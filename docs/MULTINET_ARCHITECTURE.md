# Multinet 아키텍처

`scripts/multinet.py` 안에 정의된 5개의 매크로 신호 그래프 — 9개 MVP 모듈(A–I)을 조합해 재현 가능한 사운드 아트 작품을 만들어내는 구성도다.

> 관련 문서:
> - [`META_SYMPHONY_ARCHITECTURE.md`](META_SYMPHONY_ARCHITECTURE.md) — 이 5개 net을 다시 엮는 메타 네트워크
> - [`feedback_1.md`](feedback_1.md) — 설계 원리 ("Anchored Corruption", Texture Guard)
> - [`IMPLEMENTATION_REPORT_V1.md`](IMPLEMENTATION_REPORT_V1.md) — 구현 로그
> - [`related_work.md`](related_work.md) — 관련 연구 서베이

---

## 차례

1. [기호 및 표기법](#1-기호-및-표기법)
2. [모듈 빠른 참조](#2-모듈-빠른-참조)
3. [공통 인프라](#3-공통-인프라)
4. [Net 1 — Crystal Cathedral](#4-net-1--crystal-cathedral)
5. [Net 2 — Recursive Organ](#5-net-2--recursive-organ)
6. [Net 3 — Decoding Chamber](#6-net-3--decoding-chamber)
7. [Net Max — Cathedral Hive](#7-net-max--cathedral-hive)
8. [Net Dynamic — Tempest](#8-net-dynamic--tempest)
9. [5개 net 정량 비교](#9-5개-net-정량-비교)
10. [운영 노트](#10-운영-노트)
11. [향후 작업](#11-향후-작업)

---

## 1. 기호 및 표기법

| 기호 | 의미 |
|---|---|
| `[ S ]` | 입력 시드 오디오 |
| `[ A ]` … `[ I ]` | 9개 MVP 모듈 (§2 참고) |
| `==>>` | 48 kHz 오디오 흐름 (다르면 명시) |
| `→24k→` | 24 kHz 강제 리샘플 (EnCodec 도메인 진입) |
| `→48k→` | 48 kHz 복귀 리샘플 |
| `[ + ]` | 합 / 믹스 버스 |
| `[ × ]` | 곱 (envelope, gain) |
| `α`…`θ` | 그리스 문자 라벨이 붙은 병렬 버스 |
| `dw` | dry/wet 비율 (0 = 원본만, 1 = 완전 변조) |
| `t` | MVP-D 모핑 보간 계수 (0 = guitar, 1 = organ) |
| `RaveHandle` | RAVE TorchScript + SR + latent_dim 묶음 |
| `CodecHandle` | EnCodec (24 kHz, n_q=8, codebook=1024) 묶음 |

각 MVP의 render 함수는 내부적으로 다음 절차를 거친다:

1. `chunk_seconds=4.0` 단위로 청크화 (overlap 0.05).
2. encode (RAVE → continuous latent `z`, EnCodec → discrete tokens).
3. 모듈 고유 손상 연산.
4. decode → 오디오.
5. **Texture Governor** — NaN 차단 + flatness/RMS/centroid 임계치 기반 자동 wet 감쇠.
6. `src/core/mix.py` 의 dry/wet mix — RMS match + soft tanh limiter + (옵션) 80 Hz crossover anchor.

---

## 2. 모듈 빠른 참조

| MVP | 백본 | 작용 대상 | 핵심 파라미터 | 음상 |
|---|---|---|---|---|
| **A** Latent Perturbation | RAVE | latent z | `noise_scale`, `dim_dropout`, `noise_mode={white,smoothed}`, `noise_smooth=0.98` | 유기적 음색 떨림 (smoothed) / 거친 grit (white) |
| **B** Caption Loop | Qwen2-Audio + AudioLDM2 *(또는 stub)* | audio↔text↔audio | `depth=3`, `text_mutation_prob=0.5` | 의미적 오역 표류 |
| **C** Token Bending | EnCodec 24k | RVQ tokens (n_q, T) | `mode`, `rate`, `quantizer_range`, `shuffle_window` | 디지털 grit, codec 오역 |
| **D** Checkpoint Morphing | RAVE × 2 | model weights | `t` (linear/slerp), `rebasin_mode` | 하이브리드 악기 / 붕괴-유령 |
| **E** Neural Granular | RAVE | latent + memory buffer | `grain_size=16`, `memory_size=4096`, `num_grains=5`, `mix=0.5` | 시간이 번지는 코러스 |
| **F** Spectral Frozen | RAVE | 상위 latent 차원 | `auto_upper_fraction=0.3–0.6`, `update_interval=64–128`, `crossfade_frames=16` | shimmer / aurora |
| **G** Latent Feedback | RAVE | latent + delay line | `delay_frames=16–96`, `feedback=0.30–0.55`, `mix=0.5–0.7` | 진화하는 뉴럴 에코 |
| **H** Codebook Organ | EnCodec | 생성 토큰 (입력 없음) | `mode={prime,fibonacci,random_walk}`, `stride`, `duration_frames` | 추상 드론 |
| **I** Bass Massive | EnCodec | 하위 RVQ 토큰 | `smear_delay`, `smear_quantizers=(0,2)`, `jitter_rate`, `fold_leak_rate` | 초저역 smear |

---

## 3. 공통 인프라

### 샘플레이트 경계

```
시드 오디오 (48 kHz, RAVE 네이티브)
    │
    ├── RAVE 단계 (A, D, E, F, G) ───── 48 kHz 유지
    │
    └── EnCodec 단계 (C, H, I) ──→ 48k→24k 리샘플 → encode → 토큰
                                              ↑                 │
                                              └─── decode ──→ 24k→48k 리샘플
```

`_resample_to(audio, n)` 가 경계를 처리한다 (`np.interp` 선형 보간). 왕복 한 번에 12 kHz 이상 대역에서 약 −1 dB 손실 발생.

### Texture Guard 체인

```
damaged_chunk
    ├── compute_texture_metrics()      → (rms, flatness, centroid, ZCR, has_nan)
    ├── govern_wet()                   → wet_per_chunk ∈ [min_wet, base_wet]
    │      • has_nan       → wet = 0.0  (emergency)
    │      • flatness>0.55 → wet ×= 0.55
    │      • rms<1e-4      → wet ×= 0.35
    │      • rms>0.85      → wet ×= 0.60
    │      • centroid > Nyquist·0.42 → wet ×= 0.75
    ├── dry_wet_mix()                  → (1-wet)·dry + wet·damaged
    │      • 옵션: RMS match
    │      • 옵션: 80 Hz crossover (dry 저역 보존 + sub_boost_db 게인)
    └── soft_limiter(drive=1.0–1.25)   → tanh 새춰레이션
```

이 가드는 *모든* MVP render 함수 안에서 매 청크마다 실행된다. NaN, 폭주 게인, 백색 노이즈가 master 합산까지 도달할 수 없는 구조.

---

## 4. Net 1 — Crystal Cathedral

**토폴로지 분류**: 5-bus 병렬 믹스.
**시드**: 10 초 440 Hz 사인파 @ 48 kHz.
**V100 wall time**: 약 1.5 초.

```
       ┌==>> [ I (Bass Massive) ]                              ==>> (Bus L: 35%) ┐
       │     smear=12, jitter=0.05, q=(0,2)                                      │
       │                                                                         │
       ├==>> [ A ] ==>> [ D ] ==>> [ C ]                       ==>> (Bus M: 30%) ┤
       │     noise=0.05 smoothed   t=0.005    bit_flip rate=0.03                 │
       │                                                                         │
[ S ] =┼==>> [ F ] ==>> [ A ]                                  ==>> (Bus H: 20%) ┼==> [ 80Hz Anchor +6dB ] ==> [ Soft Limiter drive=1.2 ] ==>> MASTER
       │     upper 50%  noise=0.10                                                │
       │                                                                         │
       ├==>> [ G ] ==>> [ E ]                                  ==>> (Bus T: 10%) ┤
       │     d=32 fb=0.4   mem=2048 grain=16                                     │
       │                                                                         │
     (Gen)>> [ H (prime, stride=7) ]                           ==>> (Bus D: 05%) ┘
```

### 버스별 역할

| 버스 | 기능 | 이유 |
|---|---|---|
| **L** Foundation | sub-bass 닻 | I + 80 Hz crossover 가 사인파 fundamental을 물리적으로 살려둠. 손상이 심해도 베이스가 무너지지 않는다. |
| **M** Core | 음색 본체 | A → D → C 의 3종 손상이 누적되며 dw를 낮춰 안정성 확보. |
| **H** Shimmer | 고역 띠 | F freeze가 shimmer 생성, A가 유기적 떨림 추가. |
| **T** Recursive | 에코 + 기억 | G가 피드백을 만들고 E가 과거 grain을 현재로 투사 → 유령 잔향. |
| **D** Drone | 생성 베드 | H가 소수 인덱스 패턴으로 코드 없는 배음열 연주. |

### 관측 master 메트릭

| | rms | peak | centroid | flatness | 상위 frequency bins |
|---|---|---|---|---|---|
| Net 1 master | 0.232 | 0.95 | 1446 Hz | 0.117 | 439–450 + 49–82 (드론) |

440 Hz 시드의 peak가 보존(top bin 439–442)되는 동시에 H 드론 버스가 49/76/78/81/82 Hz 저역 partial을 추가 — 6개 옥타브 대역이 동시 점유된다.

---

## 5. Net 2 — Recursive Organ

**토폴로지 분류**: 매크로 피드백 루프로 감싼 직선 체인 (N pass).
**시드**: 10 초 사인파.
**Pass 수**: 기본 3 (CLI 인자 `passes=`로 override).
**V100 wall time**: 약 2 초.

```
                    ┌────────────────────────────────────────────────────────────────┐  (Loop N=3×)
                    │                                                                │
[ S ] ==>>> seed ==>┤  [ G ]    [ A ]    [ C ]    [ F ]    ==> chunk-master ==> ──┐  │
                    │  d=64    noise=0.08  shuffle  upper 30%                     │  │
                    │  fb=0.55 drop=0.15 win=12     update=64                     │  │
                    │  mix=0.7 mode=white rate=0.04 fade=12                       │  │
                    │                                                              │  │
                    │  pass 0 → pass_0.wav                                         │  │
                    │  pass 1 → pass_1.wav  ← pass_0 을 새 시드로                  │  │
                    │  pass 2 → pass_2.wav  ← pass_1 을 새 시드로                  │  │
                    └─── 마지막 pass ── soft limiter drive=1.15 ──> MASTER ◄───────┘  
                                                                                      
                                              C 단계 안에서만 SR 변환 일어남:        
                                              48k → 24k (encode) → bend → 24k → 48k  
```

매 pass마다 손상이 누적된다. 모델의 음정 정체성이 pass를 거치며 표류 — pass 0은 440 Hz를 유지하지만 pass 2에서는 597–615 Hz(top spectral bins)로 점프.

### Pass별 상위 frequency bins

| Pass | rms | 상위 bins | 비고 |
|---|---|---|---|
| 0 | 0.075 | 439, 440, 441, 448, 452 | 시드 pitch 유지 |
| 1 | 0.052 | 398, 401, 405, 410, 422 | 약 −40 Hz 활강 |
| 2 | 0.047 | 597, 605, 608, 611, 615 | 약 +157 Hz 점프 |
| **MASTER** | 0.058 | 597–615 | 마지막 pass와 동일 |

비단조 표류는 학습된 latent 공간을 거친 재귀 피드백의 특징적 동작이다 — 네트워크가 자기 직전 출력을 어떻게 "해석"하는지가 음정을 끌고 다닌다.

---

## 6. Net 3 — Decoding Chamber

**토폴로지 분류**: sub-bass 앵커 분기를 갖는 직선 9단 체인.
**시드**: 10 초 사인파.
**단계별 dw**: 0.45 (손상을 점진 누적시키고 collapse 회피).
**V100 wall time**: 약 2 초.

```
[ S ] ==>> [ A ] ==>> [ D ] ==>> [ E ] ==>> [ F ] ==>> [ G ] ==>> [ C ] ==>> [ I ] ─┐
          n=0.03    t=0.005   mem=1024   25%       d=24    bit_flip   smear=8     │
          smoothed             grain=16   upd=128   fb=0.30 rate=0.02  jitter=0.03 │
                              num=4                                                │
                                                                                   │
                                                                              ─────┤
[ S ] ==>> [ 80 Hz Low-Pass (Butterworth 2차) ] × +8 dB ──────────────────────────┤
                                                                                   │
                                                                                   ▼
                                                       [ + ] ==>> [ Limiter drive=1.2 ] ==>> MASTER
```

단계별 rms를 보면 dry/wet 앵커링의 작동이 확인된다:

| 단계 | 적용 후 | rms | spectral 초점 |
|---|---|---|---|
| s1 | A | 0.105 | 440 ± 5 |
| s2 | D | 0.033 | 440 + 627 |
| s3 | E | 0.026 | 420–445 |
| s4 | F | 0.024 | 416 + 626 |
| s5 | G | 0.021 | 439–443 |
| s6 | C | 0.027 | 439–443 |
| s7 | I | 0.027 | 207 + 439–442 |
| **MASTER** (80 Hz 앵커 포함) | **0.177** | 207 + 439–442 |

80 Hz crossover 분기가 dry 사인파를 저역 셸프로 재주입하면서 RMS를 단번에 복원한다 — **anchored corruption** 원리의 실증.

---

## 7. Net Max — Cathedral Hive

**토폴로지 분류**: 8-bus 병렬 + cross-bus feedback + 2-pass 매크로 루프.
**시드**: 30 초 강화 사인파 (440 Hz + 660 Hz 5th + 0.13 Hz tremolo + 느린 pitch drift).
**V100 wall time**: 약 24 초.

```
                                          ┌── morph_guitar (t=0.005) ──── 한 번만 로드
                                          ├── morph_organ  (t=0.995) ──── 한 번만 로드
                                          ├── codec (EnCodec 24k, n_q=8) ─ 한 번만 로드
                                          ├── rave_guitar ────────────── 한 번만 로드
                                          
       ┌==>> [ I ] ==>> [ C(invalid lower) ] ==>> [ 80Hz anchor +10dB ] ==>> (α Foundation)
       │
       ├==>> [ A ] ==>> [ D(guitar) ] ==>> [ E ] ==>> [ G ] ==>> [ A(drop) ] ==>> [ F ]  ==>> (β Core)
       │     smooth      t=0.005       mem=4096   d=48     drop=0.2          30%
       │
       ├==>> [ F(60%) ] ==>> [ G ] ==>> [ C(shuffle) ] ==>> [ F(40%) ]                ==>> (γ Ghost)
       │
[ S ] =┤==>> [ D(organ t=0.995) ] ==>> [ A(n=0.18) ] ==>> [ C(bit_flip) ] ==>> [ E ]  ==>> (δ Twin)
       │
       ├==>> [ C(bit_flip rate=0.10) ] ==>> [ I(smear+jitter+fold) ] ==>> [ G ]       ==>> (ε Glitch)
       │
     (Gen)>>[ H(prime stride=11) ] ┐
            [ H(fibonacci stride=5)]┘──>> mix 55:45 ==>> [ A(smoothed) ]              ==>> (ζ Drone)
       │
       ├==>> [ B (caption→TTA depth=3) stub backend ]                                 ==>> (η Loop-B)
       │
       └── β 출력 탭 ───────────────────>> [ G(deep d=96 fb=0.55) ] ==>> [ I(fold) ]  ==>> (θ XFB)

         pass 1 → mix_buses (α20% β18% γ12% δ12% ε10% ζ10% η8% θ10%) → MASTER_pass1
         
         pass 2 → seed′ = 0.55·MASTER_pass1 + 0.45·seed (RMS-matched)
              → 8 버스 재실행
              → mix_buses (α15% β22% γ15% δ10% ε8% ζ8% η6% θ16%) → MASTER_pass2
         
         FINAL = pass1 × (1 − S(t)) + pass2 × S(t)   where   S(t) = ½(1 − cos(π·t/T))
```

### 버스별 의도

| 버스 | 기둥 | 텍스처에 더하는 것 |
|---|---|---|
| **α** | sub-bass 척추 | I→C가 베이스 토큰의 상위 detail을 죽이고 body만 남김; 80 Hz crossover가 깨끗한 sub 재주입. |
| **β** | 구조적 코어 | RAVE 도메인의 모든 손상 모드 (latent, weight, granular, feedback, dropout, freeze)를 6단으로 압축. |
| **γ** | shimmer | F→F freeze 사이에 G echo + C shuffle을 끼워 지속적인 aurora. |
| **δ** | 오르간 측 대선율 | D를 t=0.995로 돌려 organ 정체성으로 옮기고 codec grit 추가. |
| **ε** | 파괴 | 강한 C(rate=0.10) + I(fold) + G — 최대 손상 버스. |
| **ζ** | 입력 없는 드론 | H 생성 토큰이 시드와 독립된 코드 없는 베드 제공. |
| **η** | 캡션 루프 | 오디오 스케일의 의미 표류; stub captioner 가 결정론적 형용사 사슬 생성. |
| **θ** | cross-feedback | β 출력을 받아 deep-echo + fold — master 에 코어의 "유령" 추가. |

### Pass 2 weight 재분배

|  | α | β | γ | δ | ε | ζ | η | θ |
|---|---|---|---|---|---|---|---|---|
| pass 1 | 0.20 | 0.18 | 0.12 | 0.12 | 0.10 | 0.10 | 0.08 | 0.10 |
| pass 2 | 0.15 | 0.22 | 0.15 | 0.10 | 0.08 | 0.08 | 0.06 | 0.16 |

Pass 2는 β (재처리된 코어)와 θ (cross-feedback 유령)를 증폭하고, 생성 드론과 caption 루프는 감쇠 — 작품이 시드 파생 재료를 중심으로 단단해진다.

### 관측 메트릭

| | rms | peak | centroid | flatness | 상위 bins | governor 트립 |
|---|---|---|---|---|---|---|
| pass 1 master | 0.226 | 0.95 | 1994 Hz | 0.158 | 439–442, 449 | 0 |
| pass 2 master | 0.208 | 0.95 | 1897 Hz | 0.140 | 439, 449–451, 466 | 1 (θ flat) |
| **FINAL** | **0.260** | **0.95** | **1996 Hz** | **0.159** | 439–442, 449 | — |

최종 master에는 6개의 독립 음정 영역이 공존: sub 49–82 Hz (드론), 207 (앵커 LP 잔류), 337 (β 표류), 432–444 (δ detune), 439–442 (α/γ/ε 앵커), 660 (시드 5th + 사이드밴드).

---

## 8. Net Dynamic — Tempest

**토폴로지 분류**: 8 버스 + 버스별 시간 가변 진폭 envelope + impulse 이벤트 + master filter sweep.
**시드**: 60 초 강화 사인파 (220→660 Hz pitch sweep + ±6 Hz vibrato + 진폭 arc + 6 초 sparse onsets + 30 초 gaussian dip).
**V100 wall time**: 약 20 초.

버스 토폴로지는 Net Max **pass 1**과 동일 (매크로 루프 없음). 다이내믹스는 렌더 후 post-automation 단계에서 만들어진다.

```
[ Net Max 와 동일한 8 버스 α…θ ]   →   각각 60 초 전 구간 한 번만 렌더
        │
        ▼
[ 버스별 진폭 envelope (구간별 선형 보간 + ~0.3s 스무딩) ]

  버스   t→  0s    10s   15s   22s   30s   35s   45s   55s   60s
  α          0.45  0.40  0.40  0.30  0.25  0.35  0.30  0.25  0.20
  β          0.00  0.00  0.30  0.55  0.10  0.05  0.05  0.10  0.05
  γ          0.00  0.20  0.35  0.20  0.05  0.00  0.00  0.20  0.15
  δ          0.00  0.00  0.00  0.20  0.30→0.05  0.10        0.00
  ε          0.00              0.00  0.00  0.30  0.55  0.05  0.00
  ζ          0.00                    0.05  0.20  0.30  0.30  0.25
  η          0.00                          0.00  0.20  0.40→0.30→0.10
  θ          0.00                                0.15  0.35  0.45  0.35

        │
        ▼
[ Impulse 이벤트 ]
   ├── t=15.0–15.3 s  : γ freeze CLICK (×2.0 진폭 버스트)
   ├── t=30.0–31.0 s  : SILENCE DROP (선형 ramp 1.0 → 0.05 + 0.1s 복귀 ramp)
   └── t=44.5–45.5 s  : ζ DRONE BURST (×1.5, 50 ms attack / 100 ms release envelope)

        │
        ▼
[ Master 저역통과 sweep ]   cutoff 궤적 (Hz, 구간별 선형, 스무딩 0.4s):
   0s 250 → 10s 4k → 20s 12k → 28s 11k → 30s 500 → 32s 8k → 40s 16k → 50s 14k → 55s 6k → 60s 4k
   block 4096 샘플(~85 ms 해상도), 2차 Butterworth, lfilter_zi 로 상태 연속

        │
        ▼
[ 80 Hz crossover ] : 80 Hz 이하 dry 시드 유지 (+8 dB) + 필터된 master 상단
        │
        ▼
[ Soft limiter drive=1.25 ] ==>> MASTER (rms ≈ 0.3–0.6, 시간 가변)
```

### 초당 master RMS 샘플

| t (s) | rms | 이벤트 |
|---|---|---|
| 0 | 0.520 | α 단독 ground |
| 15 | 0.305 | freeze 클릭 착지 |
| 20 | 0.122 | 필터가 좁고 β 상승 중 |
| 30 | 0.533 | silence drop 회복 직후 |
| 40 | 0.471 | ε peak |
| 45 | 0.293 | drone burst envelope 정점 |
| 50 | 0.186 | 필터가 η에 닫히는 중 |
| 55 | 0.480 | η + θ climax |
| 60 | 0.459 | 꼬리 페이드 |

60 초 구간 master RMS 범위 **0.046 ~ 0.605** (약 22 dB 다이내믹). Net Max FINAL(평균 0.260, 범위 ≈ 0.18 – 0.31, 약 4 dB)과 비교하면 Tempest는 약 5배 더 역동적.

---

## 9. 5개 net 정량 비교

| | Net 1 | Net 2 | Net 3 | Net Max | Net Dynamic |
|---|---|---|---|---|---|
| **분류** | 병렬 믹스 | 재귀 매크로 루프 | 직선 체인 | 2-pass 병렬 + xfb | 시간 가변 병렬 |
| **버스 수** | 5 | 1 (loop) | 1 (deep) | 8 × 2 | 8 |
| **사용 MVP** | A C D F G H I (7) | A C F G (4) | A C D E F G I (7) | A B C D E F G H I (9) | A B C D E F G H I (9) |
| **MVP-B 사용?** | ✗ | ✗ | ✗ | ✓ (stub) | ✓ (stub) |
| **매크로 피드백?** | ✗ | ✓ (3 pass) | ✗ | ✓ (2 pass) | ✗ |
| **Cross-bus FB?** | ✗ | ✗ | ✗ | ✓ (θ taps β) | ✗ |
| **시간 가변?** | ✗ | pass 단위 | ✗ | pass 가중치 | 초 단위 |
| **스테레오?** | mono | mono | mono | mono | mono |
| **Filter sweep?** | ✗ | ✗ | ✗ | ✗ | ✓ (master LP) |
| **Impulse 이벤트?** | ✗ | ✗ | ✗ | ✗ | ✓ (3 이벤트) |
| **기본 시드 길이** | 10 s | 10 s | 10 s | 30 s | 60 s |
| **V100 wall time** | 1.5 s | 2 s | 2 s | 24 s | 20 s |
| **Master rms** | 0.232 | 0.058 | 0.177 | 0.260 | 0.30 mean (0.05–0.60) |
| **Governor 트립** | 0 | 0 | 0 | 1 | 다수 |
| **NaN emergency** | 0 | 0 | 0 | 0 | 0 |

Wall time은 대체로 `시드 길이 × 버스 수`에 선형 — 대부분의 단계가 GPU 한계로 realtime 5–10배 속도다.

---

## 10. 운영 노트

### 리소스 사용량

| 자원 | Net 1 | Net Max | Net Dynamic |
|---|---|---|---|
| 최대 VRAM | 약 3 GB | 약 5 GB | 약 5 GB |
| 디스크 / 렌더 | wav 6개 ≈ 6 MB | wav 19개 ≈ 56 MB | wav 9개 + csv 2개 ≈ 150 MB |
| CPU 쓰레드 | 1 (선형 I/O) | 1 | 1 |

### 재현성

모든 단계가 `rng_seed=` 를 받으므로 다음 조건이 같으면 비트 단위 재현 가능:
1. RAVE / EnCodec 모델 파일 (HF 다운로드 시 SHA 확인).
2. torch 버전 (`2.4.1+cu121`).
3. GPU (V100 vs A100은 부동소수 발산 미미).
4. 시드 오디오 파일.

### 알려진 이슈

| 이슈 | 영향 | 완화 |
|---|---|---|
| C 단계 `q_range=(-3, 0)` + 낮은 rate 조합에서 `bent=0 diffs` 로그 | Net 1 M 버스, Meta-Symphony N1 입력 | Net 3 는 정상 동작 — extreme `quantizer_range` 슬라이스와 SR 경계 상호작용 의심. `token_bend.bend_tokens` 슬라이스 경로 조사 필요. |
| MVP-D 중간 `t` 에서 silent collapse | 모든 guitar↔organ 모핑 net | endpoint 영역만 사용 (`t ∈ [0, 0.02]` 또는 `t ∈ [0.98, 1]`). [`feedback_1.md`](feedback_1.md) §5 참고. |
| Net Dynamic envelope이 60 초 후 마지막 값에 고정 | Meta-Symphony Phase 1.4 (180 초 시드 주입 시) | Meta-Symphony 설계상 의도 — 꼬리 120 초를 정적 drone으로 활용. |

### CLI

```bash
python scripts/multinet.py net1     # 10 초 시드, 5-bus 병렬
python scripts/multinet.py net2     # 10 초 시드, 3-pass 재귀
python scripts/multinet.py net3     # 10 초 시드, 직선 9단
python scripts/multinet.py max      # 30 초 강화 시드, 8-bus 2-pass
python scripts/multinet.py all      # Net 1 + 2 + 3 (+ max 시드가 있으면 max도)
```

출력 위치:
```
runs/multinet/
├── sine_440_10s.wav
├── sine_max_30s.wav
├── sine_dyn_60s.wav
├── net1/   bus_{L,M,H,T,D}.wav  + MASTER.wav
├── net2/   pass_{0..N}.wav      + MASTER.wav
├── net3/   s{1..7}_X.wav        + MASTER.wav
├── net_max/      p{1,2}_bus_{α..θ}.wav + MASTER_pass{1,2}.wav + MASTER_FINAL.wav
└── net_dynamic/  bus_{α..θ}.wav + envelopes.csv + master_rms_per_second.csv + MASTER_FINAL.wav
```

---

## 11. 향후 작업

현재 Multinet 은 **뉴럴 사운드 아트** 영역의 5종 매크로넷에 머문다. 프로젝트 명칭 "AudioLLM-Art" 가 가리키는 다음 단계는 진짜 AudioLLM 통합:

- **MVP-B 실제 백본 활성화** — Qwen2-Audio + AudioLDM2 다운로드(~18 GB) 후 stub 대체. 의미 표류가 실제 언어 모델 기반으로 재구성됨.
- **AudioLLM 조건부 버스** — 캡션 결과 텍스트를 다른 버스의 파라미터로 매핑 (예: "metallic shimmering" → F.auto_upper_fraction = 0.7). 의미 → 신호 파라미터의 비선형 사상.
- **Prompt-conditioned token bending** — MVP-C 의 quantizer_range / mode 를 텍스트 prompt 임베딩으로 조건화.
- **AudioLLM 자체 손상 (semantic perturbation)** — 캡션 텍스트의 토큰 임베딩 공간에 직접 노이즈를 주입, 모델이 만드는 "오해" 자체를 의도적으로 변질시킴.
- **Multi-agent caption ensemble** — Qwen-Audio, SALMONN, Pengi 등 여러 캡셔너를 동시에 돌리고 그 disagreement 를 사운드 차원으로 사상.

위 단계가 들어오면 새로운 매크로넷이 추가될 예정 (`net_semantic`, `net_llm_chain` 등).
