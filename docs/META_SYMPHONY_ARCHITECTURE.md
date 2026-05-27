# Meta-Symphony 아키텍처

`scripts/meta_symphony.py` 는 AudioArt 스택의 최상위 — **매크로넷의 네트워크**다. 3분 길이 sub-bass 시드를 4개의 주력 매크로넷(Net 1 / 2 / 3 / Dynamic)에 모두 통과시키고, 그 결과를 3분 타임라인 위에서 LFO crossfade + stereo drift 로 엮는다.

> 관련 문서:
> - [`MULTINET_ARCHITECTURE.md`](MULTINET_ARCHITECTURE.md) — 스템을 생성하는 매크로넷들
> - [`feedback_1.md`](feedback_1.md) — 설계 원리 ("Anchored Corruption")
> - [`IMPLEMENTATION_REPORT_V1.md`](IMPLEMENTATION_REPORT_V1.md) — 구현 로그
> - [`related_work.md`](related_work.md) — 관련 연구

---

## 차례

1. [설계 철학](#1-설계-철학)
2. [모듈 상세 용어](#2-모듈-상세-용어)
3. [매크로넷 스템 전략](#3-매크로넷-스템-전략)
4. [스템 인터위빙 (Phase 2)](#4-스템-인터위빙-phase-2)
5. [Foundation 강화 (Phase 3)](#5-foundation-강화-phase-3)
6. [End-to-End 다이어그램](#6-end-to-end-다이어그램)
7. [숫자 · 주기 · 근거](#7-숫자--주기--근거)
8. [성능과 출력](#8-성능과-출력)
9. [실패 모드와 안전장치](#9-실패-모드와-안전장치)
10. [실행 방법](#10-실행-방법)
11. [향후 작업](#11-향후-작업)

---

## 1. 설계 철학

AudioArt 의 핵심 명제는 **"Anchored Corruption" — 닻을 내린 붕괴**다. 뉴럴 오디오 모델(RAVE, EnCodec, AudioLDM2 …)을 *오용*하여 학습 manifold 바깥으로 끌어내되, 구조적 안전장치(sub-bass 앵커, dry/wet, texture governor)로 출력을 음악적 영역에 묶어둔다. 결과는 노이즈가 아니라 *모델이 만든 아름다운 오해*다.

Meta-Symphony 는 그 명제를 4개 레이어로 확장한다:

| 레이어 | 범위 | 닻 |
|---|---|---|
| **MVP** (A–I) | 단일 청크에 작용하는 손상 연산자 | dry/wet, soft limiter, texture governor |
| **매크로넷** (Net 1 / 2 / 3 / Max / Dynamic) | 10–60 초 시드 위의 다단 작곡 | per-stage dw, 80 Hz crossover, RMS match |
| **Meta-Symphony** | 3분에 걸친 매크로넷의 병렬 직조 | LFO crossfade, stereo drift, sub-bass 재주입 |
| **Mastering** | 최종 라우드니스 형상화 | tanh limiter (현재); `scripts/pyloudnorm_mastering.py` 의 LUFS 파이프라인도 사용 가능 |

각 레이어는 독립적으로 실패해도 다음 레이어를 무너뜨리지 않는다. 결과적으로 3분 stereo 작품 — 바닥에는 시드의 음정 정체성이 남고, 위 대역들은 완전히 재상상된 재료가 어른거리는 — 가 만들어진다.

---

## 2. 모듈 상세 용어

각 MVP가 사인파에 *실제로* 무엇을 하는지를, 알고리즘의 핵심 단계 기준으로 정리한다.

### MVP-A — Latent Perturbation (RAVE)

```
audio (48k, mono)
  → RAVE.encode  → z ∈ ℝ^(latent_dim=16, T_z)         (예: 4초 청크 당 93 프레임)
  → z + noise_scale · n
      where n ∈ ℝ^(16, T_z), 두 모드:
        • white   : iid N(0,1)
        • smoothed: x_t = α·x_{t−1} + (1−α)·N(0,1),  α = 0.98  ← 1차 IIR Brownian 표류
  → 옵션 dim_dropout : 시간축에서 일정 비율의 row를 0으로
  → 옵션 dim_shuffle : latent_dim 축 permute
  → RAVE.decode → audio
```

440 Hz 사인파에 `noise_scale=0.05, smoothed` 적용 시: 약 0.5–2 초 주기로 느린 음색 떨림이 생긴다 (α가 결정). `noise_scale=1.0`: 모델이 사인파임을 잊고 환각하기 시작한다.

### MVP-B — Caption Loop (Audio↔Text)

```
audio  → caption_fn(audio)  → text "a metallic drifting sound underwater"
      → mutation (확률 p로 형용사를 prepend)
      → synth_fn(text)      → 새 오디오
   (N번 반복, default depth=3)
```

기본 **stub backend**: 스펙트럴 통계 → 형용사 단어로 캡션, 텍스트 MD5 해시 → FM 합성으로 오디오. `caption.backend=qwen2_audio` + `tta.backend=audioldm2` 로 설정하고 18 GB 가량의 체크포인트를 다운로드하면 실제 의미 표류 실험 가능.

### MVP-C — Token Bending (EnCodec 24 kHz)

```
audio (48k) → 24k 리샘플 → EnCodec.encode → tokens ∈ ℤ^(n_q=8, T_t)  (75 fps)
  bit_flip       : 무작위 비트 flip을 `rate` 비율의 토큰 위치에
  quantizer_drop : `rate` 비율의 quantizer row 전체 0으로
  shuffle        : 윈도우 W 안에서 토큰 위치 재셔플
  invalid_token  : `rate` 비율을 sentinel −1 로 (decode 시 0으로 clamp)
  (옵션) quantizer_range: 손상 범위를 (lo, hi)로 한정. 음수 인덱스는 상위 layer 타겟팅.
  → EnCodec.decode → 48k 리샘플 → audio
```

440 Hz 사인파에 `bit_flip rate=0.03 q_range=(-3,0)` (상위 3 quantizer만): 주 partial 보존되지만 ±2–4 Hz 사이드밴드가 quantization error 재분배로 생긴다. `rate=0.10`: 거친 codec-rip 텍스처.

### MVP-D — Checkpoint Morphing (RAVE × 2)

```
load   guitar.state_dict,  organ.state_dict
옵션  Re-Basin 정렬 (partial=inner-block / full=encoder-chain)
merged = (1−t)·guitar  +  t·organ            (linear)
       | slerp(guitar, organ, t)             (angular)
       | guitar + ε·N(0,1)                   (random_walk)
merged 를 RAVE 컨테이너에 load → 정상 encode/decode
```

독립 학습된 guitar+organ 은 중간 `t` 에서 collapse — [`feedback_1.md`](feedback_1.md) §5 + endpoint cliff sweep (`experiments/mvp_d_ckpt_morph/results.json`) 참고. 운영 단계에서는 `t ∈ [0, 0.02]` (guitar 측 페이드) 또는 `t ∈ [0.98, 1]` (organ 측 페이드) 만 사용. Meta-Symphony 의 Net Max & Net Dynamic 도 morph-guitar 핸들에 `t=0.005`, morph-organ 핸들에 `t=0.995` 를 쓴다.

### MVP-E — Neural Granular (RAVE + memory buffer)

```
LatentMemory: 최근 `memory_size` (최대 4096) latent 프레임을 담는 원형 버퍼
각 forward 청크:
  z = encode(audio)
  memory에 z push
  for i in num_grains:
      memory에서 grain_size 길이의 random window 선택
      윈도우-페이드 후 출력 위치에 더함
  z_out = mix·grains + (1−mix)·z
  decode(z_out)
```

"시간이 번지는 코러스" — 모델이 현재와 자신의 과거 4개 메아리를 동시에 듣는다. latent 가 연속이라 RAVE 에서 안정.

### MVP-F — Spectral Frozen (RAVE 상위 차원)

```
let n_freeze = round(latent_dim · auto_upper_fraction)  (예: 16개 중 8개)
let indices = z의 상위 n_freeze 차원 (고역 latent)
cached_state ← z[indices, 0:1]            초기 freeze 샘플
각 프레임:
    if global_frame_idx % update_interval == 0:
        target_state ← current z[indices, ti]
        fade_ptr ← crossfade_frames
    if fade_ptr > 0:
        α = fade_ptr / crossfade_frames
        z[indices, ti] = α·cached_state + (1−α)·target_state
        fade_ptr -= 1
    else:
        z[indices, ti] = cached_state
```

주기적 스냅샷 + crossfade 덕분에 업데이트 경계에서 "beep" 이 들리지 않고 음색이 천천히 표류. 440 Hz 사인파에는 fundamental 주변에 shimmer halo 가 생긴다.

### MVP-G — Latent Feedback (RAVE delay line)

```
buffer: 원형 ℝ^(latent_dim, max_delay)
각 latent 프레임:
    delayed   = buffer[:, ptr]
    processed = z_curr + delayed · feedback        ← 재귀적 성장
    buffer[:, ptr] = processed
    out_z = (1-mix)·z_curr + mix·processed
    ptr = (ptr+1) % max_delay
```

`delay_frames=32, feedback=0.40`: 시간 지나며 자라는 분명한 에코. `feedback=0.55`: latent 공간이 포화 → 모델이 자기 자신을 다시 해석하기 시작하면서 기괴한 피드백 공명.

### MVP-H — Codebook Organ (EnCodec 생성, 입력 없음)

```
tokens ∈ ℤ^(n_q, duration_frames) 생성:
  prime      : pattern[t] = t번째 소수 % 1024,  output[q, t] = roll(pattern, q·stride)
  fibonacci  : pattern[t] = fib[t] % 1024,      동일 roll
  random_walk: [-5,5] 정수의 cumsum % 1024 per quantizer
EnCodec.decode(tokens) → audio
```

순수 생성: 오디오 입력 없음. 코드 없는 배음열 — 소수는 더 넓은 음정 간격, fibonacci 는 더 빽빽한 반음 클러스터를 만든다. 드론 베드로 사용.

### MVP-I — Bass Massive (EnCodec 하위 quantizer)

```
tokens (n_q=8, T):
  Temporal smearing : np.roll(tokens[lo:hi], smear_delay, axis=time)   하위 q에만
  Codebook jitter   : 하위 q의 `jitter_rate` 비율 토큰에 ±1 가산
  Quantizer folding : 확률 `fold_leak_rate` 로 상위 q 의 값을 하위 q 로 복사
```

quantizer (0, 1, 2) — 거친 오디오 구조가 사는 곳 — 만 타겟. 12 프레임 smear (75 fps 기준 ≈ 160 ms) 는 베이스 군 지연 → "심해" 흔들림. jitter 는 디지털 새춰레이션을 더한다.

---

## 3. 매크로넷 스템 전략

Meta-Symphony 는 4개 net 을 단순히 병렬로 섞지 **않는다**. 일부 net 을 직렬로 묶어, 후속 net 이 *이미 뉴럴 처리된* 시드 위에서 동작하도록 만든다.

```
seed (3 분 sub-bass)
    │
    ├── Net 1 (병렬 믹스) ─────────────────────────────────────→ stem_N1
    │                                                              │
    ├── input_N3 = RMS_match(seed, stem_N1) ─→ Net 3 ──────────→ stem_N3   (누적 파괴)
    │
    ├── Net 2 (raw seed 위 3-pass 재귀) ───────────────────────→ stem_N2
    │                                                              │
    └── input_Dyn = RMS_match(seed, stem_N2) ─→ Net Dynamic ────→ stem_Dyn (루프가 먹인 폭풍우)
```

이 의존 사슬을 만든 이유는 둘:

1. **Net 3 는 파괴적 직선 체인** — 이미 텍스처화된 Net 1 출력에 다시 통과시키면, 같은 시드를 두 번 독립 손상시키는 게 아니라 *더 깊은* 동일 작곡이 나온다.
2. **Net Dynamic 의 silence drop + filter sweep 은 이미 루프로 포화된 재료에서 더 잘 작동** — Net 2 가 그 재료를 공급한다.

결과적으로 두 개의 병렬 "스템"이 만들어지고, 둘 다 동일 시드 계보를 공유하므로(sine→spatial→destruction, sine→recursion→storm) 자연스러운 crossfade 가 가능하다.

---

## 4. 스템 인터위빙 (Phase 2)

LFO 2개 + 스테레오 pan LFO 2개:

```
t ∈ [0, 180 s]

lfo_A(t) = ½ + ½·sin(2π · t / 60)        ← Pair A 주기: 60 초
lfo_B(t) = ½ + ½·cos(2π · t / 45)        ← Pair B 주기: 45 초

pan_A(t) = 0.7 · sin(2π · t / 20)        ← Pair A 스테레오 표류: 20 초
pan_B(t) = 0.7 · cos(2π · t / 25)        ← Pair B 스테레오 표류: 25 초

mix_A    = lfo_A · stem_N1 + (1 − lfo_A) · stem_N3
mix_B    = lfo_B · stem_N2 + (1 − lfo_B) · stem_Dyn

equal-power pan:
    L_pan = cos((pan + 1)·π/4)
    R_pan = sin((pan + 1)·π/4)

stereo_A = (mix_A·L_panA, mix_A·R_panA)
stereo_B = (mix_B·L_panB, mix_B·R_panB)

master = stereo_A + stereo_B              ← 2-채널 결과
```

### 주파수와 주기의 선택

- **60 vs 45 초** Pair-A / Pair-B: 서로소가 아닌 비율 → 9분 beat 패턴 (LCM(60,45)=180=곡 전체 길이, 곡은 한 번의 완전 beat 사이클을 통과).
- **20 vs 25 초** 스테레오 pan: 다시 다른 비율, LCM=100 초 — 3분 안에 두 번 표류.
- A 에 `sin`, B 에 `cos` → mix와 pan 사이 90° 위상 오프셋 — 에너지 중심이 좌/우 진동만 하는 게 아니라 스테레오 필드를 대각선으로 가로지른다.

---

## 5. Foundation 강화 (Phase 3)

인터위빙 후 원본 시드가 sub-bass 에서 재주입된다:

```
seed_trim                                        ← 원본 3분 sub-bass
   → Butterworth 2차 lowpass @ 100 Hz
   → × 10^(8/20) ≈ ×2.51                         ← +8 dB sub-boost
   → 좌/우 두 채널 모두에 broadcast
   → master 에 합산                              ← fundamental 닻 고정

master = stereo_AB_mix + (sub_L, sub_R)

final  = tanh(master · 1.25) / tanh(1.25)        ← soft limiter, drive = 1.25
       → peak 0.95 로 정규화
```

100 Hz 2차 Butterworth를 쓰는 이유: 12 dB/oct 롤오프에 phase 가 충분히 일관 — 모델 손상된 상단 대역 뒤에 깔끔히 자리잡는다. 더 높은 차수는 kick attack 의 phase 가 너무 밀린다.

+8 dB sub-boost 이유: 상단 대역이 limiter 로 압축된 후의 perceptual loudness 를 dry sub 가 회복하도록 경험적으로 맞췄다. `scripts/final_bass_pro_master.py` 와 동일한 운영 설정.

---

## 6. End-to-End 다이어그램

```text
┌────────────────────────── PHASE 1: 스템 생성 (mono) ───────────────────────────────┐
│                                                                                    │
│  generate_seed(180 s, 48 kHz)                                                      │
│    sub        = 0.6·sin(2π·32.7·t) + 0.3·sin(2π·41.2·t) + 0.2·sin(2π·55·t)         │
│    fm         = sin(2π·(55 + 5·sin(2π·0.1·t))·t)                                   │
│    lfo        = 0.6 + 0.4·sin(2π·t/30)                                             │
│    sig        = (sub + 0.15·fm) · lfo,   0.8 정규화                                │
│                                                                                    │
│  seed ──────────┬──────────────────────────────────────────────────────┐           │
│                 │                                                      │           │
│                 ▼                                                      ▼           │
│           ┌─────────────────────┐                              ┌─────────────────┐ │
│           │ Net 1 — Cathedral   │                              │ Net 2 — Organ   │ │
│           │ (5-bus 병렬)        │                              │ (3-pass 루프)   │ │
│           └──────────┬──────────┘                              └────────┬────────┘ │
│                      │ stem_N1                                          │ stem_N2  │
│      RMS_match(seed, stem_N1)                          RMS_match(seed, stem_N2)    │
│                      │                                          │                  │
│                      ▼                                          ▼                  │
│           ┌─────────────────────┐                              ┌─────────────────┐ │
│           │ Net 3 — Decoding    │                              │ Net Dyn Tempest │ │
│           │ (직선 9단)          │                              │ (시간 가변)     │ │
│           └──────────┬──────────┘                              └────────┬────────┘ │
│                      │ stem_N3                                          │ stem_Dyn │
└──────────────────────┼──────────────────────────────────────────────────┼──────────┘
                       │                                                  │
┌──────────────────────┼──── PHASE 2: 인터위빙 (stereo) ────────────────────┼──────────┐
│                      │                                                  │           │
│  lfo_A = ½+½·sin(2π·t/60)              lfo_B = ½+½·cos(2π·t/45)                     │
│  pan_A = 0.7·sin(2π·t/20)              pan_B = 0.7·cos(2π·t/25)                     │
│                                                                                    │
│  mix_A = lfo_A·stem_N1 + (1−lfo_A)·stem_N3                                         │
│  mix_B = lfo_B·stem_N2 + (1−lfo_B)·stem_Dyn                                        │
│                                                                                    │
│  stereo_A = equal_power_pan(mix_A, pan_A)                                          │
│  stereo_B = equal_power_pan(mix_B, pan_B)                                          │
│                                                                                    │
│  master   = stereo_A + stereo_B                                                    │
└─────────────────────────────────────────────────────────────────────────┬──────────┘
                                                                          │
┌──── PHASE 3: FOUNDATION & MASTER ───────────────────────────────────────┴──────────┐
│                                                                                    │
│  sub_only = lowpass_butter_2nd(seed_trim, 100 Hz) · 10^(8/20)                      │
│  master[:,0] += sub_only ;  master[:,1] += sub_only                                │
│                                                                                    │
│  final = tanh(master · 1.25) / tanh(1.25)                                          │
│        → peak 0.95 로 정규화                                                       │
│                                                                                    │
│  → runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL.wav  (180 s, stereo, 48 kHz) │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 숫자 · 주기 · 근거

### 시드 구성

| 성분 | 주파수 | 진폭 | 이유 |
|---|---|---|---|
| Sub A | 32.7 Hz (C1) | 0.60 | 물리적 sub-bass 영역 |
| Sub B | 41.2 Hz (E1) | 0.30 | 위로 단3도 — 화성적 흥미 |
| Sub C | 55.0 Hz (A1) | 0.20 | A0의 완전5도, 음조 정박 |
| FM 톤 | 55 Hz ± 5 Hz @ 0.1 Hz | 0.15 | 모델이 "해석"할 톤을 제공 |
| Breathing LFO | 1/30 Hz (30 초 주기) | env 0.6–1.0 | 거시 작곡에 느린 호흡 부여 |

### LFO 주기의 선택

| 주기 | 빈도 | 영향 | 이유 |
|---|---|---|---|
| 60 초 | 1/60 Hz | Pair-A 믹스 (N1↔N3) | 곡 길이의 1/3 → 3 사이클 |
| 45 초 | 1/45 Hz | Pair-B 믹스 (N2↔Dyn) | 곡 길이의 1/4 → 4 사이클 |
| 20 초 | 1/20 Hz | Pair-A 스테레오 pan | 충분히 빠르게 운동감, 충분히 느리게 통일성 |
| 25 초 | 1/25 Hz | Pair-B 스테레오 pan | 20초와 어긋남 → 끊임없는 cross-pattern |

LCM(60, 45) = 180 초 = 곡 길이: Pair A 와 B 가 정확히 곡 끝에서 위상 정렬.

LCM(20, 25) = 100 초: 스테레오 이미지가 곡 안에서 두 번 반복하지 않는 Lissajous 패턴을 그린다.

### Phase 3 앵커

| 설정 | 값 | 이유 |
|---|---|---|
| Sub-cutoff | 100 Hz | 베이스 임팩트 보존하면서 mid 에너지 아래에 머무름 |
| Sub-boost | +8 dB | limiter 가 mid/high 를 누른 뒤의 라우드니스 회복 |
| Limiter drive | 1.25 | 부드러운 새춰레이션, peak 에서 약 3% THD |
| 최종 peak | 0.95 | 0.5 dB true-peak 헤드룸 |

---

## 8. 성능과 출력

### V100 단일 GPU wall time 예측

| Phase | 작업 | 예상 시간 |
|---|---|---|
| 1.1 | Net 1 on 180 s seed | 약 30 초 |
| 1.2 | Net 3 on 180 s seed | 약 35 초 |
| 1.3 | Net 2 on 180 s seed (2-pass) | 약 50 초 |
| 1.4 | Net Dynamic on 180 s seed | 약 60 초 |
| 2 | LFO crossfade + pan | < 1 초 |
| 3 | Foundation + limiter | < 1 초 |
| **총합** | | **약 3 분** |

GPU 가 공유 중이면 더 오래 걸릴 수 있음.

### 최종 출력

```
runs/masterpiece/meta_symphony/
├── seed_sub_180s.wav         (17 MB, mono, 48k)
└── META_SYMPHONY_FINAL.wav   (약 35 MB, stereo, 48k, 180 s, peak 0.95)
```

옵션: `stem_N1.wav`, `stem_N2.wav`, `stem_N3.wav`, `stem_Dyn.wav` 도 저장 가능 (현재는 in-memory만 — 향후 플래그 추가 예정).

---

## 9. 실패 모드와 안전장치

| 실패 | 발생 위치 | 안전장치 |
|---|---|---|
| **NaN 폭발** RAVE/EnCodec 체인 내부 | 어느 단계든, 특히 Net 3 의 깊이 ≥ 7 체인 | Texture Governor 가 감지 시 `wet = 0.0`. 장시간 렌더 중 2:08 지점에서 발생한 끊김 사고를 통해 검증 (`docs/IMPLEMENTATION_REPORT_V1.md` §4.2). |
| **Silent collapse** MVP-D 중간 t | Net Max δ 버스, Net Dynamic δ 버스 | 운영 t를 endpoint 영역 (0.005, 0.995) 으로 고정. |
| **클리핑** master 합 단계 | Phase 2 mix + Phase 3 sub-boost | `tanh` soft limiter drive=1.25, peak 0.95 정규화. |
| **순수 노이즈 표류** 중깊이 체인 | Net Max β/ε, Net Dynamic ε | 단계별 `wet ≤ 0.55–0.65` 로 매니폴드 부근 유지. |
| **OOM** 5분 초과 재료 | Phase 3 단일 텐서 master | 3분 초과 작품은 `scripts/run_hfo_master_sox.sh` 의 SoX 스트리밍으로 전환. |
| **리샘플 아티팩트** 48k↔24k 교차점 | 모든 C/H/I 단계 | 선형 보간 + 4-block 윈도우. band-limited 입력에는 사운드 아트 용도로 수용 가능. |

---

## 10. 실행 방법

```bash
# 단발 meta-symphony 렌더
conda activate audioart
cd /home1/irteam/proj/AudioArt
PYTHONPATH=. python scripts/meta_symphony.py

# 백그라운드 + 로그 + PID 추적 (WORKFLOW_HISTORY 패턴)
nohup python scripts/meta_symphony.py > runs/masterpiece/meta_symphony.log 2>&1 &
echo $! > runs/masterpiece/meta_symphony_pid.txt

# 실시간 모니터링
tail -f runs/masterpiece/meta_symphony.log
```

`META_SYMPHONY_FINAL.wav` 가 떨어지면:

```bash
# pyloudnorm 으로 −12 LUFS 라우드니스 마스터링
python scripts/pyloudnorm_mastering.py \
    --in  runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL.wav \
    --out runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL_LUFS12.wav
```

스템 단위 분석이 필요하면 `run_meta()` 안의 Phase 2 직전에 다음을 추가:

```python
sf.write(out_dir / "stem_N1.wav", stem_n1, sr)
sf.write(out_dir / "stem_N3.wav", stem_n3, sr)
sf.write(out_dir / "stem_N2.wav", stem_n2, sr)
sf.write(out_dir / "stem_Dyn.wav", stem_dyn, sr)
```

각 매크로넷의 기여를 독립적으로 들어볼 수 있다.

---

## 11. 향후 작업

현재 Meta-Symphony 는 **뉴럴 사운드 아트** 영역에 머문다. 프로젝트 이름 "AudioLLM-Art" 가 가리키는 다음 단계는 진짜 AudioLLM 의 본격 도입:

- **MVP-B 실제 백본 활성화** → Net Max / Net Dynamic 의 η 버스가 의미적으로 살아 움직임. 18 GB 가량의 Qwen2-Audio + AudioLDM2 다운로드.
- **새 매크로넷 후보**:
  - `net_semantic` — 캡션 텍스트가 다른 net 의 파라미터를 조건화 (예: "metallic" → C.bend.rate↑)
  - `net_llm_chain` — 캡션 → 다음 net 시드 → 캡션 → … 의 의미-신호 cross-modal loop
  - `net_prompt_morph` — text prompt 임베딩이 MVP-D 의 morph_t 를 시간 가변으로 조종
- **AudioLLM 자체 손상** — 캡션 모델의 텍스트 임베딩 공간에 직접 노이즈를 주입, 의미 표류의 "결" 자체를 변형.
- **Multi-modal Meta-Symphony 확장** — 현재의 4-stem 인터위빙에 AudioLLM 기반 5번째 stem 추가 (`stem_LLM`).
- **LUFS-aware 다이내믹 마스터링** — 마지막 `tanh` 한 단계를 `pyloudnorm` 으로 자동화 + 작품마다 −12/−14/−16 LUFS 변형 동시 출력.

이 단계가 들어오면 프로젝트 이름이 비로소 정합적으로 작동하고, 현재의 Meta-Symphony 는 *Meta-Symphony v1 — Pre-AudioLLM* 로 보존된다.
