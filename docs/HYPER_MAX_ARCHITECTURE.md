# Net Hyper-Max — 24 MVP 합주 아키텍처

`scripts/net_hyper_max.py` · 4 movement × 6 MVP × 45 s = 180 s
프로젝트 24 개 MVP 전부를 한 곡에 등장시키는 최대 규모 구성.

## 시간 구조

```
[0  ──────45──────90──────135────180]
  M1       M2       M3       M4
Genesis  Latent   Token   Final
                  Storm    Wash

xfade 3 s × 3 = 9 s overlap → 실제 171 s 단일 마스터
```

각 movement = 6 MVP × 1 bus, 가중 합 + soft limiter (peak 0.95).

## Movement × MVP 매핑

| Movement | 도메인 강조 | MVP 6 개 | Bus weights |
|---|---|---|---|
| **M1 Genesis** | 발원 (∅→audio) | Y · O · H · P · S · D | 1.0 / 0.8 / 0.7 / 0.6 / 0.5 / 0.4 |
| **M2 Latent surge** | latent 변형 | A · E · F · G · K · U | 1.0 / 0.9 / 0.8 / 0.8 / 0.7 / 0.7 |
| **M3 Token storm** | token 변형 | C · I · J · L · V · R | 1.0 / 0.9 / 0.7 / 0.5 / 0.8 / 0.6 |
| **M4 Final wash** | concat + recursive | Q · N · M · T · W · B | 1.0 / 0.7 / 0.6 / 0.7 / 0.6 / 0.5 |

총 24 MVP 모두 등장. 가중치는 mix peak 후 정규화.

## 신호 흐름

```
M1 (∅→audio 위주):
    Y phantom-weight ──┐
    O latent-drift ────┤
    H codebook-organ ──┼─► sum → soft-limit → M1 master
    P phase-halluc ────┤
    S random-prompt ───┤
    D ckpt-morph (Y 입력) ┘

M2 (M1 master 입력):
    A E F G K U (모두 in_48k 또는 in_24k 입력) → sum → M2 master

M3 (M2 master 입력):
    C I J L V R → sum → M3 master

M4 (M3 master + 전체 prior corpus):
    Q (prior phase 변조)
    N (corpus=M1, target=M2)
    M (corpus=M1+M2, target=M3)
    T (corpus=M1_O, text prompts)
    W (prior recursive loop)
    B (caption loop stub)
    → sum → M4 master

cross-fade: cosine ramp 3 s, between movements
```

## 샘플레이트 정책

- **마스터 sr** = 22050 (project unified)
- RAVE 내부 작동 = 48000
- EnCodec = 24000
- 각 stage 종료마다 `_resample(audio, src_sr, MASTER_SR)`

이유: STFT/CLAP/AudioLDM 가 모두 22050 기반이고, V100 RAM/디스크 절약.

## 백엔드 토글

```bash
# Procedural (기본)
python scripts/net_hyper_max.py
# → runs/hyper_max/

# Real backend (CLAP + AudioLDM)
HYPER_MAX_REAL=1 python scripts/net_hyper_max.py
# → runs/hyper_max_real/
```

영향 받는 MVP: S (TTA), T (CLAP embed), W (TTA loop).
나머지 16 MVP 는 외부 모델 불필요 → 두 변형 동일.

## 출력 구조

```
runs/hyper_max/                   # procedural
  master_180s.wav        7.2 M, 171 s, peak ≤ 0.95
  movement_1_genesis.wav 1.9 M, 45 s
  movement_2_latent.wav  1.9 M, 45 s
  movement_3_tokens.wav  1.9 M, 45 s
  movement_4_final.wav   1.9 M, 45 s
  M1_Y.wav  M1_O.wav  M1_S.wav     # generator stems
  M2_U.wav                          # cap-latent stem
  M3_V.wav  M3_R.wav                # token musaicing stems
  M4_Q.wav  M4_N.wav  M4_M.wav  M4_T.wav  M4_W/final.wav

runs/hyper_max_real/             # 동일 트리, real backend
```

## 성능

- procedural: wall **74.7 s** for 171 s output (2.3× real-time)
- real backend: wall 수 분 (AudioLDM step 시간이 지배)

V100 SM 7.0, RAVE/EnCodec/AudioLDM 모두 cuda. 메모리 안전.

## 설계 원칙

1. **각 movement = 한 도메인 / I/O 모드 시연** — taxonomy 매트릭스 4 행을 시간축에 매핑
2. **신호 전파** — M[k+1] 의 입력 = M[k] 의 master. 점진적 변형 누적
3. **bus 정규화** — sum_buses 가 `total_w` 로 나누어 mix peak 일관. soft_limit 으로 clipping 방지
4. **fallback safety** — 모든 V2 MVP 가 procedural 백업 → 외부 모델 없어도 24 MVP 풀 렌더
5. **결정론** — 모든 stage 에 seed 고정 (1, 2, 3, ..., 35)

## 향후 (V3)

- 80 s 짜리 movement 5 추가 (8-bus heterogeneous mix)
- live OSC dispatch (multinet GUI bus 슬라이더와 동기)
- spatialization: 좌우 → 4-ch 5.1 변환
- Hyper-Max × Hyper-Max self-reference (M5 = 전체 master 를 다시 통과)
- AudioLLM 가 실시간 movement scoring (∅→text→audio bridge)
