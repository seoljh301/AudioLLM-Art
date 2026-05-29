# MVP Taxonomy — V1 + V2 (24 개)

AudioArt 의 24 개 MVP 를 **I/O 시그니처 × 작동 도메인 × 미학 축** 으로 분류한 정리.
포스터 / GUI 탭 순서 / 문서 재편 / V2 통합의 1차 기준.

## 1. I/O × 작동 도메인 매트릭스 (V1 + V2)

|                                  | weight | continuous latent     | discrete tokens | STFT mag/phase | text / cross-modal |
|----------------------------------|:------:|:---------------------:|:---------------:|:--------------:|:------------------:|
| **∅ → audio** (generator)        | **Y**  | **O**                 | **H**           | **P**          | **S**              |
| **audio → audio** (mutator)      | **D**  | A · E · F · G · K     | C · I · J · L   | **Q**          | **B**              |
| **(corpus × target) → audio**    |        | **M**                 | **R**           | **N**          | **T**              |
| **audio → text → audio**         |        | **U**                 | **V**           | **W**          | **B**              |

B 는 mutator 와 recursive 양쪽 걸친다. V2 6 개 신규 = Y · S · T · U · V · W.

**V2 신규 (6 개):**
- **Y** Phantom Weight — ∅→audio · weight. 두 체크포인트 morph + random latent walk.
- **S** Random Prompt TTA — ∅→audio · text. 랜덤 prompt → TextToAudio.
- **T** CLAP Musaicing — corpus×text→audio. text prompt 시퀀스 → audio grain match.
- **U** Cap-Steered Latent — a→t→a · latent. caption → text embed → latent bias.
- **V** Cap-Conditioned Tokens — a→t→a · tokens. caption → seed token walk.
- **W** Cap→TTA Loop — a→t→a · STFT. caption ↔ TTA 재귀 → attractor drift.

**V2 backend 정책:**
- 기본값 `use_real: false` → procedural fallback (deterministic, 외부 모델 불필요)
- `use_real: true` → LAION-CLAP / AudioLDM / (옵션) Qwen-Audio 시도. 실패시 자동 fallback.
- 두 백엔드 모두 같은 인터페이스 (`src/core/v2/text_audio.py · TextAudioBridge`)

## 2. 미학 / 실패 모드 축 (2차 분류)

| 카테고리 | 멤버 | 핵심 메커니즘 |
|---|---|---|
| **Noise injection** | A, C | latent / token 에 직접 노이즈 주입 |
| **Mode pathology** | D | 두 체크포인트 사이 weight 공간 보간 → mode collapse |
| **Self-recursion** | B, E, G | 출력이 입력으로 되먹임 (text loop / grain memory / latent delay) |
| **Freeze / sustain** | F | 잠재 표현 일부를 시간상 정지 |
| **Time surgery** | K, L | 시간축 warp / 결손 inpaint 복원 |
| **Concatenative** | M, N | 외부 코퍼스 grain 매칭 후 재조립 |
| **Pure invention** | H | 입력 없음 — 코드북 시퀀스 자체에서 생성 |
| **Spectral zone** | I | 저역 다중 quantizer 변조 — 주파수대 한정 |
| **Codec semantic split** | J | semantic + acoustic dual-token (Mimi) |

## 3. 파이프라인 위치 (3차 — 신호 흐름)

```
                              ┌─────────────────────────────┐
                              │   external corpus           │
                              └─────────┬───────────────────┘
                                        ▼
SEED ──▶ [generators]   H  ──┬──▶ [concatenative]   M, N  ──┐
                              │                              │
         [mutators]                                          │
            ├─ latent     A, D, E, F, G, K ────────┬────────►│
            ├─ tokens     C, I, J, L ──────────────┤         │
            └─ recursive  B (text loop) ───────────┘         │
                                                             ▼
                                                       MIX / MASTER
```

- H: 발원지 — 다른 모든 stage 의 SEED 가능
- M / N: 외부 코퍼스를 끌어와 cross-pollination
- B: 자기 출력 → 자기 입력 — 가장 폭발적 / 가장 위험
- 나머지 mutator: 단순 in-place 변환, 체이닝 자유도 가장 높음

## 4. 체인 친화도 — 자연스러운 페어/쌍대

| 페어 | 도메인 일치 | 의미 |
|---|---|---|
| **A + D** | latent (RAVE) | noise 주입 → mode 보간. 검증된 핵심 체인 |
| **C + I + J** | discrete tokens | 코덱 token 변조 가족 |
| **E + M** | grain | E 출력을 M 코퍼스로 재투입 (self → external) |
| **G + F** | latent | feedback × freeze = 정상상태 진동 |
| **B + H** | invention | text 생성 ↔ token 생성 — 둘 다 발원형 |
| **N + D** | weighting | N 의 KL-NMF 가중치를 D 의 morph `t` 에 매핑 |
| **L + K** | time | restoration + warp = 시간축 손상 → 복구 페어 |

## 5. 단일 행 카드 (포스터 / 메모용)

```
H ┃ generator     ┃ tokens                 ┃ ∅ → audio              ┃ pure invention
O ┃ generator     ┃ latent (RAVE drift)    ┃ ∅ → audio              ┃ Brownian babble
P ┃ generator     ┃ STFT (random + GL)     ┃ ∅ → audio              ┃ Griffin-Lim invention
Y ┃ generator     ┃ weight (morph)         ┃ ∅ → audio              ┃ weight prior
S ┃ generator     ┃ text (prompt → TTA)    ┃ ∅ → audio              ┃ text invention
A ┃ mutator       ┃ latent (RAVE)          ┃ audio → audio          ┃ noise injection
D ┃ mutator       ┃ weight (RAVE morph)    ┃ audio → audio          ┃ mode pathology
E ┃ mutator       ┃ latent (self grain)    ┃ audio → audio          ┃ self-recursion
F ┃ mutator       ┃ latent (freeze)        ┃ audio → audio          ┃ sustain
G ┃ mutator       ┃ latent (feedback)      ┃ audio → audio          ┃ self-recursion
K ┃ mutator       ┃ latent (time warp)     ┃ audio → audio          ┃ time surgery
C ┃ mutator       ┃ tokens (EnCodec)       ┃ audio → audio          ┃ noise injection
I ┃ mutator       ┃ tokens (bass zone)     ┃ audio → audio          ┃ spectral zone
J ┃ mutator       ┃ tokens (Mimi sem+ac)   ┃ audio → audio          ┃ codec split
L ┃ mutator       ┃ STFT/token inpaint     ┃ audio → audio          ┃ time surgery
Q ┃ mutator       ┃ STFT (phase only)      ┃ audio → audio          ┃ phase surgery
B ┃ recursive     ┃ text ↔ audio           ┃ audio → text → audio   ┃ cross-modal loop
U ┃ recursive     ┃ text → latent bias     ┃ audio → text → audio   ┃ cap-steered latent
V ┃ recursive     ┃ text → token walk      ┃ audio → text → audio   ┃ cap-conditioned tokens
W ┃ recursive     ┃ text ↔ STFT (TTA)      ┃ audio → text → audio   ┃ attractor drift
M ┃ concatenative ┃ latent (codec corpus)  ┃ corpus × target → out  ┃ external grain
R ┃ concatenative ┃ tokens (EnCodec)       ┃ corpus × target → out  ┃ external grain
N ┃ concatenative ┃ STFT (PF / KL-NMF)     ┃ corpus × target → out  ┃ external grain
T ┃ concatenative ┃ text → grain (CLAP)    ┃ corpus × text → out    ┃ text-driven match
```

## 6. GUI / 문서 재편 권고

### GUI 탭 순서 (현재 알파벳 순 → 의미 단위 권고)
```
🎧 Listen
🎲 Generators           H
🌀 Latent mutators      A · D · E · F · G · K
🧬 Token mutators       C · I · J · L
🪞 Recursive            B
🧺 Concatenative        M · N
🌐 Multinet             net1/2/3/max/dynamic/meta
```

### 문서 분리 (현재 mvp_x/README.md 14 개 → 도메인별 3 문서로 통합 가능)
- `docs/family_latent.md` — A · D · E · F · G · K
- `docs/family_tokens.md` — C · H · I · J · L
- `docs/family_concat.md` — M · N
- B 는 cross-modal 단일 문서 유지 (`docs/family_recursive.md`)

### Net Max bus 매핑 후보
- bus L (bass)   ← I
- bus M (core)   ← A · D
- bus H (shimmer) ← C · J
- bus T (recursive) ← B · G
- bus D (drone) ← F · H
- bus G (grain) ← E · M
- bus S (spectral) ← N
- bus K (time)  ← K · L

## 7. V2 AudioLLM 자연 자리

| V2 추가 후보 | 위치 (매트릭스) | 비고 |
|---|---|---|
| 텍스트 prompt → token | text → tokens | M / N 에 텍스트 가중 prior 부여 |
| AudioLLM caption + TTA | text ↔ audio | B 의 후속 — 실제 LLM 도입 |
| Semantic token rearrange | text → tokens | J 의 Mimi semantic 축 활용 |
| Cross-modal interp | text × latent | A + 텍스트 임베딩 boundary |

전체적으로 V2 는 **text/cross-modal** 열이 비어있는 공간을 채워 들어간다.

## 8. 시각화

`runs/mvp_taxonomy.png` 에 동일 매트릭스를 카드 그리드로 렌더링.
`scripts/build_taxonomy_diagram.py` 로 재생성.
