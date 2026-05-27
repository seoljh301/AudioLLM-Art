# AudioArt: Comprehensive Research Survey (2026 Update)

이 문서는 신경망 오디오 코덱, 오디오 파운데이션 모델, 그리고 생성형 AI를 활용한 사운드 아트 및 학술 연구의 현황을 총망라합니다. AudioArt 프로젝트의 핵심인 "신경망 시스템의 미학적 오용(Aesthetic Misuse)"을 정립하기 위한 선행 연구들을 조사하여 정리했습니다.

> **심화 조사:** 본 서베이의 검증 결과 + V2 확장 항목 65개 (모델 9, 이론 7, 예술가 6, 컨퍼런스 6 등) 는 별도 보고서 [`UPDATES_V2.md`](UPDATES_V2.md) 에 정리되어 있다. 검증 부족 항목 표기 (`(미확인)`) 도 함께 포함.

---

## 1. 개요: 신경망 글리치 전통 (The Neural Glitch Tradition)

AudioArt는 지난 30년간 이어져 온 **글리치 아트(Glitch Art)**의 전통을 계승합니다. 1980년대 야수나오 토네(Yasunao Tone)의 "상처 입은 CD" 실험이나 1990년대 오발(Oval)의 디지털 실패 미학이 물리적/기계적 결함에 집중했다면, 현대의 신경망 글리치는 **학습된 잠재 공간(Latent Space)과 이산적 토큰(Tokens)**을 파괴의 대상으로 삼습니다.

*   **RAVE (2021~):** 실시간 잠재 공간 변조를 통해 음색의 유기적 표류를 가능케 함.
*   **EnCodec/DAC (2022~):** 오디오를 디지털 토큰으로 분해하여, 비트 플립(Bit-flip)이나 퀀타이저 조작과 같은 새로운 차원의 '데이터벤딩' 수단을 제공함.
*   **AudioLLM (2023~):** 소리와 언어 사이의 재해석 과정을 통해 의미론적 붕괴(Semantic Collapse)를 탐구함.

---

## 2. 핵심 모델 및 기술적 토대 (Foundational Models)

### 2.1 신경망 오디오 코덱

| 모델 | 연구 기관 | 주요 논문 / 기술 | AudioArt 에서의 활용 의미 |
| :--- | :--- | :--- | :--- |
| **RAVE** | ACIDS-IRCAM | [arXiv:2111.05011](https://arxiv.org/abs/2111.05011) | 실시간 VAE; 연속적 잠재 매니폴드 탐색 및 모핑 (MVP-A, D, E, F, G) |
| **EnCodec** | Meta AI | [arXiv:2210.13438](https://arxiv.org/abs/2210.13438) | RVQ 기반 신경망 코덱; 이산 토큰의 직접적인 변조 (MVP-C, H, I) |
| **DAC** | Descript | [GitHub](https://github.com/descriptinc/descript-audio-codec) | 범용 도메인 고압축 코덱; 멀티 스케일 토큰 조작 |
| **Mimi** | Kyutai Labs | [HuggingFace](https://huggingface.co/kyutai/mimi) / [Moshi 논문](https://kyutai.org/Moshi.pdf) | 12.5 fps 스트리밍 코덱; semantic + acoustic 듀얼 토큰 — V2 MVP-J 후보 |
| **BigVGAN** | NVIDIA ADLR | [arXiv:2206.04658](https://arxiv.org/abs/2206.04658) | 주기성 활성화 (Snake) 기반 universal vocoder; RAVE v2 활성화의 이론 토대 |

### 2.2 텍스트-오디오 생성 모델

| 모델 | 기관 | 논문 | 의미 |
| :--- | :--- | :--- | :--- |
| **AudioLDM 2** | CVSSP | [arXiv:2308.05734](https://arxiv.org/abs/2308.05734) | 텍스트-오디오 잠재 확산; MVP-B TTA 백본 |
| **MusicGen** | Meta AudioCraft | [arXiv:2306.05284](https://arxiv.org/abs/2306.05284) | EnCodec 토큰 기반 음악 자동회귀; 장형 음악 생성 |
| **AudioGen** | Meta | [arXiv:2209.15352](https://arxiv.org/abs/2209.15352) | 환경음 자동회귀 생성 — Net 1 H 버스 드론 베드의 학술적 선행 |
| **MAGNeT** | Meta | [arXiv:2401.04577](https://arxiv.org/abs/2401.04577) | masked non-autoregressive 토큰 생성 — 7× 속도, V2 빠른 재생성 토대 |
| **VampNet** | Adobe + Northwestern | [arXiv:2307.04686](https://arxiv.org/abs/2307.04686) | 마스크 기반 토큰 재생성; MVP-C bend 의 학술적 사촌 |
| **Stable Audio Open** | Stability AI | [Blog](https://stability.ai/news/stable-audio-open-small) / [HF](https://huggingface.co/stabilityai/stable-audio-open-1.0) | CC 데이터로만 학습된 오픈 텍스트-오디오 모델 |

### 2.3 오디오 LLM (캡션 / 이해)

| 모델 | 기관 | 논문 | 의미 |
| :--- | :--- | :--- | :--- |
| **Qwen2-Audio** | Alibaba | [arXiv:2407.10759](https://arxiv.org/abs/2407.10759) | 멀티모달 오디오-언어; MVP-B 캡션 백본 후보 |
| **SALMONN** | Tsinghua + ByteDance | [arXiv:2310.13289](https://arxiv.org/abs/2310.13289) | speech + audio + music 통합 LLM — MVP-B 대안 백본 |
| **Pengi** | Microsoft | [arXiv:2305.11834](https://arxiv.org/abs/2305.11834) | open-ended audio QA — caption variants |
| **Audio Flamingo 2** | NVIDIA | [arXiv:2503.03983](https://arxiv.org/abs/2503.03983) | 장형 오디오 추론, 향후 Semantic Governor 의 추론 엔진 후보 |
| **GAMA** | NVIDIA | [arXiv:2406.11768](https://arxiv.org/abs/2406.11768) | 복합 추론 audio LLM — V2 Semantic Governor 의 정량화 기반 |
| **LTU-AS** | MIT CSAIL | [arXiv:2402.16021](https://arxiv.org/abs/2402.16021) | speech + audio 통합 이해 |

---

## 3. 주요 사운드 아트 프로젝트 (2024~2025 최신 사례)

최근 신경망 코덱은 단순한 압축 도구를 넘어, 세계적인 예술가들의 핵심 악기로 자리 잡았습니다.

### 3.1 Holly Herndon & Mat Dryhurst: "The Call" (2024~2025)
*   **장소:** 런던 서펜타인 갤러리 (Serpentine Gallery).
*   **기술:** 커스텀 신경망(RAVE 및 Latent Diffusion 계열)을 이용한 **가창 목소리 전이(Singing Voice Transfer)**.
*   **의의:** 대중의 목소리를 실시간으로 수집하여 디지털 합창단으로 재합성함으로써, 잠재 공간 내의 집단적 정체성을 탐구.

### 3.2 Björk & Robin Meier: "Nature Manifesto" (2024)
*   **장소:** 파리 퐁피두 센터 (Centre Pompidou).
*   **기술:** **RAVE**를 사용하여 멸종 위기 종의 환경 녹음본과 비요크의 목소리, 합성 텍스쳐를 실시간으로 블렌딩.
*   **의의:** 환경과 반응하는 "포스트 내추럴" 사운드스케이프 구축.

### 3.3 Evala: "Studies for" (2024~2025)
*   **장소:** 도쿄 NTT ICC.
*   **기술:** Sony 연구진과 공동 개발한 **SpecMaskGIT** (코덱 기반 생성 모델).
*   **의의:** 200시간 분량의 개인 아카이브를 하나의 "거대 잠재 공간"으로 간주하여, 이전에 들어본 적 없는 8채널 입체 음향을 실시간 생성.

### 3.4 IRCAM: "AFTER" Framework (2024~2025)
*   **기술:** RAVE와 EnCodec의 잠재 공간 내부에서 작동하는 **잠재 확산 모델(Latent Diffusion)**.
*   **의의:** 하나의 소리에서 '스타일(음색)'을 추출하고 다른 소리에서 '구조'를 추출하여 결합하는 "뉴럴 리믹싱" 도구 대중화.

---

## 4. 예술적 방법론 연구 (Artistic Methods)

### 4.1 잠재 공간 섭동 (Latent Perturbation)
*   **방법:** VAE의 인코더가 생성한 잠재 벡터 `z`에 노이즈를 주입하거나 차원을 탈락(Dropout)시킴.
*   **효과:** 음색의 정체성이 서서히 흐려지며 유기적으로 변이되는 효과. (MVP-A, E, F, G의 근간)

### 4.2 토큰 데이터벤딩 (Token Databending)
*   **방법:** EnCodec과 같은 코덱의 이산 토큰 비트스트림에 직접 개입하여 비트를 뒤집거나 순서를 섞음.
*   **효과:** 전통적인 글리치와 유사하지만, 신경망이 이를 '오해'하여 복원하는 과정에서 발생하는 독특한 질감(Neural Texture)이 특징. (MVP-C, I의 근간)

### 4.3 모델 자식 작용 (Model Autophagy / Collapse)
*   **학술적 배경:** Shumailov et al. (2024, [arXiv:2305.17493](https://arxiv.org/abs/2305.17493))의 "재귀의 저주". 생성된 데이터를 다시 학습에 사용하면 모델이 극단적으로 단순화됨.
*   **예술적 적용:** Audio ↔ Text ↔ Audio 루프를 반복하여, 소리가 언어적 편향에 의해 기괴하게 변해가는 과정을 작품화. (MVP-B의 근간)

---

## 4.4 모델 병합 / 모드 연결성 이론 (MVP-D Re-Basin 의 학술적 토대)

| 논문 | 저자 | 출처 | AudioArt 와의 관계 |
| :--- | :--- | :--- | :--- |
| **Git Re-Basin: Merging Models modulo Permutation Symmetries** | Ainsworth, Hayase, Srinivasa | ICLR 2023, [arXiv:2209.04836](https://arxiv.org/abs/2209.04836) | 독립 학습된 네트워크를 permutation 정렬 후 보간하면 low-loss 곡선이 보존됨. MVP-D 의 `re_basin.py` (partial) + `re_basin_full.py` (full encoder) 두 구현의 직접 기반. |
| **Linear Mode Connectivity and the Lottery Ticket Hypothesis** | Frankle, Dziugaite, Roy, Carbin | ICML 2020, [arXiv:1912.05671](https://arxiv.org/abs/1912.05671) | 같은 초기값에서 학습된 두 모델이 weight 공간에서 선형 경로로 연결됨. MVP-D 의 "공유 init 두 RAVE 학습" V2 권고의 이론 토대. |
| **Model Soups: Averaging Weights to Improve Accuracy** | Wortsman et al. | ICML 2022, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) | 여러 fine-tune 결과의 가중치 평균이 단일 best 보다 견고함. V2 의 "Meta-Symphony Soup" 후보. |

## 4.5 자기-소비 / 재귀 생성 이론 (MVP-B Caption Loop 의 학술적 토대)

| 논문 | 저자 | 출처 | AudioArt 와의 관계 |
| :--- | :--- | :--- | :--- |
| **The Curse of Recursion: Training on Generated Data Makes Models Forget** | Shumailov et al. | [arXiv:2305.17493](https://arxiv.org/abs/2305.17493) | 생성 데이터로 반복 학습 시 분포 꼬리가 사라지는 "모델 붕괴" 정의. MVP-B Caption Loop 의 핵심 이론. |
| **Self-Consuming Generative Models Go MAD** | Alemohammad et al. | ICLR 2024, [arXiv:2307.01850](https://arxiv.org/abs/2307.01850) | autoregressive loop 의 3 패턴 (fully synthetic / fresh data / fixed synthetic) 분류. V2 MVP-B 운영 전략 결정 기준. |

---

## 5. 오픈소스 도구 및 프레임워크

| 도구 | 용도 | AudioArt 에서의 역할 |
| :--- | :--- | :--- |
| **nn_tilde** | Max/MSP 실시간 추론 플러그인 | RAVE 모델의 실시간 제어 인터페이스 |
| **AudioCraft** | Meta 통합 라이브러리 | EnCodec / MusicGen / AudioGen / MAGNeT 연구 및 렌더링 |
| **AFTER** | IRCAM 잠재 확산 프레임워크 | 고차원 사운드 합성 및 리믹싱 |
| **Diffusers** | HF 확산 모델 라이브러리 | AudioLDM 2 기반 캡션 루프 구현 |
| **transformers** | HuggingFace | Qwen2-Audio / SALMONN / Audio Flamingo 2 등 audio LLM 로드 |
| **Moshi** | Kyutai 대화 모델 | Mimi codec + 실시간 speech-text 통합 — V2 MVP-J 의 reference 구현 |

---

## 6. 결론: 향후 연구 방향

본 조사 결과, 신경망 코덱을 이용한 실시간 사운드 아트는 이미 현대 전자음악의 최전선에 서 있음을 확인했습니다. AudioArt는 단순히 개별 모델을 사용하는 것을 넘어, **Net Max** 나 **Meta-Symphony** 와 같이 **"네트워크 간의 복합적인 상호작용"** 을 체계적으로 설계하고 문서화했다는 점에서 기존 선행 연구들과 차별화되는 독창성을 가집니다.

### 6.1 AudioArt V1 의 현재 위치

본 프로젝트의 V1 단계 (2026년 5월 기준) 가 도달한 지점:

- **9 개 MVP** + **5 매크로 네트워크 (Multinet)** + **Meta-Symphony** (3 분 stereo) + **정적 데모 페이지** (52 트랙).
- **Texture Governor**: NaN 가드 + flatness/RMS/centroid 기반 자동 wet 감쇠로 9 MVP × 5 매크로넷 × 메타심포니 통과 중 NaN 도달 0 회.
- **Anchored Corruption** 의 실증 — 80 Hz crossover 가 모든 매크로넷의 안정성을 책임.
- **MVP-D Re-Basin** — partial (inner-block) + full (encoder-chain) 두 구현 모두 시도, mode-connectivity 한계 정량 측정 (cliff sweep 19 점).

### 6.2 향후 방향 — AudioLLM 단계 (V2)

프로젝트 명칭 **"AudioLLM-Art"** 가 가리키는 다음 단계:

1. **MVP-B 실제 백본 활성화** — 현재 stub 으로만 동작하는 Caption Loop 를 Qwen2-Audio-7B-Instruct + AudioLDM2 (~18 GB) 로 전환. **Shumailov et al. (2024) 의 모델 붕괴 이론** 의 직접 실증 실험.
2. **MVP-J: Mimi 코덱 통합** — Kyutai 의 의미론적 + 음향적 듀얼 토큰 스트림을 따로 손상시키는 변형. 텍스쳐뿐 아니라 '언어적 음악' 의 붕괴까지 탐구.
3. **AudioLLM 조건부 매크로넷** — `net_semantic`, `net_llm_chain`, `net_prompt_morph` 등 캡션 텍스트가 다른 매크로넷의 파라미터를 시간 가변으로 결정. **AFTER** 의 latent diffusion 제어를 매크로 차원으로 확장.
4. **Semantic Governor** — Texture Governor 의 의미 측 짝. 청크 캡션이 시드 의미와 너무 멀어지면 자동 wet 감쇠.
5. **Meta-Symphony v2 — 5 stem** — 현재 4 stem 에 `stem_LLM` 추가. 멀티 모달 cross-modal loop.
6. **AudioLLM 자체 손상** — 캡션 모델의 텍스트 임베딩 공간에 직접 노이즈 주입. *AudioLLM 도메인의 perturbation* — RAVE latent perturbation 의 의미 도메인 대응물.

V1 의 모든 결과 (`runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL.wav` 등) 는 *Pre-AudioLLM* 라벨로 보존하여, V2 와의 음상 비교 실험을 위한 baseline 으로 사용.

---
**마지막 업데이트:** 2026년 5월  
**조사 대상:** 신경망 오디오 코덱, 생성형 AI 사운드 아트, 모델 붕괴 이론  
