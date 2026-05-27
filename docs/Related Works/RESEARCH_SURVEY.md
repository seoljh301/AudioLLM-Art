# AudioArt: Comprehensive Research Survey (2026 Update)

이 문서는 신경망 오디오 코덱, 오디오 파운데이션 모델, 그리고 생성형 AI를 활용한 사운드 아트 및 학술 연구의 현황을 총망라합니다. AudioArt 프로젝트의 핵심인 "신경망 시스템의 미학적 오용(Aesthetic Misuse)"을 정립하기 위한 선행 연구들을 조사하여 정리했습니다.

---

## 1. 개요: 신경망 글리치 전통 (The Neural Glitch Tradition)

AudioArt는 지난 30년간 이어져 온 **글리치 아트(Glitch Art)**의 전통을 계승합니다. 1980년대 야수나오 토네(Yasunao Tone)의 "상처 입은 CD" 실험이나 1990년대 오발(Oval)의 디지털 실패 미학이 물리적/기계적 결함에 집중했다면, 현대의 신경망 글리치는 **학습된 잠재 공간(Latent Space)과 이산적 토큰(Tokens)**을 파괴의 대상으로 삼습니다.

*   **RAVE (2021~):** 실시간 잠재 공간 변조를 통해 음색의 유기적 표류를 가능케 함.
*   **EnCodec/DAC (2022~):** 오디오를 디지털 토큰으로 분해하여, 비트 플립(Bit-flip)이나 퀀타이저 조작과 같은 새로운 차원의 '데이터벤딩' 수단을 제공함.
*   **AudioLLM (2023~):** 소리와 언어 사이의 재해석 과정을 통해 의미론적 붕괴(Semantic Collapse)를 탐구함.

---

## 2. 핵심 모델 및 기술적 토대 (Foundational Models)

| 모델 | 연구 기관 | 주요 논문 / 기술 | AudioArt에서의 활용 의미 |
| :--- | :--- | :--- | :--- |
| **RAVE** | ACIDS-IRCAM | [arXiv:2111.05011](https://arxiv.org/pdf/2111.05011) | 실시간 VAE; 연속적 잠재 매니폴드 탐색 및 모핑 |
| **EnCodec** | Meta AI | [arXiv:2210.13438](https://arxiv.org/pdf/2210.13438) | RVQ 기반 신경망 코덱; 이산 토큰의 직접적인 변조 |
| **DAC** | Descript | [GitHub](https://github.com/descriptinc/descript-audio-codec) | 범용 도메인 고압축 코덱; 멀티 스케일 토큰 조작 |
| **Mimi** | Kyutai Labs | [Blog](https://kyutai.org/codec-explainer) | 스트리밍 코덱; 의미론적(Semantic) 토큰과 음향적 토큰의 공존 |
| **AudioLDM 2** | CVSSP | [arXiv:2308.05734](https://arxiv.org/pdf/2308.05734) | 텍스트-오디오 잠재 확산 모델; 재해석 루프를 통한 붕괴 |
| **Qwen2-Audio** | Alibaba Cloud | [arXiv:2407.10759](https://arxiv.org/pdf/2407.10759) | 멀티모달 오디오-언어 모델; 캡셔닝을 이용한 시맨틱 드리프트 |

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

## 5. 오픈소스 도구 및 프레임워크

| 도구 | 용도 | AudioArt에서의 역할 |
| :--- | :--- | :--- |
| **nn_tilde** | Max/MSP용 실시간 추론 플러그인 | RAVE 모델의 실시간 제어 인터페이스 |
| **AudioCraft** | Meta의 통합 라이브러리 | EnCodec 및 MusicGen 연구 및 렌더링 |
| **AFTER** | IRCAM의 잠재 확산 프레임워크 | 고차원적인 사운드 합성 및 리믹싱 |
| **Diffusers** | HF 확산 모델 라이브러리 | AudioLDM 2 기반의 캡션 루프 구현 |

---

## 6. 결론: 향후 연구 방향

본 조사 결과, 신경망 코덱을 이용한 실시간 사운드 아트는 이미 현대 전자음악의 최전선에 서 있음을 확인했습니다. AudioArt는 단순히 개별 모델을 사용하는 것을 넘어, **Net Max**나 **Meta-Symphony**와 같이 **"네트워크 간의 복합적인 상호작용"**을 체계적으로 설계하고 문서화했다는 점에서 기존 선행 연구들과 차별화되는 독창성을 가집니다.

향후에는 **Mimi 코덱**과 같이 의미론적 토큰을 다루는 최신 모델을 MVP-J로 통합하여, 텍스쳐뿐만 아니라 '언어적 음악'의 붕괴까지 탐구할 예정입니다.

---
**마지막 업데이트:** 2026년 5월  
**조사 대상:** 신경망 오디오 코덱, 생성형 AI 사운드 아트, 모델 붕괴 이론  
