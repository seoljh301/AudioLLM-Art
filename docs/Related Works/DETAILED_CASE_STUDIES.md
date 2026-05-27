# AudioArt: Detailed Research Survey (2024-2026)

이 문서는 AudioArt 프로젝트의 기술적, 예술적 정당성을 확보하기 위해 현대 신경망 사운드 아트의 최전선에 있는 프로젝트들을 아키텍처 수준에서 상세히 분석합니다.

---

## 1. 실시간 스타일 전이와 구조 분리 (Structure vs Timbre)

현대 신경망 사운드 아트의 가장 큰 화두는 **구조(구문, 피치, 리듬)**와 **음색(질감)**을 어떻게 독립적으로 제어하느냐에 있습니다.

### 1.1 IRCAM의 AFTER Framework (2024-2025)
*   **기술적 명칭:** Audio Features Transfer and Exploration in Real-time
*   **핵심 아키텍처:**
    *   **Latent Diffusion (LDM):** RAVE나 EnCodec과 같은 VAE의 잠재 공간 내부에서 확산 모델을 구동합니다.
    *   **적대적 분리 (Adversarial Disentanglement):** 학습 과정에서 음색 인코더가 피치 정보를 포함하지 못하도록 적대적 손실(Adversarial Loss)을 사용하여, 사용자가 소리의 높낮이를 유지하면서 질감만 바꿀 수 있게 합니다.
*   **AudioArt와의 연결:** MVP-A의 잠재 공간 섭동이 정적이라면, AFTER는 이를 확산 모델로 정교하게 조정합니다. AudioArt의 **Meta-Symphony**에서 사용되는 LFO 크로스페이드는 AFTER가 추구하는 '실시간 질감 전이'를 매크로 수준에서 구현한 것입니다.

### 1.2 Holly Herndon & Mat Dryhurst: "The Call" (2024)
*   **모델:** **Linked Diffusion** (AFTER 프레임워크의 커스텀 구현체)
*   **작동 원리:** 15개의 영국 지역 합창단 데이터를 학습한 **Choral Data Trust** 모델을 사용합니다. 관객의 목소리(구조)를 입력받아 실시간으로 합창단의 음색(질감)으로 변환(Singing Voice Transfer)합니다.
*   **하드웨어:** **"The Hearth"**라 불리는 120개의 GPU 팬으로 구성된 악기를 통해 AI의 '연산 과정'을 물리적 소리로 시각화합니다.

---

## 2. 생성 모델의 속도와 효율성 (Frugal & Fast Generation)

고품질의 소리를 실시간으로 생성하기 위해 확산 모델의 느린 속도를 극복하려는 시도들이 이어지고 있습니다.

### 2.1 Björk: "Nature Manifesto" (2024)
*   **핵심 기술:** **Frugal AI (저전력 AI)**
    *   **RAVE** 코덱의 극단적인 효율성을 활용하여 **별도의 GPU 없이 로컬 CPU만으로** 실시간 추론을 수행합니다.
    *   멸종 위기 동물의 소리를 재구성하기 위해 AFTER 모델을 사용하며, 인간의 목소리나 다른 동물의 소리를 '구조'로 삼아 사라진 생명체의 '음색'을 입힙니다.
*   **AudioArt와의 연결:** AudioArt가 V100 GPU 환경에서 1시간 분량의 대작을 뽑아내는 '고성능 연산'에 집중한다면, 비요크의 사례는 동일한 모델(RAVE)이 환경 설치 미술에서 '지속 가능성'을 어떻게 확보하는지 보여줍니다.

### 2.2 Sony AI & Evala: "Studies for" (2024-2025)
*   **모델:** **SpecMaskGIT** (Masked Generative Modeling)
    *   **SpecVQGAN + Masked Transformer:** 멜-스펙트로그램을 이산 토큰으로 압축한 뒤, 비결정론적(Non-autoregressive) 방식으로 토큰을 예측합니다.
    *   **성능:** 표준 확산 모델보다 훨씬 빠른 **16번의 반복(Iterations)** 만에 10초 분량의 48kHz 고품질 오디오를 생성합니다.
    *   **CLAP Embedding:** 텍스트나 오디오 쿼리를 '조건부 마스크'로 사용하여 생성 과정을 실시간으로 조타(Steering)합니다.
*   **AudioArt와의 연결:** AudioArt의 **MVP-H(Codebook Organ)**가 수학적 수열로 토큰을 직접 찍어낸다면, Sony의 방식은 CLAP을 통해 이를 예술가의 의도와 연결하는 고수준의 제어를 보여줍니다.

---

## 3. 신경망 데이터벤딩의 전통 계승 (Glitch to Codec)

물리적 손상에서 데이터 손상으로, 다시 신경망의 '오해'로 이어지는 계보입니다.

### 3.1 모델 자식 작용 (Model Autophagy / Collapse)
*   **학술적 배경:** Shumailov et al. (2024)의 연구에 따르면, 모델이 자신의 생성물을 다시 학습할 때 데이터의 꼬리(Tails) 부분이 사라지는 '붕괴' 현상이 발생합니다.
*   **예술적 적용:** **MVP-B (Caption Loop)**는 이 붕괴를 작품의 엔진으로 사용합니다. 캡셔너(Qwen2-Audio)가 소리를 '오해'하여 텍스트로 바꾸고, 다시 생성기(AudioLDM2)가 그 텍스트를 소리로 바꾸는 과정은 '의미론적 붕괴'가 만들어내는 추상적 서사를 보여줍니다.

---

## 4. 요약: AudioArt의 위치 (Positioning)

| 연구/프로젝트 | 주안점 | AudioArt의 차별점 |
| :--- | :--- | :--- |
| **AFTER / The Call** | 목소리/스타일 전이 | **비-인간적 텍스쳐**: 악기가 아닌 신경망 에러 자체의 미학 탐구 |
| **Nature Manifesto** | 저전력/환경 설치 | **Massive Scale**: 1시간 분량의 대작 렌더링을 위한 SoX 기반 파이프라인 |
| **Studies for** | 고속 생성 / 쿼리 제어 | **Macro-Architecture**: Net Max, Meta-Symphony와 같은 복합적인 네트워크 위상 설계 |

---
**업데이트:** 2026년 5월  
**분석 키워드:** Latent Diffusion, Adversarial Disentanglement, Masked Generative Modeling, Model Collapse.  
