# AudioArt: Detailed Research Survey (2024-2026)

이 문서는 AudioArt 프로젝트의 기술적, 예술적 정당성을 확보하기 위해 현대 신경망 사운드 아트의 최전선에 있는 프로젝트들을 아키텍처 수준에서 상세히 분석합니다.

> **심화 조사:** 본 케이스 스터디의 검증 결과 + 신규 65 개 항목 (모델 9, 이론 7, 예술가 6, 컨퍼런스 6 등) 은 [`UPDATES_V2.md`](UPDATES_V2.md) 에 정리. 일부 진술의 정정 (예: Björk 협업자 "Robin Meier" → "Robin Meier Wiratunga") 과 검증 부족 항목 표기 (`(미확인)`) 도 포함.

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

### 3.2 모델 가중치 병합과 모드 연결성 (MVP-D 의 학술적 정당성)

*   **Git Re-Basin (Ainsworth, Hayase, Srinivasa, ICLR 2023, [arXiv:2209.04836](https://arxiv.org/abs/2209.04836)):** 독립적으로 학습된 두 네트워크가 weight 공간에서 *서로 다른 basin* 에 위치해 단순 보간이 무너지지만, **hidden unit 의 순열 (permutation symmetry)** 을 정렬하면 동일 basin 으로 끌어올 수 있다는 주장. 알고리즘 3 가지 (Activation Matching, Weight Matching, Straight-Through Estimator) 제시.
*   **Linear Mode Connectivity (Frankle, Dziugaite, Roy, Carbin, ICML 2020, [arXiv:1912.05671](https://arxiv.org/abs/1912.05671)):** 같은 초기값에서 출발한 두 학습 trajectory 는 weight 공간에서 선형 경로로 연결됨. *공유 init 두 RAVE 학습* 권고의 이론 토대.
*   **Model Soups (Wortsman et al., ICML 2022, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)):** 여러 fine-tune 결과의 가중치 평균이 단일 best 보다 일반화에 견고. V2 의 *Meta-Symphony Soup* — Net 1–4 의 가중치 평균 ensemble — 의 토대.
*   **AudioArt 의 실증:** `src/modules/mvp_d/re_basin.py` (partial — inner-block 정렬, 21 resblock × grouped conv 인지) + `re_basin_full.py` (full — encoder chain 클래스 4종 + inner 7종 + tied-group 의미론). 함수 보존은 random-perm 검증으로 확인되었으나, **morph collapse 는 여전히 해소되지 않음** — decoder + gimbal + prior_net + latent_pca 도 정렬해야 함. `cliff_sweep_guitar_organ` 19 점에서 endpoint 영역 `t ∈ [0, 0.02] ∪ [0.98, 1]` 만이 유의미한 음악적 영역으로 확인됨.

### 3.3 비-자동회귀 토큰 생성 (Net Max / V2 빠른 재생성 토대)

*   **MAGNeT (Meta, [arXiv:2401.04577](https://arxiv.org/abs/2401.04577)):** masked non-autoregressive 토큰 생성으로 MusicGen 대비 약 7× 빠른 속도. text-to-music 의 *디코딩 병목* 을 깬다.
*   **VampNet (Adobe + NU, ISMIR 2023, [arXiv:2307.04686](https://arxiv.org/abs/2307.04686)):** 마스크된 RVQ 토큰의 parallel iterative refinement. AudioArt 의 *MVP-C bend* 가 토큰을 *손상* 시킨다면 VampNet 은 *마스크 + 재생성* — 손상 후 복원의 학술적 사촌.
*   **AudioArt 와의 연결:** V2 의 *Meta-Symphony 빠른 재생성* (현재 3 분 wall time → 30 초 목표) 에서 MAGNeT/VampNet 디코딩을 차용할 수 있음. 특히 *MVP-C 의 invalid_token 회복* 을 마스크 모델로 우회하면 더 풍부한 재구성 텍스처가 가능.

---

## 4. 요약: AudioArt의 위치 (Positioning)

| 연구/프로젝트 | 주안점 | AudioArt 의 차별점 |
| :--- | :--- | :--- |
| **AFTER / The Call** | 목소리/스타일 전이 | **비-인간적 텍스쳐** — 악기가 아닌 신경망 에러 자체의 미학 탐구 |
| **Nature Manifesto** | 저전력/환경 설치 | **Massive Scale** — 1 시간 분량 대작 렌더링용 SoX 디스크 스트리밍 파이프라인 |
| **Studies for** | 고속 생성 / 쿼리 제어 | **Macro-Architecture** — Net Max, Meta-Symphony 같은 복합 네트워크 위상 설계 |

## 5. AudioArt 의 매크로 네트워크 — 선행 사례에 대한 답

### 5.1 Net Max — 9 MVP 의 cross-feedback 통합
선행 사례들은 대부분 *단일 변환*이거나 *2 종 모델의 결합*에 머문다. AudioArt 의 Net Max 는:
- 9 개의 손상 양식 (latent perturb / caption loop / token bend / weight morph / granular memory / spectral freeze / latent feedback / generative drone / bass smear) 을 동시에 운용.
- 8 버스 (α~θ) 의 cross-feedback (θ 가 β 의 출력을 tap) 로 *네트워크 안의 네트워크* 구조.
- 2-pass 매크로 루프로 자기 강화.

AFTER 의 "실시간 질감 전이" 가 *단일 차원* 의 transfer 라면, Net Max 는 9 차원이 동시에 작동하는 *다축 transfer*.

### 5.2 Net Dynamic — 22 dB 다이내믹의 작곡적 격상
Björk 의 Nature Manifesto, Studies for 시리즈가 *지속 가능한 텍스쳐 흐름* 에 초점을 두는 반면, AudioArt 의 Net Dynamic:
- 같은 8 버스 구성에서 시간 가변 envelope + filter sweep + 3 impulse 이벤트 (15 s freeze CLICK, 30 s SILENCE DROP, 45 s drone BURST) 로 22 dB 다이내믹 레인지 달성.
- 정적 텍스쳐를 *작곡* 의 차원으로 격상.

### 5.3 Meta-Symphony — 매크로넷의 네트워크
Holly Herndon 의 "The Call" 이 단일 모델 (Linked Diffusion) 의 multi-input 통합이라면, AudioArt 의 Meta-Symphony 는:
- 4 개의 매크로넷 (Net 1 / 2 / 3 / Dynamic) 자체를 stem 으로 사용.
- 3 분 stereo 타임라인 위 LFO crossfade (60 s / 45 s, LCM 180 s = 곡 길이) + 스테레오 pan drift (20 s / 25 s, LCM 100 s).
- 100 Hz Butterworth 2 차 LPF + 시드 +8 dB sub-boost 재주입으로 anchored.

---

## 6. 향후 작업 — AudioLLM 단계 (V2)

본 케이스 스터디 시점 (2026 년 5 월) 기준 AudioArt 는 V1 = *뉴럴 사운드 아트* 영역에 해당. 다음 V2 단계 = *AudioLLM 통합* 영역:

| V1 (현재) | V2 (향후) |
|---|---|
| MVP-B stub (deterministic 형용사 + FM 합성) | MVP-B 실제 백본 (Qwen2-Audio + AudioLDM2) — Shumailov 의 model collapse 이론 실증 |
| Texture Governor (NaN + RMS + flatness + centroid) | + **Semantic Governor** — 캡션 표류 임계치 기반 자동 wet 감쇠 |
| 4 stem Meta-Symphony | 5 stem Meta-Symphony v2 — `stem_LLM` 추가 |
| RAVE latent perturbation | + AudioLLM 임베딩 공간 perturbation (의미적 noise) |
| 단일 매크로넷 토폴로지 | + `net_semantic`, `net_llm_chain`, `net_prompt_morph` — 캡션이 다른 net 의 파라미터 시간 가변 결정 |
| 4 모델 통합 (RAVE × 2, EnCodec, ckpt morph) | + Mimi codec (MVP-J) — 의미적 + 음향적 토큰 분리 손상 |

### 6.1 AudioArt V2 가 선행 사례에 다시 답하는 방식

| 선행 사례 | V2 의 추가 답 |
|---|---|
| Holly Herndon **Spawn / The Call** | AudioLLM 캡션이 다음 단계 시드 생성을 *조건화* 하는 cross-modal loop — 인간 입력의 자리에 의미적 노이즈가 들어감 |
| **Dadabots** SampleRNN 24-시간 스트림 | Semantic Governor 가 24-시간 스트림 중 *의미 표류* 까지 모니터링, 너무 멀어지면 자동 시드 재주입 |
| Yuri Suzuki **Sonic Pendulum / Vox PopulA.I** | 청중의 발화 → AudioLLM 캡션 → 다른 매크로넷 파라미터 조건화 → 실시간 다축 변형 |

---
**업데이트:** 2026년 5월  
**분석 키워드:** Latent Diffusion, Adversarial Disentanglement, Masked Generative Modeling, Model Collapse, Cross-modal Conditioning.  
