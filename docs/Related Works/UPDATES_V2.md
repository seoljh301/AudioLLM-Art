# AudioArt V2 Related Works 확장 및 검증 보고서

**작성일:** 2026년 5월 27일  
**검증 대상:** RESEARCH_SURVEY.md, DETAILED_CASE_STUDIES.md  
**목표:** 신경망 오디오 파운데이션 모델, 미학 이론, 음향예술 역사, 학술 토대의 심화 조사

---

## 1. 기존 항목 검증 결과 (Verification Report)

### 표: 기존 진술의 사실 검증

| 항목 | 검증 상태 | 출처 | 수정/보충 사항 |
|---|---|---|---|
| **Holly Herndon "The Call" (2024) @ Serpentine Gallery** | ✅ 검증됨 | Serpentine Galleries 공식 웹사이트 | 정확. 2024년 10월 4일 ~ 2025년 2월 2일 개최. 15개 영국 지역 합창단 데이터 사용 확인. |
| **Björk "Nature Manifesto" (2024) @ Pompidou** | ✅ 검증됨 (부분 수정) | Centre Pompidou 공식 안내 | "Robin Meier"가 아니라 "Robin Meier Wiratunga"가 정확한 이름. IRCAM AFTER 기술 사용 확인. 11월 20일 ~ 12월 9일 개최. |
| **Evala "Studies for" (2024) @ NTT ICC Tokyo** | ✅ 검증됨 | arXiv:2510.25228, Sony AI 블로그 | 정확. 12월 2024 ~ 3월 2025, 8채널 설치. SpecMaskGIT 아키텍처 확인. 200시간 아카이브 데이터 학습 검증. |
| **IRCAM AFTER Framework (2024-2025)** | ✅ 검증됨 | GitHub: acids-ircam/AFTER, 논문 "Combining audio control and style transfer using latent diffusion" (Aug 2024) | 정확. Nils Demerlé 리드. 실시간 timbre transfer + MIDI 제어 확인. |
| **Yasunao Tone "Wounded CD" (1984~1997)** | ✅ 검증됨 | Academia.edu, Wikipedia, Blank Forms | 정확. 1984년 CD 수정 시작, 1997년 "Solo for Wounded CD" 녹음. Fluxus 멤버 확인. |
| **Oval/Markus Popp CD Glitching (1990s)** | ✅ 검증됨 | Wikipedia, Bandcamp | 정확. Systemisch (1994)가 이정표. 손상된 CD에 칼, 테이프, 페인트 사용 검증. |
| **RAVE 모델 (2021~)** | ✅ 검증됨 | arXiv:2111.05011, GitHub | 정확. ACIDS-IRCAM, v2에서 Snake activation 포함 확인. |
| **EnCodec (Meta AI, 2022~)** | ✅ 검증됨 | arXiv:2210.13438, GitHub | 정확. RVQ 기반, 실시간 고충실도 코덱 검증. |
| **DAC (Descript)** | ✅ 검증됨 | GitHub: descriptinc/descript-audio-codec | 정확. 범용 도메인 고압축 코덱 확인. |
| **AudioLDM 2 (CVSSP)** | ✅ 검증됨 | arXiv:2308.05734 | 정확. 텍스트-오디오 잠재 확산 모델 검증. |
| **Qwen2-Audio** | ✅ 검증됨 | arXiv:2407.10759 | 정확. 멀티모달 오디오-언어 모델. |
| **Dadabots 24-hour SampleRNN** | ✅ 검증됨 | GitHub, dadabots.com | 정확. 기술적 사망금속, 자유 재즈, NASA Voyager 스트림 확인. |

---

## 2. 신규 항목: 오디오 파운데이션 모델 (New Audio Foundation Models)

### 2.1 Kyutai Mimi Codec (2024)

- **기관:** Kyutai Labs
- **출시일:** 2024년 9월 17일 (Hugging Face 추가: 9월 18일)
- **핵심 특징:** 의미론적(semantic) + 음향적(acoustic) 듀얼 토큰 스트림
- **기술 상세:**
  - 12.5 Hz 프레임율, 1.1 kbps 비트레이트
  - WavLM 기반 의미론적 토큰 증류
  - 16개 코드북의 잔여 벡터 양자화 (RVQ)
  - 완전 인과(causal) 설계 = 스트리밍 Transformer에 최적
- **AudioArt와의 연관성:** MVP-J 계획 ("Mimi 코덱 통합") — 의미적 토큰과 음향 토큰을 분리하여 손상시키는 새로운 차원의 databending. V2에서 "stem_LLM"의 토대.
- **레퍼런스:** [Kyutai Mimi HuggingFace](https://huggingface.co/kyutai/mimi), [Moshi 대화 모델](https://github.com/kyutai-labs/moshi)

### 2.2 Stability AI Stable Audio Open (2024)

- **출시일:** 2024년 (Stable Audio Open 1.0)
- **최신:** Stable Audio 3.0 (2026년 5월 기준 6분 곡 생성 능력)
- **핵심 명세:**
  - 변수 길이 오디오 생성 (최대 47초 for v1.0, 6분 for v3.0)
  - 44.1kHz 스테레오, Creative Commons 라이선스 데이터로 학습
  - T5 텍스트 인코딩 + 잠재 공간 DiT (Diffusion Transformer)
- **교육 데이터:** 약 50만 개 CC-0/CC-BY/CC-Sampling+ 녹음 (Freesound + FMA)
- **AudioArt와의 연관성:** 오픈 소스 기반 생성 모델의 참고. Stable Audio Open은 소비자 수준 GPU에서 실행 가능 (환경 설치 미술의 "지속 가능성" 관점).
- **레퍼런스:** [Stability AI 공식](https://stability.ai/research/stable-audio-open), [HuggingFace 모델](https://huggingface.co/stabilityai/stable-audio-open-1.0)

### 2.3 Meta AudioCraft: MusicGen, AudioGen, MAGNeT (2023-2024)

#### MusicGen
- **출시:** 2023년 8월
- **학습 데이터:** 20,000시간 라이선스 음악 (Meta 자체 + Shutterstock + Pond5)
- **아키텍처:** 32 kHz EnCodec 토크나이저 (4 코드북, 50 Hz)에 대한 단계 자동회귀 Transformer
- **특징:** 텍스트 + 멜로디 조건부 생성

#### AudioGen
- **초점:** 공개 음향효과, 환경음 생성
- **기술:** MusicGen과 동일 기반 아키텍처

#### MAGNeT (2024)
- **발표:** 2024년 1월 (EMNLP 2023 후 일반 공개)
- **혁신:** 순수 비자동회귀(non-autoregressive) 방식
- **성능:** 기존 모델 대비 **7배 빠름** (30초 곡 < 1초)
- **아키텍처:** 다중 오디오 토큰 스트림에 대한 마스크된 Transformer
- **모델 크기:** 300M ~ 1.5B 파라미터 변형

**AudioArt와의 연관성:** Net Dynamic의 "22 dB 다이내믹 범위 달성"을 위한 빠른 재생성 기반. AudioCraft는 RAVE + EnCodec의 토큰화 기초를 제공.

**레퍼런스:** [Meta AudioCraft](https://ai.meta.com/resources/models-and-libraries/audiocraft/), [MAGNeT 문서](https://github.com/facebookresearch/audiocraft/blob/main/docs/MAGNET.md)

### 2.4 Adobe VampNet (2023)

- **발표:** ISMIR 2023
- **기술:** 마스크된 음향 토큰 모델링 (Masked Acoustic Token Modeling)
- **아키텍처:** 비자동회귀 양방향 Transformer
- **샘플링 속도:** ~36 스텝 (자동회귀 모델은 수백 스텝)
- **기능:** 압축, inpainting, outpainting, 연속(continuation), 루핑/vamping
- **개발자:** Northwestern University + Descript Inc.

**AudioArt와의 연관성:** Token databending (MVP-C, I)의 마스크 기반 재생성 이론. 토큰 손상 후 컨텍스트 복원 메커니즘 탐구.

**레퍼런스:** [arXiv:2307.04686](https://arxiv.org/abs/2307.04686), [GitHub](https://github.com/hugofloresgarcia/vampnet)

### 2.5 NVIDIA Audio Flamingo 2 (2024-2025)

- **발표:** 2024년 초, 개선된 버전 2025년
- **명칭:** Audio-Language Model (ALM)
- **성능:**
  - 3B 소규모 LLM 기반
  - 20+ 벤치마크에서 SOTA
  - 5분까지의 장형 오디오 이해
  - Expert reasoning 능력
- **학습:** 공개 데이터셋만 사용 (AudioSkills, LongAudio)
- **기반 LLM:** Qwen-2.5

**AudioArt와의 연관성:** AudioLLM 백본 후보. Semantic Governor 설계의 참고. MVP-B 의미론적 손상의 명확성 메커니즘.

**레퍼런스:** [NVIDIA ADLR 공식](https://research.nvidia.com/labs/adlr/AF2/)

### 2.6 NVIDIA GAMA (2024)

- **정식 명칭:** General-purpose Large Audio-Language Model (LALM) with Advanced Reasoning
- **발표:** EMNLP 2024 (11월)
- **성능:** 다른 모든 LALM 대비 1% ~ 84% 우수
- **혁신:** 
  - 복합 추론 능력
  - CompA-R 데이터셋 기반 튜닝
  - Audio Q-Former를 통한 다중 오디오 표현 통합
- **평가:** CompA-R-test 제안 (인간 레이블 복합 추론 평가 세트)

**AudioArt와의 연관성:** AudioLLM 기반 조건부 매크로넷 설계 (V2: `net_semantic`, `net_llm_chain`, `net_prompt_morph`).

**레퍼균:** [ACL Anthology](https://aclanthology.org/2024.emnlp-main.361/)

### 2.7 ByteDance SALMONN (2024) & Video-SALMONN 2 (2025)

- **SALMONN (원본):** ICLR 2024 수용 (2024년 10월 발표)
- **기술:** Speech Audio Language Music Open Neural Network
- **기능:** 
  - ASR (자동 음성 인식)
  - 번역, 감정 인식, 화자 검증
  - 음악 + 오디오 캡셔닝
  - 청각 정보 기반 QA
- **통합:** Whisper 스타일 음성 인코더 + LLM

- **Video-SALMONN 2 (2025):** 오디오-비주얼 대형 언어 모델
  - 고품질 비디오 캡셔닝
  - MrDPO (Multi-round DPO) + 캡션 품질 목표
  - GPT-4o, Gemini-1.5-Pro 대비 우수

**AudioArt와의 연관성:** MVP-B (Caption Loop) 의 실제 백본 후보. 다중모달 손상 루프 (Audio→Text→Audio) 구현.

**레퍼런스:** [GitHub](https://github.com/bytedance/SALMONN), [arXiv:2310.13289](https://arxiv.org/abs/2310.13289)

### 2.8 MIT LTU-AS (Listen, Think, and Understand Audio & Speech) (2023-2024)

- **발표:** ICLR 2024 수용
- **개발:** MIT CSAIL + MIT-IBM Watson AI Lab
- **통합 구성:**
  - Whisper (음성 인식)
  - LLaMA (추론)
- **능력:**
  - 음성 텍스트 + 음성 준언어학(paralinguistics) + 비음성 오디오 동시 이해
  - Open-ended 질문 답변
  - 명령 추종 > 95% (GPT-4 평가)
- **데이터:** OpenAQA-5M 증강 → 9.6M Open-ASQA
- **용도:** 음성 + 오디오 결합 추론이 필요한 작업

**AudioArt와의 연관성:** Semantic Governor 설계, 캡션 텍스트 "표류" 감지.

**레퍼런스:** [arXiv:2305.10790](https://arxiv.org/abs/2305.10790), [GitHub](https://github.com/YuanGongND/ltu)

### 2.9 Suno & Udio (상업용 곡 레벨 생성) (2023-2026)

- **Suno:**
  - 완성된 곡 생성 (보컬 + 가사 + 악기법)
  - 30~60초 생성 시간
  - Pro 플랜: 상업 재사용권 포함
  - v5.5 (최신)

- **Udio:**
  - Suno의 경쟁 플랫폼
  - 곡 생성 능력 유사
  - 2026년 기준 플랫폼 내 콘텐츠 잠금

**AudioArt와의 연관성:** V2 메타심포니의 스템 대체 가능성, 상용 곡 레벨 장형 생성의 법적 모델. AudioLLM 통합 후 생성 가능한 상품화 경로 검토.

**레퍼런스:** [Suno.com](https://suno.com/), [Udio](https://www.udio.com/)

---

## 3. 신규 항목: 학술 이론 토대 (Academic Theory)

### 3.1 Git Re-Basin: Merging Models modulo Permutation Symmetries

- **저자:** Ainsworth, S. K., Hayase, J., Srinivasa, S.
- **발표:** ICLR 2023 (arXiv:2209.04836, 2022년 제출)
- **핵심 발견:**
  - 신경망 손실 경지(loss landscape)는 순열 대칭성을 고려하면 **단일 기저(single basin)**를 포함
  - 세 가지 알고리즘으로 두 모델을 align하여 가중치 공간에서 병합
  - 넓은 모델(wider models)은 선형 모드 연결성 더 우수
- **실험:** CIFAR-10 CNN, ResNet
- **임의 사항:** Permutation 찾기는 Hungarian algorithm 기반

**AudioArt와의 연관성:** MVP-D (Checkpoint Morphing) 의 이론적 정당성. **"partial (inner-block) + full (encoder-chain) 두 구현 모두 시도, mode-connectivity 한계 정량 측정 (cliff sweep 19 점)"** 의 과학적 배경.

**레퍼런스:** [arXiv:2209.04836](https://arxiv.org/abs/2209.04836), [GitHub](https://github.com/snimu/rebasin)

### 3.2 Linear Mode Connectivity and the Lottery Ticket Hypothesis

- **저자:** Frankle, J., Dziugaite, G. K., Roy, D., Carbin, M.
- **발표:** ICML 2020
- **주요 결과:**
  - 같은 초기화에서 시작한 다른 최솟값들은 선형 경로로 연결
  - 가중치 가지치기(pruning) 중에도 유지
  - 선형 연결된 해답들 = 동일한 로또 티켓

**AudioArt와의 연관성:** Git Re-Basin 의 선행 이론. 모델 모핑의 "선형 방향" 가정의 기반.

**레퍼런스:** [ICML 2020 논문](http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf)

### 3.3 Model Soups: Averaging Weights of Multiple Fine-tuned Models

- **저자:** Wortsman, M., Ilharco, G. et al.
- **발표:** ICML 2022 (arXiv:2203.05482)
- **핵심:**
  - 여러 하이퍼파라미터로 미세조정(fine-tuned)된 모델들의 가중치 평균화
  - 추론 비용 0 증가
  - 단일 모델 선택보다 개선 (특히 대형 기초 모델)
- **응용:** CLIP, ALIGN, ViT-G on ImageNet. 분포 외(OOD) 성능 + zero-shot 성능 개선
- **용어:** "Soup" = 앙상블 없는 weight averaging

**AudioArt와의 연관성:** 여러 MVP의 가중치를 혼합하는 Meta-Symphony 아키텍처의 학술 정당성. Net 1-4의 크로스페이드가 "Soup"의 음향 버전.

**레퍼런스:** [ICML Proceedings](https://proceedings.mlr.press/v162/wortsman22a.html), [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)

### 3.4 The Curse of Recursion: Training on Generated Data Makes Models Forget

- **저자:** Shumailov, I., Shumaylov, Z., Zhao, Y., Gal, Y., Papernot, N., Anderson, R.
- **발표:** arXiv:2305.17493 (2023년 5월 제출, 2024년 4월 최종)
- **핵심 발견:** "Model Collapse"
  - 모델이 자신의 생성 데이터로 재학습 → 원본 분포의 꼬리(tails) 부분 소실
  - VAE, GMM, LLM에서 발생
  - 완전 합성 루프 = 가장 심함
  - "신선한 데이터" 루프만 붕괴 회피
- **함의:** LLM이 인터넷 데이터의 대부분을 차지하면 어떻게 될까?

**AudioArt와의 연관성:** **MVP-B (Caption Loop)의 근간**. Audio ↔ Text ↔ Audio 재귀는 의도적 "붕괴"를 미학적 재료로 활용. V2 단계에서 Shumailov 이론을 직접 실증할 계획.

**레퍼런스:** [arXiv:2305.17493](https://arxiv.org/abs/2305.17493), [Rice DSP 분석](https://dsp.rice.edu/ai-loops/)

### 3.5 Self-Consuming Generative Models Go MAD

- **저자:** Alemohammad, S., Casco-Rodriguez, J., Luzi, L., et al.
- **발표:** ICLR 2024 (arXiv:2307.01850, 2023년 7월)
- **분석 대상:** 3가지 자식 작용(autophagous) 루프
  1. **완전 합성 루프** (fully synthetic) = 최악의 붕괴
  2. **합성 증강 루프** (synthetic augmentation)
  3. **신선한 데이터 루프** (fresh data loop) = 안전
- **실험:** 최첨단 생성 이미지 모델로 실증
- **용어:** MAD = "Model Autophagous Degradation"

**AudioArt와의 연관성:** Shumailov과의 평행 연구. 오디오 도메인에서의 자식 작용 구조 설계 (MVP-B, V2 체계화).

**레퍼런스:** [arXiv:2307.01850](https://arxiv.org/abs/2307.01850)

### 3.6 Snake Activation Function for Audio Synthesis

- **저자:** Liu, H., et al. (원래 제안)
- **함수:** `x + sin²(αx) / α`
- **특성:** 주기성 귀납 편향(periodic inductive bias) + anti-aliasing
- **응용:** RAVE v2, neural vocoders (BigVGAN 등)
- **효과:** 필터링된 비선형성으로 고주파 아티팩트 감소

**AudioArt와의 연관성:** RAVE latent perturbation (MVP-A, E, F, G)의 잠재 공간이 Snake activation으로 구성. 유기적 음색 변화의 수학적 기반.

### 3.7 BigVGAN: A Universal Neural Vocoder with Large-Scale Training

- **저자:** NVIDIA ADLR
- **발표:** ICLR 2023
- **혁신:**
  - 주기 활성화 함수 + anti-aliased representation
  - 112M 파라미터 (사상 최대 규모)
  - LibriTTS 청정 음성만으로 학습
- **성능:**
  - 24 kHz 음성 44.72배 실시간(GPU)
  - 제로샷 조건: 미보인 화자, 언어, 환경, 가성, 음악, 악기음
- **공개:** GitHub NVIDIA/BigVGAN

**AudioArt와의 연관성:** Texture Governor의 "flatness/RMS/centroid" 모니터링이 BigVGAN 같은 고급 보코더의 안정성 입력. V1에서 "NaN 도달 0회"를 달성한 기반.

**레퍼런스:** [arXiv:2206.04658](https://arxiv.org/abs/2206.04658), [NVIDIA 공식](https://research.nvidia.com/labs/adlr/projects/bigvgan/)

---

## 4. 신규 항목: 사운드 아트 & 글리치 계보 (Sound Art & Glitch Lineage)

### 4.1 Ryoji Ikeda (데이터 소니피케이션)

- **출생:** 1966년 기후현(日本)
- **활동:** 파리 + 교토 기반
- **특징:** 극단적 미니멀리즘, 정현파 + 데이터 + 수학 패턴
- **주요 작업:**
  - **"spectra II"** (2002): 글리치 마이크로사운드 설치
  - **"Test Pattern"** (2008): 텍스트/사진/영상/음성 → 깜빡이는 바코드 패턴 + 싱크 음악
  - **"matryoshka"**, **"superposition"** 등
- **기술:** 데이터를 청각시각적 코드로 변환하는 공학적 미학

**AudioArt와의 연관성:** Token databending (MVP-C, I)의 "데이터 시각화" 대응물. 신경망 토큰의 비트 플립을 음향예술로 표현하는 이론적 선행자.

**레퍼런스:** [Ryoji Ikeda 전시 분석](https://online.ucpress.edu/afterimage/article/51/4/65/204742/)

### 4.2 Mira Calix (상호작용 음향 설치)

- **주요 작품:**
  - **"Inside There Falls"** (2015, Carriageworks Sydney): 180채널 오케스트라 스코어 + 손으로 으깬 종이(1.5 km)에 임베드된 스피커 + 댄서들의 숨겨진 스피커 착용
  - **"Nothing Is Set in Stone"** (2012, London Olympics): 22개 Meyer Sound 스피커 임베드 + 모션 센서 + 9명 합창 + 전자 텍스처 + 자연음
- **방법론:** 음향을 조각적 재료로, 사용자 근접도에 따른 반응형 생성
- **역사:** 1990년대부터 디지털 음향 설치 개척자

**AudioArt와의 연관성:** Meta-Symphony의 공간 반응형 구현 모델. LFO crossfade + 스테레오 drift가 Calix의 "proximity-based spatialisation"의 계산적 버전.

**레퍼런스:** [Mira Calix VICE 인터뷰](https://www.vice.com/en/article/artist-mira-calix-creates-a-labyrinth-of-sound-and-storytelling-out-of-paper-2/)

### 4.3 Dadabots: 24-시간 AI 스트림 (2017~)

- **기술:** SampleRNN (unconditional, raw audio time-domain)
- **장르:** 기술적 사망금속(technical death metal), 자유 재즈, DnB
- **주요 스트림:**
  - YouTube 24시간 기술적 사망금속
  - Twitch 24시간 AI 피아노 (Melodrive 협력)
  - NASA Voyager 3: John Coltrane "Interstellar Space" 학습 → 24/7 자유 재즈 방송
- **영향:** 초기 장형 생성 음악 precedent

**AudioArt와의 연관성:** "3분 stereo Meta-Symphony" vs. "24시간 스트림" 비교. V2의 Semantic Governor는 Dadabots의 24시간 표류 방지 메커니즘과 병렬.

**레퍼런스:** [Dadabots 공식](https://dadabots.com/science/), [GitHub SampleRNN](https://github.com/dada-bots/dadabots_sampleRNN)

### 4.4 Yuri Suzuki (상호작용 호른 조각, AI 생성)

- **Sonic Pendulum:** 음향 펜듈럼 설치 (세부사항 미확인)
- **Vox PopulA.I** (Science Centre Singapore):
  - 5개 호른, 실시간 AI 멜로디/가사 생성
  - 지역 전통 합창음악 영감
- **The Welcome Chorus** (Turner Contemporary):
  - 12개 상호작용 호른
  - 방문객 음성 인식 → 호른이 응답 멜로디 생성
  - 톤 변화, 템포, 단어 감지
- **Arborhythm** (SFMOMA Art of Noise):
  - 혼 모양 스피커
  - 샌프란시스코 피안 + 해사자 + 케이블카 음성 리믹스

**AudioArt와의 연관성:** Yuri Suzuki의 "호른 기반 실시간 음성 조건부 생성"은 V2 `net_prompt_morph`의 선행 사례. 청중 입력 → 캡션 → 매크로넷 파라미터 조건화 구조.

**레퍼런스:** [Yuri Suzuki 공식](https://www.yurisuzuki.com/projects/vox-popula-i)

### 4.5 Holly Herndon의 초기 신경음성 작업: PROTO (2019) & Spawn

- **PROTO** (2019):
  - 최초 노래 신경망(singing neural network) 앨범
  - Spawn = AI agent (Mathew Dryhurst + Jules LaPlace와 공동 개발)
  - SampleRNN: Herndon 음성으로 학습된 비조건부 종단간 신경음성 생성 모델
  - Berlin vocal ensemble 데이터 포함
- **이론적 배경:** 2016년부터 AI research + ensemble composition 병렬 추구
- **이전 앨범:** Movement (2012), Platform

**AudioArt와의 연관성:** MVP-B (Caption Loop)의 "음성 변형" 근원. PROTO의 SampleRNN 손상은 데이터벤딩의 음성 버전.

**레퍼런스:** [Holly Herndon 인터뷰 musictech.com](https://musictech.com/features/interviews/holly-herndon-proto/), [Holly Herndon PROTO Neural](https://neural.it/2020/04/holly-herndon-%E2%80%8E-proto/)

### 4.6 Caterina Barbieri (모듈러 신스 + AI, 2024년 "Spirit Exit")

- **음악 철학:** "Ecstatic Computation"
- **방법:** 모듈러 신스로 미시적 음색 조형 → SuperCollider에서 대체 튜닝 실험
- **최근 작업:** 2022년 "Spirit Exit" 앨범 = 신성 음악 + 트랜스 + **기계 학습** + 정서 강도 혼합
- **하드웨어:** Ableton + multi-layered delay lines

**AudioArt와의 연관성:** Latent Feedback (MVP-G)의 "유기적 진화" 아이디어. Barbieri의 모듈러 자기순환은 신경망의 latent perturbation과 유사한 음향 역학.

**레퍼런스:** [Caterina Barbieri Ableton 블로그](https://www.ableton.com/en/blog/caterina-barbieri-minimalism-modular-and-live/)

---

## 5. 신규 항목: 학술 및 예술 컨퍼런스 (Venues & Conferences)

### 5.1 NIME (New Interfaces for Musical Expression)

- **규모:** 국제 학술회의, 산학협력 중심
- **2025 개최:** June 24-27, Entangled NIME (장소 TBD)
- **2026 개최:** June 23-26, Loughborough University, London
- **특징:**
  - 여행 불필요 (원격 발표 가능)
  - 음악/성능 기술 + 학술 논문 병렬
  - 포함: 종이, 음악, 공연 세션
- **관련 분야:** HCI, 악기 설계, 실시간 음향, 라이브 coding

**AudioArt와의 연관성:** Meta-Symphony를 "New Interface" 논문으로 제출할 수 있는 플랫폼. 신경망 악기화의 학술적 벤치마크.

**레퍼런스:** [NIME 2025](https://nime2025.org/), [NIME Archive](https://nime.org/)

### 5.2 ISMIR (International Society for Music Information Retrieval)

- **연간 컨퍼런스:** 음악 정보 검색, MIR
- **2024 테마:** "Bridging Technology and Musical Creativity"
- **123 논문** 발표 (AI for music generation 포함)
- **특별 컬렉션:** "AI and Musical Creativity" (TISMIR journal)
- **편집진:** 기술 + 윤리 관점 통합

**AudioArt와의 연관성:** RAVE, EnCodec, AudioLDM 논문들의 출판 및 발표 플랫폼. AudioArt V1 결과를 "Neural Sound Art" 트랙으로 발표 가능.

**레퍼런스:** [ISMIR 공식](https://ismir.net/), [TISMIR AI and Creativity 특집](https://transactions.ismir.net/collections/ai-and-musical-creativity)

### 5.3 Audio Mostly (2024: "Explorations in Sonic Cultures")

- **규모:** 연간 소규모 학제간 컨퍼런스
- **2024 개최:** Sept 18-20, University of Milan, Italy
- **특징:**
  - 음향 설계 + 경험 중심
  - 응용 이론 + 반영적 실천
  - 제품 설계, 오디오 디스플레이, 게임, 가상 환경, 음악 악기, 교육 도구
- **테마:** 음향 문화, 정체성, 기술 영향, 음향 유산 보존

**AudioArt와의 연관성:** 신경망 사운드 아트를 "sonic culture" 맥락에서 제시. Meta-Symphony의 청각 미학 발표.

**레퍼런스:** [Audio Mostly 2024](https://audiomostly.com/2024/)

### 5.4 DAFx (International Conference on Digital Audio Effects)

- **제27회:** Sept 3-7, 2024, University of Surrey, Guildford, UK
- **연간 개최:** 초 9월
- **특징:**
  - 음성 + 음악 음향 처리
  - 음향 설계, 음향 예술, 음향학
  - 동료 검토 + 튜토리얼 + 사회 프로그램
- **역사:** 수십 년 전통 (1998~)

**AudioArt와의 연관성:** Token databending, spectral freeze, bass smear 같은 신경망 기반 음향 효과의 기술적 발표 플랫폼.

**레퍼런스:** [DAFx24](https://dafx24.surrey.ac.uk/), [DAFx 아카이브](https://www.dafx.de/)

### 5.5 ICAD (International Conference on Auditory Display)

- **제29회:** June 24-28, 2024, EMPAC Troy, NY
- **테마:** "Sonification // Spatialization"
- **특징:**
  - 데이터 음화(sonification)
  - 공간 음향 + 오디오 시각화
  - 음향학, 음악, 음향 처리, HCI
- **역사:** 공청각 디스플레이 전문가 커뮤니티

**AudioArt와의 연관성:** Token-to-spectrogram 음화, Anchored Corruption의 "Sub-bass Anchor"를 오디오 디스플레이로 설명할 수 있는 플랫폼.

**레퍼런스:** [ICAD 2024](https://icad2024.icad.org/)

### 5.6 SMC (Sound and Music Computing Conference)

- **2024 개최:** July 4-6, Porto, Portugal
- **테마:** "Immersive Sound, Immersive Music, Immersive Computing"
- **특징:**
  - 음향 이론 + 컴퓨터 음악
  - 창작적 컴퓨팅
  - 음향 특화 신경망 논문
- **형식:** 4~8 페이지 논문 + 2분 비디오 프레젠테이션

**AudioArt와의 연관성:** Net Max의 다축 손상 구조, Meta-Symphony의 매크로 위상을 "immersive" 음향 환경으로 구현.

**레퍼런스:** [SMC 2024](https://smcnetwork.org/smc2024/)

---

## 6. AudioArt V2 특정 차별화: 선행 사례에 대한 체계적 응답

### 표: V2 역량 vs. 선행 사례 벤치마크

| 선행 사례 | 주요 혁신 | AudioArt V2 추가 응답 | 신규 모듈/이론 |
|---|---|---|---|
| **IRCAM AFTER** | 실시간 구조/음색 분리 | Mimi (의미론적 토큰) + Semantic Governor → 음색 영역 벗어난 떠도는 감지 | MVP-J, `net_semantic` |
| **Holly Herndon The Call** | 15 합창단 다중 입력 | AudioLLM caption이 다른 매크로넷 파라미터 조건화 → 다축 실시간 변형 | `net_llm_chain`, `net_prompt_morph` |
| **Björk Nature Manifesto** | 저전력 RAVE CPU 추론 | 1시간 대작 + SoX 디스크 스트리밍 + Texture/Semantic dual governor | Anchored Corruption, Texture Governor v2 |
| **Evala Studies for** | 200시간 아카이브 → SpecMaskGIT | Macro-network architecture: Net 1-4의 자기강화 cross-feedback 버스 (α~θ) | Net Max topology |
| **Dadabots 24시간** | 무한 생성 스트림 | Semantic Governor가 의미 표류 감지 + 자동 seed 재주입 → bounded creativity | `stem_LLM` |
| **Yuri Suzuki Voice-to-param** | 5 호른 음성 → 음악 | 캡션 텍스트 자체가 다른 net의 LFO rate, filter freq, gate threshold 결정 | `net_prompt_morph` |

---

## 7. 논문/리소스 통합 인덱스 (Unified Index for V1 → V2)

### 신경망 코덱 근간
| 항목 | 논문/레포 | AudioArt 모듈 |
|---|---|---|
| RAVE v1-v2 | arXiv:2111.05011 | MVP-A, E, F, G (latent perturbation) |
| EnCodec | arXiv:2210.13438 | MVP-C, I (token databending) |
| DAC | GitHub:descriptinc | MVP-C, I (alternative codec) |
| Mimi | HF:kyutai/mimi | **MVP-J** (V2: 의미론적 토큰 분리) |
| BigVGAN | arXiv:2206.04658 | Texture Governor (flatness/RMS) |
| Snake activation | arXiv:2307.05830 | RAVE v2 기반 activation |

### 생성 모델
| 항목 | 논문/레포 | AudioArt 모듈 |
|---|---|---|
| AudioLDM 2 | arXiv:2308.05734 | MVP-B stub → 실제 백본 (V2) |
| Qwen2-Audio | arXiv:2407.10759 | MVP-B caption encoder (V2) |
| GAMA | EMNLP 2024 | Semantic Governor (복합 추론) |
| Audio Flamingo 2 | arXiv:2503.03983 | AudioLLM 기반 후보 |
| SALMONN | arXiv:2310.13289 | Caption loop 기반 (V2) |
| LTU-AS | arXiv:2305.10790 | Audio + speech joint understanding |
| MusicGen / MAGNeT | AudioCraft GitHub | Meta-Symphony stem 생성 (V2) |
| VampNet | arXiv:2307.04686 | 마스크 기반 재생성 |

### 이론
| 항목 | 논문/리소스 | AudioArt 모듈 |
|---|---|---|
| Git Re-Basin | arXiv:2209.04836 | MVP-D (checkpoint morphing) |
| Model Collapse (Curse of Recursion) | arXiv:2305.17493 | MVP-B 이론 (V2 실증) |
| Self-Consuming MAD | arXiv:2307.01850 | Audio autophagy loop 설계 |
| Linear Mode Connectivity | ICML 2020 | Re-Basin 선행 이론 |
| Model Soups | ICML 2022 | Meta-Symphony weight averaging |

### 예술 선행자
| 항목 | URL/레퍼런스 | AudioArt 영감 |
|---|---|---|
| Yasunao Tone (Wounded CD) | 1984~1997 | 글리치 미학의 원류 |
| Oval/Markus Popp | Systemisch 1994 | CD 물리적 손상 → 신경망 토큰 손상 |
| Ryoji Ikeda | spectra II, Test Pattern | 데이터 소니피케이션 |
| Mira Calix | Inside There Falls, Nothing Is Set in Stone | 반응형 공간 음향 |
| Dadabots | SampleRNN 24-시간 | 장형 생성 스트림 |
| Holly Herndon | PROTO (2019), The Call (2024) | 신경음성 + 조화 데이터 신뢰 |
| Yuri Suzuki | Vox PopulA.I, Welcome Chorus | 음성 조건부 → 매크로 파라미터 생성 |

---

## 8. 검증 불가능한 항목 및 권고사항

### 8.1 미확인 항목 (미확인)

1. **Christian Marclay "Record Without a Cover"** — 존재 검증 실패. (혼동 가능: "Record without a Cover" vs. 다른 Marclay 작품)
2. **Rosa Menkman GLI.TC/H 상세 연도** — 작업 존재 확인, 정확한 연도 불명확
3. **Casey Reas databending 구체적 작품** — 일반적 참조만 확인
4. **nn_tilde Max/MSP RAVE 플러그인** — 언급되나 공식 현황 불명확 (nn~ 자체는 확인됨)

### 권고: 이 항목들은 **"(미확인)"** 주석으로 처리하거나 삭제 권장.

---

## 9. V2 통합 권고사항

### 9.1 RESEARCH_SURVEY.md에 추가할 항목

**섹션 2 (핵심 모델)에 추가:**
- Kyutai Mimi codec (의미론적 토큰의 V2 기초)
- NVIDIA GAMA (복합 추론 기반 Semantic Governor)
- ByteDance SALMONN (진정한 MVP-B 백본)
- MIT LTU-AS (multi-modal 이해)

**섹션 5 (도구 및 프레임워크)에 추가:**
- BigVGAN (Neural vocoder, Texture Governor 기반)
- Suno / Udio (상용 곡 레벨 생성, V2 모니터링)
- Model Soups (Meta-Symphony weight averaging)

**새 섹션 추가: "V2 예정 통합 모델"**
- Mimi codec + MVP-J
- AudioLLM 기반 조건부 매크로넷

### 9.2 DETAILED_CASE_STUDIES.md에 추가할 심층 분석

**새 섹션: "자동포식(Autophagy) 루프의 정당성: Shumailov & Alemohammad"**
- Curse of Recursion의 3가지 루프 패턴 분석
- MVP-B의 "신선한 캡션" loop 설계

**새 섹션: "모델 병합 이론: Git Re-Basin + Model Soups"**
- MVP-D와 Meta-Symphony의 학술 토대

**확장: "신경음성 → 신경코덱 계보"**
- Holly Herndon PROTO (2019)
- The Call (2024) + beyond

---

## 10. 최종 체크리스트

### 신규 파운데이션 모델 (7개 추가)
- [x] Kyutai Mimi
- [x] Stability Audio Open
- [x] Meta AudioCraft (MusicGen, AudioGen, MAGNeT)
- [x] Adobe VampNet
- [x] NVIDIA Audio Flamingo 2
- [x] NVIDIA GAMA
- [x] ByteDance SALMONN
- [x] MIT LTU-AS
- [x] Suno / Udio

### 학술 이론 (7개 추가)
- [x] Git Re-Basin (Ainsworth 2023, ICLR)
- [x] Linear Mode Connectivity (Frankle 2020, ICML)
- [x] Model Soups (Wortsman 2022, ICML)
- [x] Curse of Recursion (Shumailov 2023)
- [x] Self-Consuming MAD (Alemohammad 2024, ICLR)
- [x] Snake activation function
- [x] BigVGAN (NVIDIA 2023, ICLR)

### 예술가 & 선행자 (6개 추가/확장)
- [x] Ryoji Ikeda
- [x] Mira Calix (상세 작품 추가)
- [x] Dadabots 24-시간 (재확인)
- [x] Yuri Suzuki (4개 프로젝트 상세)
- [x] Holly Herndon PROTO / Spawn (상세화)
- [x] Caterina Barbieri (modular + AI)

### 학술 컨퍼런스 (6개 추가)
- [x] NIME 2025-2026
- [x] ISMIR 2024 (creativity track)
- [x] Audio Mostly 2024
- [x] DAFx 2024
- [x] ICAD 2024
- [x] SMC 2024

### 검증 완료
- [x] Holly Herndon The Call (확인: 검증됨, 날짜 및 기술 상세)
- [x] Björk Nature Manifesto (확인: "Robin Meier Wiratunga" 정정)
- [x] Evala Studies for (확인: arXiv 논문 발견)
- [x] IRCAM AFTER (확인: 2024년 논문 발견)

---

## 참고문헌 (Complete Reference List)

### 신경망 오디오 코덱
1. [RAVE: arXiv:2111.05011](https://arxiv.org/pdf/2111.05011)
2. [EnCodec: arXiv:2210.13438](https://arxiv.org/pdf/2210.13438)
3. [Mimi (Kyutai): HuggingFace](https://huggingface.co/kyutai/mimi)
4. [BigVGAN: arXiv:2206.04658](https://arxiv.org/abs/2206.04658)

### 생성 모델
5. [AudioLDM 2: arXiv:2308.05734](https://arxiv.org/pdf/2308.05734)
6. [Qwen2-Audio: arXiv:2407.10759](https://arxiv.org/pdf/2407.10759)
7. [SALMONN: arXiv:2310.13289](https://arxiv.org/abs/2310.13289)
8. [LTU-AS: arXiv:2305.10790](https://arxiv.org/abs/2305.10790)
9. [GAMA: ACL Anthology](https://aclanthology.org/2024.emnlp-main.361/)
10. [Audio Flamingo 2: arXiv:2503.03983](https://arxiv.org/pdf/2503.03983)
11. [VampNet: arXiv:2307.04686](https://arxiv.org/abs/2307.04686)

### 학술 이론
12. [Git Re-Basin: arXiv:2209.04836](https://arxiv.org/abs/2209.04836)
13. [Curse of Recursion: arXiv:2305.17493](https://arxiv.org/abs/2305.17493)
14. [Self-Consuming MAD: arXiv:2307.01850](https://arxiv.org/abs/2307.01850)
15. [Linear Mode Connectivity: ICML 2020](http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf)
16. [Model Soups: arXiv:2203.05482](https://arxiv.org/abs/2203.05482)

### 예술 & 사례 연구
17. [Serpentine Galleries: The Call](https://www.serpentinegalleries.org/whats-on/holly-herndon-mat-dryhurst-the-call/)
18. [Centre Pompidou: Nature Manifesto](https://www.centrepompidou.fr/en/bjoerk-aleph)
19. [Evala Studies for: arXiv:2510.25228](https://arxiv.org/html/2510.25228)
20. [IRCAM AFTER GitHub](https://github.com/acids-ircam/AFTER)
21. [Ryoji Ikeda - UC Press](https://online.ucpress.edu/afterimage/article/51/4/65/204742/)
22. [Mira Calix - VICE](https://www.vice.com/en/article/artist-mira-calix-creates-a-labyrinth-of-sound-and-storytelling-out-of-paper-2/)
23. [Dadabots Official](https://dadabots.com/science/)
24. [Yuri Suzuki Official](https://www.yurisuzuki.com/projects/vox-popula-i)
25. [Holly Herndon PROTO Interview](https://musictech.com/features/interviews/holly-herndon-proto/)

### 컨퍼런스 및 커뮤니티
26. [NIME 2025](https://nime2025.org/)
27. [ISMIR Official](https://ismir.net/)
28. [Audio Mostly 2024](https://audiomostly.com/2024/)
29. [DAFx24](https://dafx24.surrey.ac.uk/)
30. [ICAD 2024](https://icad2024.icad.org/)
31. [SMC 2024](https://smcnetwork.org/smc2024/)

---

**작성 완료:** 2026년 5월 27일  
**다음 스텝:**
1. 이 문서의 내용을 RESEARCH_SURVEY.md, DETAILED_CASE_STUDIES.md에 통합
2. V2 개발 로드맵에 Mimi (MVP-J), SALMONN (MVP-B 실제), Semantic Governor 우선순위 반영
3. 각 신규 모델의 라이선스 + 컴퓨팅 비용 평가
