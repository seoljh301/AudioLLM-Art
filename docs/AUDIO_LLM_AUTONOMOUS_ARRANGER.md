# AudioLLM Autonomous Arranger: The Model-to-Model Pipeline

이 문서는 AudioArt V2의 최종 아키텍처인 **"자율 편곡자 (Autonomous Arranger)"**의 작동 방식을 설명합니다. 

이 시스템의 가장 큰 특징은 인간의 프로그래밍이나 파라미터 조작에 의존하지 않고, **두 개의 서로 다른 Audio Foundation Model 간의 상호작용(Model-to-Model Interaction)**만으로 음악을 생성하고 편곡한다는 점입니다.

---

## 1. 아키텍처 개요 (The Pipeline)

### 🏭 Stage 1: The Generator (AudioLDM)
*   **역할**: 무한한 소리의 원자재를 생산하는 '공장'입니다. 
*   **작동 방식**: 수학적인 사인파 연산을 버리고, 거대 확산 모델(AudioLDM)을 사용해 수십~수백 개의 텍스트 프롬프트를 오디오로 변환합니다.
    *   *Prompt 1*: "A very clean, deep 808 kick drum loop at 120 bpm"
    *   *Prompt 2*: "Industrial harsh noise percussion hitting like metal pipes"
    *   *Prompt 3*: "Screaming digital glitch stutter"
*   **결과**: 수십 개의 무작위 드럼/퍼커션/질감 오디오 파일(Pool)이 생성됩니다.

### ⚖️ Stage 2: The Curator (Qwen2-Audio / AudioLLM)
*   **역할**: 생성된 소리들을 듣고 평가하는 '심사위원'이자 'DJ'입니다.
*   **작동 방식**: Generator가 만든 수십 개의 오디오 파일을 AudioLLM이 직접 **"청취(Listen)"**합니다. 그리고 각 파일의 특성을 분석하여 **메타데이터(에너지 레벨, 질감, 적합한 곡의 위치)**를 태깅합니다.
    *   *오디오 파일 #04* ➔ AudioLLM 평가: `{"Energy": 2, "Texture": "Soft, Ambient", "Role": "Intro"}`
    *   *오디오 파일 #17* ➔ AudioLLM 평가: `{"Energy": 9, "Texture": "Harsh, Chaotic", "Role": "Climax"}`

### 🎼 Stage 3: The Weaver (Timeline Orchestrator)
*   **역할**: 태깅된 소리들을 시간 순서대로 엮어 하나의 완결된 곡(Symphony)으로 만듭니다.
*   **작동 방식**: "Intro ➔ Build-up ➔ Drop ➔ Climax ➔ Outro"라는 거시적 서사 템플릿(Macro-template)에 맞춰, AudioLLM이 가장 높은 점수를 주었거나 해당 에너지가 필요한 오디오 블록을 레고처럼 끼워 맞춥니다.

---

## 2. Model-to-Model (M2M) 패러다임의 의의

*   **진정한 "기계의 미학"**: 파이썬 코드가 개입하는 여지(예: LFO 속도, 필터 값 조절)를 없앴습니다. **"A모델이 소리를 상상해서 만들고, B모델이 그 소리를 평가해서 전시한다"**는 점에서 진정한 의미의 자율적(Autonomous) 기계 예술이 성립됩니다.
*   **무한한 확장성**: 텍스트 프롬프트만 LLM이 무작위로 생성하게 둔다면, 이 파이프라인은 인간의 개입 없이 영원히 새로운 장르와 텍스쳐의 음악을 만들어낼 수 있는 24/7 스트리밍 라디오가 될 수 있습니다.
*   **환각의 증폭**: AudioLDM이 프롬프트를 잘못 이해하여 기괴한 소리를 만들고, AudioLLM이 그 기괴한 소리를 또 오해하여 클라이맥스에 배치해 버리는 **'오류의 연쇄 작용'**이야말로 이 프로젝트가 추구하는 Glitch Art의 정점입니다.
