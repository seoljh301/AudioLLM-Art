# LLM-as-a-Judge: The Autonomous Aesthetic Curator

이 문서는 AudioArt V2의 핵심 패러다임 전환인 **"LLM-as-a-Judge (심사위원으로서의 언어 모델)"** 아키텍처를 정의합니다.

기존의 AudioArt 시스템이 물리적 알고리즘과 파라미터에 의해 일방향적(One-way)으로 소리를 파괴했다면, V2 시스템은 AudioLLM(예: Qwen2-Audio)을 도입하여 시스템 스스로가 **생성(Generate) ➔ 청취(Listen) ➔ 비평(Critique) ➔ 수정(Refine)**의 루프를 도는 자아 성찰적 아티스트로 진화합니다.

---

## 1. Core Concept (핵심 개념)

오디오-언어 모델은 단순히 소리에 자막을 다는 것을 넘어, 소리가 특정한 "미학적 목표(Aesthetic Target)"에 도달했는지 평가할 수 있습니다. 우리는 이 능력을 활용하여 리듬, 퍼커션, 질감을 끝없이 조각(Sculpting)합니다.

*   **The Brief (작업 지시서)**: 사용자가 원하는 소리의 추상적 묘사. (예: "차가운 금속이 부딪히며 부서지는 듯한 불규칙한 인더스트리얼 비트")
*   **The Judge (오디오 심사위원)**: 현재 렌더링된 소리를 듣고, "The Brief"와 얼마나 일치하는지 평가(1~10점)하며 구체적인 실패 요인을 지적합니다.
*   **The Helmsman (조타수)**: 심사위원의 비평 리포트를 바탕으로 다음 렌더링에 사용할 파라미터(디스토션, 글리치 밀도, LFO 속도 등)를 수정합니다.

---

## 2. The Auto-Correction Loop (자동 수정 루프)

다음은 시스템 내부에서 일어나는 가상의 퍼커션 조각 세션(Percussion Sculpting Session) 로그입니다. 기계는 인간의 개입 없이 목표 점수(Pass Score)에 도달할 때까지 파라미터를 튜닝합니다.

### 📝 Session #042: Percussion Sculpting
**Target Aesthetic**: "차가운 금속이 부딪히며 부서지는 듯한 불규칙한 인더스트리얼 비트"

| Iteration | AudioLLM's Hearing (모델이 들은 소리) | Critique (심사평) | Action (파라미터 수정 지시) |
| :--- | :--- | :--- | :--- |
| **Draft v1.0** | "A steady and heavy electronic drum beat with some background noise." | **[Fail: 3/10]** 리듬이 너무 정직하고 예측 가능함. 금속성 질감이 부족하며 단순한 잡음으로 들림. | `glitch_density` 0.2 ➔ 0.7 상승. `filter_cutoff_bias` +0.5로 하이역대 개방. |
| **Draft v1.1** | "Fast, chaotic clicks and sharp noises in a rapid tempo." | **[Fail: 6/10]** 불규칙성은 좋아졌으나 묵직함이 사라져 가벼운 클릭음만 남음. 타격감 복원 요망. | `kick_decay` 30.0 ➔ 10.0으로 킥 꼬리 연장. `distortion_intensity` 0.4 추가. |
| **Draft v1.2** | "A heavy, distorted metallic pounding with an unpredictable stuttering rhythm." | **[Pass: 9/10]** 타겟 미학에 완벽히 부합함. 차갑고 파괴적인 인더스트리얼 질감 및 불규칙한 리듬 달성. | **최종 승인 및 렌더링 완료** |

---

## 3. Artistic Implications (예술적 의의)

1.  **Chain of Thought in Art**: 기계가 왜 특정한 소리를 최종 결과물로 선택했는지에 대한 '사고 과정'이 텍스트로 보존됩니다. 이는 결과물만큼이나 과정 자체를 예술로 만드는 개념적 접근입니다.
2.  **Infinite Curation**: 예술가는 피곤해하지만 기계는 지치지 않습니다. 밤새도록 수천 번의 루프를 돌며 가장 완벽하게 기괴한 소리 하나를 깎아내게 할 수 있습니다.
3.  **Hallucinatory Mutations (환각적 돌연변이)**: AudioLLM이 소리를 완전히 엉뚱하게 오해(Hallucination)하여 잘못된 심사평을 내릴 때, 믹싱 엔진이 그 오해를 수용하려고 파라미터를 비틀면서 인간이 상상할 수 없는 "기괴한 돌연변이 사운드"가 탄생합니다. 이 오류의 궤적은 AudioArt 철학의 핵심입니다.
