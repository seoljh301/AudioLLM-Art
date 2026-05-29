# AudioLLM Micro-Sequencer (Neural Tracker)

이 문서는 AudioArt V2의 가장 진보된 편곡 시스템인 **"Micro-Semantic Sequencer (신경망 트래커)"**의 아키텍처를 정의합니다.

기계가 음악 전체를 통째로 생성하는 흔한 AI 작곡 방식을 배제하고, 언어 모델(LLM)이 오직 **"시간 축의 미세 분할과 샘플의 배치"**만을 담당하게 하여 전자음악 특유의 IDM(Intelligent Dance Music) 프로그래밍 방법론을 기계 지능으로 구현합니다.

---

## 1. 아키텍처 철학: 매크로 생성에서 마이크로 시퀀싱으로

*   **배경**: 전체 드럼 루프를 생성 모델(AudioLDM)에 맡기면 세밀한 리듬의 통제가 불가능하고 '전형적인 AI 음악'이 됩니다.
*   **해법**: 오디오 모델은 극도로 짧은 단일 타격음(One-shot Sample)만을 생성합니다. 그리고 LLM은 1/64, 1/128 박자 단위의 그리드(Grid)를 작성하여 이 파편들을 조립(Tracking)합니다.

---

## 2. 작동 프로세스 (The Neural Tracker Pipeline)

### Phase 1: Dynamic Palette Definition (음향 파편 창조)
단순히 미리 정해진 샘플(Kick, Snare)을 쓰는 것을 넘어, LLM이 **자신이 쓸 악기를 스스로 정의(Prompting)**합니다. 텍스트 묘사를 바탕으로 AudioLDM이나 DSP 엔진이 해당 소리를 즉석에서 렌더링하여 팔레트에 등록합니다.

**Palette JSON from LLM:**
```json
[
  {"symbol": "K", "description": "heavy distorted sub kick", "duration": 0.4},
  {"symbol": "G", "description": "glass shatter bright", "duration": 0.2},
  {"symbol": "Z", "description": "laser zap synth drop", "duration": 0.1}
]
```

### Phase 2: LLM Track Programming (시퀀스 매트릭스 생성)
텍스트 LLM에게 원하는 리듬의 미학을 설명하고, 미세 시간 축(예: 1마디를 32단계로 분할)에 대한 JSON 배열을 반환하도록 프롬프팅합니다. 자신이 1단계에서 발명한 기호(Symbol)를 사용하여 리듬을 조립합니다.

**Sequence JSON from LLM:**
```json
[
  {"step": 0, "sample": "K", "ratchet": 1},
  {"step": 4, "sample": "Z", "ratchet": 4},
  {"step": 8, "sample": "G", "ratchet": 1}, 
  {"step": 12, "sample": "Z", "ratchet": 2},
  ...
]
```

### Phase 3: AudioLLM Micro-Judge (미세 리듬 비평)
파이썬 엔진이 이 JSON을 읽어 정확한 샘플 단위 오디오로 렌더링합니다.
Qwen2-Audio(심사위원)가 이 패턴을 듣고 **"리듬의 긴장감, 클라베(Clave)의 엇박자 정도, 래칫의 과도함"** 등을 평가합니다.

**AudioLLM Critique:**
> "두 번째 박자의 1/32 글리치 연타가 너무 뻔하게 들린다. 스텝 12부터 15 사이의 공간을 비우고, 스텝 16에 극단적인 1/128 박자 래칫을 집중시켜라."

이 피드백은 다시 텍스트 LLM으로 들어가 다음 세대의 JSON 매트릭스로 진화합니다.

---

## 3. 예술적 의의
이 시스템은 소리를 '음원'이 아닌 '데이터 구조'로 다루는 초창기 트래커(Tracker, 예: Renoise, FastTracker) 문화의 부활입니다. 인간이 마우스로 일일이 찍기 불가능한 1/128 박자 단위의 미세한 엇박자와 버그 같은 스터터링(Stuttering)을 기계의 논리가 스스로 계산하여 박아 넣는, 진정한 의미의 **알고리즘 IDM**이 탄생합니다.
