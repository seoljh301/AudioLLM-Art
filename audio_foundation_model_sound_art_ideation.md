# Audio Foundation Model Misuse for Sound Art

## Ideation Document for Agent-Based Prototyping

Author: User  
Status: Exploratory Research / Sound Art Prototype  
Domain: Audio Foundation Models, Experimental Electronic Music, Neural Audio Systems, Sound Art

---

# 1. Core Motivation

This project does NOT aim to build a normal AI music generator.

The goal is:

> To explore aesthetically meaningful sonic artifacts emerging from the failure, instability, misunderstanding, corruption, or recursive reinterpretation of Audio Foundation Models.

The system should treat AudioLLMs not as composers, but as:
- unstable listening machines
- hallucinating acoustic interpreters
- neural texture synthesizers
- semantic distortion devices
- recursive sonic organisms

---

# 2. High-Level Directions

1. Semantic Collapse
2. Recursive Audio Feedback
3. Audio ↔ Text ↔ Audio Misinterpretation
4. Checkpoint-as-Synthesizer
5. Codebook-as-Timbral-Control
6. Neural Databending

---

# 3. Candidate Technical Stack

## Realtime Neural Audio
- RAVE
- EnCodec
- DAC
- Mimi codec

## Streaming Audio Models
- Moshi
- Qwen-Audio
- Audio captioning models

---

# 4. System Architecture

```text
Max/MSP
↓
OSC / UDP / WebSocket
↓
Python Backend
↓
Audio Model Processing
↓
Audio Return to Max/MSP
↓
Live Processing / Spatialization / Performance
```

---

# 5. Experimental Modules

## Semantic Collapse Engine

Methods:
- semantic token dropout
- semantic token shuffle
- high-temperature generation
- partial token corruption

Desired outputs:
- pseudo-language
- ghost speech
- impossible environmental sounds

---

## Recursive Audio Feedback

```text
audio_t
→ model
→ output_t
→ partial reinjection
→ output_t+1
```

Controls:
- feedback amount
- entropy injection
- random perturbation
- memory decay

---

## Audio ↔ Text ↔ Audio Loop

```text
modular synth audio
→ audio captioning model
→ generated text
→ text-to-audio model
→ generated sound
→ repeat
```

---

## Checkpoint-as-Synthesizer

Possible techniques:
- checkpoint interpolation
- training-stage morphing
- corrupted checkpoint loading
- checkpoint routing

Concept:
> A checkpoint is a frozen auditory interpretation of the world.

---

## Codebook Organ

Operations:
- upper-RVQ corruption
- lower-RVQ preservation
- codebook swap
- token perturbation
- entropy-limited random walk

---

## Neural Databending

Techniques:
- invalid token injection
- bitrate abuse
- quantizer omission
- temporal token scrambling

---

# 6. Realtime Constraints

Recommended approach:
- realtime codec manipulation
- lightweight latent operations
- asynchronous heavy generation
- chunk-based streaming
- look-ahead buffering

---

# 7. Suggested MVP

## MVP-A
RAVE + Max/MSP latent perturbation system.

## MVP-B
Audio captioning → text-to-audio recursive loop.

## MVP-C
EnCodec token corruption playground.

## MVP-D
Checkpoint morphing texture synthesizer.

---

# 8. Final Guiding Principle

> not how AI successfully understands sound,
> but how sound transforms when AI fails to understand it.

---

# 9. Post-Prototyping Reflection (2026)

이 프로젝트는 초기 아이디어인 "신경망 오용(Misuse)"을 넘어, 9개의 MVP 모듈과 거대 매크로 네트워크(Multinet), 그리고 최종적인 Meta-Symphony로 완성되었다. 프로토타이핑 과정에서 얻은 주요 통찰은 다음과 같다.

1.  **Anchored Corruption의 유효성**: 단순히 모델을 폭주시키는 것보다, 80Hz 크로스오버와 같은 물리적 '닻(Anchor)'을 내렸을 때 훨씬 더 음악적이고 압도적인 예술적 결과물이 나왔다.
2.  **모드 연결성(Mode Connectivity)의 한계와 활용**: MVP-D(Morphing)에서 발견된 '중간 지점의 무음 붕괴'는 기술적으로는 실패였으나, 예술적으로는 '유령 같은 잔향'이라는 새로운 프리셋으로 재탄생했다.
3.  **스케일의 확장**: 실시간 처리를 넘어 1시간 분량의 대작(`Ulaanbaatar Epic`)을 렌더링하면서, 파이썬의 한계를 SoX 커맨드라인 스트리밍으로 극복한 것은 기술적 성숙도를 증명했다.

초기의 "실패의 미학"은 이제 "정교하게 설계된 신경망 교향곡"이라는 구체적인 형상을 갖추게 되었다.
