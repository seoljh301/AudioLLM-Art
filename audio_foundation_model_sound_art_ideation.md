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

이 프로젝트는 초기 아이디어인 "신경망 오용(Misuse)"을 넘어, **9 개의 MVP 모듈** + **5 개의 매크로 네트워크 (Multinet)** + 최상위 **Meta-Symphony** + 정적 **데모 페이지**까지 완성되었다.

## 9.1 실제 빌드된 구성요소

| 단계 | 구현물 | 위치 |
|---|---|---|
| MVP 9 종 (A–I) | latent perturb · caption loop · token bend · ckpt morph · neural granular · spectral freeze · latent feedback · codebook organ · bass massive | `src/modules/mvp_*` |
| 공통 인프라 | Texture Governor (NaN 가드 + flatness / RMS / centroid 임계치) + Mix Engine (dry/wet + 80 Hz crossover + soft limiter) + OSC bridge | `src/core/{mix,texture_governor,texture_metrics,osc_bridge}.py` |
| 매크로 네트워크 5 종 | Net 1 / 2 / 3 / Net Max (9 MVP 모두) / Net Dynamic (22 dB 다이내믹) | `scripts/multinet.py` |
| 메타 작곡 | Meta-Symphony — 4 매크로넷 stem 을 LFO crossfade + 스테레오 드리프트로 3 분 stereo 곡 | `scripts/meta_symphony.py` |
| 검증 데모 | 52 트랙 정적 HTML + waveform PNG 썸네일 | `demo.html` + `scripts/build_demo.py` |
| 대용량 마스터링 | LUFS −12 + SoX 디스크 스트리밍 | `scripts/pyloudnorm_mastering.py`, `scripts/run_hfo_master_sox.sh` |

## 9.2 핵심 통찰

1. **Anchored Corruption 의 유효성**: 단순 폭주보다 80 Hz 크로스오버 같은 물리적 닻이 훨씬 더 음악적 결과를 만든다. 모든 매크로넷이 이 한 줄에 의존.
2. **모드 연결성의 한계와 재활용**: MVP-D 의 *중간 무음 붕괴* 는 기술적 실패였지만 "유령 잔향" 프리셋으로 재탄생. Re-Basin (partial + full encoder) 까지 시도했으나 결국 cliff sweep 으로 `t ∈ [0, 0.02] ∪ [0.98, 1]` 운영이 최선.
3. **스케일 확장**: 1 시간 대작 `Ulaanbaatar Epic` 까지. Python 메모리 한계는 SoX 디스크 스트리밍으로 우회.
4. **Texture Governor 의 결정성**: 9 MVP × 5 매크로넷 × 메타-심포니 전체 파이프라인에서 NaN 도달이 단 한 번도 발생하지 않음. 2:08 끊김 사고 후 도입된 NaN emergency 가 결정적.
5. **다이내믹 vs 정적의 차이**: Net Max 4 dB vs Net Dynamic 22 dB — 같은 8 버스라도 시간 가변 envelope + filter sweep + impulse 이벤트가 결과를 *작곡*으로 격상.

## 9.3 §7 의 원래 MVP 제안 vs 실제 구현

| 원래 제안 | 실제 빌드 |
|---|---|
| MVP-A: RAVE + Max latent perturbation | MVP-A + smoothed Brownian noise mode ✓ |
| MVP-B: Caption ↔ TTA recursive loop | MVP-B (stub backend, 실제 백본은 V2 로) ⚠️ |
| MVP-C: EnCodec token corruption | MVP-C 4 모드 + grouped quantizer_range ✓ |
| MVP-D: Checkpoint morphing | MVP-D + partial/full Re-Basin + endpoint cliff sweep + A+D bilateral dropout ✓ |
| (계획에 없던 추가 MVP) | MVP-E/F/G/H/I — feedback_1 design 원리를 따라 자연 발생 |

§7 의 4 개 MVP 가설 모두 검증되었고, 검증 과정에서 새 MVP 5 개가 자연 발생.

## 9.4 향후 작업 — AudioLLM 단계

프로젝트 명칭 **"AudioLLM-Art"** 가 가리키듯, 현재까지는 *뉴럴 사운드 아트* 영역의 V1 — 다음 단계 V2:

- **MVP-B 실제 백본 활성화** (Qwen2-Audio + AudioLDM2, ~18 GB) — stub → 실제 의미적 표류.
- **AudioLLM 조건부 매크로넷** (`net_semantic`, `net_llm_chain`, `net_prompt_morph`) — 캡션이 다른 매크로넷의 파라미터를 시간 가변으로 결정.
- **Semantic Governor** — Texture Governor 의 의미 측 짝. 캡션 표류 과대 시 자동 감쇠.
- **Meta-Symphony v2 — 5 stem** — 현재 4 stem 에 `stem_LLM` 추가.
- **AudioLLM 자체 손상** — 캡션 모델 임베딩 공간에 직접 노이즈 — *AudioLLM 도메인의 perturbation*. feedback_1 의 §13–§16 원칙을 의미 도메인에 다시 적용.

V1 결과는 *Pre-AudioLLM* 라벨로 보존된다.

---

초기의 "실패의 미학" 은 이제 *정교하게 설계된 신경망 교향곡* 이라는 구체적 형상을 갖추었고, "AudioLLM-Art" 라는 이름이 실제로 가리킬 다음 단계가 명확해졌다.
