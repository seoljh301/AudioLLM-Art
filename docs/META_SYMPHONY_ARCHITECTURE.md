# AudioArt: The Meta-Symphony Architecture

이 문서는 **AudioArt** 프로젝트의 궁극적인 결과물인 `Meta-Symphony`가 어떻게 구성되는지, 그리고 그 근간이 되는 기초 신경망 모듈(MVP A~I)들이 어떤 원리로 작동하는지를 처음 보는 사람도 이해할 수 있도록 총망라하여 설명합니다.

AudioArt의 핵심 철학은 **"Anchored Corruption(닻을 내린 붕괴)"**입니다. 신경망 오디오 모델(RAVE, EnCodec 등)을 극단적으로 오용(Misuse)하되, 저음역대나 볼륨의 뼈대를 안전장치(Texture Guard)로 단단히 고정하여, 노이즈가 아닌 **"음악적으로 아름다운 신경망의 에러"**를 창조합니다.

---

## 1. 기초 단위: 9개의 신경망 변조 모듈 (The 9 MVPs)
이 프로젝트는 사운드를 파괴하고 재구성하는 9개의 독립적인 모듈(MVP)로 구성됩니다.

| 모듈 | 이름 | 사용 모델 | 작동 원리 및 사운드 특성 |
| :--- | :--- | :--- | :--- |
| **MVP-A** | Latent Perturbation | RAVE | 잠재 공간(Latent Space)에 브라운 운동 방식의 부드러운 노이즈를 주입합니다. 사운드의 질감이 유기적인 액체처럼 꿈틀거리며 변조됩니다. |
| **MVP-B** | Caption Loop | AudioLDM2 (TTA) | 소리를 텍스트로 설명(Caption)하고, 그 텍스트에 "글리치" 같은 단어를 섞어 다시 소리로 생성(TTA)하는 과정을 반복합니다. 소리가 논리적으로 비약하며 변이됩니다. |
| **MVP-C** | Token Bending | EnCodec | 이산적인 오디오 토큰을 강제로 뒤섞거나(Local Shuffle) 비트를 뒤집습니다. 차갑고 날카롭게 부서지는 디지털 알갱이(Grit)를 만듭니다. |
| **MVP-D** | Checkpoint Morphing | RAVE (Hybrid) | 기타와 오르간 등 서로 다른 악기로 학습된 두 모델의 가중치를 결합(Re-basin)한 하이브리드 모델을 통과시킵니다. 물리 세계에 없는 유령 같은 잔향을 만듭니다. |
| **MVP-E** | Neural Granular | RAVE | 모델이 과거 40초간 처리한 잠재 공간의 '기억'을 버퍼에 담아 현재로 무작위 투사합니다. 시간이 겹치고 번지는 듯한 몽환적인 코러스를 만듭니다. |
| **MVP-F** | Spectral Frozen | RAVE | 잠재 공간의 상위 50%(고음역대)를 특정 시점의 값으로 주기적으로 동결시킵니다. 배경은 흐르는데 음색의 표면만 오로라처럼 일렁이는 Shimmer 효과를 냅니다. |
| **MVP-G** | Latent Feedback | RAVE | 잠재 벡터 스스로가 무한한 피드백 루프에 빠집니다. 단순한 에코가 아니라, 신경망이 계속해서 소리를 재해석하며 기괴하게 진화하는 메아리를 만듭니다. |
| **MVP-H** | Codebook Organ | EnCodec | 오디오 입력 없이, 소수(Prime)나 피보나치 수열 등 수학적 패턴으로 코드북 인덱스를 직접 생성합니다. 신경망의 원시적 언어가 연주되는 추상적 드론입니다. |
| **MVP-I** | Bass Massive | EnCodec | 기초 구조를 담당하는 하위 레이어 토큰들의 시간을 강제로 늘이거나(Smearing) 오차를 줍니다. 심해의 지진처럼 압도적이고 육중한 신경망 베이스를 만듭니다. |

---

## 2. 거시 신경망 파이프라인 (The Multi-Nets)
`scripts/multinet.py`는 위에서 정의된 기초 모듈(MVP A~I)들을 레고 블록처럼 조립하여, 복합적인 사운드 스케이프를 만들어내는 **매크로 네트워크(Macro-Networks)**를 정의합니다.

### Net 1: Crystal Cathedral (병렬 믹스)
*   **구조**: 원본 소스를 A, C, D, F, I 등 서로 다른 모듈에 병렬로 통과시킨 뒤 합칩니다.
*   **특징**: 여러 변조가 동시에 일어나며 스테레오 필드를 꽉 채우는 거대한 공간감을 형성합니다.

### Net 2: Recursive Organ (매크로 피드백 루프)
*   **구조**: `G -> A -> C -> F`로 이어지는 파이프라인을 통과한 결과물이 다시 입력으로 들어가 3번 순환합니다.
*   **특징**: 데미지가 기하급수적으로 누적되며 사운드의 정체성이 완전히 바뀝니다.

### Net 3: Decoding Chamber (순차적 누적 파괴)
*   **구조**: `A -> D -> E -> F -> G -> C -> I`를 일직선으로 통과시킵니다.
*   **특징**: 모든 종류의 신경망적 오해가 겹겹이 쌓인, 가장 짙고 어두운 텍스쳐를 생성합니다.

### Net Dynamic: Tempest (시간 기반 동적 연출)
*   **구조**: 60초의 타임라인을 가지고 있으며, 시간에 따라 특정 버스의 볼륨이 변하고, 15초/30초 구간에서 무음 강하(Silence Drop)나 폭발 같은 극적인 이벤트가 발생합니다.

---

## 3. 궁극의 결합: 메타 심포니 (The Meta-Symphony)
`scripts/meta_symphony.py`는 위의 거대 네트워크(Net 1~Dynamic)들을 다시 하나의 거대한 시간 축 위로 끌어모아 얽어내는(Interweaving) **"네트워크의 네트워크"**입니다.

### 아키텍처 다이어그램

```text
[ 3-Min Sub-Bass Seed ] ──┬──────────────────────────────────────────┐
(FM & Rhythmic Grains)    │                                          │
                          ▼                                          ▼
                [ Net 1: Crystal Cathedral ]               [ Net 2: Recursive Organ ]
                (공간감 확장 줄기)                            (피드백 루프 줄기)
                          │                                          │
                          ▼                                          ▼
                [ Net 3: Decoding Chamber ]                [ Net Dynamic: Tempest ]
                (파괴감 누적 줄기)                            (폭풍우 이벤트 줄기)
                          │                                          │
                  ==== Stem A ====                           ==== Stem B ====
                          │                                          │
                          ▼                                          ▼
                [ 60s Cycle LFO Crossfade ]                [ 45s Cycle LFO Crossfade ]
                (공간감과 파괴감이 1분마다 교차)                (루프와 폭풍우가 45초마다 교차)
                          │                                          │
                          ▼                                          ▼
                [ 20s Cycle Stereo Drift ]                 [ 25s Cycle Stereo Drift ]
                (좌우 공간을 서서히 유영)                      (좌우 공간을 서서히 유영)
                          │                                          │
                          └──────────────────────┬───────────────────┘
                                                 │
                                                 ▼
[ Original Sub-Bass ] ──> [ 100Hz Low-Pass ] ──> [ +8dB Bass Boost & Summing ]
                                                 │
                                                 ▼
                               [ Tanh Soft Limiter & LUFS Mastering ]
                                                 │
                                                 ▼
                                    [ META_SYMPHONY_FINAL.wav ]
```

### 사운드 디자인 철학
이 구조는 단순히 이펙터를 섞는 것이 아니라, 소리가 신경망을 거치며 어떻게 여러 갈래의 평행우주(Stem A, B)로 분화하고, 다시 시간의 흐름(LFO)에 따라 유기적으로 섞여 들어가는지를 보여줍니다. 바닥에는 결코 흔들리지 않는 원본의 베이스(Foundation)가 깔려 있고, 그 위로 인공지능이 해석한 소리의 파편들이 폭풍우처럼 휘몰아치는 **완결된 제너러티브 사운드 아트(Generative Sound Art) 작품**입니다.
