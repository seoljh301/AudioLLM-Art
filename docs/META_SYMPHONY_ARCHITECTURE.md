# Meta-Symphony Architecture

이 문서는 `scripts/meta_symphony.py`를 통해 생성된 **"네트워크의 네트워크 (The Network of Networks)"** 구조를 시각화합니다. 
개별적인 MVP 모듈(A~I)을 넘어, `multinet.py`에 정의된 거대 파이프라인(Net 1 ~ Net Dynamic)들이 어떻게 서로 먹이고 먹히며 얽혀있는지(Interweaving) 보여줍니다.

---

## 1. 기반 시드 (The Foundation)
모든 렌더링의 뼈대가 되는 3분(180초) 길이의 초저역대 사운드입니다.
*   **Sub-Bass (30Hz, 40Hz, 55Hz)**: 육중한 바닥을 형성.
*   **FM Modulation**: 신경망 모델(RAVE)이 물고 뜯을 수 있는 질감(Grip) 제공.
*   **Macro Breathing**: 30초 주기의 거대한 볼륨 LFO.

---

## 2. 거시 신경망 라우팅 (Macro-Network Routing)
시드 오디오가 4개의 서로 다른 성격을 가진 거대 네트워크로 나뉘어 투입됩니다. 특징적인 것은 **이전 네트워크의 결과물이 다음 네트워크의 입력으로 사용된다는 점**입니다.

```text
[ 3-Min Sub-Bass Seed ] ──┬──────────────────────────────────────────┐
                          │                                          │
                          ▼                                          ▼
                [ Net 1: Crystal Cathedral ]               [ Net 2: Recursive Organ ]
                (5-Bus 병렬 공간감 확장)                     (2-Pass 매크로 피드백 루프)
                          │                                          │
                          ▼                                          ▼
                [ Net 3: Decoding Chamber ]                [ Net Dynamic: Tempest ]
                (9단계 직렬 누적 파괴)                       (60초 주기 동적 이벤트 변조)
                          │                                          │
             (Stem A: 공간감 vs 파괴감)                  (Stem B: 무한루프 vs 폭풍우)
```

---

## 3. 얽힘 및 공간화 (Interweaving & Spatialization)
생성된 두 그룹의 줄기(Stem A, Stem B)를 단순히 섞는 것이 아니라, 서로 다른 주기의 LFO를 이용해 교차시키고 스테레오 공간에서 유영하게 만듭니다.

```text
[ Stem A ]                                    [ Stem B ]
(Net 1 & Net 3)                               (Net 2 & Net Dynamic)
     │                                             │
     ▼                                             ▼
[ 60s Cycle LFO Crossfade ]                   [ 45s Cycle LFO Crossfade ]
(공간감과 파괴감이 1분에 한 번씩 교대)           (루프와 폭풍우가 45초마다 교대)
     │                                             │
     ▼                                             ▼
[ 20s Cycle Stereo Drift ]                    [ 25s Cycle Stereo Drift ]
(공간의 좌/우를 20초 주기로 천천히 유영)           (공간의 좌/우를 25초 주기로 천천히 유영)
     │                                             │
     └──────────────────────┬──────────────────────┘
                            │
                            ▼
                     [ Master Summing ]
```

---

## 4. 최종 마스터링 (Final Polish)
얽히고설킨 텍스쳐가 허공에 뜨지 않도록, 원본 시드의 극저역대만 추출하여 믹스의 정중앙에 강력하게 박아 넣습니다.

```text
[ Master Summing ] ─────────────────┐
                                    │
[ Original Seed ] ──> [ 100Hz Low-Pass Filter ] ──> ( +8dB Sub-Bass Boost )
                                    │
                                    ▼
                          [ Tanh Soft Limiter ]  <-- (피크 제어 및 아날로그 배음 포화)
                                    │
                                    ▼
                      [ META_SYMPHONY_FINAL.wav ]
```

### 사운드 디자인 요약
이 아키텍처는 하나의 소리가 신경망을 거치며 어떻게 여러 갈래의 평행우주로 분화하고, 다시 하나의 거대한 시간 축 안에서 숨을 쉬며 섞여 들어가는지를 극적으로 보여주는 **오디오 제너레이티브 아트(Audio Generative Art)의 결정체**입니다.
