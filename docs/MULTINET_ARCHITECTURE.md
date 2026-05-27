# Multinet Architecture Diagrams

이 문서는 `scripts/multinet.py`에 정의된 복합 신경망 사운드 파이프라인(Net 1 ~ Net Dynamic)의 신호 흐름을 시각적으로 설명합니다.
각 MVP 모듈(A~I)이 어떻게 직렬/병렬로 연결되고 피드백되는지 나타냅니다.

## 범례 (Legend)
*   `[ S ]` : Input Seed Audio (입력 오디오)
*   `[ A ]` : MVP-A (Latent Perturbation)
*   `[ B ]` : MVP-B (Audio-Caption-TTA Loop)
*   `[ C ]` : MVP-C (Token Bending)
*   `[ D ]` : MVP-D (Checkpoint Morphing)
*   `[ E ]` : MVP-E (Neural Latent Granular)
*   `[ F ]` : MVP-F (Neural Spectral Frozen)
*   `[ G ]` : MVP-G (Latent Feedback Echo)
*   `[ H ]` : MVP-H (Codebook Organ / Generative)
*   `[ I ]` : MVP-I (Neural Bass Massive)
*   `===>>` : Signal Flow (신호 흐름)
*   `[ + ]` : Summing / Mix Bus (믹싱 데스크)

---

## 1. Net 1: Crystal Cathedral (5-Bus Parallel Mix)
서로 다른 변조를 거친 5개의 병렬 버스를 합쳐 거대한 공간감을 만듭니다.

```text
       ┌==>> [ I (Bass Massive) ] =============================>> (Bus L: 35%) ┐
       │                                                                       │
       ├==>> [ A (Latent) ] ==>> [ D (Morph) ] ==>> [ C (Token) ] >> (Bus M: 30%) ┤
       │                                                                       │
[ S ] =┼==>> [ F (Freeze) ] ==>> [ A (Latent) ] =================>> (Bus H: 20%) ┼==> [ 80Hz Anchor Boost ] ==>> [ Soft Limiter ] ==>> MASTER
       │                                                                       │
       ├==>> [ G (Echo) ] ====>> [ E (Granular) ] ===============>> (Bus T: 10%) ┤
       │                                                                       │
     (Gen)>> [ H (Organ) ] ====================================>> (Bus D: 05%) ┘
```

---

## 2. Net 2: Recursive Organ (3-Pass Macro Loop)
출력물을 다시 입력으로 사용하는 거대한 매크로 피드백(Macro-Feedback) 루프입니다.

```text
          ┌────────────────────────────────────────────────────────┐ (Loop 3x)
          │                                                        │
[ S ] ==>>┼==> [ G ] ==>> [ A ] ==>> [ C ] ==>> [ F ] ==>> [ + ] ==┴==> [ Soft Limiter ] ==>> MASTER
               (Echo)   (Latent)   (Token)   (Freeze)    │
                                                     (Update Seed)
```

---

## 3. Net 3: Decoding Chamber (Linear 9-Stage)
모든 데미지를 직렬로 누적시키는 순차적 파괴 파이프라인입니다.

```text
[ S ] ==>> [ A ] ==>> [ D ] ==>> [ E ] ==>> [ F ] ==>> [ G ] ==>> [ C ] ==>> [ I ] ──┐
          (Latent)  (Morph)   (Granular) (Freeze)   (Echo)    (Token)  (Massive)   │
                                                                                   │
[ S ] ==>> [ 80Hz Low-Pass Filter (Anchor +8dB) ] ─────────────────────────────────┼==>> [ Limiter ] ==>> MASTER
```

---

## 4. Net Max: Cathedral Hive (8-Bus + Cross-Feedback)
8개의 버스와 버스 간 피드백(Cross-Feedback)이 결합된 가장 복잡한 구조입니다.

```text
       ┌==>> [ I ] ==>> [ C ] ==>> [ 80Hz Anchor ] ===========================>> (Bus α: Foundation) ┐
       │                                                                                           │
       ├==>> [ A ] ==>> [ D(Guitar) ] ==>> [ E ] ==>> [ G ] ==>> [ A ] ==>> [ F ] => (Bus β: Core) ┤
       │                                                                       │                   │
       ├==>> [ F ] ==>> [ G ] ==>> [ C ] ==>> [ F ] ========================== │ ==> (Bus γ: Ghost) ┤
       │                                                                       │                   │
[ S ] =┼==>> [ D(Organ) ] ==>> [ A ] ==>> [ C ] ==>> [ E ] =================== │ ==> (Bus δ: Twin)  ┼==> [ Master ]
       │                                                                       │                   │     (Pass 1 & 2
       ├==>> [ C ] ==>> [ I ] ==>> [ G ] ===================================== │ ==> (Bus ε: Glitch)┤     Blend)
       │                                                                       │                   │
     (Gen)>> [ H(Prime) + H(Fibo) ] ==>> [ A ] =============================== │ ==> (Bus ζ: Drone) ┤
       │                                                                       │                   │
       ├==>> [ B (Caption Loop) ] ============================================ │ ==> (Bus η: Loop-B)┤
       │                                                                       │                   │
       └──────────────────────────────────────────────────────────── (β Out) ──┴=>> [ G ] ==>> [ I ] => (Bus θ: XFB)
```

---

## 5. Net Dynamic: Tempest (Time-Varying Events)
Net Max와 동일한 8개의 버스를 사용하지만, 정적 믹스가 아닌 60초의 시간축(Timeline)에 따라 볼륨과 필터가 요동칩니다.

```text
[ Bus α ~ θ ]
      │
      ▼
[ Dynamic Volume Envelopes ]  <-- 각 버스의 볼륨이 0.0 ~ 0.55 사이에서 곡의 서사(Arc)에 따라 페이드 인/아웃 됨.
      │
      ▼
[ Impulse Events ]
  ├── 15s: [ F ] Freeze Click (얼어붙은 파편 폭발)
  ├── 30s: Silence Drop (모든 소리가 0.05 레벨로 추락)
  └── 45s: [ H ] Drone Burst (저음역대 오르간 폭발)
      │
      ▼
[ Master Filter Sweep ]  <-- 전체 사운드를 덮는 Moving Low-Pass Filter
      │                      (250Hz에서 시작해 16kHz까지 요동치며 열리고 닫힘)
      ▼
[ 80Hz Anchor Mix & Limiter ] ==>> MASTER (60s Composition)
```
