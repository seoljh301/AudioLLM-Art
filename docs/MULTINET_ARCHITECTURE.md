# Multinet Architecture

This document describes the macro-networks defined in `scripts/multinet.py` — five composite signal graphs that wire together the nine MVP modules (A–I) into reproducible sound-art compositions.

---

## 1. Legend & Conventions

| Symbol | Meaning |
|---|---|
| `[ S ]` | Input seed audio |
| `[ A ]` … `[ I ]` | MVP modules |
| `==>>` | Audio signal flow |
| `[ + ]` | Sum / mixing bus |
| `α`…`θ` | Greek-letter labelled buses |

---

## 2. Net 1 — Crystal Cathedral (Parallel Mix)

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

## 3. Net 2 — Recursive Organ (Macro-Feedback)

```text
          ┌────────────────────────────────────────────────────────┐ (Loop 3x)
          │                                                        │
[ S ] ==>>┼==> [ G ] ==>> [ A ] ==>> [ C ] ==>> [ F ] ==>> [ + ] ==┴==> [ Soft Limiter ] ==>> MASTER
               (Echo)   (Latent)   (Token)   (Freeze)    │
                                                     (Update Seed)
```

---

## 4. Net 3 — Decoding Chamber (Sequential Destruction)

```text
[ S ] ==>> [ A ] ==>> [ D ] ==>> [ E ] ==>> [ F ] ==>> [ G ] ==>> [ C ] ==>> [ I ] ──┐
          (Latent)  (Morph)   (Granular) (Freeze)   (Echo)    (Token)  (Massive)   │
                                                                                   │
[ S ] ==>> [ 80Hz Low-Pass Filter (Anchor +8dB) ] ─────────────────────────────────┼==>> [ Limiter ] ==>> MASTER
```

---

## 5. Net Max — Cathedral Hive (Complex Multi-Bus)

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

## 6. Net Dynamic — Tempest (Time-Varying Narrative)

```text
[ Bus α ~ θ ]
      │
      ▼
[ Dynamic Volume Envelopes ]  <-- (60s Composition Arc)
      │
      ▼
[ Impulse Events ]
  ├── 15s: [ F ] Freeze Click
  ├── 30s: Silence Drop
  └── 45s: [ H ] Drone Burst
      │
      ▼
[ Master Filter Sweep ]  <-- (250Hz → 16kHz Moving Low-Pass)
      │
      ▼
[ 80Hz Anchor Mix & Limiter ] ==>> MASTER
```

---

## 7. 성능 및 수치 비교 (Quantitative Comparison)

| 지표 | Net 1 | Net 2 | Net 3 | Net Max | Net Dynamic |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **연산 속도 (V100)** | ≈ 1.5s | ≈ 2.0s | ≈ 2.0s | ≈ 24s | ≈ 20s |
| **VRAM 점유율** | ≈ 3GB | ≈ 3GB | ≈ 3GB | ≈ 5GB | ≈ 5GB |
| **사용된 MVP 수** | 7개 | 4개 | 7개 | 9개 | 9개 |
| **다이내믹 (RMS)** | ≈ 4dB | ≈ 3dB | ≈ 5dB | ≈ 4dB | **≈ 22dB** |
