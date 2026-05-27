# Multinet Architecture

This document describes the macro-networks defined in `scripts/multinet.py` — five composite signal graphs that wire together the nine MVP modules (A–I) into reproducible sound-art compositions.

> Companion docs:
> - [`META_SYMPHONY_ARCHITECTURE.md`](META_SYMPHONY_ARCHITECTURE.md) — the meta-network that interweaves these five nets
> - [`feedback_1.md`](feedback_1.md) — design principles ("Anchored Corruption", Texture Guard)
> - [`IMPLEMENTATION_REPORT_V1.md`](IMPLEMENTATION_REPORT_V1.md) — implementation log
> - [`related_work.md`](related_work.md) — prior art survey

---

## Contents

1. [Legend & Conventions](#1-legend--conventions)
2. [Module Quick Reference](#2-module-quick-reference)
3. [Common Infrastructure](#3-common-infrastructure)
4. [Net 1 — Crystal Cathedral](#4-net-1--crystal-cathedral)
5. [Net 2 — Recursive Organ](#5-net-2--recursive-organ)
6. [Net 3 — Decoding Chamber](#6-net-3--decoding-chamber)
7. [Net Max — Cathedral Hive](#7-net-max--cathedral-hive)
8. [Net Dynamic — Tempest](#8-net-dynamic--tempest)
9. [Cross-Net Comparison](#9-cross-net-comparison)
10. [Operational Notes](#10-operational-notes)

---

## 1. Legend & Conventions

| Symbol | Meaning |
|---|---|
| `[ S ]` | Input seed audio |
| `[ A ]` … `[ I ]` | MVP modules (see §2) |
| `==>>` | Audio signal flow at 48 kHz unless marked otherwise |
| `→24k→` | Forced resample to 24 kHz (EnCodec) |
| `→48k→` | Resample back to 48 kHz |
| `[ + ]` | Sum / mixing bus |
| `[ × ]` | Multiply (envelope, gain) |
| `α`…`θ` | Greek-letter labelled buses (parallel branches) |
| `dw` | dry/wet ratio (0 = pure dry input, 1 = pure damaged) |
| `t` | Morph interpolation coefficient for MVP-D (0 = guitar, 1 = organ) |
| `RaveHandle` | Wraps RAVE TorchScript model + sample rate + latent dim |
| `CodecHandle` | Wraps EnCodec (24 kHz, 8 quantizers, codebook 1024) |

Each MVP render internally:
1. Chunks audio at `chunk_seconds=4.0` (overlap 0.05).
2. Encodes (RAVE: → latent z, EnCodec: → discrete tokens).
3. Applies module-specific damage.
4. Decodes back to audio.
5. Runs **Texture Governor** (NaN-stop + auto-reduce wet on flatness > 0.55 / RMS bounds / centroid > Nyquist × 0.42).
6. Mixes dry+wet via `src/core/mix.py` (RMS match + soft tanh limiter + optional 80 Hz crossover anchor).

---

## 2. Module Quick Reference

| MVP | Backbone | Acts on | Key params | Sonic signature |
|---|---|---|---|---|
| **A** Latent Perturbation | RAVE | latent z | `noise_scale`, `dim_dropout`, `noise_mode={white,smoothed}`, `noise_smooth=0.98` | organic timbral wobble (smoothed) or grit (white) |
| **B** Caption Loop | Qwen2-Audio + AudioLDM2 *(or stub)* | audio → text → audio | `depth=3`, `text_mutation_prob=0.5` | semantic mistranslation drift |
| **C** Token Bending | EnCodec 24k | RVQ tokens (n_q, T) | `mode={bit_flip, quantizer_drop, shuffle, invalid_token}`, `rate`, `quantizer_range`, `shuffle_window` | digital grit, codec mis-decode |
| **D** Checkpoint Morphing | RAVE (2 ckpts) | model weights | `t` (linear/slerp), `rebasin_mode={partial,full,off}` | hybrid instrument / collapse-ghost |
| **E** Neural Granular | RAVE | latent z + memory buffer | `grain_size=16`, `memory_size=4096`, `num_grains=5`, `mix=0.5` | smeared-time chorus |
| **F** Spectral Frozen | RAVE | upper latent dims | `auto_upper_fraction=0.3–0.6`, `update_interval=64–128`, `crossfade_frames=16` | shimmer / aurora |
| **G** Latent Feedback | RAVE | latent z with delay line | `delay_frames=16–96`, `feedback=0.30–0.55`, `mix=0.5–0.7` | evolving neural echo |
| **H** Codebook Organ | EnCodec | generative tokens | `mode={prime,fibonacci,random_walk}`, `stride`, `duration_frames` | abstract drone, no audio input |
| **I** Bass Massive | EnCodec | lower RVQ tokens | `smear_delay`, `smear_quantizers=(0,2)`, `jitter_rate`, `fold_leak_rate` | hyper-low sub-bass smear |

---

## 3. Common Infrastructure

### Sample-rate boundary

```
seed audio  (48 kHz, RAVE-native)
    │
    ├── RAVE stages (A, D, E, F, G)  ───── stay 48 kHz
    │
    └── EnCodec stages (C, H, I)  ──→ resample 48k → 24k → encode → tokens
                                              ↑                 │
                                              └─── decode ──→ resample 24k → 48k
```

`_resample_to(audio, n)` (linear interp, `np.interp`) handles boundary. Each round-trip introduces ≈ −1 dB high-frequency loss above 12 kHz.

### Texture Guard chain

```
damaged_chunk
    ├── compute_texture_metrics()      → (rms, flatness, centroid, ZCR, has_nan)
    ├── govern_wet()                   → wet_per_chunk ∈ [min_wet, base_wet]
    │      • has_nan       → wet = 0.0  (emergency)
    │      • flatness>0.55 → wet ×= 0.55
    │      • rms<1e-4      → wet ×= 0.35
    │      • rms>0.85      → wet ×= 0.60
    │      • centroid > Nyquist·0.42 → wet ×= 0.75
    ├── dry_wet_mix()                  → (1-wet)·dry + wet·damaged
    │      • optional RMS match
    │      • optional 80 Hz crossover  (preserve dry sub + add sub_boost_db gain)
    └── soft_limiter(drive=1.0–1.25)   → tanh saturation
```

This same guard runs inside *every* MVP render function. No bus can deliver NaN, runaway gain, or pure noise to the master sum.

---

## 4. Net 1 — Crystal Cathedral

**Topology class**: 5-bus parallel mix.
**Seed**: 10 s 440 Hz sine @ 48 kHz.
**Wall time on V100**: ≈ 1.5 s.

```
       ┌==>> [ I (Bass Massive) ]                              ==>> (Bus L: 35%) ┐
       │     smear=12, jitter=0.05, q=(0,2)                                      │
       │                                                                         │
       ├==>> [ A ] ==>> [ D ] ==>> [ C ]                       ==>> (Bus M: 30%) ┤
       │     noise=0.05 smoothed   t=0.005    bit_flip rate=0.03                 │
       │                                                                         │
[ S ] =┼==>> [ F ] ==>> [ A ]                                  ==>> (Bus H: 20%) ┼==> [ 80Hz Anchor +6dB ] ==> [ Soft Limiter drive=1.2 ] ==>> MASTER
       │     upper 50%  noise=0.10                                                │
       │                                                                         │
       ├==>> [ G ] ==>> [ E ]                                  ==>> (Bus T: 10%) ┤
       │     d=32 fb=0.4   mem=2048 grain=16                                     │
       │                                                                         │
     (Gen)>> [ H (prime, stride=7) ]                           ==>> (Bus D: 05%) ┘
```

### Bus design rationale

| Bus | Role | Why |
|---|---|---|
| **L** Foundation | sub-bass anchor | I + 80 Hz crossover keeps the sine fundamental physically present even when damage is heavy. |
| **M** Core | tonal body | A → D → C accumulates 3 different damage modes (latent noise → weight blend → token bend) at low dw. |
| **H** Shimmer | high band | F freeze creates shimmer; A adds organic wobble. |
| **T** Recursive | echo + memory | G feeds back, E projects past grains forward — produces a "ghost reverb" tail. |
| **D** Drone | generative bed | H plays prime-number tokens as a chordless harmonic series. |

### Observed master metrics

| | rms | peak | centroid | flatness | top freq bins |
|---|---|---|---|---|---|
| Net 1 master | 0.232 | 0.95 | 1446 Hz | 0.117 | 439–450 + 49–82 (drone) |

The 440 Hz seed peak is preserved (top bins 439–442) while the H drone bus introduces 49/76/78/81/82 Hz partials — six octave bands occupied simultaneously.

---

## 5. Net 2 — Recursive Organ

**Topology class**: serial chain wrapped in macro-feedback (N passes).
**Seed**: 10 s sine.
**Passes**: 3 default (overridable via `passes=` arg).
**Wall time on V100**: ≈ 2 s.

```
                    ┌────────────────────────────────────────────────────────────────┐  (Loop N=3×)
                    │                                                                │
[ S ] ==>>> seed ==>┤  [ G ]    [ A ]    [ C ]    [ F ]    ==> chunk-master ==> ──┐  │
                    │  d=64    noise=0.08  shuffle  upper 30%                     │  │
                    │  fb=0.55 drop=0.15 win=12     update=64                     │  │
                    │  mix=0.7 mode=white rate=0.04 fade=12                       │  │
                    │                                                              │  │
                    │  pass 0 → pass_0.wav                                         │  │
                    │  pass 1 → pass_1.wav  ← pass_0 as new seed                   │  │
                    │  pass 2 → pass_2.wav  ← pass_1 as new seed                   │  │
                    └─── final pass ── soft limiter drive=1.15 ──> MASTER ◄────────┘  
                                                                                      
                                              SR conversion inside C only:           
                                              48k → 24k (encode) → bend → 24k → 48k  
```

Each pass adds compounding damage. The model's pitch identity drifts upward across passes: pass 0 keeps 440 Hz, pass 2 lands at 597–615 Hz (top spectral bins).

### Per-pass observed top frequency bins

| Pass | rms | top freq bins | note |
|---|---|---|---|
| 0 | 0.075 | 439, 440, 441, 448, 452 | seed pitch held |
| 1 | 0.052 | 398, 401, 405, 410, 422 | sliding down ≈ −40 Hz |
| 2 | 0.047 | 597, 605, 608, 611, 615 | jumped up ≈ +157 Hz |
| **MASTER** | 0.058 | 597–615 | matches final pass |

The non-monotonic pitch wandering is characteristic of recursive feedback through a learned latent space — the network's "interpretation" of its own previous output drives the pitch.

---

## 6. Net 3 — Decoding Chamber

**Topology class**: linear 9-stage with sub-bass anchor branch.
**Seed**: 10 s sine.
**Per-stage dw**: 0.45 (gentle so damage accumulates without collapse).
**Wall time on V100**: ≈ 2 s.

```
[ S ] ==>> [ A ] ==>> [ D ] ==>> [ E ] ==>> [ F ] ==>> [ G ] ==>> [ C ] ==>> [ I ] ─┐
          n=0.03    t=0.005   mem=1024   25%       d=24    bit_flip   smear=8     │
          smoothed             grain=16   upd=128   fb=0.30 rate=0.02  jitter=0.03 │
                              num=4                                                │
                                                                                   │
                                                                              ─────┤
[ S ] ==>> [ 80 Hz Low-Pass (Butterworth 2nd) ] × +8 dB ──────────────────────────┤
                                                                                   │
                                                                                   ▼
                                                       [ + ] ==>> [ Limiter drive=1.2 ] ==>> MASTER
```

Layer-by-layer rms shows the dry/wet anchoring at work:

| Stage | After applying | rms | spectral focus |
|---|---|---|---|
| s1 | A | 0.105 | 440 ± 5 |
| s2 | D | 0.033 | 440 + 627 |
| s3 | E | 0.026 | 420–445 |
| s4 | F | 0.024 | 416 + 626 |
| s5 | G | 0.021 | 439–443 |
| s6 | C | 0.027 | 439–443 |
| s7 | I | 0.027 | 207 + 439–442 |
| **MASTER** (with 80 Hz anchor) | **0.177** | 207 + 439–442 |

The 80 Hz crossover branch single-handedly resurrects RMS by re-injecting the dry sine through a low-shelved path — the **anchored corruption** principle in action.

---

## 7. Net Max — Cathedral Hive

**Topology class**: 8-bus parallel mix + cross-bus feedback + 2-pass macro-loop.
**Seed**: 30 s enriched sine (440 Hz fundamental + 660 Hz 5th + 0.13 Hz tremolo + slow pitch drift).
**Wall time on V100**: ≈ 24 s.

```
                                          ┌── morph_guitar (t=0.005) ──── pre-loaded once
                                          ├── morph_organ  (t=0.995) ──── pre-loaded once
                                          ├── codec (EnCodec 24k, 8 q) ── pre-loaded once
                                          ├── rave_guitar ────────────── pre-loaded once
                                          
       ┌==>> [ I ] ==>> [ C(invalid lower) ] ==>> [ 80Hz anchor +10dB ] ==>> (α Foundation)
       │
       ├==>> [ A ] ==>> [ D(guitar) ] ==>> [ E ] ==>> [ G ] ==>> [ A(drop) ] ==>> [ F ]  ==>> (β Core)
       │     smooth      t=0.005       mem=4096   d=48     drop=0.2          30%
       │
       ├==>> [ F(60%) ] ==>> [ G ] ==>> [ C(shuffle) ] ==>> [ F(40%) ]                ==>> (γ Ghost)
       │
[ S ] =┤==>> [ D(organ t=0.995) ] ==>> [ A(n=0.18) ] ==>> [ C(bit_flip) ] ==>> [ E ]  ==>> (δ Twin)
       │
       ├==>> [ C(bit_flip rate=0.10) ] ==>> [ I(smear+jitter+fold) ] ==>> [ G ]       ==>> (ε Glitch)
       │
     (Gen)>>[ H(prime stride=11) ] ┐
            [ H(fibonacci stride=5)]┘──>> mix 55:45 ==>> [ A(smoothed) ]              ==>> (ζ Drone)
       │
       ├==>> [ B(caption→TTA depth=3) stub backend ]                                  ==>> (η Loop-B)
       │
       └── tap β output ────────────────>> [ G(deep d=96 fb=0.55) ] ==>> [ I(fold) ]  ==>> (θ XFB)

         pass 1 → mix_buses (α20% β18% γ12% δ12% ε10% ζ10% η8% θ10%) → MASTER_pass1
         
         pass 2 → seed′ = 0.55·MASTER_pass1 + 0.45·seed (RMS-matched)
              → re-run all 8 buses on seed′
              → mix_buses (α15% β22% γ15% δ10% ε8% ζ8% η6% θ16%) → MASTER_pass2
         
         FINAL = pass1 × (1 − S(t)) + pass2 × S(t)  where  S(t) = ½(1 − cos(π·t/T))
```

### Why these specific buses

| Bus | Pillar | What it adds to the texture |
|---|---|---|
| **α** | sub-bass spine | I→C combination kills the upper detail of bass tokens but leaves the body; 80 Hz crossover re-injects clean sine sub. |
| **β** | structural core | 6-stage chain that compresses through every RAVE-domain damage mode (latent, weight, granular, feedback, dropout, freeze). |
| **γ** | shimmer | nested F→F freezes between G echo + C shuffle generate a sustained aurora. |
| **δ** | organ-side counter-melody | D at t=0.995 swings model identity to organ; remaining stages add codec grit. |
| **ε** | breakage | aggressive C(rate=0.10) + I(fold) + G makes the maximally damaged bus. |
| **ζ** | non-input drone | H generative tokens give a chordless bed independent of the seed. |
| **η** | caption-loop | semantic drift at audio scale; stub captioner produces deterministic adjective chains. |
| **θ** | cross-feedback | reads β output, deep-echoes + folds — gives the master a "ghost" of the core. |

### Pass-2 weight redistribution

Weights shift to emphasise refed-seed material:

|  | α | β | γ | δ | ε | ζ | η | θ |
|---|---|---|---|---|---|---|---|---|
| pass 1 | 0.20 | 0.18 | 0.12 | 0.12 | 0.10 | 0.10 | 0.08 | 0.10 |
| pass 2 | 0.15 | 0.22 | 0.15 | 0.10 | 0.08 | 0.08 | 0.06 | 0.16 |

Pass 2 amplifies β (deeply-processed core) and θ (cross-feedback ghost) while reducing the generative drone and caption loop — the composition tightens around the seed-derived material.

### Observed metrics

| | rms | peak | centroid | flatness | top bins | governor events |
|---|---|---|---|---|---|---|
| pass 1 master | 0.226 | 0.95 | 1994 Hz | 0.158 | 439–442, 449 | 0 |
| pass 2 master | 0.208 | 0.95 | 1897 Hz | 0.140 | 439, 449–451, 466 | 1 (θ flat) |
| **FINAL** | **0.260** | **0.95** | **1996 Hz** | **0.159** | 439–442, 449 | — |

Six independent pitch zones are present in the final master: sub 49–82 Hz (drone), 207 (anchor LP residual), 337 (β drift), 432–444 (δ detune), 439–442 (α/γ/ε anchor), 660 (seed 5th + sidebands).

---

## 8. Net Dynamic — Tempest

**Topology class**: 8 buses + per-bus time-varying amplitude envelopes + impulse events + master filter sweep.
**Seed**: 60 s enriched sine (220→660 Hz pitch sweep + ±6 Hz vibrato + amplitude arc + 6 s sparse onsets + 30 s gaussian dip).
**Wall time on V100**: ≈ 20 s.

The bus topology is identical to Net Max **pass 1** (no macro loop). The dynamism comes from post-render automation.

```
[ same 8 buses α…θ as Net Max ]   →   each rendered ONCE at full 60 s length
        │
        ▼
[ per-bus amplitude envelopes  (piecewise linear breakpoints, smoothed ~0.3 s) ]

  Bus   t→  0s    10s   15s   22s   30s   35s   45s   55s   60s
  α          0.45  0.40  0.40  0.30  0.25  0.35  0.30  0.25  0.20
  β          0.00  0.00  0.30  0.55  0.10  0.05  0.05  0.10  0.05
  γ          0.00  0.20  0.35  0.20  0.05  0.00  0.00  0.20  0.15
  δ          0.00  0.00  0.00  0.20  0.30→0.05  0.10        0.00
  ε          0.00              0.00  0.00  0.30  0.55  0.05  0.00
  ζ          0.00                    0.05  0.20  0.30  0.30  0.25
  η          0.00                          0.00  0.20  0.40→0.30→0.10
  θ          0.00                                0.15  0.35  0.45  0.35

        │
        ▼
[ Impulse events ]
   ├── t=15.0–15.3 s  : γ freeze CLICK (×2.0 amplitude burst)
   ├── t=30.0–31.0 s  : SILENCE DROP (linear ramp 1.0 → 0.05, +0.1 s recovery ramp)
   └── t=44.5–45.5 s  : ζ DRONE BURST (×1.5, 50 ms attack / 100 ms release envelope)

        │
        ▼
[ Master lowpass sweep ]   cutoff trajectory (Hz, piecewise linear, smoothed):
   0s 250 → 10s 4k → 20s 12k → 28s 11k → 30s 500 → 32s 8k → 40s 16k → 50s 14k → 55s 6k → 60s 4k
   block size 4096 samples (~85 ms resolution), 2nd-order Butterworth with continuous state (lfilter_zi)

        │
        ▼
[ 80 Hz crossover ] : keep dry seed below 80 Hz (+8 dB) + filtered master above
        │
        ▼
[ Soft limiter drive=1.25 ] ==>> MASTER (rms ≈ 0.3–0.6, dynamic)
```

### Observed per-second master RMS (sample)

| t (s) | rms | event |
|---|---|---|
| 0 | 0.520 | α-only ground |
| 15 | 0.305 | freeze click landed |
| 20 | 0.122 | filter still narrow + β rising |
| 30 | 0.533 | silence drop already recovered |
| 40 | 0.471 | ε peak |
| 45 | 0.293 | drone burst envelope crest |
| 50 | 0.186 | filter closing on η |
| 55 | 0.480 | η + θ climax |
| 60 | 0.459 | tail fade |

Master RMS range across the 60 s: **0.046 to 0.605** (≈ 22 dB dynamic range). Contrast with Net Max FINAL (0.260 mean, range ≈ 0.18 – 0.31, ≈ 4 dB) — Tempest is roughly 5× more dynamic.

---

## 9. Cross-Net Comparison

| | Net 1 | Net 2 | Net 3 | Net Max | Net Dynamic |
|---|---|---|---|---|---|
| **Class** | parallel mix | recursive macro-loop | linear chain | 2-pass parallel + xfb | time-varying parallel |
| **Buses** | 5 | 1 (looped) | 1 (deep) | 8 × 2 | 8 |
| **Modules used** | A C D F G H I (7) | A C F G (4) | A C D E F G I (7) | A B C D E F G H I (9) | A B C D E F G H I (9) |
| **MVP-B used?** | no | no | no | yes | yes |
| **Macro feedback?** | no | yes (3 passes) | no | yes (2 passes) | no |
| **Cross-bus FB?** | no | no | no | yes (θ taps β) | no |
| **Time-varying?** | no | per-pass only | no | per-pass weights | per-second |
| **Stereo?** | mono | mono | mono | mono | mono |
| **Filter sweep?** | no | no | no | no | yes (master LP) |
| **Impulse events?** | no | no | no | no | yes (3 events) |
| **Default seed length** | 10 s | 10 s | 10 s | 30 s | 60 s |
| **Wall time on V100** | 1.5 s | 2 s | 2 s | 24 s | 20 s |
| **Master rms** | 0.232 | 0.058 | 0.177 | 0.260 | 0.30 mean (0.05–0.60) |
| **Texture governor trips** | 0 | 0 | 0 | 1 | several |
| **NaN emergencies** | 0 | 0 | 0 | 0 | 0 |

Wall-time scales roughly linearly with `seed_seconds × n_buses` because most stages are GPU-limited at 5–10 × realtime.

---

## 10. Operational Notes

### Resource usage

| Resource | Net 1 | Net Max | Net Dynamic |
|---|---|---|---|
| Peak VRAM | ≈ 3 GB | ≈ 5 GB | ≈ 5 GB |
| Disk per render | 6 wavs ≈ 6 MB | 19 wavs ≈ 56 MB | 9 wavs + 2 csv ≈ 150 MB |
| CPU threads | 1 (linear I/O) | 1 | 1 |

### Reproducibility

Every stage takes `rng_seed=` so renders are bit-reproducible given:
1. Same RAVE / EnCodec model files (SHA-checked against HF download).
2. Same torch version (`2.4.1+cu121`).
3. Same GPU (V100 vs A100 produces minor floating-point divergence).
4. Same seed audio file.

### Known issues

| Issue | Affected | Mitigation |
|---|---|---|
| `bent=0 diffs` log in C stage when `q_range=(-3, 0)` + low rate | Net 1 M bus, Meta-Symphony N1 input | Net 3 still bends — likely interaction with extreme `quantizer_range` slicing at high SR boundary. Investigate `token_bend.bend_tokens` slice path. |
| MVP-D collapse at intermediate `t` | all nets that morph guitar↔organ | Use only endpoint regions: `t ∈ [0, 0.02]` or `t ∈ [0.98, 1]`. See [`feedback_1.md`](feedback_1.md) §5. |
| Net Dynamic envelope freezes after 60 s if fed longer input | Meta-Symphony Phase 1.4 | Intentional drone-after-tail per Meta-Symphony design. |

### CLI

```bash
# Single net
python scripts/multinet.py net1     # 10 s seed, 5-bus parallel
python scripts/multinet.py net2     # 10 s seed, 3-pass recursive
python scripts/multinet.py net3     # 10 s seed, linear 9-stage
python scripts/multinet.py max      # 30 s enriched seed, 8-bus 2-pass
python scripts/multinet.py all      # runs Net 1 + 2 + 3 (and max if seed_max_30s.wav exists)
```

Output landing zones:
```
runs/multinet/
├── sine_440_10s.wav
├── sine_max_30s.wav
├── sine_dyn_60s.wav
├── net1/   bus_{L,M,H,T,D}.wav  + MASTER.wav
├── net2/   pass_{0..N}.wav      + MASTER.wav
├── net3/   s{1..7}_X.wav        + MASTER.wav
├── net_max/      p{1,2}_bus_{α..θ}.wav + MASTER_pass{1,2}.wav + MASTER_FINAL.wav
└── net_dynamic/  bus_{α..θ}.wav + envelopes.csv + master_rms_per_second.csv + MASTER_FINAL.wav
```
