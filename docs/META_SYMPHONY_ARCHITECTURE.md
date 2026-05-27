# Meta-Symphony Architecture

`scripts/meta_symphony.py` is the apex of the AudioArt stack: a **network of macro-networks** that takes a 3-minute sub-bass seed, runs it through all four primary nets (Net 1 / 2 / 3 / Dynamic), then interweaves their outputs across a 3-min timeline with crossfading LFOs and stereo drift.

> Companion docs:
> - [`MULTINET_ARCHITECTURE.md`](MULTINET_ARCHITECTURE.md) — the macro-nets that produce the stems
> - [`feedback_1.md`](feedback_1.md) — design principles ("Anchored Corruption")
> - [`IMPLEMENTATION_REPORT_V1.md`](IMPLEMENTATION_REPORT_V1.md) — implementation log
> - [`related_work.md`](related_work.md) — prior art survey

---

## Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Module Glossary (Detailed)](#2-module-glossary-detailed)
3. [Macro-Net Stem Strategy](#3-macro-net-stem-strategy)
4. [Stem Interweaving (Phase 2)](#4-stem-interweaving-phase-2)
5. [Foundation Reinforcement (Phase 3)](#5-foundation-reinforcement-phase-3)
6. [End-to-End Diagram](#6-end-to-end-diagram)
7. [Numbers, Cycles, & Why](#7-numbers-cycles--why)
8. [Performance & Output](#8-performance--output)
9. [Failure Modes & Safeguards](#9-failure-modes--safeguards)
10. [How to Run](#10-how-to-run)

---

## 1. Design Philosophy

The core thesis of AudioArt is **"Anchored Corruption"**: neural audio models (RAVE, EnCodec, AudioLDM2…) are *misused* — driven away from the training manifold — but the output is held to musical territory by structural safeguards (sub-bass anchor, dry/wet mix, texture governor). The result is not noise. It is *the model's beautiful mistake*.

Meta-Symphony scales that thesis from a single MVP through four layers:

| Layer | Scope | Anchored by |
|---|---|---|
| **MVP** (A–I) | single damage operator on one chunk | dry/wet, soft limiter, texture governor |
| **Macro-Net** (Net 1 / 2 / 3 / Max / Dynamic) | multi-stage composition on a 10–60 s seed | per-stage dw, 80 Hz crossover, RMS match |
| **Meta-Symphony** | parallel macro-nets braided across 3 min | LFO crossfade, stereo drift, sub-bass re-injection |
| **Mastering** | final loudness shaping | tanh limiter (currently); LUFS pipeline available in `scripts/pyloudnorm_mastering.py` |

Every layer can fail independently without bringing the next one down. The result is a 3-min stereo composition that retains the seed's pitch identity at the bottom while the upper bands shimmer through completely re-imagined material.

---

## 2. Module Glossary (Detailed)

What each module *actually does* to a sine wave, with the algorithmic step that matters:

### MVP-A — Latent Perturbation (RAVE)

```
audio (48k, mono)
  → RAVE.encode  → z ∈ ℝ^(latent_dim=16, T_z)         (e.g. 93 frames per 4 s chunk)
  → z + noise_scale · n
      where n ∈ ℝ^(16, T_z), either:
        • white   : iid N(0,1)
        • smoothed: x_t = α·x_{t−1} + (1−α)·N(0,1),  α = 0.98  ← 1st-order IIR Brownian drift
  → optional dim_dropout: zero out fraction of rows in the time axis
  → optional dim_shuffle: permute latent_dim axis
  → RAVE.decode → audio
```

On a 440 Hz sine, `noise_scale=0.05, smoothed`: the tone develops slow timbral wobble (period ~0.5–2 s, set by α). `noise_scale=1.0`: the model forgets it's a sine and starts hallucinating.

### MVP-B — Caption Loop (Audio↔Text)

```
audio  → caption_fn(audio)  → text "a metallic drifting sound underwater"
      → mutation (prepend adjective from pool with prob p)
      → synth_fn(text)      → new audio
   (repeat N times, default depth=3)
```

Default **stub backend** uses spectral statistics → adjective bag for caption, and deterministic FM synthesis from text MD5 hash for synth. Set `caption.backend=qwen2_audio` + `tta.backend=audioldm2` + download the ~18 GB checkpoints for real semantic drift.

### MVP-C — Token Bending (EnCodec 24 kHz)

```
audio (48k) → resample(24k) → EnCodec.encode → tokens ∈ ℤ^(n_q=8, T_t)  (75 fps)
  bit_flip       : flip random bit of fraction `rate` of token positions
  quantizer_drop : zero out `rate` of entire quantizer rows
  shuffle        : reshuffle token positions within local window=W frames
  invalid_token  : replace `rate` of tokens with sentinel −1 (clamped to 0 at decode)
  (optional) quantizer_range: limit damage to slice (lo, hi); negative indices target upper layers
  → EnCodec.decode → resample(48k) → audio
```

On 440 Hz sine, `bit_flip rate=0.03 q_range=(-3,0)` (upper 3 quantizers only): main partial preserved, but ±2–4 Hz sidebands appear from quantization error redistribution. `rate=0.10`: harsh codec-rip texture.

### MVP-D — Checkpoint Morphing (RAVE × 2)

```
load   guitar.state_dict,  organ.state_dict
optional Re-Basin alignment (partial=inner-block / full=encoder-chain)
merged = (1−t)·guitar  +  t·organ            (linear)
       | slerp(guitar, organ, t)             (angular)
       | guitar + ε·N(0,1)                   (random_walk)
load merged into RAVE container → encode/decode normal
```

Independently-trained guitar+organ collapse at intermediate `t` — see [`feedback_1.md`](feedback_1.md) §5 + endpoint cliff sweep in `experiments/mvp_d_ckpt_morph/results.json`. Production usage stays at `t ∈ [0, 0.02]` (guitar-side fade) or `t ∈ [0.98, 1]` (organ-side fade). Meta-Symphony's Net Max & Net Dynamic both use `t=0.005` for the morph-guitar handle and `t=0.995` for the morph-organ handle.

### MVP-E — Neural Granular (RAVE + memory buffer)

```
LatentMemory: circular buffer of last `memory_size` (up to 4096) latent frames
each forward chunk:
  z = encode(audio)
  push z into memory
  for i in num_grains:
      pick random window of `grain_size` frames from memory
      window-fade and add into output position
  z_out = mix·grains + (1−mix)·z
  decode(z_out)
```

Produces "smeared-time chorus": the model hears the present plus 4 echoes of its own past. Stable on RAVE because latent is continuous.

### MVP-F — Spectral Frozen (RAVE upper dims)

```
let n_freeze = round(latent_dim · auto_upper_fraction)  (e.g. 8 of 16)
let indices = top n_freeze dims of z (high-frequency latent)
cached_state ← z[indices, 0:1]            initial freeze sample
on each frame:
    if global_frame_idx % update_interval == 0:
        target_state ← current z[indices, ti]
        fade_ptr ← crossfade_frames
    if fade_ptr > 0:
        α = fade_ptr / crossfade_frames
        z[indices, ti] = α·cached_state + (1−α)·target_state
        fade_ptr -= 1
    else:
        z[indices, ti] = cached_state
```

Periodic snapshot + crossfade = no audible "beep" at update boundary, but the timbre slowly drifts. On 440 Hz sine: gives a shimmer halo around the fundamental.

### MVP-G — Latent Feedback (RAVE delay line)

```
buffer: circular ℝ^(latent_dim, max_delay)
on each latent frame:
    delayed   = buffer[:, ptr]
    processed = z_curr + delayed · feedback        ← recursive growth
    buffer[:, ptr] = processed
    out_z = (1-mix)·z_curr + mix·processed
    ptr = (ptr+1) % max_delay
```

`delay_frames=32, feedback=0.40`: noticeable echo that builds over time. `feedback=0.55`: latent space saturates → the model starts re-interpreting itself in a strange feedback resonance.

### MVP-H — Codebook Organ (EnCodec generative, no input)

```
generate tokens ∈ ℤ^(n_q, duration_frames):
  prime      : pattern[t] = t-th prime % 1024,  output[q, t] = roll(pattern, q·stride)
  fibonacci  : pattern[t] = fib[t] % 1024,      same roll
  random_walk: cumsum of integers in [-5,5] mod 1024 per quantizer
EnCodec.decode(tokens) → audio
```

Pure synthesis: no audio input required. Produces non-musical chordless harmonic series (primes give wider intervals, fibonacci tighter chromatic clusters). Used as a drone bed.

### MVP-I — Bass Massive (EnCodec lower quantizers)

```
tokens (n_q=8, T):
  Temporal smearing : np.roll(tokens[lo:hi], smear_delay, axis=time)   on lower q only
  Codebook jitter   : random ±1 to fraction `jitter_rate` of tokens in lower q
  Quantizer folding : with prob `fold_leak_rate`, copy a value from upper q range to lower
```

Targets quantizers (0, 1, 2) where the coarse audio structure lives. Smear delay of 12 frames @ 75 fps ≈ 160 ms group lag in the bass → "deep sea" wobble. Jitter adds digital saturation.

---

## 3. Macro-Net Stem Strategy

Meta-Symphony does **not** simply mix all 4 nets in parallel. It serialises some nets so that each subsequent net operates on a *neurally pre-processed* version of the seed.

```
seed (3 min sub-bass)
    │
    ├── Net 1 (parallel mix) ──────────────────────────────→ stem_N1
    │                                                           │
    ├── input_N3 = RMS_match(seed, stem_N1) ─→ Net 3 ──────→ stem_N3   (compounded destruction)
    │
    ├── Net 2 (3-pass recursive on raw seed) ──────────────→ stem_N2
    │                                                           │
    └── input_Dyn = RMS_match(seed, stem_N2) ─→ Net Dynamic ──→ stem_Dyn (storm fed by loops)
```

Two reasons for this dependency chain:

1. **Net 3 is a destructive linear chain.** Feeding it Net 1's already-textured output instead of the raw sine produces a *deeper* version of the same composition rather than two independent damages of the seed.
2. **Net Dynamic's silence drops + filter sweeps work better on already-loop-saturated material** than on a pure sine. Net 2 supplies that material.

This gives **two parallel "stems"** that share a lineage (sine → spatial → destruction, and sine → recursion → storm) and can be cross-faded.

---

## 4. Stem Interweaving (Phase 2)

Two LFO-driven mixes + two stereo pan LFOs:

```
t ∈ [0, 180 s]

lfo_A(t) = ½ + ½·sin(2π · t / 60)        ← Pair A cycle: 60 s
lfo_B(t) = ½ + ½·cos(2π · t / 45)        ← Pair B cycle: 45 s

pan_A(t) = 0.7 · sin(2π · t / 20)        ← Pair A stereo drift: 20 s
pan_B(t) = 0.7 · cos(2π · t / 25)        ← Pair B stereo drift: 25 s

mix_A    = lfo_A · stem_N1 + (1 − lfo_A) · stem_N3
mix_B    = lfo_B · stem_N2 + (1 − lfo_B) · stem_Dyn

apply equal-power pan:
    L_pan = cos((pan + 1)·π/4)
    R_pan = sin((pan + 1)·π/4)

stereo_A = (mix_A·L_panA, mix_A·R_panA)
stereo_B = (mix_B·L_panB, mix_B·R_panB)

master = stereo_A + stereo_B              ← 2-channel result
```

### Frequency / cycle choices

- **60 vs 45 s** Pair-A and Pair-B: incommensurate ratio → 9-minute beat pattern (LCM of 60 and 45 is 180 s exactly = full song length, so the song spans one full beat cycle).
- **20 vs 25 s** stereo pan: again incommensurate, 100 s LCM beat — moves twice within the 3-min frame.
- `sin` for A, `cos` for B → 90° phase offset between mixing and panning, so the centroid of energy moves diagonally through the stereo field rather than oscillating only L/R.

---

## 5. Foundation Reinforcement (Phase 3)

After interweaving, the original seed is re-injected at the sub-bass:

```
seed_trim                                        ← original 3-min sub-bass
   → Butterworth 2nd-order lowpass @ 100 Hz
   → × 10^(8/20) ≈ ×2.51                         ← +8 dB sub-boost
   → broadcast to both stereo channels
   → sum into master                             ← anchors fundamental

master = stereo_AB_mix + (sub_L, sub_R)

final  = tanh(master · 1.25) / tanh(1.25)        ← soft limiter, drive = 1.25
       → normalise to peak 0.95
```

Why 2nd-order @ 100 Hz: gives 12 dB/oct rolloff with phase coherent enough to sit cleanly behind the model-damaged upper bands. Higher orders would phase-shift the kick attacks too far.

Why +8 dB: empirically restores the perceptual loudness of the dry sub after limiter compression of the upper bands. Matches the operational settings used in `scripts/final_bass_pro_master.py`.

---

## 6. End-to-End Diagram

```text
┌────────────────────────── PHASE 1: STEM GENERATION (mono) ─────────────────────────┐
│                                                                                    │
│  generate_seed(180 s, 48 kHz)                                                      │
│    sub        = 0.6·sin(2π·32.7·t) + 0.3·sin(2π·41.2·t) + 0.2·sin(2π·55·t)         │
│    fm         = sin(2π·(55 + 5·sin(2π·0.1·t))·t)                                   │
│    lfo        = 0.6 + 0.4·sin(2π·t/30)                                             │
│    sig        = (sub + 0.15·fm) · lfo,   normalised to 0.8                         │
│                                                                                    │
│  seed ──────────┬──────────────────────────────────────────────────────┐           │
│                 │                                                      │           │
│                 ▼                                                      ▼           │
│           ┌─────────────────────┐                              ┌─────────────────┐ │
│           │ Net 1 — Cathedral   │                              │ Net 2 — Organ   │ │
│           │ (parallel 5-bus)    │                              │ (3-pass loop)   │ │
│           └──────────┬──────────┘                              └────────┬────────┘ │
│                      │ stem_N1                                          │ stem_N2  │
│      RMS_match(seed, stem_N1)                          RMS_match(seed, stem_N2)    │
│                      │                                          │                  │
│                      ▼                                          ▼                  │
│           ┌─────────────────────┐                              ┌─────────────────┐ │
│           │ Net 3 — Decoding    │                              │ Net Dyn Tempest │ │
│           │ (linear 9-stage)    │                              │ (time-varying)  │ │
│           └──────────┬──────────┘                              └────────┬────────┘ │
│                      │ stem_N3                                          │ stem_Dyn │
└──────────────────────┼──────────────────────────────────────────────────┼──────────┘
                       │                                                  │
┌──────────────────────┼──── PHASE 2: INTERWEAVE (stereo) ─────────────────┼──────────┐
│                      │                                                  │           │
│  lfo_A = ½+½·sin(2π·t/60)              lfo_B = ½+½·cos(2π·t/45)                     │
│  pan_A = 0.7·sin(2π·t/20)              pan_B = 0.7·cos(2π·t/25)                     │
│                                                                                    │
│  mix_A = lfo_A·stem_N1 + (1−lfo_A)·stem_N3                                         │
│  mix_B = lfo_B·stem_N2 + (1−lfo_B)·stem_Dyn                                        │
│                                                                                    │
│  stereo_A = equal_power_pan(mix_A, pan_A)                                          │
│  stereo_B = equal_power_pan(mix_B, pan_B)                                          │
│                                                                                    │
│  master   = stereo_A + stereo_B                                                    │
└─────────────────────────────────────────────────────────────────────────┬──────────┘
                                                                          │
┌──── PHASE 3: FOUNDATION & MASTER ───────────────────────────────────────┴──────────┐
│                                                                                    │
│  sub_only = lowpass_butter_2nd(seed_trim, 100 Hz) · 10^(8/20)                      │
│  master[:,0] += sub_only ;  master[:,1] += sub_only                                │
│                                                                                    │
│  final = tanh(master · 1.25) / tanh(1.25)                                          │
│        = normalise to peak 0.95                                                    │
│                                                                                    │
│  → runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL.wav  (180 s, stereo, 48 kHz) │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Numbers, Cycles, & Why

### Seed composition

| Component | Frequency | Amplitude | Reason |
|---|---|---|---|
| Sub A | 32.7 Hz (C1) | 0.60 | physical sub-bass register |
| Sub B | 41.2 Hz (E1) | 0.30 | minor 3rd above for harmonic interest |
| Sub C | 55.0 Hz (A1) | 0.20 | perfect 5th of A0, anchors tonality |
| FM tone | 55 Hz ± 5 Hz @ 0.1 Hz | 0.15 | gives the model something tonal to "interpret" |
| Breathing LFO | 1/30 Hz (30 s period) | env 0.6–1.0 | gives the macro composition a slow breath |

### LFO cycle choices

| Cycle | Frequency | Affects | Why |
|---|---|---|---|
| 60 s | 1/60 Hz | Pair-A mix (N1↔N3) | 1/3 of song length → 3 full cycles |
| 45 s | 1/45 Hz | Pair-B mix (N2↔Dyn) | 1/4 of song length → 4 full cycles |
| 20 s | 1/20 Hz | Pair-A stereo pan | fast enough to feel motion, slow enough to be coherent |
| 25 s | 1/25 Hz | Pair-B stereo pan | offset from 20 s → constantly moving cross-pattern |

LCM(60, 45) = 180 s = song length: Pair A and Pair B come back into perfect phase exactly at the end.

LCM(20, 25) = 100 s: stereo image traces a non-repeating Lissajous figure twice within the song.

### Phase 3 anchors

| Setting | Value | Reason |
|---|---|---|
| Sub-cutoff | 100 Hz | preserves perceived bass impact while staying below mid energy |
| Sub-boost | +8 dB | restores loudness after limiter ducks the mid/high bands |
| Limiter drive | 1.25 | gentle saturation, ~3% THD at peak |
| Final peak | 0.95 | leaves 0.5 dB true-peak headroom |

---

## 8. Performance & Output

### Wall-time estimate (V100, single GPU)

| Phase | Operation | Estimated time |
|---|---|---|
| 1.1 | Net 1 on 180 s seed | ≈ 30 s |
| 1.2 | Net 3 on 180 s seed | ≈ 35 s |
| 1.3 | Net 2 on 180 s seed (2 passes) | ≈ 50 s |
| 1.4 | Net Dynamic on 180 s seed | ≈ 60 s |
| 2 | LFO crossfade + pan | < 1 s |
| 3 | Foundation + limiter | < 1 s |
| **Total** | | **~3 min** |

Active background renders may be longer if the GPU is shared.

### Final output

```
runs/masterpiece/meta_symphony/
├── seed_sub_180s.wav         (17 MB, mono, 48k)
└── META_SYMPHONY_FINAL.wav   (~35 MB, stereo, 48k, 180 s, peak 0.95)
```

Optionally also written: `stem_N1.wav`, `stem_N2.wav`, `stem_N3.wav`, `stem_Dyn.wav` for stem-level audit (controlled by future flag — currently in-memory only).

---

## 9. Failure Modes & Safeguards

| Failure | Where | Safeguard |
|---|---|---|
| **NaN explosion** inside RAVE/EnCodec chain | any stage, especially Net 3 chain depth ≥ 7 | Texture Governor sets `wet = 0.0` on detection. Verified mitigated at 2:08 mark during long-form renders (`docs/IMPLEMENTATION_REPORT_V1.md` §4.2). |
| **Silent collapse** at MVP-D intermediate `t` | Net Max δ bus, Net Dynamic δ bus | Production `t` clamped to endpoint regions (0.005, 0.995). |
| **Clipping** at master sum | Phase 2 mix + Phase 3 sub-boost | `tanh` soft limiter @ drive 1.25, then peak-normalise to 0.95. |
| **Pure noise drift** in heavy chains | Net Max β/ε, Net Dynamic ε | Each per-stage `wet ≤ 0.55–0.65` keeps signal close to manifold. |
| **OOM** on >5 min material | Phase 3 single-tensor master | For >3 min, switch to SoX streaming via `scripts/run_hfo_master_sox.sh`. |
| **Resample artefacts** at 48k↔24k crossings | every C/H/I stage | Linear interp with 4-block windows acceptable for sound-art (band-limited input). |

---

## 10. How to Run

```bash
# Single-shot meta-symphony render
conda activate audioart
cd /home1/irteam/proj/AudioArt
PYTHONPATH=. python scripts/meta_symphony.py

# Background with logging + PID tracking (per WORKFLOW_HISTORY pattern)
nohup python scripts/meta_symphony.py > runs/masterpiece/meta_symphony.log 2>&1 &
echo $! > runs/masterpiece/meta_symphony_pid.txt

# Watch live
tail -f runs/masterpiece/meta_symphony.log
```

Once `META_SYMPHONY_FINAL.wav` lands you can:

```bash
# Loudness mastering to −12 LUFS via pyloudnorm
python scripts/pyloudnorm_mastering.py \
    --in  runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL.wav \
    --out runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL_LUFS12.wav
```

For per-stem inspection, swap in a debugging block:

```python
# Inside run_meta() before Phase 2
sf.write(out_dir / "stem_N1.wav", stem_n1, sr)
sf.write(out_dir / "stem_N3.wav", stem_n3, sr)
sf.write(out_dir / "stem_N2.wav", stem_n2, sr)
sf.write(out_dir / "stem_Dyn.wav", stem_dyn, sr)
```

This makes each macro-net's contribution auditable independently.
