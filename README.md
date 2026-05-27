# AudioArt

Neural sound-art prototyping around the **misuse** of Audio Foundation Models.

> Not how AI successfully understands sound, but how sound transforms when AI fails to understand it.

See [`audio_foundation_model_sound_art_ideation.md`](audio_foundation_model_sound_art_ideation.md) for the original design philosophy, and [`docs/META_SYMPHONY_ARCHITECTURE.md`](docs/META_SYMPHONY_ARCHITECTURE.md) for a comprehensive guide to the final Meta-Symphony.

---

## 🎹 Quick Links

- **🎧 Demo Page**: [`demo.html`](demo.html) — 52-track listening index with inline audio + waveform thumbnails (open via `file://` or `python -m http.server`)
- **Final Results**: `runs/masterpiece/`
- **Architecture Visualization**: [`docs/MULTINET_ARCHITECTURE.md`](docs/MULTINET_ARCHITECTURE.md)
- **Implementation Log**: [`docs/IMPLEMENTATION_REPORT_V1.md`](docs/IMPLEMENTATION_REPORT_V1.md)
- **Workflow History**: [`docs/WORKFLOW_HISTORY.md`](docs/WORKFLOW_HISTORY.md)

---

## 🧩 The MVP Matrix (9 Pillars of Destruction)

AudioArt is built on 9 foundational modules, each exploiting a different neural failure mode.

| MVP | Domain | Backbone | Failure Mode / Sonic Signature |
|:---:|:---|:---|:---|
| **A** | Latent | RAVE | **Perturbation**: Brownian noise injection into Z-space. Organic wobbles. |
| **B** | Semantic | AudioLDM2 | **Caption Loop**: Recursive audio ↔ text translation. Logic drift. |
| **C** | Codec | EnCodec | **Token Bending**: Bit-flipping and bitstream scrambling. Digital grit. |
| **D** | Weight | RAVE | **Morphing**: Weight-space interpolation between ckpts. Ghostly collapse. |
| **E** | Memory | RAVE | **Neural Granular**: Buffer-based latent grain injection. Time-smearing. |
| **F** | Spectral | RAVE | **Spectral Freeze**: High-freq latent snapshots + crossfade. Aurora shimmer. |
| **G** | Recursive | RAVE | **Latent Feedback**: Z-space delay lines. Self-evolving neural echoes. |
| **H** | Generative | EnCodec | **Codebook Organ**: Prime/Fibonacci token synthesis. Abstract drones. |
| **I** | Bass | EnCodec | **Bass Massive**: Quantizer folding and jitter. Seismic sub-bass smear. |

---

## 🌊 Composition Tools

Beyond single modules, AudioArt provides macro-scale tools for complex composition:

### 1. Multinet (`scripts/multinet.py`)
Wires MVPs into complex topologies.
- **Net 1 (Cathedral)**: 5-bus parallel spatial mix.
- **Net 2 (Organ)**: Recursive macro-feedback loops (3-pass).
- **Net 3 (Chamber)**: 9-stage linear chain of accumulated damage.
- **Net Dynamic (Tempest)**: 60s timeline with volume envelopes and impulse events.

### 2. Meta-Symphony (`scripts/meta_symphony.py`)
The "Network of Networks". Braids Net 1, 2, 3, and Dynamic across a 3-minute timeline using slow-moving LFOs, stereo drift, and +8dB sub-bass anchoring.

### 3. Massive Scale Rendering (SoX)
For 1-hour symphonies (like `Ulaanbaatar Epic`), Python memory becomes a bottleneck. We use SoX-based streaming pipelines (`scripts/run_hfo_master_sox.sh`) to perform resampling, filtering, and mixing directly on disk, bypassing OOM issues.

---

## 🛡️ Safety: The Texture Guard

To prevent neural collapse (NaNs, pure white noise, or silence), every module runs through the **Texture Governor**:
- **NaN Hardening**: Automatic fallback to 0.0 wet mix if numerical errors are detected.
- **Spectral Flattening Guard**: Damps damage if the sound becomes too "noisy" (Flatness > 0.55).
- **RMS Protection**: Ensures signal presence between 1e-4 and 0.85 RMS.

---

## 🛠️ Setup

```bash
conda env create -f environment.yaml
conda activate audioart
# Install SoX/FFmpeg for large-scale rendering
conda install -y -c conda-forge ffmpeg sox
```

### Running the Meta-Symphony
```bash
PYTHONPATH=. python scripts/meta_symphony.py
```

---

##  Repository layout

```
src/core/            Shared infra: Texture Governor, Mix Engine, Metrics
src/modules/mvp_*/   Individual modules (A through I)
experiments/         Per-module sandboxes and results.json
scripts/             Mastering, Composition (Multinet, Meta-Symphony), SoX wrappers
docs/                Comprehensive architecture and workflow documentation
runs/                Generated masterpieces (wav, csv, logs)
checkpoints/         Model weights (RAVE .ts, EnCodec)
```

---

## ✒️ Design Rationale

The system treats AudioLLMs as **unstable listening machines**. By anchoring the "corruption" with clean sub-bass crossovers and managing energy via the Texture Governor, we transform erratic model behaviors into a controlled, beautiful, and highly complex form of **Neural Sound Art**.
