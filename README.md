# AudioArt

Sound-art prototyping around the **misuse** of Audio Foundation Models.

> Not how AI successfully understands sound, but how sound transforms when AI fails to understand it.

See [`audio_foundation_model_sound_art_ideation.md`](audio_foundation_model_sound_art_ideation.md) for the design ideation, and [`docs/related_work.md`](docs/related_work.md) for prior art / context.

---

## TL;DR

- 4 MVPs wired end-to-end, each with `--mode render` (offline) and `--mode serve` (OSC).
- 3 of 4 (A, C, D) run with real pretrained models on V100 GPU; MVP-B uses local stubs until you authorize the ~18 GB Qwen2-Audio + AudioLDM2 downloads.
- All MVPs share a single Python core (`src/core/`) for OSC, audio I/O, and model registry.

---

## MVP Matrix

| MVP | Failure mode | Stack | Status | OSC ports |
|-----|--------------|-------|--------|-----------|
| **A** | Latent perturbation (noise, dropout, shuffle, bias, freeze) | RAVE TorchScript (`.ts`) | Real model (guitar) on V100 | 7400 / 7401 |
| **B** | Audio ↔ caption ↔ TTA recursive loop | Qwen2-Audio + AudioLDM2 *(real)* / FM stub *(default)* | Stub default; real backends gated | 7410 / 7411 |
| **C** | Codec token bending (bit-flip, quantizer-drop, shuffle, invalid-token) | EnCodec 24 kHz (real) | Real codec on V100 | 7420 / 7421 |
| **D** | Checkpoint morphing (linear, slerp, random walk) | RAVE × N TorchScripts | Real pair (guitar↔organ) on V100 — collapse documented | 7430 / 7431 |

Quick reference on the failure mode of each MVP and what its `results.json` shows:

- **A** — noise_scale monotonically increases `out_rms`; high noise destroys instrument identity. Sonic drift before semantic collapse.
- **B** — caption text traverses an adjective vocabulary deterministically over loop depth; mutation pool prepends adjectives to caused drift.
- **C** — bit_flip introduces sidebands near partials; quantizer_drop introduces extra partials; shuffle stutters; invalid_token punctures low-freq artifacts. All preserve dominant input partials at rate ≤ 0.1.
- **D** — naive weight-space interp between independently-trained RAVE checkpoints **collapses to near-silence** at any intermediate `t`. This is a known property of mode-connectivity in neural nets, and a usable "hollowed-out" sound art preset. A partial Git Re-Basin (inner-resblock channel alignment, `src/modules/mvp_d/re_basin.py`) was tried and is NOT sufficient on its own — full alignment would also need to permute encoder/decoder boundary convs, gimbal, prior_net, and the latent_pca rotation.

---

## Repository layout

```
src/core/            shared infra: OSCBridge, RingBuffer, ModelRegistry
src/modules/mvp_a/   latent perturbation primitives + RAVE I/O + render
src/modules/mvp_b/   caption loop + stub/real backend factories
src/modules/mvp_c/   token bending + codec I/O + render
src/modules/mvp_d/   state_dict interp + morph I/O (reuses MVP-A render)

experiments/mvp_*/   per-MVP config.yaml + main.py + run.sh + README.md + results.json

scripts/             one-off utilities (stub RAVE builder, etc.)
checkpoints/         RAVE .ts files etc. — gitignored
data/                read-only audio — gitignored
runs/                generated audio + transcripts — gitignored
max/                 Max/MSP patches (placeholder)
docs/                related-work survey + design notes
```

---

## Setup

```bash
conda env create -f environment.yaml
conda activate audioart
```

Environment pins `torch==2.4.1 + cu121` and `torchaudio==2.4.1 + cu121`. This is the latest pair supporting **V100 SM 7.0**. Newer wheels (cu130) require SM ≥ 7.5 and silently fail.

GPU verification:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True Tesla V100-PCIE-32GB
```

The Max/MSP side requires `nn~` from ACIDS-IRCAM (installed manually): https://github.com/acids-ircam/nn_tilde

---

## Quick start

### Synthesize a 5-second test signal
```bash
python - <<'PY'
import numpy as np, soundfile as sf, os
sr = 48000
t = np.linspace(0, 5.0, sr*5, endpoint=False)
sig = (0.4*np.sin(2*np.pi*220*t) + 0.3*np.sin(2*np.pi*440*t)
       + 0.2*np.sin(2*np.pi*880*t)*np.sin(2*np.pi*3*t)
       + 0.05*np.random.default_rng(0).standard_normal(len(t)))
sig = (sig/np.max(np.abs(sig))*0.9).astype('float32')
os.makedirs('runs', exist_ok=True)
sf.write('runs/seed_48k.wav', sig, sr)
PY
```

### MVP-A — RAVE latent perturb (real guitar checkpoint)
```bash
bash experiments/mvp_a_rave_latent/run.sh \
  --mode render --input runs/seed_48k.wav --output runs/mvp_a_out.wav
```
Edit `experiments/mvp_a_rave_latent/config.yaml` `perturb.noise_scale` to vary intensity. See per-MVP README for the full OSC schema.

### MVP-C — EnCodec token bending
```bash
bash experiments/mvp_c_encodec_bend/run.sh \
  --mode render --input runs/sine_test_in.wav --output runs/mvp_c_out.wav
# requires 24kHz mono input; the seed above is 48k — use a 24k seed or let load_audio_mono resample
```

### MVP-D — guitar ↔ organ morph
```bash
bash experiments/mvp_d_ckpt_morph/run.sh \
  --mode render --input runs/seed_48k.wav --output runs/mvp_d_out.wav
```
Sweep `morph.t` 0 → 1; intermediate values **collapse** by design (see results.json).

### MVP-B — recursive caption ↔ TTA loop (stub)
```bash
bash experiments/mvp_b_caption_loop/run.sh \
  --mode render --input runs/seed_16k.wav --output runs/mvp_b_loop
# emits step_NN.wav + transcript.json into the output directory
```

### Launch any MVP as an OSC server
```bash
bash experiments/mvp_a_rave_latent/run.sh --mode serve
```
Then drive from Max/MSP — OSC ports listed in the matrix above.

---

## Getting real model checkpoints

The repo ships with the wiring; weights are not redistributed. Recommended sources:

| Model | Source | Notes |
|-------|--------|-------|
| RAVE 48kHz z=16 (guitar, organ, voice) | [`Intelligent-Instruments-Lab/rave-models`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models) | All `*_b2048_r48000_z16.ts` share architecture → MVP-D pairs |
| EnCodec 24kHz | `encodec` pip package (Meta) | Auto-downloaded by EnCodec on first use |
| DAC 44.1kHz | `descript-audio-codec` pip package | Auto-downloaded |
| Qwen2-Audio-7B-Instruct | HF `Qwen/Qwen2-Audio-7B-Instruct` | ~14 GB — set `caption.backend: qwen2_audio` |
| AudioLDM2 | HF `cvssp/audioldm2` | ~4 GB — set `tta.backend: audioldm2` |

> Loading a `.ts` TorchScript file executes arbitrary code. Only use checkpoints from sources you trust.

`scripts/make_stub_rave.py` builds local untrained RAVE-API stubs for verification without downloads.

---

## Design rationale

The system treats AudioLLMs / codecs as:
- **unstable listening machines** — what does a "guitar-only" RAVE do to a sine sweep?
- **hallucinating acoustic interpreters** — captioners produce text that may have no truth value, but is generative material.
- **neural texture synthesizers** — codec round-trip with deliberate corruption is granular synthesis on a learned codebook.
- **semantic distortion devices** — recursive audio↔text loops "translate" through language, accumulating bias.
- **recursive sonic organisms** — feedback structures that drift on their own.

Each MVP isolates one of these stances so they can be combined (e.g. MVP-D morph → MVP-A latent perturb → MVP-C token bend pipeline). See per-MVP `README.md` and `results.json` for the parameters and observed behaviors.

---

## Caveats

- **V100 SM 7.0** is at the edge of supported torch — pinned at `2.4.1 + cu121`. Don't bump without re-checking.
- **MVP-D weight interp collapses** on independently-trained RAVE pairs — this is mode-connectivity, not a bug. Treat the collapse as a preset, or implement Git Re-Basin / permutation alignment for non-trivial morph.
- **Auto-classifier blocks agent-chosen large model downloads**. Real Qwen2-Audio / AudioLDM2 must be downloaded by the user.

---

## Acknowledgements

This repo is glue. The heavy lifting is done by:

- **RAVE / nn_tilde** — Caillon, Esling, ACIDS-IRCAM.
- **EnCodec** — Défossez et al., Meta AI.
- **Descript Audio Codec** — Descript.
- **AudioLDM2** — Liu et al., CVSSP / Surrey.
- **Qwen2-Audio** — Alibaba.

Pretrained RAVE models in this repo's verification runs come from [`Intelligent-Instruments-Lab`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models) — University of Iceland.
