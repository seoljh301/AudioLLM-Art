# MVP-A — RAVE Latent Perturbation

## Hypothesis
Perturbing RAVE's latent `z` (Gaussian noise, dim dropout, dim shuffle, additive bias) yields aesthetically useful timbral drift while preserving instrument identity — until perturbation magnitude crosses a threshold and the model starts hallucinating.

## Model dependencies
- A pretrained traced RAVE model (`.ts`) placed at `checkpoints/rave/<name>.ts`.
- Download from the official RAVE model zoo or train via the IRCAM RAVE repo.

## Expected sonic output
- Low `noise_scale` (~0.05): subtle timbral wobble.
- Medium (~0.3): warped texture, partials shift.
- High (>0.8): pseudo-instrument babble, the model "forgets" what it is.

## Parameters to sweep
- `noise_scale` ∈ [0, 1]
- `dim_dropout` ∈ [0, 0.5]
- `dim_shuffle` ∈ {false, true}
- `freeze_mask` — which latent dims are pinned

## OSC interface
| Address | Args | Effect |
|---------|------|--------|
| `/mvp_a/noise` | float | set noise_scale |
| `/mvp_a/dropout` | float | set dim_dropout |

## Run

### 0. Get a model
For verification only, build a stub:
```bash
python scripts/make_stub_rave.py --out checkpoints/rave/stub.ts
```
For real sonic results, download a pretrained RAVE TorchScript export, e.g. from
[`acids-ircam/rave-models`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)
(`voice_jvs_b2048_r44100_z16.ts`, `guitar_iil_b2048_r48000_z16.ts`, etc.).
Point `model.path` in `config.yaml` at the downloaded file.

> Loading `.ts` files executes arbitrary TorchScript code — only load checkpoints from sources you trust.

### 1. Offline render
```bash
bash run.sh --mode render --input path/to/in.wav --output path/to/out.wav
```

### 2. OSC server (Max sends `/mvp_a/render <in> <out>`)
```bash
bash run.sh --mode serve
```

## OSC interface
| Address | Args | Effect |
|---------|------|--------|
| `/mvp_a/noise` | float | set noise_scale |
| `/mvp_a/dropout` | float | set dim_dropout |
| `/mvp_a/shuffle` | int 0/1 | toggle dim_shuffle |
| `/mvp_a/render` | string in, string out | trigger render, replies `/mvp_a/done <out>` |

## Hardware notes
- V100 GPU supported via pinned `torch==2.4.1 + cu121` (see root `environment.yaml`). cu130 wheels require SM ≥ 7.5 and silently fail on V100.
- Real RAVE on V100: ~3× realtime (5 s audio in ~1.5–2 s wall time).
