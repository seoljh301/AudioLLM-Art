# MVP-D — Checkpoint Morphing (with optional MVP-A perturb chain)

## Hypothesis
A checkpoint is a "frozen auditory interpretation of the world." Interpolating between two checkpoints trained on disjoint corpora (e.g. percussion vs. voice) produces a single model that hears the world as a hybrid — neither instrument, both at once.

## Model dependencies
- ≥2 RAVE `.pt` or `.ts` checkpoints of identical architecture.
- (Optional) DAC checkpoints for interp at codec level.

## Expected sonic output
- `linear` mode, t∈[0,1]: smooth crossfade between the two timbral worlds.
- `slerp` mode: arc through latent geometry; midpoint often sounds more "alive."
- `random_walk`: stochastic morph; produces drifting impossible-instrument variations.

## Parameters to sweep
- `morph.t` ∈ [0, 1] (live)
- `morph.mode` ∈ {linear, slerp, random_walk}
- `morph.walk_step` ∈ [0.01, 0.2]

## OSC interface
| Address | Args | Effect |
|---------|------|--------|
| `/mvp_d/t` | float | morph coefficient |

## Run

### 0. Checkpoints
≥2 RAVE TorchScript exports of matching architecture (same `latent_dim`, `sample_rate`, `hop`).

Stubs for verification:
```bash
python scripts/make_stub_rave.py --out checkpoints/rave/stub_a.ts --seed 0
python scripts/make_stub_rave.py --out checkpoints/rave/stub_b.ts --seed 1
```

For real morphing, download two matching exports (e.g. both `b2048_r48000_z16`) from
[`acids-ircam/rave-models`](https://huggingface.co/Intelligent-Instruments-Lab/rave-models).
Point `checkpoints.paths` in `config.yaml`.

> `.ts` loading executes TorchScript — only use checkpoints from sources you trust.

### A+D chain
Add a `perturb:` block to `config.yaml` (same schema as MVP-A) to layer latent perturbation on top of the morphed model:
```yaml
morph: {mode: linear, t: 0.005}
perturb: {noise_scale: 0.5, dim_dropout: 0.3}
```
See `ad_chain_sweep` and `dropout_intensity_sweep_bilateral` in `results.json`. Bilateral 7×6 dropout sweep shows:
- Guitar side: monotonic amplification up to d ≈ 0.3–0.5 (1.9× base rms), collapse at d=0.7.
- Organ side: small bump at d ≈ 0.1–0.2, distinct dip at d=0.3, partial recovery at d=0.5, collapse at d=0.7.

Different latent geometries: guitar has many inhibitory dims (drop more → others fire louder); organ has a few critical harmonic-defining dims clustered around d≈0.3. Perturbation cannot rescue the silent plateau at intermediate t.

### 1. Offline render (single morph snapshot)
```bash
bash run.sh --mode render --input path/to/in.wav --output path/to/out.wav
```

### 2. OSC server (live remorph)
```bash
bash run.sh --mode serve
```

## OSC interface
| Address | Args | Effect |
|---------|------|--------|
| `/mvp_d/t` | float | morph coefficient; triggers state_dict remerge |
| `/mvp_d/mode` | string | switch between linear / slerp / random_walk |
| `/mvp_d/render` | string in, string out | render and emit `/mvp_d/done <out>` |

## Hardware notes
- V100 GPU supported via pinned `torch==2.4.1 + cu121` (root `environment.yaml`).
- Remorph is fast (state_dict size ≈ model size; one tensor op per key); render dominates total time.

## Known failure mode

Naive linear interpolation between two **independently-trained** RAVE checkpoints (guitar + organ in our test) **collapses to near-silence** at any intermediate `t` (rms ≈ 0.0005 vs 0.4 at endpoints). Endpoints `t=0` and `t=1` produce the original models cleanly; anything in-between hollows the model out.

This is a known property of neural-net mode connectivity — independently-trained nets do not lie on a low-loss curve in weight space. Workarounds:

1. **Use it as a preset.** "Hollowed-out RAVE" is a usable sound art texture in its own right.
2. **Train two RAVE checkpoints from a shared initialization.** Should morph cleanly.
3. **Git Re-Basin / permutation alignment.** Re-permute one model's hidden units to align with the other before interp. Two implementations attempted:
   - **Partial** (`src/modules/mvp_d/re_basin.py`, `rebasin_mode="partial"` — default): aligns inner channels of each paired resblock (21 blocks, grouped-conv aware). LAP on `.1.weight` outputs + `.3.weight` inputs. Did not prevent collapse.
   - **Full encoder** (`src/modules/mvp_d/re_basin_full.py`, `rebasin_mode="full"`): adds chain classes `enc_48 / 96 / 192 / 384` plus inner classes, with tied-group semantics for upstream channels feeding grouped convs (`.5`, `.7`, `.16`). 5-iter weight matching (Ainsworth 2023 Algorithm 3). Random-perm verification confirms topology preserves function. But **morph still collapses** at intermediate `t` — encoder alignment alone is insufficient because the LATENT output is NOT permuted (to keep the latent space consistent for downstream consumers) and the COLLAPSE is downstream. Decoder weights are naively blended across independently-trained instruments → silence. Full fix needs decoder + gimbal + prior_net + latent_pca chains too. See `rebasin_full_attempt` in `results.json` for the side-by-side data.
4. **Stay near endpoints.** Confirmed via fine sweep — `t ∈ [0, 0.02]` and `t ∈ [0.98, 1.0]` produce a clean "fading instrument" musical region before each cliff. Silent plateau covers `t ∈ [0.10, 0.90]`. See `cliff_sweep_guitar_organ` in `results.json` and the 19 rendered files in `runs/mvp_d/cliff/`.

The stub checkpoint pair (`stub_a.ts` + `stub_b.ts`, both random init) **does** produce monotonic linear morph — because they share architecture and are random in compatible ways. Real trained nets occupy distinct basins.
