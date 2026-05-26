# MVP-B — Caption ↔ Text-to-Audio Recursive Loop

## Hypothesis
Iterating `audio -> caption -> mutate text -> TTA -> audio` causes semantic drift: the system "translates" sound through language repeatedly, accumulating linguistic bias and generative artifact until the output diverges from the seed in interesting ways.

## Model dependencies
- Caption: `Qwen/Qwen2-Audio-7B-Instruct` (or alt: SALMONN, Pengi)
- TTA: `cvssp/audioldm2` (or alt: Stable Audio, Tango)
- Both via HuggingFace; will download on first use.

## Expected sonic output
- Step 1–2: recognizable transform of the seed.
- Step 3–4: stylistic drift, identity loosens.
- Step 5+: semantic collapse, captions hallucinate, TTA renders impossible-sound textures.

## Parameters to sweep
- `loop.depth` ∈ [2, 10]
- `loop.text_mutation_prob` ∈ [0, 1]
- `tta.guidance_scale` ∈ [1, 10]
- `tta.duration_s`

## OSC interface
| Address | Args | Effect |
|---------|------|--------|
| `/mvp_b/start` | string audio_path | trigger loop |
| `/mvp_b/done` | string output_path | emitted on completion |

## Backends

| Role | `backend: stub` (default) | `backend: qwen2_audio` / `audioldm2` |
|------|---------------------------|--------------------------------------|
| caption | deterministic spectral-stats text | Qwen2-Audio-7B-Instruct |
| tta | FM synth seeded by text hash | AudioLDM2 |

Stubs verify the loop control flow without downloads. Real backends require:
- ~14 GB download for Qwen2-Audio + ~4 GB for AudioLDM2.
- V100 (SM 7.0) not supported by torch 2.12 + cu130 — pin `torch<=2.4 + cu121` or use a newer GPU.
- Loading large transformer models executes arbitrary code from HF — only use repos you trust.

Switch by editing `config.yaml`:
```yaml
caption:
  backend: qwen2_audio
tta:
  backend: audioldm2
```

## Run

### Offline loop (seed → N steps → out_dir)
```bash
bash run.sh --mode render --input runs/mvp_b/seed_16k.wav --output runs/mvp_b/loop_out
```
Writes `step_00_seed.wav`, `step_01.wav`...`step_NN.wav`, and `transcript.json` to the output directory.

### OSC server
```bash
bash run.sh --mode serve
```

## OSC interface
| Address | Args | Effect |
|---------|------|--------|
| `/mvp_b/start` | string in_path, string out_dir | run loop, replies `/mvp_b/done <transcript_path>` |

## Notes
- Async — even stub takes < 1 s for depth 6; real backends are multi-second per step. Max should fire `/mvp_b/start` then listen for `/mvp_b/done`.
- Each loop step is deterministic given config seed + text_mutation_prob.
