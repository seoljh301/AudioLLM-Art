# MVP-C — EnCodec / DAC Token Bending

## Hypothesis
Corrupting RVQ tokens at different layers produces musically distinct artifacts:
- **Upper quantizers** carry fine residual detail → corruption = high-frequency texture damage.
- **Lower quantizers** carry coarse structure → corruption = identity collapse.
Selectively corrupting layers gives independent control over timbre vs. structure.

## Model dependencies
- `encodec` pip package (Meta EnCodec, 24kHz/48kHz)
- `descript-audio-codec` for DAC 44.1kHz

## Expected sonic output
- `bit_flip` (low rate, upper quantizers): glassy granular grit.
- `quantizer_drop`: hollowed-out, codec-broken texture.
- `shuffle`: temporal stutter / scramble.
- `invalid_token`: silence pockets + decoder hallucinations.

## Parameters to sweep
- `bend.mode` ∈ {bit_flip, quantizer_drop, shuffle, invalid_token}
- `bend.rate` ∈ [0, 0.3]
- `bend.quantizer_range` — upper vs. lower split

## OSC interface
| Address | Args | Effect |
|---------|------|--------|
| `/mvp_c/rate` | float | corruption rate |
| `/mvp_c/mode` | string | switch mode |

## Run

Offline render (file → bent file):
```bash
bash run.sh --mode render --input path/to/in.wav --output path/to/out.wav
```

OSC server (Max triggers `/mvp_c/render <in> <out>`):
```bash
bash run.sh --mode serve
```

## Hardware notes

- V100 GPU supported via pinned `torch==2.4.1 + cu121` (root `environment.yaml`).
- EnCodec 24 kHz on V100: 5 s audio renders in ~1.6 s (~170× faster than CPU).
- EnCodec 24 kHz on CPU: ~50× slower; tight for streaming, fine for batch.
