#!/bin/bash
set -euo pipefail

IN="runs/ulaanbaatar_master/foundation_hyperchord_final.wav"
OUT="runs/ulaanbaatar_master/foundation_hyperchord_MAX_LOUD.wav"

echo "Applying massive gain and hard limiter to maximize loudness..."
# We checked the max amplitude is ~0.03. We need to boost it by at least 20-30dB.
# Using 'gain' effect to push volume, and 'compand' as a hard brickwall limiter to catch any stray peaks at -0.1dB.
# gain -n -0.1 will normalize the file so the absolute peak hits -0.1 dB. This is the cleanest way to maximize volume.

sox $IN $OUT norm -0.1

echo "SUCCESS: Normalized to maximum safe loudness."
