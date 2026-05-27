#!/bin/bash
set -euo pipefail

IN="runs/ulaanbaatar_master/foundation_hyperchord_MAX_LOUD.wav"
OUT="runs/ulaanbaatar_master/foundation_hyperchord_EARTHQUAKE.wav"

echo "Applying Extreme Sub-Bass EQ to rattle the floor..."

# We need to boost the extreme lows (40Hz-80Hz) significantly while pulling down 
# the muddy low-mids (250Hz-400Hz) to make the sub-bass punch through.
# SoX EQ chain:
# 1. bass +12 60 0.5q : Boost 60Hz by a massive 12dB (wide Q for natural slope)
# 2. equalizer 300 1.0q -6 : Cut 300Hz by 6dB to remove mud and make room for the sub
# 3. norm -0.1 : Re-normalize to prevent clipping from the massive bass boost.

sox $IN $OUT bass +15 50 0.5q equalizer 300 1.0q -6 norm -0.1

echo "SUCCESS: Earthquake Bass version generated."
