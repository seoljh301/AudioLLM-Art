#!/bin/bash
set -euo pipefail

IN="data/Ulaanbaatar.wav"
OUT="runs/ulaanbaatar_master/Ulaanbaatar_EARTHQUAKE.wav"

echo "Applying Extreme Sub-Bass EQ directly to the Original Source..."

# SoX EQ chain for the original file:
# 1. bass +15 50 0.5q : Massive 15dB boost at 50Hz for sub-bass power
# 2. equalizer 300 1.0q -6 : 6dB cut at 300Hz to remove mud and clear space for the sub
# 3. norm -0.1 : Re-normalize to prevent clipping from the massive bass boost.

sox $IN $OUT bass +15 50 0.5q equalizer 300 1.0q -6 norm -0.1

echo "SUCCESS: Original Earthquake Bass version generated."
