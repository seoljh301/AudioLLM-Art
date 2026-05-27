#!/bin/bash
set -euo pipefail

ORIG="data/Ulaanbaatar.wav"
OUT_DIR="runs/ulaanbaatar_master"

echo "Processing Octave 3 (Pitch shift: 4200 cents, Vol: 0.050)..."
sox $ORIG $OUT_DIR/tmp_found_oct_3.wav rate -v 48000 vol 0.050 pitch 2400 pitch 1800

echo "Processing Octave 4 (Pitch shift: 5400 cents, Vol: 0.040)..."
sox $ORIG $OUT_DIR/tmp_found_oct_4.wav rate -v 48000 vol 0.040 pitch 2400 pitch 2400 pitch 600

echo "Summing all 9 layers..."
sox -m \
  $OUT_DIR/tmp_found_oct_m4.wav \
  $OUT_DIR/tmp_found_oct_m3.wav \
  $OUT_DIR/tmp_found_oct_m2.wav \
  $OUT_DIR/tmp_found_oct_m1.wav \
  $OUT_DIR/tmp_found_oct_0.wav \
  $OUT_DIR/tmp_found_oct_1.wav \
  $OUT_DIR/tmp_found_oct_2.wav \
  $OUT_DIR/tmp_found_oct_3.wav \
  $OUT_DIR/tmp_found_oct_4.wav \
  $OUT_DIR/foundation_hyperchord_final.wav

echo "Applying aggressive mastering (Loudness Maximizing)..."
# We use a much more aggressive compander curve here to squash the dynamics and bring up the average loudness.
# -60,-60 (noise floor) -> -40,-20 (bring up mid-lows) -> 0,-0.5 (squash peaks)
# We also add a +4dB output gain at the end.
sox $OUT_DIR/foundation_hyperchord_final.wav $OUT_DIR/foundation_hyperchord_limited.wav compand 0.05,0.2 6:-60,-60,-40,-20,0,-0.5 -1 -6 4

echo "SUCCESS: Loud 9-Octave Hyper-Chord Master saved."
