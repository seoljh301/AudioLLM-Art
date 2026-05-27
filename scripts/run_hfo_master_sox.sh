#!/bin/bash
set -euo pipefail

INPUT="data/Ulaanbaatar.wav"
OUT_DIR="runs/ulaanbaatar_hfo_master"
mkdir -p "$OUT_DIR"

echo "Initializing Command-Line HFO Layering with Stereo Panning..."

# 1. Foundation: Extract Low-End (<300Hz) from original
echo "Extracting Foundation Low-End..."
FOUNDATION="$OUT_DIR/foundation.wav"
sox "$INPUT" "$FOUNDATION" sinc -300 vol 6dB

TRACK_FILES=(
    "runs/ulaanbaatar_pure_chain/step_1_a.wav"
    "runs/ulaanbaatar_pure_chain/step_2_d.wav"
    "runs/ulaanbaatar_pure_chain/step_3_d.wav"
    "runs/ulaanbaatar_pure_chain/step_4_a.wav"
    "runs/ulaanbaatar_pure_chain/step_5_a.wav"
    "runs/ulaanbaatar_pure_chain/step_6_d.wav"
    "runs/ulaanbaatar_pure_chain/step_7_a.wav"
)

# Left and Right gain multipliers for each track to create a stereo spread
# Using a constant power panning law approximation (cos/sin of angle)
# Pan positions: -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8
L_GAINS=("0.99" "0.85" "0.65" "0.50" "0.35" "0.15" "0.01")
R_GAINS=("0.01" "0.15" "0.35" "0.50" "0.65" "0.85" "0.99")

declare -a PROCESSED_TRACKS
for i in "${!TRACK_FILES[@]}"; do
    FILE="${TRACK_FILES[$i]}"
    if [ -f "$FILE" ]; then
        echo "Processing ${FILE} with Pan (L:${L_GAINS[$i]}, R:${R_GAINS[$i]})..."
        TMP_OUT="$OUT_DIR/track_${i}_hfo.wav"
        
        # SoX remix syntax for stereo panning:
        # remix 1v0.8,2v0.2 1v0.2,2v0.8 
        # (Left output mixes L*0.8 + R*0.2, Right output mixes L*0.2 + R*0.8)
        # We apply the highpass filter, resample, apply overall volume reduction, and then PAN.
        
        LG=${L_GAINS[$i]}
        RG=${R_GAINS[$i]}
        
        # We mix both input channels into the left out, and both into the right out, weighted by pan.
        # Vol is set low (0.25) to avoid clipping when 7 tracks are summed.
        sox "$FILE" "$TMP_OUT" rate -v 44100 sinc 800 vol 0.25 remix 1v${LG},2v${LG} 1v${RG},2v${RG}
        
        PROCESSED_TRACKS+=("-v" "1.0" "$TMP_OUT")
    else
        echo "Warning: $FILE not found."
    fi
done

# 3. Sum everything together
echo "Summing all layers..."
FINAL_OUT="$OUT_DIR/ulaanbaatar_hfo_layered.wav"
sox -m -v 1.0 "$FOUNDATION" "${PROCESSED_TRACKS[@]}" "$FINAL_OUT"

echo "HFO Master saved to $FINAL_OUT"
