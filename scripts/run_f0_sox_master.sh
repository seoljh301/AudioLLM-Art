#!/bin/bash
set -euo pipefail

echo 'Starting SoX f0-aligned mixing with Original Foundation...'

echo 'Processing Foundation (Pitch shift: 600 cents)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_foundation.wav rate -v 48000 sinc -300 vol 1.5 pitch 600
echo 'Processing neural track 0 (Pitch shift: 600 cents)...'
sox runs/ulaanbaatar_pure_chain/step_1_a.wav runs/ulaanbaatar_master/tmp_track_0.wav vol 0.15 pitch 600
echo 'Processing neural track 1 (Pitch shift: 600 cents)...'
sox runs/ulaanbaatar_pure_chain/step_2_d.wav runs/ulaanbaatar_master/tmp_track_1.wav vol 0.15 pitch 600
echo 'Processing neural track 2 (Pitch shift: 600 cents)...'
sox runs/ulaanbaatar_pure_chain/step_3_d.wav runs/ulaanbaatar_master/tmp_track_2.wav vol 0.15 pitch 600
echo 'Processing neural track 3 (Pitch shift: 600 cents)...'
sox runs/ulaanbaatar_pure_chain/step_4_a.wav runs/ulaanbaatar_master/tmp_track_3.wav vol 0.15 pitch 600
echo 'Processing neural track 4 (Pitch shift: -500 cents)...'
sox runs/ulaanbaatar_pure_chain/step_5_a.wav runs/ulaanbaatar_master/tmp_track_4.wav vol 0.15 pitch -500
echo 'Processing neural track 5 (Pitch shift: -500 cents)...'
sox runs/ulaanbaatar_pure_chain/step_6_d.wav runs/ulaanbaatar_master/tmp_track_5.wav vol 0.15 pitch -500
echo 'Processing neural track 6 (Pitch shift: -500 cents)...'
sox runs/ulaanbaatar_pure_chain/step_7_a.wav runs/ulaanbaatar_master/tmp_track_6.wav vol 0.15 pitch -500

echo 'Summing all aligned tracks...'
sox -m runs/ulaanbaatar_master/tmp_foundation.wav runs/ulaanbaatar_master/tmp_track_0.wav runs/ulaanbaatar_master/tmp_track_1.wav runs/ulaanbaatar_master/tmp_track_2.wav runs/ulaanbaatar_master/tmp_track_3.wav runs/ulaanbaatar_master/tmp_track_4.wav runs/ulaanbaatar_master/tmp_track_5.wav runs/ulaanbaatar_master/tmp_track_6.wav runs/ulaanbaatar_master/f0_aligned_symphony.wav

echo 'Applying final limiter...'
sox runs/ulaanbaatar_master/f0_aligned_symphony.wav runs/ulaanbaatar_master/f0_aligned_symphony_limited.wav compand 0.01,0.1 -60,-60,0,-0.5 -1 -6

echo 'Cleaning up temporary files...'
rm -f runs/ulaanbaatar_master/tmp_foundation.wav runs/ulaanbaatar_master/tmp_track_0.wav runs/ulaanbaatar_master/tmp_track_1.wav runs/ulaanbaatar_master/tmp_track_2.wav runs/ulaanbaatar_master/tmp_track_3.wav runs/ulaanbaatar_master/tmp_track_4.wav runs/ulaanbaatar_master/tmp_track_5.wav runs/ulaanbaatar_master/tmp_track_6.wav runs/ulaanbaatar_master/f0_aligned_symphony.wav

echo 'SUCCESS: F0-Aligned Master saved to runs/ulaanbaatar_master/f0_aligned_symphony_limited.wav'
