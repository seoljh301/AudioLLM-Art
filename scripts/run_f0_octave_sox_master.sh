#!/bin/bash
set -euo pipefail

echo 'Starting SoX f0-octave-spread mixing...'

echo 'Processing Foundation (Pitch shift: -1800 cents)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_foundation_oct.wav rate -v 48000 sinc -300 vol 1.5 pitch -1800
echo 'Processing neural track 1 (Pitch shift: -600 cents)...'
sox runs/ulaanbaatar_pure_chain/step_1_a.wav runs/ulaanbaatar_master/tmp_track_oct_0.wav vol 0.2 pitch -600
echo 'Processing neural track 2 (Pitch shift: 600 cents)...'
sox runs/ulaanbaatar_pure_chain/step_2_d.wav runs/ulaanbaatar_master/tmp_track_oct_1.wav vol 0.15 pitch 600
echo 'Processing neural track 3 (Pitch shift: 1800 cents)...'
sox runs/ulaanbaatar_pure_chain/step_3_d.wav runs/ulaanbaatar_master/tmp_track_oct_2.wav vol 0.1 pitch 1800
echo 'Processing neural track 4 (Pitch shift: 3000 cents)...'
sox runs/ulaanbaatar_pure_chain/step_4_a.wav runs/ulaanbaatar_master/tmp_track_oct_3.wav vol 0.1 pitch 3000
echo 'Processing neural track 5 (Pitch shift: -1700 cents)...'
sox runs/ulaanbaatar_pure_chain/step_5_a.wav runs/ulaanbaatar_master/tmp_track_oct_4.wav vol 0.2 pitch -1700
echo 'Processing neural track 6 (Pitch shift: -500 cents)...'
sox runs/ulaanbaatar_pure_chain/step_6_d.wav runs/ulaanbaatar_master/tmp_track_oct_5.wav vol 0.15 pitch -500
echo 'Processing neural track 7 (Pitch shift: 700 cents)...'
sox runs/ulaanbaatar_pure_chain/step_7_a.wav runs/ulaanbaatar_master/tmp_track_oct_6.wav vol 0.1 pitch 700

echo 'Summing all octave-aligned tracks...'
sox -m runs/ulaanbaatar_master/tmp_foundation_oct.wav runs/ulaanbaatar_master/tmp_track_oct_0.wav runs/ulaanbaatar_master/tmp_track_oct_1.wav runs/ulaanbaatar_master/tmp_track_oct_2.wav runs/ulaanbaatar_master/tmp_track_oct_3.wav runs/ulaanbaatar_master/tmp_track_oct_4.wav runs/ulaanbaatar_master/tmp_track_oct_5.wav runs/ulaanbaatar_master/tmp_track_oct_6.wav runs/ulaanbaatar_master/f0_octave_symphony.wav

echo 'Applying final limiter...'
sox runs/ulaanbaatar_master/f0_octave_symphony.wav runs/ulaanbaatar_master/f0_octave_symphony_limited.wav compand 0.01,0.1 -60,-60,0,-0.5 -1 -6

echo 'Cleaning up temporary files...'
rm -f runs/ulaanbaatar_master/tmp_foundation_oct.wav runs/ulaanbaatar_master/tmp_track_oct_0.wav runs/ulaanbaatar_master/tmp_track_oct_1.wav runs/ulaanbaatar_master/tmp_track_oct_2.wav runs/ulaanbaatar_master/tmp_track_oct_3.wav runs/ulaanbaatar_master/tmp_track_oct_4.wav runs/ulaanbaatar_master/tmp_track_oct_5.wav runs/ulaanbaatar_master/tmp_track_oct_6.wav runs/ulaanbaatar_master/f0_octave_symphony.wav

echo 'SUCCESS: F0-Octave Master saved to runs/ulaanbaatar_master/f0_octave_symphony_limited.wav'
