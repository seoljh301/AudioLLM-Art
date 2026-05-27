#!/bin/bash
set -euo pipefail

echo 'Starting Foundation Hyper-Chord Generation (-4 to +4 Octaves)...'

echo 'Processing Octave -4 (Pitch shift: -4200 cents, Vol: 0.040)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_m4.wav rate -v 48000 vol 0.040  pitch -2400 pitch -1800
echo 'Processing Octave -3 (Pitch shift: -3000 cents, Vol: 0.050)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_m3.wav rate -v 48000 vol 0.050  pitch -2400 pitch -600
echo 'Processing Octave -2 (Pitch shift: -1800 cents, Vol: 0.067)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_m2.wav rate -v 48000 vol 0.067  pitch -1800
echo 'Processing Octave -1 (Pitch shift: -600 cents, Vol: 0.100)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_m1.wav rate -v 48000 vol 0.100  pitch -600
echo 'Processing Octave 0 (Pitch shift: 600 cents, Vol: 0.200)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_0.wav rate -v 48000 vol 0.200  pitch 600
echo 'Processing Octave 1 (Pitch shift: 1800 cents, Vol: 0.100)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_1.wav rate -v 48000 vol 0.100  pitch 1800
echo 'Processing Octave 2 (Pitch shift: 3000 cents, Vol: 0.067)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_2.wav rate -v 48000 vol 0.067  pitch 2400 pitch 600
echo 'Processing Octave 3 (Pitch shift: 4200 cents, Vol: 0.050)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_3.wav rate -v 48000 vol 0.050  pitch 2400 pitch 1800
echo 'Processing Octave 4 (Pitch shift: 5400 cents, Vol: 0.040)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_4.wav rate -v 48000 vol 0.040  pitch 2400 pitch 2400 pitch 600

echo 'Summing all 9 octave layers...'
sox -m runs/ulaanbaatar_master/tmp_found_oct_m4.wav runs/ulaanbaatar_master/tmp_found_oct_m3.wav runs/ulaanbaatar_master/tmp_found_oct_m2.wav runs/ulaanbaatar_master/tmp_found_oct_m1.wav runs/ulaanbaatar_master/tmp_found_oct_0.wav runs/ulaanbaatar_master/tmp_found_oct_1.wav runs/ulaanbaatar_master/tmp_found_oct_2.wav runs/ulaanbaatar_master/tmp_found_oct_3.wav runs/ulaanbaatar_master/tmp_found_oct_4.wav runs/ulaanbaatar_master/foundation_hyperchord.wav

echo 'Applying final limiter...'
sox runs/ulaanbaatar_master/foundation_hyperchord.wav runs/ulaanbaatar_master/foundation_hyperchord_limited.wav compand 0.01,0.1 -60,-60,0,-0.5 -1 -6

echo 'Cleaning up temporary files...'
rm -f runs/ulaanbaatar_master/tmp_found_oct_m4.wav runs/ulaanbaatar_master/tmp_found_oct_m3.wav runs/ulaanbaatar_master/tmp_found_oct_m2.wav runs/ulaanbaatar_master/tmp_found_oct_m1.wav runs/ulaanbaatar_master/tmp_found_oct_0.wav runs/ulaanbaatar_master/tmp_found_oct_1.wav runs/ulaanbaatar_master/tmp_found_oct_2.wav runs/ulaanbaatar_master/tmp_found_oct_3.wav runs/ulaanbaatar_master/tmp_found_oct_4.wav runs/ulaanbaatar_master/foundation_hyperchord.wav

echo 'SUCCESS: Foundation Hyper-Chord saved to runs/ulaanbaatar_master/foundation_hyperchord_limited.wav'
