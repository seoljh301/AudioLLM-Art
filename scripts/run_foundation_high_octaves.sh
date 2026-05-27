#!/bin/bash
set -euo pipefail

echo 'Starting Foundation High Octave Generation (+3 and +4)...'

echo 'Processing Octave 3 (Pitch shift: 4200 cents, Vol: 0.150)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_3.wav rate -v 48000 vol 0.150  pitch 2400 pitch 1800
echo 'Processing Octave 4 (Pitch shift: 5400 cents, Vol: 0.150)...'
sox data/Ulaanbaatar.wav runs/ulaanbaatar_master/tmp_found_oct_4.wav rate -v 48000 vol 0.150  pitch 2400 pitch 2400 pitch 600

echo 'Summing the +3 and +4 octave layers...'
sox -m runs/ulaanbaatar_master/tmp_found_oct_3.wav runs/ulaanbaatar_master/tmp_found_oct_4.wav runs/ulaanbaatar_master/foundation_high_octaves.wav

echo 'Applying final limiter...'
sox runs/ulaanbaatar_master/foundation_high_octaves.wav runs/ulaanbaatar_master/foundation_high_octaves_limited.wav compand 0.01,0.1 -60,-60,0,-0.5 -1 -6

echo 'Cleaning up temporary files...'
rm -f runs/ulaanbaatar_master/tmp_found_oct_3.wav runs/ulaanbaatar_master/tmp_found_oct_4.wav runs/ulaanbaatar_master/foundation_high_octaves.wav

echo 'SUCCESS: High Octaves saved to runs/ulaanbaatar_master/foundation_high_octaves_limited.wav'
