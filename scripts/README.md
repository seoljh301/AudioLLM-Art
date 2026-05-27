# Scripts

This directory contains high-level orchestration tools, composition engines, and mastering utilities for the AudioArt ecosystem.

## Core Engines

- **`meta_symphony.py`**: The apex composition tool. Interweaves Net 1, 2, 3, and Dynamic across a 3-minute timeline with stereo drift and LFOs.
- **`multinet.py`**: Defines macroscopic network topologies (Net 1-3, Max, Dynamic) and handles composite signal flow.
- **`hifi_enhancer.py`**: Applies spectral excitement and air boost for HD neural textures.

## Massive Scale Mastering (SoX)

For processing 1-hour files (68+ minutes) where Python memory is a bottleneck:
- **`run_hfo_master_sox.sh`**: Streams resampling and filtering directly on disk.
- **`run_f0_octave_sox_master.sh`**: Performs f0-aligned multi-octave mixing for massive symphonies.
- **`finish_hyperchord.sh`**: Finalizes the 9-octave foundation drone with aggressive loudness maximizing.

## Utilities

- **`pyloudnorm_mastering.py`**: Loudness normalization targeting -12.0 LUFS (BS.1770-4).
- **`make_stub_rave.py`**: Generates untreated RAVE .ts skeletons for API verification.
- **`check_metrics.py`**: Batch scans directories for RMS, Flatness, and NaN errors.
