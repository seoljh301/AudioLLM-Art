#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  python experiments/mvp_o_latent_drift/main.py \
    --config experiments/mvp_o_latent_drift/config.yaml "$@"
