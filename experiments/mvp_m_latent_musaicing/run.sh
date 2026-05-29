#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH=. \
  python experiments/mvp_m_latent_musaicing/main.py \
    --config experiments/mvp_m_latent_musaicing/config.yaml \
    "$@"
