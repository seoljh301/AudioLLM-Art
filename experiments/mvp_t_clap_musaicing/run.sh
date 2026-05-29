#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python experiments/mvp_t_clap_musaicing/main.py \
    --config experiments/mvp_t_clap_musaicing/config.yaml "$@"
