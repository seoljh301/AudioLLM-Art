#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python experiments/mvp_n_concatenator/main.py \
    --config experiments/mvp_n_concatenator/config.yaml "$@"
