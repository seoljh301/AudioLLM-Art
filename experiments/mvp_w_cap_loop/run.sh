#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python experiments/mvp_w_cap_loop/main.py \
    --config experiments/mvp_w_cap_loop/config.yaml "$@"
