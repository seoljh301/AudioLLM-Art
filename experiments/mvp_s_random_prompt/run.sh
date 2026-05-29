#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python experiments/mvp_s_random_prompt/main.py \
    --config experiments/mvp_s_random_prompt/config.yaml "$@"
