#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python experiments/mvp_p_phase_halluc/main.py \
    --config experiments/mvp_p_phase_halluc/config.yaml "$@"
