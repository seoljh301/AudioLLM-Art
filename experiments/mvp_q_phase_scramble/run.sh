#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHONPATH=. python experiments/mvp_q_phase_scramble/main.py \
    --config experiments/mvp_q_phase_scramble/config.yaml "$@"
