#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source /home1/irteam/miniconda3/etc/profile.d/conda.sh
conda activate audioart

PYTHONPATH="$ROOT" \
  python experiments/mvp_e_neural_granular/main.py \
    --config experiments/mvp_e_neural_granular/config.yaml \
    "$@"
