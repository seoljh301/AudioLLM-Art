#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source /home1/irteam/miniconda3/etc/profile.d/conda.sh
conda activate audioart

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  PYTHONPATH="$ROOT" \
  python experiments/mvp_c_encodec_bend/main.py \
    --config experiments/mvp_c_encodec_bend/config.yaml \
    "$@"
