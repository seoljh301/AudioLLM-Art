#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
source /home1/irteam/miniconda3/etc/profile.d/conda.sh
conda activate audioart
PYTHONPATH="$ROOT" python experiments/mvp_f_neural_freeze/main.py --config experiments/mvp_f_neural_freeze/config.yaml "$@"
