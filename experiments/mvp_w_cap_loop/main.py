"""MVP-W entry: caption→TTA recursive loop."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_w.cap_loop import LoopParams, render_cap_loop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())
    params = LoopParams(
        n_iters=int(raw["n_iters"]),
        duration_per_step_s=float(raw["duration_per_step_s"]),
        keep_mix=float(raw["keep_mix"]),
        sr=int(raw["sr"]),
        use_real=bool(raw["use_real"]),
        save_intermediates=bool(raw["save_intermediates"]),
    )
    res = render_cap_loop(args.target, args.out_dir, params)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
