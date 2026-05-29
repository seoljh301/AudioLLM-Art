"""MVP-N entry: The Concatenator offline render.

Paper: Tralie & Cantil, "The Concatenator: A Bayesian Approach To Real
Time Concatenative Musaicing", 2024 (arXiv:2411.04366).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_n.particle_filter import PFParams
from src.modules.mvp_n.render import ConcatRenderConfig, render_concatenator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())

    pf = PFParams(
        P=int(raw["P"]), p=int(raw["p"]),
        pd=float(raw["pd"]), tau=float(raw["tau"]),
        L=int(raw["L"]),
        ess_threshold=float(raw["ess_threshold"]),
        l2_reg=float(raw.get("l2_reg", 0.0)),
        seed=int(raw.get("seed", 0)),
    )
    cfg = ConcatRenderConfig(
        sample_rate=int(raw["sample_rate"]),
        n_fft=int(raw["n_fft"]), hop=int(raw["hop"]), win=int(raw["win"]),
        dry_wet=float(raw.get("dry_wet", 1.0)),
    )
    res = render_concatenator([Path(c) for c in args.corpus],
                              args.target, args.out, pf, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
