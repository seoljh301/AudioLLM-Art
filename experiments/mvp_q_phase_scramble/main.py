"""MVP-Q entry: phase scrambler."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_q.phase_scramble import ScrambleParams
from src.modules.mvp_q.render import RenderConfig, render_scramble


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())
    params = ScrambleParams(
        mode=str(raw["mode"]),
        rate=float(raw["rate"]),
        smooth=float(raw["smooth"]),
        rotate_max=float(raw["rotate_max"]),
        seed=int(raw.get("seed", 0)),
    )
    cfg = RenderConfig(
        sample_rate=int(raw["sample_rate"]),
        n_fft=int(raw["n_fft"]), hop=int(raw["hop"]), win=int(raw["win"]),
        dry_wet=float(raw.get("dry_wet", 1.0)),
    )
    res = render_scramble(args.target, args.out, params, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
