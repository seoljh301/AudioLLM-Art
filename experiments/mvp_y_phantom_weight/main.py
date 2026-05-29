"""MVP-Y entry: phantom weight generator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_y.phantom_weight import PhantomParams
from src.modules.mvp_y.render import render_phantom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())
    params = PhantomParams(
        morph_t=float(raw["morph_t"]),
        morph_mode=str(raw["morph_mode"]),
        latent_mode=str(raw["latent_mode"]),
        latent_sigma=float(raw["latent_sigma"]),
        latent_smooth=float(raw["latent_smooth"]),
        duration_s=float(raw["duration_s"]),
        seed=int(raw.get("seed", 0)),
    )
    res = render_phantom([Path(p) for p in raw["model_paths"]],
                         args.out, params,
                         device=str(raw.get("device", "cuda")))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
