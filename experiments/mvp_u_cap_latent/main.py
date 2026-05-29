"""MVP-U entry: caption-steered latent."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_u.cap_latent import CapLatentParams, render_cap_latent


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
    params = CapLatentParams(
        bias_strength=float(raw["bias_strength"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        use_real=bool(raw["use_real"]),
        model_path=str(raw["model_path"]),
        device=str(raw["device"]),
        seed=int(raw.get("seed", 0)),
        dry_wet=float(raw.get("dry_wet", 1.0)),
    )
    res = render_cap_latent(args.target, args.out, params)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
