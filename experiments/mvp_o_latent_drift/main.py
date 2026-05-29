"""MVP-O entry: latent drift generator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_o.latent_drift import DriftParams
from src.modules.mvp_o.render import RenderConfig, render_drift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())
    params = DriftParams(
        mode=str(raw["mode"]),
        sigma=float(raw["sigma"]),
        smooth=float(raw["smooth"]),
        drift_rate=float(raw.get("drift_rate", 0.0)),
        base_freqs_hz=tuple(raw.get("base_freqs_hz", (0.1, 0.23, 0.41, 0.79))),
        seed=int(raw.get("seed", 0)),
    )
    cfg = RenderConfig(
        duration_s=float(raw["duration_s"]),
        device=str(raw.get("device", "cuda")),
    )
    res = render_drift(Path(raw["model_path"]), args.out, params, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
