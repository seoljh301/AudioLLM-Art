"""MVP-M entry: latent musaicing offline render.

Paper: Kui et al., "Latent Granular Resynthesis using Neural Audio Codecs",
ISMIR 2025 (arXiv:2507.19202). Training-free; uses pretrained codec
encoder/decoder.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_m.musaicing import MusaicingParams
from src.modules.mvp_m.render import RenderConfig, render_musaicing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--corpus", nargs="+", required=True,
                    help="source corpus wav paths")
    ap.add_argument("--target", type=Path, required=True,
                    help="target wav path")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg_raw = yaml.safe_load(args.config.read_text())

    params = MusaicingParams(
        grain_size=int(cfg_raw["grain_size"]),
        stride=int(cfg_raw["stride"]),
        target_stride=int(cfg_raw["target_stride"]),
        temperature=float(cfg_raw["temperature"]),
        overlap_add=bool(cfg_raw["overlap_add"]),
        walk_strength=float(cfg_raw.get("walk_strength", 0.0)),
        seed=int(cfg_raw.get("seed", 0)),
    )
    cfg = RenderConfig(
        codec=str(cfg_raw.get("codec", "encodec_24khz")),
        device=str(cfg_raw.get("device", "cuda")),
        dry_wet=float(cfg_raw.get("dry_wet", 1.0)),
    )
    res = render_musaicing([Path(c) for c in args.corpus],
                           args.target, args.out, params, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
