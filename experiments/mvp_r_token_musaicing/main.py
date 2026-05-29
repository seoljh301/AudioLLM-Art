"""MVP-R entry: token musaicing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_r.token_musaicing import TokenMusaicParams
from src.modules.mvp_r.render import RenderConfig, render_token_musaicing


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
    params = TokenMusaicParams(
        grain_size=int(raw["grain_size"]),
        stride=int(raw["stride"]),
        target_stride=int(raw["target_stride"]),
        dist=str(raw["dist"]),
        q_weights=tuple(raw["q_weights"]),
        temperature=float(raw["temperature"]),
        walk_strength=float(raw.get("walk_strength", 0.0)),
        seed=int(raw.get("seed", 0)),
    )
    cfg = RenderConfig(
        codec=str(raw["codec"]),
        bandwidth=float(raw["bandwidth"]),
        device=str(raw.get("device", "cuda")),
        dry_wet=float(raw.get("dry_wet", 1.0)),
    )
    res = render_token_musaicing([Path(c) for c in args.corpus],
                                 args.target, args.out, params, cfg)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
