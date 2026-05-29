"""MVP-T entry: CLAP / text-driven musaicing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_t.clap_musaicing import ClapMusaicParams, render_clap_musaicing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", required=True,
                    help="ordered text prompts; output concatenates segments per prompt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())
    params = ClapMusaicParams(
        grain_seconds=float(raw["grain_seconds"]),
        stride_seconds=float(raw["stride_seconds"]),
        seg_seconds=float(raw["seg_seconds"]),
        temperature=float(raw["temperature"]),
        walk_strength=float(raw.get("walk_strength", 0.0)),
        use_real=bool(raw["use_real"]),
        sr=int(raw["sr"]),
        crossfade_seconds=float(raw["crossfade_seconds"]),
        seed=int(raw.get("seed", 0)),
    )
    res = render_clap_musaicing([Path(c) for c in args.corpus],
                                args.prompts, args.out, params)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
