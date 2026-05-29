"""MVP-V entry: caption-conditioned tokens."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_v.cap_tokens import CapTokenParams, render_cap_tokens


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
    params = CapTokenParams(
        codec=str(raw["codec"]),
        bandwidth=float(raw["bandwidth"]),
        device=str(raw["device"]),
        chunk_seconds=float(raw["chunk_seconds"]),
        walk_mode=str(raw["walk_mode"]),
        base_seed_offset=int(raw["base_seed_offset"]),
        walk_strength=int(raw["walk_strength"]),
        dry_wet=float(raw.get("dry_wet", 1.0)),
        use_real=bool(raw.get("use_real", False)),
    )
    res = render_cap_tokens(args.target, args.out, params)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
