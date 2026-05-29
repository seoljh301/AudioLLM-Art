"""MVP-P entry: phase hallucinator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_p.phase_halluc import HallucParams
from src.modules.mvp_p.render import render_halluc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())
    params = HallucParams(
        mode=str(raw["mode"]),
        duration_s=float(raw["duration_s"]),
        sr=int(raw["sr"]),
        n_fft=int(raw["n_fft"]),
        hop=int(raw["hop"]),
        griffin_lim_iters=int(raw["griffin_lim_iters"]),
        smooth_t=float(raw["smooth_t"]),
        smooth_f=float(raw["smooth_f"]),
        chord_freqs_hz=tuple(raw.get("chord_freqs_hz", (110.0, 165.0, 220.0, 330.0))),
        chord_bw_bins=int(raw["chord_bw_bins"]),
        seed=int(raw.get("seed", 0)),
    )
    res = render_halluc(args.out, params)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
