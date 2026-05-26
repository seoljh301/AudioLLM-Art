"""MVP-H entry point: Codebook Organ.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import yaml

from src.modules.mvp_c.codec_io import load_codec, save_audio_mono
from src.modules.mvp_h.codebook_organ import OrganParams
from src.modules.mvp_h.render import render_generative


def _build_params(cfg: dict) -> OrganParams:
    o = cfg["organ"]
    return OrganParams(
        mode=o.get("mode", "prime"),
        base_token=int(o.get("base_token", 0)),
        stride=int(o.get("stride", 1)),
        duration_frames=int(o.get("duration_frames", 300)),
    )


def _run_gen(cfg: dict, args: argparse.Namespace) -> None:
    handle = load_codec(cfg["codec"]["name"], device=cfg["codec"].get("device", "cpu"))
    params = _build_params(cfg)
    # If output is requested for a long time, adjust frames
    if args.duration_sec:
        # EnCodec 24kHz is 75fps
        params.duration_frames = int(float(args.duration_sec) * 75)

    rng = np.random.default_rng(cfg.get("seed", 0))
    out = render_generative(handle, params, rng)
    save_audio_mono(Path(args.output), out, handle.sample_rate)
    logging.info("wrote %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP-H: Codebook Organ")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--mode", choices=["gen"], default="gen")
    parser.add_argument("--output", type=str)
    parser.add_argument("--duration_sec", type=float, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mode == "gen":
        _run_gen(cfg, args)


if __name__ == "__main__":
    main()
