"""MVP-G entry point: Latent Feedback Echo.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import yaml

from src.core.mix import MixConfig
from src.core.texture_governor import TextureGovernorConfig
from src.modules.mvp_a.rave_io import load_audio_mono, load_rave, save_audio_mono
from src.modules.mvp_g.latent_feedback import FeedbackParams
from src.modules.mvp_g.render import RenderConfig, render


def _build_params(cfg: dict) -> FeedbackParams:
    fb = cfg["feedback"]
    return FeedbackParams(
        delay_frames=int(fb.get("delay_frames", 32)),
        feedback=float(fb.get("feedback", 0.4)),
        mix=float(fb.get("mix", 0.5)),
    )


def _build_render_config(cfg: dict) -> RenderConfig:
    render_cfg = cfg.get("render", {})
    mix_cfg = cfg.get("mix", {})
    gov_cfg = cfg.get("governor", {})

    return RenderConfig(
        chunk_seconds=float(render_cfg.get("chunk_seconds", 4.0)),
        mix=MixConfig(
            dry_wet=float(mix_cfg.get("dry_wet", 1.0)),
            low_anchor_hz=float(mix_cfg["low_anchor_hz"]) if mix_cfg.get("low_anchor_hz") else None,
        ),
        governor=TextureGovernorConfig(
            enabled=bool(gov_cfg.get("enabled", False)),
        ),
    )


def _run_render(cfg: dict, args: argparse.Namespace) -> None:
    handle = load_rave(cfg["model"]["path"], device=cfg["model"].get("device", "cpu"))
    params = _build_params(cfg)
    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))

    audio = load_audio_mono(Path(args.input), target_sr=handle.sample_rate)
    out = render(audio, handle, params, rcfg, rng)
    save_audio_mono(Path(args.output), out, handle.sample_rate)
    logging.info("wrote %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP-G: Latent Feedback Echo")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--mode", choices=["render"], default="render")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mode == "render":
        _run_render(cfg, args)


if __name__ == "__main__":
    main()
