"""MVP-I entry point: Neural Bass Massive.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import yaml

from src.core.mix import MixConfig
from src.core.texture_governor import TextureGovernorConfig
from src.modules.mvp_c.codec_io import load_codec, load_audio_mono, save_audio_mono
from src.modules.mvp_i.bass_massive import MassiveParams
from src.modules.mvp_i.render import RenderConfig, render


def _build_params(cfg: dict) -> MassiveParams:
    m = cfg["massive"]
    return MassiveParams(
        smear_delay=int(m.get("smear_delay", 0)),
        smear_quantizers=tuple(m.get("smear_quantizers", (0, 2))),
        jitter_rate=float(m.get("jitter_rate", 0.0)),
        jitter_quantizers=tuple(m.get("jitter_quantizers", (0, 3))),
        fold_leak_rate=float(m.get("fold_leak_rate", 0.0)),
        fold_source_range=tuple(m.get("fold_source_range", (8, 12))),
        fold_target_range=tuple(m.get("fold_target_range", (0, 2))),
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
            sub_boost_db=float(mix_cfg.get("sub_boost_db", 0.0)),
        ),
        governor=TextureGovernorConfig(enabled=bool(gov_cfg.get("enabled", False))),
    )


def _run_render(cfg: dict, args: argparse.Namespace) -> None:
    handle = load_codec(cfg["codec"]["name"], device=cfg["codec"].get("device", "cpu"))
    params = _build_params(cfg)
    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))

    audio = load_audio_mono(Path(args.input), target_sr=handle.sample_rate)
    out = render(audio, handle, params, rcfg, rng)
    save_audio_mono(Path(args.output), out, handle.sample_rate)
    logging.info("wrote %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP-I: Neural Bass Massive")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--mode", choices=["render"], default="render")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _run_render(cfg, args)


if __name__ == "__main__":
    main()
