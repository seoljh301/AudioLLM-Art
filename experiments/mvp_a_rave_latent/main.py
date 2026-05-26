"""MVP-A entry point: RAVE latent perturbation.

Two modes:
  --mode render --input in.wav --output out.wav    # offline file -> perturbed file
  --mode serve                                       # OSC server
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from src.core.osc_bridge import OSCBridge, OSCBridgeConfig
from src.core.mix import MixConfig
from src.core.texture_governor import TextureGovernorConfig
from src.modules.mvp_a.latent_perturb import PerturbParams
from src.modules.mvp_a.rave_io import load_audio_mono, load_rave, save_audio_mono
from src.modules.mvp_a.render import RenderConfig, render


def _build_params(cfg: dict, latent_dim: int) -> PerturbParams:
    p = cfg["perturb"]
    bias = p.get("bias_vector")
    mask = p.get("freeze_mask")
    return PerturbParams(
        noise_scale=float(p.get("noise_scale", 0.0)),
        dim_dropout=float(p.get("dim_dropout", 0.0)),
        dim_shuffle=bool(p.get("dim_shuffle", False)),
        bias_vector=np.asarray(bias, dtype=np.float32) if bias else None,
        freeze_mask=np.asarray(mask, dtype=bool) if mask else None,
        noise_mode=p.get("noise_mode", "white"),
        noise_smooth=float(p.get("noise_smooth", 0.98)),
    )


def _build_render_config(cfg: dict) -> RenderConfig:
    render_cfg = cfg.get("render", {})
    mix_cfg = cfg.get("mix", {})
    gov_cfg = cfg.get("governor", {})

    return RenderConfig(
        chunk_seconds=float(render_cfg.get("chunk_seconds", 4.0)),
        overlap_seconds=float(render_cfg.get("overlap_seconds", 0.05)),
        mix=MixConfig(
            dry_wet=float(mix_cfg.get("dry_wet", 1.0)),
            rms_match=bool(mix_cfg.get("rms_match", False)),
            limiter=bool(mix_cfg.get("limiter", True)),
            limiter_drive=float(mix_cfg.get("limiter_drive", 1.0)),
            low_anchor_hz=float(mix_cfg["low_anchor_hz"]) if mix_cfg.get("low_anchor_hz") else None,
            sub_boost_db=float(mix_cfg.get("sub_boost_db", 0.0)),
        ),
        governor=TextureGovernorConfig(
            enabled=bool(gov_cfg.get("enabled", False)),
            min_wet=float(gov_cfg.get("min_wet", 0.10)),
            flatness_max=float(gov_cfg.get("flatness_max", 0.55)),
            rms_min=float(gov_cfg.get("rms_min", 1e-4)),
            rms_max=float(gov_cfg.get("rms_max", 0.85)),
            centroid_max_ratio=float(gov_cfg.get("centroid_max_ratio", 0.42)),
        ),
    )


def _run_render(cfg: dict, args: argparse.Namespace) -> None:
    handle = load_rave(cfg["model"]["path"], device=cfg["model"].get("device", "cpu"))
    params = _build_params(cfg, handle.latent_dim)
    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))

    audio = load_audio_mono(Path(args.input), target_sr=handle.sample_rate)
    logging.info("loaded %d samp (%.2fs) from %s", len(audio),
                 len(audio) / handle.sample_rate, args.input)
    out = render(audio, handle, params, rcfg, rng)
    save_audio_mono(Path(args.output), out, handle.sample_rate)
    logging.info("wrote %s", args.output)


def _run_serve(cfg: dict) -> None:
    handle = load_rave(cfg["model"]["path"], device=cfg["model"].get("device", "cpu"))
    params = _build_params(cfg, handle.latent_dim)
    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))

    bridge = OSCBridge(OSCBridgeConfig(**cfg["osc"]))

    def on_noise(_addr: str, val: float) -> None:
        params.noise_scale = float(val)
        logging.info("noise_scale -> %.4f", params.noise_scale)

    def on_dropout(_addr: str, val: float) -> None:
        params.dim_dropout = float(val)
        logging.info("dim_dropout -> %.4f", params.dim_dropout)

    def on_shuffle(_addr: str, val: int) -> None:
        params.dim_shuffle = bool(int(val))
        logging.info("dim_shuffle -> %s", params.dim_shuffle)

    def on_render(_addr: str, in_path: str, out_path: str) -> None:
        logging.info("OSC render: %s -> %s (noise=%.3f drop=%.3f)",
                     in_path, out_path, params.noise_scale, params.dim_dropout)
        try:
            audio = load_audio_mono(Path(in_path), target_sr=handle.sample_rate)
            out = render(audio, handle, params, rcfg, rng)
            save_audio_mono(Path(out_path), out, handle.sample_rate)
            bridge.send("/mvp_a/done", out_path)
        except Exception as e:  # noqa: BLE001
            logging.exception("render failed")
            bridge.send("/mvp_a/error", repr(e))

    bridge.register("/mvp_a/noise", on_noise)
    bridge.register("/mvp_a/dropout", on_dropout)
    bridge.register("/mvp_a/shuffle", on_shuffle)
    bridge.register("/mvp_a/render", on_render)
    logging.info("MVP-A serve ready on %s:%d",
                 cfg["osc"]["listen_host"], cfg["osc"]["listen_port"])
    bridge.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP-A: RAVE latent perturbation")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--mode", choices=["render", "serve"], default="render")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mode == "render":
        if not args.input or not args.output:
            parser.error("--mode render requires --input and --output")
        _run_render(cfg, args)
    else:
        _run_serve(cfg)


if __name__ == "__main__":
    main()
