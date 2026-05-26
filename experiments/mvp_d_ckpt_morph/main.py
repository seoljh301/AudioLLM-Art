"""MVP-D entry point: checkpoint morphing.

Two modes:
  --mode render --input in.wav --output out.wav    # offline: morph once, render
  --mode serve                                       # OSC server, /mvp_d/t live remorph
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from src.core.mix import MixConfig
from src.core.texture_governor import TextureGovernorConfig
from src.core.osc_bridge import OSCBridge, OSCBridgeConfig
from src.modules.mvp_a.latent_perturb import PerturbParams
from src.modules.mvp_a.rave_io import load_audio_mono, save_audio_mono
from src.modules.mvp_a.render import RenderConfig, render
from src.modules.mvp_d.ckpt_morph import MorphParams
from src.modules.mvp_d.morph_io import load_morph, remorph_inplace


def _build_morph_params(cfg: dict) -> MorphParams:
    m = cfg["morph"]
    return MorphParams(
        mode=m.get("mode", "linear"),
        t=float(m.get("t", 0.5)),
        walk_step=float(m.get("walk_step", 0.05)),
        seed=cfg.get("seed", 0),
    )


def _build_perturb_params(cfg: dict) -> PerturbParams:
    p = cfg.get("perturb", {})
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
    params = _build_morph_params(cfg)
    morph = load_morph(
        cfg["checkpoints"]["paths"],
        params,
        device=cfg.get("device", "cpu"),
        rebasin=bool(cfg.get("rebasin", False)),
        rebasin_mode=str(cfg.get("rebasin_mode", "partial")),
    )

    # Optional MVP-A-style latent perturbation through the morphed model.
    pp = _build_perturb_params(cfg)
    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))

    audio = load_audio_mono(Path(args.input), target_sr=morph.handle.sample_rate)
    logging.info("loaded %d samp (%.2fs) from %s", len(audio),
                 len(audio) / morph.handle.sample_rate, args.input)
    out = render(audio, morph.handle, pp, rcfg, rng)
    save_audio_mono(Path(args.output), out, morph.handle.sample_rate)
    logging.info("wrote %s", args.output)


def _run_serve(cfg: dict) -> None:
    params = _build_morph_params(cfg)
    paths = [Path(p) for p in cfg["checkpoints"]["paths"]]
    device = cfg.get("device", "cpu")

    morph = load_morph(paths, params, device=device)
    # Cache source state_dicts so remorph is fast.
    dev = morph.handle.device
    source_sds = [torch.jit.load(str(p), map_location=dev).eval().state_dict() for p in paths]

    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))
    pp = _build_perturb_params(cfg)

    bridge = OSCBridge(OSCBridgeConfig(**cfg["osc"]))

    def on_t(_addr: str, val: float) -> None:
        params.t = float(val)
        remorph_inplace(morph, source_sds, params)
        logging.info("morph t -> %.4f (remorph done)", params.t)

    def on_mode(_addr: str, val: str) -> None:
        params.mode = str(val)  # type: ignore[assignment]
        remorph_inplace(morph, source_sds, params)
        logging.info("morph mode -> %s (remorph done)", params.mode)

    def on_render(_addr: str, in_path: str, out_path: str) -> None:
        logging.info("OSC render: %s -> %s (mode=%s, t=%.3f)",
                     in_path, out_path, params.mode, params.t)
        try:
            audio = load_audio_mono(Path(in_path), target_sr=morph.handle.sample_rate)
            out = render(audio, morph.handle, pp, rcfg, rng)
            save_audio_mono(Path(out_path), out, morph.handle.sample_rate)
            bridge.send("/mvp_d/done", out_path)
        except Exception as e:  # noqa: BLE001
            logging.exception("render failed")
            bridge.send("/mvp_d/error", repr(e))

    bridge.register("/mvp_d/t", on_t)
    bridge.register("/mvp_d/mode", on_mode)
    bridge.register("/mvp_d/render", on_render)
    logging.info("MVP-D serve ready on %s:%d",
                 cfg["osc"]["listen_host"], cfg["osc"]["listen_port"])
    bridge.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP-D: checkpoint morphing")
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
