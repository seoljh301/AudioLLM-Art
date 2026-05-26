"""MVP-C entry point: EnCodec/DAC token corruption.

Two modes:
  --mode render --input path.wav --output path.wav    # offline file -> bent file
  --mode serve                                         # OSC server: /mvp_c/render <in> <out>
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
from src.modules.mvp_c.codec_io import load_codec, load_audio_mono, save_audio_mono
from src.modules.mvp_c.render import RenderConfig, render
from src.modules.mvp_c.token_bend import BendParams


def _build_params(cfg: dict) -> BendParams:
    qr = cfg["bend"].get("quantizer_range")
    return BendParams(
        mode=cfg["bend"]["mode"],
        rate=float(cfg["bend"]["rate"]),
        quantizer_range=tuple(qr) if qr else None,
        codebook_size=int(cfg["bend"]["codebook_size"]),
        shuffle_window=cfg["bend"].get("shuffle_window"),
    )


def _finalize_params(params: BendParams, cfg: dict, handle) -> BendParams:
    auto_upper = cfg["bend"].get("auto_upper_fraction")

    if params.quantizer_range is None and auto_upper is not None:
        frac = float(auto_upper)
        lo = max(
            0,
            min(
                handle.n_quantizers - 1,
                int(round(handle.n_quantizers * (1.0 - frac))),
            ),
        )
        params.quantizer_range = (lo, handle.n_quantizers)

    elif params.quantizer_range is not None:
        lo, hi = params.quantizer_range

        if lo < 0:
            lo = handle.n_quantizers + lo
        if hi <= 0:
            hi = handle.n_quantizers + hi

        params.quantizer_range = (max(0, lo), min(handle.n_quantizers, hi))

    return params


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
    handle = load_codec(cfg["codec"]["name"],
                        device=cfg["codec"].get("device", "cuda"),
                        bandwidth=float(cfg["codec"].get("bandwidth", 6.0)))
    logging.info("codec=%s sr=%d n_q=%d codebook=%d",
                 handle.name, handle.sample_rate, handle.n_quantizers, handle.codebook_size)

    params = _build_params(cfg)
    params = _finalize_params(params, cfg, handle)
    params.codebook_size = handle.codebook_size

    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))

    audio = load_audio_mono(Path(args.input), target_sr=handle.sample_rate)
    logging.info("loaded %d samp (%.2fs) from %s", len(audio),
                 len(audio) / handle.sample_rate, args.input)

    bent = render(audio, handle, params, rcfg, rng)
    save_audio_mono(Path(args.output), bent, handle.sample_rate)
    logging.info("wrote %s", args.output)


def _run_serve(cfg: dict) -> None:
    handle = load_codec(cfg["codec"]["name"],
                        device=cfg["codec"].get("device", "cuda"),
                        bandwidth=float(cfg["codec"].get("bandwidth", 6.0)))
    params = _build_params(cfg)
    params = _finalize_params(params, cfg, handle)
    params.codebook_size = handle.codebook_size
    rcfg = _build_render_config(cfg)
    rng = np.random.default_rng(cfg.get("seed", 0))

    bridge = OSCBridge(OSCBridgeConfig(**cfg["osc"]))

    def on_rate(_addr: str, val: float) -> None:
        params.rate = float(val)
        logging.info("bend rate -> %.4f", params.rate)

    def on_mode(_addr: str, val: str) -> None:
        params.mode = str(val)  # type: ignore[assignment]
        logging.info("bend mode -> %s", params.mode)

    def on_render(_addr: str, in_path: str, out_path: str) -> None:
        logging.info("OSC render: %s -> %s (mode=%s, rate=%.3f)",
                     in_path, out_path, params.mode, params.rate)
        try:
            audio = load_audio_mono(Path(in_path), target_sr=handle.sample_rate)
            bent = render(audio, handle, params, rcfg, rng)
            save_audio_mono(Path(out_path), bent, handle.sample_rate)
            bridge.send("/mvp_c/done", out_path)
        except Exception as e:  # noqa: BLE001
            logging.exception("render failed")
            bridge.send("/mvp_c/error", repr(e))

    bridge.register("/mvp_c/rate", on_rate)
    bridge.register("/mvp_c/mode", on_mode)
    bridge.register("/mvp_c/render", on_render)
    logging.info("MVP-C serve ready on %s:%d", cfg["osc"]["listen_host"], cfg["osc"]["listen_port"])
    bridge.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP-C: EnCodec/DAC token bending")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--mode", choices=["render", "serve"], default="render")
    parser.add_argument("--input", type=str, default=None, help="input wav (render mode)")
    parser.add_argument("--output", type=str, default=None, help="output wav (render mode)")
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
