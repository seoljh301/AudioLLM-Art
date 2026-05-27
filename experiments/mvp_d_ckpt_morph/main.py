"""MVP-D entry point: checkpoint morphing.

Two modes:
  --mode render --input in.wav --output out.wav    # offline: morph once, render
  --mode serve                                       # OSC server, /mvp_d/t live remorph
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from src.core.mix import MixConfig
from src.core.texture_governor import TextureGovernorConfig
from src.core.osc_bridge import OSCBridge, OSCBridgeConfig
from src.modules.mvp_a.latent_perturb import PerturbParams
from src.modules.mvp_a.rave_io import load_audio_mono, save_audio_mono, encode, decode
from src.modules.mvp_a.latent_perturb import perturb_latent
from src.core.mix import dry_wet_mix
from src.core.texture_governor import govern_wet

# Custom render for MVP-D to fix logging namespace
logger = logging.getLogger(__name__)

@dataclass
class RenderConfig:
    chunk_seconds: float = 4.0
    overlap_seconds: float = 0.05
    pad_to_chunk: bool = True
    mix: MixConfig = MixConfig()
    governor: TextureGovernorConfig = TextureGovernorConfig()

from typing import Iterator
def _chunk_iter(audio: np.ndarray, chunk_n: int, pad: bool) -> Iterator[np.ndarray]:
    n = len(audio)
    for i in range(0, n, chunk_n):
        c = audio[i : i + chunk_n]
        if len(c) < chunk_n and pad:
            c = np.pad(c, (0, chunk_n - len(c)))
        yield c

def _crossfade(prev_tail: np.ndarray, next_head: np.ndarray) -> np.ndarray:
    n = min(len(prev_tail), len(next_head))
    if n == 0:
        return next_head
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = 1.0 - fade_in
    mixed = prev_tail[:n] * fade_out + next_head[:n] * fade_in
    return np.concatenate([mixed, next_head[n:]])

def render(
    audio: np.ndarray,
    handle,
    params: PerturbParams,
    cfg: RenderConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """encode -> perturb -> decode, chunked, with crossfaded joins."""
    chunk_n = int(cfg.chunk_seconds * handle.sample_rate)
    overlap_n = int(cfg.overlap_seconds * handle.sample_rate)

    pieces: list[np.ndarray] = []
    for idx, chunk in enumerate(_chunk_iter(audio, chunk_n, cfg.pad_to_chunk)):
        z = encode(handle, chunk)
        z_pert = perturb_latent(z, params, rng)
        out = decode(handle, z_pert)

        # Apply guardrails and mixing
        decision = govern_wet(out, handle.sample_rate, cfg.mix.dry_wet, cfg.governor)
        out = dry_wet_mix(chunk, out, cfg.mix, override_wet=decision.wet, sample_rate=handle.sample_rate)

        delta = float(np.sqrt(np.mean((z_pert - z) ** 2)))
        logger.info(
            "chunk %d: z=%s, delta_rms=%.4f, wet=%.3f, flat=%.3f, rms=%.4f, guard=%s",
            idx,
            z.shape,
            delta,
            decision.wet,
            decision.metrics.spectral_flatness,
            decision.metrics.rms,
            decision.reason,
        )

        if idx > 0 and overlap_n > 0 and len(pieces[-1]) >= overlap_n:
            tail = pieces[-1][-overlap_n:]
            pieces[-1] = pieces[-1][:-overlap_n]
            out = _crossfade(tail, out)
        pieces.append(out.astype(np.float32))

    rendered = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    rendered = rendered[: len(audio)]
    logger.info("render done: in=%d samp, out=%d samp", len(audio), len(rendered))
    return rendered

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
