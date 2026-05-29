"""End-to-end MVP-O render."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.modules.mvp_a.rave_io import (
    RaveHandle, load_rave, decode, save_audio_mono,
)
from src.modules.mvp_o.latent_drift import DriftParams, draw_latent_trajectory

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    duration_s: float = 10.0
    latent_rate_hz: float = 93.75   # RAVE 48k @ 512 hop; probed at runtime if mismatch
    device: str = "cuda"


def _probe_latent_rate(handle: RaveHandle) -> float:
    dummy = np.zeros(handle.sample_rate, dtype=np.float32)
    import torch
    with torch.no_grad():
        w = torch.from_numpy(dummy).float().to(handle.device)[None, None, :]
        z = handle.model.encode(w)
    return float(z.shape[-1]) / 1.0  # 1 second probe → frames/sec


def render_drift(model_path: Path,
                 out_path: Path,
                 params: DriftParams,
                 cfg: RenderConfig) -> dict:
    handle = load_rave(model_path, cfg.device)
    rate_hz = _probe_latent_rate(handle)
    logger.info("probed latent_rate=%.2f Hz", rate_hz)
    n_frames = int(round(cfg.duration_s * rate_hz))
    z = draw_latent_trajectory(handle.latent_dim, n_frames, rate_hz, params)
    audio = decode(handle, z)
    save_audio_mono(Path(out_path), audio, handle.sample_rate)
    logger.info("wrote %s (%.2fs, %d frames)",
                out_path, len(audio) / handle.sample_rate, n_frames)
    return {
        "duration_s": float(len(audio) / handle.sample_rate),
        "n_latent_frames": int(n_frames),
        "latent_rate_hz": float(rate_hz),
        "sr": int(handle.sample_rate),
        "latent_dim": int(handle.latent_dim),
    }
