"""End-to-end MVP-Y phantom weight render."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.modules.mvp_a.rave_io import decode, save_audio_mono
from src.modules.mvp_d.ckpt_morph import MorphParams
from src.modules.mvp_d.morph_io import load_morph
from src.modules.mvp_o.latent_drift import DriftParams, draw_latent_trajectory
from src.modules.mvp_y.phantom_weight import PhantomParams

logger = logging.getLogger(__name__)


def render_phantom(model_paths: Sequence[Path],
                   out_path: Path,
                   params: PhantomParams,
                   device: str = "cuda") -> dict:
    morph_params = MorphParams(mode=params.morph_mode, t=params.morph_t)
    morph = load_morph(model_paths, morph_params, device=device)
    handle = morph.handle

    # Probe latent rate.
    dummy = np.zeros(handle.sample_rate, dtype=np.float32)
    import torch
    with torch.no_grad():
        w = torch.from_numpy(dummy).float().to(handle.device)[None, None, :]
        z = handle.model.encode(w)
    rate_hz = float(z.shape[-1])

    n_frames = int(round(params.duration_s * rate_hz))
    drift = DriftParams(
        mode=params.latent_mode,
        sigma=params.latent_sigma,
        smooth=params.latent_smooth,
        seed=params.seed,
    )
    z_traj = draw_latent_trajectory(handle.latent_dim, n_frames, rate_hz, drift)
    audio = decode(handle, z_traj)

    save_audio_mono(Path(out_path), audio, handle.sample_rate)
    logger.info("wrote %s (%.2fs)  morph_t=%.3f  n_ckpts=%d",
                out_path, len(audio) / handle.sample_rate,
                params.morph_t, len(model_paths))
    return {
        "duration_s": float(len(audio) / handle.sample_rate),
        "sr": int(handle.sample_rate),
        "latent_dim": int(handle.latent_dim),
        "morph_t": float(params.morph_t),
        "n_ckpts": int(len(model_paths)),
    }
