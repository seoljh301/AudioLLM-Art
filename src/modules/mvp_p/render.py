"""End-to-end MVP-P render."""

from __future__ import annotations

import logging
from pathlib import Path

from src.modules.mvp_a.rave_io import save_audio_mono
from src.modules.mvp_p.phase_halluc import HallucParams, make_magnitude, griffin_lim

logger = logging.getLogger(__name__)


def render_halluc(out_path: Path, params: HallucParams) -> dict:
    mag = make_magnitude(params)
    logger.info("magnitude shape: %s mode=%s", mag.shape, params.mode)
    audio = griffin_lim(mag, params)
    save_audio_mono(Path(out_path), audio, params.sr)
    logger.info("wrote %s (%.2fs)", out_path, len(audio) / params.sr)
    return {
        "duration_s": float(len(audio) / params.sr),
        "sr": int(params.sr),
        "n_fft": int(params.n_fft),
        "hop": int(params.hop),
        "gl_iters": int(params.griffin_lim_iters),
        "mode": str(params.mode),
    }
