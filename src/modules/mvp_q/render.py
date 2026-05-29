"""End-to-end MVP-Q render."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.modules.mvp_c.codec_io import load_audio_mono, save_audio_mono
from src.modules.mvp_n.stft import STFTParams, stft_mag_phase, istft_complex
from src.modules.mvp_q.phase_scramble import ScrambleParams, scramble_phase

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop: int = 512
    win: int = 2048
    dry_wet: float = 1.0


def render_scramble(target_path: Path,
                    out_path: Path,
                    params: ScrambleParams,
                    cfg: RenderConfig) -> dict:
    target = load_audio_mono(Path(target_path), cfg.sample_rate)
    stft = STFTParams(n_fft=cfg.n_fft, hop=cfg.hop, win=cfg.win)
    mag, phase = stft_mag_phase(target, stft)
    new_phase = scramble_phase(phase, params)
    out = istft_complex(mag, new_phase, stft, length=len(target))
    if cfg.dry_wet < 1.0:
        n = min(len(out), len(target))
        out = (cfg.dry_wet * out[:n]
               + (1.0 - cfg.dry_wet) * target[:n]).astype("float32")
    save_audio_mono(Path(out_path), out, cfg.sample_rate)
    logger.info("wrote %s mode=%s rate=%.2f", out_path,
                params.mode, params.rate)
    return {
        "duration_s": float(len(out) / cfg.sample_rate),
        "sr": int(cfg.sample_rate),
        "frames": int(mag.shape[1]),
        "freq_bins": int(mag.shape[0]),
        "mode": str(params.mode),
        "rate": float(params.rate),
    }
