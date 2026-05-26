"""Small numpy-only audio metrics for detecting collapse/noise drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextureMetrics:
    rms: float
    spectral_flatness: float
    spectral_centroid_hz: float
    zero_crossing_rate: float
    has_nan: bool = False


def _frame_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    x = audio.astype(np.float32).reshape(-1)
    if len(x) < frame_size:
        x = np.pad(x, (0, frame_size - len(x)))
    n_frames = 1 + max(0, (len(x) - frame_size) // hop_size)
    frames = np.stack([x[i * hop_size : i * hop_size + frame_size] for i in range(n_frames)])
    return frames


def compute_texture_metrics(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = 2048,
    hop_size: int = 1024,
    eps: float = 1e-8,
) -> TextureMetrics:
    """Compute metrics that distinguish texture from noise/silence/NaNs."""
    x = audio.astype(np.float32).reshape(-1)
    if x.size == 0:
        return TextureMetrics(0.0, 0.0, 0.0, 0.0, False)

    has_nan = bool(np.isnan(x).any() or np.isinf(x).any())
    if has_nan:
        return TextureMetrics(0.0, 1.0, 0.0, 0.0, True)

    rms_val = float(np.sqrt(np.mean(x * x) + eps))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.float32))))) if len(x) > 1 else 0.0

    frames = _frame_audio(x, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32) + eps
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / float(sample_rate)).astype(np.float32)

    flatness = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    centroid = np.sum(spec * freqs[None, :], axis=1) / np.sum(spec, axis=1)

    return TextureMetrics(
        rms=rms_val,
        spectral_flatness=float(np.mean(flatness)),
        spectral_centroid_hz=float(np.mean(centroid)),
        zero_crossing_rate=zcr,
        has_nan=False,
    )
