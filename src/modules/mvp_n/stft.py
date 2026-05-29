"""STFT helpers for The Concatenator.

Uses librosa for STFT/ISTFT to stay consistent with the rest of the
project. Returns magnitude + phase separately so the particle filter
can reconstruct from corpus phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STFTParams:
    n_fft: int = 2048
    hop: int = 512
    win: int = 2048


def stft_mag_phase(audio: np.ndarray, p: STFTParams) -> tuple[np.ndarray, np.ndarray]:
    import librosa  # noqa: WPS433
    S = librosa.stft(audio.astype(np.float32), n_fft=p.n_fft,
                     hop_length=p.hop, win_length=p.win, center=True)
    return np.abs(S).astype(np.float32), np.angle(S).astype(np.float32)


def istft_complex(mag: np.ndarray, phase: np.ndarray, p: STFTParams,
                  length: int | None = None) -> np.ndarray:
    import librosa  # noqa: WPS433
    S = mag * np.exp(1j * phase)
    return librosa.istft(S, hop_length=p.hop, win_length=p.win,
                         length=length).astype(np.float32)
