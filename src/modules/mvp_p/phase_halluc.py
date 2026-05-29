"""Phase hallucinator — synthesize audio from a random magnitude
spectrogram via Griffin-Lim phase reconstruction.

Input: ∅. Output: audio of given duration with chosen spectral envelope.
Magnitude shape (M, T) is generated procedurally (noise / shaped / chord)
and then iterated through Griffin-Lim to recover plausible phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ShapeMode = Literal["white", "pink", "shaped_smooth", "chord", "comb"]


@dataclass
class HallucParams:
    mode: ShapeMode = "shaped_smooth"
    duration_s: float = 10.0
    sr: int = 22050
    n_fft: int = 2048
    hop: int = 512
    griffin_lim_iters: int = 32
    smooth_t: float = 0.92        # time-axis IIR
    smooth_f: float = 0.85        # freq-axis smooth (1-pole)
    chord_freqs_hz: tuple[float, ...] = (110.0, 165.0, 220.0, 330.0)
    chord_bw_bins: int = 3
    seed: int = 0


def _bin_for_hz(f: float, sr: int, n_fft: int) -> int:
    return int(round(f * n_fft / sr))


def make_magnitude(p: HallucParams) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    M = p.n_fft // 2 + 1
    T = int(round(p.duration_s * p.sr / p.hop))
    if p.mode == "white":
        return rng.uniform(0.0, 1.0, size=(M, T)).astype(np.float32)

    if p.mode == "pink":
        base = rng.uniform(0.0, 1.0, size=(M, T)).astype(np.float32)
        falloff = 1.0 / np.sqrt(np.arange(1, M + 1, dtype=np.float32))
        return (base * falloff[:, None]).astype(np.float32)

    if p.mode == "shaped_smooth":
        noise = rng.uniform(0.0, 1.0, size=(M, T)).astype(np.float32)
        # Time-axis IIR
        out = np.zeros_like(noise)
        out[:, 0] = noise[:, 0]
        at = float(p.smooth_t)
        for t in range(1, T):
            out[:, t] = at * out[:, t - 1] + (1 - at) * noise[:, t]
        # Freq-axis smoothing (1-pole)
        af = float(p.smooth_f)
        for f in range(1, M):
            out[f] = af * out[f - 1] + (1 - af) * out[f]
        return out.astype(np.float32)

    if p.mode == "chord":
        mag = np.zeros((M, T), dtype=np.float32)
        for f0 in p.chord_freqs_hz:
            for harmonic in range(1, 9):
                b = _bin_for_hz(f0 * harmonic, p.sr, p.n_fft)
                if b >= M:
                    break
                amp = (1.0 / harmonic)
                lo = max(0, b - p.chord_bw_bins)
                hi = min(M, b + p.chord_bw_bins + 1)
                # Time-varying envelope per harmonic
                env = rng.uniform(0.5, 1.0)
                mod = 0.5 + 0.5 * np.sin(
                    np.linspace(0, env * np.pi * 4, T))
                mag[lo:hi] += amp * mod[None, :]
        return mag.astype(np.float32)

    if p.mode == "comb":
        mag = np.zeros((M, T), dtype=np.float32)
        spacing = max(4, M // 24)
        for b in range(spacing, M, spacing):
            wobble = 0.4 * np.sin(np.linspace(0, 4 * np.pi, T)
                                  + rng.uniform(0, 2 * np.pi))
            mag[b, :] = 1.0 + wobble
        # Spread each tooth
        mag = mag + 0.05 * rng.uniform(0, 1, size=mag.shape).astype(np.float32)
        return mag.astype(np.float32)

    raise ValueError(f"unknown shape mode: {p.mode}")


def griffin_lim(mag: np.ndarray, p: HallucParams) -> np.ndarray:
    import librosa  # noqa: WPS433
    rng = np.random.default_rng(p.seed + 17)
    phase = rng.uniform(-np.pi, np.pi,
                        size=mag.shape).astype(np.float32)
    S = mag * np.exp(1j * phase)
    for _ in range(p.griffin_lim_iters):
        audio = librosa.istft(S, hop_length=p.hop, win_length=p.n_fft,
                              center=True)
        S_new = librosa.stft(audio, n_fft=p.n_fft, hop_length=p.hop,
                             win_length=p.n_fft, center=True)
        phase_new = np.angle(S_new)
        S = mag * np.exp(1j * phase_new)
    audio = librosa.istft(S, hop_length=p.hop, win_length=p.n_fft,
                          center=True).astype(np.float32)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    return (audio / peak * 0.9).astype(np.float32)
