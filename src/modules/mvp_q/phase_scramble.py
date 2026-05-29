"""Phase scrambler — keep magnitude, mutate phase.

Classic spectral surgery: human pitch/timbre largely lives in magnitude;
phase carries transients and stereo cues. Scrambling phase produces
familiar timbral shape with smeared / glitched temporal structure.

Modes:
- random_uniform : phase ~ U(-π, π) (most destructive of transients)
- frame_swap     : permute time frames of the phase tensor
- bin_swap       : permute frequency bins of the phase tensor (per frame)
- rotate         : add a per-bin random phase offset (frequency-domain delay)
- ou             : smoothly varying phase via IIR — preserves some structure
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ScrambleMode = Literal["random_uniform", "frame_swap", "bin_swap",
                       "rotate", "ou"]


@dataclass
class ScrambleParams:
    mode: ScrambleMode = "random_uniform"
    rate: float = 1.0          # fraction [0,1] of bins/frames affected
    smooth: float = 0.92       # for OU phase
    rotate_max: float = np.pi  # for rotate mode
    seed: int = 0


def scramble_phase(phase: np.ndarray, params: ScrambleParams) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    M, T = phase.shape
    out = phase.copy()
    rate = float(np.clip(params.rate, 0.0, 1.0))

    if params.mode == "random_uniform":
        new = rng.uniform(-np.pi, np.pi, size=(M, T)).astype(np.float32)
        mask = rng.random(size=(M, T)) < rate
        out = np.where(mask, new, out)
        return out.astype(np.float32)

    if params.mode == "frame_swap":
        n_swap = max(1, int(rate * T))
        idx = rng.permutation(T)[:n_swap]
        target = rng.permutation(T)[:n_swap]
        out[:, target] = phase[:, idx]
        return out.astype(np.float32)

    if params.mode == "bin_swap":
        for t in range(T):
            if rng.random() > rate:
                continue
            perm = rng.permutation(M)
            out[:, t] = phase[perm, t]
        return out.astype(np.float32)

    if params.mode == "rotate":
        rmax = float(params.rotate_max)
        rot = rng.uniform(-rmax, rmax, size=M).astype(np.float32)
        # weight by rate (linear interp toward identity)
        out = (phase + rate * rot[:, None]).astype(np.float32)
        # wrap to (-π, π]
        out = np.angle(np.exp(1j * out)).astype(np.float32)
        return out

    if params.mode == "ou":
        a = float(params.smooth)
        eps = rng.normal(0.0, 1.0, size=(M, T)).astype(np.float32)
        new = np.zeros_like(eps)
        new[:, 0] = eps[:, 0] * np.pi
        for t in range(1, T):
            new[:, t] = a * new[:, t - 1] + (1 - a) * eps[:, t] * np.pi
        new = np.angle(np.exp(1j * new)).astype(np.float32)
        out = (1.0 - rate) * phase + rate * new
        out = np.angle(np.exp(1j * out)).astype(np.float32)
        return out

    raise ValueError(f"unknown scramble mode: {params.mode}")
