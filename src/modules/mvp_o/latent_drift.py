"""Latent drift generator — pure RAVE decode without any input.

Sample a continuous latent trajectory z(t) ∈ R^{dim × T_z} via one of:
- iid normal (white)
- IIR-smoothed normal (Brownian / OU-like)
- low-frequency multi-band sinusoids per dimension

Then feed to RAVE decoder. Output = pure generative babble.
This is the input-less dual of MVP-A (which perturbs an encoded latent).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DriftMode = Literal["white", "smoothed", "sinusoid", "ou"]


@dataclass
class DriftParams:
    mode: DriftMode = "smoothed"
    sigma: float = 1.0       # white noise stddev
    smooth: float = 0.985    # IIR coef for "smoothed" / OU
    drift_rate: float = 0.0  # OU mean-reversion rate; 0 = pure smoothed walk
    base_freqs_hz: tuple[float, ...] = (0.1, 0.23, 0.41, 0.79)  # for sinusoid
    seed: int = 0


def draw_latent_trajectory(latent_dim: int,
                           n_frames: int,
                           latent_rate_hz: float,
                           params: DriftParams) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    if params.mode == "white":
        z = rng.normal(0.0, params.sigma,
                       size=(latent_dim, n_frames)).astype(np.float32)
        return z

    if params.mode == "smoothed":
        # IIR low-pass over white noise → Brownian-ish trajectory
        a = float(params.smooth)
        eps = rng.normal(0.0, params.sigma,
                         size=(latent_dim, n_frames)).astype(np.float32)
        z = np.zeros_like(eps)
        z[:, 0] = eps[:, 0]
        for t in range(1, n_frames):
            z[:, t] = a * z[:, t - 1] + (1.0 - a) * eps[:, t]
        return z

    if params.mode == "ou":
        # Ornstein-Uhlenbeck: dz = -theta z dt + sigma dW
        theta = float(params.drift_rate)
        sig = float(params.sigma)
        a = float(params.smooth)
        eps = rng.normal(0.0, sig,
                         size=(latent_dim, n_frames)).astype(np.float32)
        z = np.zeros_like(eps)
        for t in range(1, n_frames):
            z[:, t] = (1.0 - theta) * (a * z[:, t - 1]
                                       + (1.0 - a) * eps[:, t])
        return z

    if params.mode == "sinusoid":
        t = np.arange(n_frames, dtype=np.float32) / latent_rate_hz
        freqs = np.asarray(params.base_freqs_hz, dtype=np.float32)
        # one random (freq, phase) per latent dim drawn from base_freqs cycle
        choose = rng.integers(0, len(freqs), size=latent_dim)
        phases = rng.uniform(0.0, 2 * np.pi, size=latent_dim).astype(np.float32)
        amps = rng.uniform(0.5, 1.5, size=latent_dim).astype(np.float32)
        z = np.zeros((latent_dim, n_frames), dtype=np.float32)
        for d in range(latent_dim):
            w = 2 * np.pi * float(freqs[choose[d]])
            z[d] = amps[d] * np.sin(w * t + phases[d])
        return (params.sigma * z).astype(np.float32)

    raise ValueError(f"unknown drift mode: {params.mode}")
