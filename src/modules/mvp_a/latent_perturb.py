"""RAVE latent-space perturbation primitives.

The model itself (a pretrained RAVE `.ts` traced model) is expected to be
provided by the caller via the model registry. This module focuses on the
mathematical operations applied to the latent tensor `z` of shape
(batch, latent_dim, time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PerturbParams:
    noise_scale: float = 0.0
    dim_dropout: float = 0.0       # probability of zeroing a latent dimension
    dim_shuffle: bool = False
    bias_vector: np.ndarray | None = None  # shape (latent_dim,)
    freeze_mask: np.ndarray | None = None  # shape (latent_dim,), bool
    noise_mode: str = "white"      # white | smoothed
    noise_smooth: float = 0.98


def perturb_latent(z: np.ndarray, params: PerturbParams, rng: np.random.Generator) -> np.ndarray:
    """Apply perturbations to a latent tensor.

    Args:
        z: latent of shape (batch, latent_dim, time) or (latent_dim, time).
        params: perturbation parameters.
        rng: numpy RNG for reproducibility.

    Returns:
        perturbed latent of same shape as input.
    """
    squeeze = False
    if z.ndim == 2:
        z = z[None]
        squeeze = True

    out = z.copy()
    _, ldim, _ = out.shape

    if params.noise_scale > 0:
        noise = rng.normal(0, 1.0, size=out.shape).astype(out.dtype)

        if params.noise_mode == "smoothed":
            alpha = float(np.clip(params.noise_smooth, 0.0, 0.999))
            for ti in range(1, noise.shape[-1]):
                noise[:, :, ti] = alpha * noise[:, :, ti - 1] + (1.0 - alpha) * noise[:, :, ti]
            # re-normalize to unit std
            noise = noise / (np.std(noise) + 1e-6)

        elif params.noise_mode != "white":
            raise ValueError(f"unknown noise_mode: {params.noise_mode}")

        out = out + params.noise_scale * noise

    if params.dim_dropout > 0:
        mask = rng.random(ldim) < params.dim_dropout
        out[:, mask, :] = 0.0

    if params.dim_shuffle:
        perm = rng.permutation(ldim)
        out = out[:, perm, :]

    if params.bias_vector is not None:
        out = out + params.bias_vector[None, :, None]

    if params.freeze_mask is not None:
        out[:, params.freeze_mask, :] = z[:, params.freeze_mask, :]

    return out[0] if squeeze else out


def interp_latent(z_a: np.ndarray, z_b: np.ndarray, t: float, mode: str = "linear") -> np.ndarray:
    """Interpolate between two latent tensors. `mode` in {"linear", "slerp"}."""
    if mode == "linear":
        return (1 - t) * z_a + t * z_b
    if mode == "slerp":
        a_flat = z_a.reshape(-1)
        b_flat = z_b.reshape(-1)
        dot = np.clip(
            np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-9),
            -1.0,
            1.0,
        )
        omega = np.arccos(dot)
        if abs(omega) < 1e-6:
            return (1 - t) * z_a + t * z_b
        s = np.sin(omega)
        return (np.sin((1 - t) * omega) / s) * z_a + (np.sin(t * omega) / s) * z_b
    raise ValueError(f"unknown interp mode: {mode}")
