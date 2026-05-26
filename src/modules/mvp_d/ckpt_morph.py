"""Checkpoint morphing utilities.

Interpolates between two or more PyTorch state_dicts to create a "frozen
auditory interpretation" hybrid. Shapes must match; mismatched keys are
ignored with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

MorphMode = Literal["linear", "slerp", "random_walk"]


@dataclass
class MorphParams:
    mode: MorphMode = "linear"
    t: float = 0.5                # interpolation coefficient for two-ckpt morph
    walk_step: float = 0.05       # step size for random_walk mode
    seed: int = 0


def interp_state_dicts(sd_a: dict, sd_b: dict, t: float, mode: MorphMode = "linear") -> dict:
    """Interpolate two state_dicts. Requires torch at call time."""
    import torch  # noqa: WPS433

    out: dict = {}
    shared = set(sd_a.keys()) & set(sd_b.keys())
    skipped = (set(sd_a.keys()) | set(sd_b.keys())) - shared
    if skipped:
        logger.warning("ckpt_morph: %d non-shared keys skipped", len(skipped))

    for k in shared:
        a, b = sd_a[k], sd_b[k]
        if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.shape != b.shape:
            out[k] = a
            continue
        if mode == "linear":
            out[k] = (1 - t) * a + t * b
        elif mode == "slerp":
            a_f = a.flatten().float()
            b_f = b.flatten().float()
            dot = torch.clamp(
                torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-9),
                -1.0,
                1.0,
            )
            omega = torch.acos(dot)
            if omega.abs() < 1e-6:
                out[k] = (1 - t) * a + t * b
            else:
                s = torch.sin(omega)
                merged = (torch.sin((1 - t) * omega) / s) * a_f + (torch.sin(t * omega) / s) * b_f
                out[k] = merged.view_as(a).to(a.dtype)
        else:
            raise ValueError(f"unsupported mode for two-ckpt interp: {mode}")
    return out


def random_walk_state_dict(base_sd: dict, params: MorphParams) -> dict:
    """Add Gaussian noise scaled by `walk_step` to every tensor in `base_sd`."""
    import torch  # noqa: WPS433

    gen = torch.Generator().manual_seed(params.seed)
    out: dict = {}
    for k, v in base_sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            noise = torch.randn(v.shape, generator=gen) * params.walk_step
            out[k] = v + noise.to(v.dtype)
        else:
            out[k] = v
    return out
