"""Output-side guardrail for damage-based audio processors.

The governor does not try to make audio clean. It keeps the damaged signal from
becoming unusable pure noise, silence, or clipping by reducing wetness per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.texture_metrics import TextureMetrics, compute_texture_metrics


@dataclass
class TextureGovernorConfig:
    enabled: bool = False
    min_wet: float = 0.10
    flatness_max: float = 0.55
    rms_min: float = 1e-4
    rms_max: float = 0.85
    centroid_max_ratio: float = 0.42


@dataclass
class GovernorDecision:
    wet: float
    metrics: TextureMetrics
    reason: str


def govern_wet(
    damaged: np.ndarray,
    sample_rate: int,
    base_wet: float,
    cfg: TextureGovernorConfig,
) -> GovernorDecision:
    """Return a safer wet value for the current damaged chunk."""
    metrics = compute_texture_metrics(damaged, sample_rate)
    wet = float(np.clip(base_wet, 0.0, 1.0))
    reasons: list[str] = []

    if not cfg.enabled:
        return GovernorDecision(wet=wet, metrics=metrics, reason="disabled")

    if metrics.has_nan:
        # Emergency stop for this chunk
        return GovernorDecision(wet=0.0, metrics=metrics, reason="nan_emergency")

    if metrics.spectral_flatness > cfg.flatness_max:
        wet *= 0.55
        reasons.append("flatness")

    if metrics.rms < cfg.rms_min:
        wet *= 0.35
        reasons.append("silence")

    if metrics.rms > cfg.rms_max:
        wet *= 0.60
        reasons.append("rms_high")

    nyquist = sample_rate / 2.0
    if nyquist > 0 and metrics.spectral_centroid_hz / nyquist > cfg.centroid_max_ratio:
        wet *= 0.75
        reasons.append("centroid")

    wet = float(np.clip(wet, cfg.min_wet, base_wet))
    return GovernorDecision(wet=wet, metrics=metrics, reason="+".join(reasons) or "ok")
