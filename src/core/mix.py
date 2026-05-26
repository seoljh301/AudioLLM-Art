"""Dry/wet anchoring and safe output mixing.

These utilities keep damage-based processors near the input manifold instead of
letting them drift into pure noise, clipping, or silence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixConfig:
    dry_wet: float = 1.0          # 0=dry input, 1=fully damaged output
    rms_match: bool = False       # match damaged RMS to dry RMS before mixing
    limiter: bool = True          # soft-limit after mixing
    limiter_drive: float = 1.0    # >1 increases saturation before tanh
    low_anchor_hz: float | None = None  # If set, preserve dry frequencies below this Hz
    sub_boost_db: float = 0.0     # Extra gain (dB) applied specifically to the anchored low-end


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    """Return root-mean-square amplitude."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float32))) + eps))


def match_rms(reference: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale target so its RMS roughly matches reference RMS."""
    ref = rms(reference, eps=eps)
    tgt = rms(target, eps=eps)
    if tgt <= eps:
        return target.astype(np.float32)
    return (target * (ref / tgt)).astype(np.float32)


def soft_limiter(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    """Tanh limiter that prevents explosive damaged outputs."""
    drive = max(float(drive), 1e-6)
    y = np.tanh(x.astype(np.float32) * drive) / np.tanh(drive)
    return y.astype(np.float32)


def dry_wet_mix(
    dry: np.ndarray,
    wet_audio: np.ndarray,
    cfg: MixConfig,
    *,
    override_wet: float | None = None,
    sample_rate: int | None = None,
) -> np.ndarray:
    """Blend clean and damaged signals with optional RMS matching/limiting/low-anchoring."""
    n = min(len(dry), len(wet_audio))
    if n == 0:
        return wet_audio.astype(np.float32)

    dry_n = dry[:n].astype(np.float32)
    wet_n = wet_audio[:n].astype(np.float32)

    if cfg.rms_match:
        wet_n = match_rms(dry_n, wet_n)

    # Final safety against NaNs in the model output
    wet_n = np.nan_to_num(wet_n, nan=0.0, posinf=0.0, neginf=0.0)

    wet = cfg.dry_wet if override_wet is None else override_wet
    wet = float(np.clip(wet, 0.0, 1.0))
    mixed = (1.0 - wet) * dry_n + wet * wet_n

    # If low anchor is requested, supplement original low end with boost
    if cfg.low_anchor_hz is not None and sample_rate is not None:
        from scipy.signal import butter, lfilter
        # Using 2nd order for cleaner separation of subwoofer band
        b_low, a_low = butter(2, cfg.low_anchor_hz, fs=sample_rate, btype="low")
        b_high, a_high = butter(2, cfg.low_anchor_hz, fs=sample_rate, btype="high")
        
        low_end = lfilter(b_low, a_low, dry_n).astype(np.float32)
        
        # Apply sub-boost
        if cfg.sub_boost_db != 0:
            gain = 10 ** (cfg.sub_boost_db / 20.0)
            low_end *= gain
            
        mixed_hp = lfilter(b_high, a_high, mixed).astype(np.float32)
        mixed = mixed_hp + low_end

    if len(wet_audio) > n:
        mixed = np.concatenate([mixed, wet_audio[n:].astype(np.float32)])

    if cfg.limiter:
        mixed = soft_limiter(mixed, drive=cfg.limiter_drive)
    return mixed.astype(np.float32)
