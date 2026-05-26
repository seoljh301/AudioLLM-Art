"""Neural Bass Massive primitives for EnCodec/DAC.

Manipulates the hierarchical RVQ structure to create 1. Temporal Smearing,
2. Codebook Jitter, and 3. Quantizer Folding in the low-frequency layers.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MassiveParams:
    # 1. Temporal Smearing
    smear_delay: int = 0          # Number of frames to delay coarse layers
    smear_quantizers: tuple[int, int] = (0, 2)
    
    # 2. Codebook Jitter
    jitter_rate: float = 0.0      # Probability of shifting index by +/- 1
    jitter_quantizers: tuple[int, int] = (0, 3)
    
    # 3. Quantizer Folding
    fold_leak_rate: float = 0.0    # Probability of leaking detail into coarse
    fold_source_range: tuple[int, int] = (8, 12) # Upper layers to leak
    fold_target_range: tuple[int, int] = (0, 2)  # Coarse layers to target


def apply_massive_ops(tokens: np.ndarray, params: MassiveParams, rng: np.random.Generator) -> np.ndarray:
    """Apply low-end focused NAC modulations."""
    out = tokens.copy()
    n_q, t = out.shape

    # 1. Temporal Smearing (Coarse Delay)
    if params.smear_delay > 0:
        lo, hi = params.smear_quantizers
        lo, hi = int(np.clip(lo, 0, n_q)), int(np.clip(hi, 0, n_q))
        if lo < hi:
            # Shift the coarse layers forward
            sub = out[lo:hi]
            shifted = np.roll(sub, params.smear_delay, axis=1)
            # Silence the wrap-around start if preferred, or just let it loop
            shifted[:, :params.smear_delay] = sub[:, 0:1] # zero-order hold initial state
            out[lo:hi] = shifted

    # 2. Codebook Jitter (Neural Saturation)
    if params.jitter_rate > 0:
        lo, hi = params.jitter_quantizers
        lo, hi = int(np.clip(lo, 0, n_q)), int(np.clip(hi, 0, n_q))
        if lo < hi:
            sub = out[lo:hi]
            mask = rng.random(sub.shape) < params.jitter_rate
            shifts = rng.choice([-1, 1], size=sub.shape)
            sub[mask] = (sub[mask] + shifts[mask]) % 1024 # assuming standard cb size
            out[lo:hi] = sub

    # 3. Quantizer Folding (Detail Leakage)
    if params.fold_leak_rate > 0:
        s_lo, s_hi = params.fold_source_range
        t_lo, t_hi = params.fold_target_range
        
        # Clip ranges
        s_lo, s_hi = int(np.clip(s_lo, 0, n_q)), int(np.clip(s_hi, 0, n_q))
        t_lo, t_hi = int(np.clip(t_lo, 0, n_q)), int(np.clip(t_hi, 0, n_q))
        
        if s_lo < s_hi and t_lo < t_hi:
            source_pool = out[s_lo:s_hi].flatten()
            target_sub = out[t_lo:t_hi]
            
            mask = rng.random(target_sub.shape) < params.fold_leak_rate
            n_replace = np.sum(mask)
            if n_replace > 0:
                replacements = rng.choice(source_pool, size=n_replace)
                target_sub[mask] = replacements
                out[t_lo:t_hi] = target_sub

    return out
