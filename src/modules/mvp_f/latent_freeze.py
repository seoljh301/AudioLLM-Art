"""Latent freeze primitives with smoothing.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class FreezeParams:
    freeze_indices: list[int] | None = None
    auto_upper_fraction: float = 0.5
    freeze_active: bool = True
    # New: Stochastic Updating
    update_interval_frames: int = 128
 # update the 'frozen' state every N latent frames
    crossfade_frames: int = 16        # smooth transition to new frozen state
    cached_state: np.ndarray | None = None
    target_state: np.ndarray | None = None
    fade_ptr: int = 0


def apply_freeze(z: np.ndarray, params: FreezeParams, frame_idx_global: int) -> np.ndarray:
    """Apply freezing logic with periodic updates and smoothing."""
    if not params.freeze_active:
        return z

    out = z.copy()
    ldim = z.shape[1] if z.ndim == 3 else z.shape[0]
    t = z.shape[-1]
    
    n_freeze = int(round(ldim * params.auto_upper_fraction))
    indices = list(range(ldim - n_freeze, ldim))

    # Initialize cache if missing
    if params.cached_state is None:
        params.cached_state = (z[:, indices, 0:1].copy() if z.ndim == 3 else z[indices, 0:1].copy())

    # Process each frame in the chunk
    for ti in range(t):
        curr_global = frame_idx_global + ti
        
        # Periodic check for update
        if curr_global > 0 and curr_global % params.update_interval_frames == 0:
            params.target_state = (z[:, indices, ti:ti+1].copy() if z.ndim == 3 else z[indices, ti:ti+1].copy())
            params.fade_ptr = params.crossfade_frames
            
        # Apply smoothing / crossfade if fading to a new target
        if params.fade_ptr > 0 and params.target_state is not None:
            # alpha goes from 1.0 (all old) to 0.0 (all new)
            alpha = params.fade_ptr / params.crossfade_frames
            state = (1.0 - alpha) * params.target_state + alpha * params.cached_state
            params.fade_ptr -= 1
            if params.fade_ptr == 0:
                params.cached_state = params.target_state
                params.target_state = None
        else:
            state = params.cached_state
            
        if z.ndim == 3:
            out[:, indices, ti:ti+1] = state
        else:
            out[indices, ti:ti+1] = state

    return out
