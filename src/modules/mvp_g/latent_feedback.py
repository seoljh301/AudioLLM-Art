"""Latent-space recursive feedback (Echo/Loop) for RAVE.

Implements z_out = z_current + z_delayed * feedback.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class FeedbackParams:
    delay_frames: int = 32         # delay in latent frames (~350ms)
    feedback: float = 0.4          # feedback amount (0 to <1)
    mix: float = 0.5               # latent mix (0=dry z, 1=fully processed z)


class LatentFeedbackBuffer:
    def __init__(self, latent_dim: int, max_delay: int):
        self.buffer = np.zeros((latent_dim, max_delay), dtype=np.float32)
        self.ptr = 0
        self.max_delay = max_delay

    def process(self, z: np.ndarray, params: FeedbackParams) -> np.ndarray:
        # z shape: (latent_dim, time)
        ldim, t = z.shape
        out = np.zeros_like(z)
        
        for ti in range(t):
            # 1. Read from delay line
            delayed = self.buffer[:, self.ptr]
            
            # 2. Calculate output frame
            # Processed frame includes feedback
            processed = z[:, ti] + delayed * params.feedback
            
            # 3. Write back to buffer (recursive)
            self.buffer[:, self.ptr] = processed
            
            # 4. Mix for output
            out[:, ti] = (1.0 - params.mix) * z[:, ti] + params.mix * processed
            
            # Update pointer
            self.ptr = (self.ptr + 1) % self.max_delay
            
        return out


def apply_feedback(z: np.ndarray, buffer: LatentFeedbackBuffer, params: FeedbackParams) -> np.ndarray:
    squeeze = False
    if z.ndim == 3:
        z = z[0]
        squeeze = True
        
    out = buffer.process(z, params)
    
    if squeeze:
        out = out[None]
    return out.astype(np.float32)
