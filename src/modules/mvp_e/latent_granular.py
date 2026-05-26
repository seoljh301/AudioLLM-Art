"""Latent granular primitives.

Operates on the continuous latent space (batch, latent_dim, time).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class GranularParams:
    grain_size: int = 16          # size in latent frames (approx 16 * 512 / 48000 = 170ms)
    memory_size: int = 2048       # max history in latent frames (approx 20s)
    num_grains: int = 4           # overlapping grains
    mix: float = 0.5              # 0=current only, 1=pure grains


class LatentMemory:
    def __init__(self, latent_dim: int, max_frames: int):
        self.max_frames = max_frames
        self.buffer = np.zeros((latent_dim, max_frames), dtype=np.float32)
        self.ptr = 0
        self.count = 0

    def push(self, z: np.ndarray):
        # z shape: (latent_dim, time)
        if z.ndim == 3: # handle (batch, dim, time)
            z = z[0]
            
        n = z.shape[-1]
        if n > self.max_frames:
            z = z[:, -self.max_frames:]
            n = self.max_frames
        
        if self.ptr + n <= self.max_frames:
            self.buffer[:, self.ptr:self.ptr + n] = z
        else:
            first_part = self.max_frames - self.ptr
            self.buffer[:, self.ptr:] = z[:, :first_part]
            self.buffer[:, :n - first_part] = z[:, first_part:]
        
        self.ptr = (self.ptr + n) % self.max_frames
        self.count = min(self.max_frames, self.count + n)

    def get_grains(self, n_out: int, params: GranularParams, rng: np.random.Generator) -> np.ndarray:
        if self.count < params.grain_size:
            return np.zeros((self.buffer.shape[0], n_out), dtype=np.float32)

        out = np.zeros((self.buffer.shape[0], n_out), dtype=np.float32)
        
        # Simple grain allocation: fill n_out samples using num_grains layers
        for _ in range(params.num_grains):
            # Number of grains needed to fill n_out
            num_steps = max(1, int(np.ceil(n_out / params.grain_size)))
            
            layer = []
            for _ in range(num_steps):
                # Pick a random starting point in valid history
                valid_range = self.count - params.grain_size
                s = rng.integers(0, valid_range)
                
                # Convert s to buffer index
                idx = (self.ptr - self.count + s) % self.max_frames
                
                if idx + params.grain_size <= self.max_frames:
                    g = self.buffer[:, idx:idx + params.grain_size]
                else:
                    first = self.max_frames - idx
                    g = np.concatenate([self.buffer[:, idx:], self.buffer[:, :params.grain_size - first]], axis=1)
                layer.append(g)
            
            stream = np.concatenate(layer, axis=1)[:, :n_out]
            out += stream
            
        return out / float(params.num_grains)


def apply_granular(z: np.ndarray, memory: LatentMemory, params: GranularParams, rng: np.random.Generator) -> np.ndarray:
    """Sample grains from memory and mix with current latent z."""
    squeeze = False
    if z.ndim == 3:
        z = z[0]
        squeeze = True
        
    ldim, t = z.shape
    
    # 1. Get grains from memory
    grains = memory.get_grains(t, params, rng)
    
    # 2. Mix
    out = (1.0 - params.mix) * z + params.mix * grains
    
    # 3. Update memory with the ORIGINAL current latent
    memory.push(z)
    
    if squeeze:
        out = out[None]
    return out.astype(np.float32)
