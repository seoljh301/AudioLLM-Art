"""Generative Codebook Organ for EnCodec/DAC.

Produces audio from raw token sequences (prime numbers, Fibonacci, etc.)
without input audio.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class OrganParams:
    mode: str = "prime"           # prime | fibonacci | random_walk
    base_token: int = 0
    stride: int = 1
    duration_frames: int = 300    # length in tokens (approx 4s @ 75fps)


def generate_tokens(n_q: int, params: OrganParams, rng: np.random.Generator) -> np.ndarray:
    """Generate (n_quantizers, time) token tensor."""
    t = params.duration_frames
    out = np.zeros((n_q, t), dtype=np.int64)
    
    if params.mode == "prime":
        # Fill with prime numbers modulo 1024
        primes = []
        cand = 2
        while len(primes) < t:
            if all(cand % p != 0 for p in primes):
                primes.append(cand)
            cand += 1
        
        pattern = np.array(primes) % 1024
        for q in range(n_q):
            out[q] = np.roll(pattern, q * params.stride)

    elif params.mode == "fibonacci":
        fib = [0, 1]
        while len(fib) < t:
            fib.append((fib[-1] + fib[-2]) % 1024)
        
        pattern = np.array(fib)
        for q in range(n_q):
            out[q] = np.roll(pattern, q * params.stride)

    elif params.mode == "random_walk":
        for q in range(n_q):
            steps = rng.integers(-5, 6, size=t)
            walk = np.cumsum(steps) % 1024
            out[q] = walk
            
    else:
        # Default to a rhythmic pulse
        for q in range(n_q):
            out[q, ::params.stride] = params.base_token

    return out
