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


def _get_primes(n: int) -> np.ndarray:
    """Fast Sieve of Eratosthenes to get the first n primes."""
    if n <= 0: return np.array([], dtype=np.int64)
    
    # Estimation for the nth prime: n * (log n + log log n)
    # For n=306,000, max_val is roughly 4.5M.
    if n < 6:
        upper = 15
    else:
        upper = int(n * (np.log(n) + np.log(np.log(n))))
    
    sieve = np.ones(upper, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(np.sqrt(upper)) + 1):
        if sieve[i]:
            sieve[i*i : upper : i] = False
            
    primes = np.where(sieve)[0]
    return primes[:n]


def generate_tokens(n_q: int, params: OrganParams, rng: np.random.Generator) -> np.ndarray:
    """Generate (n_quantizers, time) token tensor."""
    t = params.duration_frames
    out = np.zeros((n_q, t), dtype=np.int64)
    
    if params.mode == "prime":
        # Fast generation of primes modulo 1024
        pattern = _get_primes(t) % 1024
        # If we didn't get enough primes (due to estimation), pad with random
        if len(pattern) < t:
            pad = rng.integers(0, 1024, size=t - len(pattern))
            pattern = np.concatenate([pattern, pad])
            
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
