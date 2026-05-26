"""EnCodec / DAC token-level corruption.

Operates on the discrete RVQ token tensor (n_quantizers, time). All ops are
numpy-only so the module is testable without a torch import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Mode = Literal["bit_flip", "quantizer_drop", "shuffle", "invalid_token"]


@dataclass
class BendParams:
    mode: Mode = "bit_flip"
    rate: float = 0.05            # fraction of positions affected
    quantizer_range: tuple[int, int] | None = None  # (lo, hi) of quantizers to corrupt
    codebook_size: int = 1024     # for bit-flip masking & invalid-token sentinel
    invalid_token: int = -1
    shuffle_window: int | None = None


def bend_tokens(tokens: np.ndarray, params: BendParams, rng: np.random.Generator) -> np.ndarray:
    """Corrupt RVQ token tensor of shape (n_q, time). Returns a new array."""
    out = tokens.copy()
    n_q, t = out.shape

    lo, hi = params.quantizer_range or (0, n_q)

    # Allow negative indices
    if lo < 0:
        lo = max(0, n_q + lo)
    if hi <= 0:
        hi = max(0, n_q + hi)

    lo = int(np.clip(lo, 0, n_q))
    hi = int(np.clip(hi, lo, n_q))

    if lo >= hi:
        return out

    # We work on a copy of the slice to be safe, then assign back
    sub = out[lo:hi].copy()

    if params.mode == "bit_flip":
        n = max(1, int(params.rate * sub.size))
        flat_idx = rng.choice(sub.size, size=min(n, sub.size), replace=False)
        bit_mask = 1 << rng.integers(0, max(1, int(np.log2(params.codebook_size))), size=n)
        flat = sub.ravel()
        flat[flat_idx] = (flat[flat_idx] ^ bit_mask) % params.codebook_size
        sub = flat.reshape(sub.shape)

    elif params.mode == "quantizer_drop":
        n_drop = max(1, int(params.rate * (hi - lo)))
        drop_idx = rng.choice(hi - lo, size=min(n_drop, hi - lo), replace=False)
        sub[drop_idx, :] = 0

    elif params.mode == "shuffle":
        if params.shuffle_window is not None and params.shuffle_window > 1:
            win = int(params.shuffle_window)
            for start in range(0, t, win):
                end = min(t, start + win)
                width = end - start
                if width <= 1:
                    continue
                n = max(1, int(params.rate * width))
                cols = rng.choice(width, size=min(n, width), replace=False)
                shuffled = cols.copy()
                rng.shuffle(shuffled)
                sub[:, start + cols] = sub[:, start + shuffled]
        else:
            n = max(1, int(params.rate * t))
            cols = rng.choice(t, size=min(n, t), replace=False)
            shuffled = cols.copy()
            rng.shuffle(shuffled)
            sub[:, cols] = sub[:, shuffled]

    elif params.mode == "invalid_token":
        n = max(1, int(params.rate * sub.size))
        flat_idx = rng.choice(sub.size, size=min(n, sub.size), replace=False)
        flat = sub.ravel()
        flat[flat_idx] = params.invalid_token
        sub = flat.reshape(sub.shape)

    else:
        raise ValueError(f"unknown bend mode: {params.mode}")

    out[lo:hi] = sub
    return out
