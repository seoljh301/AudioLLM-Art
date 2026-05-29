"""KL-divergence NMF activations.

Given a fixed dictionary W ∈ R^{M×p} and observation v ∈ R^M, solve
  min_h  KL(v || W h)
via Lee-Seung multiplicative updates. Used as the per-particle
observation model for The Concatenator (Tralie & Cantil 2024).
"""

from __future__ import annotations

import numpy as np


def kl_nmf_activations(v: np.ndarray, W: np.ndarray, L: int = 10,
                       eps: float = 1e-9) -> np.ndarray:
    """v: (M,), W: (M, p). Returns h: (p,) >= 0."""
    M, p = W.shape
    h = np.full(p, 1.0 / p, dtype=np.float32)
    ones = np.ones(M, dtype=np.float32)
    WT_one = W.T @ ones + eps
    for _ in range(L):
        Wh = W @ h + eps
        h = h * ((W.T @ (v / Wh)) / WT_one)
    return h.astype(np.float32)


def kl_divergence(v: np.ndarray, Wh: np.ndarray, eps: float = 1e-9) -> float:
    """Generalised KL D(v || Wh) = Σ (v log v/Wh − v + Wh)."""
    v_ = np.maximum(v, eps)
    Wh_ = np.maximum(Wh, eps)
    return float(np.sum(v_ * np.log(v_ / Wh_) - v_ + Wh_))


def kl_nmf_with_l2(v: np.ndarray, W: np.ndarray, L: int = 10,
                   l2: float = 0.0, eps: float = 1e-9) -> np.ndarray:
    """L2 regularised variant (Eq.10 of the paper) for quiet segments."""
    if l2 <= 0.0:
        return kl_nmf_activations(v, W, L=L, eps=eps)
    M, p = W.shape
    h = np.full(p, 1.0 / p, dtype=np.float32)
    ones = np.ones(M, dtype=np.float32)
    WT_one = W.T @ ones + eps
    for _ in range(L):
        Wh = W @ h + eps
        num = W.T @ (v / Wh)
        denom = WT_one + l2 * h
        h = h * (num / (denom + eps))
    return h.astype(np.float32)
