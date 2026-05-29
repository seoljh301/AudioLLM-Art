"""Token-domain musaicing.

EnCodec / DAC produce discrete code tensors (n_q, T). This module builds
a "token grain bank" from a source corpus and matches a target token
sequence window-by-window via a discrete distance metric
(Hamming or weighted-quantizer match). Selected grains are concatenated
in token space and decoded once at the end.

Token dual of MVP-M (continuous latent musaicing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

DistMode = Literal["hamming", "weighted_hamming", "first_q"]


@dataclass
class TokenGrainBank:
    grains: np.ndarray   # (N_grains, n_q, G) int64
    n_q: int
    grain_size: int
    stride: int

    @property
    def n_grains(self) -> int:
        return int(self.grains.shape[0])


@dataclass
class TokenMusaicParams:
    grain_size: int = 8        # token frames per grain
    stride: int = 4            # corpus stride
    target_stride: int = 8
    dist: DistMode = "weighted_hamming"
    q_weights: tuple[float, ...] = (1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.08)
    temperature: float = 0.5   # 0=argmin, ∞=uniform; softmax(-dist/τ)
    walk_strength: float = 0.0
    seed: int = 0


def build_token_grain_bank(corpus_tokens: list[np.ndarray],
                           grain_size: int,
                           stride: int) -> TokenGrainBank:
    """corpus_tokens: list of (n_q, T) int arrays. Concatenate grains."""
    grains: list[np.ndarray] = []
    n_q = corpus_tokens[0].shape[0]
    for c in corpus_tokens:
        if c.shape[0] != n_q:
            raise ValueError("all corpus clips must share n_q")
        T = c.shape[1]
        if T < grain_size:
            continue
        for s in range(0, T - grain_size + 1, stride):
            grains.append(c[:, s:s + grain_size])
    if not grains:
        raise ValueError("corpus too short for grain_size")
    arr = np.stack(grains, axis=0).astype(np.int64)
    logger.info("token grain bank: N=%d  n_q=%d  G=%d  stride=%d",
                arr.shape[0], n_q, grain_size, stride)
    return TokenGrainBank(grains=arr, n_q=n_q,
                          grain_size=grain_size, stride=stride)


def _softmax_neg(d: np.ndarray, tau: float) -> np.ndarray:
    """softmax(-d / tau)."""
    if tau <= 1e-6:
        out = np.zeros_like(d, dtype=np.float64)
        out[int(np.argmin(d))] = 1.0
        return out
    x = -d / tau
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float64)


def match_target_tokens(target: np.ndarray,
                        bank: TokenGrainBank,
                        params: TokenMusaicParams) -> np.ndarray:
    rng = np.random.default_rng(params.seed)
    n_q, T = target.shape
    G = bank.grain_size
    stride = params.target_stride
    if n_q != bank.n_q:
        raise ValueError(f"target n_q={n_q} != bank n_q={bank.n_q}")
    weights = np.asarray(params.q_weights[:n_q], dtype=np.float32)
    if weights.size < n_q:
        weights = np.concatenate([weights, np.full(n_q - weights.size, 0.05)])

    picks: list[int] = []
    last_idx = None
    for s in range(0, T - G + 1, stride):
        win = target[:, s:s + G]  # (n_q, G)
        # diff: per-position not-equal
        diff = (bank.grains != win[None, :, :]).astype(np.float32)
        if params.dist == "hamming":
            d = diff.sum(axis=(1, 2))
        elif params.dist == "weighted_hamming":
            d = (diff * weights[None, :, None]).sum(axis=(1, 2))
        elif params.dist == "first_q":
            d = diff[:, 0, :].sum(axis=1)
        else:
            raise ValueError(f"unknown dist: {params.dist}")

        if params.walk_strength > 0 and last_idx is not None:
            prior = np.exp(-((np.arange(bank.n_grains) - last_idx) ** 2)
                           / (2.0 * (bank.n_grains * 0.05) ** 2))
            d = d - params.walk_strength * prior * d.max()

        probs = _softmax_neg(d, params.temperature)
        # Renormalise to defend against float rounding in numpy.choice.
        probs = probs / probs.sum()
        idx = int(rng.choice(bank.n_grains, p=probs))
        picks.append(idx)
        last_idx = idx
    return np.asarray(picks, dtype=np.int64)


def assemble_token_grains(bank: TokenGrainBank,
                          indices: np.ndarray,
                          target_stride: int) -> np.ndarray:
    """Concatenate grains in token space. Overlap is resolved by
    majority vote per (quantizer, frame).
    """
    G = bank.grain_size
    n_picks = len(indices)
    if n_picks == 0:
        return np.zeros((bank.n_q, 0), dtype=np.int64)
    total = target_stride * (n_picks - 1) + G
    # voting buckets — collect candidates per cell
    votes: list[list[list[int]]] = [
        [[] for _ in range(total)] for _ in range(bank.n_q)
    ]
    for k, gidx in enumerate(indices):
        g = bank.grains[gidx]
        start = k * target_stride
        for q in range(bank.n_q):
            for t in range(G):
                votes[q][start + t].append(int(g[q, t]))
    out = np.zeros((bank.n_q, total), dtype=np.int64)
    for q in range(bank.n_q):
        for t in range(total):
            v = votes[q][t]
            if not v:
                out[q, t] = 0
            else:
                # majority vote; tie → first
                vals, counts = np.unique(v, return_counts=True)
                out[q, t] = int(vals[counts.argmax()])
    return out
