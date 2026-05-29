"""Latent musaicing: grain bank + cosine matching + assembly.

Reformulates granular synthesis as latent vector manipulation
(Kui et al. 2025, arXiv:2507.19202). Source corpus encoded once into
overlapping latent windows ("grains"). Target latent windowed and each
window matched to the grain bank via cosine similarity with softmax
temperature sampling. The selected grains are concatenated (optional
overlap-add crossfade) and decoded by the codec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GrainBank:
    embs: np.ndarray   # (N, dim, G) — full latent grains for decode assembly
    feats: np.ndarray  # (N, dim*G) — L2-normalised flatten for cosine match
    grain_size: int
    stride: int
    latent_dim: int

    @property
    def n_grains(self) -> int:
        return int(self.embs.shape[0])


@dataclass
class MusaicingParams:
    grain_size: int = 8          # latent frames; 8 @ 75Hz ≈ 107 ms
    stride: int = 4              # source corpus stride (overlap when < grain_size)
    target_stride: int = 8       # target window stride (non-overlap default)
    temperature: float = 0.1     # softmax τ; 0 = argmax, ∞ = uniform
    overlap_add: bool = True     # cosine crossfade at grain boundaries
    walk_strength: float = 0.0   # 0 = independent picks; >0 prefer adjacent corpus idx
    seed: int = 0


def _l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def build_grain_bank(corpus_embs: list[np.ndarray],
                     grain_size: int,
                     stride: int) -> GrainBank:
    """corpus_embs: list of (dim, T_lat) per source clip."""
    grains: list[np.ndarray] = []
    for emb in corpus_embs:
        dim, t = emb.shape
        if t < grain_size:
            continue
        for s in range(0, t - grain_size + 1, stride):
            grains.append(emb[:, s:s + grain_size])
    if not grains:
        raise ValueError("corpus too short for chosen grain_size")
    embs = np.stack(grains, axis=0).astype(np.float32)         # (N, dim, G)
    feats = embs.reshape(embs.shape[0], -1)
    feats = _l2norm(feats, axis=1).astype(np.float32)
    logger.info("grain bank: N=%d, dim=%d, G=%d, stride=%d",
                embs.shape[0], embs.shape[1], grain_size, stride)
    return GrainBank(embs=embs, feats=feats, grain_size=grain_size,
                     stride=stride, latent_dim=embs.shape[1])


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def match_target(target_emb: np.ndarray,
                 bank: GrainBank,
                 params: MusaicingParams) -> np.ndarray:
    """Return selected grain indices (one per target window)."""
    rng = np.random.default_rng(params.seed)
    G = bank.grain_size
    stride = params.target_stride
    dim, T = target_emb.shape

    last_idx: Optional[int] = None
    picks: list[int] = []
    for s in range(0, T - G + 1, stride):
        win = target_emb[:, s:s + G].reshape(-1)
        winn = win / (np.linalg.norm(win) + 1e-8)
        sims = bank.feats @ winn                              # (N,)
        if params.walk_strength > 0 and last_idx is not None:
            prior = np.exp(-((np.arange(bank.n_grains) - last_idx) ** 2)
                           / (2.0 * (bank.n_grains * 0.05) ** 2))
            sims = sims + params.walk_strength * prior
        if params.temperature <= 1e-6:
            idx = int(np.argmax(sims))
        else:
            probs = _softmax(sims / params.temperature)
            idx = int(rng.choice(bank.n_grains, p=probs))
        picks.append(idx)
        last_idx = idx
    return np.asarray(picks, dtype=np.int64)


def _cos_fade(n: int) -> np.ndarray:
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    t = np.linspace(0, np.pi, n, dtype=np.float32)
    return (0.5 * (1.0 - np.cos(t))).astype(np.float32)


def assemble_grains(bank: GrainBank,
                    indices: np.ndarray,
                    target_stride: int,
                    overlap_add: bool) -> np.ndarray:
    """Concatenate grains. With overlap_add, crossfade between grains by
    (grain_size - target_stride) frames using cosine ramp.
    """
    G = bank.grain_size
    n_picks = len(indices)
    if n_picks == 0:
        return np.zeros((bank.latent_dim, 0), dtype=np.float32)
    total = target_stride * (n_picks - 1) + G
    out = np.zeros((bank.latent_dim, total), dtype=np.float32)
    norm = np.zeros(total, dtype=np.float32)
    overlap = max(0, G - target_stride) if overlap_add else 0
    if overlap > 0:
        fade_in = _cos_fade(overlap)
        fade_out = fade_in[::-1]
    for k, gidx in enumerate(indices):
        s = k * target_stride
        g = bank.embs[gidx].copy()
        w = np.ones(G, dtype=np.float32)
        if overlap > 0:
            if k > 0:
                w[:overlap] *= fade_in
            if k < n_picks - 1:
                w[-overlap:] *= fade_out
        out[:, s:s + G] += g * w[None, :]
        norm[s:s + G] += w
    norm = np.maximum(norm, 1e-6)
    return (out / norm[None, :]).astype(np.float32)
