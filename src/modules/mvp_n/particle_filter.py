"""Particle filter for concatenative musaicing.

State: each of P particles holds `p` corpus-window indices.
Transition: with prob pd advance index by 1 (temporal continuity);
            with prob 1-pd jump to a uniform random corpus index.
Observation: per particle, solve KL-NMF activations h for the p columns
             of corpus dictionary W indexed by the particle; compute
             KL(v || W h). Likelihood = softmax(-KL / τ).
Resample: when effective sample size (ESS) drops below threshold,
          multinomial resample with replacement.

Tralie & Cantil 2024 (arXiv:2411.04366). Complexity O(LPMpT) — independent
of corpus size N.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.modules.mvp_n.kl_nmf import kl_nmf_with_l2, kl_divergence

logger = logging.getLogger(__name__)


@dataclass
class PFParams:
    P: int = 200          # particle count
    p: int = 5            # corpus windows per particle
    pd: float = 0.95      # prob of temporal continuity per index
    tau: float = 1.0      # softmax temperature on KL distance
    L: int = 10           # KL-NMF iterations
    ess_threshold: float = 0.5   # fraction of P for resampling
    l2_reg: float = 0.0
    seed: int = 0


@dataclass
class PFStats:
    n_resamples: int = 0
    mean_ess: float = 0.0
    n_frames: int = 0


def init_particles(N: int, params: PFParams,
                   rng: np.random.Generator) -> np.ndarray:
    """Returns particles (P, p) of corpus indices in [0, N)."""
    return rng.integers(0, N, size=(params.P, params.p), dtype=np.int64)


def transition(particles: np.ndarray, N: int, params: PFParams,
               rng: np.random.Generator) -> np.ndarray:
    advance_mask = rng.random(size=particles.shape) < params.pd
    advanced = (particles + 1) % N
    jumps = rng.integers(0, N, size=particles.shape, dtype=np.int64)
    return np.where(advance_mask, advanced, jumps)


def step(W: np.ndarray, v: np.ndarray, particles: np.ndarray,
         weights: np.ndarray, params: PFParams,
         rng: np.random.Generator, stats: PFStats
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single PF step.

    Returns:
        new_particles (P, p)
        new_weights (P,)
        h_all (P, p) — activations per particle (for output reconstruction)
    """
    N = W.shape[1]
    particles = transition(particles, N, params, rng)
    P, p = particles.shape
    dists = np.zeros(P, dtype=np.float32)
    h_all = np.zeros((P, p), dtype=np.float32)
    for k in range(P):
        W_sub = W[:, particles[k]]
        h = kl_nmf_with_l2(v, W_sub, L=params.L, l2=params.l2_reg)
        h_all[k] = h
        dists[k] = kl_divergence(v, W_sub @ h)

    # likelihood ∝ exp(-d / τ); shift for stability
    d_shift = dists - dists.min()
    lik = np.exp(-d_shift / max(params.tau, 1e-6))
    weights = weights * lik
    s = weights.sum()
    if s <= 0 or not np.isfinite(s):
        weights = np.full(P, 1.0 / P, dtype=np.float32)
    else:
        weights = (weights / s).astype(np.float32)

    ess = 1.0 / float(np.sum(weights ** 2) + 1e-12)
    stats.mean_ess += ess
    stats.n_frames += 1
    if ess < params.ess_threshold * P:
        idx = rng.choice(P, size=P, replace=True, p=weights)
        particles = particles[idx]
        h_all = h_all[idx]
        weights = np.full(P, 1.0 / P, dtype=np.float32)
        stats.n_resamples += 1
    return particles, weights, h_all
