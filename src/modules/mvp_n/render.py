"""End-to-end Concatenator render.

Pipeline:
    1. STFT corpus → magnitude dictionary W (M, N) + corpus phase (M, N)
    2. STFT target → magnitude V (M, T)
    3. For each target frame t:
         (a) PF transition + per-particle KL-NMF + likelihood + resample
         (b) Reconstruct magnitude V̂_t = Σ_k w_k (W[:, particles_k] h_k)
         (c) Pick phase from the dominant (particle, window) pair
    4. ISTFT (V̂, Φ̂) → output audio

Optional dry/wet blends the dry target signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.modules.mvp_c.codec_io import load_audio_mono, save_audio_mono
from src.modules.mvp_n.particle_filter import (
    PFParams, PFStats, init_particles, step,
)
from src.modules.mvp_n.stft import STFTParams, stft_mag_phase, istft_complex

logger = logging.getLogger(__name__)


@dataclass
class ConcatRenderConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop: int = 512
    win: int = 2048
    dry_wet: float = 1.0     # 1=pure musaicing
    log_every: int = 50


def _build_corpus_dict(corpus_paths: Sequence[Path],
                       sr: int, stft: STFTParams) -> tuple[np.ndarray, np.ndarray]:
    mags: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    for cp in corpus_paths:
        a = load_audio_mono(Path(cp), sr)
        if a.size < stft.n_fft:
            continue
        m, ph = stft_mag_phase(a, stft)
        mags.append(m)
        phases.append(ph)
    if not mags:
        raise ValueError("corpus produced no usable STFT frames")
    W = np.concatenate(mags, axis=1).astype(np.float32)        # (M, N)
    Phi = np.concatenate(phases, axis=1).astype(np.float32)
    return W, Phi


def render_concatenator(corpus_paths: Sequence[Path],
                        target_path: Path,
                        out_path: Path,
                        pf: PFParams,
                        cfg: ConcatRenderConfig) -> dict:
    stft = STFTParams(n_fft=cfg.n_fft, hop=cfg.hop, win=cfg.win)

    logger.info("encoding corpus: %d clips", len(corpus_paths))
    W, Phi = _build_corpus_dict(corpus_paths, cfg.sample_rate, stft)
    M, N = W.shape
    logger.info("corpus dictionary: M=%d freq bins, N=%d windows", M, N)

    target_audio = load_audio_mono(Path(target_path), cfg.sample_rate)
    V, _ = stft_mag_phase(target_audio, stft)
    T = V.shape[1]
    logger.info("target frames: T=%d", T)

    rng = np.random.default_rng(pf.seed)
    particles = init_particles(N, pf, rng)
    weights = np.full(pf.P, 1.0 / pf.P, dtype=np.float32)
    stats = PFStats()

    V_hat = np.zeros_like(V)
    Phi_hat = np.zeros_like(V)
    for t in range(T):
        v = V[:, t]
        particles, weights, h_all = step(W, v, particles, weights, pf, rng, stats)
        # Magnitude reconstruction (weighted over particles, summed over their p indices)
        recon = np.zeros(M, dtype=np.float32)
        # Track dominant (particle, idx-in-p) for phase pick
        best_score = -np.inf
        best_pk = 0
        best_pj = 0
        for k in range(pf.P):
            w_k = float(weights[k])
            if w_k <= 1e-9:
                continue
            W_sub = W[:, particles[k]]
            recon += w_k * (W_sub @ h_all[k])
            # update phase pick if this particle has the strongest contribution
            score = w_k * float(h_all[k].max())
            if score > best_score:
                best_score = score
                best_pk = k
                best_pj = int(np.argmax(h_all[k]))
        V_hat[:, t] = recon
        Phi_hat[:, t] = Phi[:, particles[best_pk, best_pj]]
        if (t + 1) % cfg.log_every == 0:
            logger.info("frame %d/%d  resamples=%d  ESS(mean)=%.1f",
                        t + 1, T, stats.n_resamples,
                        stats.mean_ess / max(1, stats.n_frames))

    out_audio = istft_complex(V_hat, Phi_hat, stft, length=len(target_audio))

    if cfg.dry_wet < 1.0:
        n = min(len(out_audio), len(target_audio))
        out_audio = (cfg.dry_wet * out_audio[:n]
                     + (1.0 - cfg.dry_wet) * target_audio[:n])

    save_audio_mono(Path(out_path), out_audio, cfg.sample_rate)
    logger.info("wrote %s  (%.2fs, %d frames)",
                out_path, len(out_audio) / cfg.sample_rate, T)
    return {
        "n_frames": int(T),
        "corpus_windows_N": int(N),
        "n_resamples": int(stats.n_resamples),
        "mean_ess": float(stats.mean_ess / max(1, stats.n_frames)),
        "sample_rate": int(cfg.sample_rate),
        "duration_s": float(len(out_audio) / cfg.sample_rate),
    }
