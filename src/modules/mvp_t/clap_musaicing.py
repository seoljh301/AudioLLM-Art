"""CLAP / text-driven musaicing.

Same musaicing skeleton as MVP-M (latent grain bank + cosine match)
but the matching key is a *text prompt sequence* rather than a target
audio's latents. Each grain is embedded by a text-audio bridge (real
CLAP if available, else procedural caption→hash). Text targets are
embedded the same way; cosine match selects grains in order.

Distinct from M (corpus×audio_target) and N (corpus×audio_target STFT);
this is corpus×text_target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from src.core.v2.text_audio import TextAudioBridge
from src.modules.mvp_a.rave_io import save_audio_mono
from src.modules.mvp_c.codec_io import load_audio_mono

logger = logging.getLogger(__name__)


@dataclass
class ClapMusaicParams:
    grain_seconds: float = 2.0       # corpus chunk size
    stride_seconds: float = 1.0      # corpus chunk stride
    seg_seconds: float = 4.0         # per-prompt output duration
    temperature: float = 0.2         # 0=argmax, ∞=uniform
    walk_strength: float = 0.0
    use_real: bool = False
    sr: int = 22050
    crossfade_seconds: float = 0.05
    seed: int = 0


@dataclass
class TextGrainBank:
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    embs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0),
                                                              dtype=np.float32))
    grain_size: int = 0

    @property
    def n_grains(self) -> int:
        return int(self.embs.shape[0])


def build_text_grain_bank(corpus_paths: Sequence[Path],
                          bridge: TextAudioBridge,
                          params: ClapMusaicParams) -> TextGrainBank:
    chunks: list[np.ndarray] = []
    embs: list[np.ndarray] = []
    G = int(round(params.grain_seconds * params.sr))
    S = int(round(params.stride_seconds * params.sr))
    for p in corpus_paths:
        a = load_audio_mono(Path(p), params.sr)
        if a.size < G:
            continue
        for start in range(0, a.size - G + 1, max(1, S)):
            seg = a[start:start + G].astype(np.float32)
            chunks.append(seg)
            embs.append(bridge.embed_audio(seg, params.sr))
    if not chunks:
        raise ValueError("corpus produced no grains")
    embs_np = np.stack(embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-9
    embs_np = (embs_np / norms).astype(np.float32)
    logger.info("text grain bank: N=%d  emb_dim=%d  G_sec=%.2f",
                embs_np.shape[0], embs_np.shape[1], params.grain_seconds)
    return TextGrainBank(audio_chunks=chunks, embs=embs_np, grain_size=G)


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 1e-6:
        out = np.zeros_like(x, dtype=np.float64)
        out[int(np.argmax(x))] = 1.0
        return out
    x = x / tau
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float64)


def render_clap_musaicing(corpus_paths: Sequence[Path],
                          prompts: Sequence[str],
                          out_path: Path,
                          params: ClapMusaicParams) -> dict:
    bridge = TextAudioBridge(use_real=params.use_real)
    bridge.init_defaults()
    bank = build_text_grain_bank(corpus_paths, bridge, params)

    rng = np.random.default_rng(params.seed)
    n_per_prompt = max(1, int(round(params.seg_seconds / params.grain_seconds)))
    picks: list[int] = []
    chosen_prompts: list[str] = []
    last_idx: int | None = None
    for prompt in prompts:
        t_emb = bridge.embed_text(prompt)
        t_emb = t_emb / (np.linalg.norm(t_emb) + 1e-9)
        for _ in range(n_per_prompt):
            sims = bank.embs @ t_emb
            if params.walk_strength > 0 and last_idx is not None:
                prior = np.exp(
                    -((np.arange(bank.n_grains) - last_idx) ** 2)
                    / (2.0 * (bank.n_grains * 0.05) ** 2)
                )
                sims = sims + params.walk_strength * prior.astype(np.float32)
            probs = _softmax(sims, params.temperature)
            probs = probs / probs.sum()
            idx = int(rng.choice(bank.n_grains, p=probs))
            picks.append(idx)
            chosen_prompts.append(prompt)
            last_idx = idx

    # Concatenate with cosine crossfade
    G = bank.grain_size
    xfade = int(round(params.crossfade_seconds * params.sr))
    xfade = max(0, min(xfade, G // 2))
    if xfade > 0:
        t = np.linspace(0.0, np.pi, xfade, dtype=np.float32)
        fade_in = (0.5 - 0.5 * np.cos(t)).astype(np.float32)
        fade_out = fade_in[::-1]
    total = G + (len(picks) - 1) * (G - xfade)
    out = np.zeros(total, dtype=np.float32)
    for k, idx in enumerate(picks):
        start = k * (G - xfade)
        seg = bank.audio_chunks[idx].copy()
        if xfade > 0 and k > 0:
            seg[:xfade] *= fade_in
            out[start:start + xfade] *= fade_out
        out[start:start + G] += seg
    peak = float(np.max(np.abs(out)) + 1e-9)
    out = (out / peak * 0.9).astype(np.float32)

    save_audio_mono(Path(out_path), out, params.sr)
    logger.info("wrote %s (%.2fs, picks=%d, unique=%d)",
                out_path, len(out) / params.sr, len(picks),
                int(np.unique(picks).size))
    return {
        "duration_s": float(len(out) / params.sr),
        "sr": int(params.sr),
        "n_grains": int(bank.n_grains),
        "n_picks": int(len(picks)),
        "unique": int(np.unique(picks).size),
        "prompts": list(prompts),
        "chosen_prompt_per_pick": chosen_prompts,
        "backend": "real" if params.use_real else "procedural",
    }
