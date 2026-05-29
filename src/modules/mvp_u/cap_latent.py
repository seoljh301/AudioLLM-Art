"""Caption-Steered Latent.

audio → caption → text embedding → project into RAVE latent space →
bias the latent trajectory by the projected vector → decode.

The "projection" is a deterministic linear map (random Gaussian with
fixed seed) from text-embedding dim to RAVE latent_dim. Each chunk of
the latent stream gets a constant bias added; mix-in strength is
controllable.

a → text → latent → audio. Closes the recursive loop in the latent
column.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.v2.text_audio import EMBED_DIM, TextAudioBridge
from src.modules.mvp_a.rave_io import (
    decode, encode, load_audio_mono, load_rave, save_audio_mono,
)

logger = logging.getLogger(__name__)


@dataclass
class CapLatentParams:
    bias_strength: float = 0.5    # latent += strength * proj(text_embed)
    chunk_seconds: float = 1.0    # how often to re-caption
    use_real: bool = False
    model_path: str = "checkpoints/rave/guitar_iil_b2048_r48000_z16.ts"
    device: str = "cuda"
    seed: int = 0
    dry_wet: float = 1.0


def _projection_matrix(latent_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 0xC0FFEE)
    W = rng.standard_normal((latent_dim, EMBED_DIM)).astype(np.float32)
    W /= np.sqrt(EMBED_DIM)
    return W


def render_cap_latent(target_path: Path,
                      out_path: Path,
                      params: CapLatentParams) -> dict:
    handle = load_rave(params.model_path, device=params.device)
    sr = handle.sample_rate
    audio = load_audio_mono(target_path, sr)
    z = encode(handle, audio)  # (D, T)
    latent_dim, T = z.shape

    # Probe latent_rate via 1s silence
    import torch
    with torch.no_grad():
        dummy = torch.zeros(1, 1, sr, device=handle.device)
        zp = handle.model.encode(dummy)
    rate_hz = float(zp.shape[-1])
    frames_per_chunk = max(1, int(round(params.chunk_seconds * rate_hz)))

    bridge = TextAudioBridge(use_real=params.use_real)
    bridge.init_defaults()
    W = _projection_matrix(latent_dim, params.seed)

    captions: list[str] = []
    z_biased = z.copy()
    for s in range(0, T, frames_per_chunk):
        e = min(T, s + frames_per_chunk)
        sample_start = int(s * sr / rate_hz)
        sample_end = int(e * sr / rate_hz)
        chunk_audio = audio[sample_start:sample_end]
        if chunk_audio.size == 0:
            continue
        cap = bridge.caption(chunk_audio, sr)
        captions.append(cap)
        emb = bridge.embed_text(cap)
        bias = (W @ emb).astype(np.float32)              # (latent_dim,)
        z_biased[:, s:e] = z_biased[:, s:e] + params.bias_strength * bias[:, None]

    out = decode(handle, z_biased)
    if params.dry_wet < 1.0:
        n = min(len(out), len(audio))
        out = (params.dry_wet * out[:n]
               + (1.0 - params.dry_wet) * audio[:n]).astype(np.float32)
    save_audio_mono(out_path, out, sr)
    logger.info("wrote %s (%.2fs)  captions=%d  uniq_caps=%d",
                out_path, len(out) / sr, len(captions),
                len(set(captions)))
    return {
        "duration_s": float(len(out) / sr),
        "sr": int(sr),
        "n_captions": int(len(captions)),
        "unique_captions": int(len(set(captions))),
        "captions": captions,
        "backend": "real" if params.use_real else "procedural",
    }
