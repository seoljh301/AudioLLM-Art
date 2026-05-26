"""Rendering pipeline for Latent Freeze with time-aware updates.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Iterator
import numpy as np

from .latent_freeze import FreezeParams, apply_freeze
from src.modules.mvp_a.rave_io import RaveHandle, decode, encode
from src.core.mix import MixConfig, dry_wet_mix
from src.core.texture_governor import TextureGovernorConfig, govern_wet

logger = logging.getLogger(__name__)

@dataclass
class RenderConfig:
    chunk_seconds: float = 4.0
    overlap_seconds: float = 0.05
    pad_to_chunk: bool = True
    mix: MixConfig = MixConfig()
    governor: TextureGovernorConfig = TextureGovernorConfig()

def _chunk_iter(audio: np.ndarray, chunk_n: int, pad: bool) -> Iterator[np.ndarray]:
    n = len(audio)
    for i in range(0, n, chunk_n):
        c = audio[i : i + chunk_n]
        if len(c) < chunk_n and pad:
            c = np.pad(c, (0, chunk_n - len(c)))
        yield c

def _crossfade(prev_tail: np.ndarray, next_head: np.ndarray) -> np.ndarray:
    n = min(len(prev_tail), len(next_head))
    if n == 0:
        return next_head
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = 1.0 - fade_in
    mixed = prev_tail[:n] * fade_out + next_head[:n] * fade_in
    return np.concatenate([mixed, next_head[n:]])

def render(
    audio: np.ndarray,
    handle: RaveHandle,
    params: FreezeParams,
    cfg: RenderConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """encode -> latent freeze -> decode, chunked."""
    chunk_n = int(cfg.chunk_seconds * handle.sample_rate)
    overlap_n = int(cfg.overlap_seconds * handle.sample_rate)
    
    pieces: list[np.ndarray] = []
    frame_idx_global = 0
    
    for idx, chunk in enumerate(_chunk_iter(audio, chunk_n, cfg.pad_to_chunk)):
        z = encode(handle, chunk)
        
        # Pass the global frame index to allow periodic updates
        z_frozen = apply_freeze(z, params, frame_idx_global)
        frame_idx_global += z.shape[-1]
        
        out = decode(handle, z_frozen)

        decision = govern_wet(out, handle.sample_rate, cfg.mix.dry_wet, cfg.governor)
        out = dry_wet_mix(chunk, out, cfg.mix, override_wet=decision.wet, sample_rate=handle.sample_rate)

        pieces.append(out.astype(np.float32))
        logger.info("chunk %d: freeze=%s, wet=%.3f, guard=%s", 
                    idx, params.freeze_active, decision.wet, decision.reason)

    rendered = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    rendered = rendered[: len(audio)]
    return rendered
