"""Chunked encode -> bend -> decode pipeline for MVP-C.

Splits long audio into windows, runs the codec round-trip on each window,
corrupting tokens between encode and decode, then concatenates the result.
A small overlap-and-fade is applied to mask chunk boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .codec_io import CodecHandle, decode_tokens, encode_audio
from .token_bend import BendParams, bend_tokens

from src.core.mix import MixConfig, dry_wet_mix
from src.core.texture_governor import TextureGovernorConfig, govern_wet

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    chunk_seconds: float = 4.0     # codec encode window
    overlap_seconds: float = 0.05  # equal-power crossfade between windows
    pad_to_chunk: bool = True      # zero-pad final chunk
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
    handle: CodecHandle,
    bend: BendParams,
    cfg: RenderConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run encode -> bend -> decode on `audio`, chunked, with crossfaded joins."""
    chunk_n = int(cfg.chunk_seconds * handle.sample_rate)
    overlap_n = int(cfg.overlap_seconds * handle.sample_rate)

    pieces: list[np.ndarray] = []
    diffs_total = 0
    for idx, chunk in enumerate(_chunk_iter(audio, chunk_n, cfg.pad_to_chunk)):
        tokens = encode_audio(handle, chunk)
        bent = bend_tokens(tokens, bend, rng)
        diffs_total += int((bent != tokens).sum())
        out = decode_tokens(handle, bent)

        # Apply guardrails and mixing
        decision = govern_wet(out, handle.sample_rate, cfg.mix.dry_wet, cfg.governor)
        out = dry_wet_mix(chunk, out, cfg.mix, override_wet=decision.wet, sample_rate=handle.sample_rate)

        if idx > 0 and overlap_n > 0 and len(pieces) > 0 and len(pieces[-1]) >= overlap_n:
            tail = pieces[-1][-overlap_n:]
            pieces[-1] = pieces[-1][:-overlap_n]
            out = _crossfade(tail, out)
        pieces.append(out.astype(np.float32))
        logger.info(
            "chunk %d: in=%d tokens, bent=%d diffs, wet=%.3f, flat=%.3f, rms=%.4f, guard=%s",
            idx,
            tokens.size,
            int((bent != tokens).sum()),
            decision.wet,
            decision.metrics.spectral_flatness,
            decision.metrics.rms,
            decision.reason,
        )

    rendered = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    rendered = rendered[: len(audio)]
    logger.info("render done: in=%d samp, out=%d samp, total token diffs=%d",
                len(audio), len(rendered), diffs_total)
    return rendered
