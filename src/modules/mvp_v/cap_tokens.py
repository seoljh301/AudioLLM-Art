"""Caption-Conditioned Tokens.

audio → caption → seed a deterministic token-walk in EnCodec's codebook
space → decode. The caption hash seeds (a) the starting codes per
quantizer and (b) the step pattern (prime / fibonacci / random_walk).

Closes the recursive loop in the discrete-token column without an LLM:
the caption acts as a *symbolic prior* over codebook trajectories.
Real captioner can be wired in via TextAudioBridge.use_real=True; the
procedural fallback still produces a caption that varies with audio
content.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.core.v2.text_audio import TextAudioBridge
from src.modules.mvp_a.rave_io import save_audio_mono
from src.modules.mvp_c.codec_io import (
    decode_tokens, encode_audio, load_audio_mono, load_codec,
)

logger = logging.getLogger(__name__)

WalkMode = Literal["prime", "fibonacci", "random_walk"]


@dataclass
class CapTokenParams:
    codec: str = "encodec_24khz"
    bandwidth: float = 6.0
    device: str = "cuda"
    chunk_seconds: float = 1.0      # re-caption every chunk
    walk_mode: WalkMode = "prime"
    base_seed_offset: int = 0
    walk_strength: int = 5          # max delta per token step
    dry_wet: float = 1.0
    use_real: bool = False          # CLAP zero-shot captioner


def _cap_seed(cap: str) -> int:
    return int.from_bytes(hashlib.md5(cap.encode()).digest()[:4], "little")


def _walk_sequence(start: int, n: int, mode: WalkMode,
                   strength: int, codebook: int,
                   rng: np.random.Generator) -> np.ndarray:
    seq = np.zeros(n, dtype=np.int64)
    seq[0] = int(start) % codebook
    if mode == "prime":
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29], dtype=np.int64)
        for i in range(1, n):
            step = int(primes[i % len(primes)]) * (1 if i % 2 == 0 else -1)
            seq[i] = (seq[i - 1] + step) % codebook
    elif mode == "fibonacci":
        a, b = 1, 1
        for i in range(1, n):
            step = a * (1 if i % 2 == 0 else -1)
            seq[i] = (seq[i - 1] + step) % codebook
            a, b = b, (a + b) % max(2, strength * 3)
    elif mode == "random_walk":
        for i in range(1, n):
            step = int(rng.integers(-strength, strength + 1))
            seq[i] = (seq[i - 1] + step) % codebook
    else:
        raise ValueError(f"unknown walk_mode: {mode}")
    return seq


def render_cap_tokens(target_path: Path,
                      out_path: Path,
                      params: CapTokenParams) -> dict:
    handle = load_codec(params.codec, params.device, params.bandwidth)  # type: ignore[arg-type]
    sr = handle.sample_rate
    audio = load_audio_mono(target_path, sr)
    tokens = encode_audio(handle, audio)                 # (n_q, T)
    n_q, T = tokens.shape
    codebook = int(handle.codebook_size)

    bridge = TextAudioBridge(use_real=params.use_real)
    bridge.init_defaults()

    # 75 fps token rate for EnCodec 24k
    fps = 75 if params.codec.startswith("encodec_24") else 150
    frames_per_chunk = max(1, int(round(params.chunk_seconds * fps)))

    new_tokens = tokens.copy()
    captions: list[str] = []
    for start in range(0, T, frames_per_chunk):
        end = min(T, start + frames_per_chunk)
        sample_start = int(start * sr / fps)
        sample_end = int(end * sr / fps)
        chunk_audio = audio[sample_start:sample_end]
        if chunk_audio.size == 0:
            continue
        cap = bridge.caption(chunk_audio, sr)
        captions.append(cap)
        seed = _cap_seed(cap) + params.base_seed_offset
        rng = np.random.default_rng(seed)
        for q in range(n_q):
            origin = int(rng.integers(0, codebook))
            seq = _walk_sequence(origin, end - start, params.walk_mode,
                                 params.walk_strength, codebook, rng)
            new_tokens[q, start:end] = seq

    out = decode_tokens(handle, new_tokens)
    if params.dry_wet < 1.0:
        n = min(len(out), len(audio))
        out = (params.dry_wet * out[:n]
               + (1.0 - params.dry_wet) * audio[:n]).astype(np.float32)
    save_audio_mono(out_path, out, sr)
    logger.info("wrote %s (%.2fs)  captions=%d  uniq=%d",
                out_path, len(out) / sr, len(captions),
                len(set(captions)))
    return {
        "duration_s": float(len(out) / sr),
        "sr": int(sr),
        "n_q": int(n_q),
        "n_captions": int(len(captions)),
        "unique_captions": int(len(set(captions))),
        "captions": captions,
        "walk_mode": str(params.walk_mode),
    }
