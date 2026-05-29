"""End-to-end MVP-R render."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.modules.mvp_c.codec_io import (
    load_codec, encode_audio, decode_tokens, load_audio_mono, save_audio_mono,
)
from src.modules.mvp_r.token_musaicing import (
    TokenMusaicParams, build_token_grain_bank, match_target_tokens,
    assemble_token_grains,
)

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    codec: str = "encodec_24khz"
    bandwidth: float = 6.0     # n_q for EnCodec 24k @ 6kbps = 8
    device: str = "cuda"
    dry_wet: float = 1.0


def render_token_musaicing(corpus_paths: Sequence[Path],
                           target_path: Path,
                           out_path: Path,
                           params: TokenMusaicParams,
                           cfg: RenderConfig) -> dict:
    handle = load_codec(cfg.codec, cfg.device, cfg.bandwidth)
    sr = handle.sample_rate

    logger.info("encoding corpus: %d clips", len(corpus_paths))
    corpus_tokens: list[np.ndarray] = []
    for p in corpus_paths:
        a = load_audio_mono(Path(p), sr)
        if a.size < sr // 4:
            continue
        corpus_tokens.append(encode_audio(handle, a))
    bank = build_token_grain_bank(corpus_tokens, params.grain_size, params.stride)

    target = load_audio_mono(Path(target_path), sr)
    target_tokens = encode_audio(handle, target)
    logger.info("target tokens: n_q=%d  T=%d", *target_tokens.shape)

    picks = match_target_tokens(target_tokens, bank, params)
    logger.info("matched %d windows  unique=%d / N=%d",
                len(picks), int(np.unique(picks).size), bank.n_grains)

    assembled = assemble_token_grains(bank, picks, params.target_stride)
    out = decode_tokens(handle, assembled)

    if cfg.dry_wet < 1.0:
        n = min(len(out), len(target))
        out = (cfg.dry_wet * out[:n]
               + (1.0 - cfg.dry_wet) * target[:n]).astype("float32")

    save_audio_mono(Path(out_path), out, sr)
    logger.info("wrote %s (%.2fs)", out_path, len(out) / sr)
    return {
        "n_grains": int(bank.n_grains),
        "n_picks": int(len(picks)),
        "unique_grains": int(np.unique(picks).size),
        "duration_s": float(len(out) / sr),
        "n_q": int(target_tokens.shape[0]),
        "sr": int(sr),
    }
