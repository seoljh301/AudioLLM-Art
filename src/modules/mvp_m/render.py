"""End-to-end latent musaicing render.

Pipeline:
    1. load codec
    2. encode corpus → continuous latents → grain bank
    3. encode target → continuous latent
    4. match target windows against bank (cosine + softmax τ)
    5. assemble grains (optional crossfade)
    6. decode assembled latent → audio
    7. (optional) blend with target dry signal

No model training is involved. Codec weights are pretrained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.modules.mvp_c.codec_io import load_audio_mono, save_audio_mono
from src.modules.mvp_m.codec_latent import (
    CodecLatentHandle, load_codec_latent, encode_latent, decode_latent,
)
from src.modules.mvp_m.musaicing import (
    GrainBank, MusaicingParams, build_grain_bank, match_target, assemble_grains,
)

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    codec: str = "encodec_24khz"
    device: str = "cuda"
    dry_wet: float = 1.0   # 1.0 = pure musaicing, 0.0 = passthrough target


def render_musaicing(corpus_paths: Sequence[Path],
                     target_path: Path,
                     out_path: Path,
                     params: MusaicingParams,
                     cfg: RenderConfig) -> dict:
    handle = load_codec_latent(cfg.codec, cfg.device)  # type: ignore[arg-type]
    sr = handle.sample_rate

    logger.info("encoding corpus: %d clips", len(corpus_paths))
    corpus_embs: list[np.ndarray] = []
    for p in corpus_paths:
        audio = load_audio_mono(Path(p), sr)
        if len(audio) < sr // 4:
            continue
        corpus_embs.append(encode_latent(handle, audio))
    bank = build_grain_bank(corpus_embs, params.grain_size, params.stride)

    logger.info("encoding target: %s", target_path)
    target_audio = load_audio_mono(Path(target_path), sr)
    target_emb = encode_latent(handle, target_audio)

    picks = match_target(target_emb, bank, params)
    logger.info("matched %d target windows, unique grains used=%d",
                len(picks), int(np.unique(picks).size))

    assembled = assemble_grains(bank, picks, params.target_stride,
                                params.overlap_add)
    out_audio = decode_latent(handle, assembled)

    if cfg.dry_wet < 1.0:
        n = min(len(out_audio), len(target_audio))
        out_audio = (cfg.dry_wet * out_audio[:n]
                     + (1.0 - cfg.dry_wet) * target_audio[:n])

    save_audio_mono(Path(out_path), out_audio, sr)
    logger.info("wrote %s (%.2fs)", out_path, len(out_audio) / sr)
    return {
        "n_grains": int(bank.n_grains),
        "n_picks": int(len(picks)),
        "unique_grains": int(np.unique(picks).size),
        "duration_s": float(len(out_audio) / sr),
        "sr": int(sr),
    }
