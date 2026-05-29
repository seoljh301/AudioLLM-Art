"""Random Prompt TTA.

Generate audio from random text prompts sampled from a bank (or
permuted/combined). The output is whatever the TextToAudio backend
produces — real (AudioLDM) or procedural fallback.

Pairs naturally with:
- MVP-B (recursive audio↔text): same TTA engine, just driven by captions
- MVP-H (token generator): both are "pure invention" but at different
  abstraction layers (tokens vs. text → audio)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.core.v2.text_audio import CAPTION_BANK, TextAudioBridge

logger = logging.getLogger(__name__)


@dataclass
class PromptParams:
    duration_s: float = 10.0
    sr: int = 22050
    n_prompts: int = 1
    mix_mode: str = "concat"     # "concat" | "blend"
    custom_prompts: Sequence[str] = field(default_factory=tuple)
    permute_words: bool = False
    use_real: bool = False
    seed: int = 0


def _sample_prompts(p: PromptParams) -> list[str]:
    rng = np.random.default_rng(p.seed)
    if p.custom_prompts:
        prompts = list(p.custom_prompts)
        if p.n_prompts != len(prompts):
            idx = rng.integers(0, len(prompts), size=p.n_prompts)
            prompts = [prompts[i] for i in idx]
    else:
        idx = rng.integers(0, len(CAPTION_BANK), size=p.n_prompts)
        prompts = [CAPTION_BANK[i] for i in idx]
    if p.permute_words:
        out: list[str] = []
        for s in prompts:
            words = s.split()
            rng.shuffle(words)
            out.append(" ".join(words))
        prompts = out
    return prompts


def render_random_prompt(out_path,  # noqa: ANN001 - Path runtime
                         params: PromptParams) -> dict:
    from src.modules.mvp_a.rave_io import save_audio_mono  # noqa: WPS433

    bridge = TextAudioBridge(use_real=params.use_real)
    bridge.init_defaults()
    prompts = _sample_prompts(params)
    logger.info("prompts: %s", prompts)

    if params.mix_mode == "concat":
        seg_dur = params.duration_s / max(1, len(prompts))
        parts = [bridge.synth(p, seg_dur, params.sr) for p in prompts]
        audio = np.concatenate(parts).astype(np.float32)
    elif params.mix_mode == "blend":
        parts = [bridge.synth(p, params.duration_s, params.sr) for p in prompts]
        L = min(len(a) for a in parts)
        audio = np.zeros(L, dtype=np.float32)
        for a in parts:
            audio += a[:L]
        audio = audio / max(1, len(parts))
    else:
        raise ValueError(f"unknown mix_mode: {params.mix_mode}")

    peak = float(np.max(np.abs(audio)) + 1e-9)
    audio = (audio / peak * 0.9).astype(np.float32)
    save_audio_mono(out_path, audio, params.sr)
    logger.info("wrote %s (%.2fs, %d prompts)",
                out_path, len(audio) / params.sr, len(prompts))
    return {
        "duration_s": float(len(audio) / params.sr),
        "sr": int(params.sr),
        "prompts": prompts,
        "backend": "real" if params.use_real else "procedural",
    }
