"""Caption + text-to-audio backends for MVP-B.

Two backends each:

* caption: "stub" (deterministic spectral-stats text) | "qwen2_audio" (HF Qwen2-Audio)
* tta:     "stub" (FM synth seeded by text hash)    | "audioldm2"  (HF AudioLDM2)

Stubs are fully local, no downloads — they produce non-trivial outputs that
verify the recursive loop's control flow and OSC plumbing. Real backends are
behind explicit config flags and require the user to authorize the model
downloads (~14 GB for Qwen2-Audio, ~4 GB for AudioLDM2).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

CaptionFn = Callable[[np.ndarray], str]
SynthFn = Callable[[str], np.ndarray]


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_VOCAB_TIMBRE = ["bright", "dark", "warm", "metallic", "wooden", "glassy", "vocal", "harsh"]
_VOCAB_MOTION = ["pulsing", "drifting", "stuttering", "swelling", "decaying", "shimmering"]
_VOCAB_PLACE = ["underwater", "in a cave", "through a wall", "in fog", "in static", "from below"]


def _spectral_centroid_hz(audio: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    s = spec.sum()
    return float((spec * freqs).sum() / s) if s > 0 else 0.0


def _zero_crossing_rate(audio: np.ndarray) -> float:
    return float(np.mean(np.diff(np.sign(audio)) != 0))


def make_stub_caption(sample_rate: int) -> CaptionFn:
    """Return a deterministic caption_fn that summarizes audio statistics."""

    def caption_fn(audio: np.ndarray) -> str:
        if audio.size == 0:
            return "silence"
        rms = float(np.sqrt(np.mean(audio**2)))
        sc = _spectral_centroid_hz(audio, sample_rate)
        zcr = _zero_crossing_rate(audio)
        timbre = _VOCAB_TIMBRE[int(sc / 500) % len(_VOCAB_TIMBRE)]
        motion = _VOCAB_MOTION[int(zcr * 100) % len(_VOCAB_MOTION)]
        place = _VOCAB_PLACE[int(rms * 50) % len(_VOCAB_PLACE)]
        return f"a {timbre} {motion} sound {place}"

    return caption_fn


def make_stub_synth(sample_rate: int, duration_s: float) -> SynthFn:
    """Return a deterministic synth_fn that maps text -> FM-modulated audio."""

    n_samples = int(sample_rate * duration_s)

    def synth_fn(text: str) -> np.ndarray:
        h = hashlib.md5(text.encode("utf-8")).digest()
        # Derive params deterministically from hash bytes.
        f_carrier = 110.0 + (h[0] * 4)              # 110..1130 Hz
        f_mod = 1.0 + (h[1] * 0.5)                  # 1..128 Hz
        mod_idx = 0.5 + (h[2] / 64.0)               # 0.5..4.5
        noise_amp = h[3] / 4096.0                   # 0..0.0625
        amp_env_freq = 0.5 + (h[4] * 0.05)          # 0.5..13 Hz

        t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
        env = 0.5 * (1.0 + np.sin(2 * np.pi * amp_env_freq * t))
        carrier = np.sin(2 * np.pi * f_carrier * t + mod_idx * np.sin(2 * np.pi * f_mod * t))
        noise = np.random.default_rng(int.from_bytes(h[:4], "little")).standard_normal(n_samples)
        sig = (env * carrier + noise_amp * noise).astype(np.float32)
        return (sig / max(1e-6, float(np.max(np.abs(sig)))) * 0.9).astype(np.float32)

    return synth_fn


# ---------------------------------------------------------------------------
# Real backends (lazy import — only loaded if explicitly selected)
# ---------------------------------------------------------------------------

@dataclass
class Qwen2AudioConfig:
    model_id: str = "Qwen/Qwen2-Audio-7B-Instruct"
    prompt: str = "Describe the sound in one short sentence."
    max_new_tokens: int = 48
    device: str = "cpu"


def make_qwen2_audio_caption(cfg: Qwen2AudioConfig, sample_rate: int) -> CaptionFn:
    """Real caption_fn using Qwen2-Audio. ~14 GB download; user must authorize."""
    import torch  # noqa: WPS433
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration  # noqa: WPS433

    logger.info("loading Qwen2-Audio: %s on %s", cfg.model_id, cfg.device)
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg.model_id, torch_dtype=torch.float32,
    ).to(cfg.device).eval()

    def caption_fn(audio: np.ndarray) -> str:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "in-memory"},
            {"type": "text", "text": cfg.prompt},
        ]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=[audio], sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()

    return caption_fn


@dataclass
class AudioLDM2Config:
    model_id: str = "cvssp/audioldm2"
    duration_s: float = 6.0
    guidance_scale: float = 3.5
    num_inference_steps: int = 50
    device: str = "cpu"


def make_audioldm2_synth(cfg: AudioLDM2Config) -> SynthFn:
    """Real synth_fn using AudioLDM2. ~4 GB download; user must authorize."""
    import torch  # noqa: WPS433
    from diffusers import AudioLDM2Pipeline  # noqa: WPS433

    logger.info("loading AudioLDM2: %s on %s", cfg.model_id, cfg.device)
    pipe = AudioLDM2Pipeline.from_pretrained(cfg.model_id, torch_dtype=torch.float32).to(cfg.device)

    def synth_fn(text: str) -> np.ndarray:
        with torch.no_grad():
            out = pipe(
                prompt=text,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                audio_length_in_s=cfg.duration_s,
            )
        return out.audios[0].astype(np.float32)

    return synth_fn
