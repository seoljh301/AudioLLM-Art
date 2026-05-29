"""Caption → TTA recursive loop.

Step 0: load target audio.
Step k (1..N):
    cap_k    = caption(audio_{k-1})
    audio_k  = TextToAudio(cap_k, dur)
    optionally mix_k = α·audio_k + (1-α)·audio_{k-1}

Audio drifts away from the original through iterated caption /
synthesis. Pure cousin of MVP-B (which mutates a tight caption loop
with text-domain edits); this version skips the text mutation step
and just observes the geometric drift.

Backend: TextAudioBridge — real (Qwen-Audio + AudioLDM) if available,
else procedural caption + Griffin-Lim TTA fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.v2.text_audio import TextAudioBridge
from src.modules.mvp_a.rave_io import save_audio_mono
from src.modules.mvp_c.codec_io import load_audio_mono

logger = logging.getLogger(__name__)


@dataclass
class LoopParams:
    n_iters: int = 4
    duration_per_step_s: float = 6.0
    keep_mix: float = 0.0          # 0=full replace; 1=keep original signal
    sr: int = 22050
    use_real: bool = False
    save_intermediates: bool = True


def render_cap_loop(target_path: Path,
                   out_dir: Path,
                   params: LoopParams) -> dict:
    bridge = TextAudioBridge(use_real=params.use_real)
    bridge.init_defaults()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = load_audio_mono(target_path, params.sr)
    cap_history: list[str] = []
    paths: list[str] = []
    for k in range(1, params.n_iters + 1):
        cap = bridge.caption(audio, params.sr)
        cap_history.append(cap)
        new = bridge.synth(cap, params.duration_per_step_s, params.sr)
        if params.keep_mix > 0.0:
            L = min(len(new), len(audio))
            new = (params.keep_mix * audio[:L]
                   + (1.0 - params.keep_mix) * new[:L]).astype(np.float32)
        audio = new
        if params.save_intermediates:
            p = out_dir / f"iter_{k:02d}.wav"
            save_audio_mono(p, audio, params.sr)
            paths.append(str(p))

    final_path = out_dir / "final.wav"
    save_audio_mono(final_path, audio, params.sr)
    paths.append(str(final_path))
    logger.info("loop done. captions=%s", cap_history)
    return {
        "n_iters": int(params.n_iters),
        "captions": cap_history,
        "outputs": paths,
        "sr": int(params.sr),
        "backend": "real" if params.use_real else "procedural",
    }
