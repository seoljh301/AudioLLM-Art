"""V2 text-audio bridge interfaces.

Three pluggable services:
- Captioner       audio (np.ndarray) -> str
- TextEmbedder    str -> np.ndarray (embedding)
- TextToAudio     str -> np.ndarray (audio waveform)

Each ships with a deterministic procedural fallback that runs without any
HuggingFace download. The fallback is intentionally simple but consistent
so V2 MVPs can be wired and tested end-to-end before plugging in real
models (CLAP, Qwen-Audio, AudioLDM, etc.).

Real-model adapters live alongside as optional functions
(``try_load_clap``, ``try_load_audioldm``, ``try_load_qwen_audio``); they
return None if the dependency or weight is not available — the caller
falls back to the procedural stub.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 512


# ---------- procedural fallback dictionaries ----------

CAPTION_BANK = [
    # spectral / tonal
    "bright shimmer with metallic resonance",
    "deep cavernous drone",
    "warm pulsing organ",
    "crystal bell choir",
    "underwater hum",
    "warm string ensemble",
    "low subharmonic rumble",
    "sparkling icy crystals",
    "ghostly chamber pad",
    "harmonic series sweep",
    "wind through cathedral",
    "synthesized harpsichord",
    "shimmering bowed metal",
    "swirling pad with detuned oscillators",
    "soft sine wave clusters",
    "fog horn drifting in distance",
    "frequency-shifted overtones",
    "spectral choir vowel ee",
    "spectral choir vowel ah",
    "spectral choir vowel oo",
    # vocal-like
    "fragmented vocal whisper",
    "alien chant in unknown language",
    "robotic monk humming",
    "stretched soprano holding a note",
    "throat singer overtones",
    "muffled crowd whispers",
    "child voice slowed to a drone",
    "ghostly female ahh",
    # percussive
    "metallic clang reverberation",
    "clockwork mechanism ticking",
    "wooden xylophone tumbling down stairs",
    "ceramic shards falling",
    "distant thunder rolling",
    "muted timpani roll",
    "tiny bells stuttering",
    "pulsing kick drum at slow tempo",
    "bone-dry stick clicks",
    "shattering glass into reverberant hall",
    # environmental / natural
    "rain on a tin roof",
    "ocean waves crashing on rocks",
    "forest creaks and wind",
    "bird flock taking flight at dusk",
    "rocky cave dripping water",
    "footsteps on frozen snow",
    "campfire crackle close-mic",
    "river flowing over pebbles",
    "wind howling through narrow alley",
    "leaves rustling in autumn breeze",
    # electronic / synth
    "vintage modular synth bleeps",
    "analog square wave detuned chord",
    "fm bell with cross-modulation",
    "saw lead with portamento glide",
    "lo-fi chiptune arpeggio",
    "8-bit explosion sample",
    "phase-modulated drone",
    "ring-modulated metallic chord",
    "fast-attack synth pluck",
    # noise / glitch
    "noisy granular static",
    "high frequency oscillator buzz",
    "broken cassette tape texture",
    "abrasive industrial noise",
    "soft analog tape hiss",
    "buffer underrun glitch loop",
    "bitcrushed harsh fragments",
    "cracked vinyl record loop",
    "white noise modulated by tremolo",
    "shortwave radio static",
    "garbled fax transmission",
    "swarm of digital insects",
    "stuttering data corruption",
    # textural / pad
    "long sustained brass cluster",
    "slow attack pad rising",
    "vinyl-warmth pad shifting in pitch",
    "subtle granular cloud",
    "barely audible breathing texture",
    "looped reversed cymbal swell",
    "string ensemble swelling and decaying",
    "muted accordion pad",
    # mechanical
    "old refrigerator hum cycling on",
    "factory machinery ostinato",
    "elevator humming between floors",
    "vintage typewriter clatter",
    "creaking ship hull at sea",
    "rusting hinge slowly turning",
    "wind-up music box winding down",
    "engine idling at low rpm",
    "telephone dial tone with interference",
    # transient / sharp
    "single piano note decaying with hall reverb",
    "harp glissando upward",
    "bowed cymbal swell",
    "snare hit with long reverb tail",
    "metal sheet wobbling",
    "kalimba pluck in stereo",
    "hammered dulcimer cascade",
    # ambient / slow
    "ambient pad like a distant ocean",
    "very slow sub-bass throbs",
    "ethereal woodwind choir",
    "minor seventh chord sustained",
    "single chord ringing into silence",
    "blooming reverberant texture",
    "frozen wind chime in slow motion",
    # microsound / abstract
    "tiny clicking insects in stereo",
    "geiger counter clicks scattered",
    "popping bubbles underwater",
    "static crackle from old amp",
    "rhythmic gating across stereo field",
    "spectral comb filter sweep",
    "tape splice clicks",
    "vinyl crackle bed",
    # texture qualifiers
    "dense layered drone with shifting partials",
    "sparse rhythmic clusters",
    "evolving texture from breath to grit",
    "harsh modulated metallic timbre",
    "soft modulated pad with subtle motion",
    "low rumble with high sparkle on top",
]

KEYWORD_PROFILE = {
    "shimmer":  ("comb",     "high"),
    "metallic": ("comb",     "high"),
    "drone":    ("pink",     "low"),
    "cavernous":("pink",     "low"),
    "noisy":    ("white",    "mid"),
    "granular": ("white",    "mid"),
    "organ":    ("chord",    "mid"),
    "warm":     ("chord",    "mid"),
    "pulsing":  ("chord",    "mid"),
    "vocal":    ("chord",    "mid"),
    "whisper":  ("shaped_smooth", "mid"),
    "bell":     ("comb",     "high"),
    "choir":    ("chord",    "mid"),
    "underwater":("shaped_smooth","low"),
    "hum":      ("pink",     "low"),
    "buzz":     ("comb",     "high"),
    "broken":   ("white",    "mid"),
    "static":   ("white",    "mid"),
    "string":   ("chord",    "mid"),
    "clang":    ("comb",     "mid"),
    "rumble":   ("pink",     "low"),
    "crystal":  ("comb",     "high"),
    "icy":      ("comb",     "high"),
    "ghostly":  ("shaped_smooth", "mid"),
    "chamber":  ("shaped_smooth", "mid"),
    "industrial":("white",   "mid"),
    "tape":     ("shaped_smooth", "mid"),
    "harmonic": ("chord",    "mid"),
    "clockwork":("comb",     "mid"),
    "wind":     ("shaped_smooth", "low"),
    "cathedral":("chord",    "mid"),
    "harpsichord":("chord",  "mid"),
}


# ---------- protocol types ----------

class Captioner(Protocol):
    def __call__(self, audio: np.ndarray, sr: int) -> str: ...


class TextEmbedder(Protocol):
    def __call__(self, text: str) -> np.ndarray: ...


class TextToAudio(Protocol):
    def __call__(self, prompt: str, duration_s: float,
                 sr: int) -> np.ndarray: ...


# ---------- procedural fallbacks ----------

def _audio_features(audio: np.ndarray, sr: int) -> dict:
    """Cheap descriptors used to pick a deterministic caption."""
    a = audio.astype(np.float32)
    if a.size == 0:
        return {"rms": 0.0, "centroid": 0.0, "flatness": 0.0}
    rms = float(np.sqrt(np.mean(a * a) + 1e-12))
    spec = np.abs(np.fft.rfft(a[: min(len(a), sr * 4)]))
    freqs = np.fft.rfftfreq(min(len(a), sr * 4), 1.0 / sr)
    if spec.sum() < 1e-9:
        centroid = 0.0
    else:
        centroid = float((spec * freqs).sum() / spec.sum())
    geo = float(np.exp(np.mean(np.log(spec + 1e-9))))
    arith = float(np.mean(spec + 1e-9))
    flat = geo / arith if arith > 0 else 0.0
    return {"rms": rms, "centroid": centroid, "flatness": flat}


def procedural_caption(audio: np.ndarray, sr: int) -> str:
    """Deterministic caption from cheap acoustic descriptors."""
    f = _audio_features(audio, sr)
    # Bucket centroid into low/mid/high
    if f["centroid"] < 500:
        band = "low"
    elif f["centroid"] < 3000:
        band = "mid"
    else:
        band = "high"
    if f["flatness"] > 0.4:
        kind = "noisy"
    elif f["rms"] < 0.05:
        kind = "quiet"
    else:
        kind = "tonal"
    pool = {
        ("low", "noisy"): ["deep cavernous drone", "low subharmonic rumble",
                           "underwater hum"],
        ("low", "tonal"): ["warm pulsing organ", "warm string ensemble",
                           "wind through cathedral"],
        ("low", "quiet"): ["ghostly chamber pad", "soft analog tape hiss"],
        ("mid", "noisy"): ["noisy granular static", "abrasive industrial noise",
                           "broken cassette tape texture"],
        ("mid", "tonal"): ["harmonic series sweep", "synthesized harpsichord",
                           "warm string ensemble", "crystal bell choir"],
        ("mid", "quiet"): ["fragmented vocal whisper", "ghostly chamber pad"],
        ("high", "noisy"): ["high frequency oscillator buzz",
                            "metallic clang reverberation"],
        ("high", "tonal"): ["bright shimmer with metallic resonance",
                            "sparkling icy crystals", "crystal bell choir"],
        ("high", "quiet"): ["clockwork mechanism ticking"],
    }
    options = pool.get((band, kind), CAPTION_BANK[:3])
    # Deterministic pick from rms hash
    h = int(hashlib.md5(f"{f['rms']:.4f}".encode()).hexdigest()[:8], 16)
    return options[h % len(options)]


def procedural_text_embed(text: str) -> np.ndarray:
    """Hash text to a unit-norm pseudo-embedding of EMBED_DIM."""
    h = hashlib.sha256(text.lower().strip().encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v


def procedural_text_to_audio(prompt: str, duration_s: float,
                             sr: int) -> np.ndarray:
    """Map prompt keywords to MVP-P phase hallucinator presets."""
    from src.modules.mvp_p.phase_halluc import (
        HallucParams, make_magnitude, griffin_lim,
    )
    tokens = [t.lower() for t in prompt.split() if t.isalpha()]
    mode = "shaped_smooth"
    band = "mid"
    for tok in tokens:
        if tok in KEYWORD_PROFILE:
            mode, band = KEYWORD_PROFILE[tok]
            break
    seed = int.from_bytes(
        hashlib.md5(prompt.encode()).digest()[:4], "little"
    )
    # Band → freq center for "chord" / "comb"
    chord = {
        "low":  (55.0, 82.5, 110.0, 165.0),
        "mid":  (165.0, 247.5, 330.0, 440.0),
        "high": (440.0, 660.0, 880.0, 1320.0),
    }[band]
    params = HallucParams(
        mode=mode, duration_s=duration_s, sr=sr,
        n_fft=2048, hop=512, griffin_lim_iters=24,
        smooth_t=0.92, smooth_f=0.85,
        chord_freqs_hz=chord, chord_bw_bins=3, seed=seed,
    )
    mag = make_magnitude(params)
    return griffin_lim(mag, params)


# ---------- real-model lazy loaders (best effort, optional) ----------

_CLAP_INSTANCE = {"module": None, "tried": False}


def _get_clap_module():
    """Load LAION-CLAP once and cache. Returns None on failure."""
    if _CLAP_INSTANCE["tried"]:
        return _CLAP_INSTANCE["module"]
    _CLAP_INSTANCE["tried"] = True
    try:
        import laion_clap  # type: ignore
        m = laion_clap.CLAP_Module(enable_fusion=False)
        m.load_ckpt()
        _CLAP_INSTANCE["module"] = m
    except Exception as exc:  # noqa: BLE001
        logger.info("CLAP unavailable: %s", exc)
        _CLAP_INSTANCE["module"] = None
    return _CLAP_INSTANCE["module"]


def try_load_clap_embedder() -> Optional[Callable[[str], np.ndarray]]:
    m = _get_clap_module()
    if m is None:
        return None

    def embed(text: str) -> np.ndarray:
        e = m.get_text_embedding([text], use_tensor=False)[0]
        v = e.astype(np.float32)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    return embed


def try_load_clap_audio_embedder() -> Optional[Callable[[np.ndarray, int], np.ndarray]]:
    m = _get_clap_module()
    if m is None:
        return None

    def embed(audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        if sr != 48000:
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=sr, target_sr=48000)
        e = m.get_audio_embedding_from_data(x=audio[None, :],
                                            use_tensor=False)[0]
        v = e.astype(np.float32)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    return embed


def try_load_clap_captioner(
    vocabulary: Optional[list[str]] = None,
) -> Optional[Callable[[np.ndarray, int], str]]:
    """Zero-shot audio captioning via CLAP.

    For each input audio:
        1. embed audio with CLAP
        2. precomputed embeddings for each vocabulary phrase
        3. return argmax cosine-sim phrase

    Yields varied captions across diverse audio content. No new model
    download required — reuses the cached CLAP module.
    """
    if _get_clap_module() is None:
        return None
    audio_embed = try_load_clap_audio_embedder()
    text_embed = try_load_clap_embedder()
    if audio_embed is None or text_embed is None:
        return None

    vocab = list(vocabulary) if vocabulary else list(CAPTION_BANK)
    # Precompute caption embeddings
    cap_embs = np.stack([text_embed(c) for c in vocab], axis=0)

    def caption(audio: np.ndarray, sr: int) -> str:
        a = audio_embed(audio, sr)
        sims = cap_embs @ a            # (N,)
        return vocab[int(np.argmax(sims))]

    return caption


def try_load_audioldm() -> Optional[Callable[[str, float, int], np.ndarray]]:
    """Attempt to load AudioLDM TTA pipeline."""
    try:
        from diffusers import AudioLDMPipeline  # type: ignore
        import torch  # type: ignore
        pipe = AudioLDMPipeline.from_pretrained(
            "cvssp/audioldm-s-full-v2", torch_dtype=torch.float16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
    except Exception as exc:  # noqa: BLE001
        logger.info("AudioLDM unavailable, using procedural TTA: %s", exc)
        return None

    def tta(prompt: str, duration_s: float, sr: int) -> np.ndarray:
        import librosa
        out = pipe(prompt, num_inference_steps=10,
                   audio_length_in_s=float(duration_s))
        a = np.asarray(out.audios[0], dtype=np.float32)
        if sr != 16000:
            a = librosa.resample(a, orig_sr=16000, target_sr=sr)
        return a.astype(np.float32)

    return tta


# ---------- service objects with active backend selection ----------

@dataclass
class TextAudioBridge:
    """One shared object holding the active backends. Lazy load."""
    use_real: bool = False
    captioner: Optional[Captioner] = None
    text_embed: Optional[TextEmbedder] = None
    tta: Optional[TextToAudio] = None
    _audio_embed: Optional[Callable[[np.ndarray, int], np.ndarray]] = field(
        default=None, repr=False
    )

    def init_defaults(self) -> None:
        if self.captioner is None:
            real_cap = try_load_clap_captioner() if self.use_real else None
            self.captioner = real_cap if real_cap is not None else procedural_caption
        if self.text_embed is None:
            real = try_load_clap_embedder() if self.use_real else None
            self.text_embed = real if real is not None else procedural_text_embed
        if self.tta is None:
            real = try_load_audioldm() if self.use_real else None
            self.tta = real if real is not None else procedural_text_to_audio
        if self.use_real and self._audio_embed is None:
            self._audio_embed = try_load_clap_audio_embedder()

    def caption(self, audio: np.ndarray, sr: int) -> str:
        if self.captioner is None:
            self.init_defaults()
        return self.captioner(audio, sr)  # type: ignore[misc]

    def embed_text(self, text: str) -> np.ndarray:
        if self.text_embed is None:
            self.init_defaults()
        return self.text_embed(text)  # type: ignore[misc]

    def embed_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self._audio_embed is not None:
            return self._audio_embed(audio, sr)
        # Procedural audio embedding: caption -> text embed.
        return self.embed_text(self.caption(audio, sr))

    def synth(self, prompt: str, duration_s: float, sr: int) -> np.ndarray:
        if self.tta is None:
            self.init_defaults()
        return self.tta(prompt, duration_s, sr)  # type: ignore[misc]
