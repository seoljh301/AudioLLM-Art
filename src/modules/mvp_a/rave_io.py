"""RAVE TorchScript model loader + encode/decode wrappers.

A pretrained RAVE model is expected as a traced `.ts` file exporting
`encode(audio) -> z` and `decode(z) -> audio` methods. This is the convention
used by ACIDS-IRCAM RAVE exports for the nn~ Max external.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


@dataclass
class RaveHandle:
    model: torch.jit.ScriptModule
    sample_rate: int
    latent_dim: int
    device: torch.device


def load_rave(model_path: str | Path, device: str = "cpu") -> RaveHandle:
    """Load a traced RAVE .ts model and probe its latent dim + sample rate."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"RAVE model not found: {path}")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = torch.jit.load(str(path), map_location=dev).eval()

    # Probe sample rate: RAVE exports usually expose `sr` attribute.
    sample_rate = int(getattr(model, "sr", torch.tensor(48000)).item()) \
        if hasattr(model, "sr") else 48000

    # Probe latent dim by encoding a short silence.
    with torch.no_grad():
        dummy = torch.zeros(1, 1, sample_rate, device=dev)
        z = model.encode(dummy)
    latent_dim = int(z.shape[1])

    logger.info("RAVE loaded: sr=%d latent_dim=%d device=%s path=%s",
                sample_rate, latent_dim, dev, path)
    return RaveHandle(model=model, sample_rate=sample_rate, latent_dim=latent_dim, device=dev)


@torch.no_grad()
def encode(handle: RaveHandle, audio: np.ndarray) -> np.ndarray:
    """Encode mono float32 audio (1D) -> latent (latent_dim, T_z)."""
    wav = torch.from_numpy(audio).float().to(handle.device)
    if wav.ndim == 1:
        wav = wav[None, None, :]
    elif wav.ndim == 2:
        wav = wav[None]
    z = handle.model.encode(wav)
    return z[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def decode(handle: RaveHandle, z: np.ndarray) -> np.ndarray:
    """Decode latent (latent_dim, T_z) -> mono float32 audio (1D)."""
    zt = torch.from_numpy(z).float().to(handle.device)
    if zt.ndim == 2:
        zt = zt[None]
    out = handle.model.decode(zt)
    return out[0, 0].cpu().numpy().astype(np.float32)


def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    """Same helper signature as MVP-C; kept independent so modules don't cross-import."""
    import librosa  # noqa: WPS433
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def save_audio_mono(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
