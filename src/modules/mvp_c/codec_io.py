"""EnCodec / DAC encode-decode wrappers for MVP-C token bending.

Keeps token_bend.py pure-numpy. This module owns the torch + codec model
side: load, encode chunks -> int token tensor, decode int tokens -> audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

CodecName = Literal["encodec_24khz", "encodec_48khz", "dac_44khz"]


@dataclass
class CodecHandle:
    name: CodecName
    model: object
    sample_rate: int
    n_quantizers: int
    codebook_size: int
    device: torch.device


def load_codec(name: CodecName, device: str = "cuda", bandwidth: float = 6.0) -> CodecHandle:
    """Load an audio codec. Returns a uniform handle."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if name.startswith("encodec"):
        from encodec import EncodecModel  # noqa: WPS433
        if name == "encodec_24khz":
            m = EncodecModel.encodec_model_24khz()
            sr = 24000
        elif name == "encodec_48khz":
            m = EncodecModel.encodec_model_48khz()
            sr = 48000
        else:
            raise ValueError(f"unknown encodec variant: {name}")
        m.set_target_bandwidth(bandwidth)
        m = m.to(dev).eval()
        # n_q depends on bandwidth for EnCodec. 
        # For 24kHz: 1.5kbps=2, 3kbps=4, 6kbps=8, 12kbps=16, 24kbps=32
        # We can derive it from bandwidth: (bandwidth * 1000) / (frame_rate * bits_per_codebook)
        # For 24kHz, frame_rate is 75. bits_per_codebook is 10.
        frame_rate = 75 if name == "encodec_24khz" else 150
        n_q = int((bandwidth * 1000) / (frame_rate * 10))
        
        cb = m.quantizer.vq.layers[0].codebook.shape[0]
        return CodecHandle(name=name, model=m, sample_rate=sr, n_quantizers=n_q,
                           codebook_size=cb, device=dev)

    if name == "dac_44khz":
        import dac  # noqa: WPS433
        model_path = dac.utils.download(model_type="44khz")
        m = dac.DAC.load(model_path).to(dev).eval()
        # DAC default 44.1kHz, 9 quantizers, codebook 1024
        return CodecHandle(name=name, model=m, sample_rate=44100,
                           n_quantizers=m.n_codebooks, codebook_size=m.codebook_size,
                           device=dev)

    raise ValueError(f"unknown codec: {name}")


@torch.no_grad()
def encode_audio(handle: CodecHandle, audio: np.ndarray) -> np.ndarray:
    """Encode mono float32 audio (1D) to int tokens of shape (n_q, T)."""
    wav = torch.from_numpy(audio).float().to(handle.device)
    if wav.ndim == 1:
        wav = wav[None, None, :]  # (B=1, C=1, T)
    elif wav.ndim == 2:
        wav = wav[None]            # (B=1, C, T)

    if handle.name.startswith("encodec"):
        # EncodecModel.encode returns a list[ (codes, scale) ] per frame.
        encoded_frames = handle.model.encode(wav)
        codes = torch.cat([c for c, _ in encoded_frames], dim=-1)  # (B, n_q, T)
        return codes[0].cpu().numpy().astype(np.int64)

    if handle.name == "dac_44khz":
        wav = handle.model.preprocess(wav, handle.sample_rate)
        z, codes, _, _, _ = handle.model.encode(wav)
        return codes[0].cpu().numpy().astype(np.int64)

    raise ValueError(f"unknown codec: {handle.name}")


@torch.no_grad()
def decode_tokens(handle: CodecHandle, tokens: np.ndarray) -> np.ndarray:
    """Decode int tokens of shape (n_q, T) back to mono float32 audio (1D)."""
    codes = torch.from_numpy(tokens).long().to(handle.device)
    # Clamp invalid tokens (e.g. -1 from invalid_token mode) to a safe value.
    codes = torch.clamp(codes, 0, handle.codebook_size - 1)
    
    n_q, total_frames = codes.shape
    
    # Process in chunks of 4500 frames (~1 minute) to avoid VRAM OOM on massive files
    chunk_size = 4500
    audio_chunks = []
    
    for i in range(0, total_frames, chunk_size):
        chunk_codes = codes[:, i : i + chunk_size]
        chunk_codes = chunk_codes[None]  # (B=1, n_q, T)

        if handle.name.startswith("encodec"):
            out = handle.model.decode([(chunk_codes, None)])
            audio_chunks.append(out[0, 0].cpu().numpy().astype(np.float32))

        elif handle.name == "dac_44khz":
            z = handle.model.quantizer.from_codes(chunk_codes)[0]
            out = handle.model.decode(z)
            audio_chunks.append(out[0, 0].cpu().numpy().astype(np.float32))
        else:
            raise ValueError(f"unknown codec: {handle.name}")

    return np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)


def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    """Load a wav file, downmix to mono, resample to target_sr. Returns float32 1D."""
    import librosa  # noqa: WPS433
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def save_audio_mono(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
