"""Codec latent encoder/decoder for latent musaicing.

Implements Kui et al. 2025 (arXiv:2507.19202) — operate on continuous
pre-VQ embeddings from a neural audio codec encoder, then decode the
assembled grain sequence with the same codec's decoder.

We use EnCodec (24kHz) by default: model.encoder(x) returns shape
(B, 128, T_lat) at frame_rate=75 Hz. Decoder accepts (B, 128, T_lat)
and returns waveform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

logger = logging.getLogger(__name__)

CodecName = Literal["encodec_24khz", "encodec_48khz"]


@dataclass
class CodecLatentHandle:
    name: CodecName
    model: object
    sample_rate: int
    latent_dim: int
    frame_rate: int  # Hz
    device: torch.device


def load_codec_latent(name: CodecName = "encodec_24khz",
                      device: str = "cuda") -> CodecLatentHandle:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    from encodec import EncodecModel  # noqa: WPS433
    if name == "encodec_24khz":
        m = EncodecModel.encodec_model_24khz()
        sr, fr = 24000, 75
    elif name == "encodec_48khz":
        m = EncodecModel.encodec_model_48khz()
        sr, fr = 48000, 150
    else:
        raise ValueError(f"unknown codec: {name}")
    m.set_target_bandwidth(6.0)
    m = m.to(dev).eval()
    dim = m.encoder.dimension if hasattr(m.encoder, "dimension") else 128
    return CodecLatentHandle(name=name, model=m, sample_rate=sr,
                             latent_dim=dim, frame_rate=fr, device=dev)


@torch.no_grad()
def encode_latent(handle: CodecLatentHandle, audio: np.ndarray) -> np.ndarray:
    """Mono float32 1D → continuous latent (dim, T_lat)."""
    wav = torch.from_numpy(audio).float().to(handle.device)
    if wav.ndim == 1:
        wav = wav[None, None, :]
    elif wav.ndim == 2:
        wav = wav[None]
    emb = handle.model.encoder(wav)
    return emb[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def decode_latent(handle: CodecLatentHandle, emb: np.ndarray) -> np.ndarray:
    """Continuous latent (dim, T_lat) → mono float32 1D."""
    z = torch.from_numpy(emb).float().to(handle.device)
    if z.ndim == 2:
        z = z[None]
    chunk = 6000
    parts = []
    for i in range(0, z.shape[-1], chunk):
        out = handle.model.decoder(z[..., i:i + chunk])
        parts.append(out[0, 0].cpu().numpy().astype(np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
