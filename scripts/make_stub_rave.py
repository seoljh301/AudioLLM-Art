"""Create a minimal RAVE-API-compatible TorchScript stub for testing MVP-A.

The stub exposes:
    encode(x: (B, 1, T)) -> z: (B, latent_dim, T // hop)
    decode(z: (B, latent_dim, T // hop)) -> y: (B, 1, T)
    sr: int attribute

It is NOT a trained RAVE model — it's a deterministic conv autoencoder
that survives the trace and round-trips audio with some loss. Useful only
to verify the MVP-A wiring end-to-end. Replace `checkpoints/rave/stub.ts`
with a real RAVE export (e.g. from acids-ircam/rave-models) for actual
sonic results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class StubRave(nn.Module):
    def __init__(self, latent_dim: int = 16, sample_rate: int = 48000, hop: int = 2048,
                 seed: int = 0) -> None:
        super().__init__()
        # register as buffers so TorchScript can serialize.
        self.register_buffer("sr", torch.tensor(sample_rate))
        self.register_buffer("hop_size", torch.tensor(hop))
        self.register_buffer("latent_dim_buf", torch.tensor(latent_dim))
        # encoder: 1 -> latent_dim, stride=hop
        self.enc = nn.Conv1d(1, latent_dim, kernel_size=hop * 2, stride=hop, padding=hop // 2)
        # decoder: latent_dim -> 1, ConvTranspose1d to invert
        self.dec = nn.ConvTranspose1d(latent_dim, 1, kernel_size=hop * 2, stride=hop, padding=hop // 2)
        # deterministic init so trace is reproducible; seed controls variant.
        with torch.no_grad():
            torch.manual_seed(seed)
            nn.init.orthogonal_(self.enc.weight, gain=0.5)
            nn.init.orthogonal_(self.dec.weight, gain=0.5)
            self.enc.bias.zero_()
            self.dec.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dec(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("checkpoints/rave/stub.ts"))
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--hop", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = StubRave(args.latent_dim, args.sample_rate, args.hop, seed=args.seed).eval()
    scripted = torch.jit.script(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.out))
    print(f"saved stub RAVE -> {args.out}")
    print(f"sr={int(model.sr.item())} latent_dim={int(model.latent_dim_buf.item())} "
          f"hop={int(model.hop_size.item())}")


if __name__ == "__main__":
    main()
