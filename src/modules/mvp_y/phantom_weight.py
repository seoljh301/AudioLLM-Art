"""Phantom Weight — input-less weight-space generator.

Two RAVE checkpoints + a random morph t + a random latent trajectory →
decode. No audio input. The point: hear the weight-space alone, freed
from any signal commitment.

Pairs naturally with:
- MVP-D (audio→audio · weight): same morph engine, just with input
- MVP-O (∅→audio · latent): same input-less decode, but fixed weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

LatentMode = Literal["white", "smoothed", "sinusoid"]


@dataclass
class PhantomParams:
    morph_t: float = 0.5
    morph_mode: str = "linear"        # passed to MorphParams (linear / slerp)
    latent_mode: LatentMode = "smoothed"
    latent_sigma: float = 1.0
    latent_smooth: float = 0.985
    duration_s: float = 10.0
    seed: int = 0
