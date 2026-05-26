"""Audio <-> Text recursive loop.

Pipeline:
    audio -> caption model (Qwen-Audio etc.) -> text
         -> text mutator (optional)
         -> text-to-audio model (AudioLDM2 etc.) -> audio
         -> repeat

Both models are accessed via the shared model registry; this module owns the
loop control flow only.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

CaptionFn = Callable[["any"], str]    # audio -> text
SynthFn = Callable[[str], "any"]      # text -> audio


@dataclass
class LoopParams:
    depth: int = 4
    text_mutation_prob: float = 0.2
    mutation_pool: list[str] = field(
        default_factory=lambda: [
            "fragmented", "decayed", "whispered", "underwater",
            "metallic", "haunted", "smeared", "impossible",
        ]
    )
    seed: int = 0


def mutate_text(text: str, params: LoopParams, rng: random.Random) -> str:
    """Insert one mutation adjective with probability `text_mutation_prob`."""
    if rng.random() >= params.text_mutation_prob:
        return text
    adj = rng.choice(params.mutation_pool)
    return f"{adj} {text}"


def run_loop(
    seed_audio: "any",
    caption_fn: CaptionFn,
    synth_fn: SynthFn,
    params: LoopParams,
) -> list[tuple[str, "any"]]:
    """Run the recursive loop. Returns a list of (text, audio) per step."""
    rng = random.Random(params.seed)
    audio = seed_audio
    history: list[tuple[str, "any"]] = []
    for step in range(params.depth):
        text = caption_fn(audio)
        text = mutate_text(text, params, rng)
        logger.info("loop step %d: %r", step, text)
        audio = synth_fn(text)
        history.append((text, audio))
    return history
