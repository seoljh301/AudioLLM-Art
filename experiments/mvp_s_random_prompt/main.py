"""MVP-S entry: random prompt TTA."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from src.modules.mvp_s.random_prompt import PromptParams, render_random_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=Path(__file__).parent / "config.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = yaml.safe_load(args.config.read_text())
    params = PromptParams(
        duration_s=float(raw["duration_s"]),
        sr=int(raw["sr"]),
        n_prompts=int(raw["n_prompts"]),
        mix_mode=str(raw["mix_mode"]),
        custom_prompts=tuple(raw.get("custom_prompts", []) or []),
        permute_words=bool(raw["permute_words"]),
        use_real=bool(raw["use_real"]),
        seed=int(raw.get("seed", 0)),
    )
    res = render_random_prompt(args.out, params)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
