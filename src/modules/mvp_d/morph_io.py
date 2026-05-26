"""Load 2+ RAVE TorchScript checkpoints and produce a morphed model.

The morphed model is one of the source ScriptModules with its state_dict
overwritten by the interpolated weights. encode/decode are then identical
in API to the originals, so MVP-A's render pipeline can reuse it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from src.modules.mvp_a.rave_io import RaveHandle
from src.modules.mvp_d.ckpt_morph import MorphParams, interp_state_dicts, random_walk_state_dict
from src.modules.mvp_d.re_basin import rebasin_align
from src.modules.mvp_d.re_basin_full import rebasin_full

logger = logging.getLogger(__name__)


@dataclass
class MorphHandle:
    handle: RaveHandle
    source_paths: list[Path]
    params: MorphParams


def _probe_handle(model: torch.jit.ScriptModule, device: torch.device) -> RaveHandle:
    sr = int(model.sr.item()) if hasattr(model, "sr") else 48000
    with torch.no_grad():
        z = model.encode(torch.zeros(1, 1, sr, device=device))
    return RaveHandle(model=model, sample_rate=sr, latent_dim=int(z.shape[1]), device=device)


def _load_script(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.jit.load(str(path), map_location=device).eval()


def load_morph(
    paths: Iterable[str | Path],
    params: MorphParams,
    device: str = "cpu",
    rebasin: bool = False,
    rebasin_mode: str = "partial",
) -> MorphHandle:
    """Load multiple checkpoints and apply `params` to produce a morphed model.

    If `rebasin=True`:
      - `rebasin_mode='partial'` (default): inner-block-only Re-Basin
        (`src/modules/mvp_d/re_basin.py`).
      - `rebasin_mode='full'`: full encoder chain Re-Basin
        (`src/modules/mvp_d/re_basin_full.py`).
    """
    paths = [Path(p) for p in paths]
    if len(paths) < 1:
        raise ValueError("need at least one checkpoint")

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    models = [_load_script(p, dev) for p in paths]
    sds = [m.state_dict() for m in models]

    if rebasin and len(sds) >= 2:
        ref = sds[0]
        for i in range(1, len(sds)):
            if rebasin_mode == "full":
                aligned, report = rebasin_full(ref, sds[i])
                logger.info("rebasin_full sd[%d]: %d classes, iters=%d",
                            i, report["n_classes"], report["iterations"])
            else:
                aligned, report = rebasin_align(ref, sds[i])
                logger.info("rebasin sd[%d]: %d blocks aligned", i, report["n_blocks"])
            sds[i] = aligned

    if params.mode == "random_walk":
        if len(sds) > 1:
            logger.warning("random_walk uses only the first ckpt; %d extras ignored", len(sds) - 1)
        merged = random_walk_state_dict(sds[0], params)
    elif len(sds) == 1:
        merged = sds[0]
    elif len(sds) == 2:
        merged = interp_state_dicts(sds[0], sds[1], params.t, params.mode)
    else:
        # Chain interp: collapse list into one with the configured t between adjacent pairs.
        merged = sds[0]
        for nxt in sds[1:]:
            merged = interp_state_dicts(merged, nxt, params.t, params.mode)

    target = models[0]
    target.load_state_dict(merged)
    target.eval()
    handle = _probe_handle(target, dev)
    logger.info("morph loaded: %d ckpts, mode=%s, t=%.3f, sr=%d, latent_dim=%d",
                len(paths), params.mode, params.t, handle.sample_rate, handle.latent_dim)
    return MorphHandle(handle=handle, source_paths=paths, params=params)


def remorph_inplace(
    morph: MorphHandle,
    source_sds: list[dict],
    params: MorphParams,
) -> None:
    """Re-apply morph with new params using cached source state_dicts."""
    if params.mode == "random_walk":
        merged = random_walk_state_dict(source_sds[0], params)
    elif len(source_sds) == 1:
        merged = source_sds[0]
    elif len(source_sds) == 2:
        merged = interp_state_dicts(source_sds[0], source_sds[1], params.t, params.mode)
    else:
        merged = source_sds[0]
        for nxt in source_sds[1:]:
            merged = interp_state_dicts(merged, nxt, params.t, params.mode)
    morph.handle.model.load_state_dict(merged)
    morph.params = params
