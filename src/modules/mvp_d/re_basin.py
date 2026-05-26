"""Approximate Git Re-Basin alignment for RAVE state_dicts.

Targets the *inner* channel permutation symmetry inside each residual block
of a RAVE TorchScript export. Each block has the pattern:

    branches.0.1.weight  : Conv1d(in -> inner, k)        # expansion
    branches.0.3.weight  : Conv1d(inner -> out, 1)       # projection

The `inner` channel dimension is **isolated within the block** — its activations
never leave that resblock, so permuting it does not break skip connections or
neighbouring layers. This makes it the cleanest target for the Ainsworth et al.
"Git Re-Basin" (2023) weight-matching procedure.

Algorithm (one shot, no iterative weight-matching loop):

  for each paired block in source-B:
      cost[i, j] = ||W_A.1[i] - W_B.1[j]||^2 + small * ||W_A.3[:, i] - W_B.3[:, j]||^2
      π = linear_sum_assignment(cost)
      apply π to B's:
          .1.weight (dim 0)            output channels of expansion
          .1.bias   (dim 0)            if present
          .3.weight (dim 1)            input channels of projection
          .3.cache.pad (dim 1)         streaming cache for projection input
          .3.downsampling_delay.pad (dim 1)

The other state_dict keys (skip-add channels, encoder/decoder boundaries, PQMF
filter bank, prior_net, etc.) are *not* permuted — those would require the full
permutation-symmetry graph of RAVE, which is complex and not the goal here.

This is therefore "partial Re-Basin" — it aligns the easy axes only. Sufficient
to test whether intra-block alignment alone reduces the weight-interp collapse.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

_BRANCH_RE = re.compile(r"^(.*?\.aligned\.branches\.\d+)\.(1|3)\.weight$")


def _find_paired_blocks(sd: dict) -> list[str]:
    """Return prefixes of resblock branches that have both .1 and .3 weights."""
    pairs: dict[str, set[str]] = {}
    for k in sd.keys():
        mo = _BRANCH_RE.match(k)
        if mo:
            pairs.setdefault(mo.group(1), set()).add(mo.group(2))
    return [p for p, s in pairs.items() if {"1", "3"}.issubset(s)]


def _compute_inner_perm(w_a1: torch.Tensor, w_b1: torch.Tensor,
                        w_a3: torch.Tensor, w_b3: torch.Tensor,
                        groups: int = 1) -> torch.Tensor:
    """Solve LAP per group to align B's inner channels to A's.

    For grouped conv (groups>1), inner channels are tied to specific input
    groups and can only be permuted *within* each group of size inner/groups.
    """
    inner = int(w_a1.shape[0])
    if inner % groups != 0:
        raise ValueError(f"inner {inner} not divisible by groups {groups}")
    group_size = inner // groups

    # w_a1 shape: (inner, in_per_group, k). Flatten across (in_per_group, k).
    a1 = w_a1.reshape(inner, -1).float()
    b1 = w_b1.reshape(inner, -1).float()
    # w_a3 shape: (out, inner, k). Move inner to front, flatten across (out, k).
    a3 = w_a3.permute(1, 0, 2).reshape(inner, -1).float()
    b3 = w_b3.permute(1, 0, 2).reshape(inner, -1).float()

    perm = torch.arange(inner)
    for g in range(groups):
        lo, hi = g * group_size, (g + 1) * group_size
        cost = -(a1[lo:hi] @ b1[lo:hi].T + a3[lo:hi] @ b3[lo:hi].T).cpu().numpy()
        _, col = linear_sum_assignment(cost)
        perm[lo:hi] = torch.from_numpy(col).long() + lo
    return perm.long()


def _apply_block_perm(sd: dict, prefix: str, perm: torch.Tensor) -> int:
    """Permute B's resblock state in-place. Returns number of tensors touched."""
    touched = 0

    perm_len = int(perm.shape[0])

    def _maybe(key: str, dim: int) -> None:
        nonlocal touched
        if key not in sd:
            return
        t = sd[key]
        if t.shape[dim] != perm_len:
            return  # dim is not the inner axis we're permuting
        p = perm.to(t.device)
        sd[key] = t.index_select(dim, p).clone()
        touched += 1

    _maybe(f"{prefix}.1.weight", 0)                       # output channels of expansion
    _maybe(f"{prefix}.1.bias", 0)
    _maybe(f"{prefix}.3.weight", 1)                       # input channels of projection
    _maybe(f"{prefix}.3.cache.pad", 1)                    # inner-side streaming cache
    _maybe(f"{prefix}.3.downsampling_delay.pad", 1)
    return touched


def _infer_groups(sd: dict, prefix: str) -> int:
    """Infer the conv `groups` of `.1` by comparing weight in-channels to cache channels."""
    k_weight = f"{prefix}.1.weight"
    k_cache = f"{prefix}.1.cache.pad"
    if k_weight not in sd or k_cache not in sd:
        return 1
    in_per_group = int(sd[k_weight].shape[1])
    in_total = int(sd[k_cache].shape[1])
    if in_per_group == 0:
        return 1
    if in_total % in_per_group != 0:
        return 1
    return in_total // in_per_group


def rebasin_align(sd_a: dict, sd_b: dict) -> tuple[dict, dict]:
    """Return a permuted copy of `sd_b` whose inner channels align to `sd_a`.

    The second return value is a small report dict for logging / results.
    """
    sd_b_aligned: dict = {k: v.clone() for k, v in sd_b.items()}

    paired = _find_paired_blocks(sd_a)
    report = {"n_blocks": len(paired), "blocks": []}

    for prefix in paired:
        k1 = f"{prefix}.1.weight"
        k3 = f"{prefix}.3.weight"
        if k1 not in sd_a or k1 not in sd_b_aligned:
            continue
        if sd_a[k1].shape != sd_b_aligned[k1].shape:
            logger.warning("shape mismatch at %s — skipping", k1)
            continue
        groups = _infer_groups(sd_a, prefix)
        perm = _compute_inner_perm(sd_a[k1], sd_b_aligned[k1],
                                   sd_a[k3], sd_b_aligned[k3], groups=groups)
        touched = _apply_block_perm(sd_b_aligned, prefix, perm)
        identity = int((perm == torch.arange(len(perm))).sum().item())
        report["blocks"].append({
            "prefix": prefix,
            "inner_dim": int(len(perm)),
            "groups": groups,
            "tensors_permuted": touched,
            "identity_kept": identity,
            "moved": int(len(perm) - identity),
        })
        logger.info("re-basin %s: inner=%d, groups=%d, identity_kept=%d, tensors=%d",
                    prefix, len(perm), groups, identity, touched)

    return sd_b_aligned, report


def rebasin_files(path_a: str | Path, path_b: str | Path, out_path: str | Path) -> dict:
    """Load two RAVE .ts files, align B to A, save aligned state_dict as .pt."""
    a = torch.jit.load(str(path_a), map_location="cpu").eval()
    b = torch.jit.load(str(path_b), map_location="cpu").eval()
    sd_a = a.state_dict()
    sd_b = b.state_dict()
    aligned, report = rebasin_align(sd_a, sd_b)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as plain torch tensor dict; loader script can apply via b.load_state_dict.
    torch.save(aligned, str(out_path))
    report["out_path"] = str(out_path)
    return report
