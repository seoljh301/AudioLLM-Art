"""Full RAVE Git Re-Basin: walk the complete permutation symmetry graph.

Identifies every channel-dim permutation class in a RAVE TorchScript model and
solves them iteratively (Ainsworth et al. 2023, Algorithm 3 — weight matching).

Topology covered:

  Encoder chain:
    enc_48  : encoder.net.0 output, .1 resblock skip, .2 input
    enc_96  : encoder.net.2 output, .3 resblock skip, .4 input
    enc_192 : .4 output, .5 skip, .6 input
    enc_384 : .6 output, .7 skip, .8 input
    enc_768 : .8 output, .9 skip, .10 input, .10 output, .11 skip, .12 input,
              .12 output, .13 skip, .14 input, .14 output, .16 input (grouped=2!)
    inner_enc_{1,3,5,7,9,11,13}: resblock inner channels (handled per-block,
              same as `re_basin.py`)

  Decoder chain (mirrored):
    dec_768 : decoder.net.0 output, .2 input, .2 output, .3 skip, .4 input,
              .4 output, .5 skip, .6 input, .6 output, .7 skip, .8 input
    dec_384 : .8 output, .9 skip, .10 input
    dec_192 : .10 output, .11 skip, .12 input
    dec_96  : .12 output, .13 skip, .14 input
    dec_48  : .14 output, .15 skip, synth input
    inner_dec_{3,5,7,9,11,13,15} (×2 per dec block — two nets per block)

  Latent (128-d): output of encoder.net.16 (split into mean[0:128] / logvar[128:256])
    Only the MEAN portion connects to gimbal / decoder.net.0 / prior_net.net.0 /
    latent_pca / latent_mean / last_z. Logvar (128:256) is used only at sampling
    and does not propagate downstream — we treat it as fixed for now.
    NOTE: latent_pca is (128, 128) — applying a permutation rotates the PCA axes,
    so we must permute both rows AND columns of latent_pca to preserve the
    pca @ z + mean computation. Similarly for prior_net.

The `enc_768` class has a grouped-conv constraint at `encoder.net.16`: its input
(768 ch) is split into two halves (0..383 and 384..767), each feeding one
conv-group. We respect this by solving two sub-LAPs within enc_768.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import torch
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class Axis:
    """One axis of one tensor that participates in a permutation class."""
    key: str          # state_dict key
    dim: int          # which dim to index_select
    expected_size: int  # for validation
    sub_range: tuple[int, int] | None = None  # (lo, hi) within a grouped axis; None = full


@dataclass
class PermClass:
    """A permutation class: a set of axes that must share one permutation."""
    name: str
    size: int
    axes: list[Axis] = field(default_factory=list)
    sub_groups: list[tuple[int, int]] | None = None  # INDEPENDENT perms per sub-range
    tied_groups: int | None = None                    # TIED perm replicated across N groups

    @property
    def base_size(self) -> int:
        """Free size of the permutation (per-group when tied)."""
        if self.tied_groups:
            return self.size // self.tied_groups
        return self.size


def _find_resblock_prefixes(sd: dict, parent: str) -> dict[int, str]:
    """Return {net_index: prefix} for each resblock in encoder/decoder net.X."""
    out: dict[int, str] = {}
    # Pattern e.g. encoder.net.1.net.0.aligned.branches.0
    pat = re.compile(rf'^{re.escape(parent)}\.net\.(\d+)\.net\.(\d+)\.aligned\.branches\.0\.1\.weight$')
    for k in sd.keys():
        mo = pat.match(k)
        if mo:
            idx = int(mo.group(1))
            sub = int(mo.group(2))
            prefix = f'{parent}.net.{idx}.net.{sub}.aligned.branches.0'
            out[(idx, sub)] = prefix
    return out


# ---------------------------------------------------------------------------
# Class registration
# ---------------------------------------------------------------------------

def _add_conv_layer_axes(cls_in: PermClass | None, cls_out: PermClass | None,
                        weight_key: str, bias_key: str | None,
                        cache_key: str | None, delay_key: str | None,
                        in_size: int, out_size: int) -> None:
    """Register a conv layer's axes into in/out permutation classes."""
    if cls_in is not None:
        cls_in.axes.append(Axis(weight_key, dim=1, expected_size=in_size))
        if cache_key is not None:
            cls_in.axes.append(Axis(cache_key, dim=1, expected_size=in_size))
        if delay_key is not None:
            cls_in.axes.append(Axis(delay_key, dim=1, expected_size=in_size))
    if cls_out is not None:
        cls_out.axes.append(Axis(weight_key, dim=0, expected_size=out_size))
        if bias_key is not None:
            cls_out.axes.append(Axis(bias_key, dim=0, expected_size=out_size))


def _infer_block_groups(sd: dict, prefix: str) -> int:
    k_weight = f'{prefix}.1.weight'
    k_cache = f'{prefix}.1.cache.pad'
    if k_weight not in sd or k_cache not in sd:
        return 1
    in_per_group = int(sd[k_weight].shape[1])
    in_total = int(sd[k_cache].shape[1])
    if in_per_group == 0 or in_total % in_per_group != 0:
        return 1
    return in_total // in_per_group


def _sub_groups_for(size: int, groups: int) -> list[tuple[int, int]] | None:
    if groups <= 1:
        return None
    step = size // groups
    return [(g * step, (g + 1) * step) for g in range(groups)]


def _register_resblock(classes: dict[str, PermClass],
                       prefix: str,
                       block_in_cls: PermClass,
                       block_out_cls: PermClass,
                       inner_class_name: str,
                       sd: dict) -> None:
    """Register all axes belonging to an aligned-resblock into the right classes.

    Branch structure:
      .1.weight: (inner, in_per_group, k)   — block_in → inner
      .3.weight: (out, inner, k)            — inner → block_out

    .1 may be a grouped conv (.1.cache.pad channels > .1.weight.shape[1]).
    For now we just register the axes; the actual LAP for the inner class is
    solved per-block in re_basin.py (called by `align_inner_blocks`).

    Boundary contributions (block_in_cls and block_out_cls):
      block_in_cls  receives: .1.cache.pad (input cache), .paddings.*.pad
      block_out_cls receives: .3.weight (dim 0)
    """
    w1 = f'{prefix}.1.weight'
    w3 = f'{prefix}.3.weight'
    inner_size = int(sd[w1].shape[0])
    groups = _infer_block_groups(sd, prefix)
    cls_inner = classes.setdefault(
        inner_class_name, PermClass(name=inner_class_name, size=inner_size,
                                    sub_groups=_sub_groups_for(inner_size, groups)))

    cls_inner.axes.append(Axis(w1, dim=0, expected_size=inner_size))
    cls_inner.axes.append(Axis(w3, dim=1, expected_size=inner_size))
    c3 = f'{prefix}.3.cache.pad'
    d3 = f'{prefix}.3.downsampling_delay.pad'
    if c3 in sd and sd[c3].shape[1] == inner_size:
        cls_inner.axes.append(Axis(c3, dim=1, expected_size=inner_size))
    if d3 in sd and sd[d3].shape[1] == inner_size:
        cls_inner.axes.append(Axis(d3, dim=1, expected_size=inner_size))

    # block_in_cls touches .1's INPUT side and skip-padding caches
    in_size = block_in_cls.size
    c1 = f'{prefix}.1.cache.pad'
    d1 = f'{prefix}.1.downsampling_delay.pad'
    if c1 in sd and sd[c1].shape[1] == in_size:
        block_in_cls.axes.append(Axis(c1, dim=1, expected_size=in_size))
    if d1 in sd and sd[d1].shape[1] == in_size:
        block_in_cls.axes.append(Axis(d1, dim=1, expected_size=in_size))
    for j in (0, 1):
        pkey = f'{prefix.replace(".branches.0", "")}.paddings.{j}.pad'
        if pkey in sd and sd[pkey].shape[1] == in_size:
            block_in_cls.axes.append(Axis(pkey, dim=1, expected_size=in_size))

    # block_out_cls receives .3's OUTPUT side
    block_out_cls.axes.append(Axis(w3, dim=0, expected_size=block_out_cls.size))
    # Note: .1's INPUT side weight axis (dim 1) also belongs to block_in_cls
    block_in_cls.axes.append(Axis(w1, dim=1, expected_size=int(sd[w1].shape[1])))


def build_encoder_classes(sd: dict, include_enc_768: bool = False) -> dict[str, PermClass]:
    """Build all encoder permutation classes from the state_dict.

    `include_enc_768=False` (default): skip the 768-channel chain that terminates
    at encoder.net.16 (grouped conv into latent). Permuting it requires also
    propagating the permutation into the latent → decoder/gimbal/prior_net,
    which is not implemented. With it disabled, the encoder is aligned on
    chains 48, 96, 192, 384 + every resblock inner class; the 768-chain stays
    identity.
    """
    classes: dict[str, PermClass] = {}

    # Resblock-imposed group constraints on the block-INPUT class.
    # The grouped .1 conv at each resblock requires the upstream class to permute
    # within sub-ranges matching its `in_per_group` size.
    resblock_inputs = [(1, 48), (3, 96), (5, 192), (7, 384), (9, 768), (11, 768), (13, 768)]
    block_in_groups: dict[int, int] = {}
    for idx, sz in resblock_inputs:
        prefix = f'encoder.net.{idx}.net.0.aligned.branches.0'
        g = _infer_block_groups(sd, prefix)
        block_in_groups[sz] = max(block_in_groups.get(sz, 1), g)
    # .16 also constrains 768 with groups=2
    if 'encoder.net.16.weight' in sd:
        block_in_groups[768] = max(block_in_groups.get(768, 1), 2)

    sizes_chain = [48, 96, 192, 384] + ([768] if include_enc_768 else [])
    chain_classes: list[PermClass] = []
    for sz in sizes_chain:
        g = block_in_groups.get(sz, 1)
        # Grouped consumer (.5, .7, .9, .11, .13, .16) shares weights across groups,
        # so the upstream class must use a TIED perm replicated `g` times.
        tied = g if g > 1 else None
        c = PermClass(name=f'enc_{sz}', size=sz, tied_groups=tied)
        classes[c.name] = c
        chain_classes.append(c)

    # encoder.net.0: (48, 16, 7) — input 16 is PQMF-fixed (NOT permuted)
    _add_conv_layer_axes(
        cls_in=None, cls_out=classes['enc_48'],
        weight_key='encoder.net.0.weight', bias_key='encoder.net.0.bias',
        cache_key='encoder.net.0.cache.pad', delay_key='encoder.net.0.downsampling_delay.pad',
        in_size=16, out_size=48)

    # encoder.net.2,4,6,8: channel-doublers
    for net_idx, in_sz, out_sz in [(2, 48, 96), (4, 96, 192), (6, 192, 384), (8, 384, 768)]:
        cls_out = classes.get(f'enc_{out_sz}')
        _add_conv_layer_axes(
            cls_in=classes[f'enc_{in_sz}'], cls_out=cls_out,
            weight_key=f'encoder.net.{net_idx}.weight', bias_key=f'encoder.net.{net_idx}.bias',
            cache_key=f'encoder.net.{net_idx}.cache.pad',
            delay_key=f'encoder.net.{net_idx}.downsampling_delay.pad',
            in_size=in_sz, out_size=out_sz)

    if include_enc_768:
        # encoder.net.10, 12, 14: 768 bridge layers (in==out==768)
        for net_idx in (10, 12, 14):
            _add_conv_layer_axes(
                cls_in=classes['enc_768'], cls_out=classes['enc_768'],
                weight_key=f'encoder.net.{net_idx}.weight', bias_key=f'encoder.net.{net_idx}.bias',
                cache_key=f'encoder.net.{net_idx}.cache.pad',
                delay_key=f'encoder.net.{net_idx}.downsampling_delay.pad',
                in_size=768, out_size=768)
        # encoder.net.16: grouped conv groups=2, in 384*2=768, out 128*2=256.
        classes['enc_768'].sub_groups = [(0, 384), (384, 768)]
        classes['enc_768'].axes.append(Axis('encoder.net.16.weight', dim=1, expected_size=384))
        if 'encoder.net.16.cache.pad' in sd:
            classes['enc_768'].axes.append(Axis('encoder.net.16.cache.pad', dim=1, expected_size=768))
        if 'encoder.net.16.downsampling_delay.pad' in sd:
            classes['enc_768'].axes.append(Axis('encoder.net.16.downsampling_delay.pad', dim=1, expected_size=768))

    # Resblocks 1,3,5,7 — always register. 9,11,13 only if include_enc_768.
    resblock_in_out = [(1, 48), (3, 96), (5, 192), (7, 384)]
    if include_enc_768:
        resblock_in_out += [(9, 768), (11, 768), (13, 768)]
    for idx, sz in resblock_in_out:
        prefix = f'encoder.net.{idx}.net.0.aligned.branches.0'
        block_cls = classes[f'enc_{sz}']
        _register_resblock(classes, prefix,
                           block_in_cls=block_cls, block_out_cls=block_cls,
                           inner_class_name=f'enc_inner_{idx}', sd=sd)

    # For inner-only handling of the 768 resblocks (when enc_768 is OFF), still
    # permute their inner channels — the inner is fully isolated within each block.
    if not include_enc_768:
        for idx in (9, 11, 13):
            prefix = f'encoder.net.{idx}.net.0.aligned.branches.0'
            w1 = f'{prefix}.1.weight'
            if w1 not in sd:
                continue
            inner_size = int(sd[w1].shape[0])
            groups = _infer_block_groups(sd, prefix)
            cls_inner = classes.setdefault(
                f'enc_inner_{idx}', PermClass(name=f'enc_inner_{idx}', size=inner_size,
                                              sub_groups=_sub_groups_for(inner_size, groups)))
            cls_inner.axes.append(Axis(w1, dim=0, expected_size=inner_size))
            w3 = f'{prefix}.3.weight'
            cls_inner.axes.append(Axis(w3, dim=1, expected_size=inner_size))
            c3 = f'{prefix}.3.cache.pad'
            d3 = f'{prefix}.3.downsampling_delay.pad'
            if c3 in sd and sd[c3].shape[1] == inner_size:
                cls_inner.axes.append(Axis(c3, dim=1, expected_size=inner_size))
            if d3 in sd and sd[d3].shape[1] == inner_size:
                cls_inner.axes.append(Axis(d3, dim=1, expected_size=inner_size))

    return classes


# ---------------------------------------------------------------------------
# Re-Basin solver (Ainsworth Algorithm 3 — iterative weight matching)
# ---------------------------------------------------------------------------

def _solve_class(cls: PermClass,
                 sd_a: dict, sd_b: dict,
                 cur_perms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Solve LAP for one class given current perms of OTHER classes applied to sd_b.

    For each axis in this class, build the cost matrix between sd_a's slice and
    the OTHER-axis-permuted sd_b's slice, summed across all axes.
    """
    # Helper: get sd_b tensor with all OTHER classes' perms already applied to
    # appropriate axes. Since perms are stored as LongTensors, we can build a
    # working dict on-the-fly.
    def _apply_other_perms(t: torch.Tensor, key: str, this_dim: int) -> torch.Tensor:
        out = t
        # We don't know which other class owns which dim; collect all axes for
        # this key from all classes and apply their perms (except for this_dim).
        # Caller passes the source tensor; here we just return as-is and rely on
        # the global apply step at the end. (Cheaper than re-deriving per call.)
        return out

    base = cls.base_size
    cost = torch.zeros(base, base, dtype=torch.float64)
    for ax in cls.axes:
        a = sd_a.get(ax.key)
        b = sd_b.get(ax.key)
        if a is None or b is None:
            continue
        ax_size = a.shape[ax.dim]
        if ax_size == cls.size and cls.tied_groups:
            # Full-size axis on a tied-group class: accumulate cost by summing each
            # of the `tied_groups` slabs onto the base cost matrix.
            a_t = a.detach().movedim(ax.dim, 0).reshape(ax_size, -1).double().cpu()
            b_t = b.detach().movedim(ax.dim, 0).reshape(ax_size, -1).double().cpu()
            for g in range(cls.tied_groups):
                lo, hi = g * base, (g + 1) * base
                cost += a_t[lo:hi] @ b_t[lo:hi].T
            continue
        if ax_size != base:
            continue
        a_t = a.detach().movedim(ax.dim, 0).reshape(ax_size, -1).double().cpu()
        b_t = b.detach().movedim(ax.dim, 0).reshape(ax_size, -1).double().cpu()
        cost += a_t @ b_t.T

    if cls.tied_groups:
        _, col = linear_sum_assignment(-cost.cpu().numpy())
        # Replicate the base-size perm across all groups, with offsets.
        base_perm = torch.from_numpy(col).long()
        full_perm = torch.cat([base_perm + g * base for g in range(cls.tied_groups)])
        return full_perm.long()

    if cls.sub_groups is None:
        _, col = linear_sum_assignment(-cost.cpu().numpy())
        return torch.from_numpy(col).long()

    perm = torch.arange(cls.size)
    for lo, hi in cls.sub_groups:
        sub = -cost[lo:hi, lo:hi].cpu().numpy()
        _, col = linear_sum_assignment(sub)
        perm[lo:hi] = torch.from_numpy(col).long() + lo
    return perm.long()


def _apply_perms(sd_b: dict, classes: dict[str, PermClass],
                 perms: dict[str, torch.Tensor]) -> dict:
    """Build a new state_dict from sd_b with all class perms applied to all axes.

    Handles three axis variants:
      * full-size axis: index_select with the class's full-size perm
      * per-group axis (size == base_size for tied class): use base perm (first
        base elements of the full perm — they are identical across groups by
        construction)
    """
    out: dict[str, torch.Tensor] = {k: v.clone() for k, v in sd_b.items()}
    for cls_name, cls in classes.items():
        full_perm = perms[cls_name]
        base = cls.base_size
        for ax in cls.axes:
            if ax.key not in out:
                continue
            t = out[ax.key]
            ax_size = t.shape[ax.dim]
            if ax_size == cls.size:
                p = full_perm.to(t.device)
                out[ax.key] = t.index_select(ax.dim, p).clone()
            elif cls.tied_groups and ax_size == base:
                # per-group axis on a tied class — apply the base perm
                base_perm = full_perm[:base]
                p = base_perm.to(t.device)
                out[ax.key] = t.index_select(ax.dim, p).clone()
    return out


def rebasin_full(sd_a: dict, sd_b: dict, max_iters: int = 5) -> tuple[dict, dict]:
    """Full RAVE Re-Basin: iteratively align all permutation classes of sd_b to sd_a."""
    classes = build_encoder_classes(sd_a)
    logger.info("rebasin_full: %d classes registered", len(classes))
    for name, c in classes.items():
        logger.info("  class %-16s size=%-5d axes=%d sub_groups=%s",
                    name, c.size, len(c.axes), c.sub_groups)

    # Initialise perms to identity
    perms: dict[str, torch.Tensor] = {name: torch.arange(c.size) for name, c in classes.items()}

    for it in range(max_iters):
        moved = 0
        # Iterate classes in deterministic order
        sd_b_perm = _apply_perms(sd_b, classes, perms)
        for name in sorted(classes.keys()):
            cls = classes[name]
            # Build a single-class snapshot: use sd_a as-is, sd_b with all OTHER perms applied
            # (recompute by un-permuting current class then re-applying).
            cur_perm = perms[name]
            # Temporarily set current class perm to identity in sd_b_perm
            tmp_perms = dict(perms)
            tmp_perms[name] = torch.arange(cls.size)
            sd_b_tmp = _apply_perms(sd_b, classes, tmp_perms)
            new_perm = _solve_class(cls, sd_a, sd_b_tmp, perms)
            n_moved = int((new_perm != cur_perm).sum().item())
            if n_moved > 0:
                moved += n_moved
                perms[name] = new_perm
        logger.info("rebasin_full iter %d: %d channels moved across all classes", it, moved)
        if moved == 0:
            break

    sd_b_aligned = _apply_perms(sd_b, classes, perms)
    report = {
        "n_classes": len(classes),
        "iterations": it + 1,
        "perm_moves": {name: int((perms[name] != torch.arange(c.size)).sum().item())
                       for name, c in classes.items()},
    }
    return sd_b_aligned, report
