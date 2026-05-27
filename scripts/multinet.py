"""Multinet runner: chains MVP-A through MVP-I into 3 example networks.

Net 1 — Crystal Cathedral (5-bus parallel mix)
Net 2 — Recursive Organ (G -> A -> C -> F with re-injection passes)
Net 3 — Decoding Chamber (linear 9-stage)

Loads RAVE (shared by A/D/E/F/G) + EnCodec (shared by C/H/I) once.
Saves intermediate + final wavs to runs/multinet/{net1,net2,net3}/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.core.mix import MixConfig, dry_wet_mix, match_rms, soft_limiter
from src.core.texture_governor import TextureGovernorConfig
from src.modules.mvp_a.latent_perturb import PerturbParams
from src.modules.mvp_a.rave_io import RaveHandle, load_audio_mono, load_rave, save_audio_mono
from src.modules.mvp_a.render import RenderConfig as ARenderCfg
from src.modules.mvp_a.render import render as a_render
from src.modules.mvp_c.codec_io import CodecHandle, load_codec
from src.modules.mvp_c.render import RenderConfig as CRenderCfg
from src.modules.mvp_c.render import render as c_render
from src.modules.mvp_c.token_bend import BendParams
from src.modules.mvp_d.ckpt_morph import MorphParams
from src.modules.mvp_d.morph_io import load_morph
from src.modules.mvp_e.latent_granular import GranularParams
from src.modules.mvp_e.render import RenderConfig as ERenderCfg
from src.modules.mvp_e.render import render as e_render
from src.modules.mvp_f.latent_freeze import FreezeParams
from src.modules.mvp_f.render import RenderConfig as FRenderCfg
from src.modules.mvp_f.render import render as f_render
from src.modules.mvp_g.latent_feedback import FeedbackParams
from src.modules.mvp_g.render import RenderConfig as GRenderCfg
from src.modules.mvp_g.render import render as g_render
from src.modules.mvp_h.codebook_organ import OrganParams
from src.modules.mvp_h.render import render_generative as h_render
from src.modules.mvp_i.bass_massive import MassiveParams
from src.modules.mvp_i.render import RenderConfig as IRenderCfg
from src.modules.mvp_i.render import render as i_render


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("multinet")

SEED_PATH = Path("runs/multinet/sine_440_10s.wav")
OUT_ROOT = Path("runs/multinet")
RAVE_GUITAR = "checkpoints/rave/guitar_iil_b2048_r48000_z16.ts"
RAVE_ORGAN = "checkpoints/rave/organ_archive_b2048_r48000_z16.ts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _governor(min_wet=0.10) -> TextureGovernorConfig:
    return TextureGovernorConfig(enabled=True, min_wet=min_wet, flatness_max=0.55,
                                 rms_min=1e-4, rms_max=0.85, centroid_max_ratio=0.42)


def _mix(dw=0.7, rms_match=True, low_anchor_hz=None, sub_boost_db=0.0,
         limiter_drive=1.0) -> MixConfig:
    return MixConfig(dry_wet=dw, rms_match=rms_match, limiter=True,
                     limiter_drive=limiter_drive,
                     low_anchor_hz=low_anchor_hz, sub_boost_db=sub_boost_db)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_A(audio, rave, *, noise=0.05, drop=0.0, mode="smoothed", dw=0.7, rng_seed=0):
    cfg = ARenderCfg(mix=_mix(dw=dw), governor=_governor())
    params = PerturbParams(noise_scale=noise, dim_dropout=drop, noise_mode=mode,
                           noise_smooth=0.98)
    return a_render(audio, rave, params, cfg, np.random.default_rng(rng_seed))


def stage_D(audio, morph, *, dw=0.7, rng_seed=0):
    cfg = ARenderCfg(mix=_mix(dw=dw), governor=_governor())
    return a_render(audio, morph.handle, PerturbParams(), cfg, np.random.default_rng(rng_seed))


def stage_C(audio, codec, *, mode="bit_flip", rate=0.03, q_range=None,
            shuffle_window=12, dw=0.7, rng_seed=0):
    cfg = CRenderCfg(mix=_mix(dw=dw), governor=_governor())
    params = BendParams(mode=mode, rate=rate, quantizer_range=q_range,
                        shuffle_window=shuffle_window, codebook_size=codec.codebook_size)
    return c_render(audio, codec, params, cfg, np.random.default_rng(rng_seed))


def stage_E(audio, rave, *, grain=16, mem=2048, num=4, mix_amt=0.5, dw=0.6, rng_seed=0):
    cfg = ERenderCfg(mix=_mix(dw=dw), governor=_governor())
    params = GranularParams(grain_size=grain, memory_size=mem, num_grains=num, mix=mix_amt)
    return e_render(audio, rave, params, cfg, np.random.default_rng(rng_seed))


def stage_F(audio, rave, *, auto_upper=0.5, update_interval=128, cross=16, dw=0.6, rng_seed=0):
    cfg = FRenderCfg(mix=_mix(dw=dw), governor=_governor())
    params = FreezeParams(auto_upper_fraction=auto_upper, freeze_active=True,
                          update_interval_frames=update_interval, crossfade_frames=cross)
    return f_render(audio, rave, params, cfg, np.random.default_rng(rng_seed))


def stage_G(audio, rave, *, delay=32, feedback=0.4, mix_amt=0.5, dw=0.6, rng_seed=0):
    cfg = GRenderCfg(mix=_mix(dw=dw), governor=_governor())
    params = FeedbackParams(delay_frames=delay, feedback=feedback, mix=mix_amt)
    return g_render(audio, rave, params, cfg, np.random.default_rng(rng_seed))


def stage_H(codec, *, mode="prime", base=0, stride=7, frames=750, rng_seed=0):
    params = OrganParams(mode=mode, base_token=base, stride=stride, duration_frames=frames)
    return h_render(codec, params, np.random.default_rng(rng_seed))


def stage_B(audio, rave, sr, *, depth=3, mutation_prob=0.5, rng_seed=0):
    """Caption->TTA recursive loop wrapper (stub backend).

    Returns audio of equal length to input. The B render uses the stub
    captioner/synth (no HF download). Final synthesized audio is resampled
    to target sample rate + length-matched to input.
    """
    from src.modules.mvp_b.caption_loop import LoopParams, run_loop
    from src.modules.mvp_b.models import make_stub_caption, make_stub_synth

    # B's models operate at 16k. Downsample input to 16k for captioner.
    in_len = len(audio)
    in_16k = _resample_to(audio, int(in_len * 16000 / sr))

    cap = make_stub_caption(sample_rate=16000)
    syn = make_stub_synth(sample_rate=16000, duration_s=in_len / sr)
    history = run_loop(in_16k, cap, syn,
                       LoopParams(depth=depth, text_mutation_prob=mutation_prob,
                                  seed=rng_seed))
    final_text, final_audio_16k = history[-1]
    log.info("stage_B final text=%r", final_text[:80])
    out = _resample_to(final_audio_16k, in_len)
    return out.astype(np.float32)


def stage_I(audio, codec, *, smear=12, jitter=0.05, fold=0.0,
            smear_q=(0, 2), jitter_q=(0, 3), dw=0.6, rng_seed=0):
    cfg = IRenderCfg(mix=_mix(dw=dw), governor=_governor())
    params = MassiveParams(smear_delay=smear, smear_quantizers=smear_q,
                           jitter_rate=jitter, jitter_quantizers=jitter_q,
                           fold_leak_rate=fold)
    return i_render(audio, codec, params, cfg, np.random.default_rng(rng_seed))


# ---------------------------------------------------------------------------
# Mix utilities
# ---------------------------------------------------------------------------

def _resample_to(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) == n:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return np.zeros(n, dtype=np.float32)
    # Simple linear stretch
    src = np.arange(len(audio), dtype=np.float32)
    tgt = np.linspace(0, len(audio) - 1, n, dtype=np.float32)
    return np.interp(tgt, src, audio).astype(np.float32)


def _to_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Pad or truncate to length n without resampling."""
    if len(audio) >= n:
        return audio[:n].astype(np.float32)
    return np.concatenate([audio, np.zeros(n - len(audio), dtype=np.float32)]).astype(np.float32)


def mix_buses(buses: dict[str, tuple[np.ndarray, float]], n: int,
              sr: int, anchor_hz: float | None = None, anchor_boost_db: float = 0.0,
              limiter_drive: float = 1.1) -> np.ndarray:
    """Weighted sum of named buses → soft-limited master."""
    out = np.zeros(n, dtype=np.float32)
    for name, (audio, weight) in buses.items():
        a = _to_length(audio, n)
        peak = float(np.max(np.abs(a)))
        if peak > 1e-6:
            a = a / peak * 0.95
        out += weight * a
        log.info("bus %s: weight=%.2f, in_rms=%.4f, peak=%.4f",
                 name, weight, float(np.sqrt(np.mean(a**2))), peak)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if anchor_hz is not None and anchor_boost_db != 0.0:
        from scipy.signal import butter, lfilter
        b_lo, a_lo = butter(2, anchor_hz, fs=sr, btype="low")
        b_hi, a_hi = butter(2, anchor_hz, fs=sr, btype="high")
        low = lfilter(b_lo, a_lo, out).astype(np.float32) * (10 ** (anchor_boost_db / 20.0))
        high = lfilter(b_hi, a_hi, out).astype(np.float32)
        out = low + high
    out = soft_limiter(out, drive=limiter_drive)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-6:
        out = out / peak * 0.95
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_all():
    log.info("loading RAVE guitar on %s", DEVICE)
    rave_g = load_rave(RAVE_GUITAR, device=DEVICE)
    log.info("loading morph guitar<->organ on %s", DEVICE)
    morph = load_morph([RAVE_GUITAR, RAVE_ORGAN],
                       MorphParams(mode="linear", t=0.005),
                       device=DEVICE, rebasin=False)
    log.info("loading EnCodec on %s", DEVICE)
    codec = load_codec("encodec_24khz", device=DEVICE, bandwidth=6.0)
    return rave_g, morph, codec


# ---------------------------------------------------------------------------
# Net 1 — Crystal Cathedral
# ---------------------------------------------------------------------------

def net1(seed: np.ndarray, sr: int, rave: RaveHandle, morph, codec: CodecHandle):
    out_dir = OUT_ROOT / "net1"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(seed)

    # ---- L bus: I on 24k seed (downsample needed)
    log.info("[N1 L] EnCodec 24k bass-massive")
    seed_24k = _resample_to(seed, int(n * 24000 / sr))
    L_24k = stage_I(seed_24k, codec, smear=12, jitter=0.05, smear_q=(0, 2),
                    jitter_q=(0, 3), dw=0.65)
    L = _resample_to(L_24k, n)
    sf.write(out_dir / "bus_L_bass.wav", L, sr)

    # ---- M bus: A -> D -> C
    log.info("[N1 M] A -> D -> C")
    seed_24k_m = _resample_to(seed, int(n * 24000 / sr))
    m1 = stage_A(seed, rave, noise=0.05, mode="smoothed", dw=0.65)
    m2 = stage_D(m1, morph, dw=0.6)
    m2_24k = _resample_to(m2, int(n * 24000 / sr))
    m3_24k = stage_C(m2_24k, codec, mode="bit_flip", rate=0.03,
                     q_range=(-3, 0), shuffle_window=12, dw=0.5)
    M = _resample_to(m3_24k, n)
    sf.write(out_dir / "bus_M_core.wav", M, sr)

    # ---- H bus: F -> A
    log.info("[N1 H] F -> A(high)")
    h1 = stage_F(seed, rave, auto_upper=0.5, update_interval=128, cross=16, dw=0.55)
    H = stage_A(h1, rave, noise=0.10, mode="smoothed", dw=0.55)
    sf.write(out_dir / "bus_H_shimmer.wav", H, sr)

    # ---- T bus: G -> E
    log.info("[N1 T] G -> E")
    t1 = stage_G(seed, rave, delay=32, feedback=0.40, mix_amt=0.5, dw=0.55)
    T = stage_E(t1, rave, grain=16, mem=2048, num=4, mix_amt=0.5, dw=0.55)
    sf.write(out_dir / "bus_T_recursive.wav", T, sr)

    # ---- D bus: H drone
    log.info("[N1 D] H drone bed")
    D_24k = stage_H(codec, mode="prime", base=0, stride=7, frames=750)
    D = _resample_to(D_24k, n)
    sf.write(out_dir / "bus_D_drone.wav", D, sr)

    # ---- Master mix
    master = mix_buses(
        {"L": (L, 0.35), "M": (M, 0.30), "H": (H, 0.20),
         "T": (T, 0.10), "D": (D, 0.05)},
        n, sr, anchor_hz=80.0, anchor_boost_db=6.0, limiter_drive=1.2,
    )
    sf.write(out_dir / "MASTER.wav", master, sr)
    log.info("[N1] master rms=%.4f, peak=%.4f",
             float(np.sqrt(np.mean(master**2))), float(np.max(np.abs(master))))
    return master


# ---------------------------------------------------------------------------
# Net 2 — Recursive Organ
# ---------------------------------------------------------------------------

def net2(seed: np.ndarray, sr: int, rave: RaveHandle, morph, codec: CodecHandle,
         passes: int = 3):
    out_dir = OUT_ROOT / "net2"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(seed)
    cur = seed.copy()
    for p in range(passes):
        log.info("[N2 pass %d] G -> A -> C -> F", p)
        x = stage_G(cur, rave, delay=64, feedback=0.55, mix_amt=0.7, dw=0.55,
                    rng_seed=p)
        x = stage_A(x, rave, noise=0.08, drop=0.15, mode="white", dw=0.55,
                    rng_seed=10 + p)
        x_24k = _resample_to(x, int(n * 24000 / sr))
        x_24k = stage_C(x_24k, codec, mode="shuffle", rate=0.04,
                        shuffle_window=12, dw=0.50, rng_seed=20 + p)
        x = _resample_to(x_24k, n)
        x = stage_F(x, rave, auto_upper=0.3, update_interval=64, cross=12,
                    dw=0.55, rng_seed=30 + p)
        sf.write(out_dir / f"pass_{p}.wav", x, sr)
        cur = x
    master = soft_limiter(cur, drive=1.15)
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master = master / peak * 0.95
    sf.write(out_dir / "MASTER.wav", master, sr)
    log.info("[N2] master rms=%.4f", float(np.sqrt(np.mean(master**2))))
    return master


# ---------------------------------------------------------------------------
# Net 3 — Decoding Chamber (linear 9-stage)
# ---------------------------------------------------------------------------

def net3(seed: np.ndarray, sr: int, rave: RaveHandle, morph, codec: CodecHandle):
    out_dir = OUT_ROOT / "net3"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(seed)
    sub_dw = 0.45  # accumulate damage gently

    log.info("[N3] stage 1 A noise=0.03")
    x = stage_A(seed, rave, noise=0.03, mode="smoothed", dw=sub_dw)
    sf.write(out_dir / "s1_A.wav", x, sr)

    log.info("[N3] stage 2 D morph t=0.005")
    x = stage_D(x, morph, dw=sub_dw)
    sf.write(out_dir / "s2_D.wav", x, sr)

    log.info("[N3] stage 3 E granular mem=1024")
    x = stage_E(x, rave, grain=16, mem=1024, num=4, mix_amt=0.5, dw=sub_dw)
    sf.write(out_dir / "s3_E.wav", x, sr)

    log.info("[N3] stage 4 F freeze 25%")
    x = stage_F(x, rave, auto_upper=0.25, update_interval=128, cross=16, dw=sub_dw)
    sf.write(out_dir / "s4_F.wav", x, sr)

    log.info("[N3] stage 5 G delay=24 fb=0.30")
    x = stage_G(x, rave, delay=24, feedback=0.30, mix_amt=0.5, dw=sub_dw)
    sf.write(out_dir / "s5_G.wav", x, sr)

    log.info("[N3] stage 6 C bit_flip upper rate=0.02")
    x_24k = _resample_to(x, int(n * 24000 / sr))
    x_24k = stage_C(x_24k, codec, mode="bit_flip", rate=0.02,
                    q_range=(-3, 0), shuffle_window=8, dw=sub_dw)
    x = _resample_to(x_24k, n)
    sf.write(out_dir / "s6_C.wav", x, sr)

    log.info("[N3] stage 7 I bass smear")
    x_24k = _resample_to(x, int(n * 24000 / sr))
    x_24k = stage_I(x_24k, codec, smear=8, jitter=0.03, smear_q=(0, 2),
                    jitter_q=(0, 2), dw=sub_dw)
    x = _resample_to(x_24k, n)
    sf.write(out_dir / "s7_I.wav", x, sr)

    log.info("[N3] stage 8 80Hz crossover anchor +8dB")
    from scipy.signal import butter, lfilter
    b_lo, a_lo = butter(2, 80.0, fs=sr, btype="low")
    b_hi, a_hi = butter(2, 80.0, fs=sr, btype="high")
    low = lfilter(b_lo, a_lo, seed).astype(np.float32) * (10 ** (8.0 / 20.0))
    high = lfilter(b_hi, a_hi, x).astype(np.float32)
    x = low + high

    master = soft_limiter(x, drive=1.2)
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master = master / peak * 0.95
    sf.write(out_dir / "MASTER.wav", master, sr)
    log.info("[N3] master rms=%.4f", float(np.sqrt(np.mean(master**2))))
    return master


# ---------------------------------------------------------------------------
# Net Max — Cathedral Hive (uses all A..I, multi-bus + cross-feedback + 2-pass)
# ---------------------------------------------------------------------------

def _morph_organ(rave, paths, t, device=DEVICE):
    """Load morph with given t value. Re-loads each call (rave_handle cached)."""
    return load_morph(paths, MorphParams(mode="linear", t=t), device=device, rebasin=False)


def net_max(seed: np.ndarray, sr: int, rave: RaveHandle, morph_guitar,
            codec: CodecHandle, paths: list[str]):
    """Cathedral Hive — 8 buses + cross-feedback + 2-pass macro loop.

    Buses:
      α FOUNDATION : I -> C(lower-q invalid) + 80Hz LPF + sub-boost
      β CORE       : A -> D(guitar) -> E(memory) -> G(echo) -> A(drop) -> F
      γ GHOST      : F(freeze) -> G(echo) -> C(shuffle window) -> F(freeze)
      δ TWIN       : D(organ side, t=0.995) -> A(noise) -> C(bit_flip upper) -> E
      ε GLITCH     : C(bit_flip rate=0.10) -> I(smear+jitter+fold) -> G
      ζ DRONE      : H(prime) + H(fibonacci) blended + A pass
      η LOOP-B     : B caption->TTA recursive depth=3
      θ XFB        : CORE output -> G(deep) -> I(fold) (cross-bus feedback)

    Pass 1 produces master_1. Pass 2 feeds master_1 back into β/γ as seed,
    mixing with the original sine. Final master = mix(master_1, master_2).
    """
    out_dir = OUT_ROOT / "net_max"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(seed)
    sr_codec = codec.sample_rate

    # Pre-build the organ-end morph (t=0.995, near pure organ for distinct texture)
    log.info("[NMAX] loading morph t=0.995 (organ-end)")
    morph_organ = _morph_organ(rave, paths, t=0.995)

    def to_codec(a):
        return _resample_to(a, int(len(a) * sr_codec / sr))

    def from_codec(a):
        return _resample_to(a, n)

    # Pass-builder: applies the buses on a given input
    def run_buses(seed_in: np.ndarray, pass_idx: int) -> dict[str, np.ndarray]:
        buses: dict[str, np.ndarray] = {}

        # α FOUNDATION
        log.info("[NMAX p%d α] I(low smear) -> C(lower q) + 80Hz anchor", pass_idx)
        a_c = to_codec(seed_in)
        a_i = stage_I(a_c, codec, smear=20, jitter=0.06, fold=0.03,
                      smear_q=(0, 2), jitter_q=(0, 3), dw=0.65,
                      rng_seed=100 + pass_idx)
        a_c2 = stage_C(a_i, codec, mode="invalid_token", rate=0.03,
                       q_range=(0, 2), shuffle_window=8, dw=0.55,
                       rng_seed=101 + pass_idx)
        alpha = from_codec(a_c2)
        # 80Hz crossover: keep original low + boost
        from scipy.signal import butter, lfilter
        b_lo, a_lo = butter(2, 80.0, fs=sr, btype="low")
        b_hi, a_hi = butter(2, 80.0, fs=sr, btype="high")
        low = lfilter(b_lo, a_lo, seed_in).astype(np.float32) * (10 ** (10.0 / 20.0))
        high = lfilter(b_hi, a_hi, alpha).astype(np.float32)
        alpha = (low + high).astype(np.float32)
        sf.write(out_dir / f"p{pass_idx}_bus_α_foundation.wav", alpha, sr)
        buses["α"] = alpha

        # β CORE (deep linear chain)
        log.info("[NMAX p%d β] A->D(guitar)->E->G->A(drop)->F", pass_idx)
        b1 = stage_A(seed_in, rave, noise=0.06, mode="smoothed", dw=0.60,
                     rng_seed=200 + pass_idx)
        b2 = stage_D(b1, morph_guitar, dw=0.60, rng_seed=201 + pass_idx)
        b3 = stage_E(b2, rave, grain=24, mem=4096, num=5, mix_amt=0.55, dw=0.55,
                     rng_seed=202 + pass_idx)
        b4 = stage_G(b3, rave, delay=48, feedback=0.45, mix_amt=0.55, dw=0.55,
                     rng_seed=203 + pass_idx)
        b5 = stage_A(b4, rave, noise=0.0, drop=0.20, mode="white", dw=0.55,
                     rng_seed=204 + pass_idx)
        beta = stage_F(b5, rave, auto_upper=0.30, update_interval=128, cross=16,
                       dw=0.55, rng_seed=205 + pass_idx)
        sf.write(out_dir / f"p{pass_idx}_bus_β_core.wav", beta, sr)
        buses["β"] = beta

        # γ GHOST (freeze chain)
        log.info("[NMAX p%d γ] F -> G -> C(shuffle) -> F", pass_idx)
        g1 = stage_F(seed_in, rave, auto_upper=0.6, update_interval=64, cross=12,
                     dw=0.55, rng_seed=300 + pass_idx)
        g2 = stage_G(g1, rave, delay=24, feedback=0.35, mix_amt=0.5, dw=0.55,
                     rng_seed=301 + pass_idx)
        g3_c = stage_C(to_codec(g2), codec, mode="shuffle", rate=0.04,
                       shuffle_window=12, dw=0.50, q_range=(-4, 0),
                       rng_seed=302 + pass_idx)
        g3 = from_codec(g3_c)
        gamma = stage_F(g3, rave, auto_upper=0.40, update_interval=96, cross=20,
                        dw=0.55, rng_seed=303 + pass_idx)
        sf.write(out_dir / f"p{pass_idx}_bus_γ_ghost.wav", gamma, sr)
        buses["γ"] = gamma

        # δ TWIN (organ-side morph)
        log.info("[NMAX p%d δ] D(organ) -> A -> C(bit_flip upper) -> E", pass_idx)
        d1 = stage_D(seed_in, morph_organ, dw=0.55, rng_seed=400 + pass_idx)
        d2 = stage_A(d1, rave, noise=0.18, drop=0.10, mode="white", dw=0.55,
                     rng_seed=401 + pass_idx)
        d3_c = stage_C(to_codec(d2), codec, mode="bit_flip", rate=0.05,
                       q_range=(-3, 0), shuffle_window=10, dw=0.50,
                       rng_seed=402 + pass_idx)
        d3 = from_codec(d3_c)
        delta = stage_E(d3, rave, grain=8, mem=1024, num=3, mix_amt=0.40,
                        dw=0.55, rng_seed=403 + pass_idx)
        sf.write(out_dir / f"p{pass_idx}_bus_δ_twin.wav", delta, sr)
        buses["δ"] = delta

        # ε GLITCH (heavy codec damage)
        log.info("[NMAX p%d ε] C(heavy) -> I(fold) -> G", pass_idx)
        e1_c = stage_C(to_codec(seed_in), codec, mode="bit_flip", rate=0.10,
                       q_range=(-4, 0), shuffle_window=6, dw=0.65,
                       rng_seed=500 + pass_idx)
        e2_c = stage_I(e1_c, codec, smear=15, jitter=0.10, fold=0.05,
                       smear_q=(0, 3), jitter_q=(0, 4), dw=0.65,
                       rng_seed=501 + pass_idx)
        e2 = from_codec(e2_c)
        epsilon = stage_G(e2, rave, delay=16, feedback=0.40, mix_amt=0.6,
                          dw=0.60, rng_seed=502 + pass_idx)
        sf.write(out_dir / f"p{pass_idx}_bus_ε_glitch.wav", epsilon, sr)
        buses["ε"] = epsilon

        # ζ DRONE (generative bed)
        log.info("[NMAX p%d ζ] H(prime) + H(fibonacci) -> A", pass_idx)
        frames_target = int(n * 75 / sr)  # 75fps for encodec 24k
        z1_c = stage_H(codec, mode="prime", base=0, stride=11,
                       frames=frames_target, rng_seed=600 + pass_idx)
        z2_c = stage_H(codec, mode="fibonacci", base=0, stride=5,
                       frames=frames_target, rng_seed=601 + pass_idx)
        # Mix two drone variants
        z_mix_c = 0.55 * _to_length(z1_c, len(z1_c)) + 0.45 * _to_length(z2_c, len(z1_c))
        z_mix = from_codec(z_mix_c)
        # Pass through RAVE for organic smoothing
        zeta = stage_A(z_mix, rave, noise=0.04, mode="smoothed", dw=0.55,
                       rng_seed=602 + pass_idx)
        sf.write(out_dir / f"p{pass_idx}_bus_ζ_drone.wav", zeta, sr)
        buses["ζ"] = zeta

        # η LOOP-B (caption recursive)
        log.info("[NMAX p%d η] B caption->TTA depth=3", pass_idx)
        eta = stage_B(seed_in, rave, sr, depth=3, mutation_prob=0.5,
                      rng_seed=700 + pass_idx)
        sf.write(out_dir / f"p{pass_idx}_bus_η_loopB.wav", eta, sr)
        buses["η"] = eta

        # θ XFB (cross-bus feedback: β output deep echo + fold)
        log.info("[NMAX p%d θ] β_out -> G(deep) -> I(fold)", pass_idx)
        x1 = stage_G(beta, rave, delay=96, feedback=0.55, mix_amt=0.7,
                     dw=0.60, rng_seed=800 + pass_idx)
        x2_c = stage_I(to_codec(x1), codec, smear=24, jitter=0.04, fold=0.08,
                       smear_q=(0, 2), jitter_q=(0, 4), dw=0.60,
                       rng_seed=801 + pass_idx)
        theta = from_codec(x2_c)
        sf.write(out_dir / f"p{pass_idx}_bus_θ_xfb.wav", theta, sr)
        buses["θ"] = theta

        return buses

    # ----- Pass 1 -----
    log.info("=== NMAX PASS 1 ===")
    buses_1 = run_buses(seed, pass_idx=1)
    weights_1 = {"α": 0.20, "β": 0.18, "γ": 0.12, "δ": 0.12,
                 "ε": 0.10, "ζ": 0.10, "η": 0.08, "θ": 0.10}
    master_1 = mix_buses({k: (buses_1[k], weights_1[k]) for k in buses_1},
                         n, sr, anchor_hz=80.0, anchor_boost_db=8.0,
                         limiter_drive=1.25)
    sf.write(out_dir / "MASTER_pass1.wav", master_1, sr)
    log.info("[NMAX] pass1 master rms=%.4f peak=%.3f",
             float(np.sqrt(np.mean(master_1**2))), float(np.max(np.abs(master_1))))

    # ----- Pass 2: master_1 mixed with original sine, refed into buses -----
    log.info("=== NMAX PASS 2 (refed master_1 + sine) ===")
    refed_seed = match_rms(seed, 0.55 * master_1 + 0.45 * seed).astype(np.float32)
    sf.write(out_dir / "p2_refed_seed.wav", refed_seed, sr)
    buses_2 = run_buses(refed_seed, pass_idx=2)
    # Weights shift: emphasize CORE + XFB + GHOST in pass 2
    weights_2 = {"α": 0.15, "β": 0.22, "γ": 0.15, "δ": 0.10,
                 "ε": 0.08, "ζ": 0.08, "η": 0.06, "θ": 0.16}
    master_2 = mix_buses({k: (buses_2[k], weights_2[k]) for k in buses_2},
                         n, sr, anchor_hz=80.0, anchor_boost_db=6.0,
                         limiter_drive=1.25)
    sf.write(out_dir / "MASTER_pass2.wav", master_2, sr)
    log.info("[NMAX] pass2 master rms=%.4f peak=%.3f",
             float(np.sqrt(np.mean(master_2**2))), float(np.max(np.abs(master_2))))

    # ----- Final master = blend pass1 + pass2 with slow time-varying crossfade -----
    log.info("=== NMAX FINAL BLEND (time-varying p1<->p2) ===")
    t_vec = np.linspace(0, 1, n, dtype=np.float32)
    # Smooth S-curve from p1 to p2 dominance
    alpha_curve = 0.5 - 0.5 * np.cos(np.pi * t_vec)
    final = (1 - alpha_curve) * master_1 + alpha_curve * master_2
    final = soft_limiter(final, drive=1.20)
    peak = float(np.max(np.abs(final)))
    if peak > 1e-6:
        final = final / peak * 0.95
    sf.write(out_dir / "MASTER_FINAL.wav", final, sr)
    log.info("[NMAX] FINAL master rms=%.4f peak=%.3f",
             float(np.sqrt(np.mean(final**2))), float(np.max(np.abs(final))))
    return final


# ---------------------------------------------------------------------------
# Net Dynamic — Tempest (time-varying bus envelopes + events + filter sweep)
# ---------------------------------------------------------------------------

def _env_segments(t_sec: np.ndarray, segments: list[tuple[float, float, float]],
                  smooth: float = 0.5) -> np.ndarray:
    """Piecewise-linear envelope from (t, value) breakpoints.

    `segments` = [(t_seconds, value), ...]. Linearly interp, then smooth via
    moving-average of `smooth` seconds.
    """
    ts = np.array([s[0] for s in segments], dtype=np.float32)
    vs = np.array([s[1] for s in segments], dtype=np.float32)
    env = np.interp(t_sec, ts, vs).astype(np.float32)
    if smooth > 0:
        from scipy.ndimage import uniform_filter1d
        w = max(1, int(smooth * len(t_sec) / t_sec[-1]))
        env = uniform_filter1d(env, size=w, mode="reflect")
    return env


def _moving_lowpass(audio: np.ndarray, sr: int, cutoff_hz: np.ndarray,
                    block_size: int = 4096) -> np.ndarray:
    """Time-varying lowpass: process in blocks, each with its own 2nd-order LP."""
    from scipy.signal import butter, lfilter, lfilter_zi
    out = np.zeros_like(audio, dtype=np.float32)
    n = len(audio)
    prev_zi = None
    for i in range(0, n, block_size):
        j = min(i + block_size, n)
        cut = float(np.clip(np.mean(cutoff_hz[i:j]), 20.0, sr / 2 - 100))
        b, a = butter(2, cut, fs=sr, btype="low")
        if prev_zi is None:
            zi = lfilter_zi(b, a) * audio[i]
        else:
            zi = prev_zi
        y, prev_zi = lfilter(b, a, audio[i:j], zi=zi)
        out[i:j] = y.astype(np.float32)
    return out


def net_dynamic(seed: np.ndarray, sr: int, rave: RaveHandle, morph_guitar,
                codec: CodecHandle, paths: list[str]):
    """Tempest — 8 buses, each with time-varying amplitude envelope + master
    filter sweep + 3 impulse/silence events for maximum sonic motion."""
    out_dir = OUT_ROOT / "net_dynamic"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(seed)
    dur = n / sr
    sr_codec = codec.sample_rate

    log.info("[NDYN] loading morph t=0.995 (organ-end)")
    morph_organ = _morph_organ(rave, paths, t=0.995)

    def to_codec(a):
        return _resample_to(a, int(len(a) * sr_codec / sr))

    def from_codec(a):
        return _resample_to(a, n)

    # ---- Render each bus once (full length, high-intensity variant) ----
    log.info("[NDYN] rendering 8 buses")
    buses: dict[str, np.ndarray] = {}

    # α FOUNDATION
    a_c = to_codec(seed)
    a_i = stage_I(a_c, codec, smear=20, jitter=0.06, fold=0.03,
                  smear_q=(0, 2), jitter_q=(0, 3), dw=0.65, rng_seed=100)
    a_c2 = stage_C(a_i, codec, mode="invalid_token", rate=0.03,
                   q_range=(0, 2), shuffle_window=8, dw=0.55, rng_seed=101)
    alpha = from_codec(a_c2)
    from scipy.signal import butter, lfilter
    b_lo, a_lo = butter(2, 80.0, fs=sr, btype="low")
    b_hi, a_hi = butter(2, 80.0, fs=sr, btype="high")
    low_seed = lfilter(b_lo, a_lo, seed).astype(np.float32) * (10 ** (10.0 / 20.0))
    high_alpha = lfilter(b_hi, a_hi, alpha).astype(np.float32)
    buses["α"] = (low_seed + high_alpha).astype(np.float32)
    sf.write(out_dir / "bus_α_foundation.wav", buses["α"], sr)

    # β CORE
    b1 = stage_A(seed, rave, noise=0.06, mode="smoothed", dw=0.60, rng_seed=200)
    b2 = stage_D(b1, morph_guitar, dw=0.60, rng_seed=201)
    b3 = stage_E(b2, rave, grain=24, mem=4096, num=5, mix_amt=0.55, dw=0.55, rng_seed=202)
    b4 = stage_G(b3, rave, delay=48, feedback=0.45, mix_amt=0.55, dw=0.55, rng_seed=203)
    b5 = stage_A(b4, rave, drop=0.20, mode="white", dw=0.55, rng_seed=204)
    buses["β"] = stage_F(b5, rave, auto_upper=0.30, update_interval=128, cross=16,
                         dw=0.55, rng_seed=205)
    sf.write(out_dir / "bus_β_core.wav", buses["β"], sr)

    # γ GHOST
    g1 = stage_F(seed, rave, auto_upper=0.6, update_interval=64, cross=12, dw=0.55,
                 rng_seed=300)
    g2 = stage_G(g1, rave, delay=24, feedback=0.35, mix_amt=0.5, dw=0.55, rng_seed=301)
    g3_c = stage_C(to_codec(g2), codec, mode="shuffle", rate=0.04, shuffle_window=12,
                   dw=0.50, q_range=(-4, 0), rng_seed=302)
    g3 = from_codec(g3_c)
    buses["γ"] = stage_F(g3, rave, auto_upper=0.40, update_interval=96, cross=20,
                         dw=0.55, rng_seed=303)
    sf.write(out_dir / "bus_γ_ghost.wav", buses["γ"], sr)

    # δ TWIN
    d1 = stage_D(seed, morph_organ, dw=0.55, rng_seed=400)
    d2 = stage_A(d1, rave, noise=0.18, drop=0.10, mode="white", dw=0.55, rng_seed=401)
    d3_c = stage_C(to_codec(d2), codec, mode="bit_flip", rate=0.05, q_range=(-3, 0),
                   shuffle_window=10, dw=0.50, rng_seed=402)
    d3 = from_codec(d3_c)
    buses["δ"] = stage_E(d3, rave, grain=8, mem=1024, num=3, mix_amt=0.40, dw=0.55,
                         rng_seed=403)
    sf.write(out_dir / "bus_δ_twin.wav", buses["δ"], sr)

    # ε GLITCH
    e1_c = stage_C(to_codec(seed), codec, mode="bit_flip", rate=0.10, q_range=(-4, 0),
                   shuffle_window=6, dw=0.65, rng_seed=500)
    e2_c = stage_I(e1_c, codec, smear=15, jitter=0.10, fold=0.05, smear_q=(0, 3),
                   jitter_q=(0, 4), dw=0.65, rng_seed=501)
    e2 = from_codec(e2_c)
    buses["ε"] = stage_G(e2, rave, delay=16, feedback=0.40, mix_amt=0.6, dw=0.60,
                         rng_seed=502)
    sf.write(out_dir / "bus_ε_glitch.wav", buses["ε"], sr)

    # ζ DRONE
    frames_target = int(n * 75 / sr)
    z1_c = stage_H(codec, mode="prime", base=0, stride=11, frames=frames_target,
                   rng_seed=600)
    z2_c = stage_H(codec, mode="fibonacci", base=0, stride=5, frames=frames_target,
                   rng_seed=601)
    z_mix_c = 0.55 * _to_length(z1_c, len(z1_c)) + 0.45 * _to_length(z2_c, len(z1_c))
    z_mix = from_codec(z_mix_c)
    buses["ζ"] = stage_A(z_mix, rave, noise=0.04, mode="smoothed", dw=0.55, rng_seed=602)
    sf.write(out_dir / "bus_ζ_drone.wav", buses["ζ"], sr)

    # η LOOP-B
    buses["η"] = stage_B(seed, rave, sr, depth=3, mutation_prob=0.5, rng_seed=700)
    sf.write(out_dir / "bus_η_loopB.wav", buses["η"], sr)

    # θ XFB
    x1 = stage_G(buses["β"], rave, delay=96, feedback=0.55, mix_amt=0.7, dw=0.60,
                 rng_seed=800)
    x2_c = stage_I(to_codec(x1), codec, smear=24, jitter=0.04, fold=0.08,
                   smear_q=(0, 2), jitter_q=(0, 4), dw=0.60, rng_seed=801)
    buses["θ"] = from_codec(x2_c)
    sf.write(out_dir / "bus_θ_xfb.wav", buses["θ"], sr)

    # ---- Build per-bus time envelopes (60s breakpoints) ----
    t_sec = np.linspace(0, dur, n, dtype=np.float32)
    log.info("[NDYN] building bus envelopes")
    envelopes = {
        # (t_sec, weight) breakpoints — composition arc:
        # 0-10s ground (α only), 10-20 ghost rises, 20-30 core dominates,
        # 30 SILENCE DROP, 35-45 glitch+drone, 45-55 loop-B+xfb climax,
        # 55-60 fade
        "α": _env_segments(t_sec, [(0, 0.45), (5, 0.45), (15, 0.40), (25, 0.30),
                                    (30, 0.25), (35, 0.35), (45, 0.30), (55, 0.25),
                                    (60, 0.20)], smooth=0.3),
        "β": _env_segments(t_sec, [(0, 0.0), (8, 0.0), (15, 0.30), (22, 0.55),
                                    (28, 0.50), (32, 0.10), (45, 0.05), (55, 0.10),
                                    (60, 0.05)], smooth=0.4),
        "γ": _env_segments(t_sec, [(0, 0.0), (8, 0.0), (10, 0.20), (15, 0.35),
                                    (22, 0.20), (32, 0.05), (45, 0.0), (52, 0.20),
                                    (60, 0.15)], smooth=0.3),
        "δ": _env_segments(t_sec, [(0, 0.0), (18, 0.0), (22, 0.20), (28, 0.30),
                                    (35, 0.05), (55, 0.10), (60, 0.0)], smooth=0.4),
        "ε": _env_segments(t_sec, [(0, 0.0), (32, 0.0), (35, 0.30), (40, 0.55),
                                    (47, 0.25), (55, 0.05), (60, 0.0)], smooth=0.3),
        "ζ": _env_segments(t_sec, [(0, 0.0), (30, 0.05), (35, 0.20), (45, 0.30),
                                    (55, 0.30), (60, 0.25)], smooth=0.5),
        "η": _env_segments(t_sec, [(0, 0.0), (42, 0.0), (45, 0.20), (50, 0.40),
                                    (55, 0.30), (60, 0.10)], smooth=0.3),
        "θ": _env_segments(t_sec, [(0, 0.0), (38, 0.0), (45, 0.15), (52, 0.35),
                                    (58, 0.45), (60, 0.35)], smooth=0.3),
    }

    # ---- Save envelope plot data as CSV ----
    env_stack = np.stack([envelopes[k] for k in "αβγδεζηθ"], axis=0)
    sample_idx = np.arange(0, n, sr // 2)  # 2Hz samples
    csv_lines = ["t_sec," + ",".join("αβγδεζηθ")]
    for si in sample_idx:
        vals = ",".join(f"{env_stack[i, si]:.3f}" for i in range(8))
        csv_lines.append(f"{si/sr:.2f},{vals}")
    (out_dir / "envelopes.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # ---- Apply per-bus envelopes + sum ----
    log.info("[NDYN] applying envelopes + summing")
    master = np.zeros(n, dtype=np.float32)
    for name, bus in buses.items():
        b = _to_length(bus, n)
        # Normalize each bus to peak 0.95 first
        peak = float(np.max(np.abs(b)))
        if peak > 1e-6:
            b = b / peak * 0.95
        master = master + envelopes[name] * b
        log.info("bus %s: max_env=%.3f, in_rms=%.4f",
                 name, float(envelopes[name].max()), float(np.sqrt(np.mean(b**2))))

    master = np.nan_to_num(master, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- IMPULSE EVENTS ----
    log.info("[NDYN] applying impulse events")
    # Event 1: freeze CLICK at t=15s (200ms burst of γ amplified)
    click_start = int(15.0 * sr)
    click_end = int(15.3 * sr)
    click = _to_length(buses["γ"], n)[click_start:click_end] * 2.0
    master[click_start:click_end] += click * 0.6

    # Event 2: SILENCE DROP at t=30-31s
    drop_start = int(30.0 * sr)
    drop_end = int(31.0 * sr)
    drop_env = np.linspace(1.0, 0.05, drop_end - drop_start, dtype=np.float32)
    master[drop_start:drop_end] *= drop_env
    # Hard hit recovery at end of drop (0.1s)
    recov_start = drop_end
    recov_end = int(31.1 * sr)
    if recov_end < n:
        recov_env = np.linspace(0.05, 1.0, recov_end - recov_start, dtype=np.float32)
        master[recov_start:recov_end] *= recov_env

    # Event 3: H DRONE BURST at t=45s (1s loud sub burst)
    burst_start = int(44.5 * sr)
    burst_end = int(45.5 * sr)
    burst_src = _to_length(buses["ζ"], n)[burst_start:burst_end] * 1.5
    # Hard envelope: 50ms attack, 850ms hold, 100ms release
    bn = burst_end - burst_start
    burst_env = np.ones(bn, dtype=np.float32)
    a_n = int(0.05 * sr); r_n = int(0.1 * sr)
    burst_env[:a_n] = np.linspace(0, 1, a_n)
    burst_env[-r_n:] = np.linspace(1, 0, r_n)
    master[burst_start:burst_end] += burst_src * burst_env * 0.7

    # ---- MASTER FILTER SWEEP (lowpass cutoff time-varying) ----
    log.info("[NDYN] applying master filter sweep")
    # Cutoff trajectory: 250Hz → 12kHz over 0-20s, hold 12kHz, dip to 500Hz at 30s,
    # rise to 16kHz at 40s, fall to 6kHz at 55s
    cutoff = _env_segments(t_sec,
                           [(0, 250), (10, 4000), (20, 12000), (28, 11000),
                            (30, 500), (32, 8000), (40, 16000), (50, 14000),
                            (55, 6000), (60, 4000)], smooth=0.4)
    # Apply lowpass with block-wise cutoff
    master = _moving_lowpass(master, sr, cutoff, block_size=4096)

    # ---- 80Hz crossover anchor (preserve dry sub) ----
    low = lfilter(b_lo, a_lo, seed).astype(np.float32) * (10 ** (8.0 / 20.0))
    high = lfilter(b_hi, a_hi, master).astype(np.float32)
    master = low + high

    # ---- Soft limit + normalize ----
    master = soft_limiter(master, drive=1.25)
    peak = float(np.max(np.abs(master)))
    if peak > 1e-6:
        master = master / peak * 0.95

    sf.write(out_dir / "MASTER_FINAL.wav", master, sr)
    log.info("[NDYN] FINAL master rms=%.4f peak=%.3f",
             float(np.sqrt(np.mean(master**2))), float(np.max(np.abs(master))))

    # Save a per-second RMS curve to confirm dynamics
    rms_curve = []
    for s in range(int(dur)):
        seg = master[s * sr:(s + 1) * sr]
        rms_curve.append(float(np.sqrt(np.mean(seg**2))))
    (out_dir / "master_rms_per_second.csv").write_text(
        "t_sec,rms\n" + "\n".join(f"{i},{v:.5f}" for i, v in enumerate(rms_curve)))
    log.info("[NDYN] rms curve min=%.4f max=%.4f mean=%.4f",
             min(rms_curve), max(rms_curve), float(np.mean(rms_curve)))

    return master


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which == "max":
        seed_path = Path("runs/multinet/sine_max_30s.wav")
    elif which == "dynamic":
        seed_path = Path("runs/multinet/sine_dynamic_60s.wav")
    else:
        seed_path = SEED_PATH
    seed, sr = sf.read(seed_path)
    seed = seed.astype(np.float32)
    log.info("seed loaded: %d samp @ %d Hz from %s", len(seed), sr, seed_path)

    rave, morph, codec = load_all()

    if which in ("all", "1", "net1"):
        log.info("=== Net 1 — Crystal Cathedral ===")
        net1(seed, sr, rave, morph, codec)

    if which in ("all", "2", "net2"):
        log.info("=== Net 2 — Recursive Organ ===")
        net2(seed, sr, rave, morph, codec, passes=3)

    if which in ("all", "3", "net3"):
        log.info("=== Net 3 — Decoding Chamber ===")
        net3(seed, sr, rave, morph, codec)

    if which in ("max", "all"):
        log.info("=== Net Max — Cathedral Hive ===")
        net_max(seed, sr, rave, morph, codec, [RAVE_GUITAR, RAVE_ORGAN])

    if which in ("dynamic", "all"):
        log.info("=== Net Dynamic — Tempest ===")
        net_dynamic(seed, sr, rave, morph, codec, [RAVE_GUITAR, RAVE_ORGAN])

    log.info("ALL DONE")


if __name__ == "__main__":
    main()
