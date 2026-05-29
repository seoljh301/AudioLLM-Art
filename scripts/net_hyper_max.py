"""Net Hyper-Max — 24 MVP composition.

Four 45-second movements, each sampling 6 MVPs from one or two
taxonomy quadrants. Final master = movements crossfaded at 3-second
boundaries. Total duration 180 s.

Movement plan
-------------
M1 [0, 45]    Genesis           Y · O · H · P · S · D    (∅→audio, foundation)
M2 [45, 90]   Latent surge      A · E · F · G · K · U    (latent mutators)
M3 [90, 135]  Token storm       C · I · J · L · V · R    (token mutators + concat)
M4 [135, 180] Final wash        M · N · T · Q · W · B    (concat + STFT + recursive)

Each movement renders its 6 MVPs in parallel (one bus each), sums them
with per-bus weights, and applies a soft limiter. Subsequent movements
use the prior master as their input seed (so signal evolves).

Output: runs/hyper_max/master_180s.wav  plus per-movement stems.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import multinet as mn  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hyper_max")

import os

USE_REAL = os.environ.get("HYPER_MAX_REAL", "0") == "1"
SUFFIX = "_real" if USE_REAL else ""
OUT_DIR = ROOT / "runs" / f"hyper_max{SUFFIX}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SR = 22050
SEG_SECONDS = 45.0
XFADE_SECONDS = 3.0

# Caption sequence log — populated by U/V/W stages, dumped at end
CAPTIONS_LOG: dict[str, dict] = {}


# ---------- helpers ----------

def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    import librosa
    return librosa.resample(audio.astype(np.float32),
                            orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)


def _fit_length(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) == n:
        return audio
    if len(audio) < n:
        return np.pad(audio, (0, n - len(audio))).astype(np.float32)
    return audio[:n].astype(np.float32)


def _soft_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak <= ceiling:
        return audio.astype(np.float32)
    return (audio * (ceiling / peak)).astype(np.float32)


def _sum_buses(buses: list[tuple[np.ndarray, float]], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    total_w = 0.0
    for audio, w in buses:
        out += w * _fit_length(audio, n)
        total_w += abs(w)
    if total_w > 0:
        out /= max(1.0, total_w)
    return _soft_limit(out)


def _crossfade(a: np.ndarray, b: np.ndarray, xfade_n: int) -> np.ndarray:
    if xfade_n <= 0:
        return np.concatenate([a, b]).astype(np.float32)
    head = a[:-xfade_n]
    tail_a = a[-xfade_n:]
    head_b = b[:xfade_n]
    rest_b = b[xfade_n:]
    t = np.linspace(0.0, np.pi, xfade_n, dtype=np.float32)
    fade_out = (0.5 + 0.5 * np.cos(t)).astype(np.float32)
    fade_in = (0.5 - 0.5 * np.cos(t)).astype(np.float32)
    mid = tail_a * fade_out + head_b * fade_in
    return np.concatenate([head, mid, rest_b]).astype(np.float32)


# ---------- Movement 1: Genesis ----------

def movement_genesis(dur_s: float) -> np.ndarray:
    n = int(dur_s * MASTER_SR)
    buses: list[tuple[np.ndarray, float]] = []

    # Y — phantom weight
    from src.modules.mvp_y.phantom_weight import PhantomParams
    from src.modules.mvp_y.render import render_phantom
    y_path = OUT_DIR / "M1_Y.wav"
    render_phantom([Path(mn.RAVE_GUITAR), Path(mn.RAVE_ORGAN)], y_path,
                   PhantomParams(morph_t=0.4, latent_mode="smoothed",
                                 latent_sigma=0.9, duration_s=dur_s, seed=1),
                   device=mn.DEVICE)
    y_audio, y_sr = sf.read(str(y_path))
    buses.append((_resample(y_audio, y_sr, MASTER_SR), 1.0))

    # O — latent drift
    from src.modules.mvp_o.latent_drift import DriftParams
    from src.modules.mvp_o.render import RenderConfig as ODrCfg, render_drift
    o_path = OUT_DIR / "M1_O.wav"
    render_drift(Path(mn.RAVE_GUITAR), o_path,
                 DriftParams(mode="sinusoid", sigma=0.8, smooth=0.985, seed=2),
                 ODrCfg(duration_s=dur_s, device=mn.DEVICE))
    o_audio, o_sr = sf.read(str(o_path))
    buses.append((_resample(o_audio, o_sr, MASTER_SR), 0.8))

    # H — codebook organ (gen)
    mn_codec = mn.load_codec("encodec_24khz", device=mn.DEVICE, bandwidth=6.0)
    h_audio = mn.stage_H(mn_codec, mode="fibonacci", base=100, stride=11,
                         frames=int(dur_s * 75), rng_seed=3)
    buses.append((_resample(h_audio, mn_codec.sample_rate, MASTER_SR), 0.7))

    # P — phase hallucinator
    from src.modules.mvp_p.phase_halluc import HallucParams, make_magnitude, griffin_lim
    p_params = HallucParams(mode="chord", duration_s=dur_s, sr=MASTER_SR,
                            chord_freqs_hz=(82.5, 110.0, 165.0, 220.0),
                            griffin_lim_iters=16, seed=4)
    p_audio = griffin_lim(make_magnitude(p_params), p_params)
    buses.append((p_audio, 0.6))

    # S — random prompt TTA (procedural)
    from src.modules.mvp_s.random_prompt import PromptParams, render_random_prompt
    s_path = OUT_DIR / "M1_S.wav"
    render_random_prompt(s_path,
                         PromptParams(duration_s=dur_s, sr=MASTER_SR,
                                      n_prompts=3, mix_mode="blend",
                                      custom_prompts=("warm pulsing organ",
                                                      "deep cavernous drone",
                                                      "harmonic series sweep"),
                                      use_real=USE_REAL, seed=5))
    s_audio, s_sr = sf.read(str(s_path))
    buses.append((_resample(s_audio, s_sr, MASTER_SR), 0.5))

    # D — ckpt morph (uses Y's audio as the input)
    morph = mn.load_morph([mn.RAVE_GUITAR, mn.RAVE_ORGAN],
                          mn.MorphParams(mode="linear", t=0.3),
                          device=mn.DEVICE, rebasin=False)
    d_in = _resample(y_audio, int(y_sr), morph.handle.sample_rate)
    d_audio = mn.stage_D(d_in, morph, dw=0.6, rng_seed=6)
    buses.append((_resample(d_audio, morph.handle.sample_rate, MASTER_SR), 0.4))

    return _sum_buses(buses, n)


# ---------- Movement 2: Latent surge ----------

def movement_latent(prior: np.ndarray, dur_s: float, rave) -> np.ndarray:
    n = int(dur_s * MASTER_SR)
    in_48k = _resample(prior[:n], MASTER_SR, rave.sample_rate)
    buses: list[tuple[np.ndarray, float]] = []

    # A
    a = mn.stage_A(in_48k, rave, noise=0.10, drop=0.0, mode="smoothed",
                   dw=0.7, rng_seed=11)
    buses.append((_resample(a, rave.sample_rate, MASTER_SR), 1.0))

    # E
    e = mn.stage_E(in_48k, rave, grain=24, mem=2048, num=3, mix_amt=0.6,
                   dw=0.6, rng_seed=12)
    buses.append((_resample(e, rave.sample_rate, MASTER_SR), 0.9))

    # F
    f = mn.stage_F(in_48k, rave, auto_upper=0.5, update_interval=96,
                   cross=12, dw=0.6, rng_seed=13)
    buses.append((_resample(f, rave.sample_rate, MASTER_SR), 0.8))

    # G
    g = mn.stage_G(in_48k, rave, delay=40, feedback=0.45, mix_amt=0.6,
                   dw=0.6, rng_seed=14)
    buses.append((_resample(g, rave.sample_rate, MASTER_SR), 0.8))

    # K — temporal warp on EnCodec tokens (torch tensor in/out), then decode
    mn_codec = mn.load_codec("encodec_24khz", device=mn.DEVICE, bandwidth=6.0)
    in_24k = _resample(prior[:n], MASTER_SR, mn_codec.sample_rate)
    from src.modules.mvp_c.codec_io import encode_audio, decode_tokens
    import torch as _torch
    tokens = encode_audio(mn_codec, in_24k)
    tokens_t = _torch.from_numpy(tokens.astype(np.int64))
    warped = mn.stage_K(tokens_t)
    k_audio = decode_tokens(mn_codec, warped.cpu().numpy().astype(np.int64))
    buses.append((_resample(k_audio, mn_codec.sample_rate, MASTER_SR), 0.7))

    # U — caption-steered latent
    from src.modules.mvp_u.cap_latent import CapLatentParams, render_cap_latent
    u_in_path = OUT_DIR / "M2_U_in.wav"
    sf.write(str(u_in_path), in_48k.astype(np.float32), rave.sample_rate)
    u_path = OUT_DIR / "M2_U.wav"
    u_res = render_cap_latent(u_in_path, u_path,
                      CapLatentParams(bias_strength=0.5, chunk_seconds=1.5,
                                      use_real=USE_REAL,
                                      model_path=mn.RAVE_GUITAR,
                                      device=mn.DEVICE, seed=15))
    CAPTIONS_LOG["M2_U"] = {
        "backend": u_res.get("backend"),
        "n_captions": u_res.get("n_captions"),
        "unique": u_res.get("unique_captions"),
        "captions": u_res.get("captions", []),
    }
    u_audio, sr = sf.read(str(u_path))
    buses.append((_resample(u_audio, sr, MASTER_SR), 0.7))

    return _sum_buses(buses, n)


# ---------- Movement 3: Token storm ----------

def movement_tokens(prior: np.ndarray, dur_s: float) -> np.ndarray:
    n = int(dur_s * MASTER_SR)
    mn_codec = mn.load_codec("encodec_24khz", device=mn.DEVICE, bandwidth=6.0)
    in_24k = _resample(prior[:n], MASTER_SR, mn_codec.sample_rate)
    buses: list[tuple[np.ndarray, float]] = []

    # C
    c = mn.stage_C(in_24k, mn_codec, mode="bit_flip", rate=0.04,
                   q_range=(0, 4), shuffle_window=8, dw=0.6, rng_seed=21)
    buses.append((_resample(c, mn_codec.sample_rate, MASTER_SR), 1.0))

    # I
    i = mn.stage_I(in_24k, mn_codec, smear=14, jitter=0.06, fold=0.05,
                   smear_q=(0, 2), jitter_q=(0, 3),
                   dw=0.6, rng_seed=22)
    buses.append((_resample(i, mn_codec.sample_rate, MASTER_SR), 0.9))

    # J — mimi stub
    j = mn.stage_J(in_24k, semantic_rate=0.06, acoustic_rate=0.10, rng_seed=23)
    buses.append((_resample(j, mn_codec.sample_rate, MASTER_SR), 0.7))

    # L — speculative restoration
    l = mn.stage_L(dur_s, MASTER_SR, intensity=2.0)
    buses.append((_fit_length(l, n), 0.5))

    # V — caption-conditioned tokens
    from src.modules.mvp_v.cap_tokens import CapTokenParams, render_cap_tokens
    v_in_path = OUT_DIR / "M3_V_in.wav"
    sf.write(str(v_in_path), in_24k.astype(np.float32), mn_codec.sample_rate)
    v_path = OUT_DIR / "M3_V.wav"
    v_res = render_cap_tokens(v_in_path, v_path,
                      CapTokenParams(codec="encodec_24khz", bandwidth=6.0,
                                     device=mn.DEVICE, chunk_seconds=1.5,
                                     walk_mode="prime", base_seed_offset=24,
                                     walk_strength=4, dry_wet=1.0,
                                     use_real=USE_REAL))
    CAPTIONS_LOG["M3_V"] = {
        "walk_mode": v_res.get("walk_mode"),
        "n_captions": v_res.get("n_captions"),
        "unique": v_res.get("unique_captions"),
        "captions": v_res.get("captions", []),
    }
    v_audio, sr = sf.read(str(v_path))
    buses.append((_resample(v_audio, sr, MASTER_SR), 0.8))

    # R — token musaicing  (corpus = prior, target = prior)
    from src.modules.mvp_r.token_musaicing import TokenMusaicParams
    from src.modules.mvp_r.render import RenderConfig as RCfg, render_token_musaicing
    corpus_path = OUT_DIR / "M3_R_corpus.wav"
    target_path = OUT_DIR / "M3_R_target.wav"
    sf.write(str(corpus_path), in_24k.astype(np.float32), mn_codec.sample_rate)
    sf.write(str(target_path),
             _resample(prior[:n // 2], MASTER_SR, mn_codec.sample_rate),
             mn_codec.sample_rate)
    r_path = OUT_DIR / "M3_R.wav"
    render_token_musaicing([corpus_path], target_path, r_path,
                           TokenMusaicParams(grain_size=8, stride=4,
                                             target_stride=8,
                                             dist="weighted_hamming",
                                             temperature=0.3, seed=25),
                           RCfg(codec="encodec_24khz", bandwidth=6.0,
                                device=mn.DEVICE, dry_wet=1.0))
    r_audio, sr = sf.read(str(r_path))
    buses.append((_resample(r_audio, sr, MASTER_SR), 0.6))

    return _sum_buses(buses, n)


# ---------- Movement 4: Final wash ----------

def movement_final(prior: np.ndarray, dur_s: float,
                   rave, all_prior: list[np.ndarray]) -> np.ndarray:
    n = int(dur_s * MASTER_SR)
    buses: list[tuple[np.ndarray, float]] = []

    # Q — phase scramble (on prior)
    from src.modules.mvp_q.phase_scramble import ScrambleParams
    from src.modules.mvp_q.render import RenderConfig as QCfg, render_scramble
    q_in_path = OUT_DIR / "M4_Q_in.wav"
    sf.write(str(q_in_path), prior[:n].astype(np.float32), MASTER_SR)
    q_path = OUT_DIR / "M4_Q.wav"
    render_scramble(q_in_path, q_path,
                    ScrambleParams(mode="ou", rate=0.7, smooth=0.93, seed=31),
                    QCfg(sample_rate=MASTER_SR, dry_wet=1.0))
    q_audio, sr = sf.read(str(q_path))
    buses.append((_resample(q_audio, sr, MASTER_SR), 1.0))

    # N — concatenator (corpus = M1, target = M2)
    from src.modules.mvp_n.particle_filter import PFParams
    from src.modules.mvp_n.render import ConcatRenderConfig, render_concatenator
    corp_path = OUT_DIR / "M4_N_corpus.wav"
    targ_path = OUT_DIR / "M4_N_target.wav"
    sf.write(str(corp_path), all_prior[0].astype(np.float32), MASTER_SR)
    sf.write(str(targ_path),
             _fit_length(all_prior[1], n // 3).astype(np.float32),
             MASTER_SR)
    n_path = OUT_DIR / "M4_N.wav"
    render_concatenator([corp_path], targ_path, n_path,
                        PFParams(P=60, p=5, pd=0.95, tau=8.0, L=8, seed=32),
                        ConcatRenderConfig(sample_rate=MASTER_SR, dry_wet=1.0))
    n_audio, sr = sf.read(str(n_path))
    buses.append((_resample(n_audio, sr, MASTER_SR), 0.7))

    # M — latent musaicing (corpus = M1+M2, target = M3)
    from src.modules.mvp_m.musaicing import MusaicingParams
    from src.modules.mvp_m.render import RenderConfig as MCfg, render_musaicing
    mc1 = OUT_DIR / "M4_M_corpus1.wav"
    mc2 = OUT_DIR / "M4_M_corpus2.wav"
    mt = OUT_DIR / "M4_M_target.wav"
    sf.write(str(mc1), all_prior[0].astype(np.float32), MASTER_SR)
    sf.write(str(mc2), all_prior[1].astype(np.float32), MASTER_SR)
    sf.write(str(mt), _fit_length(all_prior[2], n // 2).astype(np.float32),
             MASTER_SR)
    m_path = OUT_DIR / "M4_M.wav"
    render_musaicing([mc1, mc2], mt, m_path,
                     MusaicingParams(grain_size=8, stride=4, target_stride=8,
                                     temperature=0.15, seed=33),
                     MCfg(codec="encodec_24khz", device=mn.DEVICE, dry_wet=1.0))
    m_audio, sr = sf.read(str(m_path))
    buses.append((_resample(m_audio, sr, MASTER_SR), 0.6))

    # T — CLAP musaicing (procedural)
    from src.modules.mvp_t.clap_musaicing import ClapMusaicParams, render_clap_musaicing
    t_path = OUT_DIR / "M4_T.wav"
    render_clap_musaicing(
        [OUT_DIR / "M1_O.wav"],
        ["bright shimmer with metallic resonance",
         "deep cavernous drone",
         "warm pulsing organ"],
        t_path,
        ClapMusaicParams(grain_seconds=1.8, stride_seconds=0.9,
                         seg_seconds=dur_s / 3, temperature=0.2,
                         use_real=USE_REAL, sr=MASTER_SR,
                         crossfade_seconds=0.08, seed=34))
    t_audio, sr = sf.read(str(t_path))
    buses.append((_resample(t_audio, sr, MASTER_SR), 0.7))

    # W — caption→TTA recursive loop
    from src.modules.mvp_w.cap_loop import LoopParams, render_cap_loop
    w_in_path = OUT_DIR / "M4_W_in.wav"
    sf.write(str(w_in_path), prior[:n].astype(np.float32), MASTER_SR)
    w_dir = OUT_DIR / "M4_W"
    w_res = render_cap_loop(w_in_path, w_dir,
                    LoopParams(n_iters=3, duration_per_step_s=dur_s / 3,
                               keep_mix=0.1, sr=MASTER_SR, use_real=USE_REAL,
                               save_intermediates=False))
    CAPTIONS_LOG["M4_W"] = {
        "backend": w_res.get("backend"),
        "n_iters": w_res.get("n_iters"),
        "captions": w_res.get("captions", []),
    }
    w_audio, sr = sf.read(str(w_dir / "final.wav"))
    buses.append((_resample(w_audio, sr, MASTER_SR), 0.6))

    # B — caption loop (stub)
    b_in_48k = _resample(prior[:n], MASTER_SR, rave.sample_rate)
    b_audio = mn.stage_B(b_in_48k, rave, rave.sample_rate, depth=2,
                         mutation_prob=0.4, rng_seed=35)
    buses.append((_resample(b_audio, rave.sample_rate, MASTER_SR), 0.5))

    return _sum_buses(buses, n)


# ---------- Master assembly ----------

def main():
    t0 = time.time()
    log.info("Net Hyper-Max start. master_sr=%d, total=%ds",
             MASTER_SR, int(4 * SEG_SECONDS))

    log.info("[M1] Genesis")
    m1 = movement_genesis(SEG_SECONDS)
    sf.write(str(OUT_DIR / "movement_1_genesis.wav"),
             m1.astype(np.float32), MASTER_SR)

    log.info("[M2] Latent surge")
    rave = mn.load_rave(mn.RAVE_GUITAR, device=mn.DEVICE)
    m2 = movement_latent(m1, SEG_SECONDS, rave)
    sf.write(str(OUT_DIR / "movement_2_latent.wav"),
             m2.astype(np.float32), MASTER_SR)

    log.info("[M3] Token storm")
    m3 = movement_tokens(m2, SEG_SECONDS)
    sf.write(str(OUT_DIR / "movement_3_tokens.wav"),
             m3.astype(np.float32), MASTER_SR)

    log.info("[M4] Final wash")
    m4 = movement_final(m3, SEG_SECONDS, rave, [m1, m2, m3])
    sf.write(str(OUT_DIR / "movement_4_final.wav"),
             m4.astype(np.float32), MASTER_SR)

    log.info("crossfade master")
    xfade_n = int(XFADE_SECONDS * MASTER_SR)
    master = _crossfade(m1, m2, xfade_n)
    master = _crossfade(master, m3, xfade_n)
    master = _crossfade(master, m4, xfade_n)
    master = _soft_limit(master, 0.95)

    out_path = OUT_DIR / "master_180s.wav"
    sf.write(str(out_path), master.astype(np.float32), MASTER_SR)

    # Dump caption log
    import json as _json
    log_path = OUT_DIR / "captions.json"
    log_obj = {
        "variant": "real" if USE_REAL else "procedural",
        "wall_seconds": round(time.time() - t0, 1),
        "stages": CAPTIONS_LOG,
    }
    log_path.write_text(_json.dumps(log_obj, indent=2, ensure_ascii=False))
    log.info("captions dumped → %s", log_path)

    log.info("DONE. wrote %s (%.1fs)  wall=%.1fs",
             out_path, len(master) / MASTER_SR, time.time() - t0)


if __name__ == "__main__":
    main()
