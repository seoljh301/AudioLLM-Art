"""Gradio control GUI for AudioArt MVPs + Multinet + demo listener.

Runs on 127.0.0.1:7860 by default. Reuses the stage_X functions from
scripts/multinet.py so renders are direct (no separate OSC server needed).

Sections:
  Listen          — play any wav from runs/ (recursive)
  MVP-A latent    — noise/dropout/shuffle on a chosen RAVE checkpoint
  MVP-C bend      — EnCodec token corruption
  MVP-D morph     — checkpoint t + optional perturb chain
  MVP-E granular  — latent memory
  MVP-F freeze    — spectral freeze
  MVP-G feedback  — latent echo
  MVP-H organ     — generative codebook (no input)
  MVP-I bass      — lower RVQ smear/jitter
  Multinet        — run one of the 5 macro-nets on a seed
  Meta-Symphony   — run the full 3-min stereo composition
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import soundfile as sf

ROOT = Path("/home1/irteam/proj/AudioArt")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import multinet as mn  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gui")

RUNS = ROOT / "runs"
GUI_OUT = RUNS / "gui"
GUI_OUT.mkdir(parents=True, exist_ok=True)


# Lazy global loads — RAVE/EnCodec take a few seconds; only load on first use.
_state = {"rave": None, "morph_g": None, "morph_o": None, "codec": None}


def _ensure_models():
    if _state["rave"] is None:
        log.info("loading RAVE guitar...")
        _state["rave"] = mn.load_rave(mn.RAVE_GUITAR, device=mn.DEVICE)
    if _state["morph_g"] is None:
        log.info("loading morph guitar (t=0.005)...")
        _state["morph_g"] = mn.load_morph(
            [mn.RAVE_GUITAR, mn.RAVE_ORGAN],
            mn.MorphParams(mode="linear", t=0.005),
            device=mn.DEVICE, rebasin=False)
    if _state["morph_o"] is None:
        log.info("loading morph organ (t=0.995)...")
        _state["morph_o"] = mn.load_morph(
            [mn.RAVE_GUITAR, mn.RAVE_ORGAN],
            mn.MorphParams(mode="linear", t=0.995),
            device=mn.DEVICE, rebasin=False)
    if _state["codec"] is None:
        log.info("loading EnCodec...")
        _state["codec"] = mn.load_codec("encodec_24khz", device=mn.DEVICE,
                                         bandwidth=6.0)


def _save_out(audio: np.ndarray, sr: int, tag: str) -> str:
    ts = time.strftime("%H%M%S")
    path = GUI_OUT / f"{tag}_{ts}.wav"
    sf.write(str(path), audio.astype(np.float32), sr)
    return str(path)


def _load_seed_wav(path: str, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


# ---------------------------------------------------------------------------
# Listen tab
# ---------------------------------------------------------------------------

def list_wavs() -> list[str]:
    paths = []
    for d in [RUNS / "multinet", RUNS / "masterpiece", RUNS / "ulaanbaatar_epic_v2"]:
        if d.exists():
            for p in d.rglob("*.wav"):
                paths.append(str(p.relative_to(ROOT)))
    paths.sort()
    return paths


def play_pick(rel_path: str) -> str:
    return str(ROOT / rel_path)


# ---------------------------------------------------------------------------
# MVP render handlers
# ---------------------------------------------------------------------------

def run_A(seed_path, noise, drop, shuffle, mode, dw, seed):
    _ensure_models()
    from src.modules.mvp_a.latent_perturb import PerturbParams
    from src.modules.mvp_a.render import RenderConfig, render
    audio = _load_seed_wav(seed_path, _state["rave"].sample_rate)
    cfg = RenderConfig(mix=mn._mix(dw=dw), governor=mn._governor())
    out = render(audio, _state["rave"],
                 PerturbParams(noise_scale=noise, dim_dropout=drop,
                               dim_shuffle=bool(shuffle),
                               noise_mode=mode, noise_smooth=0.98),
                 cfg, np.random.default_rng(int(seed)))
    return _save_out(out, _state["rave"].sample_rate, "A")


def run_C(seed_path, mode, rate, q_lo, q_hi, shuffle_win, dw, seed):
    _ensure_models()
    sr = _state["codec"].sample_rate
    audio = _load_seed_wav(seed_path, sr)
    q_range = None if q_lo == 0 and q_hi == 0 else (int(q_lo), int(q_hi))
    out = mn.stage_C(audio, _state["codec"], mode=mode, rate=rate,
                     q_range=q_range, shuffle_window=int(shuffle_win),
                     dw=dw, rng_seed=int(seed))
    return _save_out(out, sr, "C")


def run_D(seed_path, t, mode, perturb_noise, perturb_drop, dw, seed):
    _ensure_models()
    # Build morph on-the-fly with chosen t
    morph = mn.load_morph(
        [mn.RAVE_GUITAR, mn.RAVE_ORGAN],
        mn.MorphParams(mode=mode, t=float(t)),
        device=mn.DEVICE, rebasin=False)
    sr = morph.handle.sample_rate
    audio = _load_seed_wav(seed_path, sr)
    from src.modules.mvp_a.latent_perturb import PerturbParams
    from src.modules.mvp_a.render import RenderConfig, render
    cfg = RenderConfig(mix=mn._mix(dw=dw), governor=mn._governor())
    out = render(audio, morph.handle,
                 PerturbParams(noise_scale=perturb_noise,
                               dim_dropout=perturb_drop,
                               noise_mode="smoothed", noise_smooth=0.98),
                 cfg, np.random.default_rng(int(seed)))
    return _save_out(out, sr, "D")


def run_E(seed_path, grain, mem, num, mix_amt, dw, seed):
    _ensure_models()
    sr = _state["rave"].sample_rate
    audio = _load_seed_wav(seed_path, sr)
    out = mn.stage_E(audio, _state["rave"],
                     grain=int(grain), mem=int(mem), num=int(num),
                     mix_amt=mix_amt, dw=dw, rng_seed=int(seed))
    return _save_out(out, sr, "E")


def run_F(seed_path, auto_upper, update_interval, cross, dw, seed):
    _ensure_models()
    sr = _state["rave"].sample_rate
    audio = _load_seed_wav(seed_path, sr)
    out = mn.stage_F(audio, _state["rave"],
                     auto_upper=auto_upper,
                     update_interval=int(update_interval),
                     cross=int(cross), dw=dw, rng_seed=int(seed))
    return _save_out(out, sr, "F")


def run_G(seed_path, delay, feedback, mix_amt, dw, seed):
    _ensure_models()
    sr = _state["rave"].sample_rate
    audio = _load_seed_wav(seed_path, sr)
    out = mn.stage_G(audio, _state["rave"],
                     delay=int(delay), feedback=feedback,
                     mix_amt=mix_amt, dw=dw, rng_seed=int(seed))
    return _save_out(out, sr, "G")


def run_H(mode, base, stride, frames, seed):
    _ensure_models()
    sr = _state["codec"].sample_rate
    out = mn.stage_H(_state["codec"], mode=mode, base=int(base),
                     stride=int(stride), frames=int(frames),
                     rng_seed=int(seed))
    return _save_out(out, sr, "H")


def run_I(seed_path, smear, jitter, fold, smear_lo, smear_hi, jitter_lo, jitter_hi,
          dw, seed):
    _ensure_models()
    sr = _state["codec"].sample_rate
    audio = _load_seed_wav(seed_path, sr)
    out = mn.stage_I(audio, _state["codec"],
                     smear=int(smear), jitter=jitter, fold=fold,
                     smear_q=(int(smear_lo), int(smear_hi)),
                     jitter_q=(int(jitter_lo), int(jitter_hi)),
                     dw=dw, rng_seed=int(seed))
    return _save_out(out, sr, "I")


# ---------------------------------------------------------------------------
# V2 render handlers (text-conditioned)
# ---------------------------------------------------------------------------

def run_Y(morph_t, latent_mode, sigma, dur, seed):
    from src.modules.mvp_y.phantom_weight import PhantomParams
    from src.modules.mvp_y.render import render_phantom
    out_path = GUI_OUT / f"Y_{time.strftime('%H%M%S')}.wav"
    params = PhantomParams(
        morph_t=float(morph_t), morph_mode="linear",
        latent_mode=str(latent_mode), latent_sigma=float(sigma),
        latent_smooth=0.985, duration_s=float(dur),
        seed=int(seed),
    )
    render_phantom([Path(mn.RAVE_GUITAR), Path(mn.RAVE_ORGAN)],
                   out_path, params, device=mn.DEVICE)
    return str(out_path)


def run_S(prompt_text, n_prompts, mix_mode, dur, use_real, seed):
    from src.modules.mvp_s.random_prompt import PromptParams, render_random_prompt
    out_path = GUI_OUT / f"S_{time.strftime('%H%M%S')}.wav"
    custom = [s.strip() for s in str(prompt_text).split("|") if s.strip()]
    params = PromptParams(
        duration_s=float(dur), sr=22050,
        n_prompts=int(n_prompts), mix_mode=str(mix_mode),
        custom_prompts=tuple(custom), permute_words=False,
        use_real=bool(use_real), seed=int(seed),
    )
    render_random_prompt(out_path, params)
    return str(out_path)


def run_T(seed_path, prompt_text, grain, seg, temp, use_real, seed):
    from src.modules.mvp_t.clap_musaicing import ClapMusaicParams, render_clap_musaicing
    out_path = GUI_OUT / f"T_{time.strftime('%H%M%S')}.wav"
    prompts = [s.strip() for s in str(prompt_text).split("|") if s.strip()]
    if not prompts:
        prompts = ["bright shimmer with metallic resonance"]
    params = ClapMusaicParams(
        grain_seconds=float(grain), stride_seconds=float(grain) / 2,
        seg_seconds=float(seg), temperature=float(temp),
        walk_strength=0.0, use_real=bool(use_real), sr=22050,
        crossfade_seconds=0.05, seed=int(seed),
    )
    render_clap_musaicing([Path(seed_path)], prompts, out_path, params)
    return str(out_path)


def run_U(seed_path, bias, chunk, use_real, seed):
    from src.modules.mvp_u.cap_latent import CapLatentParams, render_cap_latent
    out_path = GUI_OUT / f"U_{time.strftime('%H%M%S')}.wav"
    params = CapLatentParams(
        bias_strength=float(bias), chunk_seconds=float(chunk),
        use_real=bool(use_real), model_path=mn.RAVE_GUITAR,
        device=mn.DEVICE, seed=int(seed), dry_wet=1.0,
    )
    render_cap_latent(Path(seed_path), out_path, params)
    return str(out_path)


def run_V(seed_path, walk_mode, walk_strength, chunk, use_real, seed):
    from src.modules.mvp_v.cap_tokens import CapTokenParams, render_cap_tokens
    out_path = GUI_OUT / f"V_{time.strftime('%H%M%S')}.wav"
    params = CapTokenParams(
        codec="encodec_24khz", bandwidth=6.0, device=mn.DEVICE,
        chunk_seconds=float(chunk), walk_mode=str(walk_mode),
        base_seed_offset=int(seed), walk_strength=int(walk_strength),
        dry_wet=1.0, use_real=bool(use_real),
    )
    render_cap_tokens(Path(seed_path), out_path, params)
    return str(out_path)


def run_W(seed_path, n_iters, dur, keep_mix, use_real):
    from src.modules.mvp_w.cap_loop import LoopParams, render_cap_loop
    ts = time.strftime("%H%M%S")
    out_dir = GUI_OUT / f"W_{ts}"
    params = LoopParams(
        n_iters=int(n_iters), duration_per_step_s=float(dur),
        keep_mix=float(keep_mix), sr=22050,
        use_real=bool(use_real), save_intermediates=False,
    )
    render_cap_loop(Path(seed_path), out_dir, params)
    return str(out_dir / "final.wav")


def run_multinet(net_name, seed_path):
    _ensure_models()
    audio, sr = sf.read(seed_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if net_name == "net1":
        master = mn.net1(audio, sr, _state["rave"], _state["morph_g"], _state["codec"])
    elif net_name == "net2":
        master = mn.net2(audio, sr, _state["rave"], _state["morph_g"], _state["codec"], passes=3)
    elif net_name == "net3":
        master = mn.net3(audio, sr, _state["rave"], _state["morph_g"], _state["codec"])
    elif net_name == "net_max":
        master = mn.net_max(audio, sr, _state["rave"], _state["morph_g"], _state["codec"],
                            [mn.RAVE_GUITAR, mn.RAVE_ORGAN])
    elif net_name == "net_dynamic":
        master = mn.net_dynamic(audio, sr, _state["rave"], _state["morph_g"], _state["codec"],
                                 [mn.RAVE_GUITAR, mn.RAVE_ORGAN])
    else:
        raise ValueError(f"unknown net: {net_name}")
    return _save_out(master, sr, f"multinet_{net_name}")


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------

SEED_DEFAULTS = {
    "sine 10s @ 48k": str(ROOT / "runs/multinet/sine_440_10s.wav"),
    "sine 30s @ 48k": str(ROOT / "runs/multinet/sine_max_30s.wav"),
    "sine 60s @ 48k": str(ROOT / "runs/multinet/sine_dyn_60s.wav"),
    "sub-bass 180s": str(ROOT / "runs/masterpiece/meta_symphony/seed_sub_180s.wav"),
}


def build_ui():
    css = """
    body { background: #0d1117; color: #c9d1d9; }
    .gradio-container { max-width: 1200px !important; }
    h1, h2, h3 { color: #7fffd4; }
    """

    with gr.Blocks(title="AudioArt Control") as demo:
        gr.Markdown("# 🎛️ AudioArt Control Panel\n"
                    "9 MVP × 5 Multinet × Meta-Symphony — direct render GUI. "
                    "Renders save to `runs/gui/`.")

        with gr.Tabs():
            # ------ Listen tab ------
            with gr.Tab("🎧 Listen"):
                with gr.Row():
                    refresh = gr.Button("Refresh list", scale=0)
                    file_drop = gr.Dropdown(choices=list_wavs(), label="WAV file",
                                            scale=4)
                player = gr.Audio(label="Player", type="filepath", interactive=False)
                file_drop.change(play_pick, file_drop, player)
                refresh.click(lambda: gr.update(choices=list_wavs()), None, file_drop)

            # ------ MVP-A ------
            with gr.Tab("A · Latent Perturb"):
                with gr.Row():
                    a_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                          value=SEED_DEFAULTS["sine 30s @ 48k"],
                                          label="Seed wav")
                a_noise = gr.Slider(0, 2.0, 0.1, label="noise_scale")
                a_drop = gr.Slider(0, 0.7, 0.0, label="dim_dropout")
                a_shuffle = gr.Checkbox(False, label="dim_shuffle")
                a_mode = gr.Radio(["white", "smoothed"], value="smoothed",
                                   label="noise_mode")
                a_dw = gr.Slider(0, 1.0, 0.7, label="dry_wet")
                a_rng = gr.Number(0, label="rng_seed", precision=0)
                a_btn = gr.Button("Render A", variant="primary")
                a_out = gr.Audio(label="Output")
                a_btn.click(run_A,
                            inputs=[a_seed, a_noise, a_drop, a_shuffle, a_mode, a_dw, a_rng],
                            outputs=a_out)

            # ------ MVP-C ------
            with gr.Tab("C · Token Bend"):
                with gr.Row():
                    c_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                          value=SEED_DEFAULTS["sine 30s @ 48k"],
                                          label="Seed wav")
                c_mode = gr.Dropdown(["bit_flip", "quantizer_drop", "shuffle",
                                       "invalid_token"], value="bit_flip", label="bend mode")
                c_rate = gr.Slider(0, 0.3, 0.05, label="rate")
                c_q_lo = gr.Number(0, label="quantizer_range lo (0=auto/all)", precision=0)
                c_q_hi = gr.Number(0, label="quantizer_range hi (0=auto/all; use neg for upper)",
                                    precision=0)
                c_shuffle_win = gr.Slider(1, 48, 12, label="shuffle_window", step=1)
                c_dw = gr.Slider(0, 1.0, 0.7, label="dry_wet")
                c_rng = gr.Number(0, label="rng_seed", precision=0)
                c_btn = gr.Button("Render C", variant="primary")
                c_out = gr.Audio(label="Output")
                c_btn.click(run_C,
                            inputs=[c_seed, c_mode, c_rate, c_q_lo, c_q_hi, c_shuffle_win,
                                    c_dw, c_rng],
                            outputs=c_out)

            # ------ MVP-D ------
            with gr.Tab("D · Morph"):
                with gr.Row():
                    d_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                          value=SEED_DEFAULTS["sine 30s @ 48k"],
                                          label="Seed wav")
                d_t = gr.Slider(0, 1.0, 0.005, label="morph t (cliff at ~0.05~0.95)",
                                 step=0.001)
                d_mode = gr.Radio(["linear", "slerp", "random_walk"],
                                   value="linear", label="morph mode")
                d_pn = gr.Slider(0, 1.0, 0.0, label="perturb noise_scale (A+D chain)")
                d_pd = gr.Slider(0, 0.7, 0.0, label="perturb dim_dropout")
                d_dw = gr.Slider(0, 1.0, 0.7, label="dry_wet")
                d_rng = gr.Number(0, label="rng_seed", precision=0)
                d_btn = gr.Button("Render D", variant="primary")
                d_out = gr.Audio(label="Output")
                d_btn.click(run_D,
                            inputs=[d_seed, d_t, d_mode, d_pn, d_pd, d_dw, d_rng],
                            outputs=d_out)

            # ------ MVP-E ------
            with gr.Tab("E · Granular"):
                e_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                      value=SEED_DEFAULTS["sine 30s @ 48k"],
                                      label="Seed wav")
                e_grain = gr.Slider(4, 64, 16, step=1, label="grain_size (latent frames)")
                e_mem = gr.Slider(256, 8192, 2048, step=128, label="memory_size")
                e_num = gr.Slider(1, 8, 4, step=1, label="num_grains")
                e_mix = gr.Slider(0, 1.0, 0.5, label="mix")
                e_dw = gr.Slider(0, 1.0, 0.6, label="dry_wet")
                e_rng = gr.Number(0, label="rng_seed", precision=0)
                e_btn = gr.Button("Render E", variant="primary")
                e_out = gr.Audio(label="Output")
                e_btn.click(run_E,
                            inputs=[e_seed, e_grain, e_mem, e_num, e_mix, e_dw, e_rng],
                            outputs=e_out)

            # ------ MVP-F ------
            with gr.Tab("F · Freeze"):
                f_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                      value=SEED_DEFAULTS["sine 30s @ 48k"],
                                      label="Seed wav")
                f_au = gr.Slider(0, 1.0, 0.5, label="auto_upper_fraction")
                f_ui = gr.Slider(8, 512, 128, step=8, label="update_interval_frames")
                f_xf = gr.Slider(0, 64, 16, step=2, label="crossfade_frames")
                f_dw = gr.Slider(0, 1.0, 0.6, label="dry_wet")
                f_rng = gr.Number(0, label="rng_seed", precision=0)
                f_btn = gr.Button("Render F", variant="primary")
                f_out = gr.Audio(label="Output")
                f_btn.click(run_F,
                            inputs=[f_seed, f_au, f_ui, f_xf, f_dw, f_rng],
                            outputs=f_out)

            # ------ MVP-G ------
            with gr.Tab("G · Feedback"):
                g_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                      value=SEED_DEFAULTS["sine 30s @ 48k"],
                                      label="Seed wav")
                g_delay = gr.Slider(4, 256, 32, step=4, label="delay_frames")
                g_fb = gr.Slider(0, 0.95, 0.4, label="feedback")
                g_mix = gr.Slider(0, 1.0, 0.5, label="mix")
                g_dw = gr.Slider(0, 1.0, 0.6, label="dry_wet")
                g_rng = gr.Number(0, label="rng_seed", precision=0)
                g_btn = gr.Button("Render G", variant="primary")
                g_out = gr.Audio(label="Output")
                g_btn.click(run_G,
                            inputs=[g_seed, g_delay, g_fb, g_mix, g_dw, g_rng],
                            outputs=g_out)

            # ------ MVP-H ------
            with gr.Tab("H · Codebook Organ (gen)"):
                h_mode = gr.Radio(["prime", "fibonacci", "random_walk"], value="prime",
                                   label="generator mode")
                h_base = gr.Number(0, label="base_token", precision=0)
                h_stride = gr.Slider(1, 32, 7, step=1, label="stride (per quantizer)")
                h_frames = gr.Slider(75, 3000, 750, step=75,
                                      label="duration_frames (75 fps @ 24k)")
                h_rng = gr.Number(0, label="rng_seed", precision=0)
                h_btn = gr.Button("Render H", variant="primary")
                h_out = gr.Audio(label="Output")
                h_btn.click(run_H,
                            inputs=[h_mode, h_base, h_stride, h_frames, h_rng],
                            outputs=h_out)

            # ------ MVP-I ------
            with gr.Tab("I · Bass Massive"):
                i_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                      value=SEED_DEFAULTS["sine 30s @ 48k"],
                                      label="Seed wav")
                i_smear = gr.Slider(0, 64, 12, step=1, label="smear_delay")
                i_jitter = gr.Slider(0, 0.3, 0.05, label="jitter_rate")
                i_fold = gr.Slider(0, 0.3, 0.0, label="fold_leak_rate")
                with gr.Row():
                    i_smear_lo = gr.Number(0, label="smear_q lo", precision=0)
                    i_smear_hi = gr.Number(2, label="smear_q hi", precision=0)
                with gr.Row():
                    i_jitter_lo = gr.Number(0, label="jitter_q lo", precision=0)
                    i_jitter_hi = gr.Number(3, label="jitter_q hi", precision=0)
                i_dw = gr.Slider(0, 1.0, 0.6, label="dry_wet")
                i_rng = gr.Number(0, label="rng_seed", precision=0)
                i_btn = gr.Button("Render I", variant="primary")
                i_out = gr.Audio(label="Output")
                i_btn.click(run_I,
                            inputs=[i_seed, i_smear, i_jitter, i_fold,
                                    i_smear_lo, i_smear_hi, i_jitter_lo, i_jitter_hi,
                                    i_dw, i_rng],
                            outputs=i_out)

            # ============== V2 (text-conditioned) ==============
            with gr.Tab("Y · Phantom Weight (V2 ∅→weight)"):
                gr.Markdown("Input-less weight-space generator. 2 ckpts morph + random latent.")
                y_t = gr.Slider(0.0, 1.0, value=0.5, label="morph_t")
                y_mode = gr.Radio(["smoothed", "white", "ou", "sinusoid"],
                                  value="smoothed", label="latent_mode")
                y_sig = gr.Slider(0.1, 3.0, value=1.0, label="latent_sigma")
                y_dur = gr.Slider(2.0, 30.0, value=10.0, label="duration_s")
                y_seed = gr.Number(value=0, label="seed", precision=0)
                y_btn = gr.Button("Run Y", variant="primary")
                y_out = gr.Audio(label="output")
                y_btn.click(run_Y,
                            inputs=[y_t, y_mode, y_sig, y_dur, y_seed],
                            outputs=y_out)

            with gr.Tab("S · Random Prompt TTA (V2 ∅→text)"):
                gr.Markdown("Random prompt → TTA. `|`-separated for custom prompts.")
                s_text = gr.Textbox(value="", lines=2,
                                    label="prompts (pipe-separated, empty = bank random)")
                s_n = gr.Slider(1, 6, value=3, step=1, label="n_prompts (if random)")
                s_mix = gr.Radio(["concat", "blend"], value="concat", label="mix_mode")
                s_dur = gr.Slider(4.0, 40.0, value=12.0, label="duration_s")
                s_real = gr.Checkbox(value=False, label="use_real (AudioLDM, slow)")
                s_seed = gr.Number(value=0, label="seed", precision=0)
                s_btn = gr.Button("Run S", variant="primary")
                s_out = gr.Audio(label="output")
                s_btn.click(run_S,
                            inputs=[s_text, s_n, s_mix, s_dur, s_real, s_seed],
                            outputs=s_out)

            with gr.Tab("T · CLAP Musaicing (V2 corpus×text)"):
                gr.Markdown("Text prompts → audio corpus grain match.")
                t_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                     value=SEED_DEFAULTS["sine 30s @ 48k"],
                                     label="corpus wav")
                t_text = gr.Textbox(
                    value="bright shimmer with metallic resonance | deep cavernous drone",
                    lines=2, label="prompts (pipe-separated)")
                t_g = gr.Slider(0.5, 4.0, value=2.0, label="grain_seconds")
                t_seg = gr.Slider(2.0, 12.0, value=4.0, label="seg_seconds_per_prompt")
                t_temp = gr.Slider(0.001, 2.0, value=0.2, label="temperature")
                t_real = gr.Checkbox(value=False, label="use_real (CLAP)")
                t_sd = gr.Number(value=0, label="seed", precision=0)
                t_btn = gr.Button("Run T", variant="primary")
                t_out = gr.Audio(label="output")
                t_btn.click(run_T,
                            inputs=[t_seed, t_text, t_g, t_seg, t_temp, t_real, t_sd],
                            outputs=t_out)

            with gr.Tab("U · Cap-Steered Latent (V2 a→t→a · latent)"):
                gr.Markdown("audio → caption → text embed → RAVE latent bias.")
                u_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                     value=SEED_DEFAULTS["sine 10s @ 48k"],
                                     label="target wav")
                u_bias = gr.Slider(0.0, 2.0, value=0.4, label="bias_strength")
                u_chunk = gr.Slider(0.2, 4.0, value=1.0, label="chunk_seconds (re-caption)")
                u_real = gr.Checkbox(value=False, label="use_real (Qwen-Audio not wired; falls back)")
                u_sd = gr.Number(value=0, label="seed", precision=0)
                u_btn = gr.Button("Run U", variant="primary")
                u_out = gr.Audio(label="output")
                u_btn.click(run_U,
                            inputs=[u_seed, u_bias, u_chunk, u_real, u_sd],
                            outputs=u_out)

            with gr.Tab("V · Cap-Conditioned Tokens (V2 a→t→a · tokens)"):
                gr.Markdown("audio → caption hash → EnCodec codebook walk.")
                v_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                     value=SEED_DEFAULTS["sine 10s @ 48k"],
                                     label="target wav")
                v_mode = gr.Radio(["prime", "fibonacci", "random_walk"],
                                  value="prime", label="walk_mode")
                v_str = gr.Slider(1, 20, value=5, step=1, label="walk_strength")
                v_chunk = gr.Slider(0.2, 4.0, value=1.0, label="chunk_seconds")
                v_real = gr.Checkbox(value=False, label="use_real (CLAP captioner)")
                v_sd = gr.Number(value=0, label="seed", precision=0)
                v_btn = gr.Button("Run V", variant="primary")
                v_out = gr.Audio(label="output")
                v_btn.click(run_V,
                            inputs=[v_seed, v_mode, v_str, v_chunk, v_real, v_sd],
                            outputs=v_out)

            with gr.Tab("W · Cap→TTA Loop (V2 a→t→a · STFT)"):
                gr.Markdown("audio → caption → TTA → audio → ... recursive drift.")
                w_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                     value=SEED_DEFAULTS["sine 10s @ 48k"],
                                     label="seed wav")
                w_n = gr.Slider(1, 8, value=4, step=1, label="n_iters")
                w_dur = gr.Slider(2.0, 12.0, value=6.0, label="duration_per_step_s")
                w_keep = gr.Slider(0.0, 0.5, value=0.0, label="keep_mix")
                w_real = gr.Checkbox(value=False, label="use_real (AudioLDM)")
                w_btn = gr.Button("Run W (final iter)", variant="primary")
                w_out = gr.Audio(label="final output")
                w_btn.click(run_W,
                            inputs=[w_seed, w_n, w_dur, w_keep, w_real],
                            outputs=w_out)

            # ------ Net Hyper-Max ------
            with gr.Tab("🔥 Net Hyper-Max (24 MVP × 180s)"):
                gr.Markdown(
                    "**Hyper-Max master = 4 movements × 6 MVPs.** "
                    "Pre-rendered in `runs/hyper_max/` (procedural) and "
                    "`runs/hyper_max_real/` (CLAP+AudioLDM). 새 렌더는 wall ~75 s "
                    "(procedural) / 수 분 (real backend). 자세한 구조는 "
                    "`docs/HYPER_MAX_ARCHITECTURE.md`."
                )

                hm_variant = gr.Radio(
                    ["procedural (runs/hyper_max)", "real (runs/hyper_max_real)"],
                    value="procedural (runs/hyper_max)",
                    label="variant",
                )
                hm_track = gr.Radio(
                    ["master_180s.wav",
                     "movement_1_genesis.wav",
                     "movement_2_latent.wav",
                     "movement_3_tokens.wav",
                     "movement_4_final.wav"],
                    value="master_180s.wav",
                    label="track",
                )
                hm_load_btn = gr.Button("Load pre-rendered", variant="primary")
                hm_load_out = gr.Audio(label="output")

                def _hm_load(variant, track):
                    sub = "hyper_max_real" if "real" in variant else "hyper_max"
                    p = ROOT / "runs" / sub / track
                    return str(p) if p.exists() else None

                hm_load_btn.click(_hm_load,
                                  inputs=[hm_variant, hm_track],
                                  outputs=hm_load_out)

                gr.Markdown("---")
                hm_render_real = gr.Checkbox(value=False,
                                             label="render with real backend (CLAP+AudioLDM, 수 분 소요)")
                hm_render_btn = gr.Button("Render fresh master", variant="secondary")
                hm_render_out = gr.Audio(label="freshly rendered master")

                def _hm_render(real_flag):
                    import subprocess
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(ROOT)
                    env["HYPER_MAX_REAL"] = "1" if real_flag else "0"
                    log.info("rendering Hyper-Max (real=%s)...", real_flag)
                    r = subprocess.run(
                        ["python", "scripts/net_hyper_max.py"],
                        cwd=str(ROOT), env=env, capture_output=True, text=True,
                    )
                    log.info("hyper_max stderr tail: %s", r.stderr[-500:])
                    sub = "hyper_max_real" if real_flag else "hyper_max"
                    return str(ROOT / "runs" / sub / "master_180s.wav")

                hm_render_btn.click(_hm_render,
                                    inputs=[hm_render_real],
                                    outputs=hm_render_out)

            # ------ Multinet ------
            with gr.Tab("🌐 Multinet"):
                gr.Markdown(
                    "Run a full macro-net on a seed. Net Max + Net Dynamic 가 가장 무겁다 "
                    "(20-30 s wall time)."
                )
                m_net = gr.Radio(["net1", "net2", "net3", "net_max", "net_dynamic"],
                                  value="net1", label="네트 선택")
                m_seed = gr.Dropdown(choices=list(SEED_DEFAULTS.values()),
                                      value=SEED_DEFAULTS["sine 10s @ 48k"],
                                      label="Seed wav")
                m_btn = gr.Button("Run multinet", variant="primary")
                m_out = gr.Audio(label="Master output")
                m_btn.click(run_multinet, inputs=[m_net, m_seed], outputs=m_out)

        gr.Markdown(
            "---\n"
            "models lazy-loaded on first render. seed dropdown lists pre-rendered sine wavs in "
            "`runs/multinet/`. all renders save to `runs/gui/`. "
            "See `docs/MULTINET_ARCHITECTURE.md` for net topology."
        )
    return demo


CSS = """
body { background: #0d1117; color: #c9d1d9; }
.gradio-container { max-width: 1200px !important; }
h1, h2, h3 { color: #7fffd4; }
"""


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False,
              inbrowser=False, show_error=True, css=CSS)
