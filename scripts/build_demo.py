"""Build a static HTML demo page indexing all key renders.

Output: /home1/irteam/proj/AudioArt/demo.html (relative wav paths under runs/...)
Thumbnails: runs/_thumbs/<safe_id>.png (320×60 waveform PNG)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path("/home1/irteam/proj/AudioArt")
THUMB_DIR = ROOT / "runs" / "_thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Track:
    title: str
    rel_path: str           # path relative to project root
    description: str
    section: str            # group key
    order: int = 0

    @property
    def safe_id(self) -> str:
        return self.rel_path.replace("/", "_").replace(".", "_")


def stats(path: Path) -> dict:
    audio, sr = sf.read(str(path))
    audio = audio if audio.ndim > 1 else audio[:, None]
    n = audio.shape[0]
    dur = n / sr
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    peak = float(np.max(np.abs(audio)))
    channels = audio.shape[1]
    return {"sr": sr, "duration_s": dur, "rms": rms, "peak": peak,
            "channels": channels, "size_mb": path.stat().st_size / 1024 / 1024}


def build_waveform_png(audio_path: Path, png_path: Path, *,
                       w: int = 480, h: int = 80) -> None:
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    # Down-sample for plotting
    target = w * 2
    if n > target:
        bins = np.array_split(audio[: (n // target) * target], target)
        mins = np.array([b.min() for b in bins])
        maxs = np.array([b.max() for b in bins])
    else:
        mins = audio
        maxs = audio
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    x = np.arange(len(mins))
    ax.fill_between(x, mins, maxs, color="#7fffd4", linewidth=0)
    ax.set_xlim(0, len(mins))
    peak = max(0.05, float(np.max(np.abs(audio))))
    ax.set_ylim(-peak * 1.05, peak * 1.05)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(str(png_path), dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def section_block(name: str, blurb: str, tracks: list[Track]) -> str:
    rows = []
    for t in tracks:
        path = ROOT / t.rel_path
        if not path.exists():
            rows.append(f'<div class="track missing"><div class="title">{t.title}</div>'
                        f'<div class="meta">missing: {t.rel_path}</div></div>')
            continue
        try:
            st = stats(path)
        except Exception as exc:  # noqa: BLE001
            rows.append(f'<div class="track missing"><div class="title">{t.title}</div>'
                        f'<div class="meta">stats fail: {exc}</div></div>')
            continue
        png_rel = f"runs/_thumbs/{t.safe_id}.png"
        png_abs = ROOT / png_rel
        try:
            if not png_abs.exists() or png_abs.stat().st_mtime < path.stat().st_mtime:
                build_waveform_png(path, png_abs)
        except Exception as exc:  # noqa: BLE001
            print(f"thumb fail {t.rel_path}: {exc}")
        meta = (f"{st['duration_s']:.1f}s · {st['sr']/1000:.0f} kHz · "
                f"{'stereo' if st['channels'] > 1 else 'mono'} · "
                f"rms {st['rms']:.3f} · peak {st['peak']:.3f} · "
                f"{st['size_mb']:.1f} MB")
        rows.append(f'''
<div class="track">
  <div class="title">{t.title}</div>
  <div class="desc">{t.description}</div>
  <img class="wave" src="{png_rel}" alt="waveform">
  <audio controls preload="none" src="{t.rel_path}"></audio>
  <div class="meta">{meta} · <a href="{t.rel_path}">download</a></div>
</div>''')
    return f'''
<section class="section">
  <h2>{name}</h2>
  <p class="blurb">{blurb}</p>
  <div class="tracks">{"".join(rows)}</div>
</section>
'''


# ---------------------------------------------------------------------------
# Track manifest
# ---------------------------------------------------------------------------

SEEDS = [
    Track("Sine 440 Hz · 10 s", "runs/multinet/sine_440_10s.wav",
          "Clean 440 Hz seed used by Net 1 / 2 / 3.", "seeds", 0),
    Track("Sine enriched · 30 s", "runs/multinet/sine_max_30s.wav",
          "440 Hz fundamental + 660 Hz 5th + 0.13 Hz tremolo + slow drift. "
          "Seed for Net Max.", "seeds", 1),
    Track("Sine dynamic arc · 60 s", "runs/multinet/sine_dyn_60s.wav",
          "Pitch sweep 220→660 Hz + amplitude arc + sparse onsets + 30 s gaussian dip. "
          "Seed for Net Dynamic.", "seeds", 2),
    Track("Sub-bass · 180 s", "runs/masterpiece/meta_symphony/seed_sub_180s.wav",
          "32.7 + 41.2 + 55 Hz triple-sub with FM + 30 s breathing LFO. "
          "Seed for Meta-Symphony.", "seeds", 3),
]

NET1 = [
    Track("MASTER — Crystal Cathedral", "runs/multinet/net1/MASTER.wav",
          "5-bus parallel mix with 80 Hz anchor. 440 Hz still recognisable; "
          "drone bed at 49/76/82 Hz fills sub-bass.", "net1", 0),
    Track("Bus L — Bass Massive (I)", "runs/multinet/net1/bus_L_bass.wav",
          "EnCodec lower-RVQ temporal smear, 80 Hz sub-anchored.", "net1", 1),
    Track("Bus M — Core (A→D→C)", "runs/multinet/net1/bus_M_core.wav",
          "RAVE latent noise + checkpoint morph + token bit-flip.", "net1", 2),
    Track("Bus H — Shimmer (F→A)", "runs/multinet/net1/bus_H_shimmer.wav",
          "Upper-50% latent freeze + high-noise pass.", "net1", 3),
    Track("Bus T — Recursive (G→E)", "runs/multinet/net1/bus_T_recursive.wav",
          "Latent feedback echo + granular memory projection.", "net1", 4),
    Track("Bus D — Drone (H prime)", "runs/multinet/net1/bus_D_drone.wav",
          "Generative codebook organ on prime-index tokens.", "net1", 5),
]

NET2 = [
    Track("MASTER — Recursive Organ", "runs/multinet/net2/MASTER.wav",
          "After 3 macro-feedback passes through G→A→C→F. "
          "Pitch drifts from 440 to ≈600 Hz.", "net2", 0),
    Track("Pass 0", "runs/multinet/net2/pass_0.wav",
          "First pass on raw sine. Holds 440 Hz core.", "net2", 1),
    Track("Pass 1", "runs/multinet/net2/pass_1.wav",
          "Pass-0 re-fed. Pitch slides to ~400 Hz.", "net2", 2),
    Track("Pass 2", "runs/multinet/net2/pass_2.wav",
          "Pass-1 re-fed. Pitch jumps to ~600 Hz — model loses anchor.", "net2", 3),
]

NET3 = [
    Track("MASTER — Decoding Chamber", "runs/multinet/net3/MASTER.wav",
          "Linear 9-stage A→D→E→F→G→C→I + dry 80 Hz crossover anchor. "
          "Sub-bass single-handedly restores RMS after 7-stage decay.", "net3", 0),
    Track("Stage 1 — A noise=0.03", "runs/multinet/net3/s1_A.wav",
          "Light smoothed latent noise.", "net3", 1),
    Track("Stage 2 — D morph t=0.005", "runs/multinet/net3/s2_D.wav",
          "Endpoint-cliff morph (just past pure-guitar).", "net3", 2),
    Track("Stage 3 — E granular", "runs/multinet/net3/s3_E.wav",
          "Memory buffer 1024 frames, 4 grains.", "net3", 3),
    Track("Stage 4 — F freeze 25%", "runs/multinet/net3/s4_F.wav",
          "Upper-quarter latent dim shimmer.", "net3", 4),
    Track("Stage 5 — G feedback d=24", "runs/multinet/net3/s5_G.wav",
          "Latent echo, light feedback.", "net3", 5),
    Track("Stage 6 — C bit-flip upper", "runs/multinet/net3/s6_C.wav",
          "Upper-3-quantizer bit-flips.", "net3", 6),
    Track("Stage 7 — I bass smear", "runs/multinet/net3/s7_I.wav",
          "Lower-q temporal smear pre-anchor.", "net3", 7),
]

NET_MAX_PASS1 = [
    Track("MASTER pass 1", "runs/multinet/net_max/MASTER_pass1.wav",
          "8-bus parallel mix, pass 1 weights.", "max_p1", 0),
    Track("α Foundation", "runs/multinet/net_max/p1_bus_α_foundation.wav",
          "I + C(invalid_token) + 80 Hz anchor +10 dB.", "max_p1", 1),
    Track("β Core", "runs/multinet/net_max/p1_bus_β_core.wav",
          "6-stage chain A→D(guitar)→E→G→A(drop)→F.", "max_p1", 2),
    Track("γ Ghost", "runs/multinet/net_max/p1_bus_γ_ghost.wav",
          "Nested F→G→C(shuffle)→F freeze chain.", "max_p1", 3),
    Track("δ Twin", "runs/multinet/net_max/p1_bus_δ_twin.wav",
          "Organ-side morph + heavy perturb + token bend.", "max_p1", 4),
    Track("ε Glitch", "runs/multinet/net_max/p1_bus_ε_glitch.wav",
          "Aggressive C(rate=0.10) + I(fold) + G.", "max_p1", 5),
    Track("ζ Drone", "runs/multinet/net_max/p1_bus_ζ_drone.wav",
          "H(prime) + H(fibonacci) → A smoothing.", "max_p1", 6),
    Track("η Loop-B", "runs/multinet/net_max/p1_bus_η_loopB.wav",
          "Caption→TTA depth 3, stub backend.", "max_p1", 7),
    Track("θ XFB", "runs/multinet/net_max/p1_bus_θ_xfb.wav",
          "Cross-bus feedback: β output → G(deep) → I(fold).", "max_p1", 8),
]

NET_MAX_PASS2_FINAL = [
    Track("MASTER pass 2", "runs/multinet/net_max/MASTER_pass2.wav",
          "Same 8 buses, refed-seed input, β/θ weights bumped.", "max_p2", 0),
    Track("Refed seed (0.55·M1 + 0.45·sine)", "runs/multinet/net_max/p2_refed_seed.wav",
          "Input to pass 2.", "max_p2", 1),
    Track("β Core (pass 2)", "runs/multinet/net_max/p2_bus_β_core.wav",
          "Deepened chain on refed seed.", "max_p2", 2),
    Track("θ XFB (pass 2)", "runs/multinet/net_max/p2_bus_θ_xfb.wav",
          "Reinforced cross-bus feedback.", "max_p2", 3),
    Track("MASTER FINAL — Cathedral Hive", "runs/multinet/net_max/MASTER_FINAL.wav",
          "Time-varying S-curve blend of pass 1 → pass 2 across the 30 s. "
          "Six independent pitch zones present simultaneously.", "max_p2", 4),
]

NET_DYN = [
    Track("MASTER — Tempest", "runs/multinet/net_dynamic/MASTER_FINAL.wav",
          "60 s composition with per-bus envelopes + filter sweep + 3 events: "
          "15 s freeze click · 30 s silence drop · 45 s drone burst.", "dyn", 0),
    Track("Bus α — Foundation", "runs/multinet/net_dynamic/bus_α_foundation.wav",
          "Bass anchor (high envelope across full duration).", "dyn", 1),
    Track("Bus β — Core", "runs/multinet/net_dynamic/bus_β_core.wav",
          "Peaks 22 s, fades by 30 s drop.", "dyn", 2),
    Track("Bus γ — Ghost", "runs/multinet/net_dynamic/bus_γ_ghost.wav",
          "Enters 10 s, peaks 15 s (click), re-emerges 50 s.", "dyn", 3),
    Track("Bus δ — Twin", "runs/multinet/net_dynamic/bus_δ_twin.wav",
          "Enters 20 s, peaks 28 s, gone by 35 s.", "dyn", 4),
    Track("Bus ε — Glitch", "runs/multinet/net_dynamic/bus_ε_glitch.wav",
          "Spike 35→45 s.", "dyn", 5),
    Track("Bus ζ — Drone", "runs/multinet/net_dynamic/bus_ζ_drone.wav",
          "Sustains 30→60 s.", "dyn", 6),
    Track("Bus η — Loop-B", "runs/multinet/net_dynamic/bus_η_loopB.wav",
          "Caption loop, 45→55 s climax.", "dyn", 7),
    Track("Bus θ — XFB", "runs/multinet/net_dynamic/bus_θ_xfb.wav",
          "Cross-feedback, ends the piece at peak 0.45 weight.", "dyn", 8),
]

META = [
    Track("META_SYMPHONY_FINAL (stereo, 180 s)",
          "runs/masterpiece/meta_symphony/META_SYMPHONY_FINAL.wav",
          "Network of networks. Four macro-net stems (N1, N3, N2, N_dyn) "
          "interwoven via LFO crossfade (60 s & 45 s) + stereo drift (20 s & 25 s) + "
          "100 Hz sub re-injection. First stereo render in the AudioArt stack.",
          "meta", 0),
]

ARCHIVE = [
    Track("Final Symphony Bass Heavy",
          "runs/masterpiece/final_symphony_bass_heavy.wav",
          "180 s bass-heavy master. 80 Hz crossover +8 dB sub-boost.", "archive", 0),
    Track("Neural Granular Symphony",
          "runs/masterpiece/neural_granular_symphony.wav",
          "8 tracks fragmented into 250 ms grains, stochastic interweaving.",
          "archive", 1),
    Track("Full Spectrum Neural Galaxy",
          "runs/masterpiece/full_spectrum_neural_galaxy.wav",
          "9-octave hyper-chord, triangular energy balancing.", "archive", 2),
    Track("Hi-Fi Enhanced Symphony",
          "runs/masterpiece/hifi_enhanced_symphony_1.wav",
          "Harmonic exciter ≥8 kHz + 14 kHz air boost.", "archive", 3),
    Track("Defined Neural Galaxy",
          "runs/masterpiece/defined_neural_galaxy.wav",
          "Tukey-windowed granular reconstruction.", "archive", 4),
    Track("Neural Symphony 1 (original)",
          "runs/masterpiece/neural_symphony_1.wav",
          "First multi-layer assembly, harmonic alignment.", "archive", 5),
]


def main() -> None:
    sections_html = []
    sections_html.append(section_block(
        "Seeds",
        "Synthetic inputs fed to the macro-nets. The whole pipeline starts here.",
        sorted(SEEDS, key=lambda t: t.order)))
    sections_html.append(section_block(
        "Net 1 — Crystal Cathedral",
        "5 buses in parallel (L bass · M core · H shimmer · T recursive · D drone) "
        "summed through an 80 Hz anchor and a soft limiter. 10 s sine seed.",
        sorted(NET1, key=lambda t: t.order)))
    sections_html.append(section_block(
        "Net 2 — Recursive Organ",
        "G→A→C→F chain wrapped in a 3-pass macro-feedback. Each pass eats its own "
        "tail; listen for the pitch drift across passes.",
        sorted(NET2, key=lambda t: t.order)))
    sections_html.append(section_block(
        "Net 3 — Decoding Chamber",
        "Linear 9-stage pipeline + a dry 80 Hz bypass that brings the sub back. "
        "Per-stage outputs show the sound dissolving.",
        sorted(NET3, key=lambda t: t.order)))
    sections_html.append(section_block(
        "Net Max — Cathedral Hive · Pass 1",
        "8 buses (α–θ) feeding one master mix. Bus θ taps β to create cross-bus "
        "feedback.",
        sorted(NET_MAX_PASS1, key=lambda t: t.order)))
    sections_html.append(section_block(
        "Net Max — Cathedral Hive · Pass 2 & Final",
        "Pass 1's master + the seed are RMS-matched and refed. β/θ weights are "
        "bumped. Final = S-curve crossfade between pass 1 and pass 2.",
        sorted(NET_MAX_PASS2_FINAL, key=lambda t: t.order)))
    sections_html.append(section_block(
        "Net Dynamic — Tempest",
        "Same 8 buses as Net Max but post-render time-varies each bus's amplitude, "
        "stamps three impulse events, and sweeps a master lowpass cutoff. 22 dB "
        "dynamic range across the 60 s.",
        sorted(NET_DYN, key=lambda t: t.order)))
    sections_html.append(section_block(
        "Meta-Symphony",
        "Network of networks. All four macro-nets generate stems on a 3-min sub-bass "
        "seed; LFOs braid them in stereo with a sub-re-injection anchor.",
        META))
    sections_html.append(section_block(
        "Earlier Masterpieces (archive)",
        "Pre-multinet experiments from earlier sessions. 180 s 8-track layered "
        "masters with harmonic exciter / sub-boost / LUFS mastering.",
        ARCHIVE))

    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AudioArt — Demo Page</title>
<style>
  :root {{
    --bg: #0d1117;
    --fg: #c9d1d9;
    --muted: #8b949e;
    --accent: #7fffd4;
    --card: #161b22;
    --border: #30363d;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--fg);
    font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI",
          Roboto, "Helvetica Neue", Arial, "Noto Sans KR", sans-serif;
  }}
  header {{
    padding: 40px 32px 24px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
  }}
  header h1 {{
    margin: 0 0 8px;
    font-size: 32px;
    letter-spacing: -0.02em;
  }}
  header h1 .accent {{ color: var(--accent); }}
  header p.tagline {{
    color: var(--muted);
    margin: 0;
    max-width: 720px;
  }}
  nav {{
    padding: 16px 32px;
    border-bottom: 1px solid var(--border);
    background: var(--card);
    position: sticky; top: 0; z-index: 10;
    overflow-x: auto;
    white-space: nowrap;
  }}
  nav a {{
    color: var(--accent);
    text-decoration: none;
    padding: 6px 12px;
    margin-right: 4px;
    border: 1px solid var(--border);
    border-radius: 999px;
    font-size: 13px;
  }}
  nav a:hover {{ background: #1f2530; }}
  main {{ padding: 24px 32px 80px; max-width: 1200px; margin: 0 auto; }}
  .section {{ margin-top: 48px; scroll-margin-top: 72px; }}
  .section h2 {{
    margin: 0 0 4px;
    font-size: 22px;
    border-left: 4px solid var(--accent);
    padding-left: 12px;
  }}
  .section .blurb {{ color: var(--muted); margin: 0 0 20px 16px; max-width: 760px; }}
  .tracks {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); }}
  .track {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .track.missing {{ opacity: 0.4; }}
  .track .title {{ font-weight: 600; font-size: 15px; }}
  .track .desc {{ color: var(--muted); font-size: 13px; }}
  .track .wave {{
    display: block;
    width: 100%;
    height: 60px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #0d1117;
    image-rendering: -webkit-optimize-contrast;
  }}
  audio {{ width: 100%; }}
  .meta {{ color: var(--muted); font-size: 12px; font-family: ui-monospace, Menlo, monospace; }}
  .meta a {{ color: var(--accent); text-decoration: none; margin-left: 4px; }}
  footer {{
    padding: 32px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
  }}
  footer a {{ color: var(--accent); text-decoration: none; }}
  code {{ background: #1f2530; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Audio<span class="accent">Art</span> — Demo</h1>
  <p class="tagline">
    Aesthetic misuse of audio foundation models. 9 MVP modules (A–I) wired into
    5 macro-nets and 1 meta-symphony. Every track is rendered from a sine wave
    seed through some chain of latent perturbation, codec bending, checkpoint
    morphing, granular memory, freeze shimmer, feedback echo, generative drone,
    and bass massive operations.
  </p>
</header>
<nav>
  <a href="#seeds">seeds</a>
  <a href="#net1">Net 1</a>
  <a href="#net2">Net 2</a>
  <a href="#net3">Net 3</a>
  <a href="#max_p1">Net Max p1</a>
  <a href="#max_p2">Net Max p2</a>
  <a href="#dyn">Net Dynamic</a>
  <a href="#meta">Meta-Symphony</a>
  <a href="#archive">Archive</a>
</nav>
<main>
{"".join(s.replace('<section class="section">', '<section class="section" id="') for s in [])}{
"".join(
    s.replace('<section class="section">',
              f'<section class="section" id="{sid}">')
    for s, sid in zip(sections_html,
                      ["seeds", "net1", "net2", "net3", "max_p1", "max_p2",
                       "dyn", "meta", "archive"])
)}
</main>
<footer>
  Rendered <code>{__import__('datetime').date.today().isoformat()}</code> from
  <code>scripts/build_demo.py</code>. See
  <a href="docs/MULTINET_ARCHITECTURE.md">MULTINET_ARCHITECTURE.md</a> and
  <a href="docs/META_SYMPHONY_ARCHITECTURE.md">META_SYMPHONY_ARCHITECTURE.md</a>
  for the underlying design. Open this file with <code>file://</code> or serve
  the project root with <code>python -m http.server</code>.
</footer>
</body>
</html>
"""
    out = ROOT / "demo.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)/1024:.1f} KB)")
    print(f"thumbs dir: {THUMB_DIR}")


if __name__ == "__main__":
    main()
