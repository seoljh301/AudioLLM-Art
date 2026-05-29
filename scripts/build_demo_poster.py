"""Render single-PNG poster summarising all key tracks from the demo manifest.

Output: /home1/irteam/proj/AudioArt/runs/demo_poster.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pathlib import Path

# Re-use the manifest from build_demo
from build_demo import (
    SEEDS, NET1, NET2, NET3,
    NET_MAX_PASS1, NET_MAX_PASS2_FINAL, NET_DYN, META, HYPER_MAX, V2, ARCHIVE,
    stats,
)

ROOT = Path("/home1/irteam/proj/AudioArt")
OUT = ROOT / "runs" / "demo_poster.png"

BG = "#0d1117"
CARD = "#161b22"
ACCENT = "#7fffd4"
ACCENT_V2 = "#ffb84d"   # distinct color for V2 section
FG = "#c9d1d9"
MUTED = "#8b949e"
BORDER = "#30363d"

ACCENT_HYPER = "#ff7fc8"   # pink for Hyper-Max

SECTION_COLORS = {
    "Seeds": ACCENT,
    "Net 1 — Crystal Cathedral": ACCENT,
    "Net 2 — Recursive Organ": ACCENT,
    "Net 3 — Decoding Chamber": ACCENT,
    "Net Max p1 (8 buses)": ACCENT,
    "Net Max p2 + Final": ACCENT,
    "Net Dynamic — Tempest": ACCENT,
    "Meta-Symphony (stereo)": ACCENT,
    "Hyper-Max (24 MVP)": ACCENT_HYPER,
    "V2 — Ulaanbaatar Epic": ACCENT_V2,
    "Archive": "#9aa5b1",
}

SECTIONS = [
    ("Seeds", SEEDS),
    ("Net 1 — Crystal Cathedral", NET1),
    ("Net 2 — Recursive Organ", NET2),
    ("Net 3 — Decoding Chamber", NET3),
    ("Net Max p1 (8 buses)", NET_MAX_PASS1),
    ("Net Max p2 + Final", NET_MAX_PASS2_FINAL),
    ("Net Dynamic — Tempest", NET_DYN),
    ("Meta-Symphony (stereo)", META),
    ("Hyper-Max (24 MVP)", HYPER_MAX),
    ("V2 — Ulaanbaatar Epic", V2),
    ("Archive", ARCHIVE),
]


def load_audio_mono(path: Path, max_samples: int = 480000):
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio[:max_samples], sr


def draw_track(ax, track, sectionname):
    path = ROOT / track.rel_path
    title = track.title
    if not path.exists():
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                color=MUTED, fontsize=8, transform=ax.transAxes)
        ax.set_axis_off()
        return
    try:
        audio, sr = load_audio_mono(path)
        st = stats(path)
    except Exception as exc:  # noqa: BLE001
        ax.text(0.5, 0.5, f"err: {exc}", ha="center", va="center",
                color="#ff7878", fontsize=7, transform=ax.transAxes)
        ax.set_axis_off()
        return
    target_bins = 320
    if len(audio) > target_bins:
        bins = np.array_split(audio[: (len(audio) // target_bins) * target_bins],
                              target_bins)
        mins = np.array([b.min() for b in bins])
        maxs = np.array([b.max() for b in bins])
    else:
        mins = audio; maxs = audio
    x = np.arange(len(mins))
    peak = max(0.05, float(np.max(np.abs(audio))))
    ax.fill_between(x, mins, maxs, color=ACCENT, linewidth=0, alpha=0.85)
    ax.set_xlim(0, len(mins))
    ax.set_ylim(-peak * 1.05, peak * 1.05)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    ax.set_xticks([]); ax.set_yticks([])
    # Title (top inside card)
    title_short = title if len(title) <= 38 else title[:36] + "…"
    ax.text(0.02, 0.93, title_short, transform=ax.transAxes,
            color=FG, fontsize=8.5, va="top", weight="bold")
    # Meta line (bottom)
    meta = f"{st['duration_s']:>5.1f}s · {st['sr']//1000}k · " \
           f"rms {st['rms']:.3f} · peak {st['peak']:.2f}"
    ax.text(0.02, 0.08, meta, transform=ax.transAxes,
            color=MUTED, fontsize=6.5, va="top",
            family="monospace")


def main():
    n_tracks = sum(len(t[1]) for t in SECTIONS)
    cols = 4  # more compact
    n_section_headers = len(SECTIONS)
    total_track_rows = sum((len(s[1]) + cols - 1) // cols for s in SECTIONS)
    total_rows = total_track_rows + n_section_headers + 2  # title + footer
    fig_w = 20
    fig_h = max(16, total_rows * 1.05)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=110, facecolor=BG)
    gs = fig.add_gridspec(total_rows, cols, hspace=0.45, wspace=0.10,
                          left=0.020, right=0.980, top=0.988, bottom=0.010)

    # Title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    title_ax.set_facecolor(BG)
    title_ax.text(0.0, 0.55, "AudioArt — Demo Poster",
                  color=FG, fontsize=28, weight="bold",
                  transform=title_ax.transAxes)
    title_ax.text(0.0, 0.18,
                  "9 MVP × 5 Multinet × Meta-Symphony × V2 Epic — "
                  "neural sound art prototyping",
                  color=MUTED, fontsize=11, transform=title_ax.transAxes)

    row = 1
    for sec_idx, (sec_name, tracks) in enumerate(SECTIONS):
        # Section header row spanning all cols
        hdr_ax = fig.add_subplot(gs[row, :])
        hdr_ax.set_axis_off()
        hdr_ax.set_facecolor(BG)
        sec_color = SECTION_COLORS.get(sec_name, ACCENT)
        # Lock data coords so the patches below act in 0..1 axes space
        hdr_ax.set_xlim(0, 1); hdr_ax.set_ylim(0, 1)
        # Card-colored background band
        hdr_ax.add_patch(plt.Rectangle((0.0, 0.10), 1.0, 0.80,
                                       color=CARD, alpha=0.95, zorder=1,
                                       transform=hdr_ax.transAxes))
        # Thick left accent bar (3% width)
        hdr_ax.add_patch(plt.Rectangle((0.0, 0.10), 0.012, 0.80,
                                       color=sec_color, zorder=2,
                                       transform=hdr_ax.transAxes))
        # Section title in accent color (contrast against dark CARD)
        hdr_ax.text(0.025, 0.5, sec_name, color=sec_color,
                    fontsize=18, weight="bold", va="center",
                    transform=hdr_ax.transAxes, zorder=3)
        hdr_ax.text(0.985, 0.5, f"§{sec_idx+1}  ·  {len(tracks)} tracks",
                    color=FG, fontsize=11, ha="right", va="center",
                    transform=hdr_ax.transAxes,
                    family="monospace", zorder=3)
        print(f"section {sec_idx+1} '{sec_name}' header at row={row}, tracks={len(tracks)}")
        row += 1
        # Track grid for this section
        for i, t in enumerate(tracks):
            r = row + i // cols
            c = i % cols
            ax = fig.add_subplot(gs[r, c])
            draw_track(ax, t, sec_name)
        row += (len(tracks) + cols - 1) // cols

    # Footer
    footer_ax = fig.add_subplot(gs[-1, :])
    footer_ax.set_axis_off()
    footer_ax.set_facecolor(BG)
    footer_ax.text(0.0, 0.5,
                   "scripts/build_demo_poster.py · "
                   "see demo.html for interactive audio + "
                   "docs/META_SYMPHONY_ARCHITECTURE.md for design",
                   color=MUTED, fontsize=9,
                   transform=footer_ax.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=110, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{n_tracks} tracks)")


if __name__ == "__main__":
    main()
