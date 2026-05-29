"""Render runs/mvp_taxonomy.png — single-PNG categorical map of the
14 MVPs (A-N) along I/O × domain × aesthetic axes.

Outputs match docs/MVP_TAXONOMY.md.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("/home1/irteam/proj/AudioArt/runs/mvp_taxonomy.png")

BG = "#0d1117"
CARD = "#161b22"
FG = "#c9d1d9"
MUTED = "#8b949e"
BORDER = "#30363d"
GRID = "#21262d"

# Domain palette
COLOR = {
    "weight":   "#ff6b9d",   # pink/magenta
    "latent":   "#7fffd4",   # mint
    "tokens":   "#ffb84d",   # orange
    "stft":     "#9d7fff",   # purple
    "text":     "#ff7878",   # red
}

# (id, label, domain, io, aesthetic)
MVPS = [
    # Generators
    ("H", "Codebook Organ",       "tokens", "gen",     "pure invention"),
    ("O", "Latent Drift",         "latent", "gen",     "Brownian babble"),
    ("P", "Phase Halluc.",        "stft",   "gen",     "Griffin-Lim invention"),
    ("Y", "Phantom Weight",       "weight", "gen",     "weight prior"),
    ("S", "Random Prompt TTA",    "text",   "gen",     "text invention"),
    # Mutators
    ("A", "Latent Perturb",       "latent", "mutator", "noise injection"),
    ("D", "Ckpt Morph",           "weight", "mutator", "mode pathology"),
    ("E", "Latent Granular",      "latent", "mutator", "self-recursion"),
    ("F", "Latent Freeze",        "latent", "mutator", "sustain"),
    ("G", "Latent Feedback",      "latent", "mutator", "self-recursion"),
    ("K", "Temporal Warp",        "latent", "mutator", "time surgery"),
    ("C", "Token Bend",           "tokens", "mutator", "noise injection"),
    ("I", "Bass Massive",         "tokens", "mutator", "spectral zone"),
    ("J", "Mimi Bend",            "tokens", "mutator", "codec split"),
    ("L", "Spec. Restoration",    "tokens", "mutator", "time surgery"),
    ("Q", "Phase Scramble",       "stft",   "mutator", "phase surgery"),
    # Recursive (a→t→a)
    ("B", "Caption Loop",         "text",   "recurse", "cross-modal loop"),
    ("U", "Cap-Steered Latent",   "latent", "recurse", "text → latent bias"),
    ("V", "Cap-Conditioned Tok.", "tokens", "recurse", "text → token walk"),
    ("W", "Cap → TTA Loop",       "stft",   "recurse", "drift to attractor"),
    # Concatenative
    ("M", "Latent Musaicing",     "latent", "concat",  "external grain"),
    ("R", "Token Musaicing",      "tokens", "concat",  "external grain"),
    ("N", "Concatenator",         "stft",   "concat",  "external grain"),
    ("T", "CLAP Musaicing",       "text",   "concat",  "text → grain match"),
]

ROW_LABEL = {
    "gen":     "∅ → audio  (generator)",
    "mutator": "audio → audio  (mutator)",
    "concat":  "(corpus × target) → audio",
    "recurse": "audio → text → audio  (recursive)",
}

ROWS = ["gen", "mutator", "concat", "recurse"]
COLS = ["weight", "latent", "tokens", "stft", "text"]
COL_LABEL = {
    "weight": "weight-space",
    "latent": "continuous latent",
    "tokens": "discrete tokens",
    "stft":   "STFT mag/phase",
    "text":   "text / cross-modal",
}


def draw_card(ax, mvp_id: str, label: str, domain: str, aesthetic: str,
              x: float, y: float, w: float, h: float, stack_idx: int = 0):
    """Draw a single MVP card at (x,y) with width w, height h, in axes coords."""
    color = COLOR[domain]
    bg = CARD
    border = color

    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.001,rounding_size=0.006",
                          linewidth=2.0, edgecolor=border, facecolor=bg,
                          transform=ax.transAxes, zorder=3)
    ax.add_patch(rect)
    cx = x + w * 0.5
    # Big letter centered top
    ax.text(cx, y + h * 0.72, mvp_id,
            transform=ax.transAxes, color=color,
            fontsize=20, weight="bold",
            ha="center", va="center", zorder=10)
    # Label centered middle
    ax.text(cx, y + h * 0.42, label,
            transform=ax.transAxes, color=FG,
            fontsize=10.5, weight="bold",
            ha="center", va="center", zorder=10)
    # Aesthetic centered bottom (muted)
    ax.text(cx, y + h * 0.18, aesthetic,
            transform=ax.transAxes, color=MUTED,
            fontsize=7.5, ha="center", va="center",
            family="monospace", zorder=10)


def main():
    fig = plt.figure(figsize=(22, 16), dpi=120, facecolor=BG)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Title
    ax.text(0.02, 0.962, "MVP Taxonomy — V1 + V2 (24 MVPs)",
            color=FG, fontsize=30, weight="bold",
            transform=ax.transAxes)
    ax.text(0.02, 0.928,
            "I/O × domain × aesthetic  ·  24 MVPs across 5 domains and 4 I/O modes (V2 fills 6)",
            color=MUTED, fontsize=13, transform=ax.transAxes)

    # Grid params
    grid_top = 0.88
    grid_bottom = 0.10
    grid_left = 0.17
    grid_right = 0.99
    col_w = (grid_right - grid_left) / len(COLS)
    row_h = (grid_top - grid_bottom) / len(ROWS)

    # Column headers (top)
    for ci, c in enumerate(COLS):
        cx = grid_left + col_w * (ci + 0.5)
        # accent bar
        color = COLOR[c]
        ax.add_patch(plt.Rectangle((grid_left + col_w * ci + 0.003, grid_top + 0.005),
                                   col_w - 0.006, 0.014,
                                   color=color, zorder=2,
                                   transform=ax.transAxes))
        ax.text(cx, grid_top + 0.038, COL_LABEL[c],
                color=color, fontsize=12, weight="bold",
                ha="center", va="bottom",
                transform=ax.transAxes)

    # Row headers (left)
    for ri, r in enumerate(ROWS):
        ry = grid_top - row_h * (ri + 0.5)
        ax.text(grid_left - 0.012, ry, ROW_LABEL[r],
                color=FG, fontsize=11, weight="bold",
                ha="right", va="center", transform=ax.transAxes)

    # Vertical + horizontal grid lines
    for ci in range(len(COLS) + 1):
        x = grid_left + col_w * ci
        ax.plot([x, x], [grid_bottom, grid_top],
                color=GRID, linewidth=0.6, transform=ax.transAxes,
                zorder=1)
    for ri in range(len(ROWS) + 1):
        y = grid_top - row_h * ri
        ax.plot([grid_left, grid_right], [y, y],
                color=GRID, linewidth=0.6, transform=ax.transAxes,
                zorder=1)

    # Group MVPs into (row, col) buckets
    buckets: dict[tuple[str, str], list[tuple[str, str, str, str]]] = {}
    for mid, lab, dom, io, aes in MVPS:
        # B sits in both recurse-text and mutator-text; place primary in recurse.
        primary_io = "recurse" if mid == "B" else io
        buckets.setdefault((primary_io, dom), []).append((mid, lab, dom, aes))

    # Draw cards inside each cell.
    # When n >= 4, lay out in 2 sub-columns so each card stays big.
    card_margin = 0.010
    inner_gap = 0.005
    for (io, dom), items in buckets.items():
        ri = ROWS.index(io)
        ci = COLS.index(dom)
        cell_x = grid_left + col_w * ci + card_margin
        cell_y_top = grid_top - row_h * ri - card_margin
        cell_w = col_w - 2 * card_margin
        n = len(items)
        sub_cols = 2 if n >= 4 else 1
        sub_rows = (n + sub_cols - 1) // sub_cols
        usable_h = row_h - 2 * card_margin - inner_gap * (sub_rows - 1)
        usable_w = cell_w - inner_gap * (sub_cols - 1)
        each_h = usable_h / sub_rows
        each_w = usable_w / sub_cols
        for k, (mid, lab, dom2, aes) in enumerate(items):
            sc = k % sub_cols
            sr = k // sub_cols
            cx = cell_x + sc * (each_w + inner_gap)
            cy = cell_y_top - (sr + 1) * each_h - sr * inner_gap
            draw_card(ax, mid, lab, dom2, aes,
                      cx, cy, each_w, each_h, stack_idx=k)

    # B duplicate ghost card in (mutator, text) cell, faded
    ri = ROWS.index("mutator")
    ci = COLS.index("text")
    cell_x = grid_left + col_w * ci + card_margin
    cell_y_top = grid_top - row_h * ri - card_margin
    cell_w = col_w - 2 * card_margin
    each_h = (row_h - 2 * card_margin) - 0.003
    ax.add_patch(FancyBboxPatch(
        (cell_x, cell_y_top - each_h), cell_w, each_h,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        linewidth=1.3, edgecolor=COLOR["text"], facecolor=CARD,
        linestyle=(0, (3, 2)), alpha=0.5,
        transform=ax.transAxes, zorder=3))
    ax.text(cell_x + 0.018, cell_y_top - each_h * 0.45,
            "B*", transform=ax.transAxes,
            color=COLOR["text"], fontsize=20, weight="bold",
            va="center", alpha=0.7, zorder=4)
    ax.text(cell_x + cell_w - 0.012, cell_y_top - each_h * 0.30,
            "(also cross-modal)", transform=ax.transAxes,
            color=MUTED, fontsize=6.5, ha="right", va="center",
            family="monospace", zorder=4)

    # Legend (bottom)
    legend_y = 0.065
    ax.text(0.02, legend_y + 0.025, "Domain palette",
            color=FG, fontsize=10, weight="bold", transform=ax.transAxes)
    lx = 0.02
    for c in COLS:
        ax.add_patch(plt.Rectangle((lx, legend_y), 0.022, 0.014,
                                   color=COLOR[c], transform=ax.transAxes,
                                   zorder=3))
        ax.text(lx + 0.028, legend_y + 0.007, COL_LABEL[c],
                color=FG, fontsize=9, va="center",
                transform=ax.transAxes)
        lx += 0.018 + 0.005 + len(COL_LABEL[c]) * 0.0065 + 0.012

    # Footer
    ax.text(0.02, 0.018,
            "see docs/MVP_TAXONOMY.md for chain affinities, "
            "GUI tab re-ordering proposal, and V2 AudioLLM placement",
            color=MUTED, fontsize=9, transform=ax.transAxes)
    ax.text(0.98, 0.018,
            "scripts/build_taxonomy_diagram.py",
            color=MUTED, fontsize=8, ha="right", va="bottom",
            family="monospace", transform=ax.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
