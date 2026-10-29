r"""Generate the Amdahl's Law figure for the report.

This script creates ``15_amdahl_law.pdf`` (and the SVG source) in the
report's ``resources/figures/`` directory.  The figure shows the
training-time speedup predicted by Amdahl's Law as a function of the
attention share of step time, for several isolated attention speedups.

The formula is::

    training_speedup = 1 / ((1 - s) + s / k)

where *s* is the attention share of total step time and *k* is the
isolated attention speedup (e.g.\ 1.78 for FlashAttention-2 vs.\
masked EfficientAttention at ``N=128``).

Colours follow the Okabe–Ito scientific palette
(``docs/vis_guidelines.md``); the FlashAttention-2 highlight uses
vermilion (``palette.VERMILION``).

Run::

    python scripts/generate_amdahl_figure.py

No external data is required; the figure is entirely theoretical.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "betterplots"))
from betterplots import palette, set_style
set_style(usetex=False, colors="okabe-ito", despine=True,
          font_size=8.5, label_size=8.5, tick_size=7.5, legend_font_size=7.5)

import matplotlib.pyplot as plt
import numpy as np


def amdahl_speedup(attention_share: np.ndarray, isolated_speedup: float) -> np.ndarray:
    """Return Amdahl's Law training speedup.

    Parameters
    ----------
    attention_share : np.ndarray
        Fraction of step time spent in attention (0 to 1).
    isolated_speedup : float
        Speedup of the attention kernel in isolation.

    Returns
    -------
    np.ndarray
        Predicted overall training speedup.
    """
    return 1.0 / ((1.0 - attention_share) + attention_share / isolated_speedup)


def main() -> None:
    """Generate and save the Amdahl's Law figure.

    Writes `15_amdahl_law.{svg,pdf}` into the report figure directory
    (`--out-dir`, default: the live report's `report-shorten/figs`).

    The plot is zoomed to the small-s region (s <= 0.15) where the
    operating point of interest (vanvliet scale, s ~ 0.05) lives; on the
    full [0, 1] axis all curves collapse into an invisible sliver near
    1.0. The y-range covers every plotted curve without clipping (the
    k = 10x curve ends at 1.156, so 1.19 leaves headroom). A horizontal
    line marks the measured end-to-end speedup (~3 %); its intersection
    with the k = 1.78x curve at s ~ 0.067 shows the attention share the
    measurement implies -- close to the real s ~ 0.05, which validates
    the model.
    """
    # Sample on a fine grid so curves are smooth at the zoomed scale.
    attention_share = np.linspace(0.0, 0.15, 500)

    speedups = [1.5, 1.78, 2.0, 5.0, 10.0]
    # Okabe–Ito categorical palette (see docs/vis_guidelines.md).
    oi_colors = [palette.ORANGE, palette.SKYBLUE, palette.GREEN,
                 palette.BLUE, palette.VERMILION]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for isolated_speedup, color in zip(speedups, oi_colors):
        training_speedup = amdahl_speedup(attention_share, isolated_speedup)
        label = f"$k = {isolated_speedup:g}\\times$"
        if isolated_speedup == 1.78:
            label = f"$k = {isolated_speedup:.2f}\\times$ (FlashAttention-2)"
            ax.plot(attention_share, training_speedup, color=color, linewidth=2.5,
                    label=label, zorder=5)
            ax.plot(0.05, amdahl_speedup(np.array([0.05]), isolated_speedup)[0],
                    "o", color=palette.VERMILION, markersize=8, zorder=6)
        else:
            ax.plot(attention_share, training_speedup, color=color, linewidth=1.8,
                    label=label)

    # Operating point: vanvliet attention share s ~ 0.05.
    ax.axvline(x=0.05, color=palette.VERMILION, linestyle="--", linewidth=1.2,
               label="Vanvliet scale ($s \\approx 0.05$)", zorder=4)
    # Measured end-to-end gain: ~3 % (3.6 vs 3.5 min/epoch). The
    # intersection of this line with each kernel curve is the attention
    # share that would be needed to explain the measurement.
    ax.axhline(y=1.03, color=palette.BLUE, linestyle=":", linewidth=1.2,
               label="Measured end-to-end ($\\approx$3 %)", zorder=4)
    intersection_s = 0.03 * 1.78 / (1.78 - 1.0)
    ax.plot(intersection_s, 1.03, "s", color=palette.BLUE, markersize=6, zorder=6)
    ax.annotate(
        f"$k=1.78\\times$ meets 3 % at $s\\approx{intersection_s:.2f}$\n"
        "(1.3$\\times$ the real share)",
        xy=(intersection_s, 1.03),
        xytext=(0.09, 1.10),
        arrowprops=dict(arrowstyle="->", color=palette.BLUE, lw=1.0),
        color=palette.BLUE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=palette.BLUE, linewidth=0.6),
        zorder=7,
    )

    ax.set_xlabel("Attention share of step time ($s$)")
    ax.set_ylabel("Training speedup ($1 / ((1-s) + s/k)$)")
    # No internal title: the beamer frame title carries the message.
    ax.set_xlim(0, 0.15)
    ax.set_ylim(1.0, 1.21)
    ax.axhline(y=1, color="#999999", linewidth=0.5, linestyle="-")
    # Legend above the axes: cannot collide with curves or annotations.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=3,
              frameon=False, columnspacing=1.0,
              handlelength=1.5)
    ax.grid(True)

    ax.annotate(
        "$s=0.05,\\; k=1.78\\times$\n$\\rightarrow$ 2.2 % predicted",
        xy=(0.05, 1.022),
        xytext=(0.055, 1.155),
        arrowprops=dict(arrowstyle="->", color=palette.VERMILION, lw=1.2),
        color=palette.VERMILION,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=palette.VERMILION, linewidth=0.6),
        zorder=8,
    )

    fig.tight_layout()

    output_dir = Path(_REPO_ROOT) / "report-shorten" / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    svg_path = output_dir / "15_amdahl_law.svg"
    pdf_path = output_dir / "15_amdahl_law.pdf"

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Generated: {svg_path}")
    print(f"Generated: {pdf_path}")


if __name__ == "__main__":
    main()
