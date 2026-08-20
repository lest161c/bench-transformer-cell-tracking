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
vermilion (``#D55E00``).

Run::

    python scripts/generate_amdahl_figure.py

No external data is required; the figure is entirely theoretical.
"""

from pathlib import Path

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
    """Generate and save the Amdahl's Law figure."""
    attention_share = np.linspace(0.0, 1.0, 500)

    speedups = [1.5, 1.78, 2.0, 5.0, 10.0]
    # Okabe–Ito categorical palette (see docs/vis_guidelines.md).
    oi_colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for isolated_speedup, color in zip(speedups, oi_colors):
        training_speedup = amdahl_speedup(attention_share, isolated_speedup)
        label = f"$k = {isolated_speedup:g}\\times$"
        if isolated_speedup == 1.78:
            label = f"$k = {isolated_speedup:.2f}\\times$ (FlashAttention-2)"
            ax.plot(attention_share, training_speedup, color=color, linewidth=2.5,
                    label=label, zorder=5)
            ax.axvline(x=0.05, color="#D55E00", linestyle="--", linewidth=1.2, alpha=0.7,
                       label="Vanvliet scale ($s \\approx 0.05$)", zorder=4)
            ax.plot(0.05, amdahl_speedup(np.array([0.05]), isolated_speedup)[0],
                    "o", color="#D55E00", markersize=8, zorder=6)
        else:
            ax.plot(attention_share, training_speedup, color=color, linewidth=1.8,
                    label=label)

    ax.set_xlabel("Attention share of step time ($s$)", fontsize=12)
    ax.set_ylabel("Training speedup ($1 / ((1-s) + s/k)$)", fontsize=12)
    ax.set_title("Amdahl's Law: Training Speedup vs. Attention Share", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 10)
    ax.axhline(y=1, color="gray", linewidth=0.5, linestyle="-")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    ax.annotate(
        f"$s=0.05,\\; k=1.78\\times$\n→ $1.022\\times$ training\n(only $2.2\\%$ speedup)",
        xy=(0.05, 1.022),
        xytext=(0.22, 1.6),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.2),
        color="#D55E00",
    )

    fig.tight_layout()

    output_dir = Path("resources/figures")
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
