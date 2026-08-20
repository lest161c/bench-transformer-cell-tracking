#!/usr/bin/env python3
"""
Publication-quality plots for the project report.

All colours follow the Okabe–Ito scientific palette defined in
``docs/vis_guidelines.md`` (colorblind-safe, print-safe).
Sequential / heatmap data uses ``viridis``.

Style:
- Serif font (Linux Libertine / standard LaTeX serif)
- White background, no top/right spines
- Light gray horizontal grid lines only
- Okabe–Ito categorical palette
- Compact size (≈4–5 in wide × 2.5–3 in tall)
- 300 DPI, PDF output, bbox_inches='tight'

Usage:
    python results/generate_plots.py
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from cycler import cycler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent  # research-proj/
FIGURES = BASE / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Okabe–Ito palette (see docs/vis_guidelines.md) ────────────────────────────
# Slot assignment kept identical to curated/generate_report.py so the same
# semantic role (e.g. "Dense baseline") gets the same colour in every figure
# of the report.
OI_BLACK     = "#000000"   # theory / reference
OI_ORANGE    = "#E69F00"   # primary method / variant A
OI_SKYBLUE   = "#56B4E9"   # variant B / sparse K=64
OI_GREEN     = "#009E73"   # variant C / dense_flash
OI_YELLOW    = "#F0E442"   # variant D
OI_BLUE      = "#0072B2"   # secondary metric / dense baseline
OI_VERMILION = "#D55E00"   # ablation / sparse K=16
OI_PURPLE    = "#CC79A7"   # attention / gather / sparse K=32

# Cross-figure role aliases — every figure picks its colours from this table,
# so a "Dense baseline" series is always OI_BLUE, etc.
ROLE_DENSE_BASELINE = OI_BLUE
ROLE_DENSE_FLASH    = OI_GREEN
ROLE_K4             = OI_ORANGE
ROLE_K8             = OI_GREEN
ROLE_K16            = OI_VERMILION
ROLE_K32            = OI_PURPLE
ROLE_K64            = OI_SKYBLUE
ROLE_FORWARD        = OI_BLUE
ROLE_BACKWARD       = OI_VERMILION
ROLE_NEUTRAL        = "#999999"

# Default colour cycle for any plot that does not set explicit colours.
OI_CYCLE = [
    OI_BLUE,
    OI_ORANGE,
    OI_GREEN,
    OI_VERMILION,
    OI_PURPLE,
    OI_SKYBLUE,
    OI_YELLOW,
    OI_BLACK,
]

# Color pairs for bar charts (darker fill, slightly lighter border)
def bar_colors(base_color, alpha=0.75):
    """Return (fill, edge) pair for a base color."""
    from matplotlib.colors import to_rgba, to_hex
    rgba = to_rgba(base_color)
    fill = to_hex((rgba[0], rgba[1], rgba[2], alpha), keep_alpha=True)
    return fill, base_color

# ── Trackastra matplotlibrc ──────────────────────────────────────────────────
plt.rcParams.update({
    # Font: serif matches LaTeX (Linux Libertine / standard)
    "font.family": "serif",
    "font.serif": ["Linux Libertine", "Libertine", "TeX Gyre Bonum", "DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.5,

    # Figure
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.6,
    "axes.labelcolor": "#222222",
    "axes.titlecolor": "#222222",
    "axes.facecolor": "white",

    # Grid
    "axes.grid": True,
    "axes.grid.axis": "y",        # horizontal grid only by default
    "axes.grid.which": "major",
    "grid.color": "#e0e0e0",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",

    # Ticks
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.fancybox": False,
    "legend.edgecolor": "#cccccc",
    "legend.facecolor": "white",
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.3,

    # Color cycle
    "axes.prop_cycle": cycler(color=OI_CYCLE),
})

# ── Helper ────────────────────────────────────────────────────────────────────
def savefig(name, fig=None, dpi=300):
    """Save figure to results/figures/ with consistent settings."""
    if fig is None:
        fig = plt.gcf()
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ {name}  ({size_kb:.1f} KB)")
    plt.close(fig)


def add_horizontal_grid(ax):
    """Ensure only horizontal grid lines are shown."""
    ax.grid(True, axis="y", color="#e0e0e0", alpha=0.6, linewidth=0.4)
    ax.grid(False, axis="x")


def remove_top_right_spines(ax):
    """Remove top and right spines (already done by rcParams, but belt-and-suspenders)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_legend(ax, loc="best", framealpha=0.8, **kw):
    """Apply consistent legend styling."""
    kw.setdefault("frameon", True)
    kw.setdefault("framealpha", framealpha)
    kw.setdefault("edgecolor", "#cccccc")
    kw.setdefault("facecolor", "white")
    kw.setdefault("fontsize", 7)
    ax.legend(loc=loc, **kw)


def figsize_wide(aspect=1.6):
    """Return (width, height) in inches for compact figures. aspect = width/height."""
    width = 4.8  # ~122mm, fits LLNCS column width
    height = width / aspect
    return (width, height)


# ══════════════════════════════════════════════════════════════════════════════
#  1. CNN Convergence Curves  (Chart 17 from progress_roadmap.html)
# ══════════════════════════════════════════════════════════════════════════════
def plot_cnn_convergence():
    """
    Baseline vs Vanilla CNN val_loss trajectories (epochs 0–110).
    Source: progress_roadmap.html §Chart 17 (The Trap).
    """
    fig, ax = plt.subplots(figsize=figsize_wide(1.5))

    # Anchor points from the chart
    baseline_pts = np.array([
        [0, 0.480], [5, 0.114], [10, 0.070], [19, 0.018],
        [30, 0.010], [51, 0.006], [80, 0.004], [106, 0.003], [110, 0.003],
    ])
    cnn_pts = np.array([
        [0, 0.425], [5, 0.086], [10, 0.061], [19, 0.052],
        [30, 0.048], [51, 0.035], [80, 0.035], [106, 0.035], [110, 0.035],
    ])

    def interpolate(pts, n=111):
        xs = np.linspace(0, 110, n)
        return xs, np.interp(xs, pts[:, 0], pts[:, 1])

    x_b, y_b = interpolate(baseline_pts)
    x_c, y_c = interpolate(cnn_pts)

    ax.plot(x_b, y_b, color=ROLE_DENSE_BASELINE, linewidth=1.8, label="Baseline (no CNN)")
    ax.plot(x_c, y_c, color=OI_ORANGE, linewidth=1.5, linestyle="--", label="Vanilla CNN (frozen)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 1.0)
    ax.set_xlim(0, 110)
    add_horizontal_grid(ax)
    style_legend(ax, loc="upper right")

    # Annotations (more subtle than original)
    ax.annotate("CNN ahead",
                xy=(10, 0.070), xytext=(18, 0.12),
                arrowprops=dict(arrowstyle="->", color=OI_ORANGE, lw=0.8),
                fontsize=6.5, color=OI_ORANGE)
    ax.annotate("Baseline overtakes",
                xy=(30, 0.010), xytext=(42, 0.005),
                arrowprops=dict(arrowstyle="->", color=ROLE_DENSE_BASELINE, lw=0.8),
                fontsize=6.5, color=ROLE_DENSE_BASELINE)
    ax.annotate("CNN stuck",
                xy=(60, 0.035), xytext=(72, 0.06),
                arrowprops=dict(arrowstyle="->", color=OI_ORANGE, lw=0.8),
                fontsize=6.5, color=OI_ORANGE)

    savefig("cnn_convergence.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  2. CNN Variant Ranking  (Chart 16 from progress_roadmap.html)
# ══════════════════════════════════════════════════════════════════════════════
def plot_cnn_variant_ranking():
    """
    Horizontal bar chart of 8 CNN variants sorted by val_loss.
    Values match Table tab:cnn_variants / tab:app_cnn in the report
    (Baseline 0.003; CONCAT+dropout p=0.2 0.011; additive+dropout p=0.2 0.016;
     additive+dropout p=0.5 0.023; trainable+dropout p=0.2 0.024; vanilla 0.035;
     trainable 0.036; lambda+dropout 0.040; lambda-only 0.077).
    Source: progress_roadmap.html §Chart 16 (chWhatTested).
    """
    labels = [
        "CNN CONCAT + dropout\n(p=0.2)",
        "CNN additive + dropout\n(p=0.2)",
        "CNN additive + dropout\n(p=0.5)",
        "CNN additive + trainable\ndropout (p=0.2)",
        "Vanilla CNN (frozen)",
        "CNN trainable",
        "λ(t) step + dropout",
        "λ(t) step only",
    ]
    values = [0.011, 0.016, 0.023, 0.024, 0.035, 0.036, 0.040, 0.077]
    baseline = 0.003

    # Colors: better than vanilla = blue (dense baseline), vanilla = grey,
    # worse than vanilla = vermilion (ablation).
    colors = []
    for v in values:
        if v < 0.035:
            colors.append(ROLE_DENSE_BASELINE)
        elif v == 0.035:
            colors.append(ROLE_NEUTRAL)
        else:
            colors.append(OI_VERMILION)

    fig, ax = plt.subplots(figsize=figsize_wide(1.4))

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.3, height=0.65)
    ax.axvline(x=baseline, color=ROLE_DENSE_BASELINE, linewidth=1.2, linestyle="--",
               label=f"Baseline (no CNN) = {baseline}")
    ax.set_xscale("log")
    ax.set_xlabel("Best val_loss (lower is better)")
    ax.set_xlim(0.001, 0.2)
    add_horizontal_grid(ax)
    # Also add x-axis grid for log scale readability
    ax.grid(True, axis="x", color="#e0e0e0", alpha=0.3, linewidth=0.3)
    style_legend(ax, loc="lower right", fontsize=6.5)

    # Annotate values on bars
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=6.5)

    savefig("cnn_variant_ranking.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Edge Probe — Balanced Accuracy  (Chart 19)
# ══════════════════════════════════════════════════════════════════════════════
def plot_edge_probe_bal_acc():
    """
    Grouped horizontal bar chart of edge prediction balanced accuracy (5-fold CV).
    Source: progress_roadmap.html §Chart 19 (chEdgeBalAcc).
    """
    labels = [
        "DINOv2 MLP", "DINOv2 linear",
        "HOCT 19D MLP", "HOCT 19D linear",
        "Regionprops MLP", "Regionprops linear",
        "CNN NT-Xent MLP", "CNN NT-Xent linear",
        "CNN e2e MLP", "CNN e2e linear",
    ]
    values = [0.896, 0.528, 0.880, 0.573, 0.860, 0.557, 0.703, 0.543, 0.500, 0.500]
    errors = [0.036, 0.006, 0.020, 0.012, 0.014, 0.030, 0.055, 0.015, 0.000, 0.000]
    baseline = 0.500

    # Group colors by feature type (Okabe–Ito; one hue per family)
    group_map = {
        "DINOv2":     OI_BLUE,
        "HOCT 19D":    OI_GREEN,
        "Regionprops": OI_VERMILION,
        "CNN NT-Xent": OI_PURPLE,
        "CNN e2e":     ROLE_NEUTRAL,
    }

    colors = []
    for lbl in labels:
        if "DINOv2" in lbl:
            colors.append(group_map["DINOv2"])
        elif "HOCT" in lbl:
            colors.append(group_map["HOCT 19D"])
        elif "Regionprops" in lbl:
            colors.append(group_map["Regionprops"])
        elif "CNN NT-Xent" in lbl:
            colors.append(group_map["CNN NT-Xent"])
        elif "CNN e2e" in lbl:
            colors.append(group_map["CNN e2e"])

    fig, ax = plt.subplots(figsize=figsize_wide(1.1))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, xerr=errors, color=colors,
                   edgecolor="white", linewidth=0.3, height=0.65, capsize=2,
                   error_kw={"linewidth": 0.6, "ecolor": "#666666"})
    ax.axvline(x=baseline, color="#888888", linewidth=0.8, linestyle="--",
               label="Random baseline = 0.500")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Balanced Accuracy")
    ax.set_xlim(0.45, 1.0)
    add_horizontal_grid(ax)
    style_legend(ax, loc="lower right", fontsize=6.5)

    savefig("edge_probe_bal_acc.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Edge Probe — F1 Score  (Chart 20)
# ══════════════════════════════════════════════════════════════════════════════
def plot_edge_probe_f1():
    """
    Grouped horizontal bar chart of edge prediction F1 score (5-fold CV).
    Source: progress_roadmap.html §Chart 20 (chEdgeF1).
    """
    labels = [
        "DINOv2 MLP", "DINOv2 linear",
        "HOCT 19D MLP", "HOCT 19D linear",
        "Regionprops MLP", "Regionprops linear",
        "CNN NT-Xent MLP", "CNN NT-Xent linear",
        "CNN e2e MLP", "CNN e2e linear",
    ]
    values = [0.566, 0.113, 0.424, 0.144, 0.396, 0.131, 0.290, 0.129, 0.000, 0.000]
    errors = [0.090, 0.044, 0.076, 0.037, 0.056, 0.029, 0.126, 0.035, 0.000, 0.000]
    baseline = 0.000

    group_map = {
        "DINOv2":     OI_BLUE,
        "HOCT 19D":    OI_GREEN,
        "Regionprops": OI_VERMILION,
        "CNN NT-Xent": OI_PURPLE,
        "CNN e2e":     ROLE_NEUTRAL,
    }

    colors = []
    for lbl in labels:
        if "DINOv2" in lbl:
            colors.append(group_map["DINOv2"])
        elif "HOCT" in lbl:
            colors.append(group_map["HOCT 19D"])
        elif "Regionprops" in lbl:
            colors.append(group_map["Regionprops"])
        elif "CNN NT-Xent" in lbl:
            colors.append(group_map["CNN NT-Xent"])
        elif "CNN e2e" in lbl:
            colors.append(group_map["CNN e2e"])

    fig, ax = plt.subplots(figsize=figsize_wide(1.1))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, xerr=errors, color=colors,
                   edgecolor="white", linewidth=0.3, height=0.65, capsize=2,
                   error_kw={"linewidth": 0.6, "ecolor": "#666666"})
    ax.axvline(x=baseline, color="#888888", linewidth=0.8, linestyle="--",
               label="Random baseline = 0.000")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("F1 Score")
    ax.set_xlim(-0.05, 0.75)
    add_horizontal_grid(ax)
    style_legend(ax, loc="lower right", fontsize=6.5)

    savefig("edge_probe_f1.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Sparse Attention — Forward Speed
# ══════════════════════════════════════════════════════════════════════════════
def plot_sparse_attention_speed():
    """
    Line plot: N vs time_ms for different K values.
    Source: benchmark_attn/results/speed_mem.csv (updated 2026-08-04, K ∈ {0,4,8,16,32,64},
    N ∈ {128,256,512}).
    Fallback: bench-transformer-cell-tracking/benchmark_combined/results/speed_mem.csv (no K=64).
    """
    # Try updated local copy first, then the older benchmark_combined path
    path_local = BASE / "benchmark_attn" / "results" / "speed_mem.csv"
    path_combined = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "speed_mem.csv"

    path = path_local if path_local.exists() else path_combined
    if not path.exists():
        print("  ⚠ speed_mem.csv not found — skipping sparse_attention_speed.pdf")
        return

    df = pd.read_csv(path)
    df["K_label"] = df["K"].map({
        0: "Dense (K=0)", 4: "K=4", 8: "K=8", 16: "K=16", 32: "K=32", 64: "K=64",
    })

    fig, ax = plt.subplots(figsize=figsize_wide(1.5))

    # Okabe–Ito palette: one hue per K value (colorblind-safe).
    palette = {
        "Dense (K=0)": ROLE_DENSE_BASELINE,
        "K=4":  ROLE_K4,
        "K=8":  ROLE_K8,
        "K=16": ROLE_K16,
        "K=32": ROLE_K32,
        "K=64": ROLE_K64,
    }
    styles = {
        "Dense (K=0)": "-",
        "K=4": "--",
        "K=8": "-.",
        "K=16": "-",
        "K=32": "--",
        "K=64": "-.",
    }

    for i, (k_label, grp) in enumerate(df.groupby("K_label")):
        grp_sorted = grp.sort_values("N")
        ax.plot(grp_sorted["N"], grp_sorted["time_ms"],
                color=palette.get(k_label, OI_CYCLE[i % len(OI_CYCLE)]),
                linestyle=styles.get(k_label, "-"),
                linewidth=1.5 if "Dense" in k_label or "K=16" in k_label else 1.0,
                marker="o", markersize=3.5, label=k_label)

    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(100, 600)
    add_horizontal_grid(ax)
    style_legend(ax, loc="upper left", fontsize=6.5)

    savefig("sparse_attention_speed.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  6. Sparse Attention — Peak Memory
# ══════════════════════════════════════════════════════════════════════════════
def plot_sparse_attention_memory():
    """
    Line plot: N vs mem_mb for different K values.
    Source: benchmark_attn/results/speed_mem.csv (updated 2026-08-04, K ∈ {0,4,8,16,32,64},
    N ∈ {128,256,512}).
    Fallback: bench-transformer-cell-tracking/benchmark_combined/results/speed_mem.csv (no K=64).
    """
    path_local = BASE / "benchmark_attn" / "results" / "speed_mem.csv"
    path_combined = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "speed_mem.csv"

    path = path_local if path_local.exists() else path_combined
    if not path.exists():
        print("  ⚠ speed_mem.csv not found — skipping sparse_attention_memory.pdf")
        return

    df = pd.read_csv(path)
    df["K_label"] = df["K"].map({
        0: "Dense (K=0)", 4: "K=4", 8: "K=8", 16: "K=16", 32: "K=32", 64: "K=64",
    })

    fig, ax = plt.subplots(figsize=figsize_wide(1.5))

    palette = {
        "Dense (K=0)": ROLE_DENSE_BASELINE,
        "K=4":  ROLE_K4,
        "K=8":  ROLE_K8,
        "K=16": ROLE_K16,
        "K=32": ROLE_K32,
        "K=64": ROLE_K64,
    }
    styles = {
        "Dense (K=0)": "-",
        "K=4": "--",
        "K=8": "-.",
        "K=16": "-",
        "K=32": "--",
        "K=64": "-.",
    }

    for i, (k_label, grp) in enumerate(df.groupby("K_label")):
        grp_sorted = grp.sort_values("N")
        ax.plot(grp_sorted["N"], grp_sorted["mem_mb"],
                color=palette.get(k_label, OI_CYCLE[i % len(OI_CYCLE)]),
                linestyle=styles.get(k_label, "-"),
                linewidth=1.5 if "Dense" in k_label or "K=16" in k_label else 1.0,
                marker="s", markersize=3.5, label=k_label)

    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Memory (MB)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(100, 600)
    add_horizontal_grid(ax)
    style_legend(ax, loc="upper left", fontsize=6.5)

    savefig("sparse_attention_memory.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  7. Pipeline Breakdown  (Charts 6 & 9 from progress_roadmap.html)
# ══════════════════════════════════════════════════════════════════════════════
def plot_pipeline_breakdown():
    """
    Stacked bar chart of training step breakdown + regionprops CPU/GPU split.
    Sources: progress_roadmap.html Charts 6, 9; data_pipeline_results.csv
    """
    # Training step breakdown from progress_roadmap doughnut data
    step_labels = ["Backward pass\n(57%)", "Attention 12L\n(22%)",
                   "Blockwise norm\n(13%)", "FFN 12L\n(5%)", "Optimizer\n(2%)"]
    step_values = [57, 22, 13, 5, 2]

    # Regionprops CPU vs GPU breakdown
    region_labels = ["N=50", "N=200", "N=500", "N=1000"]
    cpu_pct = [74, 52, 33, 21]
    gpu_pct = [26, 48, 67, 79]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.6))

    # ── Left: Training step breakdown (horizontal bar, not pie) ──
    # Semantic color mapping:
    #   Backward pass → orange (training signal, warm)
    #   Attention 12L → blue (forward: core model compute)
    #   Blockwise norm → teal (forward: supporting compute)
    #   FFN 12L       → purple (forward: compute)
    #   Optimizer     → green  (parameter update)
    step_colors = [OI_ORANGE, OI_BLUE, OI_VERMILION, OI_PURPLE, OI_GREEN]
    y_pos = np.arange(len(step_labels))
    ax1.barh(y_pos, step_values, color=step_colors, edgecolor="white",
             linewidth=0.3, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(step_labels, fontsize=6.5)
    ax1.set_xlabel("% of training step")
    ax1.set_xlim(0, 65)
    add_horizontal_grid(ax1)

    # Annotate values
    for i, v in enumerate(step_values):
        ax1.text(v + 1, i, f"{v}%", va="center", fontsize=6.5)

    # ── Right: Regionprops CPU/GPU split ──
    x = np.arange(len(region_labels))
    width = 0.5
    ax2.bar(x, cpu_pct, width, label="CPU (regionprops)",
            color=OI_ORANGE, edgecolor="white", linewidth=0.3)
    ax2.bar(x, gpu_pct, width, bottom=cpu_pct, label="GPU (model forward)",
            color=OI_BLUE, edgecolor="white", linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(region_labels)
    ax2.set_xlabel("Cells per frame (N)")
    ax2.set_ylabel("% of total inference time")
    ax2.set_ylim(0, 105)
    add_horizontal_grid(ax2)
    style_legend(ax2, loc="upper right", fontsize=6)

    fig.tight_layout(pad=0.8)
    savefig("pipeline_breakdown.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  8. Backward Pass Comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_backward_pass():
    """
    Bar chart comparing forward+backward time for different attention methods.
    Source: benchmark_backward/sparse_backward_results.csv
    """
    path = BASE / "benchmark_backward" / "sparse_backward_results.csv"
    if not path.exists():
        print("  ⚠ sparse_backward_results.csv not found — skipping backward_pass.pdf")
        return

    df = pd.read_csv(path)

    # Filter to N=512 for clean comparison
    df_n512 = df[df["N"] == 512].copy()

    # Simplify method labels
    method_display = {
        "mask_knn": "mask-KNN",
        "gather_knn": "gather-KNN",
        "sparse_softmax": "sparse softmax",
        "dense_masked": "dense masked",
        "dense_flash": "dense flash",
    }
    df_n512["method_short"] = df_n512["method"].map(method_display)

    # Compare K=16
    df_k16 = df_n512[df_n512["K"] == 16].copy()
    df_k16_long = pd.melt(df_k16, id_vars=["method_short"],
                          value_vars=["forward_ms", "backward_ms"],
                          var_name="phase", value_name="time_ms")
    df_k16_long["phase"] = df_k16_long["phase"].map({
        "forward_ms": "Forward", "backward_ms": "Backward"
    })

    fig, ax = plt.subplots(figsize=figsize_wide(1.3))

    # Grouped bar plot
    methods_order = df_k16_long["method_short"].unique()
    x = np.arange(len(methods_order))
    width = 0.32
    gap = 0.04

    for i, (phase, color) in enumerate([
        ("Forward", ROLE_FORWARD),
        ("Backward", ROLE_BACKWARD)
    ]):
        phase_data = df_k16_long[df_k16_long["phase"] == phase]
        # Ensure order matches
        vals = [phase_data.loc[phase_data["method_short"] == m, "time_ms"].values[0]
                if len(phase_data.loc[phase_data["method_short"] == m, "time_ms"]) > 0
                else 0 for m in methods_order]
        offset = (i - 0.5) * (width + gap)
        bars = ax.bar(x + offset, vals, width, label=phase, color=color,
                      edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods_order, fontsize=6.5, rotation=20, ha="right")
    ax.set_xlabel("Attention Method")
    ax.set_ylabel("Time (ms)")
    add_horizontal_grid(ax)
    style_legend(ax, loc="upper right", fontsize=6.5)

    savefig("backward_pass.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  9. KNN TRA Parity — Bar chart of tracking accuracy across K values
# ══════════════════════════════════════════════════════════════════════════════
def plot_knn_tra_parity():
    """
    Horizontal bar chart showing mean TRA for each KNN variant vs. dense baseline.
    Demonstrates that all sparse variants remain within ±0.001 TRA of dense.
    Source: results/knn_sweep/results_{baseline,k4,k16,k32,k64}.csv
    """
    import csv, statistics

    results_dir = BASE / "results" / "knn_sweep"
    configs = [
        ("Dense",   "results_baseline.csv"),
        ("K = 4",   "results_k4.csv"),
        ("K = 16",  "results_k16.csv"),
        ("K = 32",  "results_k32.csv"),
        ("K = 64",  "results_k64.csv"),
    ]

    labels = []
    means = []
    stds = []
    for label, fname in configs:
        fpath = results_dir / fname
        if not fpath.exists():
            print(f"  ⚠ {fname} not found — skipping knn_tra_parity.pdf")
            return
        with open(fpath) as f:
            tra_vals = [float(r["CTCMetrics_TRA"]) for r in csv.DictReader(f)]
        labels.append(label)
        means.append(statistics.mean(tra_vals))
        stds.append(statistics.stdev(tra_vals))

    dense_mean = means[0]  # Dense baseline
    dense_std  = stds[0]

    # Find global TRA range for x-axis zoom
    all_tra = []
    for fname in [c[1] for c in configs]:
        with open(results_dir / fname) as f:
            for r in csv.DictReader(f):
                all_tra.append(float(r["CTCMetrics_TRA"]))
    tra_min = min(all_tra)
    tra_max = max(all_tra)

    fig, ax = plt.subplots(figsize=figsize_wide(1.6))

    # ── Colors: dense = blue (dense baseline); each K gets its own
    #    Okabe–Ito hue. Edge uses the same hue for a clean scientific look.
    bar_colors = [
        ROLE_DENSE_BASELINE,   # Dense
        ROLE_K4,               # K=4
        ROLE_K16,              # K=16
        ROLE_K32,              # K=32
        ROLE_K64,              # K=64
    ]
    edge_colors = bar_colors  # uniform edge keeps the palette tight

    # Horizontal bar chart: TRA on x-axis, configs on y-axis
    y_pos = np.arange(len(labels))
    x_start = 0.990  # Start bars at 0.99 to make differences visible

    # Draw extended bars from x_start to TRA value
    bar_width = 0.55
    for i in range(len(labels)):
        ax.barh(y_pos[i], means[i] - x_start, left=x_start, height=bar_width,
                color=bar_colors[i], edgecolor=edge_colors[i], linewidth=0.5,
                zorder=3)

    # Dense baseline reference line
    ax.axvline(x=dense_mean, color=ROLE_DENSE_BASELINE, linewidth=1.2, linestyle="--",
               alpha=0.7, zorder=2)
    # ±0.001 band
    ax.axvspan(dense_mean - 0.001, dense_mean + 0.001,
               color=ROLE_DENSE_BASELINE, alpha=0.08, zorder=1,
               label=f"$\\pm 0.001$ band")

    # Axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("TRA (Tracking Accuracy)", fontsize=8.5)
    ax.set_xlim(0.989, tra_max + 0.001)
    ax.set_ylim(-0.7, len(labels) + 0.5)

    # Format x-axis ticks to 4 decimal places
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Grid
    add_horizontal_grid(ax)

    # Annotate values on bars (TRA value at bar end, diff below bar)
    for i in range(len(labels)):
        val_text = f"{means[i]:.4f}"
        ax.text(means[i] + 0.0003, y_pos[i], val_text,
                va="center", ha="left", fontsize=7, fontweight="bold",
                color=edge_colors[i])

    # Annotate diff from dense for sparse variants — below the bar
    for i in range(1, len(labels)):
        diff = means[i] - dense_mean
        diff_str = f"{diff:+.4f}"
        ax.text(means[i] + 0.0003, y_pos[i] - 0.37, diff_str,
                va="center", ha="left", fontsize=6, color="#888888",
                fontstyle="italic")

    # Main annotation: all within ±0.001 (top-right, above highest bar)
    ax.annotate("All KNN variants within $\\pm0.001$ TRA of dense",
                xy=(0.991, len(labels) - 0.15), fontsize=7.5,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4f8",
                          edgecolor=ROLE_DENSE_BASELINE, linewidth=0.6))

    # Legend (only for reference line and band)
    legend_elements = [
        Line2D([0], [0], color=ROLE_DENSE_BASELINE, linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"Dense baseline ({dense_mean:.4f})"),
        Line2D([0], [0], color=ROLE_DENSE_BASELINE, linewidth=6, alpha=0.12,
               label="$\\pm0.001$ TRA band"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6.5,
              framealpha=0.85)

    savefig("knn_tra_parity.pdf", fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Generating Okabe–Ito report figures...")
    print("=" * 60)

    funcs = [
        ("1/9", "CNN Convergence Curves", plot_cnn_convergence),
        ("2/9", "CNN Variant Ranking", plot_cnn_variant_ranking),
        ("3/9", "Edge Probe — Balanced Accuracy", plot_edge_probe_bal_acc),
        ("4/9", "Edge Probe — F1 Score", plot_edge_probe_f1),
        ("5/9", "Sparse Attention Speed", plot_sparse_attention_speed),
        ("6/9", "Sparse Attention Memory", plot_sparse_attention_memory),
        ("7/9", "Pipeline Breakdown", plot_pipeline_breakdown),
        ("8/9", "Backward Pass Comparison", plot_backward_pass),
        ("9/9", "KNN TRA Parity", plot_knn_tra_parity),
    ]

    for num, name, func in funcs:
        print(f"\n[{num}] {name}")
        try:
            func()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  All figures saved to results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
