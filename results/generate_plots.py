#!/usr/bin/env python3
"""
Publication-quality seaborn plots for Trackastra Sparse Attention + CNN Feature Injection.
Reads raw CSV/JSON/hard-coded data and generates PDF figures in results/figures/.

Usage:
    cd /home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj
    python results/generate_plots.py

Requirements: seaborn, matplotlib, pandas, numpy, scipy
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ── Globals ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent  # research-proj/
FIGURES = BASE / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

VIRIDIS = sns.color_palette("viridis", n_colors=10)
DEEP = sns.color_palette("deep", n_colors=10)
MUTED = sns.color_palette("muted", n_colors=10)


# ── Helper ────────────────────────────────────────────────────────────────────
def savefig(name):
    """Save figure to results/figures/ and print confirmation."""
    path = FIGURES / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ {name}  ({size_kb:.1f} KB)")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  1. CNN Convergence Curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_cnn_convergence():
    """
    Baseline vs Vanilla CNN val_loss trajectories (epochs 0–110).
    Source: progress_roadmap.html §Chart 17 (The Trap).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

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
        """Linear interpolation to n points."""
        xs = np.linspace(0, 110, n)
        return xs, np.interp(xs, pts[:, 0], pts[:, 1])

    x_b, y_b = interpolate(baseline_pts)
    x_c, y_c = interpolate(cnn_pts)

    ax.plot(x_b, y_b, color="#4ade80", linewidth=2.5, label="Baseline (no CNN)")
    ax.plot(x_c, y_c, color="#ef4444", linewidth=2, linestyle="--", label="Vanilla CNN (frozen)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 1.0)
    ax.set_xlim(0, 110)
    ax.legend(loc="upper right")
    ax.set_title("CNN Convergence — The Trap")

    # Annotations
    ax.annotate("CNN ahead",
                xy=(10, 0.070), xytext=(15, 0.12),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5),
                fontsize=9, color="#ef4444", fontweight="bold")
    ax.annotate("Baseline overtakes",
                xy=(30, 0.010), xytext=(40, 0.005),
                arrowprops=dict(arrowstyle="->", color="#4ade80", lw=1.5),
                fontsize=9, color="#4ade80", fontweight="bold")
    ax.annotate("CNN stuck",
                xy=(60, 0.035), xytext=(70, 0.06),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.5),
                fontsize=9, color="#ef4444", fontweight="bold")

    savefig("cnn_convergence.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  2. CNN Variant Ranking
# ══════════════════════════════════════════════════════════════════════════════
def plot_cnn_variant_ranking():
    """
    Horizontal bar chart of 8 CNN variants sorted by val_loss.
    Source: progress_roadmap.html §Chart 16 (chWhatTested).
    """
    labels = [
        "CNN CONCAT + dropout",
        "dropout p=0.2",
        "dropout p=0.5",
        "both",
        "vanilla CNN",
        "trainable",
        "λ+dropout",
        "λ only",
    ]
    values = [0.01131, 0.016, 0.023, 0.024, 0.035, 0.036, 0.040, 0.077]
    baseline = 0.003

    # Colors: green for better than vanilla, red for worse, grey for vanilla
    colors = []
    for v in values:
        if v < 0.035:
            colors.append("#4ade80")
        elif v == 0.035:
            colors.append("#8899aa")
        else:
            colors.append("#ef4444")

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor="black", linewidth=0.5, height=0.7)
    ax.axvline(x=baseline, color="#4ade80", linewidth=2, linestyle="--",
               label=f"Baseline (no CNN) = {baseline}")
    ax.set_xscale("log")
    ax.set_xlabel("Best val_loss (lower = better)")
    ax.set_title("CNN Feature Injection — 8 Variant Ranking")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.001, 0.2)

    # Annotate values on bars
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    savefig("cnn_variant_ranking.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Edge Probe — Balanced Accuracy
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
    values = [0.896, 0.541, 0.873, 0.575, 0.860, 0.557, 0.703, 0.543, 0.500, 0.500]
    errors = [0.044, 0.018, 0.021, 0.050, 0.014, 0.030, 0.056, 0.015, 0.000, 0.000]
    baseline = 0.500

    # Group colors by feature type
    group_colors = {
        "DINOv2": "#4ade80",
        "HOCT 19D": "#14b8a6",
        "Regionprops": "#60a5fa",
        "CNN NT-Xent": "#f59e0b",
        "CNN e2e": "#ef4444",
    }

    colors = []
    for lbl in labels:
        prefix = lbl.split(" ")[0] + (" 19D" if "HOCT" in lbl else "")
        if "DINOv2" in lbl:
            colors.append(group_colors["DINOv2"])
        elif "HOCT" in lbl:
            colors.append(group_colors["HOCT 19D"])
        elif "Regionprops" in lbl:
            colors.append(group_colors["Regionprops"])
        elif "CNN NT-Xent" in lbl:
            colors.append(group_colors["CNN NT-Xent"])
        elif "CNN e2e" in lbl:
            colors.append(group_colors["CNN e2e"])

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, xerr=errors, color=colors, edgecolor="black",
                   linewidth=0.5, height=0.7, capsize=3)
    ax.axvline(x=baseline, color="#ef4444", linewidth=1.5, linestyle="--",
               label="Random baseline = 0.500")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Balanced Accuracy")
    ax.set_title("Edge Prediction — Balanced Accuracy (5-fold CV)")
    ax.set_xlim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=9)

    savefig("edge_probe_bal_acc.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  4. Edge Probe — F1 Score
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
    values = [0.571, 0.138, 0.394, 0.145, 0.396, 0.131, 0.289, 0.129, 0.000, 0.000]
    errors = [0.121, 0.037, 0.052, 0.052, 0.056, 0.029, 0.127, 0.035, 0.000, 0.000]
    baseline = 0.000

    group_colors = {
        "DINOv2": "#4ade80",
        "HOCT 19D": "#14b8a6",
        "Regionprops": "#60a5fa",
        "CNN NT-Xent": "#f59e0b",
        "CNN e2e": "#ef4444",
    }

    colors = []
    for lbl in labels:
        if "DINOv2" in lbl:
            colors.append(group_colors["DINOv2"])
        elif "HOCT" in lbl:
            colors.append(group_colors["HOCT 19D"])
        elif "Regionprops" in lbl:
            colors.append(group_colors["Regionprops"])
        elif "CNN NT-Xent" in lbl:
            colors.append(group_colors["CNN NT-Xent"])
        elif "CNN e2e" in lbl:
            colors.append(group_colors["CNN e2e"])

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, xerr=errors, color=colors, edgecolor="black",
                   linewidth=0.5, height=0.7, capsize=3)
    ax.axvline(x=baseline, color="#ef4444", linewidth=1.5, linestyle="--",
               label="Random baseline = 0.000")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title("Edge Prediction — F1 Score (5-fold CV)")
    ax.set_xlim(-0.05, 0.75)
    ax.legend(loc="lower right", fontsize=9)

    savefig("edge_probe_f1.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  5. Sparse Attention Speed
# ══════════════════════════════════════════════════════════════════════════════
def plot_sparse_attention_speed():
    """
    Line plot: N vs time_ms for different K values.
    Source: bench-transformer-cell-tracking/benchmark_combined/results/speed_mem.csv
    """
    path = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "speed_mem.csv"
    if not path.exists():
        print("  ⚠ speed_mem.csv not found — skipping sparse_attention_speed.pdf")
        return

    df = pd.read_csv(path)
    df["K_label"] = df["K"].map({0: "Dense (K=0)", 4: "K=4", 8: "K=8", 16: "K=16", 32: "K=32"})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.lineplot(data=df, x="N", y="time_ms", hue="K_label",
                 style="K_label", markers=True, dashes=False,
                 palette=VIRIDIS, ax=ax, linewidth=2)

    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Sparse Attention — Forward Speed")
    ax.legend(title="Method", loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    savefig("sparse_attention_speed.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  6. Sparse Attention Memory
# ══════════════════════════════════════════════════════════════════════════════
def plot_sparse_attention_memory():
    """
    Line plot: N vs mem_mb for different K values.
    Source: bench-transformer-cell-tracking/benchmark_combined/results/speed_mem.csv
    """
    path = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "speed_mem.csv"
    if not path.exists():
        print("  ⚠ speed_mem.csv not found — skipping sparse_attention_memory.pdf")
        return

    df = pd.read_csv(path)
    df["K_label"] = df["K"].map({0: "Dense (K=0)", 4: "K=4", 8: "K=8", 16: "K=16", 32: "K=32"})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.lineplot(data=df, x="N", y="mem_mb", hue="K_label",
                 style="K_label", markers=True, dashes=False,
                 palette=VIRIDIS, ax=ax, linewidth=2)

    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Sparse Attention — Peak Memory Usage")
    ax.legend(title="Method", loc="upper left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    savefig("sparse_attention_memory.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  7. Pipeline Breakdown
# ══════════════════════════════════════════════════════════════════════════════
def plot_pipeline_breakdown():
    """
    Stacked bar chart of training step breakdown.
    Also includes regionprops CPU/GPU split.
    Sources: benchmark_pipeline/ CSVs.
    """
    # Training step breakdown from progress_roadmap (doughnut data)
    step_labels = ["Backward pass\n(57%)", "Attention 12L\n(22%)",
                   "blockwise_norm\n(13%)", "FFN 12L\n(5%)", "Optimizer\n(2%)"]
    step_values = [57, 22, 13, 5, 2]

    # Regionprops CPU vs GPU breakdown
    region_labels = ["N=50", "N=200", "N=500", "N=1000"]
    cpu_pct = [74, 52, 33, 21]
    gpu_pct = [26, 48, 67, 79]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: Training step breakdown ──
    colors_step = ["#ef4444", "#60a5fa", "#f59e0b", "#4ade80", "#8899aa"]
    wedges, texts, autotexts = ax1.pie(
        step_values, labels=step_labels, autopct="%1.0f%%",
        colors=colors_step, startangle=90,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax1.set_title("Training Step Breakdown\n(27.4ms, N=200, d=320, 6L+6L)", fontsize=11)

    # ── Right: Regionprops CPU/GPU split ──
    x = np.arange(len(region_labels))
    width = 0.5
    ax2.bar(x, cpu_pct, width, label="CPU (regionprops)", color="#ef4444", edgecolor="black", linewidth=0.5)
    ax2.bar(x, gpu_pct, width, bottom=cpu_pct, label="GPU (model forward)", color="#60a5fa", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(region_labels)
    ax2.set_xlabel("Cells per frame (N)")
    ax2.set_ylabel("% of total inference time")
    ax2.set_title("Regionprops CPU vs GPU\n(Crossover ~N=600)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)

    fig.suptitle("Pipeline Bottleneck Analysis", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig("pipeline_breakdown.pdf")


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

    # Filter to N=512 for a clean comparison across methods
    df_n512 = df[df["N"] == 512].copy()

    # Simplify method labels
    method_map = {
        "mask_knn": "mask_knn",
        "gather_knn": "gather_knn",
        "sparse_softmax": "sparse_softmax",
        "dense_masked": "dense_masked",
        "dense_flash": "dense_flash",
    }
    df_n512["method_short"] = df_n512["method"].map(method_map)

    # Compare K=16
    df_k16 = df_n512[df_n512["K"] == 16].copy()
    df_k16_long = pd.melt(df_k16, id_vars=["method_short"],
                          value_vars=["forward_ms", "backward_ms"],
                          var_name="phase", value_name="time_ms")
    df_k16_long["phase"] = df_k16_long["phase"].map({
        "forward_ms": "Forward", "backward_ms": "Backward"
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_k16_long, x="method_short", y="time_ms", hue="phase",
                palette={"Forward": "#60a5fa", "Backward": "#ef4444"},
                edgecolor="black", linewidth=0.5, ax=ax)

    ax.set_xlabel("Attention Method")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Backward Pass Comparison (N=512, K=16)")
    ax.legend(title="Phase")
    ax.tick_params(axis="x", rotation=25)

    savefig("backward_pass.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  9. KNN TRA Parity (Dense vs Sparse Convergence)
# ══════════════════════════════════════════════════════════════════════════════
def plot_knn_tra_parity():
    """
    Line plot comparing dense vs sparseK4 vs sparseK16 convergence
    (val_loss over epoch) from the ablation_full.csv.
    Source: bench-transformer-cell-tracking/benchmark_combined/results/ablation_full.csv
    """
    path = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "ablation_full.csv"
    if not path.exists():
        print("  ⚠ ablation_full.csv not found — skipping knn_tra_parity.pdf")
        return

    df = pd.read_csv(path)

    # Filter converge phase, L=4 (more interesting), N=512
    conv = df[(df["phase"] == "converge") & (df["L"] == 4) & (df["N"] == 512)].copy()

    # Create nice labels
    conv["label"] = conv["attn"] + "_" + conv["init"]

    # Pick the main configurations
    configs = {
        "dense_rand": "Dense (rand init)",
        "dense_ssl": "Dense (SSL init)",
        "sparseK4_rand": "Sparse K=4 (rand init)",
        "sparseK4_ssl": "Sparse K=4 (SSL init)",
        "sparseK16_rand": "Sparse K=16 (rand init)",
        "sparseK16_ssl": "Sparse K=16 (SSL init)",
    }
    plot_df = conv[conv["label"].isin(configs.keys())].copy()
    plot_df["display"] = plot_df["label"].map(configs)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=plot_df, x="epoch", y="val_loss", hue="display",
                 style="display", markers=True, dashes=False,
                 palette=VIRIDIS, ax=ax, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss (log scale)")
    ax.set_yscale("log")
    ax.set_title("KNN TRA Parity — Dense vs Sparse Convergence\n(L=4, N=512)")
    ax.legend(title="Configuration", fontsize=8, title_fontsize=9)

    savefig("knn_tra_parity.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Generating publication-quality figures...")
    print("=" * 60)

    print("\n[1/9] CNN Convergence Curves")
    plot_cnn_convergence()

    print("\n[2/9] CNN Variant Ranking")
    plot_cnn_variant_ranking()

    print("\n[3/9] Edge Probe — Balanced Accuracy")
    plot_edge_probe_bal_acc()

    print("\n[4/9] Edge Probe — F1 Score")
    plot_edge_probe_f1()

    print("\n[5/9] Sparse Attention Speed")
    plot_sparse_attention_speed()

    print("\n[6/9] Sparse Attention Memory")
    plot_sparse_attention_memory()

    print("\n[7/9] Pipeline Breakdown")
    plot_pipeline_breakdown()

    print("\n[8/9] Backward Pass Comparison")
    plot_backward_pass()

    print("\n[9/9] KNN TRA Parity")
    plot_knn_tra_parity()

    print("\n" + "=" * 60)
    print("  All figures saved to results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
