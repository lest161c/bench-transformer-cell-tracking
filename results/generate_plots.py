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
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import json
from matplotlib.patches import Patch

# ── Single style source (betterplots) ────────────────────────────────────────
# Every figure in this script shares one style: betterplots applies the
# Okabe–Ito palette, the guideline §5/§6 neutrals (text/grid/spine/legend),
# despine, and the legend/grid defaults.  See docs/vis_guidelines.md.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "betterplots"))
from betterplots import palette, set_style
set_style(usetex=False, colors="okabe-ito", despine=True,
          font_size=8.5, label_size=8.5, tick_size=7.5, legend_font_size=7.5)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent  # research-proj/
FIGURES = BASE / "results" / "figures"
PROBE_JSON = BASE / "results" / "probes" / "unified_probe_results_cv10.json"
SIGNIFICANCE_JSON = BASE / "results" / "probes" / "probe_paired_significance.json"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Okabe–Ito palette (see docs/vis_guidelines.md) ────────────────────────────
# Slot assignment kept identical to curated/generate_report.py so the same
# semantic role (e.g. "Dense baseline") gets the same colour in every figure
# of the report.
OI_BLACK     = palette.BLACK
OI_ORANGE    = palette.ORANGE
OI_SKYBLUE   = palette.SKYBLUE
OI_GREEN     = palette.GREEN
OI_YELLOW    = palette.YELLOW
OI_BLUE      = palette.BLUE
OI_VERMILION = palette.VERMILION
OI_PURPLE    = palette.PURPLE

PROBE_DISPLAY = {
    "rp": "Regionprops 7D",
    "rp_fourier": "Regionprops 7D + Fourier PE",
    "hoct2d": "HOCT 2D",
    "hoct2d_fourier": "HOCT 2D + Fourier PE",
    "dino": "DINOv2 (frozen)",
    "cnn_frozen": "CNN NT-Xent (frozen)",
    "cnn_e2e": "CNN end-to-end",
}
# ── Probe family colours (betterplots palette) ────────────────────────────────
# Same semantic assignment as the report body figures, but pulled from the
# shared palette module so probe / CNN families match across scripts.
PROBE_FAMILY_COLORS = {
    key: palette.METHOD_COLORS[key]
    for key in ("rp", "rp_fourier", "hoct2d", "hoct2d_fourier",
                "dino", "cnn_frozen", "cnn_e2e")
}


# Cross-figure role aliases — every figure picks its colours from this table,
# so a "Dense baseline" series is always OI_BLUE, etc.
ROLE_DENSE_BASELINE = palette.COLORS["dense_masked"]
ROLE_DENSE_FLASH    = palette.COLORS["dense_flash"]
ROLE_K4             = palette.K_COLORS[4]
ROLE_K8             = palette.K_COLORS[8]
ROLE_K16            = palette.K_COLORS[16]
ROLE_K32            = palette.K_COLORS[32]
ROLE_K64            = palette.K_COLORS[64]
ROLE_FORWARD        = palette.COLORS["forward"]
ROLE_BACKWARD       = palette.COLORS["backward"]
ROLE_NEUTRAL       = palette.NEUTRAL

# Default colour cycle for any plot that does not set explicit colours.
OI_CYCLE = [palette.BLUE, palette.ORANGE, palette.GREEN, palette.VERMILION,
            palette.PURPLE, palette.SKYBLUE, palette.YELLOW, palette.BLACK]

# Color pairs for bar charts (darker fill, slightly lighter border)
def bar_colors(base_color, alpha=0.75):
    """Return (fill, edge) pair for a base color."""
    from matplotlib.colors import to_rgba, to_hex
    rgba = to_rgba(base_color)
    fill = to_hex((rgba[0], rgba[1], rgba[2], alpha), keep_alpha=True)
    return fill, base_color

# ── Helper ────────────────────────────────────────────────────────────────────
def _short_git_commit():
    """Return ``git rev-parse --short HEAD`` for ``BASE``, or ``"unknown"``.

    Wrapped in try/except so the plot generator never fails on a machine
    without git or without a checkout (CI containers, archives).
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BASE),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        return "unknown"


def _write_meta_sidecar(name, meta):
    """Write ``FIGURES/<stem>.meta.json`` next to the figure.

    The sidecar is the machine-checkable trace of what produced the figure:
    when, by which script, from which commit, which input files were read,
    and which numeric values were plotted. It makes figure-vs-data
    consistency verifiable by reading text, without re-running the pipeline.
    """
    stem = Path(name).stem
    if not isinstance(meta, dict):
        meta = {}
    sidecar = {
        "figure": stem,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "results/generate_plots.py",
        "git_commit": _short_git_commit(),
        "sources": list(meta.get("sources", [])),
        "values": meta.get("values", {}),
    }
    # Pass-through extra keys (e.g. ``n_folds``, ``n_frame_pairs``, ``note``)
    # so data-rich plots can expose their sample size and provenance notes.
    for key, value in meta.items():
        if key in ("sources", "values"):
            continue
        sidecar[key] = value
    sidecar_path = FIGURES / f"{stem}.meta.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str))


def savefig(name, fig=None, dpi=300, meta=None):
    """Save figure to ``results/figures/`` and write a ``<stem>.meta.json`` sidecar.

    The PDF write is unchanged. ``meta`` is an optional dict with keys
    ``"sources"`` (list of input file paths) and ``"values"`` (the essential
    plotted numbers); extra keys are copied into the sidecar. Even with
    ``meta=None`` a minimal sidecar is written so every figure has one.
    """
    if fig is None:
        fig = plt.gcf()
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ {name}  ({size_kb:.1f} KB)")
    plt.close(fig)
    _write_meta_sidecar(name, meta)


def add_horizontal_grid(ax):
    """Ensure only horizontal grid lines are shown."""
    ax.grid(True, axis="y", color="#cccccc", alpha=0.6, linewidth=0.4)
    ax.grid(False, axis="x")


def remove_top_right_spines(ax):
    """Remove top and right spines (already done by rcParams, but belt-and-suspenders)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_legend(ax, loc="best", **kw):
    """Apply consistent legend styling for the figures that share ``set_style``.

    Per ``docs/vis_guidelines.md`` §6 the legend frame is optional and
    never filled.  ``frameon`` and ``framealpha`` are left at the
    betterplots defaults (False / n/a) so the guideline-compliant look
    comes from the single shared ``set_style`` call rather than from
    this helper.
    """
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

    # Meta: anchor points ARE the data; interpolated curves are derived. Record
    # the anchors so figure-vs-data checks compare the same numbers.
    meta = {
        "sources": [],
        "note": "anchor points hardcoded in generator (progress_roadmap.html Chart 17)",
        "values": {
            "baseline_pts": baseline_pts.tolist(),
            "cnn_pts": cnn_pts.tolist(),
        },
    }
    savefig("cnn_convergence.pdf", fig, meta=meta)


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
    ax.grid(True, axis="x", color="#cccccc", alpha=0.3, linewidth=0.3)
    style_legend(ax, loc="lower right", fontsize=6.5)

    # Annotate values on bars
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=6.5)

    meta = {
        "sources": [],
        "note": "val_loss values hardcoded in generator (progress_roadmap.html Chart 16)",
        "values": {
            "labels": labels,
            "val_losses": values,
            "baseline": baseline,
        },
    }
    savefig("cnn_variant_ranking.pdf", fig, meta=meta)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Edge Probe — Balanced Accuracy  (Chart 19)
# ══════════════════════════════════════════════════════════════════════════════
def plot_edge_probe_bal_acc():
    """Grouped horizontal bar chart of edge-probe balanced accuracy.

    Source: results/probes/unified_probe_results_cv10.json (10-fold CV,
    seed 42; SPEC 0006) PLUS the 3-seed reruns (cv10_s43.json, cv10_s44.json)
    when present: bars then show the 3-seed mean of the per-seed CV means,
    error bars = std across seeds (gives the comparison a seed error bar).
    """
    seed_paths = [
        PROBE_JSON,
        PROBE_JSON.parent / "unified_probe_results_cv10_s43.json",
        PROBE_JSON.parent / "unified_probe_results_cv10_s44.json",
    ]
    seed_results = []
    for path in seed_paths:
        if path.exists():
            seed_results.append(json.loads(path.read_text())["results"])
    if not seed_results:
        raise RuntimeError("no probe result JSONs found")

    features = sorted(
        {key for results in seed_results for key in results},
        key=lambda feature: -np.mean(
            [results[feature]["mlp"]["bal_acc_mean"]
             for results in seed_results if feature in results]),
    )

    probe = {}
    for feature in features:
        probe[feature] = {}
        for probe_type in ("mlp", "linear"):
            seed_means = [
                results[feature][probe_type]["bal_acc_mean"]
                for results in seed_results
                if feature in results and probe_type in results[feature]
            ]
            if not seed_means:
                continue
            probe[feature][probe_type] = {
                "bal_acc_mean": float(np.mean(seed_means)),
                "bal_acc_std": float(np.std(seed_means)),
            }

    entries = []
    for feature in sorted(probe, key=lambda f: -probe[f]["mlp"]["bal_acc_mean"]):
        for probe_type, suffix in (("mlp", "MLP"), ("linear", "linear")):
            stats = probe[feature].get(probe_type)
            if not isinstance(stats, dict):
                continue
            entries.append((f"{PROBE_DISPLAY[feature]} {suffix}",
                            float(stats["bal_acc_mean"]), float(stats["bal_acc_std"]),
                            PROBE_FAMILY_COLORS[feature]))

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    errors = [e[2] for e in entries]
    colors = [e[3] for e in entries]
    baseline = 0.500

    fig, ax = plt.subplots(figsize=figsize_wide(1.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, xerr=errors, color=colors,
            edgecolor="white", linewidth=0.3, height=0.65, capsize=2,
            error_kw={"linewidth": 0.6, "ecolor": "#999999"})
    ax.axvline(x=baseline, color="#999999", linewidth=0.8, linestyle="--",
               label="Random baseline = 0.500")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Balanced Accuracy")
    ax.set_xlim(0.45, 1.0)
    add_horizontal_grid(ax)
    style_legend(ax, loc="lower right", fontsize=6.5)

    meta = {
        "sources": [str(PROBE_JSON)],
        "n_folds": 10,
        "values": [
            {"label": label, "mean": mean, "std": std, "n_folds": 10}
            for label, mean, std, _ in entries
        ],
    }
    savefig("edge_probe_bal_acc.pdf", fig, meta=meta)
    svg_path = FIGURES / "edge_probe_bal_acc.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)


# ══════════════════════════════════════════════════════════════════════════════
#  4. Edge Probe — F1 Score  (Chart 20)
# ══════════════════════════════════════════════════════════════════════════════
def plot_edge_probe_f1():
    """Grouped horizontal bar chart of edge-probe F1 (canonical threshold 0.5).

    Source: results/probes/unified_probe_results_cv10.json (10-fold CV, seed
    42). F1 is computed on the natural (heavily imbalanced) candidate
    distribution, so it is reported per feature but the balanced-accuracy
    chart carries the headline signal.
    """
    probe = json.loads(PROBE_JSON.read_text())["results"]
    entries = []
    for feature in sorted(probe, key=lambda f: -probe[f]["mlp"]["f1_mean"]):
        for probe_type, suffix in (("mlp", "MLP"), ("linear", "linear")):
            stats = probe[feature].get(probe_type)
            if not isinstance(stats, dict):
                continue
            entries.append((f"{PROBE_DISPLAY[feature]} {suffix}",
                            float(stats["f1_mean"]), float(stats["f1_std"]),
                            PROBE_FAMILY_COLORS[feature]))

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    errors = [e[2] for e in entries]
    colors = [e[3] for e in entries]

    fig, ax = plt.subplots(figsize=figsize_wide(1.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, xerr=errors, color=colors,
            edgecolor="white", linewidth=0.3, height=0.65, capsize=2,
            error_kw={"linewidth": 0.6, "ecolor": "#999999"})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("F1 (threshold 0.5)")
    add_horizontal_grid(ax)

    meta = {
        "sources": [str(PROBE_JSON)],
        "n_folds": 10,
        "values": [
            {"label": label, "mean": mean, "std": std, "n_folds": 10}
            for label, mean, std, _ in entries
        ],
    }
    savefig("edge_probe_f1.pdf", fig, meta=meta)
    svg_path = FIGURES / "edge_probe_f1.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)


# ══════════════════════════════════════════════════════════════════════════════
#  5. Sparse Attention — Forward Speed
# ══════════════════════════════════════════════════════════════════════════════
def plot_probe_significance():
    """Forest chart of per-frame-pair paired deltas vs the 7D reference.

    Source: results/probes/probe_paired_significance.json (SPEC 0006): paired
    t-test over the 29 matched frame pairs, bootstrap 95% CI, Monte-Carlo
    sign-flip permutation, Holm adjustment across feature sets. Communicates
    the significance result the report states in text: position-augmented 7D
    and DINOv2 improve significantly over plain 7D, plain HOCT 2D does not.
    """
    sig = json.loads(SIGNIFICANCE_JSON.read_text())["comparisons_pairs"]
    fig, axes = plt.subplots(2, 1, figsize=figsize_wide(1.3), sharex=True)
    for ax, metric, title in zip(
            axes, ("bal_acc", "f1"),
            ("Balanced accuracy", "F1 (threshold 0.5)")):
        rows = []
        for key, values in sig.items():
            feature, metric_key = key.split("|")
            if metric_key != metric:
                continue
            rows.append((PROBE_DISPLAY[feature], values["mean_difference"],
                         values["ci_low"], values["ci_high"], values["holm_p"]))
        rows.sort(key=lambda r: r[1])
        y_pos = np.arange(len(rows))
        for i, (label, mean, lo, hi, holm) in enumerate(rows):
            significant = holm < 0.05
            ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]],
                        fmt="o", ms=5, capsize=3, linewidth=0.8,
                        color=OI_BLUE, markerfacecolor=OI_BLUE if significant else "white")
            ax.annotate(f"Holm p = {holm:.1e}", (hi, i), fontsize=6,
                        va="center", xytext=(4, 0), textcoords="offset points")
        ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax.set_title(title, fontsize=8)
        add_horizontal_grid(ax)
    axes[1].set_xlabel("Paired delta vs Regionprops 7D (29 matched frame pairs)")

    meta = {
        "sources": [str(SIGNIFICANCE_JSON)],
        "n_frame_pairs": 29,
        "values": [
            {
                "feature": feature,
                "metric": metric_key,
                "mean_difference": values["mean_difference"],
                "ci_low": values["ci_low"],
                "ci_high": values["ci_high"],
                "holm_p": values["holm_p"],
                "n_frame_pairs": values.get("unit_count", 29),
            }
            for key, values in sig.items()
            for feature, metric_key in [key.split("|")]
        ],
    }
    savefig("probe_significance.pdf", fig, meta=meta)
    svg_path = FIGURES / "probe_significance.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)


def plot_training_step_k_sweep():
    """
    Three-panel figure, one aspect of the sparse-attention K-sweep benchmark
    per panel (owner decision 2026-09-28: the single 6-curve line plot was
    unreadable because K ∈ {4,8,16} differ by ≤15% and braid into one band).
    Source: benchmark_attn/results/speed_mem.csv, produced by
    benchmark_training/scripts/benchmarks/benchmark_speed_mem.py — the FULL
    training step (forward + backward + AdamW, fp32) on the A500, plus peak
    GPU memory; K ∈ {0,4,8,16,32,64}, N ∈ {128,256,512}. NOT the attention
    forward time alone (see fig:speed_vs_n for the H100 op-level benchmark).
    Fallback: bench-transformer-cell-tracking/benchmark_combined/results/speed_mem.csv (no K=64).

    Panels:
      (a) Scaling with N: absolute training-step time vs N. The K ∈ {4,8,16}
          cluster is drawn as one gray min–max band; Dense, K=32, K=64 as
          lines.
      (b) Overhead vs dense: training-step-time ratio K/dense vs N; 1.0 = dense.
      (c) Cost vs K: training-step time at the largest measured N as bars for
          all six K values (the N is stated in the LaTeX caption, not hardcoded).

    Figure stem: ``training_step_k_sweep`` — named for what the data is (the
    full training step), not for the mislabeled "attention speed".
    One shared figure legend (per-panel end labels collided with neighbouring
    panels at the 3-panel width).
    """
    # Try updated local copy first, then the older benchmark_combined path
    path_local = BASE / "benchmark_attn" / "results" / "speed_mem.csv"
    path_combined = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "speed_mem.csv"

    path = path_local if path_local.exists() else path_combined
    if not path.exists():
        print("  ⚠ speed_mem.csv not found — skipping training_step_k_sweep.pdf")
        return

    df = pd.read_csv(path)
    sequence_lengths = sorted(df["N"].unique())
    largest_n = sequence_lengths[-1]
    dense_time_by_n = df[df["K"] == 0].set_index("N")["time_ms"]
    df["time_over_dense"] = df.apply(
        lambda row: row["time_ms"] / dense_time_by_n[row["N"]], axis=1)
    cluster_k_values = [4, 8, 16]  # cluster spread ≤15%; shown as one band in (a)/(b)

    fig, (ax_scaling, ax_overhead, ax_cost_k) = plt.subplots(
        1, 3, figsize=(4.8, 2.0))

    # ── Shared encodings ──────────────────────────────────────────────────────
    dense_color = ROLE_DENSE_BASELINE
    cluster_color = "#999999"
    high_k_colors = {"K=32": ROLE_K32, "K=64": ROLE_K64}
    line_widths = {"Dense": 1.6, "K=32": 1.0, "K=64": 1.0}

    # K ∈ {4,8,16} spread ≤15% at every N (vs ~2–4× gaps to Dense/K=64);
    # drawn as one min–max band.
    # Bounds are needed in both panels' units: absolute time (a) and the
    # ratio to dense (b).
    cluster_time_by_n = df[df["K"].isin(cluster_k_values)].groupby("N")["time_ms"]
    cluster_ratio_by_n = df[df["K"].isin(cluster_k_values)].groupby("N")["time_over_dense"]
    cluster_time_lo = cluster_time_by_n.min()[sequence_lengths]
    cluster_time_hi = cluster_time_by_n.max()[sequence_lengths]
    cluster_ratio_lo = cluster_ratio_by_n.min()[sequence_lengths]
    cluster_ratio_hi = cluster_ratio_by_n.max()[sequence_lengths]

    def cluster_color_for_k(k_value):
        """Map a K value to this figure's shared colour semantics."""
        if k_value == 0:
            return dense_color
        if k_value in cluster_k_values:
            return cluster_color
        return high_k_colors[f"K={k_value}"]

    def plot_lines_with_cluster_band(ax, y_by_label, cluster_lo, cluster_hi):
        """
        Draw the Dense/K=32/K=64 lines plus the K∈{4,8,16} min–max band.

        y_by_label maps "Dense"/"K=32"/"K=64" to a list of y values ordered by
        increasing N; ``cluster_lo``/``cluster_hi`` are the band bounds in the
        same units as the y values. Returns nothing; draws onto ``ax``.
        """
        ax.fill_between(sequence_lengths, cluster_lo, cluster_hi,
                        color=cluster_color, alpha=0.35, linewidth=0, zorder=1)
        for line_label, line_color in {"Dense": dense_color, **high_k_colors}.items():
            ax.plot(sequence_lengths, y_by_label[line_label],
                    color=line_color, linewidth=line_widths[line_label],
                    marker="o", markersize=2.8, zorder=3)

    # ── (a) Scaling with N ────────────────────────────────────────────────────
    time_by_label = {
        line_label: df[df["K"] == k_value].sort_values("N")["time_ms"].tolist()
        for line_label, k_value in [("Dense", 0), ("K=32", 32), ("K=64", 64)]
    }
    plot_lines_with_cluster_band(ax_scaling, time_by_label, cluster_time_lo, cluster_time_hi)
    ax_scaling.set_xlabel("Sequence length $N$")
    ax_scaling.set_ylabel("Training-step time (ms)")
    ax_scaling.set_title("(a) Scaling with $N$", fontsize=7)

    # ── (b) Overhead vs dense ─────────────────────────────────────────────────
    ratio_by_label = {
        line_label: df[df["K"] == k_value].sort_values("N")["time_over_dense"].tolist()
        for line_label, k_value in [("Dense", 0), ("K=32", 32), ("K=64", 64)]
    }
    plot_lines_with_cluster_band(ax_overhead, ratio_by_label, cluster_ratio_lo, cluster_ratio_hi)
    ax_overhead.set_xlabel("Sequence length $N$")
    ax_overhead.set_ylabel("Time / dense (×)")
    ax_overhead.set_title("(b) Overhead vs dense", fontsize=7)

    # ── (c) Cost vs K at the largest measured N ───────────────────────────────
    cost_at_largest_n = df[df["N"] == largest_n].sort_values("K")
    bar_positions = np.arange(len(cost_at_largest_n))
    ax_cost_k.bar(bar_positions, cost_at_largest_n["time_ms"],
                  color=[cluster_color_for_k(k) for k in cost_at_largest_n["K"]],
                  edgecolor="white", linewidth=0.3, width=0.65)
    # Bar value annotations. Adjacent bars with near-equal heights (e.g. the
    # K=4/K=8 pair at 123.6/127.7 ms) would have their labels touch, so a
    # label is raised by ``bar_label_gap_ms`` whenever its bar top lies within
    # ``bar_label_gap_ms`` of the previous bar's top — deterministic and
    # data-driven.
    bar_label_gap_ms = 10
    previous_bar_time = None
    for bar_position, bar_time in zip(bar_positions, cost_at_largest_n["time_ms"]):
        label_y = bar_time + 3
        if previous_bar_time is not None and bar_time - previous_bar_time < bar_label_gap_ms:
            label_y += bar_label_gap_ms
        ax_cost_k.text(bar_position, label_y, f"{bar_time:.0f}",
                       ha="center", va="bottom", fontsize=5.5)
        previous_bar_time = bar_time
    ax_cost_k.set_xticks(bar_positions)
    ax_cost_k.set_xticklabels([str(k) for k in cost_at_largest_n["K"]])
    ax_cost_k.set_xlabel("Context size $K$")
    ax_cost_k.set_ylabel("Training-step time (ms)")
    ax_cost_k.set_ylim(0, 230)  # headroom for the bar value annotations
    ax_cost_k.set_title("(c) Cost vs $K$", fontsize=7)

    # Axes limits, grid, spines, and the single shared legend.
    for ax in (ax_scaling, ax_overhead):
        ax.set_xscale("log", base=2)
        ax.set_xticks(sequence_lengths)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlim(100, 700)
    for ax in (ax_scaling, ax_overhead, ax_cost_k):
        add_horizontal_grid(ax)
        remove_top_right_spines(ax)

    legend_handles = [
        Line2D([0], [0], color=dense_color, linewidth=1.6, marker="o",
               markersize=3, label="Dense (K=0)"),
        Patch(facecolor=cluster_color, alpha=0.35, label="K=4–16"),
        Line2D([0], [0], color=ROLE_K32, linewidth=1.0, marker="o",
               markersize=3, label="K=32"),
        Line2D([0], [0], color=ROLE_K64, linewidth=1.0, marker="o",
               markersize=3, label="K=64"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               frameon=False, fontsize=6, columnspacing=1.2, handlelength=1.6)
    fig.subplots_adjust(top=0.82, wspace=0.55)

    meta = {
        "sources": [str(path)],
        "values": [
            {"K": int(row["K"]), "N": int(row["N"]), "time_ms": float(row["time_ms"])}
            for _, row in df.iterrows()
        ],
    }
    savefig("training_step_k_sweep.pdf", fig, meta=meta)


def plot_training_step_k_sweep_summary():
    """
    Single-panel body summary of the K-sweep training-step benchmark:
    training-step time ratio to dense vs sequence length N. The low-context
    cluster K ∈ {4,8,16} is drawn as one gray min–max band (the variants agree
    within ~15%), K=32 and K=64 as lines, dashed line at 1.0 = dense.
    Compact companion to the three-panel ``training_step_k_sweep`` appendix
    figure: the body panel carries the one headline (sparse-attention
    overhead at training scale, largest at N=128 and shrinking as N grows),
    per-configuration detail lives in the appendix.

    Source: benchmark_attn/results/speed_mem.csv — the FULL training step
    (forward + backward + AdamW, fp32) on the A500; NOT the attention
    forward time alone. Fallback: bench-transformer-cell-tracking path.
    """
    path_local = BASE / "benchmark_attn" / "results" / "speed_mem.csv"
    path_combined = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "speed_mem.csv"

    path = path_local if path_local.exists() else path_combined
    if not path.exists():
        print("  ⚠ speed_mem.csv not found — skipping training_step_k_sweep_summary.pdf")
        return

    df = pd.read_csv(path)
    sequence_lengths = sorted(df["N"].unique())
    largest_n = sequence_lengths[-1]
    dense_time_by_n = df[df["K"] == 0].set_index("N")["time_ms"]
    df["time_over_dense"] = df.apply(
        lambda row: row["time_ms"] / dense_time_by_n[row["N"]], axis=1)
    cluster_k_values = [4, 8, 16]  # cluster spread ≤15%; shown as one band

    fig, ax = plt.subplots(figsize=figsize_wide(2.2))

    cluster_ratio_by_n = df[df["K"].isin(cluster_k_values)].groupby("N")["time_over_dense"]
    cluster_lo = cluster_ratio_by_n.min()[sequence_lengths]
    cluster_hi = cluster_ratio_by_n.max()[sequence_lengths]
    ax.fill_between(sequence_lengths, cluster_lo, cluster_hi,
                    color="#999999", alpha=0.35, linewidth=0, zorder=1,
                    label="K=4–16")
    for k_value, k_color in [(32, ROLE_K32), (64, ROLE_K64)]:
        variant_rows = df[df["K"] == k_value].sort_values("N")
        ax.plot(sequence_lengths, variant_rows["time_over_dense"],
                color=k_color, linewidth=1.0, marker="o", markersize=2.8,
                zorder=3, label=f"K={k_value}")
    ax.axhline(1.0, color=ROLE_DENSE_BASELINE, linewidth=1.2, linestyle="--",
               alpha=0.7, zorder=2, label="Dense (K=0)")

    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Training-step time / dense (×)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sequence_lengths)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(100, 700)
    ax.set_ylim(0.85, 1.1 * df["time_over_dense"].max())
    add_horizontal_grid(ax)
    remove_top_right_spines(ax)
    style_legend(ax, loc="upper right", fontsize=6.5)

    meta = {
        "sources": [str(path)],
        "values": [
            {"K": int(row["K"]), "N": int(row["N"]),
             "time_over_dense": float(row["time_over_dense"])}
            for _, row in df.iterrows()
        ],
    }
    savefig("training_step_k_sweep_summary.pdf", fig, meta=meta)


# ══════════════════════════════════════════════════════════════════════════════
#  6. K-Sweep Training Benchmark — Peak Memory
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_step_k_sweep_memory():
    """
    Single-panel figure: peak GPU memory vs N for the K-sweep training-step
    benchmark (owner decision 2026-09-28, SPEC 0015: the former second panel
    plotted the mask-KNN mask buffer, whose measured sizes equal the
    analytic 8·N²·4 bytes exactly — a deterministic identity, not an
    experiment — so the panel was redundant with its own caption and with
    the body figure caption ``fig:memory_vs_n``; the mask-buffer fact now
    lives in this figure's caption and in the appendix text).
    Source: benchmark_attn/results/speed_mem.csv (updated 2026-08-04,
    K ∈ {0,4,8,16,32,64}, N ∈ {128,256,512}) — dense baseline and
    gather-KNN sparse variants (O(NK) memory). The memory numbers are peak
    GPU memory of the FULL training step (forward + backward + AdamW,
    fp32, A500), same benchmark as ``training_step_k_sweep``.
    Fallback: bench-transformer-cell-tracking/benchmark_combined/results/speed_mem.csv (no K=64).

    Encoding: dense, K=32, K=64 as lines; the K ∈ {4,8,16} cluster as one
    gray min–max band (spread ≤28% in memory).

    Figure stem: ``training_step_k_sweep_memory`` — named for what the data
    is (full training-step memory), not "attention memory".
    """
    # Try updated local copy first, then the older benchmark_combined path
    path_local = BASE / "benchmark_attn" / "results" / "speed_mem.csv"
    path_combined = BASE / "bench-transformer-cell-tracking" / "benchmark_combined" / "results" / "speed_mem.csv"

    path = path_local if path_local.exists() else path_combined
    if not path.exists():
        print("  ⚠ speed_mem.csv not found — skipping training_step_k_sweep_memory.pdf")
        return

    df = pd.read_csv(path)
    sequence_lengths = sorted(df["N"].unique())
    cluster_k_values = [4, 8, 16]  # cluster spread ≤28% in memory; shown as one band

    fig, ax = plt.subplots(figsize=figsize_wide(2.2))

    # ── Encodings (same vocabulary as plot_training_step_k_sweep) ────────────
    dense_color = ROLE_DENSE_BASELINE
    cluster_color = "#999999"
    high_k_colors = {"K=32": ROLE_K32, "K=64": ROLE_K64}
    line_widths = {"Dense": 1.6, "K=32": 1.0, "K=64": 1.0}

    # K ∈ {4,8,16} spread ≤28% at every N (vs ~2× gaps to Dense/K=64); drawn
    # as one min–max band.
    cluster_mem_by_n = df[df["K"].isin(cluster_k_values)].groupby("N")["mem_mb"]
    cluster_mem_lo = cluster_mem_by_n.min()[sequence_lengths]
    cluster_mem_hi = cluster_mem_by_n.max()[sequence_lengths]
    ax.fill_between(sequence_lengths, cluster_mem_lo, cluster_mem_hi,
                    color=cluster_color, alpha=0.35, linewidth=0, zorder=1)
    for line_label, k_value in [("Dense", 0), ("K=32", 32), ("K=64", 64)]:
        variant_rows = df[df["K"] == k_value].sort_values("N")
        ax.plot(sequence_lengths, variant_rows["mem_mb"],
                color=dense_color if k_value == 0
                else high_k_colors[line_label],
                linewidth=line_widths[line_label],
                marker="o", markersize=2.8, zorder=3)

    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sequence_lengths)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(100, 700)
    ax.set_ylim(0, 1750)  # 0-baseline so relative gaps are not exaggerated
    add_horizontal_grid(ax)
    remove_top_right_spines(ax)
    legend_handles = [
        Line2D([0], [0], color=dense_color, linewidth=1.6, marker="o",
               markersize=3, label="Dense (K=0)"),
        Patch(facecolor=cluster_color, alpha=0.35, label="K=4–16"),
        Line2D([0], [0], color=ROLE_K32, linewidth=1.0, marker="o",
               markersize=3, label="K=32"),
        Line2D([0], [0], color=ROLE_K64, linewidth=1.0, marker="o",
               markersize=3, label="K=64"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              fontsize=6.5, handlelength=1.6)

    # Meta: record the speed_mem.csv series (gather/dense lines).
    meta = {
        "sources": [str(path)],
        "values": [
            {"K": int(row["K"]), "N": int(row["N"]), "mem_mb": float(row["mem_mb"])}
            for _, row in df.iterrows()
        ],
    }
    savefig("training_step_k_sweep_memory.pdf", fig, meta=meta)
    # SVG sibling for editing in report/figs/ (PDF remains the build asset).
    svg_path = FIGURES / "training_step_k_sweep_memory.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)
    size_kb = os.path.getsize(svg_path) / 1024
    print(f"  ✓ training_step_k_sweep_memory.svg  ({size_kb:.1f} KB)")


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
    style_legend(ax2, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)

    fig.tight_layout(pad=0.8)
    meta = {
        "sources": [],
        "note": "hardcoded in generator",
        "values": {
            "step_breakdown": dict(zip(step_labels, step_values)),
            "regionprops_split": [
                {"N": label, "cpu_pct": cpu, "gpu_pct": gpu}
                for label, cpu, gpu in zip(region_labels, cpu_pct, gpu_pct)
            ],
        },
    }
    savefig("pipeline_breakdown.pdf", fig, meta=meta)


# ══════════════════════════════════════════════════════════════════════════════
#  8. Backward Pass Comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_backward_pass():
    """
    Bar chart comparing forward+backward time for different attention methods.
    Source: benchmark_attn/results/sparse_backward_results.csv
    (legacy path benchmark_backward/ is kept as a fallback for older runs).
    """
    path_new = BASE / "benchmark_attn" / "results" / "sparse_backward_results.csv"
    path_old = BASE / "benchmark_backward" / "sparse_backward_results.csv"
    path = path_new if path_new.exists() else path_old
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

    # Meta: record only the N=512/K=16 rows that actually get plotted (the
    # figure filters to those; the CSV has N ∈ {128,256,512} × K ∈ {4,16,32,64}).
    meta = {
        "sources": [str(path)],
        "values": [
            {
                "method": row["method_short"],
                "phase": row["phase"],
                "time_ms": float(row["time_ms"]),
            }
            for _, row in df_k16_long.iterrows()
        ],
    }
    savefig("backward_pass.pdf", fig, meta=meta)


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
                va="center", ha="left", fontsize=6, color="#999999",
                fontstyle="italic")

    # Main annotation: all within ±0.001 (top-right, above highest bar)
    ax.annotate("All KNN variants within $\\pm0.001$ TRA of dense",
                xy=(0.991, len(labels) - 0.15), fontsize=7.5,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#cccccc",
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

    meta = {
        "sources": [str(results_dir / fname) for _, fname in configs],
        "values": [
            {"config": label, "mean_tra": mean, "std_tra": std}
            for label, mean, std in zip(labels, means, stds)
        ],
    }
    savefig("knn_tra_parity.pdf", fig, meta=meta)


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
        ("5/9", "K-Sweep Training Step (3-panel)", plot_training_step_k_sweep),
        ("5a/9", "K-Sweep Training Step (body summary)", plot_training_step_k_sweep_summary),
        ("5b/9", "Probe Significance (forest)", plot_probe_significance),
        ("6/9", "K-Sweep Training Memory", plot_training_step_k_sweep_memory),
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
