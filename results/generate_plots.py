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

# ── Cutoff masked-fraction statistic (G-01) ─────────────────────────────────
# The analysis script is self-contained: it computes the per-frame CSV only
# when it is missing (or reuses it) and renders the PNG from the CSV alone.
# Compute requires an interpreter with numpy/tifffile/scikit-image.
MASKED_FRACTION_SCRIPT = BASE / "benchmark_training" / "scripts" / "analysis" / "cutoff_masked_fraction.py"
MASKED_FRACTION_CSV = BASE / "benchmark_training" / "results" / "cutoff_masked_fraction" / "masked_fraction.csv"


def run_masked_fraction():
    """Execute the cutoff masked-fraction script when its CSV is not present.

    If the intermediate CSV already exists, computation is skipped (the
    script's ``--from-csv`` mode can regenerate the PNG from the CSV alone at
    any time).
    """
    if MASKED_FRACTION_CSV.exists():
        print("  CSV present; skipping masked-fraction computation")
        return
    subprocess.run(
        [sys.executable, str(MASKED_FRACTION_SCRIPT),
         "--csv", str(MASKED_FRACTION_CSV),
         "--png", str(FIGURES / "masked_fraction.png")],
        check=True,
    )

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
    Three-panel figure, one aspect of the multi-seed sparse-attention K-sweep
    benchmark per panel. Source: benchmark_attn/results/speed_mem_v2.csv —
    the FULL training step (forward + backward + AdamW, fp32, B=2) on the
    A500, multi-seed protocol (5 seeds x 10 timed iterations per cell;
    mean +/- sd across seed runs; SPEC 0019 amended 2026-10-01 — no
    repeat-block/machine-state protocol). K in {0,2,4,8,16,32,64},
    N in {128,256,512}; ratio reference = dense (K=0). NOT the attention
    forward time alone (see fig:speed_vs_n for the H100 op-level benchmark).

    Panels:
      (a) Scaling with N: absolute training-step time vs N (seed-mean lines
          with +/-sd bands). The gather-KNN K in {2,4,8,16} cluster is drawn
          as one gray min-max band over the three seed-means; dense, K=32,
          K=64 and the two cached-dist configurations (K=16) as lines.
      (b) Overhead vs dense: training-step-time ratio K/dense vs N; 1.0 =
          dense. Ratio sd propagates the numerator and dense sd in quadrature.
      (c) Cost vs K: training-step time (seed-mean, sd whiskers) at the
          largest measured N as bars for all cells; at K=16 three bars
          (gather, cached-dist mask, cached-dist gather).

    Figure stem: ``training_step_k_sweep``. One shared figure legend.
    """
    path = BASE / "benchmark_attn" / "results" / "speed_mem_v2.csv"
    if not path.exists():
        print("  ⚠ speed_mem_v2.csv not found — skipping training_step_k_sweep.pdf")
        return

    df_raw = pd.read_csv(path)
    df_raw = df_raw[df_raw["status"] == "ok"]
    per_seed = (df_raw.groupby(["variant", "K", "N", "repeat_id"])["time_ms"]
                .mean().reset_index())
    cell = (per_seed.groupby(["variant", "K", "N"])["time_ms"]
            .agg(["mean", "std"]).reset_index())
    cell.columns = ["variant", "K", "N", "time_ms", "time_ms_sd"]
    df = cell  # one row per (variant, K, N): seed-mean and sd across seeds

    sequence_lengths = sorted(df["N"].unique())
    largest_n = sequence_lengths[-1]
    dense_time_by_n = (df[df["variant"] == "dense"]
                       .set_index("N")["time_ms"])
    dense_sd_by_n = (df[df["variant"] == "dense"]
                     .set_index("N")["time_ms_sd"])
    df["time_over_dense"] = df.apply(
        lambda row: row["time_ms"] / dense_time_by_n[row["N"]], axis=1)
    df["ratio_sd"] = df.apply(
        lambda row: row["time_over_dense"] * np.sqrt(
            (row["time_ms_sd"] / row["time_ms"]) ** 2
            + (dense_sd_by_n[row["N"]] / dense_time_by_n[row["N"]]) ** 2), axis=1)
    cluster_k_values = [2, 4, 8, 16]  # cluster spread; shown as one band in (a)/(b)

    fig, (ax_scaling, ax_overhead, ax_cost_k) = plt.subplots(
        1, 3, figsize=(4.8, 2.0))

    # ── Shared encodings ──────────────────────────────────────────────────────
    dense_color = ROLE_DENSE_BASELINE
    cluster_color = "#999999"
    high_k_colors = {"K=32": ROLE_K32, "K=64": ROLE_K64}
    cached_colors = {"cached_dist_mask_knn": OI_VERMILION,
                     "cached_dist_gather_knn": OI_PURPLE}
    cached_labels = {"cached_dist_mask_knn": "cached-dist mask (K=16)",
                     "cached_dist_gather_knn": "cached-dist gather (K=16)"}
    line_widths = {"Dense": 1.6, "K=32": 1.0, "K=64": 1.0,
                   "cached": 1.0}

    cluster_df = df[(df["variant"] == "gather_knn")
                    & (df["K"].isin(cluster_k_values))]
    cluster_time_by_n = cluster_df.groupby("N")["time_ms"]
    cluster_ratio_by_n = cluster_df.groupby("N")["time_over_dense"]
    cluster_time_lo = cluster_time_by_n.min()[sequence_lengths]
    cluster_time_hi = cluster_time_by_n.max()[sequence_lengths]
    cluster_ratio_lo = cluster_ratio_by_n.min()[sequence_lengths]
    cluster_ratio_hi = cluster_ratio_by_n.max()[sequence_lengths]

    def series_by_variant(variant, column):
        """Ordered values of ``column`` for one variant, increasing N."""
        return df[df["variant"] == variant].sort_values("N")[column].tolist()

    def cluster_color_for_k(k_value):
        """Map a K value to this figure's shared colour semantics."""
        if k_value == 0:
            return dense_color
        if k_value in cluster_k_values:
            return cluster_color
        return high_k_colors[f"K={k_value}"]

    def plot_lines_with_cluster_band(ax, line_by_label, sd_by_label,
                                     y_col, sd_col, cluster_lo, cluster_hi):
        """
        Draw the Dense/K=32/K=64 lines (+/-sd bands) plus the K∈{2,4,8,16}
        min–max band; then the two cached-dist mean lines (+/-sd bands).

        line_by_label maps "Dense"/"K=32"/"K=64" and sd_by_label the matching
        sd lists, ordered by increasing N. Band bounds in the same units.
        """
        ax.fill_between(sequence_lengths, cluster_lo, cluster_hi,
                        color=cluster_color, alpha=0.35, linewidth=0, zorder=1)
        for line_label, line_color in {"Dense": dense_color, **high_k_colors}.items():
            ys = line_by_label[line_label]
            sds = sd_by_label[line_label]
            line, = ax.plot(sequence_lengths, ys,
                            color=line_color, linewidth=line_widths[line_label],
                            marker="o", markersize=2.8, zorder=3)
            ax.fill_between(sequence_lengths, [y - s for y, s in zip(ys, sds)],
                            [y + s for y, s in zip(ys, sds)],
                            color=line_color, alpha=0.18, linewidth=0, zorder=2)
        for variant, color in cached_colors.items():
            ys = series_by_variant(variant, y_col)
            sds = series_by_variant(variant, sd_col)
            ax.plot(sequence_lengths, ys, color=cached_colors[variant],
                    linewidth=1.0, marker="o", markersize=2.8, zorder=3)
            ax.fill_between(sequence_lengths, [y - s for y, s in zip(ys, sds)],
                            [y + s for y, s in zip(ys, sds)],
                            color=cached_colors[variant], alpha=0.18,
                            linewidth=0, zorder=2)

    # ── (a) Scaling with N ────────────────────────────────────────────────────
    time_by_label = {
        line_label: df[(df["variant"] == "gather_knn") & (df["K"] == k_value)]
        .sort_values("N")["time_ms"].tolist()
        for line_label, k_value in [("Dense", 0), ("K=32", 32), ("K=64", 64)]
    }
    time_by_label["Dense"] = series_by_variant("dense", "time_ms")
    sd_by_label = {
        line_label: df[(df["variant"] == "gather_knn") & (df["K"] == k_value)]
        .sort_values("N")["time_ms_sd"].tolist()
        for line_label, k_value in [("Dense", 0), ("K=32", 32), ("K=64", 64)]
    }
    sd_by_label["Dense"] = series_by_variant("dense", "time_ms_sd")
    plot_lines_with_cluster_band(ax_scaling, time_by_label, sd_by_label,
                                 "time_ms", "time_ms_sd",
                                 cluster_time_lo, cluster_time_hi)
    ax_scaling.set_xlabel("Sequence length $N$")
    ax_scaling.set_ylabel("Training-step time (ms)")
    ax_scaling.set_title("(a) Scaling with $N$", fontsize=7)

    # ── (b) Overhead vs dense ─────────────────────────────────────────────────
    ratio_by_label = {
        line_label: df[(df["variant"] == "gather_knn") & (df["K"] == k_value)]
        .sort_values("N")["time_over_dense"].tolist()
        for line_label, k_value in [("Dense", 0), ("K=32", 32), ("K=64", 64)]
    }
    ratio_by_label["Dense"] = series_by_variant("dense", "time_over_dense")
    ratio_sd_by_label = {
        line_label: df[(df["variant"] == "gather_knn") & (df["K"] == k_value)]
        .sort_values("N")["ratio_sd"].tolist()
        for line_label, k_value in [("Dense", 0), ("K=32", 32), ("K=64", 64)]
    }
    ratio_sd_by_label["Dense"] = series_by_variant("dense", "ratio_sd")
    plot_lines_with_cluster_band(ax_overhead, ratio_by_label, ratio_sd_by_label,
                                 "time_over_dense", "ratio_sd",
                                 cluster_ratio_lo, cluster_ratio_hi)
    ax_overhead.set_xlabel("Sequence length $N$")
    ax_overhead.set_ylabel("Time / dense (×)")
    ax_overhead.set_title("(b) Overhead vs dense", fontsize=7)

    # ── (c) Cost vs K at the largest measured N ───────────────────────────────
    cost_at_largest_n = df[df["N"] == largest_n].sort_values(["K", "variant"])
    bar_positions = np.arange(len(cost_at_largest_n))
    bar_colors = []
    for _, row in cost_at_largest_n.iterrows():
        if row["variant"] == "cached_dist_mask_knn":
            bar_colors.append(OI_VERMILION)
        elif row["variant"] == "cached_dist_gather_knn":
            bar_colors.append(OI_PURPLE)
        else:
            bar_colors.append(cluster_color_for_k(row["K"]))
    ax_cost_k.bar(bar_positions, cost_at_largest_n["time_ms"],
                  yerr=cost_at_largest_n["time_ms_sd"],
                  color=bar_colors,
                  edgecolor="white", linewidth=0.3, width=0.65,
                  error_kw={"elinewidth": 0.7, "capsize": 1.5})
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
    ax_cost_k.set_xticklabels(
        [("K=0" if row["variant"] == "dense" else
          "cd-mask" if row["variant"] == "cached_dist_mask_knn" else
          "cd-gather" if row["variant"] == "cached_dist_gather_knn" else
          f"K={int(row['K'])}")
         for _, row in cost_at_largest_n.iterrows()],
        fontsize=5.5, rotation=45, ha="right")
    ax_cost_k.set_xlabel("Context size $K$")
    ax_cost_k.set_ylabel("Training-step time (ms)")
    ax_cost_k.set_ylim(0, 230)
    ax_cost_k.set_title("(c) Cost vs $K$", fontsize=7)

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
        Patch(facecolor=cluster_color, alpha=0.35, label="K=2–16"),
        Line2D([0], [0], color=ROLE_K32, linewidth=1.0, marker="o",
               markersize=3, label="K=32"),
        Line2D([0], [0], color=ROLE_K64, linewidth=1.0, marker="o",
               markersize=3, label="K=64"),
        Line2D([0], [0], color=OI_VERMILION, linewidth=1.0, marker="o",
               markersize=3, label="cached-dist mask (K=16)"),
        Line2D([0], [0], color=OI_PURPLE, linewidth=1.0, marker="o",
               markersize=3, label="cached-dist gather (K=16)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               frameon=False, fontsize=6, columnspacing=1.2, handlelength=1.6)
    fig.subplots_adjust(top=0.78, wspace=0.55)

    meta = {
        "sources": [str(path)],
        "values": [
            {"variant": row["variant"], "K": int(row["K"]), "N": int(row["N"]),
             "time_ms": float(row["time_ms"]), "time_ms_sd": float(row["time_ms_sd"])}
            for _, row in df.iterrows()
        ],
    }
    savefig("training_step_k_sweep.pdf", fig, meta=meta)


def plot_training_step_k_sweep_summary():
    """
    Single-panel body summary of the multi-seed K-sweep training-step
    benchmark: training-step time ratio to dense vs sequence length N.
    The gather-KNN cluster K ∈ {2,4,8,16} is drawn as one gray min–max band
    (over the per-K seed-means), K=32 and K=64 as lines, dashed line at
    1.0 = dense; the two cached-dist configurations (K=16) as lines.
    Compact companion to the three-panel ``training_step_k_sweep`` appendix
    figure: the body panel carries the one headline (sparse-attention
    overhead at training scale, largest at N=128 and shrinking as N grows),
    per-configuration detail lives in the appendix.

    Source: benchmark_attn/results/speed_mem_v2.csv — the FULL training step
    (forward + backward + AdamW, fp32, B=2) on the A500, multi-seed protocol
    (5 seeds × 10 timed iterations per cell; mean ± sd across seed runs;
    SPEC 0019 amended 2026-10-01). Ratio sd propagates the numerator and
    dense sd in quadrature. NOT the attention forward time alone.
    """
    path = BASE / "benchmark_attn" / "results" / "speed_mem_v2.csv"
    if not path.exists():
        print("  ⚠ speed_mem_v2.csv not found — skipping training_step_k_sweep_summary.pdf")
        return

    df_raw = pd.read_csv(path)
    df_raw = df_raw[df_raw["status"] == "ok"]
    per_seed = (df_raw.groupby(["variant", "K", "N", "repeat_id"])["time_ms"]
                .mean().reset_index())
    cell = (per_seed.groupby(["variant", "K", "N"])["time_ms"]
            .agg(["mean", "std"]).reset_index())
    cell.columns = ["variant", "K", "N", "time_ms", "time_ms_sd"]
    df = cell  # one row per (variant, K, N): mean and sd across the 5 seed runs

    sequence_lengths = sorted(df["N"].unique())
    dense_time_by_n = df[df["variant"] == "dense"].set_index("N")["time_ms"]
    dense_sd_by_n = df[df["variant"] == "dense"].set_index("N")["time_ms_sd"]
    df["time_over_dense"] = df.apply(
        lambda row: row["time_ms"] / dense_time_by_n[row["N"]], axis=1)
    df["ratio_sd"] = df.apply(
        lambda row: row["time_over_dense"] * np.sqrt(
            (row["time_ms_sd"] / row["time_ms"]) ** 2
            + (dense_sd_by_n[row["N"]] / dense_time_by_n[row["N"]]) ** 2), axis=1)
    cluster_k_values = [2, 4, 8, 16]  # cluster spread; shown as one band

    def series_by_variant(variant, column):
        """Ordered values of ``column`` for one variant, by increasing N."""
        return df[df["variant"] == variant].sort_values("N")[column].tolist()

    fig, ax = plt.subplots(figsize=figsize_wide(2.2))

    cluster_df = df[(df["variant"] == "gather_knn")
                    & (df["K"].isin(cluster_k_values))]
    cluster_ratio_by_n = cluster_df.groupby("N")["time_over_dense"]
    cluster_lo = cluster_ratio_by_n.min()[sequence_lengths]
    cluster_hi = cluster_ratio_by_n.max()[sequence_lengths]
    ax.fill_between(sequence_lengths, cluster_lo, cluster_hi,
                    color="#999999", alpha=0.35, linewidth=0, zorder=1,
                    label="K=2–16")
    for k_value, k_color in [(32, ROLE_K32), (64, ROLE_K64)]:
        variant_rows = df[(df["variant"] == "gather_knn")
                          & (df["K"] == k_value)].sort_values("N")
        ax.fill_between(sequence_lengths,
                        variant_rows["time_over_dense"] - variant_rows["ratio_sd"],
                        variant_rows["time_over_dense"] + variant_rows["ratio_sd"],
                        color=k_color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(sequence_lengths, variant_rows["time_over_dense"],
                color=k_color, linewidth=1.0, marker="o", markersize=2.8,
                zorder=3, label=f"K={k_value}")
    cached_styles = [("cached_dist_mask_knn", OI_VERMILION, "cached-dist mask (K=16)"),
                     ("cached_dist_gather_knn", OI_PURPLE, "cached-dist gather (K=16)")]
    for variant, v_color, v_label in cached_styles:
        rows = df[df["variant"] == variant].sort_values("N")
        ax.fill_between(sequence_lengths,
                        rows["time_over_dense"] - rows["ratio_sd"],
                        rows["time_over_dense"] + rows["ratio_sd"],
                        color=v_color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(sequence_lengths, rows["time_over_dense"],
                color=v_color, linewidth=1.0, marker="o", markersize=2.8,
                zorder=3, label=v_label)
    ax.axhline(1.0, color=ROLE_DENSE_BASELINE, linewidth=1.2, linestyle="--",
               alpha=0.7, zorder=2, label="Dense (K=0)")

    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Training-step time / dense (×)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sequence_lengths)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(100, 700)
    ax.set_ylim(0.9 * df["time_over_dense"].min(),
                1.1 * df["time_over_dense"].max())
    add_horizontal_grid(ax)
    remove_top_right_spines(ax)
    style_legend(ax, loc="upper right", fontsize=6.5)

    meta = {
        "sources": [str(path)],
        "values": [
            {"variant": row["variant"], "K": int(row["K"]), "N": int(row["N"]),
             "time_over_dense": float(row["time_over_dense"]),
             "ratio_sd": float(row["ratio_sd"])}
            for _, row in df.iterrows()
        ],
    }
    savefig("training_step_k_sweep_summary.pdf", fig, meta=meta)


# ══════════════════════════════════════════════════════════════════════════════
#  6. K-Sweep Training Benchmark — Peak Memory
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_step_k_sweep_memory():
    """
    Single-panel figure: peak GPU memory vs N for the multi-seed K-sweep
    training-step benchmark. Source: benchmark_attn/results/speed_mem_v2.csv
    — the FULL training step (forward + backward + AdamW, fp32, B=2) on the
    A500, multi-seed protocol (5 seeds × 10 timed iterations per cell;
    mean ± sd across seed runs; SPEC 0019 amended 2026-10-01). Dense
    baseline, gather-KNN sparse variants (O(NK) memory) and the two
    cached-dist configurations (K=16). Same benchmark as
    ``training_step_k_sweep``.

    Encoding: dense, K=32, K=64 and the two cached-dist variants as lines
    with ±sd bands; the gather-KNN K ∈ {2,4,8,16} cluster as one gray
    min–max band over the per-K seed-means.

    Figure stem: ``training_step_k_sweep_memory`` — named for what the data
    is (full training-step memory), not "attention memory".
    """
    path = BASE / "benchmark_attn" / "results" / "speed_mem_v2.csv"
    if not path.exists():
        print("  ⚠ speed_mem_v2.csv not found — skipping training_step_k_sweep_memory.pdf")
        return

    df_raw = pd.read_csv(path)
    df_raw = df_raw[df_raw["status"] == "ok"]
    per_seed = (df_raw.groupby(["variant", "K", "N", "repeat_id"])["peak_mem_mb"]
                .mean().reset_index())
    cell = (per_seed.groupby(["variant", "K", "N"])["peak_mem_mb"]
            .agg(["mean", "std"]).reset_index())
    cell.columns = ["variant", "K", "N", "peak_mem_mb", "peak_mem_mb_sd"]
    df = cell  # one row per (variant, K, N): mean and sd across the 5 seed runs

    sequence_lengths = sorted(df["N"].unique())
    cluster_k_values = [2, 4, 8, 16]  # cluster spread in memory; shown as one band

    fig, ax = plt.subplots(figsize=figsize_wide(2.2))

    # ── Encodings (same vocabulary as plot_training_step_k_sweep) ────────────
    dense_color = ROLE_DENSE_BASELINE
    cluster_color = "#999999"
    cached_colors = {"cached_dist_mask_knn": (OI_VERMILION, "cached-dist mask (K=16)"),
                     "cached_dist_gather_knn": (OI_PURPLE, "cached-dist gather (K=16)")}

    def series_by_variant(variant, column):
        """Ordered values of ``column`` for one variant, by increasing N."""
        return df[df["variant"] == variant].sort_values("N")[column].tolist()

    # K ∈ {2,4,8,16} spread; drawn as one min–max band over the seed-means.
    cluster_df = df[(df["variant"] == "gather_knn")
                    & (df["K"].isin(cluster_k_values))]
    cluster_mem_by_n = cluster_df.groupby("N")["peak_mem_mb"]
    cluster_mem_lo = cluster_mem_by_n.min()[sequence_lengths]
    cluster_mem_hi = cluster_mem_by_n.max()[sequence_lengths]
    ax.fill_between(sequence_lengths, cluster_mem_lo, cluster_mem_hi,
                    color=cluster_color, alpha=0.35, linewidth=0, zorder=1)
    for k_value, k_color, k_width in [(0, dense_color, 1.6),
                                      (32, ROLE_K32, 1.0),
                                      (64, ROLE_K64, 1.0)]:
        variant_name = "dense" if k_value == 0 else "gather_knn"
        cell_rows = df[(df["variant"] == variant_name) & (df["K"] == k_value)].sort_values("N")
        ys = cell_rows["peak_mem_mb"].tolist()
        sds = cell_rows["peak_mem_mb_sd"].tolist()
        ax.fill_between(sequence_lengths,
                        [y - s for y, s in zip(ys, sds)],
                        [y + s for y, s in zip(ys, sds)],
                        color=k_color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(sequence_lengths, ys, color=k_color, linewidth=k_width,
                marker="o", markersize=2.8, zorder=3,
                label="Dense (K=0)" if k_value == 0 else f"K={k_value}")
    for variant, (v_color, v_label) in cached_colors.items():
        ys = series_by_variant(variant, "peak_mem_mb")
        sds = series_by_variant(variant, "peak_mem_mb_sd")
        ax.fill_between(sequence_lengths,
                        [y - s for y, s in zip(ys, sds)],
                        [y + s for y, s in zip(ys, sds)],
                        color=v_color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(sequence_lengths, ys, color=v_color, linewidth=1.0,
                marker="o", markersize=2.8, zorder=3, label=v_label)

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
        Patch(facecolor=cluster_color, alpha=0.35, label="K=2–16"),
        Line2D([0], [0], color=ROLE_K32, linewidth=1.0, marker="o",
               markersize=3, label="K=32"),
        Line2D([0], [0], color=ROLE_K64, linewidth=1.0, marker="o",
               markersize=3, label="K=64"),
        Line2D([0], [0], color=OI_VERMILION, linewidth=1.0, marker="o",
               markersize=3, label="cached-dist mask (K=16)"),
        Line2D([0], [0], color=OI_PURPLE, linewidth=1.0, marker="o",
               markersize=3, label="cached-dist gather (K=16)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              fontsize=6.5, handlelength=1.6)

    meta = {
        "sources": [str(path)],
        "values": [
            {"variant": row["variant"], "K": int(row["K"]), "N": int(row["N"]),
             "peak_mem_mb": float(row["peak_mem_mb"]),
             "peak_mem_mb_sd": float(row["peak_mem_mb_sd"])}
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


def plot_epoch_decomposition_pie():
    """
    Pie chart of the measured epoch wall-time distribution (SPEC 0041).

    Single-partition leaf decomposition of one training epoch into backward,
    forward (excluding the attention module), the attention module itself,
    optimizer + host remainder, validation, and dataloader data wait.
    Phase shares (data wait, batch compute, validation) come from the
    EpochDecompositionProfiler callback (median over 18 measured epochs on
    five exclusive H100 nodes); the within-step span split (backward /
    forward / attention / host remainder) comes from the PyTorch trace of
    the profile leg, aggregated in ``results/epoch_profile/trace_aggregation.json``.
    Replaces the deprecated hardcoded ``pipeline_breakdown`` composition.

    Sources: results/epoch_profile/*/epoch_profile.json (7 runs);
    results/epoch_profile/trace_aggregation.json.
    """
    trace_path = BASE / "results" / "epoch_profile" / "trace_aggregation.json"
    with open(trace_path) as fh:
        trace = json.load(fh)
    parts = trace["epoch_leaf_partition_pct"]
    labels_full = [
        ("Backward (autograd + backward kernels)", parts["backward"], OI_ORANGE),
        ("Forward, model excl. attention", parts["forward_excl_attention"], OI_BLUE),
        ("Attention module", parts["attention_module"], OI_SKYBLUE),
        ("Optimizer + host dispatch", parts["optimizer_and_host"], OI_GREEN),
        ("Validation", parts["validation"], palette.NEUTRAL),
        ("Data wait (loader)", parts["data_wait"], OI_YELLOW),
    ]
    values = [v for _, v, _ in labels_full]
    colors = [c for _, _, c in labels_full]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 0.6},
    )
    # Legend carries name + share; wedge labels would collide on the two
    # sub-percent slices (validation 1.1%, data wait 0.5%).
    legend_labels = [f"{name}  ({value:.1f}%)" for name, value, _ in labels_full]
    style_legend(
        ax,
        handles=wedges,
        labels=legend_labels,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        fontsize=7,
    )
    ax.set_aspect("equal")

    fig.tight_layout(pad=0.8)
    meta = {
        "sources": [
            str(trace_path),
            "results/epoch_profile/*/epoch_profile.json (SPEC 0041, job 4422737)",
        ],
        "values": {name: value for name, value, _ in labels_full},
        "note": (
            "Phase shares measured by EpochDecompositionProfiler on five "
            "exclusive H100 nodes (spread 1.4%); span split within the step "
            "from the 10-step PyTorch trace (profiler inflates host shares "
            "~20%). Supersedes the deprecated hardcoded pipeline_breakdown."
        ),
    }
    savefig("epoch_decomposition.pdf", fig, meta=meta)


# ══════════════════════════════════════════════════════════════════════════════
#  8. Backward Pass Comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_backward_pass():
    """
    Bar chart comparing forward+backward time for different attention methods.
    Source: benchmark_attn/results/sparse_backward_results_v2.csv
    (consolidated backward benchmark, SPEC 0023; N=8192 chosen as the most
    discriminative shape).
    """
    path_new = BASE / "benchmark_attn" / "results" / "sparse_backward_results_v2.csv"
    path_old = BASE / "benchmark_backward" / "sparse_backward_results_v2.csv"
    path = path_new if path_new.exists() else path_old
    if not path.exists():
        print("  ⚠ sparse_backward_results_v2.csv not found — skipping backward_pass.pdf")
        return

    df = pd.read_csv(path)

    # Filter to N=8192 for clean comparison
    df_n8192 = df[df["N"] == 8192].copy()

    # Simplify method labels
    method_display = {
        "mask_knn": "Mask-KNN",
        "gather_knn": "Gather-KNN",
        "sparse_softmax": "Sparse softmax",
        "dense_masked": "Dense masked",
        "dense_flash": "Dense flash",
        "flex_attn": "FlexAttention",
    }
    df_n8192["method_short"] = df_n8192["method"].map(method_display)

    # Compare K=16
    df_k16 = df_n8192[df_n8192["K"] == 16].copy()
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
#  10. Multi-seed K-Sweep — Tracking Metrics and Per-Epoch Runtime
# ══════════════════════════════════════════════════════════════════════════════
# K axis runs in ascending K with the dense baseline LAST: dense means all
# neighbours, i.e. the K->infinity limit, so it belongs at the right end of
# the axis (owner directive 2026-10-15; leftmost placement broke the ordinal
# K axis and produced misleading curve shapes).
K_SWEEP_CONFIGS = ["k2", "k4", "k8", "k16", "k32", "k64", "baseline"]
K_SWEEP_TICKS = ["2", "4", "8", "16", "32", "64", "dense"]


def plot_knn_sweep_metrics():
    """
    Two-panel body figure for the multi-seed tracking K-sweep: TRA (left)
    and AOGM (right) vs neighbour count K, each with the two evaluation
    scopes of the corrected protocol — the in-sample mean over the five
    training conditions and the genuinely held-out trpL condition — as
    mean ± std across the three training seeds (42/43/44).

    Source: results/knn_sweep_multiseed/summary.csv, written by
    report/scripts/aggregate_multiseed_eval.py from the per-condition
    evaluation files of the corrected-protocol rerun (trpL excluded from
    training). The previous-protocol "trpL in training" pair of the report
    table is deliberately NOT plotted: it exists only for the five
    configurations trained before the launcher fix, not for K=2/8.
    """
    path = BASE / "results" / "knn_sweep_multiseed" / "summary.csv"
    if not path.exists():
        print("  ⚠ summary.csv not found — skipping knn_sweep_metrics.pdf")
        return
    import csv, statistics

    rows = list(csv.DictReader(open(path)))
    metrics = [("tra_insample", "tra_trpl_heldout", "TRA"),
               ("aogm_insample", "aogm_trpl_heldout", "AOGM")]

    stats = {}
    for config in K_SWEEP_CONFIGS:
        sel = [r for r in rows if r["config"] == config]
        if not sel:
            continue
        for insample_key, heldout_key, _ in metrics:
            for key in (insample_key, heldout_key):
                values = [float(r[key]) for r in sel]
                stats[(config, key)] = (statistics.mean(values),
                                        statistics.stdev(values) if len(values) > 1 else 0.0)

    positions = list(range(len(K_SWEEP_CONFIGS)))
    # wspace raised above the matplotlib default: the right panel's y-label
    # ("AOGM") otherwise collides with the left panel's plot area.
    fig, axes = plt.subplots(1, 2, figsize=figsize_wide(2.4),
                             gridspec_kw={"wspace": 0.45})
    for ax, (insample_key, heldout_key, y_label) in zip(axes, metrics):
        for color, marker, key, label in [
                (OI_BLUE, "o", insample_key, "training conds. (in-sample)"),
                (OI_VERMILION, "s", heldout_key, "trpL (held out)")]:
            xs, means, stds = [], [], []
            for pos, config in zip(positions, K_SWEEP_CONFIGS):
                if (config, key) not in stats:
                    continue
                mean, std = stats[(config, key)]
                xs.append(pos); means.append(mean); stds.append(std)
            ax.fill_between(xs, [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.18, linewidth=0, zorder=1)
            ax.plot(xs, means, color=color, marker=marker, markersize=3.2,
                    linewidth=1.1, zorder=3, label=label)
        ax.set_xticks(positions)
        ax.set_xticklabels(K_SWEEP_TICKS, fontsize=7)
        ax.set_xlabel("Neighbour count $K$")
        ax.set_ylabel(y_label)
        add_horizontal_grid(ax)
        remove_top_right_spines(ax)
    axes[0].set_ylim(0.9755, 0.9965)
    axes[1].set_ylim(0, 400)
    # Reserve a bottom band for tick labels, x-labels, and the shared legend;
    # without it the legend anchored below the axes overlaps them.
    fig.subplots_adjust(bottom=0.32)
    # One shared legend below both panels, inside the reserved band, so it
    # cannot overlap the data or the axis labels (previously it sat inside
    # the TRA panel and covered the curves).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.0), borderaxespad=0.0)
    meta = {
        "sources": [str(path)],
        "values": [
            {"config": config, "metric": key, "mean": mean, "std": std}
            for (config, key), (mean, std) in sorted(stats.items())
        ],
    }
    savefig("knn_sweep_metrics.pdf", fig, meta=meta)


def plot_knn_sweep_epoch_time():
    """
    Single-panel figure: wall time per training epoch vs neighbour count K
    for the multi-seed K-sweep (refreshed 500-epoch pipeline, H100 cluster).
    One marker per (config, seed) run — per-epoch time is the per-run median
    over epochs (robust to single-epoch stalls) — plus the median and the
    min-max range across the three seeds per configuration.

    The centre marker is the median across seeds, not the mean, so the
    curve shape tracks the configuration effect directly: with n=3 seeds a
    single slow node (the K=16 seed at ~272 s/epoch throughout) cannot move
    a median, and the measured flatness-in-K stays visible as a flat curve;
    a mean would manufacture a phantom K=16 bump on an otherwise flat
    profile. The min-max whiskers keep the node-to-node variance visible,
    with the y-axis clipped to the healthy range (180-235 s): the slow
    run's whisker runs off the top edge and is annotated with its value,
    so one environment artifact cannot dominate the visual field.

    Source: results/knn_sweep_multiseed/epoch_times.json, written by
    report/scripts/extract_epoch_times.py from the tensorboard ``val_loss``
    wall-time stamps of the completed cluster runs. Node-to-node variance on
    the cluster (one k16 seed ran at ~272 s/epoch throughout) exceeds any
    configuration effect: no K-dependence is resolvable, consistent with the
    Amdahl analysis in the report.
    """
    path = BASE / "results" / "knn_sweep_multiseed" / "epoch_times.json"
    if not path.exists():
        print("  ⚠ epoch_times.json not found — skipping knn_sweep_epoch_time.pdf")
        return
    import json, statistics

    runs = json.load(open(path))
    by_config = {}
    for run in runs.values():
        by_config.setdefault(run["config"], []).append(run)

    positions = list(range(len(K_SWEEP_CONFIGS)))
    fig, ax = plt.subplots(figsize=figsize_wide(1.6))
    xs, centers, lows, highs, meta_values = [], [], [], [], []
    for pos, config in zip(positions, K_SWEEP_CONFIGS):
        runs_cfg = by_config.get(config, [])
        if not runs_cfg:
            continue
        medians = [run["median_epoch_s"] for run in runs_cfg]
        ax.plot([pos] * len(medians), medians, marker="o", linestyle="none",
                markersize=3.5, color=OI_BLUE, alpha=0.55, zorder=2)
        center = statistics.median(medians)
        xs.append(pos)
        centers.append(center)
        lows.append(center - min(medians))
        highs.append(max(medians) - center)
        meta_values.append({"config": config, "n_runs": len(runs_cfg),
                            "median_epoch_s": medians,
                            "median_across_seeds_s": center,
                            "min_s": min(medians), "max_s": max(medians),
                            "mean_of_medians_s": statistics.mean(medians),
                            "n_epochs": [run["n_epochs"] for run in runs_cfg]})
    ax.errorbar(xs, centers, yerr=[lows, highs], color=OI_BLACK, marker="D",
                markersize=3.5, linewidth=1.1, capsize=2.5, zorder=3,
                label="median, min–max range (3 seeds)")
    y_top = 235.0
    ax.set_ylim(180, y_top)
    ax.set_yticks([180, 200, 220])
    for pos, runs_cfg in zip(xs, [by_config.get(config, [])
                                  for config in K_SWEEP_CONFIGS]):
        if not runs_cfg:
            continue
        worst = max(run["median_epoch_s"] for run in runs_cfg)
        if worst > y_top:
            ax.annotate(f"1 seed: {worst:.0f} s (off scale)",
                        xy=(pos, y_top - 1), xytext=(pos + 0.55, y_top - 6),
                        fontsize=6.5, color="#262626", ha="left", va="top",
                        arrowprops=dict(arrowstyle="->", color="#262626",
                                        lw=0.8))
    ax.set_xticks(positions)
    ax.set_xticklabels(K_SWEEP_TICKS, fontsize=7)
    ax.set_xlabel("Neighbour count $K$")
    ax.set_ylabel("Wall time per epoch (s)")
    add_horizontal_grid(ax)
    remove_top_right_spines(ax)
    style_legend(ax, loc="upper left", fontsize=7)
    meta = {"sources": [str(path)], "values": meta_values}
    savefig("knn_sweep_epoch_time.pdf", fig, meta=meta)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Generating Okabe–Ito report figures...")
    print("=" * 60)

    funcs = [
        ("0/9", "Masked Fraction (cutoff statistic)", run_masked_fraction),
        ("1/9", "CNN Convergence Curves", plot_cnn_convergence),
        ("2/9", "CNN Variant Ranking", plot_cnn_variant_ranking),
        ("3/9", "Edge Probe — Balanced Accuracy", plot_edge_probe_bal_acc),
        ("4/9", "Edge Probe — F1 Score", plot_edge_probe_f1),
        ("5/9", "K-Sweep Training Step (3-panel)", plot_training_step_k_sweep),
        ("5a/9", "K-Sweep Training Step (body summary)", plot_training_step_k_sweep_summary),
        ("5b/9", "Probe Significance (forest)", plot_probe_significance),
        ("6/9", "K-Sweep Training Memory", plot_training_step_k_sweep_memory),
        ("7/9", "Pipeline Breakdown", plot_pipeline_breakdown),
        ("7a/9", "Epoch Phase Decomposition (pie, SPEC 0041)", plot_epoch_decomposition_pie),
        ("8/9", "Backward Pass Comparison", plot_backward_pass),
        ("9/9", "KNN TRA Parity", plot_knn_tra_parity),
        ("9a/9", "K-Sweep Tracking Metrics (multi-seed)", plot_knn_sweep_metrics),
        ("9b/9", "K-Sweep Per-Epoch Runtime (multi-seed)", plot_knn_sweep_epoch_time),
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
