#!/usr/bin/env python3
"""Generate combined-benchmark plots and HTML report.

Reads ``results/combined_attention_ssl_2x2.csv`` (produced by
:mod:`scripts.benchmarks.benchmark_combined`) and writes
``results/combined_attention_ssl_report.html`` with embedded PNG
figures.

Usage
-----
::

    python scripts/analysis/plot_combined.py
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_DIR = _WORKSPACE_ROOT / "benchmark_training" / "results"

sns.set_theme(style="whitegrid")

PALETTE = {
    "dense+rand": "#3498db", "dense+ssl": "#2ecc71",
    "sparseK4+rand": "#e67e22", "sparseK4+ssl": "#f39c12",
    "sparseK16+rand": "#e74c3c", "sparseK16+ssl": "#9b59b6",
}
ORDER = ["dense+rand", "dense+ssl", "sparseK4+rand", "sparseK4+ssl",
         "sparseK16+rand", "sparseK16+ssl"]


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Read a benchmark CSV and coerce numeric columns."""
    rows = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            for col, value in row.items():
                try:
                    row[col] = float(value) if value else float("nan")
                except ValueError:
                    pass
            rows.append(row)
    return pd.DataFrame(rows)


def _figure_to_base64_png(fig) -> str:
    """Return a base64-encoded PNG representation of ``fig``."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def _build_figures(df: pd.DataFrame) -> list:
    """Build every matplotlib figure for the report."""
    figures = []

    # 1. Val loss over epochs
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant in ORDER:
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue
        sns.lineplot(data=sub, x="epoch", y="val_loss", marker="o",
                     label=variant, color=PALETTE.get(variant, "gray"), ax=ax)
    ax.set_title("Combined: Validation Loss (K=4 vs K=16)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.legend(title="Variant", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 2. Epoch-1 val loss bar
    ep1 = df[df["epoch"] == 1]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE.get(v, "gray") for v in ep1["variant"]]
    ax.bar(range(len(ep1)), ep1["val_loss"], color=colors, width=0.5)
    ax.set_xticks(range(len(ep1)))
    ax.set_xticklabels(ep1["variant"], rotation=15)
    ax.set_ylabel("Epoch 1 Val Loss")
    ax.set_title("Epoch 1 Initial Val Loss (lower = better)")
    for i, (_, row) in enumerate(ep1.iterrows()):
        ax.text(i, row["val_loss"] + 0.0005, f"{row['val_loss']:.4f}", ha="center", fontsize=9)
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 3. Best val loss bar
    best = df.loc[df.groupby("variant")["val_loss"].idxmin()]
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [PALETTE.get(v, "gray") for v in best["variant"]]
    ax.bar(range(len(best)), best["val_loss"], color=colors, width=0.5)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(best["variant"], rotation=15)
    ax.set_ylabel("Best Val Loss")
    ax.set_title("Best Val Loss Achieved (lower = better)")
    for i, (_, row) in enumerate(best.iterrows()):
        ax.text(i, row["val_loss"] + 0.0002, f"{row['val_loss']:.4f}", ha="center", fontsize=9)
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 4. Time per epoch
    fig, ax = plt.subplots(figsize=(10, 4))
    for variant in ORDER:
        sub = df[df["variant"] == variant]
        if sub.empty:
            continue
        ax.plot(sub["epoch"], sub["time_s"], marker="o", label=variant,
                color=PALETTE.get(variant, "gray"))
    ax.set_title("Time per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.legend(title="Variant", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 5. Cumulative time to target
    target = 0.030
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant in ORDER:
        sub = df[df["variant"] == variant].copy()
        if sub.empty:
            continue
        sub["cum"] = sub["time_s"].cumsum()
        ax.plot(sub["epoch"], sub["cum"], marker="o", label=variant,
                color=PALETTE.get(variant, "gray"))
        hit = sub[sub["val_loss"] <= target]
        if not hit.empty:
            h = hit.iloc[0]
            ax.scatter([h["epoch"]], [h["cum"]], color=PALETTE.get(variant, "gray"),
                       s=80, zorder=5)
            ax.annotate(f"{h['cum']:.1f}s", (h["epoch"], h["cum"]),
                        textcoords="offset points", xytext=(8, 8), fontsize=7,
                        color=PALETTE.get(variant, "gray"))
    ax.set_title(f"Wall Time to Val Loss ≤ {target}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative Time (s)")
    ax.legend(title="Variant", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 6. K comparison (random only, epoch 1)
    k_comp = df[(df["epoch"] == 1) & (df["variant"].str.contains("rand"))]
    fig, ax = plt.subplots(figsize=(7, 4))
    x_labels = ["Dense", "Sparse K=4", "Sparse K=16"]
    vals = [k_comp[k_comp["variant"] == v]["val_loss"].values[0]
            for v in ["dense+rand", "sparseK4+rand", "sparseK16+rand"]]
    ax.bar(range(3), vals, color=["#3498db", "#e67e22", "#e74c3c"], width=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Epoch 1 Val Loss")
    ax.set_title("Effect of K on Initial Convergence (Random Init)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=10)
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    return figures


def main() -> None:
    """Generate the combined benchmark HTML report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=_RESULTS_DIR / "combined_attention_ssl_2x2.csv",
        help="Input CSV (default: results/combined_attention_ssl_2x2.csv).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_RESULTS_DIR / "combined_attention_ssl_report.html",
        help="Output HTML (default: results/combined_attention_ssl_report.html).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Input CSV not found: {args.csv}")

    df = _load_dataframe(args.csv)
    figures = _build_figures(df)

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Combined Benchmark: Sparse + SSL + K ablation</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}</style>",
        "</head><body>",
        "<h1>Combined Benchmark: Sparse Attention + SSL + K Ablation</h1>",
        "<p>6 variants: Dense, Sparse K=4, Sparse K=16 x Random/SSL init.</p>",
    ]
    for index, fig in enumerate(figures, start=1):
        encoded = _figure_to_base64_png(fig)
        html_parts.append(
            f"<figure><figcaption>Figure {index}</figcaption>"
            f"<img src='data:image/png;base64,{encoded}' /></figure>"
        )
    html_parts.append("</body></html>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(html_parts))
    print(f"Saved {args.out} with {len(figures)} figures")


if __name__ == "__main__":
    main()
