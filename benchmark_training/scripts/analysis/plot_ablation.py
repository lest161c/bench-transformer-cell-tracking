#!/usr/bin/env python3
"""Generate ablation plots and HTML report.

Reads ``results/full_attention_ablation_results.csv`` (produced by
:mod:`scripts.benchmarks.benchmark_ablation`) and writes
``results/full_attention_ablation_report.html`` with embedded PNG
figures.

Figures produced
----------------
1. Time scaling at L=1 (dense vs sparse K=4 vs sparse K=16)
2. Time scaling at L=4
3. Memory scaling at L=1
4. Convergence at N=512, L=4 (all variants)
5. Convergence at N=512, L=1 (all variants)
6. Final val-loss bar at N=512, L=4, epoch 15
7. Epoch-1 val-loss bar at N=512
8. Speed-comparison table
9. Convergence-ratio table

Usage
-----
::

    python scripts/analysis/plot_ablation.py
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

PALETTE = {"dense": "#3498db", "sparseK4": "#e67e22", "sparseK16": "#e74c3c"}


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
    speed = df[(df["phase"] == "speed") & (df["status"] == "ok")]
    conv = df[(df["phase"] == "converge") & (df["status"] == "ok") & (df["L"] == 4)]
    conv_l1 = df[(df["phase"] == "converge") & (df["status"] == "ok") & (df["L"] == 1)]

    # 1. Time scaling at L=1
    fig, ax = plt.subplots(figsize=(9, 5))
    for attn in ("dense", "sparseK4", "sparseK16"):
        sub = speed[(speed["attn"] == attn) & (speed["L"] == 1)].sort_values("N")
        ax.plot(sub["N"], sub["time_s"] * 1000, marker="o", label=attn, color=PALETTE.get(attn, "gray"))
    ax.set_xlabel("N (cells)")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("Time Scaling: Dense vs Sparse Attention (L=1)")
    ax.legend(title="Attention")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 2. Time scaling at L=4
    fig, ax = plt.subplots(figsize=(9, 5))
    for attn in ("dense", "sparseK4", "sparseK16"):
        sub = speed[(speed["attn"] == attn) & (speed["L"] == 4)].sort_values("N")
        ax.plot(sub["N"], sub["time_s"] * 1000, marker="o", label=attn, color=PALETTE.get(attn, "gray"))
    ax.set_xlabel("N (cells)")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("Time Scaling: Dense vs Sparse Attention (L=4)")
    ax.legend(title="Attention")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 3. Memory scaling at L=1
    fig, ax = plt.subplots(figsize=(9, 5))
    for attn in ("dense", "sparseK4", "sparseK16"):
        sub = speed[(speed["attn"] == attn) & (speed["L"] == 1)].sort_values("N")
        ax.plot(sub["N"], sub["mem_mb"], marker="o", label=attn, color=PALETTE.get(attn, "gray"))
    ax.set_xlabel("N (cells)")
    ax.set_ylabel("Incremental GPU Memory (MB)")
    ax.set_title("Memory Scaling: Dense vs Sparse (L=1)")
    ax.legend(title="Attention")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xscale("log", base=2)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 4. Convergence at N=512, L=4
    fig, ax = plt.subplots(figsize=(10, 5))
    for attn in ("dense", "sparseK4", "sparseK16"):
        for init in ("rand", "ssl"):
            sub = conv[(conv["attn"] == attn) & (conv["init"] == init)]
            if sub.empty:
                continue
            label = f"{attn}+{init}"
            linestyle = "-" if init == "rand" else "--"
            ax.plot(sub["epoch"], sub["val_loss"], marker="o", label=label,
                    color=PALETTE.get(attn, "gray"), linestyle=linestyle, markersize=4)
    ax.set_title("Convergence at N=512, L=4: All Variants")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.legend(title="Variant", fontsize=7)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 5. Convergence at N=512, L=1
    fig, ax = plt.subplots(figsize=(10, 5))
    for attn in ("dense", "sparseK4", "sparseK16"):
        for init in ("rand", "ssl"):
            sub = conv_l1[(conv_l1["attn"] == attn) & (conv_l1["init"] == init)]
            if sub.empty:
                continue
            label = f"{attn}+{init}"
            linestyle = "-" if init == "rand" else "--"
            ax.plot(sub["epoch"], sub["val_loss"], marker="o", label=label,
                    color=PALETTE.get(attn, "gray"), linestyle=linestyle, markersize=4)
    ax.set_title("Convergence at N=512, L=1: All Variants")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.legend(title="Variant", fontsize=7)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 6. Final-loss bar at N=512, L=4, epoch 15
    final = conv[(conv["epoch"] == 15) & (conv["L"] == 4)]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = final["attn"] + "+" + final["init"]
    colors = [PALETTE.get(a, "gray") for a in final["attn"]]
    ax.bar(range(len(final)), final["val_loss"], color=colors, width=0.5)
    ax.set_xticks(range(len(final)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Val Loss (epoch 15)")
    ax.set_title("Final Val Loss at N=512, L=4")
    for i, (_, row) in enumerate(final.iterrows()):
        ax.text(i, row["val_loss"] + 0.01, f"{row['val_loss']:.3f}", ha="center", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 7. Epoch-1 bar
    ep1 = conv[conv["epoch"] == 1]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ep1["attn"] + "+" + ep1["init"]
    colors = [PALETTE.get(a, "gray") for a in ep1["attn"]]
    ax.bar(range(len(ep1)), ep1["val_loss"], color=colors, width=0.5)
    ax.set_xticks(range(len(ep1)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Val Loss (epoch 1)")
    ax.set_title("Epoch 1 Val Loss at N=512, L=4")
    for i, (_, row) in enumerate(ep1.iterrows()):
        ax.text(i, row["val_loss"] + 0.02, f"{row['val_loss']:.3f}", ha="center", fontsize=8)
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 8. Speed-comparison table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    table_rows = [["Metric", "Dense", "Sparse K=4", "Sparse K=16", "Best"]]
    for L in (1, 4):
        for N in (128, 256, 512):
            d = speed[(speed["attn"] == "dense") & (speed["L"] == L) & (speed["N"] == N)]
            if d.empty:
                continue
            s4 = speed[(speed["attn"] == "sparseK4") & (speed["L"] == L) & (speed["N"] == N)]
            s16 = speed[(speed["attn"] == "sparseK16") & (speed["L"] == L) & (speed["N"] == N)]
            dt = d["time_s"].values[0] * 1000
            s4t = s4["time_s"].values[0] * 1000 if not s4.empty else float("nan")
            s16t = s16["time_s"].values[0] * 1000 if not s16.empty else float("nan")
            best = min([v for v in (s4t, s16t) if not np.isnan(v)], default=dt)
            table_rows.append([f"L={L} N={N} (ms)", f"{dt:.0f}", f"{s4t:.0f}", f"{s16t:.0f}", f"{best:.0f}"])
    table = ax.table(cellText=table_rows, cellLoc="center", loc="center", colWidths=[0.25] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Speed Comparison: ms/step (lower = better)", fontsize=12, pad=15)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # 9. Convergence-ratio table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    conv_rows = [["Variant", "Epoch 1 Loss", "Epoch 15 Loss", "Final / Init Ratio"]]
    for attn in ("dense", "sparseK4", "sparseK16"):
        for init in ("rand", "ssl"):
            sub = conv[(conv["attn"] == attn) & (conv["init"] == init)]
            if sub.empty:
                continue
            e1 = sub[sub["epoch"] == 1]["val_loss"].values[0]
            e15 = sub[sub["epoch"] == 15]["val_loss"].values[0]
            ratio = e15 / e1 if e1 > 0 else 0
            conv_rows.append([f"{attn}+{init}", f"{e1:.3f}", f"{e15:.3f}", f"{ratio:.2f}"])
    table = ax.table(cellText=conv_rows, cellLoc="center", loc="center", colWidths=[0.25] * 4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Convergence at N=512, L=4 (lower = better)", fontsize=12, pad=15)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    return figures


def main() -> None:
    """Generate the ablation HTML report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=_RESULTS_DIR / "full_attention_ablation_results.csv",
        help="Input CSV (default: results/full_attention_ablation_results.csv).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_RESULTS_DIR / "full_attention_ablation_report.html",
        help="Output HTML (default: results/full_attention_ablation_report.html).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Input CSV not found: {args.csv}")

    df = _load_dataframe(args.csv)
    figures = _build_figures(df)

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Full Ablation: Attention x K x L x Init</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}</style>",
        "</head><body>",
        "<h1>Full Ablation Study</h1>",
        "<p>All combinations: Dense / SparseK4 / SparseK16 x L=1/4 x N=128-1024 x Random/SSL init.</p>",
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
