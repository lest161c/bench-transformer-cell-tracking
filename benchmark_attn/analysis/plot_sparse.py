"""Visualize gather-sparse attention benchmark results as a single HTML page.

Reads ``benchmark_sparse_results.csv`` from the ``results/`` directory
and generates multi-panel figures (time vs N, memory, reorder speedup,
feasibility map, KNN speedup heatmap) embedded as base64 PNGs in a
self-contained HTML file.

Usage::

    python plot_sparse.py

Output: ``benchmark_attn/results/benchmark_sparse.html``
"""

import base64
import csv
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set_theme(style="whitegrid")

# Colour map for each method label in the benchmark
COLOR_MAP = {
    "dense": "tab:blue",
    "dense Flash": "tab:cyan",
    "NSA": "tab:purple",
    "sparse K=2": "tab:orange",
    "sparse K=4": "tab:orange",
    "sparse K=8": "tab:green",
    "sparse K=16": "tab:green",
    "sparse K=64": "tab:red",
    "sparse V2 K=4": "tab:olive",
    "sparse V2 K=16": "tab:brown",
    "sparse V2 K=64": "tab:brown",
    "sparse K=4 +reorder": "orange",
    "sparse K=16 +reorder": "limegreen",
    "sparse K=64 +reorder": "coral",
}

RESULT_DIR = Path(__file__).resolve().parents[1] / "results"


def make_label(row):
    """Generate a human-readable label from a benchmark result row.

    Args:
        row: A dict-like row with keys ``method``, ``K``, ``reorder``.

    Returns:
        A short label string such as ``"sparse K=16 +reorder"``.
    """
    method = row["method"]
    if method == "dense":
        return "dense"
    if method == "dense_flash":
        return "dense Flash"
    if method == "nsa":
        return "NSA"
    if method == "sparse_v2":
        return f"sparse V2 K={int(row['K'])}"
    if row["reorder"]:
        return f"sparse K={int(row['K'])} +reorder"
    return f"sparse K={int(row['K'])}"


def fig_to_b64(fig):
    """Convert a matplotlib Figure to a base64-encoded PNG string.

    Args:
        fig: The matplotlib Figure to convert.

    Returns:
        A base64 string suitable for embedding in HTML ``<img>`` tags.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_results():
    """Load the sparse benchmark CSV into a DataFrame.

    Returns:
        A pandas DataFrame with columns ``time_s``, ``mem_mb``,
        ``N``, ``L``, ``K``, ``reorder``, ``status``, and ``label``.
    """
    rows = []
    with open(RESULT_DIR / "benchmark_sparse_results.csv") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            row["time_s"] = float(row["time_s"]) if row["time_s"] else float("nan")
            row["mem_mb"] = float(row["mem_mb"]) if row["mem_mb"] else float("nan")
            row["N"] = int(row["N"])
            row["L"] = int(row["L"])
            row["K"] = int(row["K"])
            row["reorder"] = int(row["reorder"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["label"] = df.apply(make_label, axis=1)
    return df


def plot_time_vs_n(df_ok, all_L):
    """Generate log-log Time vs N figures, one per L value."""
    figures = []
    for L in all_L:
        fig, ax = plt.subplots(figsize=(9, 5))
        sub = df_ok[df_ok["L"] == L]
        sns.lineplot(data=sub, x="N", y="time_s", hue="label",
                     palette=COLOR_MAP, marker="o", markersize=7,
                     linewidth=2, ax=ax)
        ax.set_title(f"Time vs N, L={L} layers")
        ax.set_xlabel("N")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)
    return figures


def plot_reorder_speedup(df_ok, all_L):
    """Generate reorder speedup heatmaps, one per L value."""
    figures = []
    for L in all_L:
        sub = df_ok[df_ok["L"] == L].copy()
        no_re = sub[sub["reorder"] == 0].copy()
        with_re = sub[sub["reorder"] == 1].copy()
        merged = no_re.merge(with_re, on=["L", "N", "K"], suffixes=("_base", "_re"))
        if merged.empty:
            continue
        merged["speedup"] = merged["time_s_base"] / merged["time_s_re"]
        pivot = merged.pivot_table(index="N", columns="K", values="speedup")

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=1.0, ax=ax,
                    cbar_kws={"label": "speedup (reorder / base)"},
                    linewidths=0.5, linecolor="gray")
        ax.set_title(f"Reorder speedup (L={L}) — >1 = reorder faster")
        ax.set_ylabel("N")
        ax.set_xlabel("K")
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)
    return figures


def plot_memory_vs_n(df_ok, all_L):
    """Generate incremental peak GPU memory log-log figures."""
    figures = []
    for L in all_L:
        sub = df_ok[df_ok["L"] == L].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=sub, x="N", y="mem_mb", hue="label",
                     palette=COLOR_MAP, marker="o", markersize=7,
                     linewidth=2, ax=ax)
        ax.set_title(f"Incremental Peak GPU Memory (L={L})")
        ax.set_xlabel("N")
        ax.set_ylabel("Incremental peak (MiB)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)
    return figures


def plot_feasibility_map(df, all_L):
    """Generate feasibility maps (OK vs OOM) per L value."""
    figures = []
    for L in all_L:
        sub = df[df["L"] == L].copy()
        sparse_rows = sub[sub["method"] == "sparse"].copy()
        if sparse_rows.empty:
            continue
        sparse_rows["feas_key"] = sparse_rows.apply(
            lambda row: f"K={int(row['K'])}" + ("+r" if row["reorder"] else ""), axis=1
        )
        sparse_rows["status_code"] = (sparse_rows["status"] == "ok").astype(int)
        pivot = sparse_rows.pivot_table(index="N", columns="feas_key",
                                         values="status_code", aggfunc="first")

        fig, ax = plt.subplots(figsize=(8, 2 + 0.4 * len(pivot)))
        cmap = ListedColormap(["#ff6b6b", "#51cf66"])
        sns.heatmap(pivot, annot=False, cmap=cmap, cbar=False, ax=ax,
                    vmin=0, vmax=1, linewidths=1, linecolor="black")
        for i, n in enumerate(pivot.index):
            for j, k in enumerate(pivot.columns):
                val = pivot.loc[n, k]
                ax.text(j + 0.5, i + 0.5, "OK" if val else "OOM",
                        ha="center", va="center", fontsize=7,
                        color="white" if val else "black", weight="bold")
        ax.set_title(f"Feasibility Map (L={L}) — Green=OK, Red=OOM")
        ax.set_ylabel("N")
        ax.set_xlabel("Config")
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)
    return figures


def plot_knn_speedup_heatmap(df_ok, all_L):
    """Generate KNN speedup and memory saving heatmaps over dense baseline."""
    figures = []
    for L in all_L:
        sub = df_ok[df_ok["L"] == L].copy()
        dense_rows = sub[sub["method"] == "dense"].copy()
        sparse_rows = sub[(sub["method"] == "sparse") & (sub["reorder"] == 0)].copy()
        if sparse_rows.empty:
            continue
        merged = sparse_rows.merge(
            dense_rows[["N", "time_s", "mem_mb"]],
            on="N", suffixes=("_sparse", "_dense")
        )
        merged["speedup"] = merged["time_s_dense"] / merged["time_s_sparse"]
        merged["mem_save"] = merged["mem_mb_dense"] / merged["mem_mb_sparse"]
        pivot_speed = merged.pivot_table(index="N", columns="K", values="speedup")
        pivot_mem = merged.pivot_table(index="N", columns="K", values="mem_save")

        fig, ax = plt.subplots(figsize=(8, 3 + 0.35 * len(pivot_speed)))
        sns.heatmap(pivot_speed, annot=True, fmt=".2f", cmap="RdYlGn",
                    center=1.0, ax=ax, linewidths=1, linecolor="gray",
                    cbar_kws={"label": "speedup (dense/sparse)"},
                    vmin=0, vmax=max(2, pivot_speed.values.max()))
        ax.set_title(f"KNN Speedup over Dense (L={L}) — >1 = sparse faster")
        ax.set_ylabel("N"); ax.set_xlabel("K")
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3 + 0.35 * len(pivot_mem)))
        sns.heatmap(pivot_mem, annot=True, fmt=".1f", cmap="RdYlGn",
                    center=1.0, ax=ax, linewidths=1, linecolor="gray",
                    cbar_kws={"label": "mem saving (dense/sparse)"},
                    vmin=0, vmax=max(2, pivot_mem.values.max()))
        ax.set_title(f"KNN Memory Savings over Dense (L={L}) — >1 = less mem")
        ax.set_ylabel("N"); ax.set_xlabel("K")
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)
    return figures


def main():
    """Load benchmark CSV, generate all figures, and write HTML report."""
    df = load_results()
    df_ok = df[df["status"] == "ok"].copy()
    df_oom = df[df["status"] == "oom"].copy()
    all_L = sorted(df["L"].unique())

    figures = []
    figures.extend(plot_time_vs_n(df_ok, all_L))
    figures.extend(plot_reorder_speedup(df_ok, all_L))
    figures.extend(plot_memory_vs_n(df_ok, all_L))
    figures.extend(plot_feasibility_map(df, all_L))
    figures.extend(plot_knn_speedup_heatmap(df_ok, all_L))

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Gather-Sparse Attention Benchmark (with Reorder)</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}",
        "h1{color:#333} h2{color:#555} img{max-width:100%;margin:20px 0;border:1px solid #ddd;border-radius:6px}</style>",
        "</head><body>",
        "<h1>Dense Masked vs Gather-Sparse Attention</h1>",
        "<p>With token-reordering ablation (Hassani et al. 2024) — reorder sequence by spatial proximity for memory locality.</p>",
        f"<p>Configs: {len(df_ok)} OK, {len(df_oom)} OOM</p>",
    ]

    for i, fig in enumerate(figures):
        b64 = fig_to_b64(fig)
        html_parts.append(f"<figure><figcaption>Figure {i+1}</figcaption>")
        html_parts.append(f'<img src="data:image/png;base64,{b64}" /></figure>')

    html_parts.append("</body></html>")

    with open(RESULT_DIR / "benchmark_sparse.html", "w") as file_handle:
        file_handle.write("\n".join(html_parts))

    print(f"Saved benchmark_sparse.html with {len(figures)} figures")


if __name__ == "__main__":
    main()
