"""Visualize corrected H100 attention benchmark results.

Reads ``results/full_bench_h100.csv`` (corrected H100 full benchmark, job 3920627,
with KNN cost included via ``--with-knn``). Generates multi-panel figures
(time vs N, memory, crossover analysis) embedded as base64 PNGs in a
self-contained HTML file.

The legacy A500 sweep (``benchmark_sparse_results.csv``) used the pre-fix
benchmark that had two bugs:
1. ``dense_masked`` was mapped to ``RelativePositionalAttention``
   (per-layer cdist, O(N²) × 12 layers).
2. KNN index computation was excluded from ``mask_knn`` timing.

These bugs made sparse attention appear 2.7× faster than dense at N=2048.
After the fixes, dense is faster at N≤2048 and sparse only wins at N≥4096.
The corrected H100 data is the only authoritative source.

Usage::

    python plot_sparse.py

Output: ``benchmark_attn/results/benchmark_sparse.html``
"""

import base64
import csv
import datetime
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

# Colour map for the corrected H100 full benchmark methods
H100_COLOR_MAP = {
    "dense_flash": "tab:cyan",
    "dense_masked": "tab:blue",
    "mask_knn": "tab:orange",
    "gather_sdpa": "tab:green",
    "nsa": "tab:purple",
    "minimax": "tab:red",
    "gather_fused": "tab:olive",
    "gather_matmul": "tab:brown",
    "knn_relpos": "tab:pink",
}


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


def load_h100_results():
    """Load the corrected H100 full-benchmark CSV into a DataFrame.

    Reads ``results/full_bench_h100.csv`` (schema
    ``method,L,N,K,time_ms,memory_mb,error``, produced by job 3920627 with
    ``--with-knn`` so KNN index cost is included in ``time_ms``).

    Returns:
        A pandas DataFrame with columns ``method``, ``N``, ``K``, ``time_ms``,
        ``memory_mb``, and a ``label`` column (e.g. ``"mask_knn K=4"``).
    """
    df = pd.read_csv(RESULT_DIR / "full_bench_h100.csv")
    df = df.dropna(subset=["time_ms"])
    df["label"] = df.apply(
        lambda row: row["method"] if row["K"] == 0 else f"{row['method']} K={row['K']}",
        axis=1,
    )
    return df


def select_lowest_k_rows(df):
    """Keep only the lowest-K row per method for each N.

    Dense methods (K=0) keep their single row; K-gated methods (gather_*,
    mask_knn, knn_relpos) collapse onto K=4 so every method contributes
    exactly one line to the H100 time/memory figures.

    Args:
        df: DataFrame from :func:`load_h100_results`.

    Returns:
        Filtered DataFrame with one row per (method, N).
    """
    min_k = df.groupby("method")["K"].transform("min")
    return df[df["K"] == min_k].copy()


def plot_h100_time_vs_n(df):
    """Generate a log-log Time vs N figure for the corrected H100 benchmark.

    One line per method (lowest K), color-coded via ``H100_COLOR_MAP``.
    Highlights dense_flash as unrealistic (no mask → FlashAttention-2) and
    shades the two regions (dense wins below ~N=4000, sparse wins above).

    Args:
        df: DataFrame from :func:`load_h100_results`.

    Returns:
        A matplotlib Figure, or None if the input is empty.
    """
    rows = select_lowest_k_rows(df)
    if rows.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color in H100_COLOR_MAP.items():
        sub = rows[rows["method"] == method].sort_values("N")
        if sub.empty:
            continue
        linestyle = "--" if method == "dense_flash" else "-"
        alpha = 0.5 if method == "dense_flash" else 1.0
        ax.plot(sub["N"], sub["time_ms"], "o" + linestyle, color=color, label=method,
                markersize=7, linewidth=2, alpha=alpha)
    # Shade regions
    ax.axvspan(64, 4000, alpha=0.06, color="#1f77b4", label="dense wins")
    ax.axvspan(4000, 16384, alpha=0.06, color="#ff7f0e", label="sparse wins")
    ax.axvline(4000, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.axvline(140, color="green", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.text(4000, ax.get_ylim()[1] * 0.05, "crossover\nN≈4000",
            fontsize=8, ha="center", color="red")
    ax.text(140, ax.get_ylim()[1] * 0.05, "vanvliet\nN≈140",
            fontsize=8, ha="center", color="green")
    ax.set_title("H100 Time vs N (corrected: --with-knn, CachedDistAttention)\n"
                 "dense_flash (dashed) is unrealistic — no mask → FlashAttention-2")
    ax.set_xlabel("N")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_h100_memory_vs_n(df):
    """Generate a log-log Memory vs N figure for the corrected H100 benchmark.

    One line per method (lowest K), color-coded via ``H100_COLOR_MAP``.

    Args:
        df: DataFrame from :func:`load_h100_results`.

    Returns:
        A matplotlib Figure, or None if the input is empty.
    """
    rows = select_lowest_k_rows(df)
    if rows.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color in H100_COLOR_MAP.items():
        sub = rows[rows["method"] == method].sort_values("N")
        if sub.empty:
            continue
        linestyle = "--" if method == "dense_flash" else "-"
        alpha = 0.5 if method == "dense_flash" else 1.0
        ax.plot(sub["N"], sub["memory_mb"], "o" + linestyle, color=color, label=method,
                markersize=7, linewidth=2, alpha=alpha)
    ax.set_title("H100 Peak Memory vs N (corrected: --with-knn)")
    ax.set_xlabel("N")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_crossover_analysis(df):
    """Generate a figure showing where mask_knn crosses dense_masked.

    Plots ``dense_masked`` and ``mask_knn K=4`` time vs N (log-log) and draws
    a vertical line at the crossover N (~4000) where ``mask_knn`` becomes
    faster than ``dense_masked``.

    Args:
        df: DataFrame from :func:`load_h100_results`.

    Returns:
        A matplotlib Figure, or None if the required methods are missing.
    """
    dense = df[df["method"] == "dense_masked"].sort_values("N")
    knn = df[(df["method"] == "mask_knn") & (df["K"] == 4)].sort_values("N")
    if dense.empty or knn.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dense["N"], dense["time_ms"], "o-", color=H100_COLOR_MAP["dense_masked"],
            label="dense_masked", markersize=7, linewidth=2)
    ax.plot(knn["N"], knn["time_ms"], "s-", color=H100_COLOR_MAP["mask_knn"],
            label="mask_knn K=4", markersize=7, linewidth=2)
    ax.axvline(4000, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
               label="crossover ~ N=4000")
    ax.set_title("Crossover: mask_knn vs dense_masked (corrected H100)")
    ax.set_xlabel("N")
    ax.set_ylabel("Time (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    """Load corrected H100 benchmark CSV, generate figures, write HTML report.

    Only the corrected H100 data (job 3920627, ``--with-knn``) is shown.
    The legacy A500 sweep (``benchmark_sparse_results.csv``) was produced
    with the pre-fix benchmark that had two bugs (wrong class for
    dense_masked, KNN cost excluded). It is not shown here.
    """
    # Corrected H100 full benchmark (job 3920627, --with-knn)
    h100 = load_h100_results()

    figures = []
    for h100_fig in (plot_h100_time_vs_n(h100),
                     plot_h100_memory_vs_n(h100),
                     plot_crossover_analysis(h100)):
        if h100_fig is not None:
            figures.append(h100_fig)

    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Dense vs Sparse Attention Benchmark — Corrected H100 (--with-knn)</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:24px;background:#0d1117;color:#c9d1d9}",
        "h1{color:#58a6ff;border-bottom:2px solid #30363d;padding-bottom:8px}",
        "h2{color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:32px}",
        "h3{color:#8b949e;margin-top:20px}",
        "img{max-width:100%;margin:12px 0;border:1px solid #30363d;border-radius:6px}",
        ".good{color:#3fb950} .bad{color:#f85149} .warn{color:#d2991d}",
        ".box{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;margin:16px 0}",
        ".highlight{color:#58a6ff;font-weight:600}",
        "table{border-collapse:collapse;width:100%;margin:16px 0}",
        "th,td{border:1px solid #30363d;padding:8px 12px;text-align:right;font-size:13px}",
        "th{background:#21262d;color:#8b949e;font-weight:600}",
        "td:first-child,th:first-child{text-align:left}",
        "</style></head><body>",
        "<h1>Dense vs Sparse Attention Benchmark — Corrected H100</h1>",
        f"<p>Generated {datetime.datetime.now().strftime('%Y-%m-%d')} &middot; "
        "d_model=320, nhead=8, fp16, H100 (80 GB), single-layer (L=1)</p>",

        # ── Key Findings ──
        "<div class='box'>",
        "<h2 style='margin-top:0'>Key Findings</h2>",
        "<ol>",
        "<li><b class='bad'>Sparse attention does NOT accelerate cell tracking.</b> "
        "At the vanvliet training regime (N≈140), <code>dense_masked</code> is "
        "<b>1.7× faster</b> than <code>mask_knn</code> (0.199 ms vs 0.335 ms at N=128). "
        "The crossover where sparse becomes faster is at <b>N≈4000</b> — "
        "28× above the vanvliet scale.</li>",
        "<li><b class='warn'>dense_flash is an unrealistic upper bound.</b> "
        "It uses no mask → dispatches FlashAttention-2. But training always enforces "
        "the spatial cutoff mask → forces fallback to EfficientAttention. "
        "No training configuration can use FlashAttention-2.</li>",
        "<li><b>Both <code>dense_masked</code> and <code>mask_knn</code> use "
        "EfficientAttention</b> at training scale. The difference is only the mask "
        "construction: dense uses (dist_2d &gt; cutoff) comparison, sparse uses "
        "KNN scatter.</li>",
        "<li><b>Attention is &lt;15% of step time</b> at N≈140. "
        "Optimizing attention yields negligible training speedup.</li>",
        "</ol>",
        "</div>",

        # ── Crossover table ──
        "<div class='box'>",
        "<h2 style='margin-top:0'>Crossover Table: dense_masked vs mask_knn K=4</h2>",
        "<table>",
        "<tr><th>N</th><th>dense_masked (ms)</th><th>mask_knn K=4 (ms)</th>"
        "<th>Winner</th><th>Sparse speedup</th></tr>",
        "<tr><td>128</td><td>0.199</td><td>0.335</td>"
        "<td class='good'>dense_masked</td><td>0.59× (1.68× slower)</td></tr>",
        "<tr><td>256</td><td>0.200</td><td>0.338</td>"
        "<td class='good'>dense_masked</td><td>0.59× (1.69× slower)</td></tr>",
        "<tr><td>512</td><td>0.200</td><td>0.339</td>"
        "<td class='good'>dense_masked</td><td>0.59× (1.70× slower)</td></tr>",
        "<tr><td>1024</td><td>0.201</td><td>0.397</td>"
        "<td class='good'>dense_masked</td><td>0.51× (1.98× slower)</td></tr>",
        "<tr><td>2048</td><td>0.344</td><td>0.396</td>"
        "<td class='good'>dense_masked</td><td>0.87× (1.15× slower)</td></tr>",
        "<tr><td>4096</td><td>1.311</td><td>0.895</td>"
        "<td class='warn'>mask_knn</td><td>1.47× faster</td></tr>",
        "<tr><td>8192</td><td>5.113</td><td>2.914</td>"
        "<td class='warn'>mask_knn</td><td>1.75× faster</td></tr>",
        "</table>",
        "<p><b>Crossover at ~N=4000.</b> The vanvliet training regime "
        "(N≈140, window=4, ~35 cells/frame) is <b>28× below</b> the crossover.</p>",
        "</div>",
    ]

    # Embed figures
    figure_titles = [
        "Figure 1: H100 Time vs N (corrected) — all methods",
        "Figure 2: H100 Peak Memory vs N (corrected)",
        "Figure 3: Crossover Analysis — dense_masked vs mask_knn K=4",
    ]
    for i, fig in enumerate(figures):
        b64 = fig_to_b64(fig)
        title = figure_titles[i] if i < len(figure_titles) else f"Figure {i+1}"
        html_parts.append(f"<h3>{title}</h3>")
        html_parts.append(f'<img src="data:image/png;base64,{b64}">')
        plt.close(fig)

    # ── Bug fix notes ──
    html_parts.append("<div class='box'>")
    html_parts.append("<h2 style='margin-top:0'>Benchmark Bug Fixes (2026-08-17)</h2>")
    html_parts.append("<ol>")
    html_parts.append("<li><b>dense_masked class fix:</b> Registry mapped <code>dense_masked</code> "
                      "to <code>RelativePositionalAttention</code> (recomputes <code>cdist</code> "
                      "per layer, O(N²) × 12 layers). Fixed to <code>CachedDistAttention</code> "
                      "(uses pre-computed <code>dist_2d</code> once, matching real training).</li>")
    html_parts.append("<li><b>KNN cost inclusion:</b> <code>--with-knn</code> flag now includes "
                      "the O(N²) <code>cdist + topk</code> cost in timing, matching training "
                      "behavior where KNN indices are recomputed every forward pass.</li>")
    html_parts.append("</ol>")
    html_parts.append("<p><b>Effect at N=2048:</b> dense_masked 0.41→0.344 ms (faster), "
                      "mask_knn K=4 0.15→0.396 ms (slower). Ranking flipped — dense_masked is "
                      "now faster than mask_knn at N≤2048.</p>")
    html_parts.append("</div>")

    html_parts.append("<hr><p style='text-align:center;color:#8b949e;font-size:0.8em'>")
    html_parts.append(f"Corrected H100 data (job 3920627, --with-knn) &middot; Generated {datetime.datetime.now().strftime('%Y-%m-%d')}</p>")
    html_parts.append("</body></html>")

    with open(RESULT_DIR / "benchmark_sparse.html", "w") as file_handle:
        file_handle.write("\n".join(html_parts))

    print(f"Saved benchmark_sparse.html with {len(figures)} figures")


if __name__ == "__main__":
    main()
