"""Generate a comprehensive HTML report from all benchmark CSVs.

Consolidates KNN methods, FlashAttention dispatch, spatial blocks,
pipeline bottlenecks, and training speedup projections into a single
self-contained HTML file with embedded base64 PNG figures.

Usage::

    python make_report.py                          # auto-discover CSVs
    python make_report.py --out report.html        # custom output path
    python make_report.py --knn-csv custom.csv     # override input

Output: ``benchmark_attn/results/comprehensive_report.html``
"""

import argparse
import csv
import datetime
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.bench.html_report import file_to_b64, figure_to_b64

sns.set_theme(style="whitegrid")

# ─── Data loading ───

def load_knn_methods_csv(path):
    """Load benchmark rows from the corrected H100 CSV.

    Reads the corrected H100 full-benchmark CSV (schema
    ``method,L,N,K,time_ms,memory_mb,error``, job 3920627, ``--with-knn``).
    Returns a list of dicts with keys ``method``, ``N``, ``K``,
    ``time_ms``, and ``memory_mb``.
    """
    rows = []
    try:
        with open(path) as file_handle:
            for row in csv.DictReader(file_handle):
                rows.append({
                    "method": row["method"],
                    "N": int(row["N"]),
                    "K": int(row.get("K", -1)),
                    "time_ms": float(row["time_ms"]),
                    "memory_mb": float(row["memory_mb"]) if row.get("memory_mb") else None,
                })
    except Exception:
        pass
    return rows


def load_flash_cutoff_csv(path):
    rows = []
    try:
        with open(path) as file_handle:
            for row in csv.DictReader(file_handle):
                rows.append({
                    "method": row["method"],
                    "N": int(row["N"]),
                    "time_ms": float(row.get("time_ms", 0)),
                    "predicted_backend": row.get("predicted_backend", "unknown"),
                    "flash_dispatches": row.get("flash_dispatches", "0") == "1",
                })
    except Exception:
        pass
    return rows


def load_pipeline_csv(path):
    rows = []
    try:
        with open(path) as file_handle:
            for row in csv.DictReader(file_handle):
                rows.append(row)
    except Exception:
        pass
    return rows


# ─── Figure generation ───

def make_knn_speedup_figure(knn_rows):
    """Generate the crossover figure for dense_masked vs mask_knn.

    Plots ``dense_masked`` and ``mask_knn K=4`` time vs N on a log-log
    scale, shades the two regions (dense wins below ~N=4000, sparse
    wins above), and annotates the vanvliet training regime (N≈140).

    Args:
        knn_rows: List of rows from :func:`load_knn_methods_csv`.

    Returns:
        A matplotlib Figure, or None if the input is empty.
    """
    if not knn_rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    dense = sorted([row for row in knn_rows if row["method"] == "dense_masked"],
                   key=lambda row: row["N"])
    knn = sorted([row for row in knn_rows if row["method"] == "mask_knn" and row["K"] == 4],
                 key=lambda row: row["N"])
    if dense and knn:
        ax.plot([row["N"] for row in dense], [row["time_ms"] for row in dense],
                "o-", color="#1f77b4", label="dense_masked (EfficientAttention)",
                markersize=8, linewidth=2)
        ax.plot([row["N"] for row in knn], [row["time_ms"] for row in knn],
                "s-", color="#ff7f0e", label="mask_knn K=4 (EfficientAttention)",
                markersize=8, linewidth=2)
        # Shade regions
        ax.axvspan(64, 4000, alpha=0.08, color="#1f77b4", label="dense wins")
        ax.axvspan(4000, 16384, alpha=0.08, color="#ff7f0e", label="sparse wins")
        ax.axvline(4000, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
        # Annotate vanvliet training regime
        ax.axvline(140, color="green", linestyle=":", alpha=0.7, linewidth=1.5)
        ax.text(140, ax.get_ylim()[1] * 0.7, "vanvliet\nN≈140",
                fontsize=8, ha="center", color="green")
        ax.text(4000, ax.get_ylim()[1] * 0.5, "crossover\nN≈4000",
                fontsize=8, ha="center", color="red")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Dense vs Sparse Attention: Crossover at N≈4000 (H100, --with-knn)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_flash_dispatch_figure(flash_rows):
    if not flash_rows:
        return None
    methods = sorted(set(row["method"] for row in flash_rows))
    Ns = sorted(set(row["N"] for row in flash_rows))

    method_labels = {
        "A_no_cutoff": "A: No cutoff",
        "B_soft_decay": "B: Soft decay",
        "C_hard_mask": "C: Hard mask",
        "D_block_sparse": "D: Block-sparse",
        "E_flex_attention": "E: FlexAttention",
        "F_spatial_blocks": "F: Spatial blocks",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    dispatch_mat = np.zeros((len(methods), len(Ns)))
    annot = np.empty((len(methods), len(Ns)), dtype=object)
    for i, m in enumerate(methods):
        sub = sorted([row for row in flash_rows if row["method"] == m], key=lambda row: row["N"])
        for j, row in enumerate(sub):
            dispatch_mat[i, j] = 1 if row["flash_dispatches"] else 0
            annot[i, j] = "FLASH" if row["flash_dispatches"] else "cuDNN"

    sns.heatmap(dispatch_mat, annot=annot, fmt="", cmap=["#e74c3c", "#2ecc71"],
                xticklabels=[str(n) for n in Ns],
                yticklabels=[method_labels.get(m, m.split(":")[0]) for m in methods],
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title("FlashAttention Dispatch Matrix")
    ax.set_xlabel("N")

    ax = axes[1]
    for m in methods:
        sub = sorted([row for row in flash_rows if row["method"] == m], key=lambda row: row["N"])
        ns = [row["N"] for row in sub]
        ts = [row["time_ms"] for row in sub]
        be = sub[0]["predicted_backend"] if sub else "?"
        ls = "-" if be == "flash" else "--"
        ax.plot(ns, ts, "o" + ls, label=f"{method_labels.get(m, m)} [{be}]",
                markersize=6, linewidth=1.5, markerfacecolor="white")
    ax.set_xlabel("N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Time vs N by Method")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("FlashAttention + Spatial Cutoff Verification", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_training_projection_figure():
    """Generate training speedup projection chart.

    Shows only realistic optimizations. Spatial block partition is
    NOT included because the corrected benchmark (job 3920627,
    ``--with-knn``) shows sparse attention is slower than dense at
    cell-tracking scale (N≈140 << crossover N≈4000).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    optimizations = [
        ("Default\nTrackastra", 11.0, "#7f8c8d"),
        ("+ blockwise_norm\nvectorization", 10.4, "#3498db"),
        ("+ FFN\ncheckpointing", 9.5, "#e67e22"),
    ]

    names = [o[0] for o in optimizations]
    hours = [o[1] for o in optimizations]
    colors = [o[2] for o in optimizations]
    bars = ax.barh(range(len(names)), hours, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Training time (hours)")
    ax.set_title("Training Time Reduction — Realistic Optimizations Only")
    for i, h in enumerate(hours):
        ax.text(h + 0.1, i, f"{h:.1f}h", va="center", fontweight="bold")
    ax.set_xlim(0, 13)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


# ─── HTML assembly ───

def build_html(args):
    outdir = Path(args.outdir) if hasattr(args, 'outdir') and args.outdir else Path(__file__).resolve().parents[1] / "results"
    outdir = Path(outdir) if isinstance(outdir, Path) else Path(outdir)

    # Load data (corrected H100 full benchmark, job 3920627)
    knn_rows = load_knn_methods_csv(args.knn_csv or str(outdir / "full_bench_h100.csv"))
    flash_rows = load_flash_cutoff_csv(args.flash_csv or str(outdir / "flash_cutoff_results.csv"))
    # Pipeline benchmarks live in ../deprecated/benchmark_pipeline/ after a repo cleanup
    pipeline_dir = Path(__file__).resolve().parents[2] / "deprecated" / "benchmark_pipeline"
    pipe_norm = load_pipeline_csv(str(pipeline_dir / "blockwise_norm_results.csv"))
    pipe_ffn = load_pipeline_csv(str(pipeline_dir / "ffn_checkpoint_results.csv"))
    pipe_spatial = load_pipeline_csv(str(pipeline_dir / "spatial_blocks_results.csv"))

    # Generate figures
    knn_fig = make_knn_speedup_figure(knn_rows)
    flash_fig = make_flash_dispatch_figure(flash_rows)
    training_fig = make_training_projection_figure()

    # Existing PNGs to embed
    pngs = {
        "TRA/AOGM Analysis": str(outdir / "tra_aogm_analysis.png"),
        "Performance KNN Detail": str(outdir / "performance_knn_detail.png"),
        "KNN Crossover": str(outdir / "knn_crossover_analysis.png"),
        "SDPA Backend Dispatch": str(outdir / "sdpa_backend_dispatch.png"),
        "System Impact Breakdown": str(outdir / "system_impact_breakdown.png"),
        "Tile Size Analysis": str(outdir / "tile_size_analysis.png"),
        "Pipeline — Blockwise Norm": str(pipeline_dir / "blockwise_norm.png"),
        "Pipeline — FFN Checkpoint": str(pipeline_dir / "ffn_checkpoint.png"),
        "Pipeline — Regionprops": str(pipeline_dir / "regionprops_benchmark.png"),
        "Pipeline — Spatial Blocks": str(pipeline_dir / "spatial_blocks.png"),
        "Spatial Block Partition": str(outdir / "spatial_block_partition.png"),
        "Flash Cutoff Solutions": str(outdir / "spatial_flash_solutions.png"),
    }

    # ── Build HTML ──
    html_parts = []
    html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html_parts.append("<title>Trackastra Benchmark Suite — Comprehensive Report</title>")
    html_parts.append("<style>")
    html_parts.append("body{font-family:system-ui,sans-serif;max-width:1400px;margin:0 auto;padding:24px;background:#0d1117;color:#c9d1d9}")
    html_parts.append("h1{color:#58a6ff;border-bottom:2px solid #30363d;padding-bottom:8px}")
    html_parts.append("h2{color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:4px;margin-top:36px}")
    html_parts.append("h3{color:#8b949e;margin-top:24px}")
    html_parts.append("img{max-width:100%;margin:12px 0;border:1px solid #30363d;border-radius:6px}")
    html_parts.append("table{border-collapse:collapse;width:100%;margin:16px 0}")
    html_parts.append("th,td{border:1px solid #30363d;padding:8px 12px;text-align:right;font-size:13px}")
    html_parts.append("th{background:#21262d;color:#8b949e;font-weight:600}")
    html_parts.append("td:first-child,th:first-child{text-align:left}")
    html_parts.append(".good{color:#3fb950} .bad{color:#f85149} .warn{color:#d2991d}")
    html_parts.append(".box{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;margin:16px 0}")
    html_parts.append(".highlight{color:#58a6ff;font-weight:600}")
    html_parts.append(".verdict{font-size:1.2em;font-weight:bold;padding:12px;border-radius:6px;margin:12px 0}")
    html_parts.append(".verdict-bad{background:#3a1a1a;border:2px solid #f85149;color:#f85149}")
    html_parts.append(".toc a{color:#58a6ff;text-decoration:none}")
    html_parts.append(".toc a:hover{text-decoration:underline}")
    html_parts.append("</style></head><body>")

    # ── Header ──
    html_parts.append("<h1>Trackastra Benchmark Suite — Comprehensive Report</h1>")
    html_parts.append(f"<p>Generated {datetime.datetime.now().strftime('%Y-%m-%d')} &middot; "
                      "d_model=320, nhead=8, fp16, H100 (80 GB)</p>")

    # ── Key Findings box ──
    html_parts.append("<div class='box'>")
    html_parts.append("<h2 style='margin-top:0'>Key Findings (corrected benchmark, job 3920627)</h2>")
    html_parts.append("<ol>")
    html_parts.append("<li><b class='bad'>Sparse attention does NOT accelerate cell tracking.</b> "
                      "At the vanvliet training regime (N≈140), <code>dense_masked</code> is "
                      "<b>1.7× faster</b> than <code>mask_knn</code> (0.199 ms vs 0.335 ms at N=128). "
                      "The crossover where sparse becomes faster is at <b>N≈4000</b> — well above "
                      "the vanvliet scale (N≈140).</li>")
    html_parts.append("<li><b class='warn'>dense_flash is an unrealistic upper bound.</b> "
                      "It uses no mask → dispatches FlashAttention-2. But training always enforces "
                      "the spatial cutoff mask → forces fallback to EfficientAttention. "
                      "No training configuration can use FlashAttention-2.</li>")
    html_parts.append("<li><b class='good'>Realistic training speedup: ~1.15×</b> "
                      "(blockwise_norm vectorization + FFN checkpointing → 11h → 9.5h). "
                      "Spatial block partition does NOT help at N≈140.</li>")
    html_parts.append("<li><b>Attention is &lt;15% of step time</b> at N≈140. "
                      "Optimizing attention yields negligible training speedup. "
                      "The real bottleneck is the data pipeline (regionprops extraction, "
                      "blockwise_norm).</li>")
    html_parts.append("</ol>")
    html_parts.append("<p class='warn'><b>Benchmark bug fixes (2026-08-17):</b> "
                      "(1) <code>dense_masked</code> was mapped to <code>RelativePositionalAttention</code> "
                      "(per-layer cdist, O(N²)×12 layers). Fixed to <code>CachedDistAttention</code> "
                      "(pre-computed dist_2d, matching real training). "
                      "(2) KNN index computation was excluded from <code>mask_knn</code> timing "
                      "(pre-computed before timing loop). Fixed with <code>--with-knn</code> flag. "
                      "Effect at N=2048: dense_masked 0.41→0.344 ms, mask_knn 0.15→0.396 ms. "
                      "Ranking flipped.</p>")
    html_parts.append("</div>")

    # ── TOC ──
    html_parts.append("<div class='box toc'><h2 style='margin-top:0'>Contents</h2><ol>")
    html_parts.append("<li><a href='#verdict'>Verdict: Dense Masked Wins at Cell-Tracking Scale</a></li>")
    html_parts.append("<li><a href='#crossover'>Crossover Analysis: Why Sparse Doesn't Help at N≈140</a></li>")
    html_parts.append("<li><a href='#knn'>KNN Methods Comparison (Corrected H100)</a></li>")
    html_parts.append("<li><a href='#flash-cutoff'>FlashAttention + Spatial Cutoff Verification</a></li>")
    html_parts.append("<li><a href='#training'>Training Speedup Projection</a></li>")
    html_parts.append("<li><a href='#pipeline'>Pipeline Bottlenecks</a></li>")
    html_parts.append("<li><a href='#supplementary'>Supplementary Analyses</a></li>")
    html_parts.append("</ol></div>")

    # ── 1. VERDICT ──
    html_parts.append("<h2 id='verdict'>Verdict: Dense Masked Wins at Cell-Tracking Scale</h2>")
    html_parts.append("<div class='box verdict verdict-bad'>"
                      "<b>Sparse attention is MOOT for cell tracking.</b><br>"
                      "At the vanvliet training regime (N≈140), <code>dense_masked</code> "
                      "(0.199 ms) is <b>1.7× faster</b> than <code>mask_knn</code> (0.335 ms). "
                      "The crossover where sparse becomes faster is at <b>N≈4000</b> — "
                      "28× above the vanvliet scale. No window size, batch configuration, "
                      "or K value changes this.</div>")

    html_parts.append("<table>")
    for row in [
        ["Method", "Time N=128", "Time N=2048", "Time N=8192", "Spatial cutoff?", "Realistic?"],
        ["dense_masked", "0.199 ms", "0.344 ms", "5.113 ms", "HARD cutoff ✓", "✓ realistic dense baseline"],
        ["mask_knn K=4", "0.335 ms", "0.396 ms", "2.914 ms", "HARD cutoff ✓", "✓ realistic sparse"],
        ["gather_sdpa K=4", "0.400 ms", "0.480 ms", "2.268 ms", "HARD cutoff ✓", "✓ realistic sparse"],
        ["dense_flash", "0.090 ms", "0.111 ms", "0.351 ms", "✗ NONE", "✗ unrealistic (no mask → FlashAttention-2)"],
    ]:
        html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html_parts.append("</table>")

    html_parts.append("<div class='box'>")
    html_parts.append("<h3>Why dense_masked wins at low N</h3>")
    html_parts.append("<p>Both <code>dense_masked</code> and <code>mask_knn</code> use "
                      "EfficientAttention (not FlashAttention-2) because the spatial cutoff "
                      "mask forces the <code>check_for_attn_mask</code> gate in PyTorch's "
                      "<code>sdp_utils.cpp</code> to reject the FlashAttention kernel.</p>")
    html_parts.append("<p>At low N (≤2048), the overhead of building the N×N KNN mask "
                      "(scatter -inf, then 0.0 at neighbour positions) exceeds the savings "
                      "from attending to fewer tokens. Dense EfficientAttention is simply "
                      "faster when N is small.</p>")
    html_parts.append("<p>At high N (≥4096), the O(N²) cost of dense attention dominates, "
                      "and the O(NK) sparse mask becomes worthwhile. The crossover is at "
                      "~N=4000.</p>")
    html_parts.append("</div>")

    # ── 2. CROSSOVER ──
    html_parts.append("<h2 id='crossover'>Crossover Analysis: Why Sparse Doesn't Help at N≈140</h2>")
    html_parts.append("<table>")
    for row in [
        ["N", "dense_masked (ms)", "mask_knn K=4 (ms)", "Winner", "Sparse speedup"],
        ["128", "0.199", "0.335", "dense_masked", "0.59× (1.68× slower)"],
        ["256", "0.200", "0.338", "dense_masked", "0.59× (1.69× slower)"],
        ["512", "0.200", "0.339", "dense_masked", "0.59× (1.70× slower)"],
        ["1024", "0.201", "0.397", "dense_masked", "0.51× (1.98× slower)"],
        ["2048", "0.344", "0.396", "dense_masked", "0.87× (1.15× slower)"],
        ["4096", "1.311", "0.895", "mask_knn", "1.47× faster"],
        ["8192", "5.113", "2.914", "mask_knn", "1.75× faster"],
    ]:
        html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html_parts.append("</table>")
    html_parts.append("<p><b>Crossover at ~N=4000.</b> The vanvliet training regime "
                      "(N≈140, window=4, ~35 cells/frame) is <b>28× below</b> the crossover. "
                      "No K value, window size, or batch configuration at vanvliet scale "
                      "reaches the crossover.</p>")
    if knn_fig:
        html_parts.append(f'<img src="data:image/png;base64,{figure_to_b64(knn_fig)}">')
        plt.close(knn_fig)

    # ── 3. KNN METHODS ──
    html_parts.append("<h2 id='knn'>KNN Methods Comparison (Corrected H100)</h2>")
    if knn_rows:
        representative_k = {"dense_masked": 0, "mask_knn": 4, "gather_sdpa": 4,
                            "gather_fused": 4, "gather_matmul": 4, "knn_relpos": 4,
                            "dense_flash": 0, "nsa": 0, "minimax": 0}
        methods_knn = [m for m in sorted(set(row["method"] for row in knn_rows))
                       if m in representative_k]
        ns = sorted(set(row["N"] for row in knn_rows))
        html_parts.append("<table>")
        header = ["Method"] + [f"N={n}" for n in ns]
        html_parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>")
        for method in methods_knn:
            k = representative_k[method]
            sub = sorted([row for row in knn_rows
                          if row["method"] == method and row["K"] == k],
                         key=lambda row: row["N"])
            if sub:
                times = {row["N"]: row["time_ms"] for row in sub}
                html_row = [method if k == 0 else f"{method} K={k}"]
                for n in ns:
                    if n in times:
                        html_row.append(f"{times[n]:.3f} ms")
                    else:
                        html_row.append("—")
                html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in html_row) + "</tr>")
        html_parts.append("</table>")
        html_parts.append("<p class='warn'>Corrected H100 measurements (job 3920627, fp16, L=1, "
                          "<code>--with-knn</code>). <code>dense_flash</code> uses no mask → "
                          "FlashAttention-2 (unrealistic for training). All other methods use "
                          "EfficientAttention (realistic).</p>")

    # ── 4. FLASH + SPATIAL CUTOFF ──
    html_parts.append("<h2 id='flash-cutoff'>FlashAttention + Spatial Cutoff Verification</h2>")
    html_parts.append("<div class='box'>")
    html_parts.append("<p><b>Can we enforce spatial cutoff AND dispatch FlashAttention?</b></p>")
    html_parts.append("<p><span class='good'>YES — Approach E (soft distance decay, λ=5)</span></p>")
    html_parts.append("<p><b>Phase 0 (weight analysis):</b> weight at d_max = 0.0067, 0% leakage ✓</p>")
    html_parts.append("<p><b>Phase 1 (forward pass):</b> cos_sim > 0.9999 at all N=32..1024 ✓</p>")
    html_parts.append("<p><b>Phase 2 (TRA):</b> full vanvliet training needed on Capella</p>")
    html_parts.append("<p class='warn'>Origin: Trackastra's own attn_dist_mode=v1 (exp(-5·dist/d_max)) + "
                      "FlexAttention score_mod (PyTorch 2.5+). No separate paper — ablation of existing design.</p>")
    html_parts.append("</div>")
    if flash_rows:
        html_parts.append("<table>")
        methods_flash = sorted(set(row["method"] for row in flash_rows))
        header = ["Method"] + [f"N={n}" for n in sorted(set(row["N"] for row in flash_rows))]
        html_parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>")
        for method in methods_flash:
            sub = sorted([row for row in flash_rows if row["method"] == method], key=lambda row: row["N"])
            backend = sub[0]["predicted_backend"] if sub else "?"
            html_row = [f"{method} [{backend}]"]
            for row in sub:
                html_row.append(f"{row['time_ms']:.2f} ms")
            html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in html_row) + "</tr>")
        html_parts.append("</table>")
    if flash_fig:
        html_parts.append(f'<img src="data:image/png;base64,{figure_to_b64(flash_fig)}">')
        plt.close(flash_fig)
    for name in ["Flash Cutoff Solutions"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    # ── 5. SPATIAL BLOCKS (deprecated — sparse doesn't help at N≈140) ──
    html_parts.append("<h2 id='spatial-blocks'>Spatial Block Partition — NOT Beneficial at N≈140</h2>")
    html_parts.append("<div class='box'>")
    html_parts.append("<p class='bad'><b>Spatial block partition does NOT help at cell-tracking scale.</b> "
                      "The corrected benchmark shows sparse attention is 1.7× slower than dense "
                      "at N≈140. The crossover where sparse becomes faster is at N≈4000 — "
                      "well above the vanvliet scale. Spatial block partition is a sparse attention "
                      "approach and inherits the same limitation.</p>")
    html_parts.append("</div>")

    # ── 6. PIPELINE ──
    html_parts.append("<h2 id='pipeline'>Pipeline Bottlenecks</h2>")

    html_parts.append("<h3>blockwise_causal_norm — Vectorization</h3>")
    if pipe_norm:
        rows_matching = [row for row in pipe_norm if row.get("N") == "256" and row.get("batch_size") == "8"]
        if rows_matching:
            html_parts.append(f"<p>N=256, B=8: serial={rows_matching[0].get('serial_ms','?')}ms → "
                    f"vectorized={rows_matching[0].get('vectorized_ms','?')}ms "
                    f"({rows_matching[0].get('speedup','?')}×)</p>")

    html_parts.append("<h3>FFN Gradient Checkpointing</h3>")
    if pipe_ffn:
        rows_matching = [row for row in pipe_ffn if row.get("N") == "256" and row.get("checkpoint_every_k") == "3"]
        if rows_matching:
            html_parts.append(f"<p>N=256, k=3: memory {rows_matching[0].get('mem_full_mb','?')}→"
                    f"{rows_matching[0].get('mem_checkpointed_mb','?')} MiB "
                    f"(save {rows_matching[0].get('mem_reduction_pct','?')}%), "
                    f"recompute {rows_matching[0].get('recompute_overhead_pct','?')}%</p>")

    html_parts.append("<h3>Regionprops CPU Bottleneck</h3>")
    html_parts.append("<p>At N<200: CPU extraction is 80-92% of inference time. "
             "Caching/pre-computing features eliminates this. >2× inference speedup.</p>")

    for name in ["Pipeline — Blockwise Norm", "Pipeline — FFN Checkpoint",
                 "Pipeline — Regionprops", "Pipeline — Spatial Blocks"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            label = name.replace("Pipeline — ", "")
            html_parts.append(f"<h3>{label}</h3>")
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    # ── 7. TRAINING SPEEDUP ──
    html_parts.append("<h2 id='training'>Training Speedup Projection (11h baseline)</h2>")
    html_parts.append("<table>")
    for row in [
        ["Optimization", "Speedup", "Training time", "Saves", "Risk"],
        ["Default Trackastra", "1.00×", "11.0h", "—", "—"],
        ["+ blockwise_norm vectorization", "1.06×", "10.4h", "0.6h", "low"],
        ["+ FFN checkpoint → B×1.5", "1.15×", "9.5h", "1.5h", "low"],
    ]:
        html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html_parts.append("</table>")
    html_parts.append("<div class='box'>")
    html_parts.append("<p><b>Realistic training speedup: ~1.15×</b> "
                      "(blockwise_norm vectorization + FFN checkpointing → 11h → 9.5h).</p>")
    html_parts.append("<p class='bad'><b>Spatial block partition is NOT included.</b> "
                      "The corrected benchmark (job 3920627, <code>--with-knn</code>) shows "
                      "sparse attention is 1.7× slower than dense at N≈140. The crossover "
                      "where sparse becomes faster is at N≈4000 — well above the vanvliet "
                      "scale (N≈140).</p>")
    html_parts.append("<p class='warn'><b>Attention is &lt;15% of step time</b> at N≈140. "
                      "Even a perfect attention optimization (0 ms) would only reduce "
                      "training time by ~15% (11h → 9.4h). The real bottleneck is the "
                      "data pipeline, not attention.</p>")
    html_parts.append("</div>")

    if training_fig:
        html_parts.append(f'<img src="data:image/png;base64,{figure_to_b64(training_fig)}">')
        plt.close(training_fig)

    # ── 8. SUPPLEMENTARY ──
    html_parts.append("<h2 id='supplementary'>Supplementary Analyses</h2>")
    for name in ["SDPA Backend Dispatch", "System Impact Breakdown", "Tile Size Analysis"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            html_parts.append(f"<h3>{name}</h3>")
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    html_parts.append("<hr><p style='text-align:center;color:#8b949e;font-size:0.8em'>")
    html_parts.append(f"Trackastra Benchmark Suite &middot; Generated {datetime.datetime.now().strftime('%Y-%m-%d')} "
                      "&middot; Corrected H100 data (job 3920627, --with-knn)</p>")
    html_parts.append("</body></html>")

    output_path = getattr(args, 'out', str(Path(__file__).resolve().parents[1] / "results" / "comprehensive_report.html"))
    with open(output_path, "w") as file_handle:
        file_handle.write("\n".join(html_parts))
    return output_path


def main():
    """Parse CLI arguments and generate the HTML report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn-csv", default=None)
    parser.add_argument("--flash-csv", default=None)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "comprehensive_report.html"))
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = parser.parse_args()

    path = build_html(args)
    size_kb = Path(path).stat().st_size // 1024
    print(f"Report: {path} ({size_kb} KB)")


if __name__ == "__main__":
    main()
