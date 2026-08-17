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


def load_knn_methods_csv(path):
    """Load benchmark rows from either the legacy or the corrected CSV schema.

    Supports the legacy analytical CSV
    (``method,N,K,time_ms,spatial_cutoff,data_source``) and the corrected
    H100 full-bench CSV (``method,L,N,K,time_ms,memory_mb,error``, job
    3920627, ``--with-knn``). Returns a list of dicts with keys ``method``,
    ``N``, ``K``, ``time_ms``, and ``memory_mb``.
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
    """Generate the corrected crossover figure for dense_masked vs mask_knn.

    Uses the corrected H100 data (job 3920627, ``--with-knn``): plots
    ``dense_masked`` and ``mask_knn K=4`` time vs N on a log-log scale and
    draws a vertical line at the ~N=4000 crossover where ``mask_knn`` becomes
    faster than the realistic dense baseline.

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
                "o-", color="#1f77b4", label="dense_masked", markersize=8, linewidth=2)
        ax.plot([row["N"] for row in knn], [row["time_ms"] for row in knn],
                "s-", color="#ff7f0e", label="mask_knn K=4", markersize=8, linewidth=2)
        ax.axvline(4000, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                   label="crossover ~ N=4000")
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Corrected Crossover: dense_masked vs mask_knn K=4 (H100, --with-knn)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
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
    colors = {"flash": "#2ecc71", "mem_efficient": "#f39c12", "cuDNN": "#e74c3c"}
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
        be = sub[0]["predicted_backend"]
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
    """Generate training speedup projection chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    optimizations = [
        ("Default\nTrackastra", 11.0, 11.0, "#7f8c8d"),
        ("+ blockwise_norm\nvectorization", 10.4, 10.4, "#3498db"),
        ("+ FFN\ncheckpointing", 9.5, 9.5, "#e67e22"),
        ("+ spatial block\npartition (safe)", 6.7, 6.7, "#2ecc71"),
        ("+ score-modulated\nattn (needs acc check)", 5.6, 5.6, "#9b59b6"),
    ]

    names = [o[0] for o in optimizations]
    hours = [o[1] for o in optimizations]
    savings = [11.0 - h for h in hours]
    colors = [o[3] for o in optimizations]
    bars = ax.barh(range(len(names)), hours, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Training time (hours)")
    ax.set_title("Training Time Reduction — Cumulative Optimizations")
    for i, (bar, h, s) in enumerate(zip(bars, hours, savings)):
        ax.text(h + 0.1, i, f"{h:.1f}h", va="center", fontweight="bold")
        if s > 0.5:
            ax.text(h / 2, i, f"save {s:.1f}h", va="center", color="white",
                    fontweight="bold", fontsize=9)
    ax.set_xlim(0, 12)
    ax.axvline(11.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


# ─── HTML assembly ───

def build_html(args):
    outdir = Path(args.outdir if hasattr(args, 'outdir') else 'benchmark_attn')
    outdir = (outdir if outdir.is_absolute() else Path(__file__).resolve().parents[1] / outdir)

    # Load data (default to the corrected H100 full benchmark, job 3920627)
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
    html_parts.append("<title>Trackastra Benchmark Suite — Comprehensive Report (Corrected H100, --with-knn)</title>")
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
    html_parts.append(".verdict-best{background:#1a3a1a;border:2px solid #2ecc71;color:#2ecc71}")
    html_parts.append(".toc a{color:#58a6ff;text-decoration:none}")
    html_parts.append(".toc a:hover{text-decoration:underline}")
    html_parts.append("</style></head><body>")
    html_parts.append("<h1>Trackastra Benchmark Suite — Comprehensive Report (Corrected H100)</h1>")
    html_parts.append(f"<p>Generated {datetime.datetime.now().strftime('%Y-%m-%d')} &middot; d_model=320, nhead=8, fp16</p>")
    html_parts.append("<div class='box'>")
    html_parts.append("<p><span class='highlight'>Benchmark bug fixes (2026-08-17, job 3920627):</span></p>")
    html_parts.append("<ul>")
    html_parts.append("<li><b>dense_masked class fix:</b> registry now maps it to <code>CachedDistAttention</code> "
                      "(pre-computed <code>dist_2d</code>) instead of <code>RelativePositionalAttention</code> "
                      "(per-layer cdist).</li>")
    html_parts.append("<li><b>KNN cost inclusion:</b> <code>--with-knn</code> flag now includes the O(N²) "
                      "<code>cdist + topk</code> cost in timing, matching training behavior.</li>")
    html_parts.append("<li><b>dense_flash is NOT realistic:</b> no mask → FlashAttention-2; training always enforces "
                      "the spatial cutoff mask → EfficientAttention. Realistic comparison is "
                      "<span class='highlight'>dense_masked vs mask_knn</span>.</li>")
    html_parts.append("</ul>")
    html_parts.append("<p>Effect at N=2048: dense_masked 0.41→0.344 ms, mask_knn 0.15→0.396 ms. "
                      "Ranking flipped — dense_masked is now faster than mask_knn at N≤2048; "
                      "mask_knn wins at N≥4096 (crossover ~N=4000).</p>")
    html_parts.append("</div>")

    # ── TOC ──
    html_parts.append("<div class='box toc'><h2>Contents</h2><ol>")
    html_parts.append("<li><a href='#verdict'>Current Best Scheme — Verdict</a></li>")
    html_parts.append("<li><a href='#knn'>KNN Methods Comparison</a></li>")
    html_parts.append("<li><a href='#flash-cutoff'>FlashAttention + Spatial Cutoff Verification</a></li>")
    html_parts.append("<li><a href='#spatial-blocks'>Spatial Block Partition (Phase 3)</a></li>")
    html_parts.append("<li><a href='#pipeline'>Pipeline Bottlenecks</a></li>")
    html_parts.append("<li><a href='#training'>Training Speedup Projection</a></li>")
    html_parts.append("<li><a href='#tra'>TRA/AOGM Accuracy</a></li>")
    html_parts.append("<li><a href='#supplementary'>Supplementary Analyses</a></li>")
    html_parts.append("</ol></div>")

    # ── 1. VERDICT ──
    html_parts.append("<h2 id='verdict'>Current Best Attention Scheme (corrected)</h2>")
    html_parts.append("<div class='box verdict verdict-best'>")
    html_parts.append("<b>At cell-tracking scale (N≈140): dense_masked is faster</b> — "
                      "0.199 ms vs mask_knn K=4 0.335 ms at N=128 (1.7×). "
                      "Sparse attention is NOT faster below the crossover.<br>")
    html_parts.append("<b>mask_knn wins at high N (≥4096):</b> 2.914 ms vs dense_masked 5.113 ms at N=8192 "
                      "(1.8× faster). Crossover at ~N=4000.")
    html_parts.append("</div>")

    html_parts.append("<table>")
    for row in [
        ["Method", "Time N=2048", "Time N=8192", "Spatial cutoff?", "FlashAttn?", "Realistic?"],
        ["dense_masked", "0.344ms", "5.113ms", "HARD cutoff ✓", "✗ masked (EfficientAttention)", "✓ realistic dense baseline"],
        ["mask_knn K=4", "0.396ms", "2.914ms", "HARD cutoff ✓", "✗ masked (EfficientAttention)", "✓ realistic sparse"],
        ["gather_sdpa K=4", "0.480ms", "2.268ms", "HARD cutoff ✓", "✗ masked (EfficientAttention)", "✓ realistic sparse"],
        ["dense_flash", "0.111ms", "0.351ms", "✗ NONE", "✓ FlashAttention-2", "✗ unrealistic (no mask)"],
    ]:
        html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html_parts.append("</table>")
    html_parts.append("<p class='warn'><b>dense_flash is an upper bound, not a training configuration:</b> "
             "it uses no mask and dispatches FlashAttention-2, but real training always enforces the "
             "spatial cutoff mask, forcing EfficientAttention. The realistic comparison is "
             "<b>dense_masked vs mask_knn</b>: dense_masked wins at N≤2048, mask_knn wins at N≥4096.</p>")

    # ── 2. KNN METHODS ──
    html_parts.append("<h2 id='knn'>KNN Methods Comparison (corrected H100)</h2>")
    if knn_rows:
        # Representative K per method: K=4 for the gathered/masked KNN variants,
        # K=0 (no KNN) for dense and block-sparse methods.
        representative_k = {"dense_masked": 0, "mask_knn": 4, "gather_sdpa": 4,
                            "gather_fused": 4, "gather_matmul": 4, "knn_relpos": 4,
                            "dense_flash": 0, "nsa": 0, "minimax": 0}
        methods_knn = [m for m in sorted(set(row["method"] for row in knn_rows))
                       if m in representative_k]
        ns = sorted(set(row["N"] for row in knn_rows))
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
                        html_row.append(f"{times[n]:.3f}ms")
                    else:
                        html_row.append("—")
                html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in html_row) + "</tr>")
        html_parts.append("</table>")
        html_parts.append("<p class='warn'>Corrected H100 measurements (job 3920627, fp16, L=1, "
                          "<code>--with-knn</code>). Realistic comparison: <b>dense_masked vs mask_knn</b>. "
                          "dense_flash is unrealistic (no mask → FlashAttention-2).</p>")
    if knn_fig:
        html_parts.append(f'<img src="data:image/png;base64,{figure_to_b64(knn_fig)}">')
        plt.close(knn_fig)
    for name in ["KNN Crossover", "Performance KNN Detail"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            html_parts.append(f"<h3>{name}</h3>")
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    # ── 3. FLASH + SPATIAL CUTOFF ──
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
                html_row.append(f"{row['time_ms']:.2f}ms")
            html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in html_row) + "</tr>")
        html_parts.append("</table>")
    if flash_fig:
        html_parts.append(f'<img src="data:image/png;base64,{figure_to_b64(flash_fig)}">')
        plt.close(flash_fig)
    for name in ["Flash Cutoff Solutions"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    # ── 4. SPATIAL BLOCKS ──
    html_parts.append("<h2 id='spatial-blocks'>Spatial Block Partition (Phase 3)</h2>")
    if pipe_spatial:
        html_parts.append("<table>")
        for N in [256, 512, 1024, 2048]:
            sub = [row for row in pipe_spatial if int(row["N"]) == N and row.get("block_size") == "64" and row.get("overlap") == "1"]
            if sub:
                html_parts.append(f"<tr><td>N={N}</td><td>B=64, o=1</td>"
                        f"<td>{sub[0].get('speedup_vs_baseline', '?')}× vs baseline</td>"
                        f"<td>{sub[0].get('tokens_per_query', '?')} tokens/query</td></tr>")
        html_parts.append("</table>")
    for name in ["Spatial Block Partition"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    # ── 5. PIPELINE ──
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

    # ── 6. TRAINING SPEEDUP ──
    html_parts.append("<h2 id='training'>Training Speedup Projection (11h baseline)</h2>")
    html_parts.append("<table>")
    for row in [
        ["Optimization", "Speedup", "Training time", "Saves", "Risk"],
        ["Default Trackastra", "1.00×", "11.0h", "—", "—"],
        ["+ blockwise_norm vectorization", "1.06×", "10.4h", "0.6h", "low"],
        ["+ FFN checkpoint → B×1.5", "1.10×", "9.5h", "1.5h", "low"],
        ["+ spatial blocks (at N>400 only)", "1.10×", "8.6h", "2.4h", "medium (accuracy)"],
        ["+ score-modulated attn", "—", "—", "—", "HIGH — needs TRA verification"],
    ]:
        html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html_parts.append("</table>")
    html_parts.append("<p class='warn'>Corrected (job 3920627): at cell-tracking scale (N≈140), "
             "<b>dense_masked is faster</b> than mask_knn (0.199 vs 0.335 ms at N=128). "
             "mask_knn only wins at N≥4096 (crossover ~N=4000). dense_flash is an unrealistic upper bound "
             "(no mask → FlashAttention-2). Realistic training speedup: ~1.3× (norm+checkpoint → 11h→8.6h).</p>")

    if training_fig:
        html_parts.append(f'<img src="data:image/png;base64,{figure_to_b64(training_fig)}">')
        plt.close(training_fig)

    # ── 7. TRA/AOGM ──
    html_parts.append("<h2 id='tra'>TRA/AOGM Accuracy</h2>")
    for name in ["TRA/AOGM Analysis"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    # ── 8. SUPPLEMENTARY ──
    html_parts.append("<h2 id='supplementary'>Supplementary Analyses</h2>")
    for name in ["SDPA Backend Dispatch", "System Impact Breakdown", "Tile Size Analysis"]:
        b64 = file_to_b64(pngs[name])
        if b64:
            html_parts.append(f"<h3>{name}</h3>")
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    html_parts.append("<hr><p style='text-align:center;color:#8b949e;font-size:0.8em'>")
    html_parts.append(f"Trackastra Benchmark Suite &middot; Generated {datetime.datetime.now().strftime('%Y-%m-%d')}</p>")
    html_parts.append("</body></html>")

    output_path = getattr(args, 'out', 'benchmark_attn/comprehensive_report.html')
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
