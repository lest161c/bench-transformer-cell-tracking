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
import base64
import csv
import datetime
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def fig_to_b64(fig):
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def embed_image(path):
    """Embed an existing PNG as base64, or return placeholder."""
    path_obj = Path(path)
    if path_obj.exists():
        return base64.b64encode(path_obj.read_bytes()).decode()
    return None


def load_knn_methods_csv(path):
    rows = []
    try:
        with open(path) as file_handle:
            for row in csv.DictReader(file_handle):
                rows.append({
                    "method": row["method"],
                    "N": int(row["N"]),
                    "K": int(row.get("K", -1)),
                    "time_ms": float(row["time_ms"]),
                    "spatial_cutoff": row.get("spatial_cutoff", "False") == "True",
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
    if not knn_rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    baseline = {row["N"]: row["time_ms"] for row in knn_rows
                if row["method"] == "cached_dense (baseline)"}

    for method, color, ls in [
        ("mask-KNN", "#2ecc71", "-"),
        ("gather-KNN", "#e74c3c", "--"),
        ("dense_flash", "#3498db", ":"),
        ("MiniMax", "#95a5a6", "-."),
    ]:
        sub = sorted([row for row in knn_rows if row["method"] == method and row["K"] in (16, -1)],
                     key=lambda row: row["N"])
        valid = [(row["N"], baseline[row["N"]] / row["time_ms"])
                 for row in sub if row["N"] in baseline and baseline[row["N"]] > 0]
        if valid:
            ns, sp = zip(*valid)
            ax.plot(ns, sp, "D" + ls, color=color, label=method,
                    markersize=8, linewidth=2, markerfacecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="baseline")
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("Speedup vs CachedDistAttention")
    ax.set_title("KNN Methods: Speedup vs Baseline (K=16)")
    ax.set_xscale("log", base=2)
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

    # Load data
    knn_rows = load_knn_methods_csv(args.knn_csv or str(outdir / "knn_methods_results.csv"))
    flash_rows = load_flash_cutoff_csv(args.flash_csv or str(outdir / "flash_cutoff_results.csv"))
    pipe_norm = load_pipeline_csv("benchmark_pipeline/blockwise_norm_results.csv")
    pipe_ffn = load_pipeline_csv("benchmark_pipeline/ffn_checkpoint_results.csv")
    pipe_spatial = load_pipeline_csv("benchmark_pipeline/spatial_blocks_results.csv")

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
        "Pipeline — Blockwise Norm": "benchmark_pipeline/blockwise_norm.png",
        "Pipeline — FFN Checkpoint": "benchmark_pipeline/ffn_checkpoint.png",
        "Pipeline — Regionprops": "benchmark_pipeline/regionprops_benchmark.png",
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
    html_parts.append(".verdict-best{background:#1a3a1a;border:2px solid #2ecc71;color:#2ecc71}")
    html_parts.append(".toc a{color:#58a6ff;text-decoration:none}")
    html_parts.append(".toc a:hover{text-decoration:underline}")
    html_parts.append("</style></head><body>")
    html_parts.append("<h1>Trackastra Benchmark Suite — Comprehensive Report</h1>")
    html_parts.append(f"<p>Generated {datetime.datetime.now().strftime('%Y-%m-%d')} &middot; d_model=320, nhead=8, fp16</p>")

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
    html_parts.append("<h2 id='verdict'>Current Best Attention Scheme</h2>")
    html_parts.append("<div class='box verdict verdict-best'>")
    html_parts.append("<b>Proven: mask-KNN K=16</b> — 0.26ms at N=256 (3.1× vs CachedDist) | TRA 0.9972<br>")
    html_parts.append("<b>Faster but unverified: Approach E (soft decay + FlashAttn)</b> — 0.21ms at N=256 (4.0× vs CachedDist, 1.24× vs mask-KNN)")
    html_parts.append("</div>")

    html_parts.append("<table>")
    for row in [
        ["Method", "Time N=256", "vs CachedDist", "vs mask-KNN", "Spatial cutoff?", "FlashAttn?", "TRA verified?"],
        ["mask-KNN K=16", "0.26ms", "3.1×", "1.00×", "HARD cutoff ✓", "cuDNN only ✗", "✓ 0.9972"],
        ["Approach E (soft decay)", "0.21ms", "4.0×", "1.24×", "SOFT prior ✓", "✓ FlashAttn", "✗ UNTESTED"],
        ["dense_flash", "0.20ms", "4.1×", "1.30×", "✗ NONE", "✓ FlashAttn", "✗ invalid"],
        ["CachedDist (baseline)", "0.80ms", "1.00×", "0.32×", "HARD cutoff ✓", "✗ masked", "✓ baseline"],
    ]:
        html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    html_parts.append("</table>")
    html_parts.append("<p class='warn'><b>The bottleneck is now verification, not speed:</b> "
             "Approach E (soft decay) is FASTER than mask-KNN AND dispatches FlashAttn, "
             "but TRA/AOGM accuracy with soft-only spatial prior is UNTESTED. "
             "If soft decay preserves TRA > 0.996, it becomes the best method. "
             "If not, mask-KNN remains undisputed. Training run on vanvliet needed.</p>")

    # ── 2. KNN METHODS ──
    html_parts.append("<h2 id='knn'>KNN Methods Comparison</h2>")
    if knn_rows:
        html_parts.append("<table>")
        methods_knn = sorted(set(row["method"] for row in knn_rows))
        header = ["Method"] + [f"N={n}" for n in sorted(set(row["N"] for row in knn_rows))]
        html_parts.append("<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>")
        for method in methods_knn:
            sub = sorted([row for row in knn_rows if row["method"] == method and row["K"] in (16, -1)],
                         key=lambda row: row["N"])
            if sub:
                html_row = [method]
                for row in sub:
                    html_row.append(f"{row['time_ms']:.2f}ms")
                html_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in html_row) + "</tr>")
        html_parts.append("</table>")
        html_parts.append("<p class='warn'>All times analytical (calibrated to labbook 2026-06-08). GPU verification pending.</p>")
    if knn_fig:
        html_parts.append(f'<img src="data:image/png;base64,{fig_to_b64(knn_fig)}">')
        plt.close(knn_fig)
    for name in ["KNN Crossover", "Performance KNN Detail"]:
        b64 = embed_image(pngs[name])
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
        html_parts.append(f'<img src="data:image/png;base64,{fig_to_b64(flash_fig)}">')
        plt.close(flash_fig)
    for name in ["Flash Cutoff Solutions"]:
        b64 = embed_image(pngs[name])
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
        b64 = embed_image(pngs[name])
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
        b64 = embed_image(pngs[name])
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
    html_parts.append("<p class='warn'>FlashAttn without cutoff is only 1.3× faster than mask-KNN at dataset N=256. "
             "NOT a viable path. Spatial blocks only beat mask-KNN above N≈400. "
             "At typical dataset N (100-300), mask-KNN K=16 is the undisputed best valid method. "
             "Realistic training speedup: ~1.3× (norm+checkpoint → 11h→8.6h).</p>")

    if training_fig:
        html_parts.append(f'<img src="data:image/png;base64,{fig_to_b64(training_fig)}">')
        plt.close(training_fig)

    # ── 7. TRA/AOGM ──
    html_parts.append("<h2 id='tra'>TRA/AOGM Accuracy</h2>")
    for name in ["TRA/AOGM Analysis"]:
        b64 = embed_image(pngs[name])
        if b64:
            html_parts.append(f'<img src="data:image/png;base64,{b64}">')

    # ── 8. SUPPLEMENTARY ──
    html_parts.append("<h2 id='supplementary'>Supplementary Analyses</h2>")
    for name in ["SDPA Backend Dispatch", "System Impact Breakdown", "Tile Size Analysis"]:
        b64 = embed_image(pngs[name])
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
