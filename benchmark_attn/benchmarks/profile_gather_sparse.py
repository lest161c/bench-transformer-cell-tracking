"""Targeted profiler for GatherSparseAttention at N=128,256,512 with K=16.
Outputs per-operator CUDA time percentages, marks gather/contiguous vs SDPA.
Profiles both forward-only and forward+backward.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import json
import base64

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_parts import GatherSparseAttention


def make_profile_input(N, K, B=2, d=256, h=4, coord_dim=3, mode="none",
                       device="cuda", dtype=torch.float16):
    m = GatherSparseAttention(embed_dim=d, n_head=h, knn_neighbors=K, mode=mode).to(device, dtype)
    q = torch.randn(B, N, d, device=device, dtype=dtype)
    coords = torch.randn(B, N, coord_dim, device=device, dtype=dtype)
    yx = coords[..., 1:]
    dist = torch.cdist(yx, yx)
    _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)
    return m, q, coords, knn_idx


def profile_forward(N, K, device, dtype, mode="none", warmup=3, output_dir="profiler_out"):
    m, q, coords, knn_idx = make_profile_input(N, K, device=device, dtype=dtype, mode=mode)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"sparse_N{N}_K{K}_{mode}"

    # warmup
    for _ in range(warmup):
        _ = m(q, q, q, knn_idx, coords)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function(tag):
            y = m(q, q, q, knn_idx, coords)
            torch.cuda.synchronize()

    key = prof.key_averages()
    table = key.table(sort_by="cuda_time_total", row_limit=20)
    print(f"\n{'='*70}")
    print(f"  PROFILE: {tag} (forward only)")
    print(f"{'='*70}")
    print(table)

    trace_path = os.path.join(output_dir, f"{tag}_fwd_trace.json")
    prof.export_chrome_trace(trace_path)

    records = []
    total_cuda = 0.0
    for evt in key:
        cu = max(getattr(evt, "cuda_time_total", 0), 0)
        total_cuda += cu
        records.append({
            "name": evt.key,
            "count": evt.count,
            "cuda_time_us": cu,
            "cpu_time_us": evt.cpu_time_total,
            "self_cuda_mem_mb": getattr(evt, "self_cuda_memory_usage", 0) / (1024**2),
        })

    records.sort(key=lambda r: r["cuda_time_us"], reverse=True)

    # Relative percentages
    for r in records:
        r["cuda_pct"] = (r["cuda_time_us"] / total_cuda * 100) if total_cuda > 0 else 0

    # ----- chart -----
    top12 = records[:12]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    labels = [r["name"] for r in top12][::-1]
    values = [r["cuda_time_us"] / 1000 for r in top12][::-1]
    pcts = [r["cuda_pct"] for r in top12][::-1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(labels)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.set_xlabel("CUDA Time (ms)")
    ax.set_title(f"{tag} — CUDA Time by Operator (forward only)")
    for bar, val, pct in zip(bars, values, pcts):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ms ({pct:.1f}%)", va="center", fontsize=8)
    fig.tight_layout()
    chart_path = os.path.join(output_dir, f"{tag}_fwd_time.png")
    fig.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "tag": tag,
        "records": records,
        "total_cuda_ms": total_cuda / 1000,
        "table": table,
        "trace_path": trace_path,
        "chart_path": chart_path,
    }


def profile_forward_backward(N, K, device, dtype, mode="none", warmup=3, output_dir="profiler_out"):
    m, q, coords, knn_idx = make_profile_input(N, K, device=device, dtype=dtype, mode=mode)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"sparse_N{N}_K{K}_{mode}"

    # warmup
    for _ in range(warmup):
        m.zero_grad(set_to_none=True)
        y = m(q, q, q, knn_idx, coords)
        loss = y.sum()
        loss.backward()
    torch.cuda.synchronize()

    m.zero_grad(set_to_none=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function(f"{tag}_fwbw"):
            y = m(q, q, q, knn_idx, coords)
            loss = y.sum()
            loss.backward()
            torch.cuda.synchronize()

    key = prof.key_averages()
    table = key.table(sort_by="cuda_time_total", row_limit=25)
    print(f"\n{'='*70}")
    print(f"  PROFILE: {tag} (forward + backward)")
    print(f"{'='*70}")
    print(table)

    trace_path = os.path.join(output_dir, f"{tag}_fwbw_trace.json")
    prof.export_chrome_trace(trace_path)

    records = []
    total_cuda = 0.0
    for evt in key:
        cu = max(getattr(evt, "cuda_time_total", 0), 0)
        total_cuda += cu
        records.append({
            "name": evt.key,
            "count": evt.count,
            "cuda_time_us": cu,
            "cpu_time_us": evt.cpu_time_total,
            "self_cuda_mem_mb": getattr(evt, "self_cuda_memory_usage", 0) / (1024**2),
        })

    records.sort(key=lambda r: r["cuda_time_us"], reverse=True)
    for r in records:
        r["cuda_pct"] = (r["cuda_time_us"] / total_cuda * 100) if total_cuda > 0 else 0

    # ----- chart -----
    top12 = records[:12]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    labels = [r["name"] for r in top12][::-1]
    values = [r["cuda_time_us"] / 1000 for r in top12][::-1]
    pcts = [r["cuda_pct"] for r in top12][::-1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(labels)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.set_xlabel("CUDA Time (ms)")
    ax.set_title(f"{tag} — CUDA Time by Operator (forward + backward)")
    for bar, val, pct in zip(bars, values, pcts):
        ax.text(bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ms ({pct:.1f}%)", va="center", fontsize=8)
    fig.tight_layout()
    chart_path = os.path.join(output_dir, f"{tag}_fwbw_time.png")
    fig.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "tag": tag,
        "records": records,
        "total_cuda_ms": total_cuda / 1000,
        "table": table,
        "trace_path": trace_path,
        "chart_path": chart_path,
    }


def categorize_records(records):
    """Group ops into categories: qkv_proj, gather, transpose_reshape, sdpa, proj, other."""
    cats = {
        "qkv_proj": ["aten::linear", "aten::addmm", "aten::mm"],
        "gather_index": ["aten::index", "aten::index_put_"],
        "transpose_reshape": ["aten::transpose", "aten::view", "aten::reshape", "aten::contiguous", "aten::as_strided"],
        "sdpa": ["aten::scaled_dot_product_attention"],
        "proj_out": ["aten::linear", "aten::addmm", "aten::mm"],  # overlaps, check context by name
        "expand": ["aten::expand"],
        "other": [],
    }
    buckets = {k: 0.0 for k in cats}
    for r in records:
        name = r["name"]
        placed = False
        for cat, patterns in cats.items():
            if cat == "other":
                continue
            for p in patterns:
                if p in name:
                    buckets[cat] += r["cuda_time_us"]
                    placed = True
                    break
            if placed:
                break
        if not placed:
            buckets["other"] += r["cuda_time_us"]
    total = sum(buckets.values())
    for k in buckets:
        buckets[k] = (buckets[k] / total * 100) if total > 0 else 0
    return buckets


def build_html_report(all_results, output_dir):
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>GatherSparseAttention Profiler Report</title>",
        "<style>",
        "body{font-family:monospace;max-width:1400px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}",
        "h1{color:#58a6ff} h2{color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:4px}",
        "pre{background:#161b22;padding:12px;border-radius:6px;overflow-x:auto;font-size:11px;line-height:1.3}",
        "img{max-width:100%;margin:12px 0;border:1px solid #30363d;border-radius:6px}",
        ".section{margin:32px 0} .cat{padding:6px;border-radius:4px;margin:4px 0}",
        ".cat-gather{background:#553322} .cat-sdpa{background:#224455} .cat-proj{background:#335533}",
        "</style></head><body>",
        "<h1>GatherSparseAttention Profiler — N=128,256,512 K=16</h1>",
    ]

    for res in all_results:
        tag = res["tag"]
        parts.append(f"<div class='section'><h2>{tag}</h2>")
        parts.append(f"<p>Total CUDA time: <b>{res['total_cuda_ms']:.4f} ms</b></p>")

        # Category breakdown
        cats = categorize_records(res["records"])
        parts.append("<h3>Category Breakdown</h3><pre>")
        for cat, pct in cats.items():
            bar = "#" * int(pct / 2)
            parts.append(f"  {cat:<20s} {pct:5.1f}%  {bar}")
        parts.append("</pre>")

        # Chart
        parts.append(f'<img src="data:image/png;base64,{_b64(res["chart_path"])}" />')

        # Table
        if res.get("table"):
            parts.append(f"<h3>Operator Table</h3><pre>{res['table']}</pre>")

        parts.append(f"<p>Trace: <code>{res.get('trace_path', '')}</code></p>")
        parts.append("</div>")

    parts.append("</body></html>")
    html_path = os.path.join(output_dir, "profile_gather_report.html")
    with open(html_path, "w") as f:
        f.write("\n".join(parts))
    print(f"\nReport: {html_path}")
    return html_path


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    device = torch.device("cuda")
    dtype = torch.float16
    output_dir = str(Path(__file__).resolve().parents[2] / "profiler_out")
    B = 2
    d = 256
    h = 4
    K = 16

    print(f"Device: {device}, dtype: {dtype}, K={K}")
    print(f"Output:  {output_dir}\n")

    all_results = []

    # 1. Forward-only at N=128, 256, 512
    for N in [128, 256, 512]:
        print(f"\n>>> N={N}: forward-only")
        res = profile_forward(N, K, device, dtype, mode="none", output_dir=output_dir)
        all_results.append(res)

        # Explicit category summary
        cats = categorize_records(res["records"])
        print(f"  Category breakdown: gather={cats['gather_index']:.1f}% "
              f"transpose/reshape={cats['transpose_reshape']:.1f}% "
              f"sdpa={cats['sdpa']:.1f}% "
              f"qkv_proj={cats['qkv_proj']:.1f}% "
              f"other={cats['other']:.1f}%")

    # 2. Forward+backward at N=128, 256, 512
    for N in [128, 256, 512]:
        print(f"\n>>> N={N}: forward+backward")
        res = profile_forward_backward(N, K, device, dtype, mode="none", output_dir=output_dir)
        all_results.append(res)

        cats = categorize_records(res["records"])
        print(f"  Category breakdown: gather={cats['gather_index']:.1f}% "
              f"transpose/reshape={cats['transpose_reshape']:.1f}% "
              f"sdpa={cats['sdpa']:.1f}% "
              f"qkv_proj={cats['qkv_proj']:.1f}% "
              f"other={cats['other']:.1f}%")

    # 3. Save JSON results
    json_path = os.path.join(output_dir, "profile_gather_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results: {json_path}")

    # 4. Build HTML
    build_html_report(all_results, output_dir)


if __name__ == "__main__":
    main()
