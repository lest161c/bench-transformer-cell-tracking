"""PyTorch profiler wrapper for attention benchmarks.

Wraps benchmark functions with torch.profiler, outputs:
  - Operator-level time breakdown (table + bar chart)
  - Operator-level memory breakdown (table + bar chart)
  - Memory timeline (allocated over time)
  - Chrome trace JSON for chrome://tracing
  - Combined HTML report with all figures
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import io
import base64
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, schedule

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_parts import GatherSparseAttention, RelativePositionalAttention, SpatialReorder


def _env(name, default):
    """Read env var with numeric fallback support."""
    val = os.environ.get(name, None)
    if val is None:
        val = os.environ.get(name.lower(), default)
    if isinstance(default, bool):
        return str(val).lower() in ("1", "true", "yes")
    return type(default)(val)


PROFILER_RECORD_SHAPES = _env("PROFILER_RECORD_SHAPES", True)
PROFILER_PROFILE_MEMORY = _env("PROFILER_PROFILE_MEMORY", True)
PROFILER_WITH_STACK  = _env("PROFILER_WITH_STACK",  False)
PROFILER_WARMUP_ITERS = _env("PROFILER_WARMUP", 3)
PROFILER_OUTPUT_DIR = _env("PROFILER_OUTPUT_DIR", "profiler_out")


def get_device():
    """Return (device, dtype) — CUDA+fp16 if available, else CPU+fp32."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def build_activities(device):
    """Build the list of ProfilerActivity targets for the given device."""
    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    return acts


# ------------------------------------------------------------------
# Benchmark workloads  (mirror benchmark_sparse.py, single forward)
# ------------------------------------------------------------------

def make_workload_dense(seq_len, device, dtype, n_layers=1, batch_size=2,
                        embed_dim=256, n_head=4, coord_dim=3):
    """Build a closure that runs n_layers layers of RelativePositionalAttention (dense).

    Args:
        seq_len: Sequence length.
        device: torch device.
        dtype: torch dtype.
        n_layers: Number of layers.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.

    Returns:
        A callable ``fn()`` that runs the forward pass.
    """
    layers = torch.nn.ModuleList([
        RelativePositionalAttention(
            coord_dim=coord_dim, embed_dim=embed_dim, n_head=n_head,
            cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
        ).to(device).to(dtype)
        for _ in range(n_layers)
    ])
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)

    def fn():
        x = query
        for layer in layers:
            x = layer(x, x, x, coords)
        return x
    return fn


def make_workload_sparse(seq_len, knn_neighbors, device, dtype, n_layers=1,
                         batch_size=2, embed_dim=256, n_head=4,
                         coord_dim=3, reorder=False):
    """Build a closure that runs n_layers layers of GatherSparseAttention.

    Args:
        seq_len: Sequence length.
        knn_neighbors: Number of KNN neighbors.
        device: torch device.
        dtype: torch dtype.
        n_layers: Number of layers.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        reorder: If True, reorder tokens by spatial proximity.

    Returns:
        A callable ``fn()`` that runs the forward pass.
    """
    layers = torch.nn.ModuleList([
        GatherSparseAttention(embed_dim=embed_dim, n_head=n_head, knn_neighbors=knn_neighbors, mode="none")
        .to(device).to(dtype)
        for _ in range(n_layers)
    ])
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)

    yx = coords[..., 1:]
    dist = torch.cdist(yx, yx)
    _, knn_idx = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)

    if reorder:
        sr = SpatialReorder(n_bins=32)
        reorder_idx, unreorder_idx = sr.compute_idx(coords)
        coords_re = sr.reorder(coords, reorder_idx)
        yx_re = coords_re[..., 1:]
        dist_re = torch.cdist(yx_re, yx_re)
        _, knn_idx_re = torch.topk(dist_re, k=knn_neighbors, dim=-1, largest=False)

        def fn():
            query_re = sr.reorder(query, reorder_idx)
            x = query_re
            for layer in layers:
                x = layer(x, x, x, knn_idx_re, coords_re)
            return sr.unreorder(x, unreorder_idx)
    else:
        def fn():
            x = query
            for layer in layers:
                x = layer(x, x, x, knn_idx, coords)
            return x

    return fn


# ------------------------------------------------------------------
# Profiler runner
# ------------------------------------------------------------------

def profile_workload(name, make_fn, device, dtype, output_dir,
                     warmup=PROFILER_WARMUP_ITERS):
    """Profile one workload and return dict of stats and paths."""
    print(f"\n{'='*60}")
    print(f"  Profiling: {name}")
    print(f"{'='*60}")

    fn = make_fn()
    activities = build_activities(device)
    device_side = "cuda" if device.type == "cuda" else "cpu"

    with profile(
        activities=activities,
        record_shapes=PROFILER_RECORD_SHAPES,
        profile_memory=PROFILER_PROFILE_MEMORY,
        with_stack=PROFILER_WITH_STACK,
    ) as prof:
        with record_function(name):
            for _ in range(warmup):
                fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()

    os.makedirs(output_dir, exist_ok=True)

    key = prof.key_averages()

    # -- tables --
    time_table = key.table(
        sort_by=f"{device_side}_time_total", row_limit=15
    )
    print("\n[Time breakdown]")
    print(time_table)

    mem_table = key.table(
        sort_by=f"self_{device_side}_memory_usage", row_limit=15
    ) if PROFILER_PROFILE_MEMORY else ""
    if mem_table:
        print("\n[Memory breakdown]")
        print(mem_table)

    # -- chrome trace --
    trace_path = os.path.join(output_dir, f"{name}_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace:  {trace_path}")

    # -- key averages as list of dicts for plotting --
    avg_records = []
    for evt in key:
        avg_records.append({
            "name": evt.key,
            "count": evt.count,
            "cpu_time_total": evt.cpu_time_total,
            "self_cpu_time_total": evt.self_cpu_time_total,
            "cuda_time_total": getattr(evt, "cuda_time_total", 0),
            "self_cuda_time_total": getattr(evt, "self_cuda_time_total", 0),
            "cpu_memory_usage": getattr(evt, "cpu_memory_usage", 0),
            "self_cpu_memory_usage": getattr(evt, "self_cpu_memory_usage", 0),
            "cuda_memory_usage": getattr(evt, "cuda_memory_usage", 0),
            "self_cuda_memory_usage": getattr(evt, "self_cuda_memory_usage", 0),
        })

    # -- memory timeline from trace --
    mem_timeline = extract_memory_timeline_from_trace(trace_path)

    result = {
        "name": name,
        "records": avg_records,
        "time_table": time_table,
        "mem_table": mem_table,
        "trace_path": trace_path,
        "mem_timeline": mem_timeline,
    }

    # -- generate charts --
    chart_paths = make_charts(result, output_dir, device_side)
    result["charts"] = chart_paths

    return result


def extract_memory_timeline_from_trace(trace_path):
    """Parse chrome trace json for memory timeline events."""
    try:
        with open(trace_path) as file_handle:
            trace = json.load(file_handle)
        mem_events = []
        t0 = trace["traceEvents"][0]["ts"] if trace.get("traceEvents") else 0
        for evt in trace.get("traceEvents", []):
            cats = evt.get("cat", "")
            if isinstance(cats, str) and "memory" in cats.lower():
                # PyTorch profiler memory events have different shapes;
                # try the structured way first
                args = evt.get("args", {})
                mem_bytes = 0
                if "Bytes" in args:
                    mem_bytes = args["Bytes"]
                elif "bytes" in args:
                    mem_bytes = args["bytes"]
                elif "Total Reserved" in args:
                    mem_bytes = args["Total Reserved"]
                elif "Allocated" in args:
                    mem_bytes = args["Allocated"]
                mem_events.append({
                    "ts_us": (evt.get("ts", 0) - t0),
                    "mem_mb": mem_bytes / (1024 * 1024),
                    "name": evt.get("name", ""),
                })
        mem_events.sort(key=lambda x: x["ts_us"])
        return mem_events
    except Exception:
        return []


# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------

def make_charts(prof_result, output_dir, device_side):
    """Generate matplotlib charts for a profiler result.

    Creates bar charts for top operators by time and memory, a memory
    timeline, and a call-count chart.  All charts are saved as PNGs
    in ``output_dir``.

    Args:
        prof_result: Dict with keys "name", "records", "mem_timeline".
        output_dir: Directory to save chart PNGs.
        device_side: "cuda" or "cpu" — determines which time/memory
            columns to use.

    Returns:
        Dict mapping chart names to file paths.
    """
    name = prof_result["name"]
    records = prof_result["records"]
    mem_timeline = prof_result["mem_timeline"]
    paths = {}

    # --- Top-12 time bar (total device time) ---
    time_key = f"{device_side}_time_total"
    time_recs = sorted(
        [record for record in records if record.get(time_key, 0) > 0],
        key=lambda record: record.get(time_key, 0), reverse=True
    )[:12]

    if time_recs:
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = [record["name"] for record in time_recs][::-1]
        values = [record[time_key] / 1000 for record in time_recs][::-1]  # us -> ms
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(labels)))
        bars = ax.barh(labels, values, color=colors, edgecolor="white")
        ax.set_xlabel(f"Total {device_side.upper()} Time (ms)")
        ax.set_title(f"{name} — Top Operators by {device_side.upper()} Time")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f} ms", va="center", fontsize=8)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}_time.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths["time_bar"] = path

    # --- Self vs total time comparison ---
    self_key = f"self_{device_side}_time_total"
    top_time = sorted(
        [record for record in records if record.get(time_key, 0) > 0],
        key=lambda record: record.get(time_key, 0), reverse=True
    )[:10]

    if top_time:
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = [record["name"] for record in top_time][::-1]
        self_vals = [record.get(self_key, 0) / 1000 for record in top_time][::-1]
        total_vals = [record[time_key] / 1000 for record in top_time][::-1]
        x = np.arange(len(labels))
        w = 0.35
        ax.barh(x + w / 2, total_vals, w, label="Total", color="steelblue", edgecolor="white")
        ax.barh(x - w / 2, self_vals, w, label="Self", color="darkorange", edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"Time (ms)")
        ax.set_title(f"{name} — Self vs Total {device_side.upper()} Time")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}_time_selfvtotal.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths["time_selfvtotal"] = path

    # --- Top-12 memory bar ---
    mem_key = f"self_{device_side}_memory_usage"
    mem_recs = sorted(
        [record for record in records if record.get(mem_key, 0) > 0],
        key=lambda record: record.get(mem_key, 0), reverse=True
    )[:12]

    if mem_recs:
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = [record["name"] for record in mem_recs][::-1]
        values = [record[mem_key] / (1024 ** 2) for record in mem_recs][::-1]
        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(labels)))
        bars = ax.barh(labels, values, color=colors, edgecolor="white")
        ax.set_xlabel("Self Memory Usage (MiB)")
        ax.set_title(f"{name} — Top Operators by Self Memory")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f} MiB", va="center", fontsize=8)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}_memory.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths["memory_bar"] = path

    # --- Memory timeline ---
    if mem_timeline:
        fig, ax = plt.subplots(figsize=(12, 4))
        ts = np.array([e["ts_us"] for e in mem_timeline])
        mb = np.array([e["mem_mb"] for e in mem_timeline])
        ts_ms = ts / 1000
        ax.plot(ts_ms, mb, color="steelblue", linewidth=1.2, alpha=0.9)
        ax.fill_between(ts_ms, 0, mb, alpha=0.15, color="steelblue")
        if len(ts_ms) > 1:
            ax.set_xlim(ts_ms[0], ts_ms[-1])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Memory (MiB)")
        ax.set_title(f"{name} — Memory Timeline (from trace)")
        ax.grid(True, alpha=0.3, ls="--")
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}_memory_timeline.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths["mem_timeline"] = path

    # --- Calls count ---
    calls_recs = sorted(
        [record for record in records if record["count"] > 1],
        key=lambda record: record["count"], reverse=True
    )[:12]
    if calls_recs:
        fig, ax = plt.subplots(figsize=(12, 4))
        labels = [record["name"] for record in calls_recs][::-1]
        counts = [record["count"] for record in calls_recs][::-1]
        bars = ax.barh(labels, counts, color="mediumseagreen", edgecolor="white")
        ax.set_xlabel("# of Calls")
        ax.set_title(f"{name} — Operator Call Counts")
        for bar, val in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name}_calls.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths["calls"] = path

    return paths


# ------------------------------------------------------------------
# HTML report
# ------------------------------------------------------------------

def fig_to_b64(fig_path):
    """Read a PNG file and return its base64-encoded string."""
    with open(fig_path, "rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode()


def build_html_report(all_results, output_dir):
    """Create a single HTML report with all charts and tables.

    Args:
        all_results: List of profiler result dicts.
        output_dir: Directory to save the HTML report.

    Returns:
        Path to the generated HTML file.
    """
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>PyTorch Profiler Report — Attention Benchmark</title>",
        "<style>",
        "body{font-family:monospace;max-width:1400px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}",
        "h1{color:#58a6ff} h2{color:#f0f6fc;border-bottom:1px solid #30363d;padding-bottom:4px}",
        "pre{background:#161b22;padding:12px;border-radius:6px;overflow-x:auto;font-size:12px;line-height:1.4}",
        "img{max-width:100%;margin:12px 0;border:1px solid #30363d;border-radius:6px}",
        ".section{margin:32px 0}",
        "</style></head><body>",
        "<h1>PyTorch Profiler — Attention Benchmark</h1>",
    ]

    for i, res in enumerate(all_results):
        parts.append(f"<div class='section'><h2>{i+1}. {res['name']}</h2>")

        charts = res.get("charts", {})

        if "time_bar" in charts:
            parts.append(f"<h3>Time Breakdown</h3>")
            parts.append(f'<img src="data:image/png;base64,{fig_to_b64(charts["time_bar"])}" />')

        if "time_selfvtotal" in charts:
            parts.append(f'<img src="data:image/png;base64,{fig_to_b64(charts["time_selfvtotal"])}" />')

        if "memory_bar" in charts:
            parts.append(f"<h3>Memory Breakdown</h3>")
            parts.append(f'<img src="data:image/png;base64,{fig_to_b64(charts["memory_bar"])}" />')

        if "mem_timeline" in charts:
            parts.append(f"<h3>Memory Timeline</h3>")
            parts.append(f'<img src="data:image/png;base64,{fig_to_b64(charts["mem_timeline"])}" />')

        if "calls" in charts:
            parts.append(f"<h3>Call Counts</h3>")
            parts.append(f'<img src="data:image/png;base64,{fig_to_b64(charts["calls"])}" />')

        if res.get("time_table"):
            parts.append(f"<h3>Operator Table (Time)</h3><pre>{res['time_table']}</pre>")
        if res.get("mem_table"):
            parts.append(f"<h3>Operator Table (Memory)</h3><pre>{res['mem_table']}</pre>")

        parts.append(f"<p>Chrome trace: <code>{res.get('trace_path', '')}</code></p>")

        parts.append("</div>")

    parts.append("</body></html>")

    html_path = os.path.join(output_dir, "profiler_report.html")
    with open(html_path, "w") as file_handle:
        file_handle.write("\n".join(parts))
    return html_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    """Run profiler across dense and sparse workloads, generate HTML report.

    Sweeps N=[512,2048,8192] with K=[16,64], with and without spatial
    reorder.  Generates per-workload charts and a combined HTML report
    in PROFILER_OUTPUT_DIR.
    """
    device, dtype = get_device()
    output_dir = PROFILER_OUTPUT_DIR

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Output:  {output_dir}\n")

    batch_size = 2
    embed_dim = 256
    n_head = 4
    coord_dim = 3
    n_layers = 1

    all_results = []

    # --- Scan over N, K ---
    for N in [512, 2048, 8192]:
        # Dense
        name = f"dense_N{N}_L{n_layers}"
        try:
            res = profile_workload(
                name,
                lambda: make_workload_dense(N, device, dtype, n_layers=n_layers, batch_size=batch_size, embed_dim=embed_dim, n_head=n_head, coord_dim=coord_dim),
                device, dtype, output_dir,
            )
            all_results.append(res)
        except RuntimeError as e:
            print(f"  OOM/Error for {name}: {e}")

        # Sparse
        for K in [16, 64]:
            for reorder in [False, True]:
                tag = "reorder" if reorder else "noreorder"
                name = f"sparse_N{N}_K{K}_{tag}_L{n_layers}"
                try:
                    res = profile_workload(
                        name,
                        lambda N=N, K=K, reorder=reorder: make_workload_sparse(
                            N, K, device, dtype, n_layers=n_layers, batch_size=batch_size,
                            embed_dim=embed_dim, n_head=n_head,
                            coord_dim=coord_dim, reorder=reorder,
                        ),
                        device, dtype, output_dir,
                    )
                    all_results.append(res)
                except RuntimeError as e:
                    print(f"  OOM/Error for {name}: {e}")

    # --- Build HTML ---
    if all_results:
        html = build_html_report(all_results, output_dir)
        print(f"\nReport: {html}")


if __name__ == "__main__":
    main()
