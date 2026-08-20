"""Sparse backward pass benchmark.

Compares forward, backward, and total time for dense-masked, dense-flash,
gather-KNN, and mask-KNN attention across sequence lengths (N) and
neighborhood sizes (K).  The benchmark uses synthetic random inputs with
fp16 tensors on CUDA.

Run::

    python scripts/benchmarks/benchmark_backward.py --gpu
    python scripts/benchmarks/benchmark_backward.py --profile
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.benchmark as benchmark_timer

from src.attention_modules import (
    DenseFlashAttention,
    GatherSparseAttention,
    KNNMaskSparseAttention,
    RelativePositionalAttention,
)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_forward(forward_fn, num_warmup=5, min_run_time=0.3):
    """Time only the forward pass.

    Parameters
    ----------
    forward_fn : callable
        Zero-argument callable that runs the forward pass.
    num_warmup : int
        Number of untimed warmup iterations.
    min_run_time : float
        Minimum total measurement time in seconds.

    Returns
    -------
    (float, float)
        ``(time_seconds, peak_memory_mb)``.
    """
    for _ in range(num_warmup):
        forward_fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    forward_fn()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    memory_mb = (peak_memory - baseline_memory) / (1024 ** 2)
    timer = benchmark_timer.Timer(
        "forward_fn()", globals={"forward_fn": forward_fn}, num_threads=1
    )
    time_seconds = timer.blocked_autorange(min_run_time=min_run_time).mean
    return time_seconds, memory_mb


def measure_backward(forward_fn, full_fn, num_warmup=5, min_run_time=0.3):
    """Measure forward and backward times separately.

    Parameters
    ----------
    forward_fn : callable
        Zero-argument callable that runs only the forward pass.
    full_fn : callable
        Zero-argument callable that runs forward + backward.
    num_warmup : int
        Number of untimed warmup iterations.
    min_run_time : float
        Minimum total measurement time in seconds.

    Returns
    -------
    (float, float, float, float)
        ``(forward_seconds, backward_seconds, total_seconds, peak_memory_mb)``.
    """
    forward_seconds, forward_memory = measure_forward(
        forward_fn, num_warmup, min_run_time
    )
    for _ in range(num_warmup):
        full_fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    full_fn()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    total_memory_mb = (peak_memory - baseline_memory) / (1024 ** 2)
    timer = benchmark_timer.Timer(
        "full_fn()", globals={"full_fn": full_fn}, num_threads=1
    )
    total_seconds = timer.blocked_autorange(min_run_time=min_run_time).mean
    backward_seconds = total_seconds - forward_seconds
    peak_memory_mb = max(forward_memory, total_memory_mb)
    return forward_seconds, backward_seconds, total_seconds, peak_memory_mb


# ---------------------------------------------------------------------------
# KNN index generation
# ---------------------------------------------------------------------------

def compute_knn_indices(coordinates, num_neighbors):
    """Generate K-nearest-neighbor indices from spatial coordinates.

    Parameters
    ----------
    coordinates : torch.Tensor
        Spatial coordinates of shape ``(batch_size, seq_len, coord_dim)``.
    num_neighbors : int
        Number of nearest neighbors to select per query.  Use ``-1`` for
        all-to-all (dense) attention.

    Returns
    -------
    torch.Tensor
        KNN index tensor of shape ``(batch_size, seq_len, num_neighbors)``.
    """
    if num_neighbors == -1:
        batch_size, seq_len = coordinates.shape[:2]
        return torch.arange(seq_len, device=coordinates.device).view(
            1, seq_len
        ).expand(batch_size, seq_len)
    distances = torch.cdist(coordinates, coordinates, p=2)
    _, indices = torch.topk(distances, k=num_neighbors, dim=-1, largest=False)
    return indices


# ---------------------------------------------------------------------------
# Method factory functions
# ---------------------------------------------------------------------------

def build_dense_masked(embed_dim, num_heads):
    """Build the dense masked attention module (Trackastra baseline)."""
    return RelativePositionalAttention(
        coord_dim=3,
        embed_dim=embed_dim,
        n_head=num_heads,
        cutoff_spatial=128.0,
        mode="none",
        attn_dist_mode="v0",
    )


def build_dense_flash(embed_dim, num_heads):
    """Build the dense FlashAttention module (no mask)."""
    return DenseFlashAttention(
        embed_dim=embed_dim, n_head=num_heads
    )


def build_mask_knn(embed_dim, num_heads, num_neighbors):
    """Build the mask-KNN sparse attention module."""
    return KNNMaskSparseAttention(
        embed_dim=embed_dim,
        n_head=num_heads,
        knn_neighbors=num_neighbors,
        mode="none",
    )


def build_gather_knn(embed_dim, num_heads, num_neighbors):
    """Build the gather-KNN sparse attention module."""
    return GatherSparseAttention(
        embed_dim=embed_dim,
        n_head=num_heads,
        knn_neighbors=num_neighbors,
        mode="none",
    )


# ---------------------------------------------------------------------------
# Per-method measurement
# ---------------------------------------------------------------------------

def measure_method(method_name, build_fn, forward_style, seq_len,
                   num_neighbors, input_tensor, target_tensor, loss_fn,
                   extra_args):
    """Measure forward/backward for one configuration.

    Parameters
    ----------
    method_name : str
        Human-readable name of the method.
    build_fn : callable
        Zero-argument callable that returns an ``nn.Module``.
    forward_style : str
        One of ``"masked"`` (needs coords), ``"flash"`` (qkv only),
        ``"knn"`` (needs knn_indices).
    seq_len : int
        Sequence length N.
    num_neighbors : int
        Neighborhood size K.
    input_tensor : torch.Tensor
        Input query/key/value tensor (same tensor used for all three).
    target_tensor : torch.Tensor
        Target tensor for loss computation.
    loss_fn : callable
        Loss function accepting ``(prediction, target)``.
    extra_args : tuple
        Additional arguments to pass to the module's forward method.

    Returns
    -------
    dict or None
        Result row dict, or ``None`` if the configuration failed.
    """
    module = build_fn().to(input_tensor.device, input_tensor.dtype)
    module.train()

    def forward():
        """Run the forward pass only."""
        return module(input_tensor, input_tensor, input_tensor, *extra_args)

    def full():
        """Run forward + backward pass."""
        output = forward()
        loss = loss_fn(output, target_tensor)
        loss.backward()

    try:
        forward_seconds, backward_seconds, total_seconds, memory_mb = (
            measure_backward(forward, full)
        )
    except Exception as exc:
        print(f"  ERROR: {method_name} N={seq_len} K={num_neighbors}: {exc}")
        del module
        gc.collect()
        torch.cuda.empty_cache()
        return None

    ratio = forward_seconds / backward_seconds if backward_seconds > 0 else float("inf")
    row = {
        "method": method_name,
        "N": seq_len,
        "K": num_neighbors,
        "forward_ms": forward_seconds * 1000,
        "backward_ms": backward_seconds * 1000,
        "total_ms": total_seconds * 1000,
        "peak_memory_mb": round(memory_mb, 2),
        "fwd_bwd_ratio": round(ratio, 3),
    }
    print(
        f"  {method_name:15s} N={seq_len:4d} K={num_neighbors:2d}  "
        f"fwd={forward_seconds * 1000:8.3f}ms  bwd={backward_seconds * 1000:8.3f}ms  "
        f"mem={memory_mb:6.1f}MB  ratio={ratio:.2f}"
    )
    del module
    gc.collect()
    torch.cuda.empty_cache()
    return row


# ---------------------------------------------------------------------------
# Main GPU benchmark
# ---------------------------------------------------------------------------

def run_gpu():
    """Run the GPU benchmark for all methods and N/K combinations.

    Returns
    -------
    (list, list, list)
        ``(rows, sequence_lengths, neighborhood_sizes)``.
    """
    device = torch.device("cuda")
    dtype = torch.float16

    sequence_lengths = [128, 256, 512, 1024, 2048, 8192]
    neighborhood_sizes = [4, 16, 32]
    embed_dim = 320
    num_heads = 8

    # Forward style per method: "masked" = (coords), "flash" = (), "knn" = (knn_indices)
    forward_styles = {
        "dense_masked": "masked",
        "dense_flash": "flash",
        "mask_knn": "knn",
        "gather_knn": "knn",
    }

    rows = []

    for seq_len in sequence_lengths:
        input_tensor = torch.randn(1, seq_len, embed_dim, device=device, dtype=dtype)
        coords_3d = torch.randn(1, seq_len, 3, device=device, dtype=dtype)
        target_tensor = torch.randn_like(input_tensor)
        loss_fn = torch.nn.MSELoss()

        for num_neighbors in neighborhood_sizes:
            knn_indices = compute_knn_indices(coords_3d[..., 1:], num_neighbors)

            for method_name, style in forward_styles.items():
                if style == "masked":
                    extra_args = (coords_3d,)
                elif style == "flash":
                    extra_args = ()
                elif style == "knn":
                    extra_args = (knn_indices,)
                else:
                    raise ValueError(f"Unknown forward style: {style}")

                build_fn = _make_build_fn(
                    method_name, embed_dim, num_heads, num_neighbors
                )

                row = measure_method(
                    method_name,
                    build_fn,
                    style,
                    seq_len,
                    num_neighbors,
                    input_tensor,
                    target_tensor,
                    loss_fn,
                    extra_args,
                )
                if row is not None:
                    rows.append(row)

    return rows, sequence_lengths, neighborhood_sizes


def _make_build_fn(method_name, embed_dim, num_heads, num_neighbors):
    """Return a zero-argument callable that builds the named attention module.

    Parameters
    ----------
    method_name : str
        One of ``"dense_masked"``, ``"dense_flash"``, ``"mask_knn"``,
        ``"gather_knn"``.
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_neighbors : int
        Neighborhood size K for KNN methods (ignored otherwise).

    Returns
    -------
    callable
        Zero-argument function that returns the configured attention module.
    """
    if method_name == "dense_masked":
        return lambda: build_dense_masked(embed_dim, num_heads)
    if method_name == "dense_flash":
        return lambda: build_dense_flash(embed_dim, num_heads)
    if method_name == "mask_knn":
        return lambda: build_mask_knn(embed_dim, num_heads, num_neighbors)
    if method_name == "gather_knn":
        return lambda: build_gather_knn(embed_dim, num_heads, num_neighbors)
    raise ValueError(f"Unknown method name: {method_name}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_csv(rows, path):
    """Save benchmark rows to a CSV file.

    Parameters
    ----------
    rows : list of dict
        Benchmark result rows.
    path : str or Path
        Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("  WARNING: no rows to save!")
        return
    with open(path, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV: {path}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_figures(rows, sequence_lengths, neighborhood_sizes, output_dir="results"):
    """Generate the 2×2 benchmark visualization.

    Parameters
    ----------
    rows : list of dict
        Benchmark result rows.
    sequence_lengths : list of int
        The N values that were benchmarked.
    neighborhood_sizes : list of int
        The K values that were benchmarked.
    output_dir : str
        Directory to write the output PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    palette = {
        "dense_masked": "#4C72B0",
        "dense_flash": "#8172B3",
        "mask_knn": "#DD8452",
        "gather_knn": "#55A868",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel 1 (top-left): Backward time vs N at K=16
    ax = axes[0, 0]
    fixed_k = 16
    for method_name, color in palette.items():
        points = [row for row in rows if row["method"] == method_name and row["K"] == fixed_k]
        if not points:
            continue
        points_sorted = sorted(points, key=lambda row: row["N"])
        seq_lengths_filtered = [row["N"] for row in points_sorted]
        backward_ms_filtered = [row["backward_ms"] for row in points_sorted]
        ax.plot(seq_lengths_filtered, backward_ms_filtered, "o-",
                label=method_name, color=color, markersize=7, linewidth=2)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Backward time (ms)")
    ax.set_title(f"Backward Time vs N (K={fixed_k})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2 (top-right): Forward vs Backward breakdown at N=512, K=16
    ax = axes[0, 1]
    fixed_n_panel2, fixed_k_panel2 = 512, 16
    methods_for_plot = []
    forward_ms_values = []
    backward_ms_values = []
    ratio_values = []
    for method_name, color in palette.items():
        match = [row for row in rows if row["method"] == method_name
                 and row["N"] == fixed_n_panel2 and row["K"] == fixed_k_panel2]
        if match:
            methods_for_plot.append(method_name)
            forward_ms_values.append(match[0]["forward_ms"])
            backward_ms_values.append(match[0]["backward_ms"])
            ratio_values.append(match[0]["fwd_bwd_ratio"])
    x_positions = np.arange(len(methods_for_plot))
    bar_width = 0.35
    ax.bar(x_positions - bar_width / 2, forward_ms_values, bar_width,
           label="Forward", color="#3498db", alpha=0.85)
    ax.bar(x_positions + bar_width / 2, backward_ms_values, bar_width,
           label="Backward", color="#e74c3c", alpha=0.85)
    for bar_index, (method_name, ratio_val) in enumerate(
        zip(methods_for_plot, ratio_values)
    ):
        ax.text(bar_index, forward_ms_values[bar_index] + backward_ms_values[bar_index] + 2,
                f"r={ratio_val:.2f}", ha="center", va="bottom",
                fontsize=8, rotation=45)
    ax.set_xlabel("Method")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Forward vs Backward (N={fixed_n_panel2}, K={fixed_k_panel2})")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods_for_plot, rotation=25, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3 (bottom-left): Heatmap — Backward speedup vs dense_masked
    ax = axes[1, 0]
    dense_rows = [row for row in rows if row["method"] == "dense_masked"]
    dense_lookup = {(row["N"], row["K"]): row["backward_ms"] for row in dense_rows}
    method_names = ["dense_flash", "mask_knn", "gather_knn"]
    speedup_mat = np.full((len(sequence_lengths), len(method_names)), np.nan)
    for row_index, seq_len in enumerate(sequence_lengths):
        for col_index, method_name in enumerate(method_names):
            ref = dense_lookup.get((seq_len, 16), None)
            if ref is None or ref == 0:
                continue
            match = [row for row in rows if row["method"] == method_name
                     and row["N"] == seq_len and row["K"] == 16]
            if match:
                speedup_mat[row_index, col_index] = ref / match[0]["backward_ms"]
    sns.heatmap(speedup_mat, annot=True, fmt=".1f", cmap="RdYlGn", center=1.0,
                xticklabels=method_names,
                yticklabels=[f"N={seq_len}" for seq_len in sequence_lengths],
                ax=ax, cbar_kws={"label": "speedup vs dense_masked"})
    ax.set_title("Backward Speedup vs Dense Masked (K=16)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)

    # Panel 4 (bottom-right): Backward time vs K at N=512
    ax = axes[1, 1]
    fixed_n_panel4 = 512
    for method_name, color in palette.items():
        if method_name in ("dense_masked", "dense_flash"):
            continue
        points = [row for row in rows if row["method"] == method_name
                  and row["N"] == fixed_n_panel4]
        if not points:
            continue
        points_sorted = sorted(points, key=lambda row: row["K"])
        k_values_filtered = [row["K"] for row in points_sorted]
        backward_ms_filtered = [row["backward_ms"] for row in points_sorted]
        ax.plot(k_values_filtered, backward_ms_filtered, "o-",
                label=method_name, color=color, markersize=8, linewidth=2)
    ax.set_xlabel("K (number of neighbors)")
    ax.set_ylabel("Backward time (ms)")
    ax.set_title(f"Backward Time vs K (N={fixed_n_panel4})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Sparse Backward Pass Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = str(output_dir / "sparse_backward.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png_path}")


# ---------------------------------------------------------------------------
# Profiler mode
# ---------------------------------------------------------------------------

def run_profiler():
    """Run torch profiler for each method at N=512, K=16.

    Produces ``results/profile_results.csv`` with per-operator CUDA times.
    """
    from torch.profiler import profile, record_function, ProfilerActivity

    device = torch.device("cuda")
    dtype = torch.float16

    seq_len = 512
    num_neighbors = 16
    embed_dim = 320
    num_heads = 8

    input_tensor = torch.randn(1, seq_len, embed_dim, device=device, dtype=dtype)
    coords_3d = torch.randn(1, seq_len, 3, device=device, dtype=dtype)
    knn_indices = compute_knn_indices(coords_3d[..., 1:], num_neighbors)
    target_tensor = torch.randn_like(input_tensor)
    loss_fn = torch.nn.MSELoss()

    methods = {
        "dense_masked": lambda: RelativePositionalAttention(
            coord_dim=3, embed_dim=embed_dim, n_head=num_heads,
            cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
        ).to(device, dtype),
        "dense_flash": lambda: DenseFlashAttention(
            embed_dim=embed_dim, n_head=num_heads,
        ).to(device, dtype),
        "mask_knn": lambda: KNNMaskSparseAttention(
            embed_dim=embed_dim, n_head=num_heads,
            knn_neighbors=num_neighbors, mode="none",
        ).to(device, dtype),
        "gather_knn": lambda: GatherSparseAttention(
            embed_dim=embed_dim, n_head=num_heads,
            knn_neighbors=num_neighbors, mode="none",
        ).to(device, dtype),
    }

    all_profile_rows = []

    for method_name, build_mod in methods.items():
        print(f"\n  Profiling {method_name} ...")
        module = build_mod()
        module.train()

        for _ in range(3):
            if method_name == "dense_masked":
                output = module(input_tensor, input_tensor, input_tensor, coords_3d)
            elif method_name == "dense_flash":
                output = module(input_tensor, input_tensor, input_tensor)
            else:
                output = module(input_tensor, input_tensor, input_tensor, knn_indices)
            loss = loss_fn(output, target_tensor)
            loss.backward()
            module.zero_grad()
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            with record_function(method_name):
                if method_name == "dense_masked":
                    output = module(input_tensor, input_tensor, input_tensor, coords_3d)
                elif method_name == "dense_flash":
                    output = module(input_tensor, input_tensor, input_tensor)
                else:
                    output = module(input_tensor, input_tensor, input_tensor, knn_indices)
                loss = loss_fn(output, target_tensor)
                loss.backward()
            torch.cuda.synchronize()

        events = prof.key_averages()
        method_names_set = set(methods.keys())
        real_events = [event for event in events if event.key not in method_names_set]
        real_events.sort(
            key=lambda event: event.device_time_total if event.device_time_total is not None else 0,
            reverse=True,
        )

        print(f"  {'Operator':<55s} {'CUDA (us)':>12s} {'CPU (us)':>12s} {'Calls':>8s}")
        print(f"  {'-'*55} {'-'*12} {'-'*12} {'-'*8}")
        count = 0
        for event in real_events:
            if event.device_time_total is None or event.device_time_total == 0:
                continue
            if count >= 5:
                break
            cuda_time_us = event.device_time_total
            cpu_time_us = event.cpu_time_total if event.cpu_time_total is not None else 0
            calls = event.count
            print(f"  {event.key:<55s} {cuda_time_us:>12.0f} {cpu_time_us:>12.0f} {calls:>8d}")
            all_profile_rows.append({
                "method": method_name,
                "operator_name": event.key,
                "cuda_time_us": cuda_time_us,
                "cpu_time_us": cpu_time_us,
                "calls": calls,
            })
            count += 1

        del module
        gc.collect()
        torch.cuda.empty_cache()

    output_csv = Path("results/profile_results.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_profile_rows:
        with open(output_csv, "w", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=list(all_profile_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_profile_rows)
        print(f"\n  Saved profile CSV: {output_csv}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Parse arguments and dispatch to GPU benchmark or profiler."""
    parser = argparse.ArgumentParser(
        description="Sparse backward pass benchmark"
    )
    parser.add_argument("--gpu", action="store_true", help="Run GPU benchmark")
    parser.add_argument("--profile", action="store_true", help="Run torch profiler")
    parser.add_argument("--out", default="results/sparse_backward_results.csv",
                        help="Output CSV path")
    parser.add_argument("--outdir", default="results",
                        help="Output directory for figures")
    args = parser.parse_args()

    if not args.gpu and not args.profile:
        print("Specify --gpu for benchmarks or --profile for profiler analysis")
        return

    if args.gpu:
        rows, seq_lengths, k_sizes = run_gpu()
        save_csv(rows, args.out)
        generate_figures(rows, seq_lengths, k_sizes, args.outdir)

    if args.profile:
        run_profiler()


if __name__ == "__main__":
    main()
