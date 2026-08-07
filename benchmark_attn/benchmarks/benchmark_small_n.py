"""Targeted benchmark: small N (128,256,512) — all variants at K=16 L=1.
Measures forward-only time and memory. Outputs CSV + text summary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.utils.benchmark as benchmark
import gc
import csv
import time

from model_parts import (
    RelativePositionalAttention,
    GatherSparseAttention,
    GatherSparseAttentionV2,
    DenseFlashAttention,
)


def bench_dense(seq_len, batch_size, embed_dim, n_head, coord_dim, device, dtype):
    """Build a closure that runs RelativePositionalAttention (dense masked).

    Args:
        seq_len: Sequence length.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        device: torch device.
        dtype: torch dtype.

    Returns:
        A callable ``fn()`` that runs the forward pass.
    """
    layer = RelativePositionalAttention(
        coord_dim=coord_dim, embed_dim=embed_dim, n_head=n_head,
        cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
    ).to(device).to(dtype)
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)

    def fn():
        return layer(query, query, query, coords)
    return fn


def bench_dense_flash(seq_len, batch_size, embed_dim, n_head, coord_dim, device, dtype):
    """Build a closure that runs DenseFlashAttention (no mask, no KNN).

    Args:
        seq_len: Sequence length.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Unused; kept for interface consistency.
        device: torch device.
        dtype: torch dtype.

    Returns:
        A callable ``fn()`` that runs the forward pass.
    """
    layer = DenseFlashAttention(embed_dim=embed_dim, n_head=n_head).to(device).to(dtype)
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn():
        return layer(query, query, query)
    return fn


def bench_sparse_v1(seq_len, knn_neighbors, batch_size, embed_dim, n_head, coord_dim, device, dtype):
    """Build a closure that runs GatherSparseAttention (V1: SDPA gather).

    Args:
        seq_len: Sequence length.
        knn_neighbors: Number of KNN neighbors.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        device: torch device.
        dtype: torch dtype.

    Returns:
        A callable ``fn()`` that runs the forward pass.
    """
    layer = GatherSparseAttention(embed_dim=embed_dim, n_head=n_head, knn_neighbors=knn_neighbors, mode="none").to(device).to(dtype)
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)
    yx = coords[..., 1:]
    dist = torch.cdist(yx, yx)
    _, knn_idx = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)

    def fn():
        return layer(query, query, query, knn_idx, coords)
    return fn


def bench_sparse_v2(seq_len, knn_neighbors, batch_size, embed_dim, n_head, coord_dim, device, dtype):
    """Build a closure that runs GatherSparseAttentionV2 (optimised gather).

    V2 eliminates unnecessary copies via view+unsqueeze for flat_query.

    Args:
        seq_len: Sequence length.
        knn_neighbors: Number of KNN neighbors.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        device: torch device.
        dtype: torch dtype.

    Returns:
        A callable ``fn()`` that runs the forward pass.
    """
    layer = GatherSparseAttentionV2(embed_dim=embed_dim, n_head=n_head, knn_neighbors=knn_neighbors, mode="none").to(device).to(dtype)
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)
    yx = coords[..., 1:]
    dist = torch.cdist(yx, yx)
    _, knn_idx = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)

    def fn():
        return layer(query, query, query, knn_idx, coords)
    return fn


def measure(fn, warmup=10, min_run_time=1.0):
    """Benchmark a callable, returning (mean_time_s, incremental_peak_mem_mb).

    Args:
        fn: Callable to benchmark.
        warmup: Number of warmup iterations.
        min_run_time: Minimum run time for the benchmark timer.

    Returns:
        Tuple (mean_time_s, peak_memory_mb).
    """
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        _ = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        mem_mb = (peak - baseline) / (1024 ** 2)
    else:
        mem_mb = 0.0

    timer = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    result = timer.blocked_autorange(min_run_time=min_run_time)

    return result.mean, mem_mb


def main():
    """Run small-N benchmark (N=128,256,512) for all attention variants.

    Compares dense, dense_flash, sparse_v1, and sparse_v2 at K=16.
    Writes results to ``benchmark_small_n_results.csv`` and prints
    a speedup summary.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_properties(0).name}, mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    batch_size = 2
    embed_dim = 256
    n_head = 4
    coord_dim = 3
    knn_neighbors = 16
    seq_len_vals = [128, 256, 512]
    n_repeat = 5  # number of timing runs

    configs = [
        ("dense", bench_dense),
        ("dense_flash", bench_dense_flash),
        ("sparse_v1", bench_sparse_v1),
        ("sparse_v2", bench_sparse_v2),
    ]

    out_csv = "benchmark_small_n_results.csv"
    with open(out_csv, "w", newline="") as file_handle:
        w = csv.writer(file_handle)
        w.writerow(["method", "L", "N", "K", "time_s", "mem_mb", "speedup_vs_dense"])

        for seq_len in seq_len_vals:
            print(f"\n{'='*60}")
            print(f"  N = {seq_len}")
            print(f"{'='*60}")

            # Collect all timing data first, then compute speedup
            results = {}

            for method_name, bench_fn in configs:
                if method_name == "sparse_v1" or method_name == "sparse_v2":
                    fn = bench_fn(seq_len, knn_neighbors, batch_size, embed_dim, n_head, coord_dim, device, dtype)
                elif method_name == "dense":
                    fn = bench_fn(seq_len, batch_size, embed_dim, n_head, coord_dim, device, dtype)
                elif method_name == "dense_flash":
                    fn = bench_fn(seq_len, batch_size, embed_dim, n_head, coord_dim, device, dtype)

                # Multiple measurements
                times = []
                for _ in range(n_repeat):
                    time_s, memory_mb = measure(fn, warmup=3, min_run_time=0.3)
                    times.append(time_s)
                mean_t = sum(times) / len(times)
                results[method_name] = (mean_t, memory_mb)

            # Print and write results
            dense_t = results["dense"][0]
            for method_name, (time_s, memory_mb) in results.items():
                speedup = dense_t / time_s if time_s > 0 else 0
                print(f"  {method_name:<14s}  {time_s*1e6:8.1f} us  {memory_mb:6.1f} MB  speedup={speedup:.2f}x")
                w.writerow([method_name, 1, seq_len, knn_neighbors, f"{time_s:.9f}", f"{memory_mb:.1f}", f"{speedup:.4f}"])
                file_handle.flush()

    print(f"\nResults saved to: {out_csv}")

    # Quick summary
    print(f"\n{'='*60}")
    print("  SUMMARY: Speedup vs dense (dense masked SDPA)")
    print(f"{'='*60}")
    with open(out_csv, "r") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            if row["method"] == "dense":
                continue
            print(f"  {row['method']:<14s} N={int(row['N']):<5}  {float(row['speedup_vs_dense']):.3f}x")


if __name__ == "__main__":
    main()
