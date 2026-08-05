"""Targeted benchmark: small N (128,256,512) — all variants at K=16 L=1.
Measures forward-only time and memory. Outputs CSV + text summary.
"""

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


def bench_dense(N, B, d, h, coord_dim, device, dtype):
    layer = RelativePositionalAttention(
        coord_dim=coord_dim, embed_dim=d, n_head=h,
        cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
    ).to(device).to(dtype)
    q = torch.randn(B, N, d, device=device, dtype=dtype)
    coords = torch.randn(B, N, coord_dim, device=device, dtype=dtype)

    def fn():
        return layer(q, q, q, coords)
    return fn


def bench_dense_flash(N, B, d, h, coord_dim, device, dtype):
    layer = DenseFlashAttention(embed_dim=d, n_head=h).to(device).to(dtype)
    q = torch.randn(B, N, d, device=device, dtype=dtype)

    def fn():
        return layer(q, q, q)
    return fn


def bench_sparse_v1(N, K, B, d, h, coord_dim, device, dtype):
    layer = GatherSparseAttention(embed_dim=d, n_head=h, knn_neighbors=K, mode="none").to(device).to(dtype)
    q = torch.randn(B, N, d, device=device, dtype=dtype)
    coords = torch.randn(B, N, coord_dim, device=device, dtype=dtype)
    yx = coords[..., 1:]
    dist = torch.cdist(yx, yx)
    _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)

    def fn():
        return layer(q, q, q, knn_idx, coords)
    return fn


def bench_sparse_v2(N, K, B, d, h, coord_dim, device, dtype):
    layer = GatherSparseAttentionV2(embed_dim=d, n_head=h, knn_neighbors=K, mode="none").to(device).to(dtype)
    q = torch.randn(B, N, d, device=device, dtype=dtype)
    coords = torch.randn(B, N, coord_dim, device=device, dtype=dtype)
    yx = coords[..., 1:]
    dist = torch.cdist(yx, yx)
    _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)

    def fn():
        return layer(q, q, q, knn_idx, coords)
    return fn


def measure(fn, warmup=10, min_run_time=1.0):
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

    t = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    result = t.blocked_autorange(min_run_time=min_run_time)

    return result.mean, mem_mb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_properties(0).name}, mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    B = 2
    d = 256
    h = 4
    coord_dim = 3
    K = 16
    N_vals = [128, 256, 512]
    n_repeat = 5  # number of timing runs

    configs = [
        ("dense", bench_dense),
        ("dense_flash", bench_dense_flash),
        ("sparse_v1", bench_sparse_v1),
        ("sparse_v2", bench_sparse_v2),
    ]

    out_csv = "benchmark_small_n_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "L", "N", "K", "time_s", "mem_mb", "speedup_vs_dense"])

        for N in N_vals:
            print(f"\n{'='*60}")
            print(f"  N = {N}")
            print(f"{'='*60}")

            # Collect all timing data first, then compute speedup
            results = {}

            for method_name, bench_fn in configs:
                if method_name == "sparse_v1" or method_name == "sparse_v2":
                    fn = bench_fn(N, K, B, d, h, coord_dim, device, dtype)
                elif method_name == "dense":
                    fn = bench_fn(N, B, d, h, coord_dim, device, dtype)
                elif method_name == "dense_flash":
                    fn = bench_fn(N, B, d, h, coord_dim, device, dtype)

                # Multiple measurements
                times = []
                for _ in range(n_repeat):
                    t, mem = measure(fn, warmup=3, min_run_time=0.3)
                    times.append(t)
                mean_t = sum(times) / len(times)
                results[method_name] = (mean_t, mem)

            # Print and write results
            dense_t = results["dense"][0]
            for method_name, (t, mem) in results.items():
                speedup = dense_t / t if t > 0 else 0
                print(f"  {method_name:<14s}  {t*1e6:8.1f} us  {mem:6.1f} MB  speedup={speedup:.2f}x")
                w.writerow([method_name, 1, N, K, f"{t:.9f}", f"{mem:.1f}", f"{speedup:.4f}"])
                f.flush()

    print(f"\nResults saved to: {out_csv}")

    # Quick summary
    print(f"\n{'='*60}")
    print("  SUMMARY: Speedup vs dense (dense masked SDPA)")
    print(f"{'='*60}")
    with open(out_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["method"] == "dense":
                continue
            print(f"  {row['method']:<14s} N={int(row['N']):<5}  {float(row['speedup_vs_dense']):.3f}x")


if __name__ == "__main__":
    main()
