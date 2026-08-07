"""Unified benchmark: all methods × all N, forward time + peak memory → CSV.

Usage:
    python benchmark_full.py [--d DPB] [--warmup N] [--rep N] [--out results.csv]
    python benchmark_full.py --d 320 --nhead 8 --out full_bench.csv

N = [128, 256, 512, 1024, 2048, 4096, 8192]
K = [4, 16] for methods that support it.

Methods:
    dense_masked   = RelativePositionalAttention (cdist per call, explicit N×N mask)
    dense_flash    = DenseFlashAttention (FlashAttn, no mask, no spatial cutoff)
    gather-KNN     = GatherSparseAttention (KNN indices → gather → SDPA on N×K)
    mask-KNN       = KNNMaskSparseAttention (KNN scatter → N×N mask → SDPA)
    NSA            = NSASparseAttention (3-path sparse attention)
    KNN-RelPos     = KNNRelativePositionalAttention (KNN + relative positional bias)
    MiniMax        = MiniMaxSparseAttention (blockwise sparse, index-branch + top-k blocks)

Note: CachedDistAttention is not in this standalone benchmark — it requires the main
Trackastra model's forward() which amortizes cdist across layers. The timing difference
is ~1.5× at L=12 (see separate CachedDist benchmark).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, csv, gc
import torch, torch.nn.functional as F
import torch.utils.benchmark as benchmark

from model_parts import (
    RelativePositionalAttention,
    KNNRelativePositionalAttention,
    GatherSparseAttention,
    KNNMaskSparseAttention,
    DenseFlashAttention,
    NSASparseAttention,
    MiniMaxSparseAttention,
)


def knn_indices(coords, knn_neighbors):
    """Compute K-nearest-neighbor indices from spatial coordinates.

    Args:
        coords: Coordinate tensor of shape (batch_size, seq_len, coord_dim).
        knn_neighbors: Number of nearest neighbors.

    Returns:
        KNN index tensor of shape (batch_size, seq_len, knn_neighbors).
    """
    yx = coords[..., 1:].float()
    dist = torch.cdist(yx, yx)
    _, knn = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)
    return knn


def measure(fn, warmup=5, min_run_time=0.3):
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
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    _ = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    memory_mb = (peak - baseline) / (1024 ** 2)
    timer = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    t_mean = timer.blocked_autorange(min_run_time=min_run_time).mean
    return t_mean, memory_mb


def run(device, dtype, d_model, n_head, coord_dim, Ns, Ks, mode, dist_mode, warmup, rep, seed=42):
    """Run the full benchmark sweep across all methods and sequence lengths.

    For each N in Ns, benchmarks: dense_masked, dense_flash, gather-KNN,
    mask-KNN, NSA, KNN-RelPos, and MiniMax with various block sizes.

    Args:
        device: torch device.
        dtype: torch dtype.
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        Ns: List of sequence lengths to benchmark.
        Ks: List of KNN neighbor counts.
        mode: Positional encoding mode ("none", "bias", "rope").
        dist_mode: Distance decay mode ("v0" or "v1").
        warmup: Number of warmup iterations.
        rep: Minimum run time in milliseconds for the benchmark timer.
        seed: Random seed for reproducibility.

    Returns:
        List of result rows [method, N, time_ms, memory_mb, error?].
    """
    rows = []

    for N in Ns:
        torch.manual_seed(seed)
        x = torch.randn(1, N, d_model, device=device, dtype=dtype)
        coords = torch.randn(1, N, coord_dim + 1, device=device, dtype=dtype)
        coords[..., 0] *= 4

        knn_idx = {K: knn_indices(coords, K) for K in Ks}

        # (A) dense_masked
        try:
            attn = RelativePositionalAttention(
                coord_dim, d_model, n_head, mode=mode, attn_dist_mode=dist_mode,
            ).to(device, dtype)
            time_s, memory_mb = measure(lambda: attn(x, x, x, coords), warmup=warmup, min_run_time=rep / 1000)
            rows.append(["dense_masked", N, time_s * 1000, memory_mb])
        except RuntimeError as e:
            rows.append(["dense_masked", N, None, None, str(e)[:120]])

        # (B) dense_flash
        try:
            attn = DenseFlashAttention(d_model, n_head).to(device, dtype)
            time_s, memory_mb = measure(lambda: attn(x, x, x), warmup=warmup, min_run_time=rep / 1000)
            rows.append(["dense_flash", N, time_s * 1000, memory_mb])
        except RuntimeError as e:
            rows.append(["dense_flash", N, None, None, str(e)[:120]])

        # (C) gather-KNN / mask-KNN for each K
        for K in Ks:
            kidx = knn_idx[K]
            for cls, label in [
                (GatherSparseAttention, "gather-KNN"),
                (KNNMaskSparseAttention, "mask-KNN"),
            ]:
                try:
                    attn = cls(d_model, n_head, knn_neighbors=K,
                               coord_dim=coord_dim, mode=mode).to(device, dtype)
                    time_s, memory_mb = measure(
                        lambda: attn(x, x, x, knn_indices=kidx, coords=coords),
                        warmup=warmup, min_run_time=rep / 1000,
                    )
                    rows.append([f"{label}_K={K}", N, time_s * 1000, memory_mb])
                except RuntimeError as e:
                    rows.append([f"{label}_K={K}", N, None, None, str(e)[:120]])

        # (D) NSA
        try:
            attn = NSASparseAttention(d_model, n_head).to(device, dtype)
            time_s, memory_mb = measure(lambda: attn(x, x, x), warmup=warmup, min_run_time=rep / 1000)
            rows.append(["NSA", N, time_s * 1000, memory_mb])
        except RuntimeError as e:
            rows.append(["NSA", N, None, None, str(e)[:120]])

        # (E) KNN-RelPos
        for K in Ks:
            try:
                attn = KNNRelativePositionalAttention(
                    coord_dim, d_model, n_head, knn_neighbors=K,
                    mode=mode, attn_dist_mode=dist_mode,
                ).to(device, dtype)
                time_s, memory_mb = measure(
                    lambda: attn(x, x, x, coords, knn_indices=knn_idx[K]),
                    warmup=warmup, min_run_time=rep / 1000,
                )
                rows.append([f"KNN-RelPos_K={K}", N, time_s * 1000, memory_mb])
            except RuntimeError as e:
                rows.append([f"KNN-RelPos_K={K}", N, None, None, str(e)[:120]])

        # (F) MiniMax — blockwise sparse attention
        for Bk in [32, 64, 128]:
            ksel = max(1, min(4, (N + Bk - 1) // Bk - 1))
            try:
                attn = MiniMaxSparseAttention(
                    d_model, n_head, block_size=Bk,
                    num_selected_blocks=ksel, mode="none",
                ).to(device, dtype)
                time_s, memory_mb = measure(
                    lambda: attn(x, x, x),
                    warmup=warmup, min_run_time=rep / 1000,
                )
                rows.append([f"MiniMax_Bk={Bk}", N, time_s * 1000, memory_mb])
            except RuntimeError as e:
                rows.append([f"MiniMax_Bk={Bk}", N, None, None, str(e)[:120]])

        print(f"  N={N} done", flush=True)

    return rows


def main():
    """Parse CLI arguments and run the full benchmark suite.

    Sweeps over N=[128..8192] and K=[4,16,32,64,128], comparing all
    attention methods. Results are written to the specified CSV file
    and a summary table is printed to stdout.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--out", default="benchmark_full_results.csv")
    parser.add_argument("--mode", default="none", choices=["none", "bias", "rope"])
    parser.add_argument("--dist-mode", default="v1", choices=["v0", "v1"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda"); dtype = torch.float16
    Ns = [128, 256, 512, 1024, 2048, 4096, 8192]; Ks = [4, 16, 32, 64, 128]

    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"flash={torch.backends.cuda.flash_sdp_enabled()}")
    print(f"N = {Ns}  K = {Ks}\n")

    rows = run(device, dtype, args.d, args.nhead, 2, Ns, Ks, args.mode, args.dist_mode,
               args.warmup, args.rep, seed=args.seed)

    header = ["method", "N", "time_ms", "memory_mb", "error"]
    with open(args.out, "w", newline="") as file_handle:
        w = csv.writer(file_handle); w.writerow(header)
        for row in rows:
            while len(row) < len(header): row.append("")
            w.writerow(row)

    print(f"\nWrote {len(rows)} rows → {args.out}")

    methods = sorted(set(row[0] for row in rows))
    print(f"\n{'method':>26s}", end="")
    for N in Ns: print(f"  N={N:>4d}", end="")
    print()
    for m in methods:
        print(f"{m:>26s}", end="")
        for N in Ns:
            matching_rows = [row for row in rows if row[0] == m and row[1] == N]
            if matching_rows and matching_rows[0][2] is not None:
                print(f" {matching_rows[0][2]:7.2f}", end="")
            else:
                tag = "OOM" if matching_rows and matching_rows[0][4] and "out of memory" in str(matching_rows[0][4]).lower() else "ERR"
                print(f" {tag:>7s}", end="")
        print()


if __name__ == "__main__":
    main()
