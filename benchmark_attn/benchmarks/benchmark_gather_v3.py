"""GatherSparseAttention V3 — manual-matmul gather benchmark.

V3 (model_parts.py:583) replaces F.scaled_dot_product_attention on the flattened
(B*N, nH, 1, Dh) tensors with native-shape torch.matmul, avoiding the SDPA kernel
launch overhead for q_len=1. This script measures V3 against V1 (SDPA gather) and
V2 (spatial reorder) for forward time + peak memory on the A500.

Usage:
    python benchmark_gather_v3.py [--d 320] [--nhead 8] [--warmup 5]
        [--rep 30] [--out benchmark_attn/gather_v3_results.csv]
        [--Ns 128,256,512,1024,2048,4096,8192] [--Ks 4,16,64]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, csv, gc
import torch
import torch.utils.benchmark as benchmark

from model_parts import (
    GatherSparseAttention,   # V1
    GatherSparseAttentionV2,  # V2 (reorder)
    GatherSparseAttentionV3,  # V3 (manual matmul)
)


def knn_indices(coords, knn_neighbors):
    """Compute K-nearest-neighbor indices from spatial coordinates.

    Args:
        coords: Coordinate tensor of shape (B, N, coord_dim).
        K: Number of nearest neighbors.

    Returns:
        KNN index tensor of shape (B, N, K).
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


def run(device, dtype, d_model, n_head, coord_dim, Ns, Ks, warmup, rep, seed=42):
    """Run the V1/V2/V3 gather-sparse attention benchmark.

    For each N in Ns and K in Ks, benchmarks three variants:
      - V1: GatherSparseAttention (SDPA on flattened 1×K tensors)
      - V2: GatherSparseAttentionV2 (optimised, fewer copies)
      - V3: GatherSparseAttentionV3 (manual matmul, no SDPA)

    Args:
        device: torch device.
        dtype: torch dtype.
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        Ns: List of sequence lengths.
        Ks: List of KNN neighbor counts.
        warmup: Number of warmup iterations.
        rep: Minimum run time in milliseconds for the benchmark timer.

    Returns:
        List of result rows [method, N, K, time_ms, memory_mb, error?].
    """
    rows = []
    print(f"{'N':>5} | {'K':>3} | {'V1 gather':>14} {'mem':>8} | {'V2 reorder':>14} {'mem':>8} | "
          f"{'V3 matmul':>14} {'mem':>8} | {'V3/V1':>8} | {'V3/V2':>8}")
    print("-" * 115)

    for N in Ns:
        torch.manual_seed(seed)
        x = torch.randn(1, N, d_model, device=device, dtype=dtype)
        coords = torch.randn(1, N, coord_dim + 1, device=device, dtype=dtype)
        coords[..., 0] *= 4

        for K in Ks:
            if K > N:
                continue
            kidx = knn_indices(coords, K)
            row = {}
            for label, cls, kwargs in [
                ("V1", GatherSparseAttention, {}),
                ("V2", GatherSparseAttentionV2, {}),
                ("V3", GatherSparseAttentionV3, {}),
            ]:
                try:
                    attn = cls(d_model, n_head, knn_neighbors=K,
                               coord_dim=coord_dim, mode="none").to(device, dtype)
                    time_s, memory_mb = measure(
                        lambda: attn(x, x, x, knn_indices=kidx, coords=coords),
                        warmup=warmup, min_run_time=rep / 1000,
                    )
                    row[label] = (time_s * 1000, memory_mb)
                    rows.append([f"gather-{label}", N, K, time_s * 1000, memory_mb])
                except RuntimeError as e:
                    rows.append([f"gather-{label}", N, K, None, None, str(e)[:120]])
                    row[label] = None

            v1, v2, v3 = row.get("V1"), row.get("V2"), row.get("V3")
            r31 = (f"{v3[0]/v1[0]:>6.2f}x" if v3 and v1 and v1[0] > 0 else "   --  ")
            r32 = (f"{v3[0]/v2[0]:>6.2f}x" if v3 and v2 and v2[0] > 0 else "   --  ")
            fmt = lambda v: (f"{v[0]:>9.2f}ms {v[1]:>6.1f}MB" if v else f"{'OOM/ERR':>18}")
            print(f"{N:>5} | {K:>3} | {fmt(v1)} | {fmt(v2)} | {fmt(v3)} | {r31} | {r32}",
                  flush=True)

    return rows


def main():
    """Parse CLI arguments and run the V1/V2/V3 gather benchmark.

    Sweeps N=[128..8192] with K=[4,16,64], comparing three
    GatherSparseAttention variants for forward time and peak memory.
    Results are written to a CSV file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "gather_v3_results.csv"))
    parser.add_argument("--Ns", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--Ks", default="4,16,64")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda"); dtype = torch.float16
    Ns = [int(seq_len) for seq_len in args.Ns.split(",")]
    Ks = [int(k) for k in args.Ks.split(",")]

    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"flash={torch.backends.cuda.flash_sdp_enabled()}")
    print(f"N = {Ns}  K = {Ks}\n")

    rows = run(device, dtype, args.d, args.nhead, 2, Ns, Ks, args.warmup, args.rep, seed=args.seed)

    header = ["method", "N", "K", "time_ms", "memory_mb", "error"]
    with open(args.out, "w", newline="") as file_handle:
        w = csv.writer(file_handle); w.writerow(header)
        for row in rows:
            while len(row) < len(header): row.append("")
            w.writerow(row)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
