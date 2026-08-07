"""CachedDistAttention — real measurement of the dense attention baseline.

The report claims a "~2x speedup" for CachedDistAttention over the per-layer
RelativePositionalAttention baseline by computing the 2D pairwise distance
matrix once and sharing it across all L transformer layers. Until now this was
only backed by an *analytical* model (benchmark_knn_methods.py, cached_dense).
This script measures the REAL classes:

  dense_masked  = RelativePositionalAttention — per-layer 2D cdist
  cached_dist   = CachedDistAttention          — uses precomputed dist_2d
  cdist_2d      = the one-time 2D torch.cdist cost (amortized once per model)

It reports per-layer forward time and peak memory for N = [128..8192], then
computes the L-layer totals for a full model (L default 12: 6 enc + 6 dec):

  total_baseline = L * t_dense_masked
  total_cached   = t_cdist_2d + L * t_cached_dist

Usage:
    python benchmark_cached_dist.py [--d 320] [--nhead 8] [--warmup 5]
        [--rep 30] [--out benchmark_attn/cached_dist_results.csv]
        [--layers 12] [--Ns 128,256,512,1024,2048,4096,8192]
        [--cutoff 256] [--dist-mode v1]

N=8192 (fp16, mask N*N) exceeds the A500 4 GB VRAM and is recorded as OOM.
Both attention classes are self-contained copies in the local model_parts.py
(no trackastra dependency).
"""

import argparse, csv, gc, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_parts import CachedDistAttention, RelativePositionalAttention


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
    time_s = timer.blocked_autorange(min_run_time=min_run_time).mean
    return time_s, memory_mb


def run(device, dtype, d_model, n_head, coord_dim, Ns, warmup, rep,
        cutoff_spatial, dist_mode, layers, seed=42):
    """Run CachedDistAttention vs RelativePositionalAttention benchmark.

    For each N in Ns, benchmarks:
      - dense_masked: RelativePositionalAttention (per-layer cdist)
      - cached_dist: CachedDistAttention (precomputed dist_2d)
      - cdist_2d: one-time 2D cdist cost (amortized once per model forward)

    Args:
        device: torch device.
        dtype: torch dtype.
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        Ns: List of sequence lengths.
        warmup: Number of warmup iterations.
        rep: Minimum run time in milliseconds.
        cutoff_spatial: Spatial cutoff distance.
        dist_mode: Distance decay mode ("v0" or "v1").
        layers: Number of transformer layers for amortized total.
        seed: Random seed for reproducibility.

    Returns:
        List of result rows [method, N, time_ms, memory_mb, error?].
    """
    rows = []
    print(f"{'N':>5} | {'dense_masked':>14} {'mem':>8} | {'cached_dist':>13} {'mem':>8} | {'cdist_2d':>10} | {'speedup':>8} (per-layer) | {'total L={}':>8}"
          .format(layers))
    print("-" * 110)

    for N in Ns:
        torch.manual_seed(seed)
        x = torch.randn(1, N, d_model, device=device, dtype=dtype)
        coords = torch.randn(1, N, coord_dim + 1, device=device, dtype=dtype)
        coords[..., 0] *= 4
        dist_2d = torch.cdist(coords[..., 1:].float(), coords[..., 1:].float())
        dist_2d_m = dist_2d.to(dtype)

        # (A) dense_masked = RelativePositionalAttention (per-layer cdist)
        time_s_dense = time_s_cached = time_s_cdist = None
        memory_mb_dense = memory_mb_cached = None
        try:
            attn = RelativePositionalAttention(
                coord_dim, d_model, n_head, cutoff_spatial=cutoff_spatial,
                mode="none", attn_dist_mode=dist_mode,
            ).to(device, dtype)
            time_s, memory_mb = measure(lambda: attn(x, x, x, coords), warmup=warmup, min_run_time=rep / 1000)
            time_s_dense, memory_mb_dense = time_s, memory_mb
            rows.append(["dense_masked", N, time_s * 1000, memory_mb])
        except RuntimeError as e:
            rows.append(["dense_masked", N, None, None, str(e)[:120]])

        # (B) cached_dist = CachedDistAttention with precomputed dist_2d
        try:
            attn = CachedDistAttention(
                coord_dim, d_model, n_head, cutoff_spatial=cutoff_spatial,
                mode="none", attn_dist_mode=dist_mode,
            ).to(device, dtype)
            time_s, memory_mb = measure(
                lambda: attn(x, x, x, coords, dist_2d=dist_2d_m),
                warmup=warmup, min_run_time=rep / 1000,
            )
            time_s_cached, memory_mb_cached = time_s, memory_mb
            rows.append(["cached_dist", N, time_s * 1000, memory_mb])
        except RuntimeError as e:
            rows.append(["cached_dist", N, None, None, str(e)[:120]])

        # (C) one-time 2D cdist cost (amortized once per full model forward)
        try:
            time_s, _ = measure(
                lambda: torch.cdist(coords[..., 1:].float(), coords[..., 1:].float()),
                warmup=warmup, min_run_time=rep / 1000,
            )
            time_s_cdist = time_s
            rows.append(["cdist_2d", N, time_s * 1000, None])
        except RuntimeError as e:
            rows.append(["cdist_2d", N, None, None, str(e)[:120]])

        if time_s_dense is not None and time_s_cached is not None:
            per = time_s_dense / time_s_cached if time_s_cached > 0 else float("nan")
            total_baseline = layers * time_s_dense
            total_cached = (time_s_cdist or 0) + layers * time_s_cached
            tot = total_baseline / total_cached if total_cached > 0 else float("nan")
            print(f"{N:>5} | {time_s_dense*1000:>9.2f}ms {memory_mb_dense or 0:>6.1f}MB | "
                  f"{time_s_cached*1000:>9.2f}ms {memory_mb_cached or 0:>6.1f}MB | "
                  f"{(time_s_cdist or 0)*1000:>7.2f}ms | {per:>6.2f}x | L={layers} total {tot:>6.2f}x",
                  flush=True)
        else:
            errs = [row[4] for row in rows[-2:] if len(row) > 4 and row[4]]
            tag = "OOM" if any("out of memory" in str(e).lower() for e in errs) else "ERR"
            print(f"{N:>5} | {tag}", flush=True)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "cached_dist_results.csv"))
    parser.add_argument("--layers", type=int, default=12,
                        help="encoder+decoder layers for the amortized total (default 12)")
    parser.add_argument("--Ns", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--cutoff", type=float, default=256)
    parser.add_argument("--dist-mode", default="v1", choices=["v0", "v1"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda"); dtype = torch.float16
    Ns = [int(n) for n in args.Ns.split(",")]

    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"flash={torch.backends.cuda.flash_sdp_enabled()}  layers={args.layers}")
    print(f"N = {Ns}  cutoff_spatial={args.cutoff}  dist_mode={args.dist_mode}\n")

    rows = run(device, dtype, args.d, args.nhead, 2, Ns, args.warmup,
               args.rep, args.cutoff, args.dist_mode, args.layers, seed=args.seed)

    header = ["method", "N", "time_ms", "memory_mb", "error"]
    with open(args.out, "w", newline="") as file_handle:
        w = csv.writer(file_handle); w.writerow(header)
        for row in rows:
            while len(row) < len(header): row.append("")
            w.writerow(row)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
