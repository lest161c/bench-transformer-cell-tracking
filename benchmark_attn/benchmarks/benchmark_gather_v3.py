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


def knn_indices(coords, K):
    yx = coords[..., 1:].float()
    dist = torch.cdist(yx, yx)
    _, knn = torch.topk(dist, k=K, dim=-1, largest=False)
    return knn


def measure(fn, warmup=5, min_run_time=0.3):
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
    mem = (peak - baseline) / (1024 ** 2)
    t = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    t_mean = t.blocked_autorange(min_run_time=min_run_time).mean
    return t_mean, mem


def run(device, dtype, d_model, n_head, coord_dim, Ns, Ks, warmup, rep):
    rows = []
    print(f"{'N':>5} | {'K':>3} | {'V1 gather':>14} {'mem':>8} | {'V2 reorder':>14} {'mem':>8} | "
          f"{'V3 matmul':>14} {'mem':>8} | {'V3/V1':>8} | {'V3/V2':>8}")
    print("-" * 115)

    for N in Ns:
        torch.manual_seed(42)
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
                    t, mem = measure(
                        lambda: attn(x, x, x, knn_indices=kidx, coords=coords),
                        warmup=warmup, min_run_time=rep / 1000,
                    )
                    row[label] = (t * 1000, mem)
                    rows.append([f"gather-{label}", N, K, t * 1000, mem])
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
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=320)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=30)
    p.add_argument("--out", default="benchmark_attn/gather_v3_results.csv")
    p.add_argument("--Ns", default="128,256,512,1024,2048,4096,8192")
    p.add_argument("--Ks", default="4,16,64")
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda"); dtype = torch.float16
    Ns = [int(n) for n in args.Ns.split(",")]
    Ks = [int(k) for k in args.Ks.split(",")]

    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"flash={torch.backends.cuda.flash_sdp_enabled()}")
    print(f"N = {Ns}  K = {Ks}\n")

    rows = run(device, dtype, args.d, args.nhead, 2, Ns, Ks, args.warmup, args.rep)

    header = ["method", "N", "K", "time_ms", "memory_mb", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            while len(r) < len(header): r.append("")
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
