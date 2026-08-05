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


def run(device, dtype, d_model, n_head, coord_dim, Ns, Ks, mode, dist_mode, warmup, rep):
    rows = []

    for N in Ns:
        torch.manual_seed(42)
        x = torch.randn(1, N, d_model, device=device, dtype=dtype)
        coords = torch.randn(1, N, coord_dim + 1, device=device, dtype=dtype)
        coords[..., 0] *= 4

        knn_idx = {K: knn_indices(coords, K) for K in Ks}

        # (A) dense_masked
        try:
            attn = RelativePositionalAttention(
                coord_dim, d_model, n_head, mode=mode, attn_dist_mode=dist_mode,
            ).to(device, dtype)
            t, mem = measure(lambda: attn(x, x, x, coords), warmup=warmup, min_run_time=rep / 1000)
            rows.append(["dense_masked", N, t * 1000, mem])
        except RuntimeError as e:
            rows.append(["dense_masked", N, None, None, str(e)[:120]])

        # (B) dense_flash
        try:
            attn = DenseFlashAttention(d_model, n_head).to(device, dtype)
            t, mem = measure(lambda: attn(x, x, x), warmup=warmup, min_run_time=rep / 1000)
            rows.append(["dense_flash", N, t * 1000, mem])
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
                    t, mem = measure(
                        lambda: attn(x, x, x, knn_indices=kidx, coords=coords),
                        warmup=warmup, min_run_time=rep / 1000,
                    )
                    rows.append([f"{label}_K={K}", N, t * 1000, mem])
                except RuntimeError as e:
                    rows.append([f"{label}_K={K}", N, None, None, str(e)[:120]])

        # (D) NSA
        try:
            attn = NSASparseAttention(d_model, n_head).to(device, dtype)
            t, mem = measure(lambda: attn(x, x, x), warmup=warmup, min_run_time=rep / 1000)
            rows.append(["NSA", N, t * 1000, mem])
        except RuntimeError as e:
            rows.append(["NSA", N, None, None, str(e)[:120]])

        # (E) KNN-RelPos
        for K in Ks:
            try:
                attn = KNNRelativePositionalAttention(
                    coord_dim, d_model, n_head, knn_neighbors=K,
                    mode=mode, attn_dist_mode=dist_mode,
                ).to(device, dtype)
                t, mem = measure(
                    lambda: attn(x, x, x, coords, knn_indices=knn_idx[K]),
                    warmup=warmup, min_run_time=rep / 1000,
                )
                rows.append([f"KNN-RelPos_K={K}", N, t * 1000, mem])
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
                t, mem = measure(
                    lambda: attn(x, x, x),
                    warmup=warmup, min_run_time=rep / 1000,
                )
                rows.append([f"MiniMax_Bk={Bk}", N, t * 1000, mem])
            except RuntimeError as e:
                rows.append([f"MiniMax_Bk={Bk}", N, None, None, str(e)[:120]])

        print(f"  N={N} done", flush=True)

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=320)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rep", type=int, default=30)
    p.add_argument("--out", default="benchmark_full_results.csv")
    p.add_argument("--mode", default="none", choices=["none", "bias", "rope"])
    p.add_argument("--dist-mode", default="v1", choices=["v0", "v1"])
    args = p.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda"); dtype = torch.float16
    Ns = [128, 256, 512, 1024, 2048, 4096, 8192]; Ks = [4, 16, 32, 64, 128]

    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"flash={torch.backends.cuda.flash_sdp_enabled()}")
    print(f"N = {Ns}  K = {Ks}\n")

    rows = run(device, dtype, args.d, args.nhead, 2, Ns, Ks, args.mode, args.dist_mode,
               args.warmup, args.rep)

    header = ["method", "N", "time_ms", "memory_mb", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            while len(r) < len(header): r.append("")
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows → {args.out}")

    methods = sorted(set(r[0] for r in rows))
    print(f"\n{'method':>26s}", end="")
    for N in Ns: print(f"  N={N:>4d}", end="")
    print()
    for m in methods:
        print(f"{m:>26s}", end="")
        for N in Ns:
            r = [row for row in rows if row[0] == m and row[1] == N]
            if r and r[0][2] is not None:
                print(f" {r[0][2]:7.2f}", end="")
            else:
                tag = "OOM" if r and r[0][4] and "out of memory" in str(r[0][4]).lower() else "ERR"
                print(f" {tag:>7s}", end="")
        print()


if __name__ == "__main__":
    main()
