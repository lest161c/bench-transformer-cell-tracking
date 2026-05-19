"""Standalone benchmark: dense masked SDPA vs gather-based sparse attention.
Expanded sweep: N up to 16384, K up to 128, with/without spatial reorder.
OOM → marked in CSV.
"""

import torch
import torch.utils.benchmark as benchmark
import gc
import csv

from model_parts import RelativePositionalAttention, GatherSparseAttention, SpatialReorder


def bench_dense(N, L, B, d, h, coord_dim, device, dtype):
    layers = torch.nn.ModuleList([
        RelativePositionalAttention(
            coord_dim=coord_dim, embed_dim=d, n_head=h,
            cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
        ).to(device).to(dtype)
        for _ in range(L)
    ])
    q = torch.randn(B, N, d, device=device, dtype=dtype)
    coords = torch.randn(B, N, coord_dim, device=device, dtype=dtype)

    def fn():
        x = q
        for layer in layers:
            x = layer(x, x, x, coords)
        return x
    return fn


def bench_sparse(N, K, L, B, d, h, coord_dim, device, dtype, reorder=False):
    layers = torch.nn.ModuleList([
        GatherSparseAttention(embed_dim=d, n_head=h, knn_neighbors=K, mode="none").to(device).to(dtype)
        for _ in range(L)
    ])
    q = torch.randn(B, N, d, device=device, dtype=dtype)
    coords = torch.randn(B, N, coord_dim, device=device, dtype=dtype)

    if reorder:
        sr = SpatialReorder(n_bins=32)
        reorder_idx, unreorder_idx = sr.compute_idx(coords)
        coords_re = sr.reorder(coords, reorder_idx)
        yx = coords_re[..., 1:]
        dist = torch.cdist(yx, yx)
        _, knn_idx_re = torch.topk(dist, k=K, dim=-1, largest=False)

        def fn():
            q_re = sr.reorder(q, reorder_idx)
            x = q_re
            for layer in layers:
                x = layer(x, x, x, knn_idx_re, coords_re)
            return sr.unreorder(x, unreorder_idx)
    else:
        yx = coords[..., 1:]
        dist = torch.cdist(yx, yx)
        _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)

        def fn():
            x = q
            for layer in layers:
                x = layer(x, x, x, knn_idx, coords)
            return x

    return fn


def measure(fn, warmup=3, min_run_time=0.5):
    """Benchmark fn, return (mean_time_s, incremental_peak_mem_mb)."""
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


def try_bench(bench_fn, *args, **kwargs):
    try:
        fn = bench_fn(*args, **kwargs)
        t, mem = measure(fn)
        return t, mem, "ok"
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg and ("memory" in msg or "alloc" in msg):
            return -1, -1, "oom"
        return -1, -1, f"err: {e}"
    except Exception as e:
        return -1, -1, f"err: {e}"


def sanity_check(device):
    if device.type != "cuda":
        print("  [skip] no CUDA")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print("=== Sanity: sparse attention + FlashAttention dispatch ===")
    B, N, K = 2, 512, 8
    q = torch.randn(B, N, 256, device=device)
    knn_idx = torch.randint(0, N, (B, N, K), device=device)
    flash_backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]

    def test(dtype):
        m = GatherSparseAttention(embed_dim=256, n_head=4, knn_neighbors=8, mode="none").to(device, dtype)
        try:
            with sdpa_kernel(flash_backends):
                _ = m(q.to(dtype), q.to(dtype), q.to(dtype), knn_idx)
            return "OK"
        except RuntimeError as e:
            return f"FAIL ({e})"

    fp16_result = test(torch.float16)
    fp32_result = test(torch.float32)
    print(f"  fp16 + flash/cudnn backends: {fp16_result}")
    print(f"  fp32 + flash/cudnn backends: {fp32_result}")
    if "FAIL" in fp16_result:
        raise RuntimeError("fp16 should dispatch flash/cudnn but failed")
    if "FAIL" not in fp32_result:
        print("  ** NOTE: fp32 unexpectedly dispatched flash/cudnn (architecture fallback?)")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    print()

    sanity_check(device)

    B = 2
    d = 256
    h = 4
    coord_dim = 3
    L_vals = [1, 4]
    N_vals = [128, 512, 2048, 8192]
    K_vals = [4, 16, 64]

    out_csv = "benchmark_sparse_results.csv"

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "L", "N", "K", "reorder", "time_s", "mem_mb", "status"])

        for L in L_vals:
            for N in N_vals:
                # --- dense ---
                t, mem, status = try_bench(bench_dense, N, L, B, d, h, coord_dim, device, dtype)
                w.writerow(["dense", L, N, 0, 0, f"{t:.6f}" if t >= 0 else "", f"{mem:.1f}" if mem >= 0 else "", status])
                f.flush()
                if t >= 0:
                    print(f"dense          L={L} N={N:<5} -> {status:>6}  {t:.6f}s  mem={mem:.0f}MB")
                else:
                    print(f"dense          L={L} N={N:<5} -> {status:>6}")

                # --- sparse variants ---
                for reorder in [False, True]:
                    tag = "sparse+r" if reorder else "sparse   "
                    for K in K_vals:
                        if K >= N:
                            continue
                        t, mem, status = try_bench(bench_sparse, N, K, L, B, d, h, coord_dim, device, dtype, reorder=reorder)
                        w.writerow(["sparse", L, N, K, 1 if reorder else 0, f"{t:.6f}" if t >= 0 else "", f"{mem:.1f}" if mem >= 0 else "", status])
                        f.flush()
                        if t >= 0:
                            print(f"{tag} L={L} N={N:<5} K={K:<3} -> {status:>6}  {t:.6f}s  mem={mem:.0f}MB")
                        else:
                            print(f"{tag} L={L} N={N:<5} K={K:<3} -> {status:>6}")


if __name__ == "__main__":
    main()
