"""Standalone benchmark: dense masked SDPA vs gather-based sparse attention.
Expanded sweep: N up to 16384, K up to 128, with/without spatial reorder.
OOM → marked in CSV.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.utils.benchmark as benchmark
import gc
import csv

from model_parts import RelativePositionalAttention, GatherSparseAttention, GatherSparseAttentionV2, DenseFlashAttention, NSASparseAttention, SpatialReorder


def bench_dense(seq_len, L, batch_size, embed_dim, n_head, coord_dim, device, dtype):
    """Build a closure that runs L layers of RelativePositionalAttention.

    Args:
        seq_len: Sequence length.
        L: Number of transformer layers.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        device: torch device.
        dtype: torch dtype.

    Returns:
        A callable ``fn()`` that runs the forward pass and returns the output.
    """
    layers = torch.nn.ModuleList([
        RelativePositionalAttention(
            coord_dim=coord_dim, embed_dim=embed_dim, n_head=n_head,
            cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
        ).to(device).to(dtype)
        for _ in range(L)
    ])
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)

    def fn():
        x = query
        for layer in layers:
            x = layer(x, x, x, coords)
        return x
    return fn


def bench_sparse(seq_len, knn_neighbors, L, batch_size, embed_dim, n_head, coord_dim, device, dtype, reorder=False):
    """Build a closure that runs L layers of GatherSparseAttention.

    Args:
        seq_len: Sequence length.
        knn_neighbors: Number of KNN neighbors.
        L: Number of transformer layers.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        device: torch device.
        dtype: torch dtype.
        reorder: If True, reorder tokens by spatial proximity.

    Returns:
        A callable ``fn()`` that runs the forward pass and returns the output.
    """
    layers = torch.nn.ModuleList([
        GatherSparseAttention(embed_dim=embed_dim, n_head=n_head, knn_neighbors=knn_neighbors, mode="none").to(device).to(dtype)
        for _ in range(L)
    ])
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)

    if reorder:
        sr = SpatialReorder(n_bins=32)
        reorder_idx, unreorder_idx = sr.compute_idx(coords)
        coords_re = sr.reorder(coords, reorder_idx)
        yx = coords_re[..., 1:]
        dist = torch.cdist(yx, yx)
        _, knn_idx_re = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)

        def fn():
            query_re = sr.reorder(query, reorder_idx)
            x = query_re
            for layer in layers:
                x = layer(x, x, x, knn_idx_re, coords_re)
            return sr.unreorder(x, unreorder_idx)
    else:
        yx = coords[..., 1:]
        dist = torch.cdist(yx, yx)
        _, knn_idx = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)

        def fn():
            x = query
            for layer in layers:
                x = layer(x, x, x, knn_idx, coords)
            return x

    return fn


def bench_sparse_v2(seq_len, knn_neighbors, L, batch_size, embed_dim, n_head, coord_dim, device, dtype):
    """Build a closure that runs L layers of GatherSparseAttentionV2.

    V2 eliminates unnecessary copies via view+unsqueeze for flat_query.

    Args:
        seq_len: Sequence length.
        knn_neighbors: Number of KNN neighbors.
        L: Number of transformer layers.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions.
        device: torch device.
        dtype: torch dtype.

    Returns:
        A callable ``fn()`` that runs the forward pass and returns the output.
    """
    layers = torch.nn.ModuleList([
        GatherSparseAttentionV2(embed_dim=embed_dim, n_head=n_head, knn_neighbors=knn_neighbors, mode="none").to(device).to(dtype)
        for _ in range(L)
    ])
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)
    yx = coords[..., 1:]
    dist = torch.cdist(yx, yx)
    _, knn_idx = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)

    def fn():
        x = query
        for layer in layers:
            x = layer(x, x, x, knn_idx, coords)
        return x
    return fn


def bench_dense_flash(seq_len, L, batch_size, embed_dim, n_head, coord_dim, device, dtype):
    """Build a closure that runs L layers of DenseFlashAttention.

    Args:
        seq_len: Sequence length.
        L: Number of transformer layers.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions (unused, kept for interface).
        device: torch device.
        dtype: torch dtype.

    Returns:
        A callable ``fn()`` that runs the forward pass and returns the output.
    """
    layers = torch.nn.ModuleList([
        DenseFlashAttention(embed_dim=embed_dim, n_head=n_head).to(device).to(dtype)
        for _ in range(L)
    ])
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn():
        x = query
        for layer in layers:
            x = layer(x, x, x)
        return x
    return fn


def bench_nsa(seq_len, L, batch_size, embed_dim, n_head, coord_dim, device, dtype,
              sliding_window_size=64, compress_block_size=32,
              compress_block_sliding_stride=16, selection_block_size=32,
              num_selected_blocks=4):
    """Build a closure that runs L layers of NSASparseAttention.

    Args:
        seq_len: Sequence length.
        L: Number of transformer layers.
        batch_size: Batch size.
        embed_dim: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of coordinate dimensions (unused, kept for interface).
        device: torch device.
        dtype: torch dtype.
        sliding_window_size: NSA sliding window size.
        compress_block_size: NSA compressed block size.
        compress_block_sliding_stride: NSA compressed block sliding stride.
        selection_block_size: NSA selection block size.
        num_selected_blocks: NSA number of selected blocks.

    Returns:
        A callable ``fn()`` that runs the forward pass and returns the output.
    """
    layers = torch.nn.ModuleList([
        NSASparseAttention(
            embed_dim=embed_dim, n_head=n_head,
            sliding_window_size=sliding_window_size,
            compress_block_size=compress_block_size,
            compress_block_sliding_stride=compress_block_sliding_stride,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks,
        ).to(device).to(dtype)
        for _ in range(L)
    ])
    query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    def fn():
        x = query
        for layer in layers:
            x = layer(x, x, x)
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
        memory_mb = (peak - baseline) / (1024 ** 2)
    else:
        memory_mb = 0.0

    timer = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    result = timer.blocked_autorange(min_run_time=min_run_time)

    return result.mean, memory_mb


def try_bench(bench_fn, *args, **kwargs):
    """Try to build and benchmark a function, catching OOM/errors.

    Args:
        bench_fn: Function that builds and returns a callable.
        *args: Positional arguments passed to bench_fn.
        **kwargs: Keyword arguments passed to bench_fn.

    Returns:
        Tuple (time_s, memory_mb, status) where status is "ok", "oom", or "err: ...".
    """
    try:
        fn = bench_fn(*args, **kwargs)
        time_s, memory_mb = measure(fn)
        return time_s, memory_mb, "ok"
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg and ("memory" in msg or "alloc" in msg):
            return -1, -1, "oom"
        return -1, -1, f"err: {e}"
    except Exception as e:
        return -1, -1, f"err: {e}"


def sanity_check(device):
    """Verify that gather-sparse attention dispatches FlashAttention.

    Tests both fp16 and fp32 with CUDNN_ATTENTION and FLASH_ATTENTION
    backends. Raises RuntimeError if fp16 fails to dispatch.

    Args:
        device: torch device to run the check on.
    """
    if device.type != "cuda":
        print("  [skip] no CUDA")
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print("=== Sanity: sparse attention + FlashAttention dispatch ===")
    batch_size, seq_len, knn_neighbors = 2, 512, 8
    query = torch.randn(batch_size, seq_len, 256, device=device)
    knn_idx = torch.randint(0, seq_len, (batch_size, seq_len, knn_neighbors), device=device)
    flash_backends = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]

    def test(dtype):
        m = GatherSparseAttention(embed_dim=256, n_head=4, knn_neighbors=8, mode="none").to(device, dtype)
        try:
            with sdpa_kernel(flash_backends):
                _ = m(query.to(dtype), query.to(dtype), query.to(dtype), knn_idx)
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
    """Run the full sparse vs dense benchmark and write results to CSV.

    Sweeps over sequence lengths N and KNN neighbors K, comparing:
    - dense (RelativePositionalAttention)
    - dense_flash (DenseFlashAttention)
    - nsa (NSASparseAttention with varying selected blocks)
    - sparse (GatherSparseAttention)
    - sparse_v2 (GatherSparseAttentionV2)

    Results are written to benchmark_attn/results/benchmark_sparse_results.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
    print()

    sanity_check(device)

    batch_size = 2
    embed_dim = 256
    n_head = 4
    coord_dim = 3
    L_vals = [1, 4]
    N_vals = [128, 256, 512, 1024, 2048, 4096, 8192]
    K_vals = [4, 16, 64]

    out_csv = str(Path(__file__).resolve().parents[1] / "results" / "benchmark_sparse_results.csv")

    with open(out_csv, "w", newline="") as file_handle:
        w = csv.writer(file_handle)
        w.writerow(["method", "L", "N", "K", "reorder", "time_s", "mem_mb", "status"])

        for L in L_vals:
            for N in N_vals:
                # --- dense (baseline RelativePositionalAttention) ---
                time_s, memory_mb, status = try_bench(bench_dense, N, L, batch_size, embed_dim, n_head, coord_dim, device, dtype)
                w.writerow(["dense", L, N, 0, 0, f"{time_s:.6f}" if time_s >= 0 else "", f"{memory_mb:.1f}" if memory_mb >= 0 else "", status])
                file_handle.flush()
                if time_s >= 0:
                    print(f"dense          L={L} N={N:<5} -> {status:>6}  {time_s:.6f}s  mem={memory_mb:.0f}MB")
                else:
                    print(f"dense          L={L} N={N:<5} -> {status:>6}")

                # --- dense_flash (no mask, no KNN) ---
                time_s, memory_mb, status = try_bench(bench_dense_flash, N, L, batch_size, embed_dim, n_head, coord_dim, device, dtype)
                w.writerow(["dense_flash", L, N, 0, 0, f"{time_s:.6f}" if time_s >= 0 else "", f"{memory_mb:.1f}" if memory_mb >= 0 else "", status])
                file_handle.flush()
                if time_s >= 0:
                    print(f"dense_flash    L={L} N={N:<5} -> {status:>6}  {time_s:.6f}s  mem={memory_mb:.0f}MB")
                else:
                    print(f"dense_flash    L={L} N={N:<5} -> {status:>6}")

                # --- nsa (Native Sparse Attention) ---
                for sel_blocks in [2, 4, 8, 16, 64]:
                    time_s, memory_mb, status = try_bench(bench_nsa, N, L, batch_size, embed_dim, n_head, coord_dim, device, dtype,
                                                          sliding_window_size=64, compress_block_size=32,
                                                          compress_block_sliding_stride=16, selection_block_size=32,
                                                          num_selected_blocks=sel_blocks)
                    w.writerow(["nsa", L, N, sel_blocks, 0, f"{time_s:.6f}" if time_s >= 0 else "", f"{memory_mb:.1f}" if memory_mb >= 0 else "", status])
                    file_handle.flush()
                    if time_s >= 0:
                        print(f"nsa            L={L} N={N:<5} sel={sel_blocks:<3} -> {status:>6}  {time_s:.6f}s  mem={memory_mb:.0f}MB")
                    else:
                        print(f"nsa            L={L} N={N:<5} sel={sel_blocks:<3} -> {status:>6}")

                # --- sparse v1 ---
                for reorder in [False]:
                    for K in K_vals:
                        if K >= N:
                            continue
                        time_s, memory_mb, status = try_bench(bench_sparse, N, K, L, batch_size, embed_dim, n_head, coord_dim, device, dtype, reorder=reorder)
                        w.writerow(["sparse", L, N, K, 0, f"{time_s:.6f}" if time_s >= 0 else "", f"{memory_mb:.1f}" if memory_mb >= 0 else "", status])
                        file_handle.flush()
                        if time_s >= 0:
                            print(f"sparse         L={L} N={N:<5} K={K:<3} -> {status:>6}  {time_s:.6f}s  mem={memory_mb:.0f}MB")
                        else:
                            print(f"sparse         L={L} N={N:<5} K={K:<3} -> {status:>6}")

                # --- sparse v2 ---
                for K in K_vals:
                    if K >= N:
                        continue
                    time_s, memory_mb, status = try_bench(bench_sparse_v2, N, K, L, batch_size, embed_dim, n_head, coord_dim, device, dtype)
                    w.writerow(["sparse_v2", L, N, K, 0, f"{time_s:.6f}" if time_s >= 0 else "", f"{memory_mb:.1f}" if memory_mb >= 0 else "", status])
                    file_handle.flush()
                    if time_s >= 0:
                        print(f"sparse_v2      L={L} N={N:<5} K={K:<3} -> {status:>6}  {time_s:.6f}s  mem={memory_mb:.0f}MB")
                    else:
                        print(f"sparse_v2      L={L} N={N:<5} K={K:<3} -> {status:>6}")


if __name__ == "__main__":
    main()
