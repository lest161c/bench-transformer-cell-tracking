"""Compare KNNMaskSparseAttention (new) vs GatherSparseAttention (old)
vs DenseFlashAttention. All at N=128,256,512 with K=16 L=1 mode=none."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch, torch.nn.functional as F, gc, csv
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark

from model_parts import (
    GatherSparseAttention, DenseFlashAttention, KNNMaskSparseAttention,
    RelativePositionalAttention,
)

def measure(fn, warmup=5, min_run_time=0.5):
    """Benchmark a callable, returning (mean_time_s, incremental_peak_mem_mb).

    Args:
        fn: Callable to benchmark.
        warmup: Number of warmup iterations.
        min_run_time: Minimum run time for the benchmark timer.

    Returns:
        Tuple (mean_time_s, peak_memory_mb).
    """
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    _ = fn(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    memory_mb = (peak - baseline) / (1024**2)
    timer = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    return timer.blocked_autorange(min_run_time=min_run_time).mean, memory_mb

def profile_sdpa_backend(name, query, key, value, mask=None):
    """Profile which SDPA backend is dispatched for the given tensors.

    Args:
        name: Label for the profiled operation.
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        mask: Optional attention mask.

    Returns:
        Tuple (backend_name, device_time_total_us).
    """
    for _ in range(3): F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=0.125)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        with record_function(name):
            _ = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=0.125)
        torch.cuda.synchronize()
    for evt in prof.key_averages():
        if 'attention' in evt.key and 'backward' not in evt.key:
            return evt.key, evt.device_time_total
    return "unknown", 0

def main():
    """Run mask vs gather benchmark and write results to CSV.

    Benchmarks KNNMaskSparseAttention, GatherSparseAttention,
    DenseFlashAttention, and RelativePositionalAttention at N=128,256,512.
    Also profiles which SDPA backend is dispatched for each method.
    """
    device = torch.device("cuda"); dtype = torch.float16
    print(f"Device: {device}, dtype: {dtype}")
    print(f"flash={torch.backends.cuda.flash_sdp_enabled()}  mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()}  cudnn={torch.backends.cuda.cudnn_sdp_enabled()}\n")

    batch_size, embed_dim, n_head, coord_dim = 2, 256, 4, 3
    knn_neighbors = 16

    # === SDPA backend comparison (just the attention call) ===
    for seq_len in [128, 256, 512]:
        print(f"={'='*50}")
        print(f"  N={seq_len}")
        print(f"={'='*50}")

        # (A) Dense no-mask: (batch_size, n_head, seq_len, head_dim)
        query = torch.randn(batch_size, n_head, seq_len, embed_dim//n_head, device=device, dtype=dtype)
        backend, time_s = profile_sdpa_backend("dense_flash", query, query, query)
        print(f"  dense_flash        (B,nH,N,Dh)              {backend:<55s} {time_s:6.0f} us")

        # (B) Dense with KNN mask: (batch_size, n_head, seq_len, head_dim) + seq_len x seq_len mask
        yx = torch.randn(batch_size, seq_len, 2, device=device, dtype=torch.float32)
        dist = torch.cdist(yx, yx)
        _, knn = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)
        src = knn.unsqueeze(1).expand(batch_size, n_head, seq_len, knn_neighbors)
        mask = torch.full((batch_size, n_head, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask.scatter_(3, src, 0.0)
        backend, time_s = profile_sdpa_backend("masked_knn", query, query, query, mask=mask)
        print(f"  masked_knn         (B,nH,N,Dh) + KNN mask   {backend:<55s} {time_s:6.0f} us")

        # (C) Sparse gather shape: (batch_size*seq_len, n_head, 1, head_dim) / (batch_size*seq_len, n_head, knn_neighbors, head_dim)
        query_sparse = torch.randn(batch_size*seq_len, n_head, 1, embed_dim//n_head, device=device, dtype=dtype)
        key_sparse = torch.randn(batch_size*seq_len, n_head, knn_neighbors, embed_dim//n_head, device=device, dtype=dtype)
        backend, time_s = profile_sdpa_backend("sparse_gather", query_sparse, key_sparse, key_sparse)
        print(f"  sparse_gather      (B*N,nH,1,Dh) / (K,Dh)   {backend:<55s} {time_s:6.0f} us")

        # (D) Sparse gather forced efficient
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                backend, time_s = profile_sdpa_backend("sparse+efficient", query_sparse, key_sparse, key_sparse)
            print(f"  sparse+efficient   (B*N,nH,1,Dh) / (K,Dh)   {backend:<55s} {time_s:6.0f} us")
        except RuntimeError as e:
            print(f"  sparse+efficient   FAILED: {str(e)[:80]}")

    # === Full forward time benchmark ===
    print(f"\n{'='*60}")
    print("  FULL FORWARD TIME (with QKV proj + output proj)")
    print(f"{'='*60}")

    out_csv = "benchmark_mask_vs_gather.csv"
    with open(out_csv, "w", newline="") as file_handle:
        w = csv.writer(file_handle)
        w.writerow(["method", "N", "K", "time_s", "mem_mb", "speedup_vs_gather"])

        for seq_len in [128, 256, 512]:
            coords = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)
            yx = coords[..., 1:]
            dist = torch.cdist(yx, yx)
            _, knn_idx = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)

            print(f"\n  N={seq_len}:")
            results = {}

            # Gather sparse (original)
            m = GatherSparseAttention(embed_dim=embed_dim, n_head=n_head, knn_neighbors=knn_neighbors,
                                      mode="none").to(device, dtype)
            qin = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
            time_s, memory_mb = measure(lambda: m(qin, qin, qin, knn_idx, coords))
            results["sparse_gather"] = (time_s, memory_mb)
            print(f"    sparse_gather: {time_s*1e6:8.1f} us  {memory_mb:6.1f} MB")

            # KNN mask sparse
            m = KNNMaskSparseAttention(embed_dim=embed_dim, n_head=n_head, knn_neighbors=knn_neighbors,
                                       mode="none").to(device, dtype)
            time_s, memory_mb = measure(lambda: m(qin, qin, qin, knn_idx, coords))
            results["mask_knn"] = (time_s, memory_mb)
            speedup = results["sparse_gather"][0] / time_s
            print(f"    mask_knn:     {time_s*1e6:8.1f} us  {memory_mb:6.1f} MB  ({speedup:.1f}x)")
            w.writerow(["mask_knn", seq_len, knn_neighbors, f"{time_s:.9f}", f"{memory_mb:.1f}", f"{speedup:.4f}"])

            # Dense flash
            m = DenseFlashAttention(embed_dim=embed_dim, n_head=n_head).to(device, dtype)
            time_s, memory_mb = measure(lambda: m(qin, qin, qin))
            results["dense_flash"] = (time_s, memory_mb)
            speedup = results["sparse_gather"][0] / time_s
            print(f"    dense_flash:  {time_s*1e6:8.1f} us  {memory_mb:6.1f} MB  ({speedup:.1f}x)")
            w.writerow(["dense_flash", seq_len, knn_neighbors, f"{time_s:.9f}", f"{memory_mb:.1f}", f"{speedup:.4f}"])

            # Dense masked (original RelativePositionalAttention)
            m = RelativePositionalAttention(
                coord_dim=coord_dim, embed_dim=embed_dim, n_head=n_head,
                cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
            ).to(device, dtype)
            coords2 = torch.randn(batch_size, seq_len, coord_dim, device=device, dtype=dtype)
            time_s, memory_mb = measure(lambda: m(qin, qin, qin, coords2))
            results["dense_masked"] = (time_s, memory_mb)
            speedup = results["sparse_gather"][0] / time_s
            print(f"    dense_masked: {time_s*1e6:8.1f} us  {memory_mb:6.1f} MB  ({speedup:.1f}x)")
            w.writerow(["dense_masked", seq_len, knn_neighbors, f"{time_s:.9f}", f"{memory_mb:.1f}", f"{speedup:.4f}"])

            # Also write sparse_gather row
            time_s, memory_mb = results["sparse_gather"]
            w.writerow(["sparse_gather", seq_len, knn_neighbors, f"{time_s:.9f}", f"{memory_mb:.1f}", "1.0000"])

    print(f"\nResults: {out_csv}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY: Speedup over GatherSparseAttention")
    print(f"{'='*60}")
    with open(out_csv) as file_handle:
        for row in csv.DictReader(file_handle):
            if row["method"] == "sparse_gather":
                continue
            print(f"  {row['method']:<16s} N={int(row['N']):<5} {float(row['speedup_vs_gather']):.1f}x")


if __name__ == "__main__":
    main()
