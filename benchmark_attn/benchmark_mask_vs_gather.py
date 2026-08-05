"""Compare KNNMaskSparseAttention (new) vs GatherSparseAttention (old)
vs DenseFlashAttention. All at N=128,256,512 with K=16 L=1 mode=none."""

import torch, torch.nn.functional as F, gc, csv
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.benchmark as benchmark

from model_parts import (
    GatherSparseAttention, DenseFlashAttention, KNNMaskSparseAttention,
    RelativePositionalAttention,
)

def measure(fn, warmup=5, min_run_time=0.5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    _ = fn(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    mem = (peak - baseline) / (1024**2)
    t = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    return t.blocked_autorange(min_run_time=min_run_time).mean, mem

def profile_sdpa_backend(name, q, k, v, mask=None):
    for _ in range(3): F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=0.125)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
        with record_function(name):
            _ = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=0.125)
        torch.cuda.synchronize()
    for evt in prof.key_averages():
        if 'attention' in evt.key and 'backward' not in evt.key:
            return evt.key, evt.device_time_total
    return "unknown", 0

def main():
    device = torch.device("cuda"); dtype = torch.float16
    print(f"Device: {device}, dtype: {dtype}")
    print(f"flash={torch.backends.cuda.flash_sdp_enabled()}  mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()}  cudnn={torch.backends.cuda.cudnn_sdp_enabled()}\n")

    B, d, h, coord_dim = 2, 256, 4, 3
    K = 16

    # === SDPA backend comparison (just the attention call) ===
    for N in [128, 256, 512]:
        print(f"={'='*50}")
        print(f"  N={N}")
        print(f"={'='*50}")

        # (A) Dense no-mask: (B, nH, N, Dh)
        q = torch.randn(B, h, N, d//h, device=device, dtype=dtype)
        backend, t = profile_sdpa_backend("dense_flash", q, q, q)
        print(f"  dense_flash        (B,nH,N,Dh)              {backend:<55s} {t:6.0f} us")

        # (B) Dense with KNN mask: (B, nH, N, Dh) + NxN mask
        yx = torch.randn(B, N, 2, device=device, dtype=torch.float32)
        dist = torch.cdist(yx, yx)
        _, knn = torch.topk(dist, k=K, dim=-1, largest=False)
        src = knn.unsqueeze(1).expand(B, h, N, K)
        mask = torch.full((B, h, N, N), float("-inf"), device=device, dtype=dtype)
        mask.scatter_(3, src, 0.0)
        backend, t = profile_sdpa_backend("masked_knn", q, q, q, mask=mask)
        print(f"  masked_knn         (B,nH,N,Dh) + KNN mask   {backend:<55s} {t:6.0f} us")

        # (C) Sparse gather shape: (B*N, nH, 1, Dh) / (B*N, nH, K, Dh)
        qs = torch.randn(B*N, h, 1, d//h, device=device, dtype=dtype)
        ks = torch.randn(B*N, h, K, d//h, device=device, dtype=dtype)
        backend, t = profile_sdpa_backend("sparse_gather", qs, ks, ks)
        print(f"  sparse_gather      (B*N,nH,1,Dh) / (K,Dh)   {backend:<55s} {t:6.0f} us")

        # (D) Sparse gather forced efficient
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                backend, t = profile_sdpa_backend("sparse+efficient", qs, ks, ks)
            print(f"  sparse+efficient   (B*N,nH,1,Dh) / (K,Dh)   {backend:<55s} {t:6.0f} us")
        except RuntimeError as e:
            print(f"  sparse+efficient   FAILED: {str(e)[:80]}")

    # === Full forward time benchmark ===
    print(f"\n{'='*60}")
    print("  FULL FORWARD TIME (with QKV proj + output proj)")
    print(f"{'='*60}")

    out_csv = "benchmark_mask_vs_gather.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "N", "K", "time_s", "mem_mb", "speedup_vs_gather"])

        for N in [128, 256, 512]:
            coords = torch.randn(B, N, coord_dim, device=device, dtype=dtype)
            yx = coords[..., 1:]
            dist = torch.cdist(yx, yx)
            _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)

            print(f"\n  N={N}:")
            results = {}

            # Gather sparse (original)
            m = GatherSparseAttention(embed_dim=d, n_head=h, knn_neighbors=K,
                                      mode="none").to(device, dtype)
            qin = torch.randn(B, N, d, device=device, dtype=dtype)
            t, mem = measure(lambda: m(qin, qin, qin, knn_idx, coords))
            results["sparse_gather"] = (t, mem)
            print(f"    sparse_gather: {t*1e6:8.1f} us  {mem:6.1f} MB")

            # KNN mask sparse
            m = KNNMaskSparseAttention(embed_dim=d, n_head=h, knn_neighbors=K,
                                       mode="none").to(device, dtype)
            t, mem = measure(lambda: m(qin, qin, qin, knn_idx, coords))
            results["mask_knn"] = (t, mem)
            speedup = results["sparse_gather"][0] / t
            print(f"    mask_knn:     {t*1e6:8.1f} us  {mem:6.1f} MB  ({speedup:.1f}x)")
            w.writerow(["mask_knn", N, K, f"{t:.9f}", f"{mem:.1f}", f"{speedup:.4f}"])

            # Dense flash
            m = DenseFlashAttention(embed_dim=d, n_head=h).to(device, dtype)
            t, mem = measure(lambda: m(qin, qin, qin))
            results["dense_flash"] = (t, mem)
            speedup = results["sparse_gather"][0] / t
            print(f"    dense_flash:  {t*1e6:8.1f} us  {mem:6.1f} MB  ({speedup:.1f}x)")
            w.writerow(["dense_flash", N, K, f"{t:.9f}", f"{mem:.1f}", f"{speedup:.4f}"])

            # Dense masked (original RelativePositionalAttention)
            m = RelativePositionalAttention(
                coord_dim=coord_dim, embed_dim=d, n_head=h,
                cutoff_spatial=128.0, mode="none", attn_dist_mode="v0",
            ).to(device, dtype)
            coords2 = torch.randn(B, N, coord_dim, device=device, dtype=dtype)
            t, mem = measure(lambda: m(qin, qin, qin, coords2))
            results["dense_masked"] = (t, mem)
            speedup = results["sparse_gather"][0] / t
            print(f"    dense_masked: {t*1e6:8.1f} us  {mem:6.1f} MB  ({speedup:.1f}x)")
            w.writerow(["dense_masked", N, K, f"{t:.9f}", f"{mem:.1f}", f"{speedup:.4f}"])

            # Also write sparse_gather row
            t, mem = results["sparse_gather"]
            w.writerow(["sparse_gather", N, K, f"{t:.9f}", f"{mem:.1f}", "1.0000"])

    print(f"\nResults: {out_csv}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY: Speedup over GatherSparseAttention")
    print(f"{'='*60}")
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            if row["method"] == "sparse_gather":
                continue
            print(f"  {row['method']:<16s} N={int(row['N']):<5} {float(row['speedup_vs_gather']):.1f}x")


if __name__ == "__main__":
    main()
