"""Synthetic attention scaling benchmark on H100.

Sweeps N ∈ [128, 256, 512, 1024, 2048, 4096, 8192], K ∈ [4, 16, 32, 64],
L ∈ [1, 6, 12] layers. Dense baseline at each N (OOMs at N≥4096).
Uses random Q/K/V tensors, float16, measures time + peak memory.

Output: runs/phase1/scaling_h100.csv
"""
import torch, time, csv, sys
from pathlib import Path

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

d_model = 256
nhead = 4
head_dim = d_model // nhead
B = 8
warmup = 5
repeat = 20

Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
Ls = [1, 6, 12]
Ks = [4, 16, 32, 64]
results = []

for L in Ls:
    for N in Ns:
        # --- Dense baseline (masked SDPA — simulates Trackastra's attn_mask path) ---
        if N <= 4096:  # dense OOMs at 8192
            try:
                attn_mask = torch.zeros(N, N, device=device, dtype=torch.float16)
                # Simulate spatial cutoff: mask cells beyond radius
                dist = torch.cdist(torch.randn(N, 2, device=device), torch.randn(N, 2, device=device))
                attn_mask[dist > 100] = -1e9

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                for _ in range(warmup):
                    for _ in range(L):
                        x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                        q = k = v = x
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                        scores = scores + attn_mask
                        attn_w = torch.softmax(scores, dim=-1)
                        x_tmp = torch.matmul(attn_w, v)
                        x_tmp = torch.nn.functional.gelu(torch.nn.Linear(d_model, d_model*2, device=device, dtype=torch.float16)(x_tmp))
                        x_tmp = torch.nn.Linear(d_model*2, d_model, device=device, dtype=torch.float16)(x_tmp)
                        x = x + x_tmp

                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024**2)

                t0 = time.perf_counter()
                for _ in range(repeat):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    q = k = v = x
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                    scores = scores + attn_mask
                    attn_w = torch.softmax(scores, dim=-1)
                    x = torch.matmul(attn_w, v)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / repeat * 1000
                results.append({"N": N, "L": L, "K": "dense", "variant": "masked",
                               "time_ms_per_step": f"{dt:.3f}", "mem_mb": f"{mem:.1f}"})
                print(f"  DENSE  N={N:5d} L={L:2d}: {dt:8.3f}ms  {mem:8.1f}MB")
            except RuntimeError as e:
                results.append({"N": N, "L": L, "K": "dense", "variant": "masked",
                               "time_ms_per_step": "OOM", "mem_mb": "OOM"})
                print(f"  DENSE  N={N:5d} L={L:2d}: OOM")

        # --- Sparse KNN (gather-based, FlashAttention-compatible) ---
        for K in Ks:
            if K >= N:
                continue
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

                for _ in range(warmup):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    coords = torch.randn(B, N, 2, device=device, dtype=torch.float16)
                    dist = torch.cdist(coords, coords)
                    _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)  # (B, N, K)
                    for _ in range(L):
                        q = k = v = x
                        k_sel = torch.gather(k.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        v_sel = torch.gather(v.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        x = torch.nn.functional.scaled_dot_product_attention(q, k_sel, v_sel)

                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024**2)

                t0 = time.perf_counter()
                for _ in range(repeat):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    coords = torch.randn(B, N, 2, device=device, dtype=torch.float16)
                    dist = torch.cdist(coords, coords)
                    _, knn_idx = torch.topk(dist, k=K, dim=-1, largest=False)
                    for _ in range(L):
                        q = k = v = x
                        k_sel = torch.gather(k.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        v_sel = torch.gather(v.unsqueeze(2).expand(-1, -1, K, -1), 1,
                                            knn_idx.unsqueeze(-1).expand(-1, -1, -1, d_model))
                        x = torch.nn.functional.scaled_dot_product_attention(q, k_sel, v_sel)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / repeat * 1000
                results.append({"N": N, "L": L, "K": f"K={K}", "variant": "knn_gather",
                               "time_ms_per_step": f"{dt:.3f}", "mem_mb": f"{mem:.1f}"})
                print(f"  KNN K={K:2d} N={N:5d} L={L:2d}: {dt:8.3f}ms  {mem:8.1f}MB")
            except RuntimeError as e:
                results.append({"N": N, "L": L, "K": f"K={K}", "variant": "knn_gather",
                               "time_ms_per_step": "OOM", "mem_mb": "OOM"})
                print(f"  KNN K={K:2d} N={N:5d} L={L:2d}: OOM")
        print()

outdir = Path("runs/phase1")
outdir.mkdir(parents=True, exist_ok=True)
csv_path = outdir / "scaling_h100.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N", "L", "K", "variant", "time_ms_per_step", "mem_mb"])
    w.writeheader()
    w.writerows(results)
print(f"\nResults written to {csv_path}")
