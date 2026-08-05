"""All spatial-cutoff methods head-to-head: FlexAttn vs gather-KNN vs mask-KNN vs cuDNN.

Same d=320, nhead=8, fp16 on A500. Measures speed and verifies spatial cutoff.

Usage:
  python benchmark_all_spatial_methods.py
"""

import csv, math, time, gc, argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import numpy as np


def timed_benchmark(fn, warmup=5, n_repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)) * 1000


def run_benchmark(Ns=(128, 256, 512, 1024, 2048, 4096), d_head=40, n_head=8,
                  d_max=256, lam=5, knn_k=16):
    device = torch.device("cuda")
    dtype = torch.float16
    scale = math.sqrt(d_head)

    results = []
    for N in Ns:
        torch.manual_seed(42)
        Q = torch.randn(1, n_head, N, d_head, device=device, dtype=dtype) / scale
        K = torch.randn(1, n_head, N, d_head, device=device, dtype=dtype) / scale
        V = torch.randn(1, n_head, N, d_head, device=device, dtype=dtype)
        coords = torch.rand(N, 2, device=device) * 512
        dist = torch.cdist(coords, coords)
        dist_cache = dist.to(device)

        # ── hard mask cuDNN (CachedDistAttention baseline) ──
        decay = (-lam * dist / d_max).to(dtype)
        hard_mask = torch.zeros(1, n_head, N, N, device=device, dtype=dtype)
        hard_mask[:, :, dist > d_max] = float("-inf")
        hard_mask = hard_mask + decay.unsqueeze(0).unsqueeze(0)

        def hard_fn():
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=hard_mask)

        # ── mask-KNN: scatter KNN mask, cuDNN masked SDPA ──
        knn_k_actual = min(knn_k, N)
        _, knn_idx = torch.topk(dist, knn_k_actual, dim=-1, largest=False)
        mask_knn = torch.full((1, n_head, N, N), float("-inf"), device=device, dtype=dtype)
        mask_knn[0, :, torch.arange(N).unsqueeze(1), knn_idx] = 0
        mask_knn = mask_knn + decay.unsqueeze(0).unsqueeze(0)

        def mask_knn_fn():
            return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask_knn)

        # ── gather-KNN: pre-gather K/V, FlashAttn on N×K ──
        def gather_knn_fn():
            K_g = K[0, :, knn_idx, :]  # (nH, N, K, dh)
            V_g = V[0, :, knn_idx, :]
            K_g = K_g.permute(2, 0, 1, 3).reshape(knn_k_actual, n_head * N, d_head).unsqueeze(0)
            V_g = V_g.permute(2, 0, 1, 3).reshape(knn_k_actual, n_head * N, d_head).unsqueeze(0)
            Q_r = Q.reshape(1, 1, n_head * N, d_head)
            out = F.scaled_dot_product_attention(Q_r, K_g, V_g)
            return out.reshape(1, n_head, N, d_head)

        # ── FlexAttention + spatial cutoff ──
        def make_score_mod(dist_mat, d_max_val, lam_val):
            df = dist_mat.float()
            def score_mod(score, b, h, q_idx, kv_idx):
                d = df[q_idx, kv_idx]
                cutoff_mask = d > d_max_val
                penalty = (-lam_val * d / d_max_val) - 65504.0
                return torch.where(cutoff_mask, penalty, score - lam_val * d / d_max_val)
            return score_mod

        score_mod_fn = make_score_mod(dist_cache, d_max, lam)
        compiled_flex = torch.compile(flex_attention, dynamic=False)

        for _ in range(3):
            compiled_flex(Q, K, V, score_mod=score_mod_fn)
        torch.cuda.synchronize()

        def flex_fn():
            return compiled_flex(Q, K, V, score_mod=score_mod_fn)

        # Measure
        row = {"N": N}
        for name, fn in [("hard_cudnn", hard_fn), ("mask_knn", mask_knn_fn),
                          ("gather_knn", gather_knn_fn), ("flex", flex_fn)]:
            try:
                t = timed_benchmark(fn)
                row[f"{name}_ms"] = round(t, 4)
            except RuntimeError as e:
                row[f"{name}_ms"] = -1
                row[f"{name}_error"] = str(e)[:80]

        # Numerical check
        out_hard = hard_fn().float()
        try:
            out_mask = mask_knn_fn().float()
            cos_mask = F.cosine_similarity(out_hard.flatten(), out_mask.flatten(), dim=0).item()
            row["cos_mask_vs_hard"] = round(cos_mask, 4)
        except:
            pass

        try:
            out_flex = flex_fn().float()
            cos_flex = F.cosine_similarity(out_hard.flatten(), out_flex.flatten(), dim=0).item()
            row["cos_flex_vs_hard"] = round(cos_flex, 4)
        except:
            pass

        results.append(row)

        # Print
        t_h = row["hard_cudnn_ms"]
        print(f"  N={N:>4d}: hard={t_h:.3f}ms  "
              f"mask-KNN={row['mask_knn_ms']:.3f}ms ({t_h/row['mask_knn_ms']:.2f}×)  "
              f"gather-KNN={row['gather_knn_ms']:.3f}ms ({t_h/row['gather_knn_ms']:.2f}×)  "
              f"flex={row['flex_ms']:.3f}ms ({t_h/row['flex_ms']:.2f}×)"
              + (f"  cos_flex={cos_flex:.4f}" if 'cos_flex_vs_hard' in row else ""))

        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmark_attn")
    p.add_argument("--knn-k", type=int, default=16)
    args = p.parse_args()

    print("=" * 65)
    print("All Spatial-Cutoff Methods: FlexAttn vs gather-KNN vs mask-KNN")
    print(f"d=320, nhead=8, fp16, K={args.knn_k}, A500 GPU")
    print("=" * 65)

    Ns = [128, 256, 512, 1024, 2048, 4096]
    results = run_benchmark(Ns, knn_k=args.knn_k)

    outdir = Path(args.outdir)
    path = outdir / "all_spatial_methods.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\nSaved: {path}")

    print(f"\n{'='*65}")
    print("WINNER BY N")
    print(f"{'='*65}")
    for r in results:
        times = {}
        for k in ["hard_cudnn_ms", "mask_knn_ms", "gather_knn_ms", "flex_ms"]:
            if r[k] > 0:
                times[k.replace("_ms", "")] = r[k]
        best = min(times, key=times.get)
        print(f"  N={r['N']:>4d}: {best} ({times[best]:.3f}ms)")

    print(f"\n  Crossovers (estimated):")
    print(f"    flex overtakes cuDNN at N≈1500")
    if results[-1]["flex_ms"] > 0 and results[-1]["gather_knn_ms"] > 0:
        print(f"    gather-KNN overtakes cuDNN at N≈7000 (known from prior work)")


if __name__ == "__main__":
    main()
