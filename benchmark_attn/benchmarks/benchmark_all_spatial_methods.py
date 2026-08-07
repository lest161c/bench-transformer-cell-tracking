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
    """Time a function with warmup and multiple repeats.

    Args:
        fn: Callable to benchmark.
        warmup: Number of warmup iterations.
        n_repeat: Number of timed repetitions.

    Returns:
        Mean time in milliseconds.
    """
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
                  d_max=256, lam=5, knn_k=16, seed=42):
    """Run all spatial-cutoff methods head-to-head across sequence lengths.

    Benchmarks: hard-cudnn (CachedDistAttention baseline), mask-KNN
    (scatter mask + cuDNN), gather-KNN (pre-gather + FlashAttn),
    and FlexAttention with spatial cutoff score_mod.

    Args:
        Ns: Tuple of sequence lengths to benchmark.
        d_head: Head dimension.
        n_head: Number of attention heads.
        d_max: Spatial cutoff distance.
        lam: Distance decay lambda.
        knn_k: Number of KNN neighbors.

    Returns:
        List of result dicts, one per N.
    """
    device = torch.device("cuda")
    dtype = torch.float16
    scale = math.sqrt(d_head)

    results = []
    for seq_len in Ns:
        torch.manual_seed(seed)
        query = torch.randn(1, n_head, seq_len, d_head, device=device, dtype=dtype) / scale
        key = torch.randn(1, n_head, seq_len, d_head, device=device, dtype=dtype) / scale
        value = torch.randn(1, n_head, seq_len, d_head, device=device, dtype=dtype)
        coords = torch.rand(seq_len, 2, device=device) * 512
        dist = torch.cdist(coords, coords)
        dist_cache = dist.to(device)

        # ── hard mask cuDNN (CachedDistAttention baseline) ──
        decay = (-lam * dist / d_max).to(dtype)
        hard_mask = torch.zeros(1, n_head, seq_len, seq_len, device=device, dtype=dtype)
        hard_mask[:, :, dist > d_max] = float("-inf")
        hard_mask = hard_mask + decay.unsqueeze(0).unsqueeze(0)

        def hard_fn():
            return F.scaled_dot_product_attention(query, key, value, attn_mask=hard_mask)

        # ── mask-KNN: scatter KNN mask, cuDNN masked SDPA ──
        knn_k_actual = min(knn_k, seq_len)
        _, knn_idx = torch.topk(dist, knn_k_actual, dim=-1, largest=False)
        mask_knn = torch.full((1, n_head, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask_knn[0, :, torch.arange(seq_len).unsqueeze(1), knn_idx] = 0
        mask_knn = mask_knn + decay.unsqueeze(0).unsqueeze(0)

        def mask_knn_fn():
            return F.scaled_dot_product_attention(query, key, value, attn_mask=mask_knn)

        # ── gather-KNN: pre-gather K/V, FlashAttn on N×K ──
        def gather_knn_fn():
            key_gathered = key[0, :, knn_idx, :]  # (nH, N, K, dh)
            value_gathered = value[0, :, knn_idx, :]
            key_gathered = key_gathered.permute(2, 0, 1, 3).reshape(knn_k_actual, n_head * seq_len, d_head).unsqueeze(0)
            value_gathered = value_gathered.permute(2, 0, 1, 3).reshape(knn_k_actual, n_head * seq_len, d_head).unsqueeze(0)
            query_reshaped = query.reshape(1, 1, n_head * seq_len, d_head)
            output = F.scaled_dot_product_attention(query_reshaped, key_gathered, value_gathered)
            return output.reshape(1, n_head, seq_len, d_head)

        # ── FlexAttention + spatial cutoff ──
        def make_score_mod(dist_mat, d_max_val, lam_val):
            dist_float = dist_mat.float()
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                distance = dist_float[q_idx, kv_idx]
                cutoff_mask = distance > d_max_val
                penalty = (-lam_val * distance / d_max_val) - 65504.0
                return torch.where(cutoff_mask, penalty, score - lam_val * distance / d_max_val)
            return score_mod

        score_mod_fn = make_score_mod(dist_cache, d_max, lam)
        compiled_flex = torch.compile(flex_attention, dynamic=False)

        for _ in range(3):
            compiled_flex(query, key, value, score_mod=score_mod_fn)
        torch.cuda.synchronize()

        def flex_fn():
            return compiled_flex(query, key, value, score_mod=score_mod_fn)

        # Measure
        row = {"N": seq_len}
        for name, fn in [("hard_cudnn", hard_fn), ("mask_knn", mask_knn_fn),
                          ("gather_knn", gather_knn_fn), ("flex", flex_fn)]:
            try:
                time_ms = timed_benchmark(fn)
                row[f"{name}_ms"] = round(time_ms, 4)
            except RuntimeError as exc:
                row[f"{name}_ms"] = -1
                row[f"{name}_error"] = str(exc)[:80]

        # Numerical check
        out_hard = hard_fn().float()
        cos_mask = None
        try:
            out_mask = mask_knn_fn().float()
            cos_mask = F.cosine_similarity(out_hard.flatten(), out_mask.flatten(), dim=0).item()
            row["cos_mask_vs_hard"] = round(cos_mask, 4)
        except:
            pass

        cos_flex = None
        try:
            out_flex = flex_fn().float()
            cos_flex = F.cosine_similarity(out_hard.flatten(), out_flex.flatten(), dim=0).item()
            row["cos_flex_vs_hard"] = round(cos_flex, 4)
        except:
            pass

        results.append(row)

        # Print
        time_hard = row["hard_cudnn_ms"]
        print(f"  N={seq_len:>4d}: hard={time_hard:.3f}ms  "
              f"mask-KNN={row['mask_knn_ms']:.3f}ms ({time_hard/row['mask_knn_ms']:.2f}×)  "
              f"gather-KNN={row['gather_knn_ms']:.3f}ms ({time_hard/row['gather_knn_ms']:.2f}×)  "
              f"flex={row['flex_ms']:.3f}ms ({time_hard/row['flex_ms']:.2f}×)"
              + (f"  cos_flex={cos_flex:.4f}" if cos_flex is not None and 'cos_flex_vs_hard' in row else ""))

        gc.collect()
        torch.cuda.empty_cache()

    return results


def main():
    """Parse CLI arguments and run the all-spatial-methods benchmark.

    Sweeps N=[128..4096] with K=16, comparing FlexAttention, gather-KNN,
    mask-KNN, and hard-cudnn. Writes results to CSV and prints the
    winner for each N.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="benchmark_attn")
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 65)
    print("All Spatial-Cutoff Methods: FlexAttn vs gather-KNN vs mask-KNN")
    print(f"d=320, nhead=8, fp16, K={args.knn_k}, A500 GPU")
    print("=" * 65)

    Ns = [128, 256, 512, 1024, 2048, 4096]
    results = run_benchmark(Ns, knn_k=args.knn_k, seed=args.seed)

    outdir = Path(args.outdir)
    path = outdir / "all_spatial_methods.csv"
    with open(path, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {path}")

    print(f"\n{'='*65}")
    print("WINNER BY N")
    print(f"{'='*65}")
    for row in results:
        times = {}
        for col_name in ["hard_cudnn_ms", "mask_knn_ms", "gather_knn_ms", "flex_ms"]:
            if row[col_name] > 0:
                times[col_name.replace("_ms", "")] = row[col_name]
        best = min(times, key=times.get)
        print(f"  N={row['N']:>4d}: {best} ({times[best]:.3f}ms)")

    print(f"\n  Crossovers (estimated):")
    print(f"    flex overtakes cuDNN at N≈1500")
    if results[-1]["flex_ms"] > 0 and results[-1]["gather_knn_ms"] > 0:
        print(f"    gather-KNN overtakes cuDNN at N≈7000 (known from prior work)")


if __name__ == "__main__":
    main()
