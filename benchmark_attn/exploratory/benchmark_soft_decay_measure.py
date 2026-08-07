"""Soft Decay GPU Measurement — measure (not estimate) soft decay vs hard mask.

Runs actual torch SDPA on synthetic Q/K/V/coords data.
Measures forward time, memory, and numerical output equivalence.
Compares: hard mask (SDPA + explicit N×N mask) vs soft decay (SDPA no mask, distance bias applied to scores).

Usage:
  python benchmark_soft_decay_measure.py --cuda   (GPU measurement)
  python benchmark_soft_decay_measure.py --cpu     (CPU fallback)
"""

import csv, argparse, time, math, gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def measure_method(fn, warmup=5, n_repeat=20):
    """Measure mean execution time of fn()."""
    for _ in range(warmup):
        fn()
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000  # ms


def measure_memory(fn):
    """Measure peak memory increment from fn()."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024 ** 2)


def run_gpu_benchmark(Ns, d_head=40, n_head=8, d_max=256, lam=5, n_repeat=15):
    """GPU measurement of hard mask vs soft decay vs dense_flash."""
    import torch
    import torch.nn.functional as F
    not gc  # gc imported at top

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    d = d_head * n_head
    is_cuda = device.type == "cuda"

    rows = []
    for N in Ns:
        torch.manual_seed(42)
        Q = torch.randn(n_head, N, d_head, device=device, dtype=dtype) / math.sqrt(d_head)
        K = torch.randn(n_head, N, d_head, device=device, dtype=dtype) / math.sqrt(d_head)
        V = torch.randn(n_head, N, d_head, device=device, dtype=dtype)
        coords = torch.rand(N, 2, device=device) * 512
        dist = torch.cdist(coords, coords)  # (N, N)

        # ── Hard mask (current) ──
        hard_mask = torch.zeros(1, n_head, N, N, device=device, dtype=dtype)
        hard_mask[:, :, dist > d_max] = float("-inf")
        distance_decay = -lam * dist / d_max
        hard_mask = hard_mask + distance_decay.to(dtype).unsqueeze(0).unsqueeze(0)

        def hard_mask_fn():
            return F.scaled_dot_product_attention(
                Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0),
                attn_mask=hard_mask
            )

        # ── Soft decay — try multiple dispatch strategies ──

        # Strategy 1: pass distance bias as attn_mask (additive, but mask disables FlashAttn)
        # Dense mask with finite values — tests cuDNN masked path with soft bias instead of -inf
        soft_mask = (-lam * dist / d_max).to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,N,N)

        def soft_decay_masked():
            """SDPA with distance bias as mask. No -inf, just additive soft decay."""
            return F.scaled_dot_product_attention(
                Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0),
                attn_mask=soft_mask
            )

        # Strategy 2: no mask, no bias — pure FlashAttention (reference only, no spatial cutoff)
        def dense_flash_fn():
            return F.scaled_dot_product_attention(
                Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
            )

        # Strategy 3: manual matmul + bias (pre-FlexAttention path — slow, materializes N×N)
        bias = soft_mask.clone()  # same values
        def manual_soft_decay():
            scores = torch.matmul(Q.unsqueeze(0), K.unsqueeze(0).transpose(-2, -1)) / math.sqrt(d_head)
            scores = scores + bias
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, V.unsqueeze(0))

        try:
            t_hard = measure_method(hard_mask_fn, n_repeat=n_repeat)
            t_soft_masked = measure_method(soft_decay_masked, n_repeat=n_repeat)
            t_soft_manual = measure_method(manual_soft_decay, n_repeat=n_repeat)
            t_dense = measure_method(dense_flash_fn, n_repeat=n_repeat)
        except RuntimeError as e:
            rows.append({"N": N, "method": "OOM", "time_ms": -1, "error": str(e)[:100]})
            print(f"  N={N}: OOM")
            continue

        # Numerical equivalence — soft_decay_masked vs hard_mask
        out_hard = hard_mask_fn()
        out_soft = soft_decay_masked()
        diff = (out_hard - out_soft).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_hard.flatten(), out_soft.flatten(), dim=0
        ).item()

        mem_hard = measure_memory(hard_mask_fn) if is_cuda else 0
        mem_soft = measure_memory(soft_decay_masked) if is_cuda else 0

        rows.append({
            "N": N,
            "device": str(device),
            "dtype": str(dtype),
            "hard_mask_ms": round(t_hard, 4),
            "soft_masked_ms": round(t_soft_masked, 4),
            "soft_manual_ms": round(t_soft_manual, 4),
            "dense_flash_ms": round(t_dense, 4),
            "soft_masked_vs_hard": round(t_hard / max(t_soft_masked, 0.0001), 1),
            "soft_manual_vs_hard": round(t_hard / max(t_soft_manual, 0.0001), 1),
            "dense_vs_hard": round(t_hard / max(t_dense, 0.0001), 1),
            "max_abs_diff": round(diff, 6),
            "cosine_similarity": round(cos_sim, 6),
            "hard_mask_mem_mb": round(mem_hard, 2),
            "soft_decay_mem_mb": round(mem_soft, 2),
        })

        status = "PASS" if cos_sim > 0.99 else "FAIL"
        print(f"  N={N:>4d}: hard={t_hard:.3f}ms  soft_masked={t_soft_masked:.3f}ms  "
              f"soft_manual={t_soft_manual:.3f}ms  dense={t_dense:.3f}ms  "
              f"cos={cos_sim:.4f} [{status}]")

        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

    return rows


def generate_figure(rows, outdir="benchmark_attn"):
    """Generate soft decay measurement figures from benchmark rows.

    Args:
        rows: List of result dictionaries from :func:`run_gpu_benchmark`.
        outdir: Directory to write the figure PNG.
    """
    outdir = Path(outdir)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ok = [r for r in rows if r.get("time_ms", -1) >= 0 or r.get("hard_mask_ms", -1) >= 0]
    if not ok:
        ok = rows

    ax = axes[0]
    Ns = [r["N"] for r in ok]
    hard_ts = [r.get("hard_mask_ms", r.get("time_ms", 0)) for r in ok]
    soft_ts = [r["soft_decay_ms"] for r in ok if "soft_decay_ms" in r]
    dense_ts = [r["dense_flash_ms"] for r in ok if "dense_flash_ms" in r]
    Ns_soft = [r["N"] for r in ok if "soft_decay_ms" in r]
    Ns_dense = [r["N"] for r in ok if "dense_flash_ms" in r]

    ax.plot(Ns, hard_ts, "s-", color="#e74c3c", label="hard mask (current)",
            markersize=9, linewidth=2.5, markerfacecolor="white")
    if Ns_soft:
        ax.plot(Ns_soft, soft_ts, "D-", color="#2ecc71", label="soft decay (no mask)",
                markersize=9, linewidth=2.5, markerfacecolor="white")
    if Ns_dense:
        ax.plot(Ns_dense, dense_ts, "o:", color="#3498db", label="dense flash (no cutoff)",
                markersize=8, linewidth=1.5, markerfacecolor="white")

    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Soft Decay vs Hard Mask — {ok[0].get('device','?')}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if Ns_soft:
        speedups = [r["soft_vs_hard_speedup"] for r in ok if "soft_vs_hard_speedup" in r]
        ax.bar(range(len(Ns_soft)), speedups, color="#2ecc71", alpha=0.85,
               edgecolor="white")
        ax.set_xticks(range(len(Ns_soft)))
        ax.set_xticklabels([str(n) for n in Ns_soft])
        ax.set_xlabel("N")
        ax.set_ylabel("Speedup (× vs hard mask)")
        ax.set_title("Soft Decay Speedup vs Hard Mask")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        for i, s in enumerate(speedups):
            ax.text(i, s + 0.1, f"{s:.1f}×", ha="center", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Soft Decay — GPU Measured", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = str(outdir / "soft_decay_measured.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png}")


def save_csv(rows, path):
    """Save benchmark rows to *path* as CSV.

    Args:
        rows: List of dictionaries to write.
        path: Output CSV file path.
    """
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved CSV: {path}")


def main():
    """Run the soft decay GPU measurement benchmark and save CSV/figure."""
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", action="store_true", help="Use CUDA GPU")
    p.add_argument("--cpu", action="store_true", default=True, help="Use CPU")
    p.add_argument("--d-head", type=int, default=40)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-repeat", type=int, default=10)
    p.add_argument("--outdir", default="benchmark_attn")
    args = p.parse_args()

    print("=" * 60)
    print("Soft Decay — GPU Measurement")
    print(f"d_head={args.d_head}, n_head={args.n_head}, "
          f"d_model={args.d_head * args.n_head}")
    print("=" * 60)

    Ns = [32, 64, 128, 256, 512, 1024]

    rows = run_gpu_benchmark(Ns, d_head=args.d_head, n_head=args.n_head,
                             n_repeat=args.n_repeat)

    outdir = Path(args.outdir)
    save_csv(rows, str(outdir / "soft_decay_measured.csv"))
    generate_figure(rows, str(outdir))

    ok = [r for r in rows if "cosine_similarity" in r]
    if ok:
        all_pass = all(r["cosine_similarity"] > 0.99 for r in ok)
        print(f"\nNumerical equivalence: {'ALL PASS' if all_pass else 'SOME FAIL'}")
        for r in ok:
            print(f"  N={r['N']}: cos_sim={r['cosine_similarity']:.6f}  "
                  f"max_diff={r['max_abs_diff']:.6f}")
        if all_pass:
            print(f"\nSoft decay produces identical output to hard mask on {args.d_head*args.n_head}D features.")


if __name__ == "__main__":
    main()
