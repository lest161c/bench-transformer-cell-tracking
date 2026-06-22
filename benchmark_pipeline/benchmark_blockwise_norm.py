"""blockwise_causal_norm benchmark — serial vs vectorized scatter_reduce.

Tests the Python for-loop bottleneck in Trackastra's loss and inference
normalization. Measures actual GPU time for scatter_reduce on N×N tensors.

Bottleneck: train.py:228 runs a Python for-loop over batch samples:
    for b in range(B):
        A_norm[b] = blockwise_causal_norm(A[b], ...)

Each iteration: 4x scatter_reduce (2 amax + 2 sum) on N×N matrix.
For B=8: 32 separate kernel launches with ~0.04ms overhead each.

Fix: Replace serial loop with single batched scatter_reduce on (B, N, N).
Saves (B-1) × 4 kernel launches + Python loop overhead.

Usage:
  python benchmark_blockwise_norm.py --analytical   (theory)
  python benchmark_blockwise_norm.py --gpu           (CUDA required)
"""

import csv, argparse, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def run_analytical():
    """Theoretical model with kernel launch overhead."""
    Ns = [32, 64, 128, 256, 512, 1024]
    Bs = [1, 4, 8, 16]
    alpha = 0.15  # ms for scatter_reduce(N=256,N=256)
    launch = 0.04  # ms per kernel launch
    python_ov = 0.02  # ms per Python iteration
    eta = 0.85  # vectorized batch efficiency

    rows = []
    for N in Ns:
        scatter_cost = alpha * (N / 256) ** 2
        for B in Bs:
            t_serial = B * 4 * (scatter_cost + launch) + B * python_ov
            t_vectorized = 4 * (scatter_cost * B * eta + launch)
            rows.append({
                "N": N, "batch_size": B,
                "serial_ms": round(t_serial, 3),
                "vectorized_ms": round(t_vectorized, 3),
                "speedup": round(t_serial / max(t_vectorized, 0.0001), 2),
                "kernel_launches_serial": B * 4,
                "kernel_launches_vec": 4,
            })
    return rows, Ns, Bs


def run_gpu():
    """Actual GPU measurement using torch."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        device = torch.device("cuda")
    except (ImportError, RuntimeError) as e:
        print(f"GPU not available: {e}. Falling back to analytical.")
        return run_analytical()

    Ns = [32, 64, 128, 256, 512, 1024]
    Bs = [1, 4, 8, 16]
    rows = []

    for N in Ns:
        for B in Bs:
            # Serial: loop over batch
            tensors = [torch.randn(N, N, device=device) for _ in range(B)]

            # Warmup
            for _ in range(5):
                for t in tensors:
                    torch.scatter_reduce(t, 0,
                        torch.randint(0, N, (N, N), device=device),
                        torch.randn(N, N, device=device), reduce="amax")

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for t in tensors:
                for _ in range(4):
                    torch.scatter_reduce(t, 0,
                        torch.randint(0, N, (N, N), device=device),
                        torch.randn(N, N, device=device), reduce="amax")
            torch.cuda.synchronize()
            t_serial = (time.perf_counter() - t0) * 1000

            # Vectorized: single batch call
            batched = torch.stack(tensors)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(4):
                torch.scatter_reduce(batched, 1,
                    torch.randint(0, N, (B, N, N), device=device),
                    torch.randn(B, N, N, device=device), reduce="amax")
            torch.cuda.synchronize()
            t_vec = (time.perf_counter() - t0) * 1000

            rows.append({
                "N": N, "batch_size": B,
                "serial_ms": round(t_serial, 3),
                "vectorized_ms": round(t_vec, 3),
                "speedup": round(t_serial / max(t_vec, 0.0001), 2),
                "kernel_launches_serial": B * 4,
                "kernel_launches_vec": 4,
            })
            print(f"  N={N} B={B}: serial={t_serial:.2f}ms vec={t_vec:.2f}ms "
                  f"speedup={t_serial/max(t_vec,0.001):.1f}x")

    return rows, Ns, Bs


def generate_figures(rows, Ns, Bs, outdir="benchmark_pipeline"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for B in [4, 8, 16]:
        sub = sorted([r for r in rows if r["batch_size"] == B], key=lambda r: r["N"])
        ns = [r["N"] for r in sub]
        ts_ser = [r["serial_ms"] for r in sub]
        ts_vec = [r["vectorized_ms"] for r in sub]
        ax.plot(ns, ts_ser, "o-", label=f"serial B={B}", markersize=7, linewidth=2)
        ax.plot(ns, ts_vec, "s:", label=f"vectorized B={B}", markersize=7, linewidth=1.5)
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Serial vs Vectorized scatter_reduce")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    speedup_mat = np.zeros((len(Ns), len(Bs)))
    for i, N in enumerate(Ns):
        for j, B in enumerate(Bs):
            r = [r for r in rows if r["N"] == N and r["batch_size"] == B]
            if r: speedup_mat[i, j] = r[0]["speedup"]
    sns.heatmap(speedup_mat, annot=True, fmt=".1f", cmap="RdYlGn", center=1.0,
                xticklabels=[f"B={b}" for b in Bs],
                yticklabels=[f"N={n}" for n in Ns],
                ax=ax, cbar_kws={"label": "speedup"})
    ax.set_title("Speedup from Vectorization")

    fig.suptitle("blockwise_causal_norm — Vectorization Benchmark",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = str(outdir / "blockwise_norm.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png}")


def save_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved CSV: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analytical", action="store_true", default=True, help="Theoretical model")
    p.add_argument("--gpu", action="store_true", help="Actual GPU measurements")
    p.add_argument("--out", default="benchmark_pipeline/blockwise_norm_results.csv")
    p.add_argument("--outdir", default="benchmark_pipeline")
    args = p.parse_args()

    if args.gpu:
        rows, Ns, Bs = run_gpu()
    else:
        print("Analytical mode (no GPU). Use --gpu for actual measurements.")
        rows, Ns, Bs = run_analytical()

    save_csv(rows, args.out)
    generate_figures(rows, Ns, Bs, args.outdir)

    r = [r for r in rows if r["N"] == 256 and r["batch_size"] == 8][0]
    print(f"\nAt N=256, B=8: serial={r['serial_ms']}ms, "
          f"vectorized={r['vectorized_ms']}ms, speedup={r['speedup']}×")


if __name__ == "__main__":
    main()
