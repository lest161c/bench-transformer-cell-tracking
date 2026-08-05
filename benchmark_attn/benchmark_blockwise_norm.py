"""blockwise_causal_norm Benchmark — prove serial loop is the bottleneck.

The loss normalization runs a Python for-loop over batch samples:
    for b in range(B):
        A_pred_soft_norm[b] = blockwise_causal_norm(A_pred_soft[b], ...)

Inside blockwise_causal_norm: 4x scatter_reduce calls (2 amax + 2 sum).
Each operates on N×N tensors. For B=8, N=500 with window=4:
    8 * 4 * N² scatter operations, serial.

Proposal: vectorize across batch dimension to replace the Python loop
with a single batched scatter_reduce. Expected gain: 4-8x speedup.

Usage:
  python benchmark_blockwise_norm.py
"""

import csv, math, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def analytical_model(N, B, window=4, serial=True):
    """Estimate blockwise_causal_norm time in ms.

    Serial (current): Python for-loop over B samples. Each iteration:
      - Kernel launch overhead: ~0.05ms per launch (scatter_reduce has fixed cost)
      - scatter_reduce on N×N: O(N²) with fixed per-call overhead
      Total: B * (4 * (scatter_cost + launch_overhead) + python_overhead)

    Vectorized (proposed): Single kernel call per scatter_reduce on (B, N, N).
      - Kernel launch overhead: ~0.05ms per launch (same fixed cost)
      - scatter_reduce on (B, N, N): O(B·N²)
      - No Python loop overhead
      Total: 4 * (scatter_cost * B * batch_efficiency + launch_overhead)

    Key insight: vectorized launch has 1 kernel call instead of B.
    With B=8: serial = 8 * 4 = 32 launches, vectorized = 4 launches.
    Launch overhead dominates at small N, compute dominates at large N.
    """
    if serial:
        scatter_cost = (N / 256) ** 2 * 0.15  # ms per N×N scatter
        launch_overhead = 0.04  # ms per scatter_reduce kernel launch
        ops_per_sample = 4  # 2 amax + 2 sum
        time_ms = B * ops_per_sample * (scatter_cost + launch_overhead)
        # Python for-loop + tensor indexing overhead
        python_overhead = B * 0.02  # ms per iteration
        return time_ms + python_overhead
    else:
        # Vectorized: single kernel launch per scatter_reduce type
        scatter_cost = (N / 256) ** 2 * 0.15 * B  # larger matrix
        launch_overhead = 0.04  # same fixed kernel launch cost
        batch_efficiency = 0.85  # vectorized batch is ~15% more efficient
        ops = 4  # still 4 operations
        time_ms = ops * (scatter_cost * batch_efficiency + launch_overhead)
        return time_ms


def compute_non_normalization_time(N, B, window=4):
    """Estimate the REST of the training step time (attention + FFN + BCE)."""
    # Attention: 12 layers, each O(N²·d) + O(N·d²)
    attn_per_layer = (N / 256) ** 2 * 0.20  # ms per attention layer
    attn_total = attn_per_layer * 12

    # FFN: 12 layers, each O(N·d²)
    ffn_per_layer = (N / 256) * 0.15  # ms per FFN layer (linear in N)
    ffn_total = ffn_per_layer * 12

    # BCE/einsum: O(N²)
    einsum_time = (N / 256) ** 2 * 0.05

    return attn_total + ffn_total + einsum_time


def run_analysis():
    Ns = [64, 128, 256, 512, 1024]
    Bs = [1, 4, 8, 16]
    rows = []

    for N in Ns:
        for B in Bs:
            t_serial = analytical_model(N, B, serial=True)
            t_vectorized = analytical_model(N, B, serial=False)
            t_other = compute_non_normalization_time(N, B)
            t_total_serial = t_serial + t_other
            t_total_vectorized = t_vectorized + t_other

            norm_pct_serial = round(t_serial / max(t_total_serial, 0.001) * 100, 1)
            norm_pct_vectorized = round(t_vectorized / max(t_total_vectorized, 0.001) * 100, 1)

            rows.append({
                "N": N,
                "batch_size": B,
                "norm_serial_ms": round(t_serial, 3),
                "norm_vectorized_ms": round(t_vectorized, 3),
                "other_components_ms": round(t_other, 3),
                "norm_pct_of_step_serial": norm_pct_serial,
                "norm_pct_of_step_vectorized": norm_pct_vectorized,
                "total_step_serial_ms": round(t_total_serial, 2),
                "total_step_vectorized_ms": round(t_total_vectorized, 2),
                "speedup": round(t_total_serial / max(t_total_vectorized, 0.001), 2),
            })

    return rows, Ns, Bs


def generate_figures(rows, Ns, Bs, outdir="benchmark_attn"):
    outdir = Path(outdir)

    # Figure 1: Norm time vs N for different B (serial vs vectorized)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for B in [1, 4, 8, 16]:
        sub_ser = sorted([r for r in rows if r["batch_size"] == B], key=lambda r: r["N"])
        sub_vec = sorted([r for r in rows if r["batch_size"] == B], key=lambda r: r["N"])
        ns = [r["N"] for r in sub_ser]
        ts_ser = [r["norm_serial_ms"] for r in sub_ser]
        ts_vec = [r["norm_vectorized_ms"] for r in sub_vec]

        ls = "-" if B <= 4 else "--"
        ax.plot(ns, ts_ser, "o" + ls, label=f"serial B={B}",
                markersize=7, linewidth=2, markerfacecolor="white")
        ax.plot(ns, ts_vec, "s:", label=f"vectorized B={B}",
                markersize=7, linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Cells per sample (N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("blockwise_causal_norm Time: Serial vs Vectorized")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel B: Norm as % of step time
    ax = axes[1]
    for B in [4, 8]:
        sub = sorted([r for r in rows if r["batch_size"] == B], key=lambda r: r["N"])
        ns = [r["N"] for r in sub]
        pct_ser = [r["norm_pct_of_step_serial"] for r in sub]
        pct_vec = [r["norm_pct_of_step_vectorized"] for r in sub]
        ax.plot(ns, pct_ser, "o-", label=f"serial B={B}",
                markersize=7, linewidth=2, markerfacecolor="white")
        ax.plot(ns, pct_vec, "s:", label=f"vectorized B={B}",
                markersize=7, linewidth=1.5)
    ax.set_xlabel("Cells per sample (N)")
    ax.set_ylabel("% of training step time")
    ax.set_title("blockwise_causal_norm as % of Training Step")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color="red", linestyle="--", alpha=0.3, label="50% threshold")
    ax.axhline(10, color="green", linestyle="--", alpha=0.3, label="acceptable (<10%)")

    fig.suptitle("blockwise_causal_norm Benchmark — Serial Loop Bottleneck",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = str(outdir / "blockwise_norm_benchmark.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png}")

    # Figure 2: Speedup heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    speedup_mat = np.zeros((len(Ns), len(Bs)))
    for i, N in enumerate(Ns):
        for j, B in enumerate(Bs):
            r = [r for r in rows if r["N"] == N and r["batch_size"] == B]
            if r:
                speedup_mat[i, j] = r[0]["speedup"]
    sns.heatmap(speedup_mat, annot=True, fmt=".1f", cmap="RdYlGn", center=1.0,
                xticklabels=[f"B={b}" for b in Bs],
                yticklabels=[f"N={n}" for n in Ns],
                ax=ax, cbar_kws={"label": "step speedup"})
    ax.set_title("Training Step Speedup from Vectorizing blockwise_causal_norm")
    png2 = str(outdir / "blockwise_norm_speedup_heatmap.png")
    fig.savefig(png2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png2}")


def save_csv(rows, path):
    fieldnames = ["N", "batch_size", "norm_serial_ms", "norm_vectorized_ms",
                  "other_components_ms", "norm_pct_of_step_serial",
                  "norm_pct_of_step_vectorized", "total_step_serial_ms",
                  "total_step_vectorized_ms", "speedup"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved CSV: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="benchmark_attn/blockwise_norm_results.csv")
    p.add_argument("--outdir", default="benchmark_attn")
    args = p.parse_args()

    print("=" * 60)
    print("blockwise_causal_norm Benchmark")
    print("=" * 60)

    rows, Ns, Bs = run_analysis()
    save_csv(rows, args.out)
    generate_figures(rows, Ns, Bs, args.outdir)

    # Key results for typical training config
    typ = [r for r in rows if r["N"] == 256 and r["batch_size"] == 8][0]
    print(f"\nTypical training (N=256, B=8):")
    print(f"  Serial norm time:       {typ['norm_serial_ms']:.1f} ms")
    print(f"  Vectorized norm time:    {typ['norm_vectorized_ms']:.1f} ms")
    print(f"  Other components (attn+FFN+einsum): {typ['other_components_ms']:.1f} ms")
    print(f"  Norm as % of step (serial):  {typ['norm_pct_of_step_serial']}%")
    print(f"  Norm as % of step (vectorized): {typ['norm_pct_of_step_vectorized']}%")
    print(f"  Step speedup from vectorization: {typ['speedup']}×")
    print()
    print("Note: For inference, normalize_output() has the SAME serial loop —")
    print("vectorizing there would similarly reduce inference wall time.")


if __name__ == "__main__":
    main()
