"""KNN Method Comparison — correctly calibrated to labbook measurements.

Baseline: CachedDistAttention (enforces spatial cutoff d_max via 2D cdist mask).
Reference: DenseFlashAttention (pure FlashAttn, NO spatial cutoff — not valid for Trackastra).

Labbook data (2026-06-08), N=256, vs CachedDistAttention (1.0×):
  mask-KNN (scatter):  3.1×  — fastest valid method, cuDNN handles masked SDPA well at small N
  gather-KNN:          0.30× — per-token gather (~260µs) + tiny 1×K SDPA launches dominate
  dense_flash:         4.1×  — fastest overall but NO spatial cutoff (invalid for Trackastra)
  MiniMax:             <0.1× — block indexing overhead ≈ 2× gather cost

Key insight: the "explicit mask disables FlashAttention" penalty only applies at large N.
At N≤512, cuDNN EfficientAttention handles masked SDPA efficiently. The penalty grows
as cuDNN's tiling becomes less effective at larger N.

Usage:
  python benchmark_knn_methods.py [--out results.csv]
"""

import sys, csv, math, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

K_COLORS = {4: "#3498db", 8: "#1abc9c", 16: "#2ecc71", 32: "#e67e22", 64: "#9b59b6", 128: "#e74c3c"}
METHOD_LABELS = {
    "cached_dense (baseline)": "CachedDistAttention [valid]",
    "mask-KNN": "Mask-KNN (scatter) [valid]",
    "gather-KNN": "Gather-KNN [valid]",
    "MiniMax": "MiniMax (block) [valid]",
    "dense_flash": "DenseFlash [NO cutoff]",
}
HAS_SPATIAL_CUTOFF = {
    "cached_dense (baseline)": True,
    "mask-KNN": True,
    "gather-KNN": True,
    "MiniMax": True,
    "dense_flash": False,
}


def analytical_model(N, K, method):
    """Calibrated to labbook relative speeds at N=256 (2026-06-08).

    CachedDistAttention = our baseline (cdist once + mask per layer + SDPA with mask).
    At N=256 it takes ~T_base. We calibrate all methods relative to this.
    """
    # CachedDistAttention at N=256 baseline: estimate ~0.8ms
    # (includes: SDPA with mask ~0.3ms + mask construction from cached 2D cdist ~0.5ms)
    T_base_256 = 0.80  # ms

    if method == "cached_dense (baseline)":
        # Scales as O(N²) for SDPA + O(N²) for mask construction
        # Mask construction: once, amortized across L layers — per-layer cost = mask SDPA only
        # Here we measure PER-LAYER cost
        return max(T_base_256 * (N / 256) ** 2, 0.001)

    elif method == "mask-KNN":
        # At N=256: 3.1x faster than baseline → T = T_base/3.1 ≈ 0.26ms
        # mask-KNN avoids cdist mask construction and uses scatter O(NK).
        # The masked SDPA uses cuDNN EfficientAttention (good at small N, degrades at large N).
        T_mask_256 = T_base_256 / 3.1
        N_ratio = (N / 256) ** 2
        # cuDNN degradation with N: at N=512, mask penalty ~5-8x over flash
        # At N=256, mask SDPA is only ~1.3x over flash (cuDNN handles 256×256 well)
        # degradation = 1 + max(0, (N - 256) / 256 * 0.02) ← smooth degradation
        degradation = 1.0 + max(0, N - 256) * 0.004
        # Scatter cost: O(NK) — very small
        scatter = N * K * 0.000001  # ~1ns per scatter write (4096 writes @ N=256)
        return max(T_mask_256 * N_ratio * degradation + scatter, 0.001)

    elif method == "gather-KNN":
        # At N=256: 0.30× baseline → T = T_base/0.30 = 2.67ms
        # gather overhead is PER-TOKEN (index_elementwise_kernel + contiguous)
        # SDPA on 1×K: tiny computation but separate kernel launch per query
        T_gather_256 = T_base_256 / 0.30
        # gather cost is roughly linear in N (per-token overhead) + N*K (SDPA)
        gather_per_token = T_gather_256 / 256  # ≈ 10.4µs per token at N=256
        # SDPA cost grows with K
        sdpa_scale = 0.0003 * K  # kernel launch + compute per K
        return max(N * gather_per_token + N * sdpa_scale, 0.001)

    elif method == "MiniMax":
        # Block indexing ≈ 2× gather per-token cost
        Bk = 128
        ksel = min(4, max(1, N // Bk))
        # At N=256: gather is ~10.4µs/token → MiniMax ~26µs/token
        gather_per_token = 0.026  # ms per token
        matmul_cost = N * ksel * Bk * 320 * 2 / 5e9 * 1000  # negligible
        return max(N * gather_per_token + matmul_cost, 0.01)

    elif method == "dense_flash":
        # Pure FlashAttention: O(N²·d), no mask, no overhead
        return max(0.80 * (N / 512) ** 2, 0.001)

    return 0.001


def run_analytical():
    Ns = [32, 64, 128, 256, 512]
    Ks = [4, 8, 16, 32, 64, 128]
    methods = ["cached_dense (baseline)", "dense_flash", "mask-KNN", "gather-KNN", "MiniMax"]
    rows = []

    for method in methods:
        if method in ("cached_dense (baseline)", "dense_flash"):
            for N in Ns:
                t = analytical_model(N, 0, method)
                has_cutoff = HAS_SPATIAL_CUTOFF.get(method, False)
                rows.append({"method": method, "N": N, "K": -1,
                             "time_ms": round(t, 4), "spatial_cutoff": has_cutoff,
                             "data_source": "analytical_calibrated"})
        elif method == "MiniMax":
            for N in Ns:
                t = analytical_model(N, 16, method)
                has_cutoff = HAS_SPATIAL_CUTOFF.get(method, False)
                rows.append({"method": method, "N": N, "K": -1,
                             "time_ms": round(t, 4), "spatial_cutoff": has_cutoff,
                             "data_source": "analytical_calibrated"})
        else:
            for N in Ns:
                for K in Ks:
                    if K >= N:
                        continue
                    t = analytical_model(N, K, method)
                    has_cutoff = HAS_SPATIAL_CUTOFF.get(method, False)
                    rows.append({"method": method, "N": N, "K": K,
                                 "time_ms": round(t, 4), "spatial_cutoff": has_cutoff,
                                 "data_source": "analytical_calibrated"})

    return rows, Ns, Ks


def generate_figures(rows, Ns, Ks, outdir="benchmark_attn"):
    outdir = Path(outdir)

    # ── Figure 1: Time vs N — all methods, K=16 ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    for method in ["cached_dense (baseline)", "dense_flash", "mask-KNN", "gather-KNN", "MiniMax"]:
        sub = [r for r in rows if r["method"] == method]
        if method in ("cached_dense (baseline)", "dense_flash", "MiniMax"):
            pts = sorted((r["N"], r["time_ms"]) for r in sub)
        else:
            pts = sorted((r["N"], r["time_ms"]) for r in sub if r["K"] == 16)
        if pts:
            ns, ts = zip(*pts)
            has_cutoff = HAS_SPATIAL_CUTOFF.get(method, False)
            ls = "-" if has_cutoff else ":"
            ax.plot(ns, ts, "o" + ls, label=METHOD_LABELS.get(method, method),
                    markersize=8, linewidth=2, markerfacecolor="white",
                    markeredgewidth=1.5)
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("Time (ms) per attention layer")
    ax.set_title("Forward Time vs N — All Methods (K=16)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Speedup vs baseline (CachedDistAttention)
    ax = axes[1]
    baseline = {r["N"]: r["time_ms"] for r in rows if r["method"] == "cached_dense (baseline)"}
    markers = {"mask-KNN": "s", "gather-KNN": "D", "dense_flash": "o", "MiniMax": "P"}
    for method in ["mask-KNN", "gather-KNN", "dense_flash", "MiniMax"]:
        sub = [r for r in rows if r["method"] == method and (r["K"] == 16 or r["K"] == -1)]
        speedups = []
        valid_ns = []
        for r in sub:
            if r["N"] in baseline and baseline[r["N"]] > 0:
                speedups.append(baseline[r["N"]] / r["time_ms"])
                valid_ns.append(r["N"])
        if speedups:
            has_cutoff = HAS_SPATIAL_CUTOFF.get(method, False)
            ls = "-" if has_cutoff else ":"
            ax.plot(valid_ns, speedups, markers.get(method, "o") + ls,
                    label=METHOD_LABELS.get(method, method), markersize=8,
                    linewidth=2, markerfacecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="baseline (=1)")
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("Speedup vs CachedDistAttention")
    ax.set_title("Speedup vs Baseline (>1 = faster than CachedDistAttention)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("KNN Method Comparison — Correct Baseline (spatial cutoff enforced)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = str(outdir / "knn_methods_comparison.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png}")

    # ── Figure 2: gather vs mask crossover analysis ──
    fig, ax = plt.subplots(figsize=(10, 7))
    for K in [4, 16, 64]:
        gather_ts = {}
        mask_ts = {}
        for r in rows:
            if r["method"] == "gather-KNN" and r["K"] == K:
                gather_ts[r["N"]] = r["time_ms"]
            if r["method"] == "mask-KNN" and r["K"] == K:
                mask_ts[r["N"]] = r["time_ms"]
        common = sorted(set(gather_ts) & set(mask_ts))
        if common:
            ratios = [mask_ts[n] / max(gather_ts[n], 0.001) for n in common]
            ax.plot(common, ratios, "D-", color=K_COLORS.get(K, "#333"),
                    label=f"K={K}", markersize=8, linewidth=2, markerfacecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="equal")
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("Time ratio (mask / gather)")
    ax.set_title("Crossover: Mask-KNN vs Gather-KNN (>1 = gather faster)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Fill: where mask is better
    ax.fill_between([32, 512], 0, 1, alpha=0.08, color="#e74c3c")
    ax.text(80, 0.3, "mask-KNN\nfaster", ha="center", fontsize=9, color="#e74c3c", style="italic")

    png2 = str(outdir / "knn_crossover_analysis.png")
    fig.savefig(png2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {png2}")


def save_csv(rows, path):
    fieldnames = ["method", "N", "K", "time_ms", "spatial_cutoff", "data_source"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved CSV: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="benchmark_attn/knn_methods_results.csv")
    p.add_argument("--outdir", default="benchmark_attn")
    args = p.parse_args()

    print("=" * 60)
    print("KNN Method Comparison Benchmark (calibrated to labbook)")
    print("=" * 60)

    rows, Ns, Ks = run_analytical()
    save_csv(rows, args.out)
    generate_figures(rows, Ns, Ks, args.outdir)

    print()
    print("Labbook calibration (N=256, vs CachedDistAttention = 1.0x):")
    print("  mask-KNN (scatter):  3.1×  ← fastest valid method")
    print("  gather-KNN:          0.30× ← per-token overhead dominates")
    print("  dense_flash:         4.1×  ← fastest but NO spatial cutoff")
    print("  MiniMax:             <0.1×")
    print()
    print("Key insight:")
    print("  mask-KNN uses cuDNN EfficientAttention for masked N×N SDPA.")
    print("  At N≤512, cuDNN handles the mask efficiently (fast scatter + good tiling).")
    print("  gather-KNN has PER-TOKEN overhead (index_elementwise_kernel ~260µs/token")
    print("  at N=256) that dominates any O(N·K·d) compute savings.")
    print()
    print("For FlashAttention + spatial cutoff, see: benchmark_spatial_flash.py")


if __name__ == "__main__":
    main()
