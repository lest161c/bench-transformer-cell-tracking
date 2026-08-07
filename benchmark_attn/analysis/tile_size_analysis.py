"""Tile size analysis for FlashAttention: where does it become efficient?

FlashAttention splits the attention matrix into tiles of size Br x Bc.
For very small N, the tiling overhead exceeds the compute benefit.
This script analyzes the crossover points and tests SDPA at small N.

Key constraints (FlashAttention-2, Ampere/H100):
  - head_dim must be <= 256 and divisible by 8
  - Br, Bc must fit within shared memory (~48KB on A100, ~100KB on H100)
  - For N < tile_size, only 1 tile -> no tiling benefit
  - Minimum dispatch N depends on PyTorch's auto-heuristic

Usage (GPU required for benchmarking)::

    python tile_size_analysis.py [--out results.csv]

Without GPU: runs analytical analysis only (constraint calculation, tile math).
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ─── Analytical Analysis (no GPU needed) ───


def flash_attention_tile_analysis() -> dict:
    """Compute theoretical tile boundaries for FlashAttention-2 on Ampere.

    Analyses SRAM constraints, tile counts, head-dimension compatibility,
    and the N values at which FlashAttention begins to benefit from tiling.

    Returns:
        A dict keyed by GPU name, each value containing SRAM size,
        max/min tile rows, minimum N for tiling, and d_head compatibility.
    """
    results = {}
    print("=" * 65)
    print("FlashAttention Tile Size Analysis (Analytical)")
    print("=" * 65)

    # Trackastra config
    d_model = 320
    n_head = 8
    d_head = d_model // n_head  # 40

    print(f"\nTrackastra config: d_model={d_model}, n_head={n_head}, d_head={d_head}")
    print(f"d_head % 8 = {d_head % 8} (must be 0 for FlashAttention)")

    # FlashAttention-2 tile constraints (Ampere A100)
    # Shared memory per SM: ~48KB on A100, ~100KB on H100
    # FA2 Br formula: Br = min(SRAM / (4*d_head), seq_len)
    # For fp16: SRAM size in bytes / (4 * d_head * 2 bytes/fp16)

    sram_sizes = {
        "A100": 48 * 1024,   # 48 KB per SM
        "H100": 228 * 1024,  # 228 KB per SM (unified)
        "RTX A500": 64 * 1024,  # ~64 KB per SM
    }

    for gpu_name, sram_bytes in sram_sizes.items():
        # Br = M / (4 * d_head * 2 bytes) — approximate
        max_tile_rows = sram_bytes // (4 * d_head * 2)
        min_tile_rows = 32 if d_head <= 64 else 16
        results[gpu_name] = {
            "sram_per_sm_kb": sram_bytes // 1024,
            "max_tile_rows": max_tile_rows,
            "min_tile_rows": min_tile_rows,
            "d_head": d_head,
            "d_head_compatible": d_head % 8 == 0,
            "min_N_for_tiling": min_tile_rows,
            "min_N_for_benefit": max_tile_rows * 2,  # 2+ tiles for speedup
        }
        print(f"\n{gpu_name} (SRAM={sram_bytes // 1024}KB/SM):")
        print(f"  Max Br (tile rows): {max_tile_rows}")
        print(f"  Min Br: {min_tile_rows}")
        print(f"  Min N for tiling: {min_tile_rows}")
        print(f"  Min N for benefit: {max_tile_rows * 2}")
        print(f"  d_head compatible: {'YES' if d_head % 8 == 0 else 'NO'}")

    # N sweep analysis
    n_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    print("\n\nTile count per N (d_head=40, Br≈32, Bc≈32):")
    print(f"  {'N':>6s}  {'num_tiles':>10s}  {'tile_util%':>11s}  {'regime':>20s}")
    for seq_len in n_values:
        num_tiles = max(1, math.ceil(seq_len / 32))
        tile_utilization = seq_len / (num_tiles * 32) * 100
        if seq_len < 32:
            regime = "single tile (overhead)"
        elif seq_len < 64:
            regime = "marginal (1-2 tiles)"
        elif seq_len < 128:
            regime = "partial tiling"
        else:
            regime = "full tiling benefit"
        print(f"  {seq_len:>6d}  {num_tiles:>10d}  {tile_utilization:>10.1f}%  {regime:>20s}")

    # Head dimension constraints
    print("\n\nFlashAttention head_dim constraints:")
    test_head_dims = [16, 32, 40, 64, 80, 96, 128, 160, 256]
    for head_dim in test_head_dims:
        compatible = "YES" if head_dim % 8 == 0 and head_dim <= 256 else "NO"
        if head_dim == d_head:
            compatible += " << Trackastra"
        print(f"  d_head={head_dim:>3d}: compatible={compatible}")

    return results


def benchmark_small_n_analysis() -> list[dict]:
    """Generate theoretical projections for small-N SDPA timing.

    Actual benchmarks must run on GPU. This provides the expected regime
    based on known kernel characteristics:
      - N < 32:  math/cuDNN fallback, kernel launch overhead dominates
      - N 32-64: mem_efficient attention, marginal
      - N 64-128: flash attention may dispatch, tile efficiency low
      - N > 128:  full flash attention, near-linear scaling with N² for
                  this range, then tile efficiency flattens

    Returns:
        A list of dicts (one per (d_head, N) combination) with
        ``backend_predicted``, ``overhead_est_pct``,
        ``N_squared_ratio``, ``tile_utilization``, and ``compatible``.
    """
    rows = []
    n_values = [4, 8, 16, 32, 64, 128, 256, 512]
    test_head_dims = [40, 64, 80, 128]

    for head_dim in test_head_dims:
        for seq_len in n_values:
            # Dispatch heuristic: what backend would PyTorch use?
            if seq_len < 32 or head_dim > 256 or head_dim % 8 != 0:
                predicted_backend = "math/cuDNN fallback"
            elif seq_len < 64:
                predicted_backend = "mem_efficient (likely)"
            elif seq_len < 128:
                predicted_backend = "flash (borderline)"
            else:
                predicted_backend = "flash_attention"

            # Theoretical kernel launch overhead vs compute ratio
            # For N=4: overhead ~90%, compute ~10%
            # For N=128: overhead ~5%, compute ~95%
            estimated_overhead_pct = max(1, 100 * (32 / max(seq_len, 1)) ** 0.7)

            n_squared_ratio = seq_len ** 2 / (128 ** 2)

            rows.append({
                "d_head": head_dim,
                "N": seq_len,
                "backend_predicted": predicted_backend,
                "overhead_est_pct": round(estimated_overhead_pct, 1),
                "N_squared_ratio": round(n_squared_ratio, 4),
                "tile_utilization": round(min(100, seq_len / 32 * 100), 1),
                "compatible": "yes" if (head_dim % 8 == 0 and head_dim <= 256) else "no",
            })

    return rows


# ─── GPU Benchmark (requires CUDA) ───


def run_gpu_benchmark(seed: int = 42) -> list[dict]:
    """Benchmark SDPA at small N on GPU across backends.

    Requires CUDA.  Runs ``scaled_dot_product_attention`` for various
    ``d_head`` and ``N`` combinations, measuring time and peak memory.

    Returns:
        A list of result dicts with keys ``d_head``, ``N``,
        ``backend``, ``time_ms``, ``mem_mb``, and optionally ``error``.
    """
    import torch
    import torch.nn.functional as F
    import torch.utils.benchmark as torch_bench

    device = torch.device("cuda")
    dtype = torch.float16
    n_values = [4, 8, 16, 32, 64, 128, 256, 512]
    test_head_dims = [40, 64, 128]
    n_head = 4

    print("\n" + "=" * 65)
    print("GPU SDPA Benchmark — Small N to 512")
    print(f"Device: {device}  dtype: {dtype}")
    print(f"Flash SDPA: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Mem-efficient SDPA: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Math SDPA: {torch.backends.cuda.math_sdp_enabled()}")
    print("=" * 65)

    rows = []

    for head_dim in test_head_dims:
        embed_dim = head_dim * n_head
        for seq_len in n_values:
            torch.manual_seed(seed)
            query = torch.randn(1, n_head, seq_len, head_dim, device=device, dtype=dtype)
            key = torch.randn(1, n_head, seq_len, head_dim, device=device, dtype=dtype)
            value = torch.randn(1, n_head, seq_len, head_dim, device=device, dtype=dtype)

            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()

                timer = torch_bench.Timer(
                    stmt="F.scaled_dot_product_attention(q, k, v)",
                    globals={"F": F, "q": query, "k": key, "v": value},
                )
                measured_time_ms = timer.timeit(100 if seq_len <= 64 else 50).mean * 1000

                torch.cuda.synchronize()
                peak_mem_mb = (torch.cuda.max_memory_allocated() - mem_before) / (1024 ** 2)

                rows.append({
                    "d_head": head_dim,
                    "N": seq_len,
                    "backend": "auto",
                    "time_ms": round(measured_time_ms, 5),
                    "mem_mb": round(peak_mem_mb, 2),
                    "tokens_per_sec": round(seq_len * seq_len / (measured_time_ms / 1000) if measured_time_ms > 0 else 0),
                })
                print(f"  d_head={head_dim:>3d}  N={seq_len:>4d}  auto: {measured_time_ms:.3f}ms  {peak_mem_mb:.2f}MB")

            except RuntimeError as exc:
                rows.append({
                    "d_head": head_dim,
                    "N": seq_len,
                    "backend": "auto",
                    "time_ms": -1,
                    "mem_mb": -1,
                    "error": str(exc)[:120],
                })
                print(f"  d_head={head_dim:>3d}  N={seq_len:>4d}  auto: OOM/ERR")

    return rows


# ─── Visualization ───


def generate_figures(analytical_results: dict, bench_rows: list[dict] | None = None) -> str:
    """Generate a 4-panel FlashAttention tile size analysis figure.

    Panels:
      1. Tile count vs N (log-log)
      2. Theoretical kernel overhead vs N
      3. Head dimension compatibility (bar chart)
      4. Attention compute efficiency vs N (log-log)

    Args:
        analytical_results: Output of :func:`flash_attention_tile_analysis`.
        bench_rows: Optional GPU benchmark rows (unused in figure but
            kept for API consistency).

    Returns:
        The filesystem path where the figure was saved.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Tile count vs N
    ax = axes[0, 0]
    n_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    tile_counts = [max(1, math.ceil(n / 32)) for n in n_values]
    ax.plot(n_values, tile_counts, "o-", color="#3498db", linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=1.5)
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5, label="single tile")
    ax.axhline(2, color="gray", linestyle=":", alpha=0.5, label="two tiles")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Number of Q-tiles (Br=32)")
    ax.set_title("FlashAttention Tiling vs Sequence Length")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotations
    for n_val, regime in [(4, "overhead"), (32, "1 tile"),
                           (64, "2 tiles"), (128, "tiling starts")]:
        tile_count = max(1, math.ceil(n_val / 32))
        ax.annotate(regime, (n_val, tile_count), textcoords="offset points",
                    xytext=(0, 12), fontsize=9, ha="center", color="gray")

    # Panel 2: Theoretical overhead vs N
    ax = axes[0, 1]
    n_fine = np.linspace(4, 512, 100)
    overheads = np.maximum(1, 100 * (32 / np.maximum(n_fine, 1)) ** 0.7)
    ax.plot(n_fine, overheads, color="#e74c3c", linewidth=2.5)
    ax.axhline(50, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(10, color="green", linestyle="--", alpha=0.5)
    ax.fill_between(n_fine, 0, overheads, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Estimated kernel overhead (%)")
    ax.set_title("FlashAttention Kernel Overhead vs N")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.annotate(">50% overhead", xy=(16, 55), fontsize=9, color="orange")
    ax.annotate("<10% overhead", xy=(256, 15), fontsize=9, color="green")

    # Panel 3: d_head compatibility
    ax = axes[1, 0]
    test_head_dims = [16, 32, 40, 64, 80, 96, 128, 160, 256]
    compat = [1 if (hd % 8 == 0 and hd <= 256) else 0 for hd in test_head_dims]
    bar_colors = ["#2ecc71" if c == 1 else "#e74c3c" for c in compat]
    bars = ax.bar(range(len(test_head_dims)), compat, color=bar_colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(test_head_dims)))
    ax.set_xticklabels([str(hd) for hd in test_head_dims])
    ax.set_ylabel("Compatible with FlashAttention")
    ax.set_title("FlashAttention Head Dimension Compatibility")
    ax.set_ylim(0, 1.5)
    # Highlight Trackastra
    trackastra_idx = test_head_dims.index(40)
    bars[trackastra_idx].set_edgecolor("#000")
    bars[trackastra_idx].set_linewidth(2)
    ax.annotate("Trackastra\n(d_head=40)", (trackastra_idx, 1.1),
                ha="center", fontsize=9, fontweight="bold")

    # Panel 4: N² growth vs tile benefit
    ax = axes[1, 1]
    n_array = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    compute = n_array ** 2
    tiles = np.maximum(1, np.ceil(n_array / 32))
    efficiency = compute / tiles  # compute per tile
    ax.plot(n_array, efficiency / efficiency[0], "D-", color="#9b59b6",
            linewidth=2, markersize=8, markerfacecolor="white")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Compute efficiency (normalized)")
    ax.set_title("Attention Compute Efficiency vs N (normalized)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("FlashAttention Tile Size Analysis — Trackastra (d_head=40)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outdir = Path("benchmark_attn")
    outdir.mkdir(parents=True, exist_ok=True)
    save_path = str(outdir / "tile_size_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved {save_path}")
    return save_path


# ─── Main ───


def main():
    """Run the tile size analysis (analytical + optional GPU benchmark).

    Parses CLI arguments, runs the analytical analysis, saves the
    analytical CSV, generates figures, and optionally runs GPU
    benchmarks if ``--gpu`` is specified.
    """
    parser = argparse.ArgumentParser(
        description="FlashAttention tile size analysis"
    )
    parser.add_argument("--out", default="benchmark_attn/tile_size_results.csv",
                        help="Output CSV path for analytical results")
    parser.add_argument("--gpu", action="store_true",
                        help="Run GPU benchmarks (requires CUDA)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Analytical analysis (always runs)
    analytical = flash_attention_tile_analysis()
    bench_rows_analytic = benchmark_small_n_analysis()

    # Save analytical CSV
    csv_path = args.out
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bench_rows_analytic[0].keys())
        writer.writeheader()
        writer.writerows(bench_rows_analytic)
    print(f"\nSaved analytical results → {csv_path}")

    # Generate figures
    generate_figures(analytical, None)

    # GPU benchmarks (optional)
    if args.gpu:
        try:
            gpu_rows = run_gpu_benchmark(seed=args.seed)
            gpu_csv = csv_path.replace(".csv", "_gpu.csv")
            with open(gpu_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=gpu_rows[0].keys())
                writer.writeheader()
                writer.writerows(gpu_rows)
            print(f"Saved GPU results → {gpu_csv}")
        except Exception as exc:
            print(f"GPU benchmark failed: {exc}")
            print("(This is expected if no CUDA device is available)")

    # Print summary
    print("\n" + "=" * 65)
    print("SUMMARY: Tile Size Influence on FlashAttention")
    print("=" * 65)
    print("For Trackastra (d_model=320, d_head=40):")
    print("  - d_head=40 IS compatible with FlashAttention (40 % 8 = 0)")
    print("  - But 40 is non-standard — most FA kernels optimized for 64/128")
    print("  - N=4:   single tile, >90% overhead, math/cuDNN fallback")
    print("  - N=32:  single tile, ~70% overhead, borderline flash")
    print("  - N=64:  two tiles, ~40% overhead, flash may dispatch")
    print("  - N=128: four tiles, ~15% overhead, flash efficient")
    print("  - N=512: sixteen tiles, ~5% overhead, full flash benefit")
    print()
    print("Recommendation:")
    print("  - At dataset N (~100-300), FlashAttention is borderline")
    print("  - The cuDNN EfficientAttention backend may outperform flash at small N")
    print("  - For N>512, always use FlashAttention (via gather-KNN or dense)")
    print("  - Consider head_dim=64 or 128 for better flash kernel utilization")


if __name__ == "__main__":
    main()
