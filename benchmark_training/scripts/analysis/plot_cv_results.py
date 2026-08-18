#!/usr/bin/env python3
"""Generate figures for the report rework.

Produces:
1. CV comparison: TRA and AOGM for masked vs dense_flash across 6 folds
2. Amdahl's law: training speedup vs attention share of step
3. Per-epoch time comparison: CV runs + vanilla vs optimal
4. Attention share of step at different N values

Usage:
    python plot_cv_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths
TRK = Path("/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601")
BENCH = Path("/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking")
TRACKASTRA_RUNS = TRK / "trackastra" / "runs"
OUTPUT_DIR = BENCH / "report" / "resources" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["rpsM", "recA", "pheA", "metA", "cib", "trpL"]


def load_cv_results():
    """Load all 12 CV results."""
    masked = {}
    flash = {}
    for d in sorted(TRACKASTRA_RUNS.glob("2026-08-18_*cv*")):
        name = d.name
        sfile = d / "cv_eval" / "summary.json"
        if not sfile.exists():
            continue
        with open(sfile) as f:
            s = json.load(f)
        fold = int(name.split("fold")[1].split("_")[0])
        if "cv_masked" in name:
            masked[fold] = s
        elif "cv_dense_flash" in name:
            flash[fold] = s
    return masked, flash


def plot_cv_comparison(masked, flash):
    """Plot TRA and AOGM comparison across 6 folds."""
    common = sorted(set(masked.keys()) & set(flash.keys()))
    folds = [CONDITIONS[f] for f in common]

    masked_tra = [masked[f]["mean_TRA"] for f in common]
    flash_tra = [flash[f]["mean_TRA"] for f in common]
    masked_aogm = [masked[f]["mean_AOGM"] for f in common]
    flash_aogm = [flash[f]["mean_AOGM"] for f in common]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(common))
    width = 0.35

    bars1 = ax1.bar(x - width/2, masked_tra, width, label='Masked (knn=-1)', color='tab:blue')
    bars2 = ax1.bar(x + width/2, flash_tra, width, label='Dense Flash (knn=-2)', color='tab:orange')
    ax1.set_ylabel('TRA')
    ax1.set_title('Tracking Accuracy by Fold (6-Fold CV)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(folds, rotation=45)
    ax1.legend()
    ax1.set_ylim(0.9, 1.0)
    for i, (m, f) in enumerate(zip(masked_tra, flash_tra)):
        ax1.text(i - width/2, m + 0.002, f'{m:.3f}', ha='center', fontsize=8)
        ax1.text(i + width/2, f + 0.002, f'{f:.3f}', ha='center', fontsize=8)

    bars1 = ax2.bar(x - width/2, masked_aogm, width, label='Masked (knn=-1)', color='tab:blue')
    bars2 = ax2.bar(x + width/2, flash_aogm, width, label='Dense Flash (knn=-2)', color='tab:orange')
    ax2.set_ylabel('AOGM')
    ax2.set_title('AOGM by Fold (6-Fold CV)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(folds, rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_cv_comparison.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "14_cv_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Print summary
    print("CV Results Summary:")
    print(f"  Masked:      TRA = {np.mean(masked_tra):.4f} ± {np.std(masked_tra):.4f}")
    print(f"  Dense Flash: TRA = {np.mean(flash_tra):.4f} ± {np.std(flash_tra):.4f}")
    print(f"  TRA diff:    {np.mean(flash_tra) - np.mean(masked_tra):+.4f}")
    print(f"  Masked:      AOGM = {np.mean(masked_aogm):.1f} ± {np.std(masked_aogm):.1f}")
    print(f"  Dense Flash: AOGM = {np.mean(flash_aogm):.1f} ± {np.std(flash_aogm):.1f}")
    print(f"  AOGM diff:   {np.mean(flash_aogm) - np.mean(masked_aogm):+.1f}")


def plot_amdahl():
    """Plot Amdahl's law: training speedup vs attention share of step."""
    fig, ax = plt.subplots(figsize=(8, 5))

    attn_speedups = [1.5, 2.0, 3.0, 5.0, 10.0, 16.0]
    colors = ['tab:gray', 'tab:olive', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    labels = ['1.5× (labbook)', '2.0×', '3.0× (measured)', '5.0×', '10.0×', '16.0× (N=8192)']

    attn_share = np.linspace(0.01, 1.0, 200)

    for speedup, color, label in zip(attn_speedups, colors, labels):
        training_speedup = 1.0 / (1.0 - attn_share + attn_share / speedup)
        ax.plot(attn_share * 100, training_speedup, color=color, linewidth=2, label=label)

    # Mark vanvliet point
    ax.axvline(x=5, color='tab:cyan', linestyle='--', alpha=0.7)
    ax.axhline(y=1.035, color='tab:cyan', linestyle='--', alpha=0.7)
    ax.plot(5, 1.035, 'o', color='tab:cyan', markersize=10, zorder=5)
    ax.annotate('Vanvliet (N≈140)\n5% attention share\n3.5% training speedup',
                xy=(5, 1.035), xytext=(15, 1.4),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Attention share of step time (%)')
    ax.set_ylabel('Training speedup (×)')
    ax.set_title("Amdahl's Law: Training Speedup vs Attention Share")
    ax.legend(loc='upper right')
    ax.set_xlim(0, 100)
    ax.set_ylim(1.0, 17)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_amdahl_law.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "15_amdahl_law.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_per_epoch_time():
    """Plot per-epoch training times."""
    fig, ax = plt.subplots(figsize=(8, 5))

    runs = ['vanilla_train\n(main branch)', 'optimal_train\n(cached-dist-attn)', 'CV masked\n(knn=-1)', 'CV dense_flash\n(knn=-2)']
    times = [214.0, 219.4, 217.7, 211.4]
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    configs = ['RelativePositional\n+ skimage\n+ serial', 'CachedDist\n+ fast-regionprops\n+ vectorized', 'CachedDist + mask\n+ all optimizations', 'DenseFlash\n+ all optimizations']

    x = np.arange(len(runs))
    bars = ax.bar(x, times, color=colors)

    for i, (bar, t) in enumerate(zip(bars, times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{t:.1f}s\n({t/60:.1f} min)', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylabel('Per-epoch training time (s)')
    ax.set_title('Per-Epoch Training Time Comparison (window=4, N≈140)')
    ax.set_ylim(0, 250)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "16_per_epoch_time.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "16_per_epoch_time.png", bbox_inches="tight", dpi=150)
    plt.close()


def plot_attention_share():
    """Plot attention share of step time at different N values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    N = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    attn_share = [2, 3, 5, 10, 25, 55, 75, 95]

    ax.plot(N, attn_share, 'o-', color='tab:blue', linewidth=2, markersize=8)

    # Highlight vanvliet scale
    ax.axvspan(100, 200, alpha=0.2, color='tab:cyan', label='Vanvliet scale (N≈140)')
    ax.axhline(y=5, color='tab:cyan', linestyle='--', alpha=0.7)
    ax.annotate('5% attention share\nat vanvliet scale',
                xy=(140, 5), xytext=(300, 30),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xscale('log')
    ax.set_xlabel('Number of tokens (N)')
    ax.set_ylabel('Attention share of step time (%)')
    ax.set_title('Attention Share of Step Time vs N')
    ax.set_xlim(50, 10000)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_attention_share.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "17_attention_share.png", bbox_inches="tight", dpi=150)
    plt.close()


def main():
    print("Generating CV comparison plot...")
    masked, flash = load_cv_results()
    plot_cv_comparison(masked, flash)

    print("Generating Amdahl's law plot...")
    plot_amdahl()

    print("Generating per-epoch time plot...")
    plot_per_epoch_time()

    print("Generating attention share plot...")
    plot_attention_share()

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
