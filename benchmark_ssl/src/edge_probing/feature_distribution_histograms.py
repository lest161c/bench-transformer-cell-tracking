"""Per-feature distribution histograms for probing sanity checks.

Extracts regionprops features from real microscopy frames, then generates
two sets of per-feature histograms for comparison:

  without_normalization/  — raw feature values as extracted
  with_normalization/     — same features after z-score standardization

A heavily skewed feature will distort the z-score mapping (outliers pull
the mean, compressing the bulk), which in turn biases any linear probe.

Usage::

    python -m src.edge_probing.feature_distribution_histograms \\
        --level hu --max-frames 200

Output goes to ``results/without_normalization/`` and
``results/with_normalization/``.
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feature_dist")

# Feature names for each extraction level.  These must match the ordering
# returned by ``src.data.extract()``.
FEATURE_NAMES = {
    "basic": ["eq_diam", "intensity", "inertia_00", "inertia_01",
              "inertia_10", "inertia_11", "border_dist"],
    "shape": ["eq_diam", "intensity", "inertia_00", "inertia_01",
              "inertia_10", "inertia_11", "border_dist",
              "eccentricity", "perimeter", "solidity", "extent",
              "axis_major", "axis_minor", "orientation"],
    "hu": ["eq_diam", "intensity", "inertia_00", "inertia_01",
           "inertia_10", "inertia_11", "border_dist",
           "eccentricity", "perimeter", "solidity", "extent",
           "axis_major", "axis_minor", "orientation",
           "hu_0", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6"],
    "patch": None,  # determined dynamically
}


def load_frames(config_path, conditions, max_frames):
    """Load experiment frames from the SSL benchmark config.

    Args:
        config_path: Path to the benchmark_ssl config YAML.
        conditions: Comma-separated condition names.
        max_frames: Maximum number of frames to load.

    Returns:
        List of frame tuples from ``load_experiment_frames``.
    """
    from src.data import load_experiment_frames

    with open(config_path) as file_handle:
        cfg = yaml.safe_load(file_handle)
    cfg["conditions"] = [c.strip() for c in conditions.split(",")]

    frames = load_experiment_frames(cfg["data_root"], conditions=cfg["conditions"])
    logger.info(f"Total frames: {len(frames)}")
    return frames[:max_frames]


def collect_features(frames, level="hu"):
    """Extract and collect features from all frames.

    Iterates over every frame, extracts the requested feature level
    using ``src.data.extract()``, and concatenates the per-cell
    feature vectors into a single array.

    Args:
        frames: List of frame tuples.
        level: Feature extraction level ("basic", "shape", "hu", "patch").

    Returns:
        Tuple ``(features, feature_names)`` where ``features`` is an
        ``(N_cells, D)`` float32 array and ``feature_names`` is a list
        of length ``D``.
    """
    from src.data import extract as extract_rich

    all_features = []
    n_frames = 0

    for frame in frames:
        _, _, _, mask_path, img_path = frame
        mask = imread(mask_path)
        img = imread(img_path).astype(np.float32)
        p1, p998 = np.percentile(img, (1, 99.8))
        img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

        result = extract_rich(level, mask, img, ndim=2)
        if result is None or len(result[1]) < 2:
            continue

        _, _, feats_dict = result
        feats = np.concatenate(list(feats_dict.values()), axis=-1).astype(np.float32)
        all_features.append(feats)

        n_frames += 1
        if n_frames % 10 == 0:
            logger.info(f"  Processed {n_frames} frames...")

    if not all_features:
        raise RuntimeError("No features collected — check data paths and config.")

    features = np.concatenate(all_features, axis=0)
    feature_names = FEATURE_NAMES.get(level)

    if level == "patch":
        feature_names = [f"patch_{i}" for i in range(features.shape[1])]

    logger.info(f"Collected {features.shape[0]} cells, {features.shape[1]} features from {n_frames} frames")
    return features, feature_names


def plot_feature_histograms(features, feature_names, output_dir, title_suffix=""):
    """Plot one histogram per feature dimension.

    Generates a multi-panel figure where each panel shows the distribution
    of a single feature across all cells.  The number of panels is
    ``min(n_features, 25)``; if there are more features, only the first
    25 are plotted.

    Args:
        features: ``(N, D)`` feature array.
        feature_names: List of ``D`` feature names.
        output_dir: Directory to save the PNG figure.
        title_suffix: Extra text appended to the figure title (e.g.
            "— Raw" or "— Z-scored").
    """
    n_features = features.shape[1]
    n_plot = min(n_features, 25)
    n_cols = min(5, n_plot)
    n_rows = (n_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).flatten()

    for i in range(n_plot):
        ax = axes[i]
        vals = features[:, i]
        ax.hist(vals, bins=50, color="#3498db", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.set_title(feature_names[i], fontsize=9)
        ax.tick_params(labelsize=7)
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        skew = float(((vals - mean_val) ** 3).mean() / (std_val ** 3 + 1e-12))
        ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1)
        ax.text(0.02, 0.95, f"μ={mean_val:.2f}\nσ={std_val:.2f}\nskew={skew:.2f}",
                transform=ax.transAxes, fontsize=6, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    for i in range(n_plot, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Per-Feature Distributions {title_suffix}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_distributions.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def main():
    """Generate per-feature distribution histograms with and without normalization."""
    parser = argparse.ArgumentParser(description="Per-feature distribution histograms")
    parser.add_argument("--config", default="configs/config.yaml", help="benchmark_ssl config path")
    parser.add_argument("--conditions", default="rpsM", help="comma-separated conditions")
    parser.add_argument("--max-frames", type=int, default=200, help="maximum frames to process")
    parser.add_argument("--level", default="hu", choices=["basic", "shape", "hu", "patch"],
                        help="feature extraction level")
    parser.add_argument("--outdir", default="results", help="base output directory")
    args = parser.parse_args()

    frames = load_frames(args.config, args.conditions, args.max_frames)
    features, feature_names = collect_features(frames, level=args.level)

    base_outdir = Path(args.outdir)
    raw_dir = base_outdir / "without_normalization"
    norm_dir = base_outdir / "with_normalization"

    # --- Without normalization (raw features) ---
    plot_feature_histograms(
        features, feature_names, raw_dir, title_suffix="— Raw (no normalization)"
    )

    # --- With z-score normalization ---
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.maximum(std, 1e-6)  # clamp constant features
    features_standardized = (features - mean) / std

    plot_feature_histograms(
        features_standardized, feature_names, norm_dir, title_suffix="— Z-scored (mean=0, std=1)"
    )

    # Console summary
    print("\n" + "=" * 70)
    print(f"PER-FEATURE DISTRIBUTION SUMMARY ({args.level} level)")
    print("=" * 70)
    print(f"{'Feature':<20} {'Mean':>10} {'Std':>10} {'Skew':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    for i, name in enumerate(feature_names):
        vals = features[:, i]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        skew = float(((vals - mean_val) ** 3).mean() / (std_val ** 3 + 1e-12))
        print(f"{name:<20} {mean_val:>10.2f} {std_val:>10.2f} {skew:>10.2f} "
              f"{np.min(vals):>10.2f} {np.max(vals):>10.2f}")

    print(f"\nOutputs:")
    print(f"  Raw:       {raw_dir / 'feature_distributions.png'}")
    print(f"  Z-scored:  {norm_dir / 'feature_distributions.png'}")


if __name__ == "__main__":
    main()
