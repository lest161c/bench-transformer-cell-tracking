"""Per-feature mean/std verification after z-score standardization.

Histograms cannot reveal whether z-score standardization was applied: the
z-score mapping is a per-feature linear transform, so the shape of every
feature distribution is preserved and the normalized and unnormalized
histograms look identical.  A working standardization is instead verified
by per-feature statistics — after z = (x - mean) / std every feature must
have mean ≈ 0 and std ≈ 1.

This script is a diagnostic / sanity-check tool, not an automated
verification pipeline.  It extracts regionprops features from real
microscopy frames, computes the z-score statistics exactly as the linear
probe does (population std, clamped to >= 1e-6), applies the transform,
and produces a four-panel figure plus a console summary table:

  Panel A — per-feature mean, raw (blue) vs standardized (green)
  Panel B — per-feature std, raw (blue) vs standardized (green)
  Panel C — per-feature mean after standardization (target 0, annotated)
  Panel D — per-feature std after standardization (target 1, annotated)

Raw features span wildly different scales (centroids ~10 px, intensities
~[0, 1], Hu moments ~1e-3), so in panels A/B the raw bars dominate and the
standardized bars sit at the zero line; panels C and D are the decisive
diagnostics because they isolate the standardized statistics.

Usage::

    python -m src.edge_probing.verify_normalization \\
        --level hu --max-frames 200

Output goes to ``results/normalization_verification/``.
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
logger = logging.getLogger("verify_normalization")

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


def compute_standardization_stats(features):
    """Compute per-feature z-score statistics (mean, std).

    Mirrors ``fit_feature_standardizer`` in ``unified_edge_probe.py``: the
    mean and population std (ddof=0) are taken over all cells per feature,
    and std is clamped to >= 1e-6 so constant features stay finite and map
    to 0 after transformation.

    Args:
        features: ``(N_cells, D)`` float32 array.

    Returns:
        Tuple ``(mean, std)`` of ``(D,)`` arrays.  ``mean`` is the raw
        per-feature mean; ``std`` is the per-feature standard deviation
        clamped to >= 1e-6.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean, std


def apply_standardization(features, mean, std):
    """Apply the per-feature z-score transform z = (x - mean) / std.

    Args:
        features: ``(N_cells, D)`` float32 array.
        mean, std: ``(D,)`` arrays of per-feature statistics.

    Returns:
        ``(N_cells, D)`` float32 array of standardized features.
    """
    std = np.maximum(std, 1e-6)  # guard against zero / near-zero std
    return (features - mean) / std


def plot_normalization_verification(features_raw, features_standardized, feature_names, output_dir):
    """Plot per-feature mean/std before and after standardization.

    Produces a four-panel figure: raw-vs-standardized mean (A),
    raw-vs-standardized std (B), standardized mean annotated per feature
    (C, target 0), and standardized std annotated per feature (D, target 1).
    Panels C and D are the decisive diagnostics: with a correct z-score
    transform all bars sit on 0 and 1 respectively.

    Args:
        features_raw: ``(N_cells, D)`` unstandardized feature array.
        features_standardized: ``(N_cells, D)`` z-scored feature array.
        feature_names: List of ``D`` feature names.
        output_dir: Directory where the PNG is saved (created if missing).
    """
    mean_raw = features_raw.mean(axis=0)
    std_raw = features_raw.std(axis=0)
    mean_standardized = features_standardized.mean(axis=0)
    std_standardized = features_standardized.std(axis=0)

    n_features = len(feature_names)
    x_positions = np.arange(n_features)
    bar_width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Normalization Verification — per-feature mean/std before and after z-score",
        fontsize=14, fontweight="bold",
    )

    # Panel A: per-feature mean, raw (blue) vs standardized (green).
    ax_mean = axes[0, 0]
    ax_mean.bar(x_positions - bar_width / 2, mean_raw, width=bar_width,
                label="Raw", color="#3498db")
    ax_mean.bar(x_positions + bar_width / 2, mean_standardized, width=bar_width,
                label="Standardized", color="#2ecc71")
    ax_mean.set_xticks(x_positions)
    ax_mean.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax_mean.set_ylabel("Mean value")
    ax_mean.set_title("A — Per-feature mean (raw vs standardized)", fontsize=11)
    ax_mean.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_mean.legend(fontsize=9)

    # Panel B: per-feature std, raw (blue) vs standardized (green).
    ax_std = axes[0, 1]
    ax_std.bar(x_positions - bar_width / 2, std_raw, width=bar_width,
               label="Raw", color="#3498db")
    ax_std.bar(x_positions + bar_width / 2, std_standardized, width=bar_width,
               label="Standardized", color="#2ecc71")
    ax_std.set_xticks(x_positions)
    ax_std.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax_std.set_ylabel("Std value")
    ax_std.set_title("B — Per-feature std (raw vs standardized)", fontsize=11)
    ax_std.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax_std.legend(fontsize=9)

    # Panel C: per-feature mean after standardization, annotated with values.
    ax_mean_scaled = axes[1, 0]
    ax_mean_scaled.bar(x_positions, mean_standardized, width=bar_width * 2,
                       color="#2ecc71")
    ax_mean_scaled.axhline(0, color="black", linewidth=0.8, linestyle="--")
    max_abs_mean = max(abs(value) for value in mean_standardized) if n_features else 0.0
    mean_text_offset = max(max_abs_mean * 0.08, 0.02)
    for x_pos, value in zip(x_positions, mean_standardized):
        label_y = value + mean_text_offset if value >= 0 else value - mean_text_offset
        label_va = "bottom" if value >= 0 else "top"
        ax_mean_scaled.text(x_pos, label_y, f"{value:.4f}", ha="center", va=label_va,
                            fontsize=8, rotation=90)
    ax_mean_scaled.set_xticks(x_positions)
    ax_mean_scaled.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax_mean_scaled.set_ylabel("Mean after z-score")
    ax_mean_scaled.set_title("C — Standardized mean (target 0)", fontsize=11)

    # Panel D: per-feature std after standardization, annotated with values.
    ax_std_scaled = axes[1, 1]
    ax_std_scaled.bar(x_positions, std_standardized, width=bar_width * 2,
                      color="#2ecc71")
    ax_std_scaled.axhline(1, color="black", linewidth=0.8, linestyle="--")
    std_text_offset = max(float(np.max(std_standardized)) * 0.03, 0.02)
    for x_pos, value in zip(x_positions, std_standardized):
        ax_std_scaled.text(x_pos, value + std_text_offset, f"{value:.4f}",
                           ha="center", va="bottom", fontsize=8, rotation=90)
    ax_std_scaled.set_xticks(x_positions)
    ax_std_scaled.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax_std_scaled.set_ylabel("Std after z-score")
    ax_std_scaled.set_title("D — Standardized std (target 1)", fontsize=11)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "normalization_verification.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def print_summary_table(features_raw, features_standardized, feature_names):
    """Print a console table of per-feature mean/std before and after z-score.

    Args:
        features_raw: ``(N_cells, D)`` unstandardized feature array.
        features_standardized: ``(N_cells, D)`` z-scored feature array.
        feature_names: List of ``D`` feature names.
    """
    mean_raw = features_raw.mean(axis=0)
    std_raw = features_raw.std(axis=0)
    mean_standardized = features_standardized.mean(axis=0)
    std_standardized = features_standardized.std(axis=0)

    print("\n" + "=" * 78)
    print("NORMALIZATION VERIFICATION SUMMARY")
    print("=" * 78)
    print(f"{'Feature':<18} {'Raw mean':>12} {'Raw std':>12} "
          f"{'Std mean':>12} {'Std std':>12}")
    print("-" * 78)
    for i, name in enumerate(feature_names):
        print(f"{name:<18} {mean_raw[i]:>12.4f} {std_raw[i]:>12.4f} "
              f"{mean_standardized[i]:>12.4f} {std_standardized[i]:>12.4f}")
    print("=" * 78)
    print("Expected after correct z-score: mean ≈ 0 and std ≈ 1 for every feature.")


def main():
    """Generate the normalization-verification figure and summary table."""
    parser = argparse.ArgumentParser(description="Verify z-score standardization (per-feature mean/std)")
    parser.add_argument("--config", default="configs/config.yaml", help="benchmark_ssl config path")
    parser.add_argument("--conditions", default="rpsM", help="comma-separated conditions")
    parser.add_argument("--max-frames", type=int, default=200, help="maximum frames to process")
    parser.add_argument("--level", default="hu", choices=["basic", "shape", "hu", "patch"],
                        help="feature extraction level")
    parser.add_argument("--outdir", default="results/normalization_verification",
                        help="output directory for the figure")
    args = parser.parse_args()

    frames = load_frames(args.config, args.conditions, args.max_frames)
    features_raw, feature_names = collect_features(frames, level=args.level)

    mean, std = compute_standardization_stats(features_raw)
    features_standardized = apply_standardization(features_raw, mean, std)

    plot_normalization_verification(features_raw, features_standardized, feature_names, args.outdir)
    print_summary_table(features_raw, features_standardized, feature_names)

    print(f"\nOutput:")
    print(f"  Figure:  {Path(args.outdir) / 'normalization_verification.png'}")


if __name__ == "__main__":
    main()
