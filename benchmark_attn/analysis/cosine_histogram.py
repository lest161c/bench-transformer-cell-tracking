"""Cosine similarity histogram: intra-cell vs inter-cell distributions.

Generates full distributions from real (distorted) regionprops data and
DINOv2/3 features, showing both intra-sample and inter-sample histograms.
Addresses meeting_18_06_26.txt lines 4-8.

Output:
    benchmark_attn/cosine_similarity_histogram.png   — multi-panel figure
    benchmark_attn/cosine_similarity_results.csv     — summary statistics
"""

import argparse
import csv
import logging
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cosine_histogram")

warnings.filterwarnings("ignore", category=UserWarning)


def load_benchmark_ssl_modules():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark_ssl"))
    from ssl_pipeline import load_experiment_frames
    from rich_features import extract as extract_rich, FEATURE_DIMS, FEATURE_NAMES
    from distortions import (
        AffineDistortion, ElasticDistortion, JitterDistortion, DropoutDistortion,
        PhotometricDistortion, FeatureNoise, DistortionPipeline,
    )
    return (load_experiment_frames, extract_rich, FEATURE_DIMS, FEATURE_NAMES,
            DistortionPipeline)


def compute_histograms(feats1, feats2, labels1, labels2, nbins=50):
    """Compute intra- and inter-cell cosine similarity histograms.

    Returns:
        intra:  (nbins,) bin counts for same-cell pairs
        inter:  (nbins,) bin counts for different-cell pairs
        bins:   (nbins+1,) bin edges
        stats:  dict with mean, std, median per category
    """
    f1 = feats1 / (np.linalg.norm(feats1, axis=1, keepdims=True) + 1e-12)
    f2 = feats2 / (np.linalg.norm(feats2, axis=1, keepdims=True) + 1e-12)
    sim = f1 @ f2.T

    intra_vals, inter_vals = [], []
    for i, lbl_i in enumerate(labels1):
        matches = np.where(labels2 == lbl_i)[0]
        for j in range(len(labels2)):
            if j in matches:
                intra_vals.append(float(sim[i, j]))
            else:
                inter_vals.append(float(sim[i, j]))

    intra_vals = np.array(intra_vals)
    inter_vals = np.array(inter_vals)
    bins = np.linspace(-0.2, 1.0, nbins + 1)

    intra_hist, _ = np.histogram(intra_vals, bins=bins)
    inter_hist, _ = np.histogram(inter_vals, bins=bins)

    stats = {}
    for name, vals in [("intra", intra_vals), ("inter", inter_vals)]:
        if len(vals) > 0:
            stats[f"{name}_mean"] = float(np.mean(vals))
            stats[f"{name}_std"] = float(np.std(vals))
            stats[f"{name}_median"] = float(np.median(vals))
            stats[f"{name}_skew"] = float(((vals - np.mean(vals)) ** 3).mean() / (np.std(vals) ** 3 + 1e-12))
        else:
            stats[f"{name}_mean"] = np.nan
            stats[f"{name}_std"] = np.nan
            stats[f"{name}_median"] = np.nan
            stats[f"{name}_skew"] = np.nan

    stats["sep_gap"] = stats.get("intra_mean", np.nan) - stats.get("inter_mean", np.nan)
    stats["n_intra_pairs"] = len(intra_vals)
    stats["n_inter_pairs"] = len(inter_vals)

    return intra_hist, inter_hist, bins, stats


def extract_dino_features(patches, device="cuda", use_v3=False):
    """Extract DINOv2 or DINOv3 features from cell patches.

    Args:
        patches: (N, H, W) or (N, C, H, W) numpy array of cell patches.
        device: torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        use_v3: If ``True``, load DINOv3 (``dinov3_vits16``);
            otherwise load DINOv2 (``dinov2_vits14``).

    Returns:
        ``(N, 384)`` numpy array of features, or ``None`` on failure.
    """
    try:
        import torch
        import torch.nn.functional as F

        if use_v3:
            dino = torch.hub.load("facebookresearch/dinov3", "dinov3_vits16")
        else:
            dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        dino = dino.to(device).eval()

        feats_list = []
        batch_size = 32
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start:start + batch_size]).float().to(device)
            if batch.dim() == 3:
                batch = batch.unsqueeze(1)
            batch = F.interpolate(batch, size=(224, 224), mode="bilinear", align_corners=False)
            batch = batch.expand(-1, 3, -1, -1)
            dino_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            dino_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            batch = (batch - dino_mean) / dino_std
            with torch.no_grad():
                feats = dino(batch).cpu().numpy()
            feats_list.append(feats)
        return np.concatenate(feats_list, axis=0)
    except Exception as exc:
        logger.warning(f"DINO extraction failed: {exc}")
        return None


def extract_patches(img, centroids, patch_size=64):
    """Extract square patches around centroids."""
    height, width = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i = int(round(float(cy)))
        cx_i = int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, height - 1)
        cx_i = np.clip(cx_i, 0, width - 1)
        y1, y2 = cy_i - half, cy_i + half
        x1, x2 = cx_i - half, cx_i + half
        pt = max(0, -y1); pb = max(0, y2 - height)
        pl = max(0, -x1); pr = max(0, x2 - width)
        y1c, y2c = max(0, y1), min(height, y2)
        x1c, x2c = max(0, x1), min(width, x2)
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(crop, ((0, max(0, patch_size - crop.shape[0])),
                                 (0, max(0, patch_size - crop.shape[1]))),
                          mode="reflect")[:patch_size, :patch_size]
        patches.append(crop)
    return np.stack(patches) if patches else np.zeros((0, patch_size, patch_size), dtype=np.float32)


def generate_histogram_figure(feature_results, save_path):
    """Multi-panel figure with histograms for each feature type."""
    n_features = len(feature_results)
    fig, axes = plt.subplots(2, n_features, figsize=(5 * n_features, 9))

    if n_features == 1:
        axes = axes.reshape(2, 1)

    bin_centers = None
    for fi, (fname, data) in enumerate(feature_results.items()):
        intra, inter, bins, stats = data
        bc = (bins[:-1] + bins[1:]) / 2

        # Top row: overlaid histograms
        ax = axes[0, fi]
        ax.bar(bc, intra, width=np.diff(bins)[0], alpha=0.7, color="#2ecc71",
               edgecolor="white", linewidth=0.3,
               label=f"intra-cell\nμ={stats['intra_mean']:.3f}, σ={stats['intra_std']:.3f}")
        ax.bar(bc, inter, width=np.diff(bins)[0], alpha=0.5, color="#e74c3c",
               edgecolor="white", linewidth=0.3,
               label=f"inter-cell\nμ={stats['inter_mean']:.3f}, σ={stats['inter_std']:.3f}")
        ax.axvline(stats["intra_mean"], color="#2ecc71", linestyle="--", linewidth=1.5)
        ax.axvline(stats["inter_mean"], color="#e74c3c", linestyle="--", linewidth=1.5)
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("pair count")
        ax.set_title(f"{fname}\ngap = {stats['sep_gap']:.4f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

        # Bottom row: CDF / cumulative distribution
        ax = axes[1, fi]
        intra_cdf = np.cumsum(intra) / max(intra.sum(), 1)
        inter_cdf = np.cumsum(inter) / max(inter.sum(), 1)
        ax.plot(bc, intra_cdf, color="#2ecc71", linewidth=2, label="intra-cell")
        ax.plot(bc, inter_cdf, color="#e74c3c", linewidth=2, label="inter-cell")
        ax.axvline(stats["intra_mean"], color="#2ecc71", linestyle=":", linewidth=1)
        ax.axvline(stats["inter_mean"], color="#e74c3c", linestyle=":", linewidth=1)
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("cumulative fraction")
        ax.set_title(f"CDF — KS distance = {np.max(np.abs(intra_cdf - inter_cdf)):.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        bin_centers = bc

    fig.suptitle("Cosine Similarity Distributions: Intra-Cell vs Inter-Cell",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved histogram figure to {save_path}")
    return fig


def main():
    """Generate cosine similarity histograms from experiment frames.

    Loads frames from the SSL benchmark, extracts regionprops and DINO
    features, computes intra-cell and inter-cell cosine similarity
    distributions, saves a multi-panel PNG figure and a CSV summary.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="benchmark_ssl config path")
    parser.add_argument("--conditions", default="rpsM", help="comma-separated conditions")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--outdir", default="benchmark_attn")
    parser.add_argument("--no-dino", action="store_true", help="skip DINO feature extraction")
    parser.add_argument("--dinov3", action="store_true", help="use DINOv3 instead of DINOv2")
    parser.add_argument("--nbins", type=int, default=50)
    args = parser.parse_args()

    (load_experiment_frames, extract_rich, FEATURE_DIMS, FEATURE_NAMES,
     DistortionPipeline) = load_benchmark_ssl_modules()

    with open(args.config) as file_handle:
        cfg = yaml.safe_load(file_handle)
    cfg["conditions"] = [c.strip() for c in args.conditions.split(",")]

    frames = load_experiment_frames(cfg["data_root"], conditions=cfg["conditions"])
    logger.info(f"Total frames: {len(frames)}")

    dist = DistortionPipeline.from_config(cfg)

    # Collect samples
    rp_intra_all, rp_inter_all = [], []
    dino_intra_all, dino_inter_all = [], []
    n_frames = 0

    for idx in range(min(len(frames), args.max_frames)):
        _, _, _, mask_path, img_path = frames[idx]
        mask = imread(mask_path)
        img = imread(img_path).astype(np.float32)
        p1, p998 = np.percentile(img, (1, 99.8))
        img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

        result = extract_rich("basic", mask, img, ndim=2)
        if result is None or len(result[1]) < 2:
            continue
        coords_src, labels_src, feats_dict_src = result
        feats_orig = np.concatenate(list(feats_dict_src.values()), axis=-1).astype(np.float32)

        c1, f1_dict, l1, c2, f2_dict, l2 = dist(coords_src, feats_dict_src, labels_src)
        f1 = np.concatenate(list(f1_dict.values()), axis=-1).astype(np.float32)
        f2 = np.concatenate(list(f2_dict.values()), axis=-1).astype(np.float32)

        # Regionprops features
        intra_rp, inter_rp, _, _ = compute_histograms(
            f1, f2, l1, l2, nbins=args.nbins
        )
        # Actually compute raw values for aggregation
        f1n = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-12)
        f2n = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-12)
        sim = f1n @ f2n.T
        for i, lbl_i in enumerate(l1):
            matches = np.where(l2 == lbl_i)[0]
            for j in range(len(l2)):
                val = float(sim[i, j])
                if j in matches:
                    rp_intra_all.append(val)
                else:
                    rp_inter_all.append(val)

        # DINO features (optional)
        if not args.no_dino:
            centroids = coords_src[:, 1:]  # (N, 2)
            patches = extract_patches(img, centroids)
            if len(patches) > 0:
                if args.dinov3:
                    dino_feats = extract_dino_features(patches, use_v3=True)
                else:
                    dino_feats = extract_dino_features(patches)
                if dino_feats is not None:
                    dino_feats = dino_feats.astype(np.float32)
                    dino_feats_n = dino_feats / (np.linalg.norm(dino_feats, axis=1, keepdims=True) + 1e-12)
                    sim_dino = dino_feats_n @ dino_feats_n.T
                    for i, lbl_i in enumerate(l1):
                        matches = np.where(l2 == lbl_i)[0]
                        for j in range(len(l2)):
                            val = float(sim_dino[i, j])
                            if j in matches:
                                dino_intra_all.append(val)
                            else:
                                dino_inter_all.append(val)

        n_frames += 1
        if n_frames % 10 == 0:
            logger.info(f"  Processed {n_frames} frames...")

    logger.info(f"Collected data from {n_frames} frames")

    # Build histogram data
    feature_results = {}
    csv_rows = []

    # Regionprops (7D)
    rp_intra_arr = np.array(rp_intra_all) if rp_intra_all else np.array([])
    rp_inter_arr = np.array(rp_inter_all) if rp_inter_all else np.array([])
    bins = np.linspace(-0.2, 1.0, args.nbins + 1)
    rp_intra_hist, _ = np.histogram(rp_intra_arr, bins=bins)
    rp_inter_hist, _ = np.histogram(rp_inter_arr, bins=bins)
    rp_stats = {
        "intra_mean": float(rp_intra_arr.mean()) if len(rp_intra_arr) > 0 else np.nan,
        "intra_std": float(rp_intra_arr.std()) if len(rp_intra_arr) > 0 else np.nan,
        "inter_mean": float(rp_inter_arr.mean()) if len(rp_inter_arr) > 0 else np.nan,
        "inter_std": float(rp_inter_arr.std()) if len(rp_inter_arr) > 0 else np.nan,
        "sep_gap": (float(rp_intra_arr.mean()) - float(rp_inter_arr.mean()))
        if len(rp_intra_arr) > 0 else np.nan,
        "n_intra": len(rp_intra_arr),
        "n_inter": len(rp_inter_arr),
    }
    dino_version = "DINOv3" if args.dinov3 else "DINOv2"
    feature_results[f"regionprops (7D)"] = (rp_intra_hist, rp_inter_hist, bins, rp_stats)
    csv_rows.append({"feature_set": "regionprops_7D", **rp_stats})

    # DINO features
    if dino_intra_all:
        dino_intra_arr = np.array(dino_intra_all)
        dino_inter_arr = np.array(dino_inter_all)
        dino_intra_hist, _ = np.histogram(dino_intra_arr, bins=bins)
        dino_inter_hist, _ = np.histogram(dino_inter_arr, bins=bins)
        dino_stats = {
            "intra_mean": float(dino_intra_arr.mean()),
            "intra_std": float(dino_intra_arr.std()),
            "inter_mean": float(dino_inter_arr.mean()),
            "inter_std": float(dino_inter_arr.std()),
            "sep_gap": float(dino_intra_arr.mean()) - float(dino_inter_arr.mean()),
            "n_intra": len(dino_intra_arr),
            "n_inter": len(dino_inter_arr),
        }
        feature_results[f"{dino_version} (384D)"] = (dino_intra_hist, dino_inter_hist, bins, dino_stats)
        csv_rows.append({"feature_set": dino_version, **dino_stats})

    # Save CSV
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "cosine_similarity_results.csv"
    with open(csv_path, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Saved results to {csv_path}")

    # Generate figure
    png_path = outdir / "cosine_similarity_histogram.png"
    generate_histogram_figure(feature_results, str(png_path))

    # Console summary
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY HISTOGRAM — SUMMARY")
    print("=" * 60)
    for fname, (_, _, _, stats) in feature_results.items():
        print(f"\n{fname}:")
        print(f"  intra-cell: μ={stats['intra_mean']:.4f} σ={stats['intra_std']:.4f} "
              f"(n={stats.get('n_intra', '?')})")
        print(f"  inter-cell: μ={stats['inter_mean']:.4f} σ={stats['inter_std']:.4f} "
              f"(n={stats.get('n_inter', '?')})")
        print(f"  separation gap:  {stats['sep_gap']:.4f}")
        if stats['sep_gap'] < 0.05:
            print(f"  ⚠ COLLAPSE — features cannot distinguish cells")
        elif stats['sep_gap'] < 0.15:
            print(f"  ~ BORDERLINE — weak but present signal")
        else:
            print(f"  ✓ PASS — sufficient signal for contrastive learning")


if __name__ == "__main__":
    main()
