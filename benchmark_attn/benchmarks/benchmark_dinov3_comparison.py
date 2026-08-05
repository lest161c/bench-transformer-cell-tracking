"""DINOv2 vs DINOv3 comparison — quantify if v3 is better.

Extracts features from identical cell patches using both models,
computes separation metrics: intra/inter cosine sim gap, effective
dimensionality, and extraction throughput.

Usage:
  python benchmark_dinov3_comparison.py
"""

import csv, time, argparse, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")


def load_model(version="v2"):
    """Load DINO model via torch hub. Falls back gracefully."""
    import torch
    specs = {
        "v2": ("facebookresearch/dinov2", "dinov2_vits14", 384),
        "v3": ("facebookresearch/dinov3", "dinov3_vits16", 384),
    }
    repo, model_name, dim = specs[version]
    try:
        model = torch.hub.load(repo, model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, dim
    except Exception as e:
        print(f"  Failed to load {version}: {e}")
        return None, None


def extract_features(model, patches, device="cpu"):
    """Extract DINO embeddings from cell patches."""
    import torch
    import torch.nn.functional as F

    DINO_INPUT = 224
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    model = model.to(device)
    feats = []
    batch_size = 16

    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[start:start+batch_size]).float().to(device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(1)
        batch = F.interpolate(batch, size=(DINO_INPUT, DINO_INPUT),
                              mode="bilinear", align_corners=False)
        batch = batch.expand(-1, 3, -1, -1)
        batch = (batch - mean) / std
        with torch.no_grad():
            f = model(batch).cpu().numpy()
        feats.append(f)

    return np.concatenate(feats, axis=0).astype(np.float32)


def compute_metrics(feats, labels):
    """Compute separation metrics for feature embeddings."""
    N = len(feats)
    if N < 3:
        return {"gap": np.nan, "eff_rank": np.nan, "intra_sim": np.nan,
                "inter_sim": np.nan}

    # Normalize
    f_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sim = f_norm @ f_norm.T

    intra, inter = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if labels[i] == labels[j]:
                intra.append(float(sim[i, j]))
            else:
                inter.append(float(sim[i, j]))

    intra_mean = np.mean(intra) if intra else np.nan
    inter_mean = np.mean(inter) if inter else np.nan
    gap = intra_mean - inter_mean if not (np.isnan(intra_mean) or np.isnan(inter_mean)) else np.nan

    # Effective rank (PCA, 95% variance)
    feats_c = feats - feats.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(N, feats.shape[1])).fit(feats_c)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_rank = int(np.searchsorted(cumvar, 0.95) + 1)
    except Exception:
        # Manual PCA
        U, S, Vt = np.linalg.svd(feats_c, full_matrices=False)
        var = S**2 / (N - 1)
        cumvar = np.cumsum(var) / var.sum()
        eff_rank = int(np.searchsorted(cumvar, 0.95) + 1)

    return {"gap": round(gap, 5), "eff_rank": eff_rank,
            "intra_sim": round(intra_mean, 5), "inter_sim": round(inter_mean, 5),
            "n_intra": len(intra), "n_inter": len(inter)}


def extract_patches(img, centroids, patch_size=64):
    """Extract square patches around centroids."""
    h, w = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i = int(round(float(cy)))
        cx_i = int(round(float(cx)))
        cy_i, cx_i = np.clip(cy_i, 0, h-1), np.clip(cx_i, 0, w-1)
        y1, y2 = cy_i - half, cy_i + half
        x1, x2 = cx_i - half, cx_i + half
        pt, pb = max(0, -y1), max(0, y2 - h)
        pl, pr = max(0, -x1), max(0, x2 - w)
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(crop, ((0, max(0, patch_size-crop.shape[0])),
                                 (0, max(0, patch_size-crop.shape[1]))),
                          mode="reflect")[:patch_size, :patch_size]
        patches.append(crop)
    return np.stack(patches).astype(np.float32) if patches else np.zeros((0, patch_size, patch_size), dtype=np.float32)


def load_real_data(max_frames=20):
    """Load vanvliet frames with segmentations."""
    try:
        from benchmark_ssl.ssl_pipeline import load_experiment_frames
        from tifffile import imread
        frames = load_experiment_frames(
            "/data/cat/ws/mawe985g-data/data/celltracking/vanvliet/",
            conditions=["rpsM"]
        )
    except Exception:
        print("  No real vanvliet data available. Using synthetic data.")
        return generate_synthetic_data(200, 10)

    all_patches = []
    all_labels = []
    n_frames = 0

    for idx in range(min(len(frames), max_frames)):
        _, _, _, mask_path, img_path = frames[idx]
        mask = imread(mask_path)
        img = imread(img_path).astype(np.float32)
        p1, p998 = np.percentile(img, (1, 99.8))
        img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

        from skimage.measure import regionprops_table
        props = regionprops_table(
            mask.astype(int),
            intensity_image=img,
            properties=("label", "centroid")
        )
        if len(props["label"]) < 2:
            continue

        centroids = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1)
        labels = props["label"]

        patches = extract_patches(img, centroids)
        if len(patches) > 0:
            all_patches.append(patches)
            all_labels.append(labels)
            n_frames += 1

    if all_patches:
        return np.concatenate(all_patches, axis=0), np.concatenate(all_labels, axis=0), n_frames
    return generate_synthetic_data(200, 10)


def generate_synthetic_data(N=200, n_groups=10):
    """Synthetic cell-like patches for fallback testing."""
    rng = np.random.RandomState(42)
    patches = rng.uniform(0, 1, (N, 64, 64)).astype(np.float32)
    labels = rng.randint(0, n_groups, (N,))
    return patches, labels, 0


def run_comparison():
    print("=" * 60)
    print("DINOv2 vs DINOv3 — Feature Quality Comparison")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    patches, labels, n_real_frames = load_real_data(max_frames=20)
    print(f"  {len(patches)} patches from {n_real_frames} real frames"
          if n_real_frames else f"  {len(patches)} synthetic patches")

    rows = []

    for version in ["v2", "v3"]:
        print(f"\nLoading DINO{version}...")
        model, dim = load_model(version)
        if model is None:
            rows.append({"version": version, "status": "FAILED", "dim": 0,
                         "gap": np.nan, "eff_rank": np.nan})
            continue

        print(f"  Extracting features ({dim}D)...")
        t0 = time.perf_counter()
        feats = extract_features(model, patches, device="cpu")
        t_feat = time.perf_counter() - t0

        print(f"  Computing metrics...")
        metrics = compute_metrics(feats, labels)

        rows.append({
            "version": version,
            "status": "OK",
            "dim": dim,
            "gap": metrics["gap"],
            "eff_rank": metrics["eff_rank"],
            "intra_sim": metrics["intra_sim"],
            "inter_sim": metrics["inter_sim"],
            "n_intra_pairs": metrics["n_intra"],
            "n_inter_pairs": metrics["n_inter"],
            "extraction_time_s": round(t_feat, 2),
            "patches_per_second": round(len(patches) / max(t_feat, 0.01)),
            "n_patches": len(patches),
        })

        print(f"  Time: {t_feat:.1f}s ({len(patches)/max(t_feat,0.01):.0f} patches/s)")
        print(f"  Gap: {metrics['gap']:.4f}  eff_rank: {metrics['eff_rank']}")
        print(f"  Intra: {metrics['intra_sim']:.4f}  Inter: {metrics['inter_sim']:.4f}")

    return rows, patches, labels


def generate_figure(rows, patches, labels, outdir="benchmark_attn"):
    outdir = Path(outdir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart of separation metrics
    ax = axes[0]
    versions = [r["version"] for r in rows if r["status"] == "OK"]
    gaps = [r["gap"] for r in rows if r["status"] == "OK"]
    ranks = [r["eff_rank"] for r in rows if r["status"] == "OK"]
    intra = [r["intra_sim"] for r in rows if r["status"] == "OK"]
    inter = [r["inter_sim"] for r in rows if r["status"] == "OK"]

    x = np.arange(len(versions))
    w = 0.2
    colors_v = {"v2": "#3498db", "v3": "#e879f9"}

    for i, v in enumerate(versions):
        c = colors_v.get(v, "#333")
        ax.bar(i - w*1.5, intra[i], w, color=c, alpha=0.6, label=f"{v} intra")
        ax.bar(i - w/2, inter[i], w, color=c, alpha=0.3, label=f"{v} inter")
        ax.bar(i + w, gaps[i], w, color=c, alpha=1.0, label=f"{v} gap")

    ax.set_xticks(x)
    ax.set_xticklabels([f"DINO{v}" for v in versions])
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Separation Metrics: DINOv2 vs DINOv3")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotation with eff_rank
    for i, (v, g, r) in enumerate(zip(versions, gaps, ranks)):
        ax.annotate(f"eff_rank={r}", (i + w, g + 0.02),
                    ha="center", fontsize=10, fontweight="bold")

    # Bar chart of throughput
    ax = axes[1]
    tps = [r["patches_per_second"] for r in rows if r["status"] == "OK"]
    bars = ax.bar(versions, tps, color=[colors_v.get(v, "#333") for v in versions],
                  alpha=0.85, edgecolor="white")
    ax.set_ylabel("Patches/second (CPU)")
    ax.set_title("Feature Extraction Throughput")
    for bar, t in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{t}", ha="center", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("DINOv2 vs DINOv3 — Quantitative Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = str(outdir / "dinov3_comparison.png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {png}")


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader(); w.writerows(rows)
    print(f"Saved CSV: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="benchmark_attn")
    args = p.parse_args()

    rows, patches, labels = run_comparison()
    save_csv(rows, f"{args.outdir}/dinov3_comparison.csv")

    ok_rows = [r for r in rows if r["status"] == "OK"]
    if len(ok_rows) >= 2:
        generate_figure(rows, patches, labels, args.outdir)

        v2 = ok_rows[0]
        v3 = ok_rows[1]
        print(f"\n{'='*60}")
        print(f"VERDICT")
        print(f"{'='*60}")
        print(f"         gap      eff_rank   intra_sim  inter_sim  throughput")
        print(f"DINOv2:  {v2['gap']:.4f}    {v2['eff_rank']:>3d}       "
              f"{v2['intra_sim']:.4f}    {v2['inter_sim']:.4f}    {v2['patches_per_second']}")
        print(f"DINOv3:  {v3['gap']:.4f}    {v3['eff_rank']:>3d}       "
              f"{v3['intra_sim']:.4f}    {v3['inter_sim']:.4f}    {v3['patches_per_second']}")

        gap_ratio = v3["gap"] / max(v2["gap"], 0.001)
        rank_ratio = v3["eff_rank"] / max(v2["eff_rank"], 1)
        speed_ratio = v3["patches_per_second"] / max(v2["patches_per_second"], 1)

        if gap_ratio > 1.05 and rank_ratio >= 0.95:
            print(f"\n→ DINOv3 IS better: gap {gap_ratio:.1f}×, rank {rank_ratio:.1f}×, speed {speed_ratio:.1f}×")
        elif gap_ratio >= 0.95 and rank_ratio >= 0.95:
            print(f"\n→ DINOv3 is EQUIVALENT to v2 (within 5%)")
        else:
            print(f"\n→ DINOv2 is BETTER for SSL separation (gap {1/gap_ratio:.1f}× larger)")
    else:
        print(f"\nERROR: One or both models failed to load.")
        print(f"v2: {rows[0] if len(rows)>0 else 'not attempted'}")
        print(f"v3: {rows[1] if len(rows)>1 else 'not attempted'}")
        print(f"\nDINOv3 requires torch hub access. Verify:")
        print(f"  python3 -c \"import torch; torch.hub.load('facebookresearch/dinov3','dinov3_vits16')\"")


if __name__ == "__main__":
    main()
