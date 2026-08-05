"""DINOv2 vs DINOv3 comparison — fresh benchmark with HF-loaded models.

Compares embedding quality on the same synthetic cell patches:
  - Intra/inter cosine similarity gap
  - Effective dimensionality (PCA 95% variance)
  - Throughput (patches/sec)

Usage:
  python benchmark_dinov3_comparison_v2.py
"""

import csv, time, math, argparse, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")

PATCH_SIZE = 64
DINO_INPUT = 224
N_PATCHES = 500
N_CENTROIDS = 20  # simulate 20 cells per frame
JITTER_STD = 4    # augment patches with small jitter

DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
DINO_STD = torch.tensor([0.229, 0.224, 0.225])


def generate_synthetic_patches(n_frames=25, cells_per_frame=20, seed=42):
    """Generate synthetic cell patches with labels."""
    rng = np.random.default_rng(seed)
    patches, labels = [], []
    for frame in range(n_frames):
        base = rng.exponential(0.1, (512, 512)).astype(np.float32)
        for c in range(cells_per_frame):
            cy = rng.integers(32, 480)
            cx = rng.integers(32, 480)
            # Add cell blob
            y, x = np.ogrid[-PATCH_SIZE//2:PATCH_SIZE//2, -PATCH_SIZE//2:PATCH_SIZE//2]
            r2 = y**2 + x**2
            blob = np.exp(-r2 / (2*(3+rng.random()*5)**2))
            blob *= 0.3 + rng.random() * 0.7
            patch = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            y1, y2 = max(0, cy-PATCH_SIZE//2), min(512, cy+PATCH_SIZE//2)
            x1, x2 = max(0, cx-PATCH_SIZE//2), min(512, cx+PATCH_SIZE//2)
            py1, py2 = PATCH_SIZE//2 - (cy-y1), PATCH_SIZE//2 + (y2-cy)
            px1, px2 = PATCH_SIZE//2 - (cx-x1), PATCH_SIZE//2 + (x2-cx)
            patch[py1:py2, px1:px2] = base[y1:y2, x1:x2] + blob[py1:py2, px1:px2]
            patch = np.clip(patch, 0, 1)

            # Add slight jitter for "same cell" variants
            p1 = patch.copy()
            p2 = patch + rng.normal(0, 0.01, patch.shape).astype(np.float32)
            p2 = np.clip(p2, 0, 1)
            patches.extend([p1, p2])
            labels.extend([c, c])
    return np.array(patches, dtype=np.float32), np.array(labels)


def compute_metrics(feats, labels):
    """intra/inter cosine sim gap and effective rank."""
    f_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sim = f_norm @ f_norm.T
    intra, inter = [], []
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            if labels[i] == labels[j]:
                intra.append(float(sim[i, j]))
            else:
                inter.append(float(sim[i, j]))
    intra_mean = np.mean(intra)
    inter_mean = np.mean(inter)
    gap = intra_mean - inter_mean

    feats_c = feats - feats.mean(axis=0)
    U, S, _ = np.linalg.svd(feats_c, full_matrices=False)
    var = S**2 / (len(feats) - 1)
    cumvar = np.cumsum(var) / var.sum()
    eff_rank = int(np.searchsorted(cumvar, 0.95) + 1)

    return {"gap": round(gap, 5), "eff_rank": eff_rank,
            "intra_sim": round(float(intra_mean), 5),
            "inter_sim": round(float(inter_mean), 5),
            "n_intra": len(intra), "n_inter": len(inter)}


def extract_features(model, patches, device, batch_size=32):
    """Extract DINO embeddings from patches."""
    model = model.to(device).eval()
    mean = DINO_MEAN.view(1, 3, 1, 1).to(device)
    std = DINO_STD.view(1, 3, 1, 1).to(device)

    feats = []
    t0 = time.perf_counter()
    for start in range(0, len(patches), batch_size):
        batch = torch.from_numpy(patches[start:start+batch_size]).float().to(device)
        batch = batch.unsqueeze(1).expand(-1, 3, -1, -1)
        batch = F.interpolate(batch, size=(DINO_INPUT, DINO_INPUT),
                              mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        with torch.no_grad():
            out = model(batch)
        if hasattr(out, "last_hidden_state"):
            f = out.last_hidden_state[:, 0, :].cpu().numpy()
        elif isinstance(out, dict):
            k = next(k for k in out if isinstance(out[k], torch.Tensor) and out[k].dim() >= 2)
            f = out[k][:, 0].cpu().numpy()
        else:
            f = out.cpu().numpy()
        feats.append(f)
    elapsed = time.perf_counter() - t0
    all_feats = np.concatenate(feats, axis=0)
    return all_feats.astype(np.float32), len(patches) / max(elapsed, 0.001)


def load_model_hf(model_id, device="cpu"):
    """Load DINO model from HuggingFace cache (pre-downloaded via snapshot_download)."""
    from transformers import AutoModel
    import os
    cache = "/tmp/hf_cache/models--facebook--dinov3-vits16-pretrain-lvd1689m/snapshots"
    if os.path.isdir(cache):
        snaps = sorted(os.listdir(cache))
        if snaps:
            local = os.path.join(cache, snaps[-1])
            model = AutoModel.from_pretrained(local, trust_remote_code=True)
            model.to(device).eval()
            return model, 384
    # fallback: try HF hub
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device).eval()
    return model, 384


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-patches", type=int, default=N_PATCHES)
    p.add_argument("--outdir", default="benchmark_attn")
    args = p.parse_args()

    print("=" * 60)
    print("DINOv2 vs DINOv3 — Feature Quality Comparison")
    print(f"Device: {args.device}, patches: {args.n_patches}")
    print("=" * 60)

    patches, labels = generate_synthetic_patches(n_frames=25,
                                                   cells_per_frame=args.n_patches // 20)
    print(f"  Generated {len(patches)} patches from 25 frames")
    print()

    results = []
    for version, model_id, repo in [
        ("v2", "facebook/dinov2-small", "facebookresearch/dinov2"),
        ("v3", "facebook/dinov3-vits16-pretrain-lvd1689m", "facebookresearch/dinov3"),
    ]:
        print(f"  Loading {version}...")
        if version == "v2":
            model = torch.hub.load(repo, "dinov2_vits14")
            model.eval()
            dim = 384
        else:
            try:
                model, dim = load_model_hf(model_id, args.device)
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({"version": version, "status": f"FAILED: {str(e)[:100]}"})
                continue

        print(f"    Extracting features ({len(patches)} patches)...")
        feats, throughput = extract_features(model, patches, args.device)
        metrics = compute_metrics(feats, labels)
        metrics["version"] = version
        metrics["status"] = "OK"
        metrics["dim"] = dim
        metrics["throughput_pps"] = round(throughput, 1)
        metrics["n_patches"] = len(patches)
        results.append(metrics)
        print(f"    gap={metrics['gap']:.4f}  eff_rank={metrics['eff_rank']}  "
              f"intra={metrics['intra_sim']:.4f}  inter={metrics['inter_sim']:.4f}  "
              f"{throughput:.0f} patches/s")
        print()

    # Save
    outdir = Path(args.outdir)
    path = outdir / "dinov3_comparison_v2.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {path}")

    # Compare
    if len(results) == 2 and results[0]["status"] == "OK" and results[1]["status"] == "OK":
        gap_ratio = results[1]["gap"] / max(results[0]["gap"], 0.0001)
        rank_ratio = results[1]["eff_rank"] / max(results[0]["eff_rank"], 1)
        print(f"\nComparison:")
        print(f"  v2 gap={results[0]['gap']:.4f}  v3 gap={results[1]['gap']:.4f}  "
              f"ratio={gap_ratio:.1f}×")
        print(f"  v2 rank={results[0]['eff_rank']}  v3 rank={results[1]['eff_rank']}  "
              f"ratio={rank_ratio:.1f}×")
        print(f"  v2 throughput={results[0]['throughput_pps']:.0f} pps  "
              f"v3 throughput={results[1]['throughput_pps']:.0f} pps")


if __name__ == "__main__":
    main()
