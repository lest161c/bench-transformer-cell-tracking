"""DINO backbone comparison: feature quality on vanvliet bacteria.

Measures per-backbone:
  - Intra-cell cosine similarity (two jittered views of same cell)
  - Inter-cell cosine similarity (different cells, same frame)
  - Gap = intra - inter  (higher = better separation)
  - Recall@1 under mild jitter
  - Effective rank (PCA 95% variance)
  - Throughput (patches/sec)
  - GPU memory (VRAM MiB)
  - Model parameters (M)

Outputs: CSV, figure, verdict.txt

Usage:
    python compare_dino_backbones.py --data-root ../data/vanvliet --outdir runs/dino_comparison
"""

import argparse, logging, time, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.measure import regionprops_table
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("compare_dino")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

PATCH_SIZE = 64
DINO_INPUT_SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ─── data loading ───────────────────────────────────────────────────────────────

def load_frame(mask_path, img_path):
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) == 0:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    labels = np.array(props["label"], dtype=np.int32)
    return coords, labels, img


def scan_frames(data_root, conditions, max_frames=200):
    frames = []
    dr = Path(data_root)
    for cond in conditions:
        for exp in sorted((dr / cond).iterdir()):
            if not exp.is_dir(): continue
            tra, img_dir = exp / "TRA", exp / "img"
            if not tra.exists() or not img_dir.exists(): continue
            for m in sorted(tra.glob("man_track*.tif")):
                stem = m.stem.replace("man_track", "")
                try: fi = int(stem)
                except ValueError: continue
                ip = img_dir / f"t{fi:06d}.tif"
                if ip.exists():
                    frames.append((str(m), str(ip)))
                    if len(frames) >= max_frames: return frames
    return frames


def extract_patches(img, centroids):
    h, w = img.shape[-2:]
    half = PATCH_SIZE // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, h - 1)
        cx_i = np.clip(cx_i, 0, w - 1)
        y1, y2 = cy_i - half, cy_i + half
        x1, x2 = cx_i - half, cx_i + half
        pt = max(0, -y1); pb = max(0, y2 - h)
        pl = max(0, -x1); pr = max(0, x2 - w)
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(crop, ((0, max(0, PATCH_SIZE - crop.shape[0])),
                                  (0, max(0, PATCH_SIZE - crop.shape[1]))),
                          mode="reflect")[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    return np.stack(patches) if patches else np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


# ─── DINO backbone loading ──────────────────────────────────────────────────────

def load_dino(repo, model_name):
    """Load a DINO model, return (model, dim, n_params_M)."""
    m = torch.hub.load(repo, model_name).to(device).eval()
    dim = m.embed_dim if hasattr(m, 'embed_dim') else (m.dim if hasattr(m, 'dim') else 384)
    n_params = sum(p.numel() for p in m.parameters())
    for p in m.parameters():
        p.requires_grad = False
    return m, dim, n_params / 1e6


@torch.no_grad()
def compute_dino_embs(patches_np, dino_model, batch_size=None):
    """patches_np: (N, H, W) float32 [0,1] → (N, dim) numpy.
    Batched to fit in 4GB VRAM (A500). Auto-sizes batch if not given."""
    if len(patches_np) == 0:
        return np.zeros((0, dino_model.embed_dim), dtype=np.float32)
    if batch_size is None:
        # heuristic: aim for ~200MB per batch (patch: 3×224×224×4B ≈ 0.6MB)
        batch_size = max(16, min(64, 200 // 1))  # ~64 for 4GB A500

    # per-patch min-max normalization
    pmin = patches_np.min(axis=(1, 2), keepdims=True)
    pmax = patches_np.max(axis=(1, 2), keepdims=True)
    pn = (patches_np - pmin) / (pmax - pmin + 1e-8)

    all_embs = []
    for start in range(0, len(pn), batch_size):
        batch = pn[start:start + batch_size]
        t = torch.from_numpy(batch).float().unsqueeze(1).to(device)
        t = F.interpolate(t, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE), mode="bilinear", align_corners=False)
        t = t.expand(-1, 3, -1, -1)
        mean = IMAGENET_MEAN.to(device)
        std = IMAGENET_STD.to(device)
        t = (t - mean) / std
        all_embs.append(dino_model(t).cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ─── metrics ────────────────────────────────────────────────────────────────────

def compute_gap(embs_a, embs_b, labels_a, labels_b):
    fa = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-12)
    fb = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-12)
    sim = fa @ fb.T
    intra, inter = [], []
    for i, lbl in enumerate(labels_a):
        for j in range(len(labels_b)):
            if labels_b[j] == lbl:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])
    return {
        "intra": float(np.mean(intra)) if intra else np.nan,
        "inter": float(np.mean(inter)) if inter else np.nan,
        "gap": float(np.mean(intra) - np.mean(inter)) if intra and inter else np.nan,
    }


def compute_recall(embs_a, embs_b, labels_a, labels_b, topk=1):
    fa = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-12)
    fb = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-12)
    sim = fa @ fb.T
    hits, total = 0, 0
    for i, lbl in enumerate(labels_a):
        matches = np.where(labels_b == lbl)[0]
        if len(matches) == 0: continue
        true_j = matches[0]
        if true_j in np.argsort(-sim[i])[:topk]:
            hits += 1
        total += 1
    return hits / total if total else np.nan


def compute_effective_rank(embs, threshold=0.95):
    """Effective rank: number of PCA components to explain threshold variance."""
    embs_c = embs - embs.mean(axis=0)
    cov = embs_c.T @ embs_c / (len(embs_c) - 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    evals = np.maximum(evals, 0)
    total = evals.sum()
    if total == 0: return 0
    cumulative = np.cumsum(evals) / total
    rank = np.searchsorted(cumulative, threshold) + 1
    return min(rank, len(evals))


# ─── backbone definitions ───────────────────────────────────────────────────────

BACKBONE_SPECS = [
    {
        "name": "DINOv2-vits14",
        "repo": "facebookresearch/dinov2",
        "model": "dinov2_vits14",
        "short": "v2-S",
        "expected_dim": 384,
    },
    {
        "name": "DINOv2-vitb14",
        "repo": "facebookresearch/dinov2",
        "model": "dinov2_vitb14",
        "short": "v2-B",
        "expected_dim": 768,
    },
]


# ─── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare DINO backbones on vanvliet bacteria")
    parser.add_argument("--data-root", default="../data/vanvliet", help="Path to vanvliet data")
    parser.add_argument("--outdir", default="runs/dino_comparison", help="Output directory")
    parser.add_argument("--conditions", nargs="+", default=["rpsM", "recA", "pheA", "metA", "cib", "trpL"])
    parser.add_argument("--max-frames", type=int, default=40, help="Max frames to scan")
    parser.add_argument("--jitter-std", type=float, default=4.0, help="Jitter std for intra/inter gap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── scan frames ──
    logger.info(f"Scanning frames from {args.data_root} ({args.conditions})...")
    frames = scan_frames(args.data_root, args.conditions, args.max_frames)
    logger.info(f"  Found {len(frames)} frames")

    # ── load all frames into memory ──
    logger.info("Loading frames into memory...")
    all_data = []
    for mp, ip in frames:
        r = load_frame(mp, ip)
        if r is None: continue
        coords, labels, img = r
        if len(labels) < 2: continue
        # pre-extract patches (same for all backbones)
        patches = extract_patches(img, coords)
        # create jittered view
        jit_coords = coords + np.random.randn(*coords.shape).astype(np.float32) * args.jitter_std
        jit_patches = extract_patches(img, jit_coords)
        all_data.append((patches, jit_patches, labels, coords.shape[0]))
    logger.info(f"  Loaded {len(all_data)} valid frames, "
                f"{sum(d[3] for d in all_data)} total cells")

    # ── evaluate each backbone ──
    results = []
    for spec in BACKBONE_SPECS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {spec['name']}...")

        # load
        t0 = time.time()
        dino_model, dim, n_params = load_dino(spec["repo"], spec["model"])
        load_time = time.time() - t0
        logger.info(f"  dim={dim}, params={n_params:.1f}M, load_time={load_time:.1f}s")

        # throughput benchmark (on subset to save time)
        throughput_n = min(400, sum(d[3] for d in all_data[:10]))
        throughput_patches = np.concatenate([d[0] for d in all_data[:10]], axis=0)[:throughput_n]
        logger.info(f"  Throughput benchmark: {len(throughput_patches)} patches")
        # warmup
        _ = compute_dino_embs(throughput_patches[:32], dino_model)
        torch.cuda.synchronize()
        t0 = time.time()
        _ = compute_dino_embs(throughput_patches, dino_model)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        pps = len(throughput_patches) / elapsed
        logger.info(f"  Throughput: {pps:.1f} patches/sec ({elapsed:.1f}s)")

        # memory
        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            torch.cuda.reset_peak_memory_stats()
        else:
            mem = 0.0
        logger.info(f"  Peak VRAM: {mem:.0f} MiB")

        # compute gap per frame (original vs jittered)
        gaps, intra_sims, inter_sims, recalls = [], [], [], []
        for patches, jit_patches, labels, _ in all_data:
            emb_orig = compute_dino_embs(patches, dino_model)
            emb_jit = compute_dino_embs(jit_patches, dino_model)
            g = compute_gap(emb_orig, emb_jit, labels, labels)
            gaps.append(g["gap"])
            intra_sims.append(g["intra"])
            inter_sims.append(g["inter"])
            r = compute_recall(emb_orig, emb_jit, labels, labels)
            recalls.append(r)

        mean_gap = np.nanmean(gaps)
        mean_intra = np.nanmean(intra_sims)
        mean_inter = np.nanmean(inter_sims)
        mean_recall = np.nanmean(recalls)

        # effective rank (on ALL embeddings pooled)
        all_embs = np.concatenate([
            compute_dino_embs(d[0], dino_model) for d in all_data
        ], axis=0)
        eff_rank = compute_effective_rank(all_embs)

        row = {
            "backbone": spec["name"],
            "short": spec["short"],
            "dim": dim,
            "params_M": round(n_params, 1),
            "intra_cos_sim": round(mean_intra, 4),
            "inter_cos_sim": round(mean_inter, 4),
            "gap": round(mean_gap, 4),
            "recall_at_1": round(mean_recall, 4),
            "effective_rank": eff_rank,
            "throughput_pps": round(pps, 1),
            "peak_vram_mib": round(mem, 0),
            "load_time_s": round(load_time, 1),
            "n_frames": len(all_data),
            "n_cells": sum(d[3] for d in all_data),
        }
        results.append(row)
        logger.info(f"  Gap: {mean_gap:.4f}, Intra: {mean_intra:.4f}, "
                    f"Inter: {mean_inter:.4f}, Recall@1: {mean_recall:.4f}")
        logger.info(f"  Effective rank: {eff_rank}")

        # free model
        del dino_model
        torch.cuda.empty_cache()

    # ── save results ──
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = outdir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved {csv_path}")

    # ── figure ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("DINO Backbone Comparison on vanvliet Bacteria", fontsize=13, fontweight="bold")

    names = [r["short"] for r in results]

    # Gap
    ax = axes[0, 0]
    gaps = [r["gap"] for r in results]
    bars = ax.bar(names, gaps, color=["#2196F3", "#FF9800"])
    ax.set_title("Gap (intra - inter cos sim)")
    ax.set_ylabel("Cosine similarity gap")
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(gaps) * 1.3)

    # Intra/Inter
    ax = axes[0, 1]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, [r["intra_cos_sim"] for r in results], w, label="Intra", color="#4CAF50")
    ax.bar(x + w/2, [r["inter_cos_sim"] for r in results], w, label="Inter", color="#F44336")
    ax.set_title("Cosine Similarity")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend()

    # Recall@1
    ax = axes[0, 2]
    recalls = [r["recall_at_1"] for r in results]
    bars = ax.bar(names, recalls, color=["#2196F3", "#FF9800"])
    ax.set_title(f"Recall@1 (jitter={args.jitter_std}px)")
    ax.set_ylabel("Recall")
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)

    # Effective rank
    ax = axes[1, 0]
    ranks = [r["effective_rank"] for r in results]
    bars = ax.bar(names, ranks, color=["#2196F3", "#FF9800"])
    ax.set_title("Effective Rank (95% var)")
    ax.set_ylabel("Components")
    for bar, val in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", fontsize=9)

    # Throughput
    ax = axes[1, 1]
    pps = [r["throughput_pps"] for r in results]
    bars = ax.bar(names, pps, color=["#2196F3", "#FF9800"])
    ax.set_title("Throughput (patches/sec)")
    ax.set_ylabel("Patches/sec")
    for bar, val in zip(bars, pps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", fontsize=9)

    # Model size
    ax = axes[1, 2]
    params = [r["params_M"] for r in results]
    bars = ax.bar(names, params, color=["#2196F3", "#FF9800"])
    ax.set_title("Model Parameters")
    ax.set_ylabel("Millions")
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}M", ha="center", fontsize=9)

    plt.tight_layout()
    fig_path = outdir / "comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {fig_path}")

    # ── verdict ──
    lines = []
    lines.append("DINO Backbone Comparison Verdict")
    lines.append("=" * 50)
    lines.append(f"Data: {len(all_data)} frames, {sum(d[3] for d in all_data)} cells from {args.conditions}")
    lines.append(f"Jitter std: {args.jitter_std}px")
    lines.append(f"Device: {device}")
    lines.append("")
    lines.append("AVAILABLE vs UNAVAILABLE BACKBONES")
    lines.append("-" * 40)
    lines.append("  DINOv2 vits14 (384D)    ✓ cached, works")
    lines.append("  DINOv3 vits16 (384D)    ✗ 403 Forbidden (dl.fbaipublicfiles.com)")
    lines.append("  Cell-DINO CP vits8      ✗ pretrained_url=None (not released)")
    lines.append("  Cell-DINO HPA vitl14    ✗ pretrained_url=None (not released)")
    lines.append("  Cell-DINO HPA vitl16    ✗ pretrained_url=None (not released)")
    lines.append("  DINOv2 vitb14 (768D)    ✓ downloaded, works")
    lines.append("  DINOv2 vitl14 (1024D)   ~ partial download (1.13GB, timed out)")
    lines.append("  DINOv2 vitg14 (1536D)   ~ not tested (would need 2+ GB)")
    lines.append("")
    lines.append("RESULTS")
    lines.append("-" * 40)
    for r in results:
        lines.append(f"  {r['short']}: gap={r['gap']:.4f}, recall={r['recall_at_1']:.4f}, "
                     f"rank={r['effective_rank']}, pps={r['throughput_pps']:.0f}, "
                     f"vram={r['peak_vram_mib']:.0f}MiB, params={r['params_M']:.1f}M")
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    if len(results) >= 2:
        d0, d1 = results[0], results[1]
        gap_ratio = d1["gap"] / d0["gap"] if d0["gap"] > 0 else float("inf")
        lines.append(f"  Gap ratio (B/S): {gap_ratio:.2f}x")
        lines.append(f"  Recall ratio (B/S): {d1['recall_at_1']/d0['recall_at_1']:.2f}x")
        lines.append(f"  VRAM ratio (B/S): {d1['peak_vram_mib']/d0['peak_vram_mib']:.2f}x")
        lines.append(f"  Speed ratio (B/S): {d1['throughput_pps']/d0['throughput_pps']:.2f}x")
        lines.append("")
        if gap_ratio > 1.15:
            lines.append(f"  Vit-B provides {gap_ratio:.1f}x better feature separation but at "
                         f"{d1['peak_vram_mib']/d0['peak_vram_mib']:.1f}x VRAM and "
                         f"{d1['throughput_pps']/d0['throughput_pps']:.1f}x speed.")
        else:
            lines.append(f"  Minimal improvement from Vit-B over Vit-S ({gap_ratio:.2f}x gap). "
                         f"Feature quality bottleneck is domain mismatch, not model capacity.")
    lines.append("")
    lines.append("CONCLUSION")
    lines.append("-" * 40)
    lines.append("  Cell-DINO and DINOv3 are inaccessible without manual weight download.")
    lines.append("  The DINOv2 variants (ViT-S/B/L/G) differ only in capacity, not domain.")
    lines.append("  If ViT-B vs ViT-S shows minimal improvement, the bottleneck is domain")
    lines.append("  mismatch (ImageNet vs bacteria), not model capacity. In that case,")
    lines.append("  a microscopy-finetuned backbone would be the only path forward —")
    lines.append("  but Cell-DINO weights need to be obtained separately from Meta.")

    verdict_path = outdir / "verdict.txt"
    verdict_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {verdict_path}")

    # print verdict
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
