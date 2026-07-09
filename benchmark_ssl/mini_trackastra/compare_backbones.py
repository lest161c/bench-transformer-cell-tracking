#!/usr/bin/env python3
"""Quick comparison: DINOv2 vs OpenPhenom feature gap on bacteria.

Measures intra-cell vs inter-cell cosine similarity to assess which
backbone produces more discriminative features for bacteria tracking.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import regionprops_table
from skimage.transform import resize
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_backbones")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PATCH_SIZE = 64        # For DINOv2
PHENOM_CROP = 256      # For OpenPhenom


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return coords, props["label"], img

def scan_frames(data_root, max_frames=40):
    dr = Path(data_root)
    frames = []
    for cond_dir in sorted(dr.glob("rpsM/*")) if (dr / "rpsM").exists() else sorted(dr.glob("*/*")):
        if not cond_dir.is_dir():
            continue
        tra = cond_dir / "TRA"
        img_dir = cond_dir / "img"
        if not tra.exists() or not img_dir.exists():
            continue
        for m in sorted(tra.glob("man_track*.tif")):
            stem = m.stem.replace("man_track", "")
            try:
                fi = int(stem)
            except ValueError:
                continue
            ip = img_dir / f"t{fi:06d}.tif"
            if ip.exists():
                frames.append((str(m), str(ip)))
            if len(frames) >= max_frames:
                return frames
    return frames

def extract_patch(img, cy, cx, size):
    """Extract square patch around centroid, with reflection padding."""
    h, w = img.shape[-2:]
    half = size // 2
    cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
    cy_i, cx_i = np.clip(cy_i, 0, h-1), np.clip(cx_i, 0, w-1)
    y1, x1 = cy_i - half, cx_i - half
    y2, x2 = cy_i + half, cx_i + half
    pt, pb = max(0, -y1), max(0, y2 - h)
    pl, pr = max(0, -x1), max(0, x2 - w)
    y1c, x1c = max(0, y1), max(0, x1)
    y2c, x2c = min(h, y2), min(w, x2)
    crop = img[y1c:y2c, x1c:x2c] if y2c > y1c and x2c > x1c else np.zeros((1,1), dtype=np.float32)
    if pt or pb or pl or pr:
        crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
    if crop.shape[0] < size or crop.shape[1] < size:
        crop = np.pad(crop, tuple((0, max(0, size - s)) for s in crop.shape), mode="reflect")
    return crop[:size, :size]


# ═══════════════════════════════════════════════════════════════════════════════
# DINOv2 backbone
# ═══════════════════════════════════════════════════════════════════════════════

_dino = None

def get_dino():
    global _dino
    if _dino is None:
        logger.info("Loading DINOv2 vits14...")
        _dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    elif next(_dino.parameters()).device != device:
        _dino = _dino.to(device)
    return _dino

@torch.no_grad()
def extract_dino(img, coords):
    """Extract DINOv2 embeddings for cells at given coordinates."""
    get_dino()
    patches = []
    for cy, cx in coords:
        p = extract_patch(img, cy, cx, PATCH_SIZE)
        patches.append(p)
    if not patches:
        return np.zeros((0, 384), dtype=np.float32)
    patches = np.stack(patches)  # (N, 64, 64)

    # Normalize per patch then resize to 224x224
    pmin = patches.min(axis=(1, 2), keepdims=True)
    pmax = patches.max(axis=(1, 2), keepdims=True)
    pn = (patches - pmin) / (pmax - pmin + 1e-8)
    t = torch.from_numpy(pn).float().unsqueeze(1).to(device)  # (N, 1, 64, 64)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.expand(-1, 3, -1, -1)  # grayscale -> RGB
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return _dino(t).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# OpenPhenom backbone
# ═══════════════════════════════════════════════════════════════════════════════

_phenom = None

def get_phenom():
    global _phenom
    if _phenom is None:
        logger.info("Loading OpenPhenom (recursionpharma/OpenPhenom)...")
        from transformers import AutoModel
        _phenom = AutoModel.from_pretrained(
            "recursionpharma/OpenPhenom",
            trust_remote_code=True,
        ).to(device).eval()
        # Some models return embeddings via .predict(), some via forward
    elif next(_phenom.parameters()).device != device:
        _phenom = _phenom.to(device)
    return _phenom

@torch.no_grad()
def extract_phenom(img, coords):
    """Extract OpenPhenom embeddings for cells."""
    get_phenom()
    crops = []
    for cy, cx in coords:
        p = extract_patch(img, cy, cx, PHENOM_CROP)
        # Normalize to [0, 255] uint8 range (as OpenPhenom expects)
        pmin, pmax = p.min(), p.max()
        if pmax > pmin:
            p = (p - pmin) / (pmax - pmin) * 255.0
        p = np.clip(p, 0, 255)
        crops.append(p.astype(np.uint8))
    if not crops:
        return np.zeros((0, 384), dtype=np.float32)
    crops = np.stack(crops)  # (N, 256, 256)

    # OpenPhenom expects (B, C, H, W) uint8
    t = torch.from_numpy(crops).unsqueeze(1).to(device)  # (N, 1, 256, 256)
    t = t.float()  # .predict() handles uint8, but we pass float for safety

    # Try .predict() first, fallback to forward()
    try:
        emb = _phenom.predict(t)
    except (AttributeError, TypeError, RuntimeError) as e:
        # Fallback: call forward and pool
        out = _phenom(t)
        if isinstance(out, dict):
            emb = out.get("pooler_output", out.get("last_hidden_state", list(out.values())[0]))
        elif isinstance(out, (tuple, list)):
            emb = out[0]
        else:
            emb = out
        if emb.dim() == 3:
            emb = emb.mean(dim=1)  # Pool sequences

    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    return emb


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gap(feats):
    """Intra-inter cosine similarity gap."""
    f = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    sim = f @ f.T
    N = f.shape[0]
    intra = np.diag(sim).mean()  # Always 1.0 for normalized
    mask = ~np.eye(N, dtype=bool)
    inter = sim[mask].mean()
    return inter  # Lower = more discriminative


def compute_effective_rank(feats, threshold=0.95):
    """Effective dimensionality (PCA var explained > threshold)."""
    f = feats - feats.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(f, full_matrices=False)
        var = S**2
        cumsum = np.cumsum(var) / np.sum(var)
        rank = np.searchsorted(cumsum, threshold) + 1
        return int(rank)
    except np.linalg.LinAlgError:
        return f.shape[-1]

def compute_recall_at_1(feats, labels):
    """Recall@1: fraction of cells where nearest neighbor has same label."""
    f = feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argmax(sim, axis=1)
    return float((labels[nn] == labels).mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    data_root = "../../data/vanvliet"
    max_frames = 30

    logger.info(f"Scanning frames from {data_root}...")
    frames = scan_frames(data_root, max_frames=max_frames)
    logger.info(f"Found {len(frames)} frames")

    all_dino, all_phenom, all_labels = [], [], []
    total_cells = 0

    for i, (mp, ip) in enumerate(frames):
        r = load_frame(mp, ip)
        if r is None:
            continue
        coords, labels, img = r
        n = len(coords)
        total_cells += n

        logger.info(f"  Frame {i+1}/{len(frames)}: {n} cells...")

        # Extract DINO features
        dino = extract_dino(img, coords)
        all_dino.append(dino)
        all_labels.append(labels[:len(dino)])

        # Extract OpenPhenom features
        try:
            phenom = extract_phenom(img, coords)
            all_phenom.append(phenom)
        except Exception as e:
            logger.warning(f"  OpenPhenom failed on frame {i+1}: {e}")
            all_phenom.append(np.zeros((n, 384), dtype=np.float32))

    # Aggregate
    dino_all = np.concatenate([d for d in all_dino if len(d) > 0])
    phenom_all = np.concatenate([p for p in all_phenom if len(p) > 0])
    labels_all = np.concatenate([l for l in all_labels if len(l) > 0])

    logger.info(f"\nTotal cells: {len(dino_all)}")

    # Clean up GPU
    global _dino, _phenom
    if _dino is not None:
        _dino = _dino.cpu()
    if _phenom is not None:
        _phenom = _phenom.cpu()
    torch.cuda.empty_cache()

    # Compute metrics
    print("\n" + "=" * 60)
    print("BACKBONE COMPARISON: DINOv2 vs OpenPhenom on Bacteria")
    print("=" * 60)

    for name, feats in [("DINOv2 vits14", dino_all), ("OpenPhenom", phenom_all)]:
        if len(feats) < 10:
            print(f"\n  {name}: too few features ({len(feats)}), skipping")
            continue
        gap = compute_gap(feats)
        rank = compute_effective_rank(feats)
        recall = compute_recall_at_1(feats, labels_all)
        print(f"\n  {name}:")
        print(f"    Inter-cell cosine sim: {gap:.4f}  (lower = more discriminative)")
        print(f"    Recall@1:               {recall:.4f}")
        print(f"    Effective rank (95%):   {rank}")
        print(f"    Feature shape:          {feats.shape}")

    # Comparison
    if len(dino_all) > 10 and len(phenom_all) > 10:
        dino_gap = compute_gap(dino_all)
        phenom_gap = compute_gap(phenom_all)
        delta = dino_gap - phenom_gap
        print(f"\n  Gap comparison:")
        print(f"    DINOv2:      {dino_gap:.4f}")
        print(f"    OpenPhenom:  {phenom_gap:.4f}")
        if delta > 0.02:
            print(f"    ✓ OpenPhenom {abs(delta):.4f} better")
        elif delta < -0.02:
            print(f"    ✓ DINOv2 {abs(delta):.4f} better")
        else:
            print(f"    ~ Equivalent ({delta:+.4f})")

    print("=" * 60)

if __name__ == "__main__":
    main()
