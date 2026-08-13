#!/usr/bin/env python3
"""Feature utility probing benchmark.

Tests how well different feature sources encode track-relevant information
by training linear/MLP probes to predict cell-cell edges (same-tracklet pairs)
between consecutive frames.

Feature extractors (all produce per-cell embeddings):
  - Regionprops 7D (baseline): eq_diameter_area, intensity_mean, inertia_tensor(4), border_dist
  - Rich Regionprops (13D):    7D + eccentricity, perimeter, solidity, extent, major_axis, minor_axis
  - Hu Moments (20D):          13D + 7 log-scaled Hu moments
  - DINOv2 frozen:             384D from dinov2_vits14
  - Tiny CNN (trained):        64D convnet trained end-to-end with probe
  - Tiny CNN (random):         64D frozen random convnet

Usage:
    uv run python -m src.analysis.feature_probe --data-root ../data/vanvliet --max-pairs 30 --steps 200
    uv run python -m src.analysis.feature_probe --features dino
    uv run python -m src.analysis.feature_probe --features all
"""

import argparse
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from skimage.measure import regionprops_table, regionprops as sk_regionprops
from skimage.transform import resize
from tifffile import imread

from src.data.feature_extraction import _border_dist_fast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("probe_features")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PATCH_SIZE = 64  # extraction patch size for DINO and CNN


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extractors
# ═══════════════════════════════════════════════════════════════════════════════

def _concatenate_features(features_dict):
    """Concatenate all feature arrays in an OrderedDict into a single (N, D) array."""
    arrays = []
    for name, arr in features_dict.items():
        arrays.append(arr)
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def extract_regionprops_7d(mask, img):
    """7D regionprops baseline: eq_diam, intensity_mean, inertia_tensor(4), border_dist."""
    ndim = mask.ndim
    props = ("equivalent_diameter_area", "intensity_mean", "inertia_tensor")
    df = pd.DataFrame(
        regionprops_table(mask, intensity_image=img, properties=("label", "centroid", *props))
    )
    if len(df) == 0:
        return None, None, None
    coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    inertias = np.stack(
        [np.column_stack([df[f"inertia_tensor-{i}-{j}"] for j in range(ndim)])
         for i in range(ndim)], axis=-1
    ).reshape(len(df), -1).astype(np.float32)

    features = OrderedDict()
    features["eq_diam"] = df["equivalent_diameter_area"].values.astype(np.float32)[:, None]
    features["intensity"] = df["intensity_mean"].values.astype(np.float32)[:, None]
    features["inertia"] = inertias
    features["border"] = np.array(list(_border_dist_fast(mask)), dtype=np.float32)[:, None]
    return coords, labels, _concatenate_features(features)


def extract_regionprops_13d(mask, img):
    """Rich 13D: 7D + eccentricity, perimeter, solidity, extent, major_axis, minor_axis."""
    ndim = mask.ndim
    basic_props = ("equivalent_diameter_area", "intensity_mean", "inertia_tensor")
    shape_props = ("eccentricity", "perimeter", "solidity", "extent",
                   "major_axis_length", "minor_axis_length")
    all_props = ("label", "centroid", *basic_props, *shape_props)
    df = pd.DataFrame(
        regionprops_table(mask, intensity_image=img, properties=all_props)
    )
    if len(df) == 0:
        return None, None, None
    coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    inertias = np.stack(
        [np.column_stack([df[f"inertia_tensor-{i}-{j}"] for j in range(ndim)])
         for i in range(ndim)], axis=-1
    ).reshape(len(df), -1).astype(np.float32)

    features = OrderedDict()
    features["eq_diam"] = df["equivalent_diameter_area"].values.astype(np.float32)[:, None]
    features["intensity"] = df["intensity_mean"].values.astype(np.float32)[:, None]
    features["inertia"] = inertias
    features["border"] = np.array(list(_border_dist_fast(mask)), dtype=np.float32)[:, None]
    features["eccentricity"] = df["eccentricity"].values.astype(np.float32)[:, None]
    features["perimeter"] = df["perimeter"].values.astype(np.float32)[:, None]
    features["solidity"] = df["solidity"].values.astype(np.float32)[:, None]
    features["extent"] = df["extent"].values.astype(np.float32)[:, None]
    features["axis_major"] = df["major_axis_length"].values.astype(np.float32)[:, None]
    features["axis_minor"] = df["minor_axis_length"].values.astype(np.float32)[:, None]
    return coords, labels, _concatenate_features(features)


def extract_hu_20d(mask, img):
    """Hu 20D: 13D + 7 log-scaled Hu moments."""
    result = extract_regionprops_13d(mask, img)
    if result[0] is None:
        return None, None, None
    coords, labels, feats_13d = result

    props_list = sk_regionprops(mask, intensity_image=img)
    hu_moments = np.array([r.moments_hu for r in props_list], dtype=np.float32)
    hu_log = np.sign(hu_moments) * np.log1p(np.abs(hu_moments))
    features_hu = np.concatenate([feats_13d, hu_log], axis=-1)
    return coords, labels, features_hu


def extract_patches(img, centroids, patch_size=PATCH_SIZE):
    """Extract square patches centered on centroids. Used for DINO and CNN."""
    h, w = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i = int(round(float(cy)))
        cx_i = int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, h - 1)
        cx_i = np.clip(cx_i, 0, w - 1)
        y1 = cy_i - half
        x1 = cx_i - half
        y2 = cy_i + half
        x2 = cx_i + half
        pt = max(0, -y1)
        pb = max(0, y2 - h)
        pl = max(0, -x1)
        pr = max(0, x2 - w)
        y1c = max(0, y1)
        x1c = max(0, x1)
        y2c = min(h, y2)
        x2c = min(w, x2)
        crop = img[y1c:y2c, x1c:x2c] if (y2c > y1c and x2c > x1c) else np.zeros((1, 1), dtype=np.float32)
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(
                crop,
                tuple((0, max(0, t)) for t in [patch_size - s for s in crop.shape]),
                mode="reflect",
            )[:patch_size, :patch_size]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, patch_size, patch_size), dtype=np.float32)


# ─── DINOv2 ──────────────────────────────────────────────────────────────────

DINO_LOADED = False
_dino_model = None

def load_dino():
    """Load the DINOv2-s backbone once and cache it on the module."""
    global DINO_LOADED, _dino_model
    if not DINO_LOADED:
        logger.info("Loading DINOv2...")
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        DINO_LOADED = True
    return _dino_model


def free_dino():
    """Free the cached DINOv2 model from GPU memory."""
    global DINO_LOADED, _dino_model
    if DINO_LOADED and _dino_model is not None:
        _dino_model.cpu()
        del _dino_model
        _dino_model = None
        DINO_LOADED = False
        torch.cuda.empty_cache()
        logger.info("DINOv2 freed from GPU")


@torch.no_grad()
def compute_dino_embs(patches_np, dino_model=None):
    """Compute 384D DINOv2 embeddings for a batch of patches."""
    if dino_model is None:
        dino_model = load_dino()
    if len(patches_np) == 0:
        return np.zeros((0, 384), dtype=np.float32)
    pmin = patches_np.min(axis=(1, 2), keepdims=True)
    pmax = patches_np.max(axis=(1, 2), keepdims=True)
    pn = (patches_np - pmin) / (pmax - pmin + 1e-8)
    t = torch.from_numpy(pn).float().unsqueeze(1).to(device)  # (N, 1, 64, 64)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.expand(-1, 3, -1, -1)  # → (N, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return dino_model(t).cpu().numpy()


# ─── Tiny CNN ────────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """Small convnet for bacteria patch feature extraction.

    Architecture: Conv(1→16, 3×3) → ReLU → MaxPool(2)
                  → Conv(16→32, 3×3) → ReLU → AdaptiveAvgPool(1)
                  → Linear(32, out_dim)
    """
    def __init__(self, out_dim=64):
        """Build the conv stack and the output linear layer."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, patches):
        """patches: (N, 1, 64, 64) or (N, 1, H, W) → (N, out_dim)."""
        x = self.conv(patches)  # (N, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (N, 32)
        return self.fc(x)  # (N, out_dim)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Load a single mask+image frame. Returns (coords, labels, image, mask) or None."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return coords, props["label"].astype(np.int32), img, mask


def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame PAIRS from the same experiment.

    Returns list of (mask_t, mask_t+1, img_t, img_t+1) paths.
    """
    dr = Path(data_root)
    pairs = []
    for cond in conditions:
        for exp in sorted((dr / cond).iterdir()):
            if not exp.is_dir():
                continue
            tra_dir = exp / "TRA"
            img_dir = exp / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            masks = sorted(tra_dir.glob("man_track*.tif"))
            for i in range(len(masks) - 1):
                stem1 = masks[i].stem.replace("man_track", "")
                stem2 = masks[i + 1].stem.replace("man_track", "")
                try:
                    f1, f2 = int(stem1), int(stem2)
                except ValueError:
                    continue
                if f2 == f1 + 1:
                    ip1 = img_dir / f"t{f1:06d}.tif"
                    ip2 = img_dir / f"t{f2:06d}.tif"
                    if ip1.exists() and ip2.exists():
                        pairs.append((
                            str(masks[i]), str(masks[i + 1]),
                            str(ip1), str(ip2),
                        ))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def build_edge_data(pairs, feature_key, max_pairs=30):
    """Build edge prediction data from consecutive frame pairs.

    For each pair:
      - Extract features for cells in frame t and frame t+1
      - Build target matrix: (N1, N2) with 1 for same-tracklet pairs, 0 otherwise

    Returns list of dicts with keys: feat_t, feat_n, target, n_pos, n_neg.
    feature_key controls which extractor is used.
    """
    edge_data = []
    total_pairs_processed = 0

    for mt, mn, it_, in_ in pairs[:max_pairs]:
        rt = load_frame(mt, it_)
        rn = load_frame(mn, in_)
        if rt is None or rn is None:
            continue
        coords_t, labels_t, imgt, mask_t = rt
        coords_n, labels_n, imgn, mask_n = rn

        # Only keep cells present in BOTH frames (shared tracking labels)
        shared_labels = set(labels_t) & set(labels_n)
        if len(shared_labels) < 2:
            continue

        idx_t = [i for i, l in enumerate(labels_t) if l in shared_labels]
        idx_n = [i for i, l in enumerate(labels_n) if l in shared_labels]
        labels_t_shared = labels_t[idx_t]
        labels_n_shared = labels_n[idx_n]

        # Extract features
        if feature_key == "regionprops_7d":
            _, labels_all_t, feats_all_t = extract_regionprops_7d(mask_t, imgt)
            _, labels_all_n, feats_all_n = extract_regionprops_7d(mask_n, imgn)
        elif feature_key == "regionprops_13d":
            _, labels_all_t, feats_all_t = extract_regionprops_13d(mask_t, imgt)
            _, labels_all_n, feats_all_n = extract_regionprops_13d(mask_n, imgn)
        elif feature_key == "hu_20d":
            _, labels_all_t, feats_all_t = extract_hu_20d(mask_t, imgt)
            _, labels_all_n, feats_all_n = extract_hu_20d(mask_n, imgn)
        elif feature_key in ("dino", "cnn_frozen", "cnn_trained"):
            # These need patches; features are computed elsewhere
            coords_t_shared = coords_t[idx_t]
            coords_n_shared = coords_n[idx_n]
            patches_t = extract_patches(imgt, coords_t_shared)
            patches_n = extract_patches(imgn, coords_n_shared)
            edge_data.append({
                "labels_t": labels_t_shared,
                "labels_n": labels_n_shared,
                "patches_t": patches_t,
                "patches_n": patches_n,
                "mask_t": mask_t,
                "mask_n": mask_n,
                "img_t": imgt,
                "img_n": imgn,
            })
            total_pairs_processed += 1
            continue
        else:
            raise ValueError(f"Unknown feature_key: {feature_key}")

        if feats_all_t is None or feats_all_n is None:
            continue

        # Filter features to only include labels present in shared_labels
        # regionprops returns rows sorted by label value
        # Build a label→index map for fast lookup
        label_to_idx_t = {l: i for i, l in enumerate(labels_all_t)}
        label_to_idx_n = {l: i for i, l in enumerate(labels_all_n)}

        idx_t_feat = [label_to_idx_t[l] for l in labels_t_shared if l in label_to_idx_t]
        idx_n_feat = [label_to_idx_n[l] for l in labels_n_shared if l in label_to_idx_n]

        if len(idx_t_feat) < 2 or len(idx_n_feat) < 2:
            continue

        feats_t = feats_all_t[idx_t_feat]
        feats_n = feats_all_n[idx_n_feat]

        # Build target matrix
        N1 = len(labels_t_shared)
        N2 = len(labels_n_shared)
        target = torch.zeros(N1, N2, dtype=torch.float32)
        for i, lt in enumerate(labels_t_shared):
            for j, ln in enumerate(labels_n_shared):
                if lt == ln:
                    target[i, j] = 1.0

        n_pos = target.sum().item()
        n_neg = N1 * N2 - n_pos

        edge_data.append({
            "feat_t": torch.from_numpy(feats_t).float(),
            "feat_n": torch.from_numpy(feats_n).float(),
            "target": target,
            "n_pos": n_pos,
            "n_neg": n_neg,
        })
        total_pairs_processed += 1

    logger.info(f"  Built {total_pairs_processed} pairs for {feature_key}")
    return edge_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Probe models
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Linear probe: concat(feat_t[i], feat_n[j]) → score."""
    def __init__(self, feat_dim):
        """Build a single linear layer over concatenated source/target features."""
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, 1)

    def forward(self, feat_t, feat_n):
        """feat_t: (N1, D), feat_n: (N2, D) → scores: (N1, N2)"""
        N1, D = feat_t.shape
        N2 = feat_n.shape[0]
        # Broadcasting: (N1, 1, D) + (1, N2, D) → (N1, N2, D)
        # Then concat along last dim → (N1, N2, 2D)
        feat_t_exp = feat_t.unsqueeze(1).expand(-1, N2, -1)   # (N1, N2, D)
        feat_n_exp = feat_n.unsqueeze(0).expand(N1, -1, -1)   # (N1, N2, D)
        pairs = torch.cat([feat_t_exp, feat_n_exp], dim=-1)   # (N1, N2, 2*D)
        scores = self.fc(pairs.view(-1, 2 * D)).view(N1, N2)  # (N1, N2)
        return scores


class MLPProbe(nn.Module):
    """2-layer MLP probe: concat(feat_t[i], feat_n[j]) → hidden → score."""
    def __init__(self, feat_dim, hidden=128):
        """Build a 2-layer MLP over concatenated source/target features."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_t, feat_n):
        """feat_t: (N1, D), feat_n: (N2, D) → scores: (N1, N2)"""
        N1, D = feat_t.shape
        N2 = feat_n.shape[0]
        feat_t_exp = feat_t.unsqueeze(1).expand(-1, N2, -1)
        feat_n_exp = feat_n.unsqueeze(0).expand(N1, -1, -1)
        pairs = torch.cat([feat_t_exp, feat_n_exp], dim=-1)
        scores = self.net(pairs.view(-1, 2 * D)).view(N1, N2)
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores, target):
    """Compute accuracy, balanced accuracy, F1, precision, recall."""
    pred = (scores > 0.0).float()  # scores are logits (BCEWithLogitsLoss)
    target_np = target.cpu().numpy().ravel()
    pred_np = pred.cpu().numpy().ravel()

    acc = accuracy_score(target_np, pred_np)
    bal_acc = balanced_accuracy_score(target_np, pred_np)
    f1 = f1_score(target_np, pred_np, zero_division=0)
    prec = precision_score(target_np, pred_np, zero_division=0)
    rec = recall_score(target_np, pred_np, zero_division=0)
    return acc, bal_acc, f1, prec, rec


def train_probe(probe, train_data, val_data, steps=200, lr=1e-3, patience=20, eval_every=20):
    """Train a probe model. Returns dict of metrics and trained probe."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_bal_acc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for step in range(steps):
        probe.train()
        train_losses = []
        for item in train_data:
            feat_t = item["feat_t"].to(device)
            feat_n = item["feat_n"].to(device)
            target = item["target"].to(device)

            scores = probe(feat_t, feat_n)
            loss = criterion(scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate
        if step % eval_every == 0 or step == steps - 1:
            probe.eval()
            val_metrics = {"acc": [], "bal_acc": [], "f1": [], "precision": [], "recall": []}
            val_losses = []
            with torch.no_grad():
                for item in val_data:
                    feat_t = item["feat_t"].to(device)
                    feat_n = item["feat_n"].to(device)
                    target = item["target"].to(device)
                    scores = probe(feat_t, feat_n)
                    loss = criterion(scores, target)
                    val_losses.append(loss.item())
                    acc, bal_acc, f1, prec, rec = compute_metrics(scores, target)
                    val_metrics["acc"].append(acc)
                    val_metrics["bal_acc"].append(bal_acc)
                    val_metrics["f1"].append(f1)
                    val_metrics["precision"].append(prec)
                    val_metrics["recall"].append(rec)

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss = float(np.mean(val_losses))
            avg_val_bal_acc = float(np.mean(val_metrics["bal_acc"]))
            avg_val_acc = float(np.mean(val_metrics["acc"]))
            avg_val_f1 = float(np.mean(val_metrics["f1"]))

            history.append({
                "step": step,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc,
                "val_bal_acc": avg_val_bal_acc,
                "val_f1": avg_val_f1,
            })

            # Early stopping check
            if avg_val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = avg_val_bal_acc
                patience_counter = 0
                best_state = probe.state_dict()
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"    Early stopping at step {step}")
                    break

    # Restore best state
    if best_state is not None:
        probe.load_state_dict(best_state)

    # Final evaluation
    probe.eval()
    final_metrics = {"acc": [], "bal_acc": [], "f1": [], "precision": [], "recall": []}
    with torch.no_grad():
        for item in val_data:
            feat_t = item["feat_t"].to(device)
            feat_n = item["feat_n"].to(device)
            target = item["target"].to(device)
            scores = probe(feat_t, feat_n)
            acc, bal_acc, f1, prec, rec = compute_metrics(scores, target)
            final_metrics["acc"].append(acc)
            final_metrics["bal_acc"].append(bal_acc)
            final_metrics["f1"].append(f1)
            final_metrics["precision"].append(prec)
            final_metrics["recall"].append(rec)

    result = {
        "probe": probe.cpu(),
        "history": history,
        "best_val_bal_acc": best_val_bal_acc,
        "final_acc": float(np.mean(final_metrics["acc"])),
        "final_bal_acc": float(np.mean(final_metrics["bal_acc"])),
        "final_f1": float(np.mean(final_metrics["f1"])),
        "final_precision": float(np.mean(final_metrics["precision"])),
        "final_recall": float(np.mean(final_metrics["recall"])),
    }
    return result


def precompute_dino_features(edge_data_raw):
    """Compute DINOv2 features for items that have patches stored."""
    logger.info("  Computing DINOv2 features for all patches...")
    dino_model = load_dino()

    edge_data = []
    for idx, item in enumerate(edge_data_raw):
        patches_t = item["patches_t"]
        patches_n = item["patches_n"]
        labels_t = item["labels_t"]
        labels_n = item["labels_n"]

        feats_t = compute_dino_embs(patches_t, dino_model)
        feats_n = compute_dino_embs(patches_n, dino_model)

        N1 = len(labels_t)
        N2 = len(labels_n)
        target = torch.zeros(N1, N2, dtype=torch.float32)
        for i, lt in enumerate(labels_t):
            for j, ln in enumerate(labels_n):
                if lt == ln:
                    target[i, j] = 1.0

        edge_data.append({
            "feat_t": torch.from_numpy(feats_t).float(),
            "feat_n": torch.from_numpy(feats_n).float(),
            "target": target,
            "n_pos": target.sum().item(),
            "n_neg": N1 * N2 - target.sum().item(),
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"    DINO features: {idx + 1}/{len(edge_data_raw)}")

    return edge_data


def prepare_cnn_edge_data(edge_data_raw, cnn_model):
    """Compute CNN features for all items (used for both frozen and trained)."""
    edge_data = []
    for item in edge_data_raw:
        patches_t = item["patches_t"]
        patches_n = item["patches_n"]
        labels_t = item["labels_t"]
        labels_n = item["labels_n"]

        # Features computed on-the-fly during training via the CNN forward pass
        patches_t_tensor = torch.from_numpy(patches_t).float().unsqueeze(1)  # (N, 1, H, W)
        patches_n_tensor = torch.from_numpy(patches_n).float().unsqueeze(1)

        N1 = len(labels_t)
        N2 = len(labels_n)
        target = torch.zeros(N1, N2, dtype=torch.float32)
        for i, lt in enumerate(labels_t):
            for j, ln in enumerate(labels_n):
                if lt == ln:
                    target[i, j] = 1.0

        edge_data.append({
            "patches_t": patches_t_tensor,
            "patches_n": patches_n_tensor,
            "target": target,
            "n_pos": target.sum().item(),
            "n_neg": N1 * N2 - target.sum().item(),
            "labels_t": labels_t,
            "labels_n": labels_n,
        })
    return edge_data


# ═══════════════════════════════════════════════════════════════════════════════
#  End-to-end trained CNN probe
# ═══════════════════════════════════════════════════════════════════════════════

class CNNProbe(nn.Module):
    """CNN feature extractor + probe, trained jointly."""
    def __init__(self, cnn_out_dim=64, probe_hidden=128, probe_type="linear"):
        """Build a TinyCNN backbone and a linear or MLP probe on top."""
        super().__init__()
        self.cnn = TinyCNN(out_dim=cnn_out_dim)
        if probe_type == "linear":
            self.probe = nn.Linear(2 * cnn_out_dim, 1)
        else:
            self.probe = nn.Sequential(
                nn.Linear(2 * cnn_out_dim, probe_hidden),
                nn.ReLU(),
                nn.Linear(probe_hidden, 1),
            )
        self.probe_type = probe_type

    def forward(self, patches_t, patches_n):
        """patches_t: (N1, 1, H, W), patches_n: (N2, 1, H, W) → scores: (N1, N2)"""
        feat_t = self.cnn(patches_t)  # (N1, D)
        feat_n = self.cnn(patches_n)  # (N2, D)
        N1, D = feat_t.shape
        N2 = feat_n.shape[0]
        feat_t_exp = feat_t.unsqueeze(1).expand(-1, N2, -1)
        feat_n_exp = feat_n.unsqueeze(0).expand(N1, -1, -1)
        pairs = torch.cat([feat_t_exp, feat_n_exp], dim=-1)
        scores = self.probe(pairs.view(-1, 2 * D)).view(N1, N2)
        return scores


def train_cnn_probe(train_data, val_data, probe_type="linear", steps=200, lr=1e-4,
                     patience=20, eval_every=20):
    """Train a CNNProbe end-to-end on edge prediction."""
    model = CNNProbe(probe_type=probe_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_bal_acc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for step in range(steps):
        model.train()
        train_losses = []
        for item in train_data:
            patches_t = item["patches_t"].to(device)
            patches_n = item["patches_n"].to(device)
            target = item["target"].to(device)

            scores = model(patches_t, patches_n)
            loss = criterion(scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if step % eval_every == 0 or step == steps - 1:
            model.eval()
            val_metrics_list = []
            val_losses = []
            with torch.no_grad():
                for item in val_data:
                    patches_t = item["patches_t"].to(device)
                    patches_n = item["patches_n"].to(device)
                    target = item["target"].to(device)
                    scores = model(patches_t, patches_n)
                    loss = criterion(scores, target)
                    val_losses.append(loss.item())
                    acc, bal_acc, f1, prec, rec = compute_metrics(scores, target)
                    val_metrics_list.append({
                        "acc": acc, "bal_acc": bal_acc, "f1": f1,
                        "precision": prec, "recall": rec,
                    })

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss = float(np.mean(val_losses))
            avg_val_bal_acc = float(np.mean([m["bal_acc"] for m in val_metrics_list]))
            avg_val_acc = float(np.mean([m["acc"] for m in val_metrics_list]))
            avg_val_f1 = float(np.mean([m["f1"] for m in val_metrics_list]))

            history.append({
                "step": step,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc,
                "val_bal_acc": avg_val_bal_acc,
                "val_f1": avg_val_f1,
            })

            if avg_val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = avg_val_bal_acc
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"    Early stopping at step {step}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    final_metrics_list = []
    with torch.no_grad():
        for item in val_data:
            patches_t = item["patches_t"].to(device)
            patches_n = item["patches_n"].to(device)
            target = item["target"].to(device)
            scores = model(patches_t, patches_n)
            acc, bal_acc, f1, prec, rec = compute_metrics(scores, target)
            final_metrics_list.append({
                "acc": acc, "bal_acc": bal_acc, "f1": f1,
                "precision": prec, "recall": rec,
            })

    result = {
        "model": model.cpu(),
        "history": history,
        "best_val_bal_acc": best_val_bal_acc,
        "final_acc": float(np.mean([m["acc"] for m in final_metrics_list])),
        "final_bal_acc": float(np.mean([m["bal_acc"] for m in final_metrics_list])),
        "final_f1": float(np.mean([m["f1"] for m in final_metrics_list])),
        "final_precision": float(np.mean([m["precision"] for m in final_metrics_list])),
        "final_recall": float(np.mean([m["recall"] for m in final_metrics_list])),
    }
    return result


def probe_with_frozen_cnn(train_data, val_data, cnn_model, probe_type="linear",
                          steps=200, lr=1e-3, patience=20, eval_every=20):
    """Train a probe on top of frozen CNN features."""
    # Move CNN to device
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    # Precompute frozen CNN features
    train_data_feat = []
    for item in train_data:
        patches_t = item["patches_t"].to(device)
        patches_n = item["patches_n"].to(device)
        with torch.no_grad():
            feat_t = cnn_model(patches_t).cpu()
            feat_n = cnn_model(patches_n).cpu()
        train_data_feat.append({
            "feat_t": feat_t,
            "feat_n": feat_n,
            "target": item["target"],
        })

    val_data_feat = []
    for item in val_data:
        patches_t = item["patches_t"].to(device)
        patches_n = item["patches_n"].to(device)
        with torch.no_grad():
            feat_t = cnn_model(patches_t).cpu()
            feat_n = cnn_model(patches_n).cpu()
        val_data_feat.append({
            "feat_t": feat_t,
            "feat_n": feat_n,
            "target": item["target"],
        })

    feat_dim = cnn_model.fc.out_features
    if probe_type == "linear":
        probe = LinearProbe(feat_dim)
        probe_lr = 1e-3
    else:
        probe = MLPProbe(feat_dim)
        probe_lr = 1e-4

    return train_probe(probe, train_data_feat, val_data_feat,
                       steps=steps, lr=probe_lr, patience=patience, eval_every=eval_every)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_feature_evaluation(feature_key, pairs, args):
    """Run full evaluation for a single feature extractor.

    Returns dict with linear/MLP probe results.
    """
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Feature: {feature_key}")
    logger.info(f"{'─' * 60}")

    # ── Build edge data ──────────────────────────────────────────────────
    raw_pairs = list(pairs)

    if feature_key in ("regionprops_7d", "regionprops_13d", "hu_20d"):
        # Build data directly
        edge_data = build_edge_data(raw_pairs, feature_key, args.max_pairs)
    elif feature_key == "dino":
        # Build raw data with patches, then compute DINO features
        edge_data_raw = build_edge_data(raw_pairs, feature_key, args.max_pairs)
        edge_data = precompute_dino_features(edge_data_raw)
        free_dino()
    elif feature_key == "cnn_trained":
        edge_data_raw = build_edge_data(raw_pairs, feature_key, args.max_pairs)
        edge_data = prepare_cnn_edge_data(edge_data_raw, None)
    elif feature_key == "cnn_frozen":
        edge_data_raw = build_edge_data(raw_pairs, feature_key, args.max_pairs)
        edge_data = prepare_cnn_edge_data(edge_data_raw, None)
    else:
        raise ValueError(f"Unknown feature_key: {feature_key}")

    if len(edge_data) < 3:
        logger.warning(f"  Not enough data pairs ({len(edge_data)}), skipping")
        return None

    # ── Train/val split ──────────────────────────────────────────────────
    np.random.seed(SEED)
    idx = np.random.permutation(len(edge_data))
    n_val = max(1, int(len(edge_data) * 0.2))
    train_data = [edge_data[i] for i in idx[:-n_val]]
    val_data = [edge_data[i] for i in idx[-n_val:]]

    pos_ratios = [d["n_pos"] / (d["n_pos"] + d["n_neg"] + 1e-8) for d in train_data]
    logger.info(f"  Train pairs: {len(train_data)}, Val pairs: {len(val_data)}")
    logger.info(f"  Avg positive ratio: {np.mean(pos_ratios):.4f}")

    # ── Determine feature dimension ──────────────────────────────────────
    if feature_key in ("regionprops_7d",):
        feat_dim = 7
    elif feature_key in ("regionprops_13d",):
        feat_dim = 13
    elif feature_key in ("hu_20d",):
        feat_dim = 20
    elif feature_key == "dino":
        feat_dim = 384
    elif feature_key in ("cnn_trained", "cnn_frozen"):
        feat_dim = 64
    else:
        feat_dim = edge_data[0]["feat_t"].shape[-1]

    results = {}

    # ── Linear probe ─────────────────────────────────────────────────────
    logger.info(f"  Training linear probe (feat_dim={feat_dim})...")
    t0 = time.time()

    if feature_key == "cnn_trained":
        lin_result = train_cnn_probe(
            train_data, val_data, probe_type="linear",
            steps=args.steps, lr=1e-4, patience=20, eval_every=20,
        )
    elif feature_key == "cnn_frozen":
        cnn_frozen = TinyCNN(out_dim=64).eval()
        lin_result = probe_with_frozen_cnn(
            train_data, val_data, cnn_frozen, probe_type="linear",
            steps=args.steps, lr=1e-3, patience=20, eval_every=20,
        )
    else:
        probe = LinearProbe(feat_dim)
        lin_result = train_probe(
            probe, train_data, val_data,
            steps=args.steps, lr=1e-3, patience=20, eval_every=20,
        )

    logger.info(f"    Linear probe done in {time.time() - t0:.1f}s")
    logger.info(f"    Val bal_acc={lin_result['final_bal_acc']:.4f}, "
                f"acc={lin_result['final_acc']:.4f}, f1={lin_result['final_f1']:.4f}")
    results["linear"] = lin_result

    # ── MLP probe ────────────────────────────────────────────────────────
    logger.info(f"  Training MLP probe (feat_dim={feat_dim}, hidden=128)...")
    t0 = time.time()

    if feature_key == "cnn_trained":
        mlp_result = train_cnn_probe(
            train_data, val_data, probe_type="mlp",
            steps=args.steps, lr=1e-4, patience=20, eval_every=20,
        )
    elif feature_key == "cnn_frozen":
        cnn_frozen = TinyCNN(out_dim=64).eval()
        mlp_result = probe_with_frozen_cnn(
            train_data, val_data, cnn_frozen, probe_type="mlp",
            steps=args.steps, lr=1e-4, patience=20, eval_every=20,
        )
    else:
        probe = MLPProbe(feat_dim)
        mlp_result = train_probe(
            probe, train_data, val_data,
            steps=args.steps, lr=1e-4, patience=20, eval_every=20,
        )

    logger.info(f"    MLP probe done in {time.time() - t0:.1f}s")
    logger.info(f"    Val bal_acc={mlp_result['final_bal_acc']:.4f}, "
                f"acc={mlp_result['final_acc']:.4f}, f1={mlp_result['final_f1']:.4f}")
    results["mlp"] = mlp_result

    return results


def print_results_table(all_results):
    """Print a summary table of all probe results."""
    print()
    print("=" * 80)
    print("Feature Utility Probing Benchmark — Results")
    print("=" * 80)
    print(f"{'Feature Extractor':<30} {'Linear BalAcc':<18} {'MLP BalAcc':<18} {'Notes'}")
    print("-" * 80)

    feature_notes = {
        "regionprops_7d": "baseline",
        "regionprops_13d": "",
        "hu_20d": "",
        "dino": "~22M params, ~20ms/cell",
        "cnn_trained": "~5K params, ~0.1ms/cell",
        "cnn_frozen": "untrained baseline",
    }
    feature_display = {
        "regionprops_7d": "Regionprops 7D",
        "regionprops_13d": "Rich Regionprops 13D",
        "hu_20d": "Hu Moments 20D",
        "dino": "DINOv2 384D (frozen)",
        "cnn_trained": "Tiny CNN 64D (trained)",
        "cnn_frozen": "Tiny CNN 64D (random)",
    }

    for fkey in ["regionprops_7d", "regionprops_13d", "hu_20d", "dino", "cnn_trained", "cnn_frozen"]:
        if fkey not in all_results or all_results[fkey] is None:
            print(f"{feature_display[fkey]:<30} {'—':<18} {'—':<18} {feature_notes[fkey]}")
            continue
        r = all_results[fkey]
        lin_bal = r["linear"]["final_bal_acc"] if "linear" in r and r["linear"] else float("nan")
        mlp_bal = r["mlp"]["final_bal_acc"] if "mlp" in r and r["mlp"] else float("nan")
        notes = feature_notes[fkey]
        print(f"{feature_display[fkey]:<30} {lin_bal:<18.4f} {mlp_bal:<18.4f} {notes}")

    print("=" * 80)

    # Also print detailed metrics
    print()
    print("Detailed metrics:")
    print("-" * 80)
    header = f"{'Feature':<25} {'Probe':<8} {'Acc':<10} {'BalAcc':<10} {'F1':<10} {'Prec':<10} {'Recall':<10}"
    print(header)
    print("-" * 80)
    for fkey in ["regionprops_7d", "regionprops_13d", "hu_20d", "dino", "cnn_trained", "cnn_frozen"]:
        if fkey not in all_results or all_results[fkey] is None:
            continue
        r = all_results[fkey]
        for ptype in ["linear", "mlp"]:
            if ptype not in r or r[ptype] is None:
                continue
            m = r[ptype]
            name = feature_display[fkey]
            print(f"{name:<25} {ptype:<8} {m['final_acc']:<10.4f} {m['final_bal_acc']:<10.4f} "
                  f"{m['final_f1']:<10.4f} {m['final_precision']:<10.4f} {m['final_recall']:<10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_FEATURES = ["regionprops_7d", "regionprops_13d", "hu_20d", "dino", "cnn_trained", "cnn_frozen"]
FEATURE_ALIASES = {
    "all": AVAILABLE_FEATURES,
    "regionprops": ["regionprops_7d", "regionprops_13d", "hu_20d"],
    "cnn": ["cnn_trained", "cnn_frozen"],
}


def parse_features(features_arg):
    """Parse --features argument into list of feature keys."""
    if features_arg in FEATURE_ALIASES:
        return list(FEATURE_ALIASES[features_arg])
    # Could be comma-separated list
    keys = [f.strip() for f in features_arg.split(",")]
    for k in keys:
        if k not in AVAILABLE_FEATURES:
            logger.warning(f"Unknown feature: {k}. Available: {AVAILABLE_FEATURES}")
    return [k for k in keys if k in AVAILABLE_FEATURES]


def main():
    """CLI entry point: run the feature utility probing benchmark."""
    parser = argparse.ArgumentParser(
        description="Feature utility probing benchmark for cell tracking"
    )
    parser.add_argument(
        "--data-root", default="../data/vanvliet",
        help="Path to vanvliet data root"
    )
    parser.add_argument(
        "--conditions", default="rpsM,recA",
        help="Comma-separated list of conditions (default: rpsM,recA)"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=30,
        help="Maximum number of consecutive frame pairs to use"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of training steps for each probe"
    )
    parser.add_argument(
        "--features", type=str, default="all",
        help=f"Features to test: comma-separated or alias. "
             f"Available: {AVAILABLE_FEATURES}. Aliases: all, regionprops, cnn"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    conditions = [c.strip() for c in args.conditions.split(",")]
    features_to_run = parse_features(args.features)

    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Features: {features_to_run}")
    logger.info(f"Max pairs: {args.max_pairs}, Steps: {args.steps}")
    logger.info(f"Device: {device}")

    # ── Scan pairs ──────────────────────────────────────────────────────
    logger.info("Scanning for consecutive frame pairs...")
    all_pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Found {len(all_pairs)} consecutive frame pairs")

    if len(all_pairs) < 3:
        logger.error("Need at least 3 consecutive frame pairs. Check data.")
        sys.exit(1)

    # ── Run each feature ────────────────────────────────────────────────
    all_results = {}
    for fkey in features_to_run:
        try:
            result = run_feature_evaluation(fkey, all_pairs, args)
            all_results[fkey] = result
        except Exception as e:
            logger.error(f"Error running {fkey}: {e}", exc_info=True)
            all_results[fkey] = None

    # ── Print results ───────────────────────────────────────────────────
    print_results_table(all_results)

    logger.info("Done.")


if __name__ == "__main__":
    main()
