#!/usr/bin/env python3
"""
Unified Edge Probing Benchmark

Tests how well different feature types predict cell-cell associations (edges)
between consecutive frames using linear/MLP probes.

Feature types:
  rp        — Regionprops 7D (wrfeat-style): eq_diam, intensity_mean,
              inertia_tensor(4), border_dist            [7 dim]
  cnn_frozen — ScaledCNN with NT-Xent checkpoint         [128 dim]
  cnn_e2e   — ScaledCNN trained jointly with probe       [128 dim]
  dino      — DINOv2 (dinov2_vits14, frozen)             [384 dim]

Probe architectures (tested for each feature type):
  Linear: nn.Linear(2*feat_dim, 1)
  MLP:    nn.Sequential(Linear(2*feat_dim, 128), ReLU, Linear(128, 1))

Usage:
  # test all features with both probes
  python unified_edge_probe.py --features all --probe both

  # quick test: only regionprops, linear probe
  python unified_edge_probe.py --features rp --probe linear --epochs 50

  # specific feature
  python unified_edge_probe.py --features dino --probe both --epochs 200

  # save results to custom path
  python unified_edge_probe.py --output results.json
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from skimage.measure import regionprops_table, regionprops as sk_regionprops
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tifffile import imread

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("unified_edge_probe")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PATCH_SIZE = 64

# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading (shared across all feature types)
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Load mask + image, return (coords, labels, img, mask) or None."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack(
        [props["centroid-0"], props["centroid-1"]], axis=-1
    ).astype(np.float32)
    return coords, props["label"].astype(np.int32), img, mask


def load_tracklets(man_track_path):
    """Parse man_track.txt into dict: label -> {t1, t2, parent}."""
    df = pd.read_csv(
        man_track_path, delimiter=' ', header=None,
        names=['label', 't1', 't2', 'parent']
    )
    tracklets = {}
    for _, row in df.iterrows():
        tracklets[int(row['label'])] = {
            't1': int(row['t1']),
            't2': int(row['t2']),
            'parent': int(row['parent']),
        }
    return tracklets


def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame pairs. Returns list of (mask_t, mask_n, img_t,
    img_n, man_track_path, condition, experiment)."""
    dr = Path(data_root)
    pairs = []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir():
                continue
            tra_dir = exp / "TRA"
            img_dir = exp / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            man_track_txt = tra_dir / "man_track.txt"
            if not man_track_txt.exists():
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
                            str(man_track_txt),
                            cond, exp.name,
                        ))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def extract_patches(img, centroids, patch_size=PATCH_SIZE):
    """Extract square patches centered on centroids. Handles edge padding."""
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
        crop = (
            img[y1c:y2c, x1c:x2c]
            if y2c > y1c and x2c > x1c
            else np.zeros((1, 1), dtype=np.float32)
        )
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extractors
# ═══════════════════════════════════════════════════════════════════════════════

# ── Regionprops 7D (wrfeat-style) ────────────────────────────────────────────

def _border_dist_fast(mask, cutoff=5):
    """Per-cell border distance (normalized 0=far, 1=border)."""
    cutoff = int(cutoff)
    border = np.ones(mask.shape, dtype=np.float32)
    ndim = mask.ndim
    for axis, size in enumerate(mask.shape):
        if axis < ndim - 2:
            continue
        band_vals = np.arange(cutoff, dtype=np.float32) / cutoff
        band_vals = band_vals[:size]
        low_slices = [slice(None)] * ndim
        low_slices[axis] = slice(0, cutoff)
        border_low = border[tuple(low_slices)]
        border_low_vals = np.minimum(
            border_low, band_vals[(...,) + (None,) * (ndim - axis - 1)]
        )
        border[tuple(low_slices)] = border_low_vals
        high_slices = [slice(None)] * ndim
        high_slices[axis] = slice(max(0, size - cutoff), size)
        band_vals_rev = band_vals[::-1]
        border_high = border[tuple(high_slices)]
        border_high_vals = np.minimum(
            border_high, band_vals_rev[(...,) + (None,) * (ndim - axis - 1)]
        )
        border[tuple(high_slices)] = border_high_vals
    dist = 1 - border
    return tuple(r.intensity_max for r in sk_regionprops(mask, intensity_image=dist))


def extract_regionprops_7d(mask, img):
    """
    7D regionprops (wrfeat 'regionprops2' style):
      eq_diam, intensity_mean, inertia_tensor(4), border_dist.
    Returns (coords, labels, features) or (None, None, None).
    """
    ndim = mask.ndim
    props = ("equivalent_diameter_area", "intensity_mean", "inertia_tensor")
    df = pd.DataFrame(
        regionprops_table(mask, intensity_image=img,
                          properties=("label", "centroid", *props))
    )
    if len(df) == 0:
        return None, None, None
    coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    # inertia_tensor columns come as inertia_tensor-0-0, inertia_tensor-0-1, etc.
    inertias = np.stack(
        [np.column_stack([df[f"inertia_tensor-{i}-{j}"] for j in range(ndim)])
         for i in range(ndim)], axis=-1
    ).reshape(len(df), -1).astype(np.float32)

    features = OrderedDict()
    features["eq_diam"] = df["equivalent_diameter_area"].values.astype(np.float32)[:, None]
    features["intensity"] = df["intensity_mean"].values.astype(np.float32)[:, None]
    features["inertia"] = inertias
    features["border"] = np.array(list(_border_dist_fast(mask)), dtype=np.float32)[:, None]

    feats_combined = np.concatenate(list(features.values()), axis=-1).astype(np.float32)
    return coords, labels, feats_combined


# ── ScaledCNN (same arch as cnn_ssl.py) ──────────────────────────────────────

class ScaledCNN(nn.Module):
    """ConvNet for 64x64 grayscale patches. Output: 128-dim embedding."""
    def __init__(self, scale='large', out_dim=128):
        super().__init__()
        if scale == 'small':
            ch = [8, 16, 32]
        elif scale == 'medium':
            ch = [16, 32, 64]
        else:  # large
            ch = [32, 64, 128, 256]

        layers = []
        in_ch = 1
        for c in ch:
            layers += [nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = c
        self.conv = nn.Sequential(*layers)
        spatial = PATCH_SIZE // (2 ** len(ch))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, patches):
        return self.fc(self.conv(patches))


# ── DINOv2 (lazy-loaded, cached) ────────────────────────────────────────────

_DINO_MODEL = None

def load_dino():
    global _DINO_MODEL
    if _DINO_MODEL is None:
        logger.info("  Loading DINOv2 (dinov2_vits14)...")
        _DINO_MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
        _DINO_MODEL.eval()
    return _DINO_MODEL


@torch.no_grad()
def compute_dino_embs(patches_np):
    """Compute 384D DINOv2 embeddings for a batch of 64x64 patches."""
    if len(patches_np) == 0:
        return np.zeros((0, 384), dtype=np.float32)
    model = load_dino()
    # Per-patch min-max normalization
    pmin = patches_np.min(axis=(1, 2), keepdims=True)
    pmax = patches_np.max(axis=(1, 2), keepdims=True)
    pn = (patches_np - pmin) / (pmax - pmin + 1e-8)
    t = torch.from_numpy(pn).float().unsqueeze(1).to(device)  # (N, 1, 64, 64)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.expand(-1, 3, -1, -1)  # -> (N, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return model(t).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature caching
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent / "feature_cache"

def _cache_path(cond, exp, frame, feat_type):
    """Return path to cached .npy file for a given frame and feature type."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{cond}_{exp}_t{frame:06d}_{feat_type}.npy"
    return CACHE_DIR / safe_name


def _load_cached_features(cond, exp, frame, feat_type, labels):
    """Load cached features. Returns (features, labels) or (None, None)."""
    path = _cache_path(cond, exp, frame, feat_type)
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    # data is a dict with keys 'features' and 'labels'
    if isinstance(data, np.ndarray) and data.dtype == np.object_:
        data = data.item()
    if isinstance(data, dict):
        return data.get("features"), data.get("labels")
    return None, None


def _save_cached_features(cond, exp, frame, feat_type, features, labels):
    """Save features to cache."""
    path = _cache_path(cond, exp, frame, feat_type)
    np.save(path, {"features": features, "labels": labels})
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Pair-level dataset (balanced per-frame-pair)
# ═══════════════════════════════════════════════════════════════════════════════

class EdgePairDataset(Dataset):
    """
    Dataset of individual cell-cell pairs.
    Each sample is (feat_i, feat_j, label) where
      feat_i = feature of cell i in frame t
      feat_j = feature of cell j in frame t+1
      label  = 1 if same-tracklet, 0 otherwise
    """
    def __init__(self, feat_t, feat_n, target):
        N1, N2 = target.shape
        pairs_t, pairs_n, labels = [], [], []
        for i in range(N1):
            for j in range(N2):
                pairs_t.append(feat_t[i])
                pairs_n.append(feat_n[j])
                labels.append(int(target[i, j].item()))
        self.pairs_t = torch.stack(pairs_t) if pairs_t else torch.empty(0, feat_t.shape[-1])
        self.pairs_n = torch.stack(pairs_n) if pairs_n else torch.empty(0, feat_n.shape[-1])
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pairs_t[idx], self.pairs_n[idx], self.labels[idx]


class EdgePairDatasetPatches(Dataset):
    """
    Dataset of individual cell-cell pairs where features are patches
    (for CNN end-to-end training).
    """
    def __init__(self, patches_t, patches_n, target):
        N1, N2 = target.shape
        p_t, p_n, labels = [], [], []
        for i in range(N1):
            for j in range(N2):
                p_t.append(patches_t[i])
                p_n.append(patches_n[j])
                labels.append(int(target[i, j].item()))
        self.patches_t = torch.stack(p_t) if p_t else torch.empty(0, *patches_t.shape[1:])
        self.patches_n = torch.stack(p_n) if p_n else torch.empty(0, *patches_n.shape[1:])
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches_t[idx], self.patches_n[idx], self.labels[idx]


def _gather_labels(dataset):
    """Gather all labels from an EdgePairDataset or ConcatDataset of them."""
    if hasattr(dataset, 'labels'):
        return dataset.labels.numpy()
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        labels_list = []
        for ds in dataset.datasets:
            if hasattr(ds, 'labels'):
                labels_list.append(ds.labels.numpy())
        if labels_list:
            return np.concatenate(labels_list)
    raise TypeError(f"Cannot gather labels from {type(dataset)}")


def make_balanced_dataloader(dataset, batch_size, shuffle=True):
    """Create a DataLoader with WeightedRandomSampler for balanced batches."""
    labels = _gather_labels(dataset)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        # Fall back to regular loader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Weight: minority class gets higher weight
    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg
    sample_weights = np.where(labels == 1, weight_pos, weight_neg)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def collate_pairs_to_full(batch):
    """
    Collate a batch of (feat_t, feat_n, label) triples.
    For metrics we need the full scores, but for the loss we use per-pair BCE.
    Just return stacked tensors.
    """
    feat_t = torch.stack([b[0] for b in batch])
    feat_n = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return feat_t, feat_n, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Probe architectures
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Linear probe: concat(feat_t[i], feat_n[j]) -> score."""
    def __init__(self, feat_dim):
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, 1)

    def forward(self, feat_t, feat_n):
        """feat_t: (N, D), feat_n: (N, D) -> scores: (N,)"""
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.fc(pairs).squeeze(-1)  # (N,)


class MLPProbe(nn.Module):
    """2-layer MLP probe: concat(feat_t[i], feat_n[j]) -> hidden -> score."""
    def __init__(self, feat_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_t, feat_n):
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.net(pairs).squeeze(-1)  # (N,)


class CNNProbeE2E(nn.Module):
    """ScaledCNN + probe, trained end-to-end."""
    def __init__(self, scale='large', out_dim=128, probe_type='linear'):
        super().__init__()
        self.cnn = ScaledCNN(scale=scale, out_dim=out_dim)
        if probe_type == 'linear':
            self.probe = nn.Linear(2 * out_dim, 1)
        else:
            self.probe = nn.Sequential(
                nn.Linear(2 * out_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def forward(self, patches_t, patches_n):
        """patches_t: (N, 1, 64, 64), patches_n: (N, 1, 64, 64) -> scores: (N,)"""
        feat_t = self.cnn(patches_t)  # (N, D)
        feat_n = self.cnn(patches_n)  # (N, D)
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.probe(pairs).squeeze(-1)  # (N,)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training and evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores, target):
    """Compute all metrics from logits and binary targets."""
    pred = (scores > 0.0).float()
    target_np = target.cpu().numpy()
    pred_np = pred.cpu().numpy()
    bal_acc = balanced_accuracy_score(target_np, pred_np)
    f1 = f1_score(target_np, pred_np, zero_division=0)
    prec = precision_score(target_np, pred_np, zero_division=0)
    rec = recall_score(target_np, pred_np, zero_division=0)
    return bal_acc, f1, prec, rec


def train_probe(probe, train_loader, val_loader, epochs=200, lr=1e-3,
                patience=10, eval_every=1, is_e2e=False):
    """
    Train a probe (or CNN+probe) on pair-level data.
    Returns dict of metrics.
    """
    probe = probe.to(device)
    if is_e2e:
        # End-to-end: train all params (CNN + probe)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_bal_acc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        probe.train()
        train_losses = []
        for feat_t, feat_n, target in train_loader:
            feat_t = feat_t.to(device)
            feat_n = feat_n.to(device)
            target = target.to(device)

            scores = probe(feat_t, feat_n)
            loss = criterion(scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate
        if epoch % eval_every == 0 or epoch == epochs - 1:
            probe.eval()
            val_losses = []
            val_metrics_list = []
            with torch.no_grad():
                for feat_t, feat_n, target in val_loader:
                    feat_t = feat_t.to(device)
                    feat_n = feat_n.to(device)
                    target = target.to(device)

                    scores = probe(feat_t, feat_n)
                    loss = criterion(scores, target)
                    val_losses.append(loss.item())
                    ba, f1, prec, rec = compute_metrics(scores, target)
                    val_metrics_list.append({
                        "bal_acc": ba, "f1": f1,
                        "precision": prec, "recall": rec,
                    })

            avg_train_loss = float(np.mean(train_losses))
            avg_val_loss = float(np.mean(val_losses))
            avg_val_bal_acc = float(np.mean([m["bal_acc"] for m in val_metrics_list]))
            avg_val_f1 = float(np.mean([m["f1"] for m in val_metrics_list]))

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_bal_acc": avg_val_bal_acc,
                "val_f1": avg_val_f1,
            })

            # Early stopping
            if avg_val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = avg_val_bal_acc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"      Early stopping at epoch {epoch}")
                    break

    # Restore best state
    if best_state is not None:
        probe.load_state_dict(best_state)

    # Final evaluation
    probe.eval()
    final_metrics_list = []
    with torch.no_grad():
        for feat_t, feat_n, target in val_loader:
            feat_t = feat_t.to(device)
            feat_n = feat_n.to(device)
            target = target.to(device)
            scores = probe(feat_t, feat_n)
            ba, f1, prec, rec = compute_metrics(scores, target)
            final_metrics_list.append({
                "bal_acc": ba, "f1": f1,
                "precision": prec, "recall": rec,
            })

    result = {
        "final_bal_acc": float(np.mean([m["bal_acc"] for m in final_metrics_list])),
        "final_f1": float(np.mean([m["f1"] for m in final_metrics_list])),
        "final_precision": float(np.mean([m["precision"] for m in final_metrics_list])),
        "final_recall": float(np.mean([m["recall"] for m in final_metrics_list])),
        "history": history,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-frame-pair data building
# ═══════════════════════════════════════════════════════════════════════════════

def build_frame_pairs(pairs, max_pairs, feature_type, checkpoint_path=None):
    """
    Build per-frame-pair data for a given feature type.

    Returns list of dicts, each containing:
      - feat_t: (N1, D) or patches_t: (N1, 1, 64, 64)
      - feat_n: (N2, D) or patches_n: (N2, 1, 64, 64)
      - target: (N1, N2) binary matrix
      - n_pos, n_neg: counts
      - condition: string
    """
    edge_data = []
    total = 0

    # For CNN frozen: load model once
    cnn_frozen_model = None
    if feature_type == 'cnn_frozen':
        logger.info("  Loading ScaledCNN (large) with NT-Xent checkpoint...")
        cnn_frozen_model = ScaledCNN(scale='large', out_dim=128).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        cnn_frozen_model.load_state_dict(ckpt["model_state_dict"])
        cnn_frozen_model.eval()
        logger.info(f"  Loaded checkpoint: {checkpoint_path}")

    for idx, (mt, mn, it_, in_, man_txt, cond, exp) in enumerate(pairs[:max_pairs]):
        rt = load_frame(mt, it_)
        rn = load_frame(mn, in_)
        if rt is None or rn is None:
            continue
        coords_t, labels_t, imgt, mask_t = rt
        coords_n, labels_n, imgn, mask_n = rn

        if len(labels_t) < 3 or len(labels_n) < 3:
            continue

        tracklets = load_tracklets(man_txt)

        # Extract features based on type
        if feature_type == 'rp':
            # Regionprops 7D
            _, labels_rp_t, feats_t = extract_regionprops_7d(mask_t, imgt)
            _, labels_rp_n, feats_n = extract_regionprops_7d(mask_n, imgn)
            if feats_t is None or feats_n is None:
                continue
            # Align by label
            lt_map = {l: i for i, l in enumerate(labels_rp_t)}
            ln_map = {l: i for i, l in enumerate(labels_rp_n)}
            idx_t = [lt_map[l] for l in labels_t if l in lt_map]
            idx_n = [ln_map[l] for l in labels_n if l in ln_map]
            if len(idx_t) < 2 or len(idx_n) < 2:
                continue
            feats_t = torch.from_numpy(feats_t[idx_t]).float()
            feats_n = torch.from_numpy(feats_n[idx_n]).float()
            labels_t_use = labels_t[[l in lt_map for l in labels_t]]
            labels_n_use = labels_n[[l in ln_map for l in labels_n]]

        elif feature_type in ('cnn_frozen', 'dino'):
            # Check cache first
            frame_t_num = int(Path(mt).stem.replace("man_track", ""))
            frame_n_num = int(Path(mn).stem.replace("man_track", ""))
            prefix = f"f{frame_t_num:06d}"
            cached_t, cached_labels_t = _load_cached_features(cond, exp, frame_t_num, feature_type, labels_t)
            cached_n, cached_labels_n = _load_cached_features(cond, exp, frame_n_num, feature_type, labels_n)

            if cached_t is not None and cached_n is not None:
                feats_t = torch.from_numpy(cached_t).float()
                feats_n = torch.from_numpy(cached_n).float()
                labels_t_use = cached_labels_t
                labels_n_use = cached_labels_n
            else:
                # Extract patches and compute features
                patches_t = extract_patches(imgt, coords_t)
                patches_n = extract_patches(imgn, coords_n)

                if feature_type == 'cnn_frozen':
                    pt_t = torch.from_numpy(patches_t).float().unsqueeze(1).to(device)
                    pn_t = torch.from_numpy(patches_n).float().unsqueeze(1).to(device)
                    with torch.no_grad():
                        feats_t_np = cnn_frozen_model(pt_t).cpu().numpy()
                        feats_n_np = cnn_frozen_model(pn_t).cpu().numpy()
                else:  # dino
                    feats_t_np = compute_dino_embs(patches_t)
                    feats_n_np = compute_dino_embs(patches_n)

                feats_t = torch.from_numpy(feats_t_np).float()
                feats_n = torch.from_numpy(feats_n_np).float()
                labels_t_use = labels_t
                labels_n_use = labels_n

                # Cache
                _save_cached_features(cond, exp, frame_t_num, feature_type, feats_t_np, labels_t)
                _save_cached_features(cond, exp, frame_n_num, feature_type, feats_n_np, labels_n)

        elif feature_type == 'cnn_e2e':
            # Just store patches; model is trained jointly
            patches_t = extract_patches(imgt, coords_t)
            patches_n = extract_patches(imgn, coords_n)
            N1, N2 = len(labels_t), len(labels_n)
            target = torch.zeros(N1, N2, dtype=torch.float32)
            n_pos = 0
            for i, lt in enumerate(labels_t):
                lt_int = int(lt)
                for j, ln in enumerate(labels_n):
                    ln_int = int(ln)
                    if lt_int == ln_int:
                        target[i, j] = 1.0
                        n_pos += 1
                    elif ln_int in tracklets and tracklets[ln_int]['parent'] == lt_int:
                        target[i, j] = 1.0
                        n_pos += 1
            if target.sum() < 1:
                continue
            edge_data.append({
                "patches_t": torch.from_numpy(patches_t).float().unsqueeze(1),  # (N, 1, H, W)
                "patches_n": torch.from_numpy(patches_n).float().unsqueeze(1),
                "target": target,
                "n_pos": n_pos,
                "n_neg": N1 * N2 - n_pos,
                "condition": cond,
                "experiment": exp,
            })
            total += 1
            continue
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        # Build target matrix
        if feature_type != 'cnn_e2e':
            N1, N2 = len(labels_t_use), len(labels_n_use)
            target = torch.zeros(N1, N2, dtype=torch.float32)
            n_pos = 0
            for i, lt in enumerate(labels_t_use):
                lt_int = int(lt)
                for j, ln in enumerate(labels_n_use):
                    ln_int = int(ln)
                    if lt_int == ln_int:
                        target[i, j] = 1.0
                        n_pos += 1
                    elif ln_int in tracklets and tracklets[ln_int]['parent'] == lt_int:
                        target[i, j] = 1.0
                        n_pos += 1

            if target.sum() < 1:
                continue

            edge_data.append({
                "feat_t": feats_t,
                "feat_n": feats_n,
                "target": target,
                "n_pos": n_pos,
                "n_neg": N1 * N2 - n_pos,
                "condition": cond,
                "experiment": exp,
            })
            total += 1

    logger.info(f"  Built {total} frame pairs for {feature_type}")
    return edge_data, cnn_frozen_model


def split_by_condition(edge_data, train_conditions, val_conditions):
    """Split edge data by condition (experimental group)."""
    train_data = [d for d in edge_data if d["condition"] in train_conditions]
    val_data = [d for d in edge_data if d["condition"] in val_conditions]
    return train_data, val_data


def flatten_to_pairs(edge_data):
    """
    Convert list of per-frame-pair dicts into a flat list of datasets,
    one per frame pair (for later concatenation into a single dataset).
    """
    datasets = []
    for item in edge_data:
        if "feat_t" in item:
            ds = EdgePairDataset(item["feat_t"], item["feat_n"], item["target"])
        else:
            # For cnn_e2e, patches are stored
            ds = EdgePairDatasetPatches(item["patches_t"], item["patches_n"],
                                         item.get("target"))
        if len(ds) > 0:
            datasets.append(ds)
    return datasets


def concatenate_datasets(datasets):
    """Concatenate multiple EdgePairDatasets (or EdgePairDatasetPatches) into one."""
    if not datasets:
        return None
    # Use torch.utils.data.ConcatDataset
    return torch.utils.data.ConcatDataset(datasets)


def shuffle_edge_data_features(edge_data_list, seed=42):
    """
    Create a copy of edge_data with feature vectors shuffled independently
    per frame pair to destroy all structure.
    Returns a new list with same structure but shuffled features.
    """
    rng = np.random.RandomState(seed)
    shuffled = []
    for item in edge_data_list:
        item_copy = dict(item)
        if "feat_t" in item:
            n1 = len(item["feat_t"])
            n2 = len(item["feat_n"])
            idx_t = torch.from_numpy(rng.permutation(n1))
            idx_n = torch.from_numpy(rng.permutation(n2))
            item_copy["feat_t"] = item["feat_t"][idx_t].clone()
            item_copy["feat_n"] = item["feat_n"][idx_n].clone()
        elif "patches_t" in item:
            n1 = len(item["patches_t"])
            n2 = len(item["patches_n"])
            idx_t = torch.from_numpy(rng.permutation(n1))
            idx_n = torch.from_numpy(rng.permutation(n2))
            item_copy["patches_t"] = item["patches_t"][idx_t].clone()
            item_copy["patches_n"] = item["patches_n"][idx_n].clone()
        shuffled.append(item_copy)
    return shuffled


def run_cross_validation(frame_pair_datasets, probe_name, feat_dim, args,
                         n_folds=5, is_e2e=False):
    """
    Run K-fold cross-validation on frame-pair-level datasets.

    Each fold: train on K-1 folds, validate on 1 held-out fold.
    Returns dict with mean, std, min, max of metrics across folds.
    """
    from sklearn.model_selection import KFold

    if len(frame_pair_datasets) < n_folds:
        n_folds = len(frame_pair_datasets)
        logger.warning(f"    Reducing folds to {n_folds} (not enough frame pairs)")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    all_fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(frame_pair_datasets)):
        logger.info(f"    Fold {fold_idx + 1}/{n_folds}: "
                    f"{len(train_indices)} train / {len(val_indices)} val frame pairs")

        train_datasets = [frame_pair_datasets[i] for i in train_indices]
        val_datasets = [frame_pair_datasets[i] for i in val_indices]

        train_dataset = concatenate_datasets(train_datasets)
        val_dataset = concatenate_datasets(val_datasets)

        lr = args.lr if probe_name == "Linear" else args.lr / 10
        batch_size = args.batch_size_linear if probe_name == "Linear" else args.batch_size_mlp

        if is_e2e:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=lambda b: (
                    torch.stack([x[0] for x in b]),
                    torch.stack([x[1] for x in b]),
                    torch.stack([x[2] for x in b]),
                )
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda b: (
                    torch.stack([x[0] for x in b]),
                    torch.stack([x[1] for x in b]),
                    torch.stack([x[2] for x in b]),
                )
            )
            use_probe = "linear" if probe_name == "Linear" else "mlp"
            model = CNNProbeE2E(scale='large', out_dim=128, probe_type=use_probe)
            result = train_probe(
                model, train_loader, val_loader,
                epochs=args.epochs, lr=lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=True,
            )
        else:
            train_loader = make_balanced_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
            )

            if probe_name == "Linear":
                probe = LinearProbe(feat_dim)
            else:
                probe = MLPProbe(feat_dim)

            result = train_probe(
                probe, train_loader, val_loader,
                epochs=args.epochs, lr=lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=False,
            )

        all_fold_results.append(result)

    bal_accs = [r["final_bal_acc"] for r in all_fold_results]
    f1s = [r["final_f1"] for r in all_fold_results]

    return {
        "fold_results": all_fold_results,
        "bal_acc_mean": float(np.mean(bal_accs)),
        "bal_acc_std": float(np.std(bal_accs)),
        "bal_acc_min": float(np.min(bal_accs)),
        "bal_acc_max": float(np.max(bal_accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_min": float(np.min(f1s)),
        "f1_max": float(np.max(f1s)),
        "is_cv": True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-feature evaluation
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_CONFIGS = {
    'rp': {
        'feat_dim': 7,
        'display': 'Regionprops 7D',
        'needs_patches': False,
    },
    'cnn_frozen': {
        'feat_dim': 128,
        'display': 'CNN NT-Xent (frozen)',
        'needs_patches': False,
    },
    'cnn_e2e': {
        'feat_dim': 128,
        'display': 'CNN end-to-end',
        'needs_patches': True,
    },
    'dino': {
        'feat_dim': 384,
        'display': 'DINOv2 (frozen)',
        'needs_patches': False,
    },
}

# Train conditions (use majority for training)
TRAIN_CONDITIONS = ["rpsM", "recA", "pheA", "metA"]
VAL_CONDITIONS = ["cib", "trpL"]


def evaluate_feature(feature_type, all_pairs, args, checkpoint_path):
    """Run full evaluation for a single feature type. Returns dict of results."""
    cfg = FEATURE_CONFIGS[feature_type]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Feature: {cfg['display']} ({cfg['feat_dim']}D)")
    logger.info(f"{'=' * 60}")

    # ── Build frame-pair data ────────────────────────────────────────────
    t0 = time.time()
    edge_data, cnn_model = build_frame_pairs(
        all_pairs, args.max_pairs, feature_type, checkpoint_path
    )
    logger.info(f"  Built {len(edge_data)} frame pairs in {time.time() - t0:.1f}s")

    if len(edge_data) < 2:
        logger.warning(f"  Not enough data ({len(edge_data)}), skipping")
        return None

    # ═══════════════════════════════════════════════════════════════════════
    #  CV MODE (overrides train/val split)
    # ═══════════════════════════════════════════════════════════════════════
    if args.cv_folds > 0:
        logger.info(f"  Running {args.cv_folds}-fold CV on {len(edge_data)} frame pairs (all conditions)")
        all_datasets = flatten_to_pairs(edge_data)
        logger.info(f"  Total frame-pair datasets: {len(all_datasets)}")
        if not all_datasets:
            logger.warning("  No pairs after flattening, skipping")
            return None

        results = {}

        def _run_cv(probe_name, feat_dim, is_e2e):
            return run_cross_validation(
                all_datasets, probe_name, feat_dim, args,
                n_folds=args.cv_folds, is_e2e=is_e2e,
            )

        # Linear probe
        if args.probe in ("linear", "both"):
            t1 = time.time()
            cv_result = _run_cv("Linear", cfg['feat_dim'],
                                is_e2e=(feature_type == 'cnn_e2e'))
            elapsed = time.time() - t1
            if cv_result:
                logger.info(f"    Linear CV: bal_acc={cv_result['bal_acc_mean']:.4f}\u00b1{cv_result['bal_acc_std']:.4f}, "
                            f"f1={cv_result['f1_mean']:.4f}\u00b1{cv_result['f1_std']:.4f} ({elapsed:.1f}s)")
                results["linear"] = cv_result

        # MLP probe
        if args.probe in ("mlp", "both"):
            t1 = time.time()
            cv_result = _run_cv("MLP", cfg['feat_dim'],
                                is_e2e=(feature_type == 'cnn_e2e'))
            elapsed = time.time() - t1
            if cv_result:
                logger.info(f"    MLP CV: bal_acc={cv_result['bal_acc_mean']:.4f}\u00b1{cv_result['bal_acc_std']:.4f}, "
                            f"f1={cv_result['f1_mean']:.4f}\u00b1{cv_result['f1_std']:.4f} ({elapsed:.1f}s)")
                results["mlp"] = cv_result

        # Shuffle baseline
        if args.shuffle_baseline:
            logger.info(f"  Running shuffled feature baseline ({args.cv_folds}-fold CV)...")
            shuffled_edge_data = shuffle_edge_data_features(edge_data, seed=SEED)
            shuffled_datasets = flatten_to_pairs(shuffled_edge_data)
            if shuffled_datasets:
                if args.probe in ("linear", "both"):
                    t1 = time.time()
                    shuf_result = run_cross_validation(
                        shuffled_datasets, "Linear", cfg['feat_dim'], args,
                        n_folds=args.cv_folds, is_e2e=(feature_type == 'cnn_e2e'),
                    )
                    elapsed = time.time() - t1
                    if shuf_result:
                        logger.info(f"    Shuffled Linear CV: bal_acc={shuf_result['bal_acc_mean']:.4f}\u00b1{shuf_result['bal_acc_std']:.4f} "
                                    f"({elapsed:.1f}s)")
                        results["linear_shuffled"] = shuf_result

                if args.probe in ("mlp", "both"):
                    t1 = time.time()
                    shuf_result = run_cross_validation(
                        shuffled_datasets, "MLP", cfg['feat_dim'], args,
                        n_folds=args.cv_folds, is_e2e=(feature_type == 'cnn_e2e'),
                    )
                    elapsed = time.time() - t1
                    if shuf_result:
                        logger.info(f"    Shuffled MLP CV: bal_acc={shuf_result['bal_acc_mean']:.4f}\u00b1{shuf_result['bal_acc_std']:.4f} "
                                    f"({elapsed:.1f}s)")
                        results["mlp_shuffled"] = shuf_result
            else:
                logger.warning("  No shuffled datasets, skipping shuffle baseline")

        return results

    # ── Split by condition (non-CV mode) ──────────────────────────────────
    train_data, val_data = split_by_condition(
        edge_data, TRAIN_CONDITIONS, VAL_CONDITIONS
    )
    logger.info(f"  Train conditions: {TRAIN_CONDITIONS} -> {len(train_data)} pairs")
    logger.info(f"  Val conditions:   {VAL_CONDITIONS} -> {len(val_data)} pairs")

    if len(val_data) < 1:
        train_frac = 1.0 - args.val_frac
        logger.warning(f"  Val empty ({VAL_CONDITIONS} have 0 pairs), falling back to random "
                       f"{train_frac:.0%}/{args.val_frac:.0%} split")
        import random
        random.shuffle(train_data)
        split = max(1, int(train_frac * len(train_data)))
        val_data = train_data[split:]
        train_data = train_data[:split]
        logger.info(f"  Random split: {len(train_data)} train, {len(val_data)} val")
    if len(train_data) < 1 or len(val_data) < 1:
        logger.warning("  Need at least 1 train and 1 val pair, skipping")
        return None

    # ── Flatten to pair-level datasets ────────────────────────────────────
    train_datasets = flatten_to_pairs(train_data)
    val_datasets = flatten_to_pairs(val_data)

    if not train_datasets or not val_datasets:
        logger.warning("  No pairs after flattening, skipping")
        return None

    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)

    n_train_pos = sum(int(d.labels.sum()) for d in train_datasets)
    n_train_neg = sum(len(d) - int(d.labels.sum()) for d in train_datasets)
    logger.info(f"  Train pairs: {len(train_dataset)} ({n_train_pos} pos / {n_train_neg} neg)")
    n_val_pos = sum(int(d.labels.sum()) for d in val_datasets)
    n_val_neg = sum(len(d) - int(d.labels.sum()) for d in val_datasets)
    logger.info(f"  Val pairs:   {len(val_dataset)} ({n_val_pos} pos / {n_val_neg} neg)")

    results = {}

    # ── Helper to run one probe type ──────────────────────────────────────
    def _run_probe(probe_name, probe_lr, batch_size):
        logger.info(f"  Training {probe_name} probe (lr={probe_lr}, bs={batch_size})...")

        if feature_type == 'cnn_e2e':
            # End-to-end: patches stored in datasets, build loaders specially
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=lambda b: (
                    torch.stack([x[0] for x in b]),
                    torch.stack([x[1] for x in b]),
                    torch.stack([x[2] for x in b]),
                )
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda b: (
                    torch.stack([x[0] for x in b]),
                    torch.stack([x[1] for x in b]),
                    torch.stack([x[2] for x in b]),
                )
            )

            use_probe = "linear" if probe_name == "Linear" else "mlp"
            model = CNNProbeE2E(scale='large', out_dim=128, probe_type=use_probe)
            result = train_probe(
                model, train_loader, val_loader,
                epochs=args.epochs, lr=probe_lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=True,
            )
        else:
            # Feature-based: use balanced sampling
            train_loader = make_balanced_dataloader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
            )

            feat_dim = cfg['feat_dim']
            if probe_name == "Linear":
                probe = LinearProbe(feat_dim)
            else:
                probe = MLPProbe(feat_dim)

            result = train_probe(
                probe, train_loader, val_loader,
                epochs=args.epochs, lr=probe_lr,
                patience=args.patience, eval_every=args.eval_every,
                is_e2e=False,
            )

        return result

    # ── Linear probe ─────────────────────────────────────────────────────
    if args.probe in ("linear", "both"):
        t1 = time.time()
        result = _run_probe("Linear", args.lr, args.batch_size_linear)
        elapsed = time.time() - t1
        if result:
            logger.info(f"    Linear: bal_acc={result['final_bal_acc']:.4f}, "
                        f"f1={result['final_f1']:.4f} ({elapsed:.1f}s)")
            results["linear"] = result

    # ── MLP probe ────────────────────────────────────────────────────────
    if args.probe in ("mlp", "both"):
        t1 = time.time()
        result = _run_probe("MLP", args.lr / 10, args.batch_size_mlp)
        elapsed = time.time() - t1
        if result:
            logger.info(f"    MLP: bal_acc={result['final_bal_acc']:.4f}, "
                        f"f1={result['final_f1']:.4f} ({elapsed:.1f}s)")
            results["mlp"] = result

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Results display
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(all_results):
    """Print formatted results table. Handles both CV and non-CV results."""
    # Auto-detect CV mode
    is_cv_mode = any(
        r is not None
        and any(isinstance(m, dict) and m.get("is_cv", False) for m in r.values())
        for r in all_results.values()
    )

    print()
    print("=" * 120)
    print("Unified Edge Probing Benchmark \u2014 Results")
    print("=" * 120)

    if is_cv_mode:
        header = (f"{'Feature':<28} {'Probe':<8} {'BalAcc (mean±std)':<22} "
                  f"{'F1 (mean±std)':<22} {'Status':<10}")
    else:
        header = (f"{'Feature':<25} {'Probe':<8} {'BalAcc':<10} {'F1':<10} "
                  f"{'Precision':<10} {'Recall':<10} {'Status':<10}")
    print(header)
    print("-" * 120)

    rows = []
    has_shuffled = False

    for fkey, cfg in FEATURE_CONFIGS.items():
        if fkey not in all_results or all_results[fkey] is None:
            if is_cv_mode:
                print(f"{cfg['display']:<28} {'\u2014':<8} {'\u2014':<22} {'\u2014':<22} {'SKIP':<10}")
            else:
                print(f"{cfg['display']:<25} {'\u2014':<8} {'\u2014':<10} {'\u2014':<10} "
                      f"{'\u2014':<10} {'\u2014':<10} {'SKIP':<10}")
            continue

        r = all_results[fkey]
        for ptype in ["linear", "mlp"]:
            if ptype not in r or r[ptype] is None:
                continue
            m = r[ptype]

            if m.get("is_cv", False):
                print(f"{cfg['display']:<28} {ptype:<8} "
                      f"{m['bal_acc_mean']:.4f}\u00b1{m['bal_acc_std']:<.4f}        "
                      f"{m['f1_mean']:.4f}\u00b1{m['f1_std']:<.4f}        "
                      f"{'OK':<10}")
                rows.append({
                    "feature": cfg['display'],
                    "probe": ptype,
                    "bal_acc": m['bal_acc_mean'],
                    "bal_acc_std": m['bal_acc_std'],
                    "f1": m['f1_mean'],
                    "f1_std": m['f1_std'],
                    "is_cv": True,
                })
            else:
                print(f"{cfg['display']:<25} {ptype:<8} "
                      f"{m['final_bal_acc']:<10.4f} {m['final_f1']:<10.4f} "
                      f"{m['final_precision']:<10.4f} {m['final_recall']:<10.4f} "
                      f"{'OK':<10}")
                rows.append({
                    "feature": cfg['display'],
                    "probe": ptype,
                    "bal_acc": m['final_bal_acc'],
                    "f1": m['final_f1'],
                    "precision": m['final_precision'],
                    "recall": m['final_recall'],
                })

        # Check for shuffled results
        for ptype in ["linear_shuffled", "mlp_shuffled"]:
            if ptype in r and r[ptype] is not None:
                has_shuffled = True

    # Print shuffled baseline section
    if has_shuffled:
        print("--- shuffled baseline ---")
        for fkey, cfg in FEATURE_CONFIGS.items():
            if fkey not in all_results or all_results[fkey] is None:
                continue
            r = all_results[fkey]
            for ptype, label in [("linear_shuffled", "linear"), ("mlp_shuffled", "mlp")]:
                if ptype not in r or r[ptype] is None:
                    continue
                m = r[ptype]
                display_name = f"{cfg['display']} (shuffled)"
                print(f"{display_name:<28} {label:<8} "
                      f"{m['bal_acc_mean']:.4f}\u00b1{m['bal_acc_std']:<.4f}        "
                      f"{m['f1_mean']:.4f}\u00b1{m['f1_std']:<.4f}        "
                      f"{'SHUF':<10}")

    print("-" * 120)
    print()

    # Best performer
    if rows:
        if is_cv_mode:
            best = max(rows, key=lambda r: r["bal_acc"])
            print(f"Best performer: {best['feature']} + {best['probe']} "
                  f"(bal_acc={best['bal_acc']:.4f}\u00b1{best['bal_acc_std']:.4f}, "
                  f"f1={best['f1']:.4f}\u00b1{best['f1_std']:.4f})")
        else:
            best = max(rows, key=lambda r: r["bal_acc"])
            print(f"Best performer: {best['feature']} + {best['probe']} "
                  f"(bal_acc={best['bal_acc']:.4f}, f1={best['f1']:.4f})")
        print()

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_FEATURES = list(FEATURE_CONFIGS.keys())
FEATURE_ALIASES = {
    "all": AVAILABLE_FEATURES,
    "shallow": ["rp"],
    "deep": ["cnn_frozen", "cnn_e2e", "dino"],
}


def parse_features(features_arg):
    """Parse --features argument into list of feature keys."""
    if features_arg in FEATURE_ALIASES:
        return list(FEATURE_ALIASES[features_arg])
    keys = [f.strip() for f in features_arg.split(",")]
    for k in keys:
        if k not in AVAILABLE_FEATURES:
            logger.warning(f"Unknown feature: {k}. Available: {AVAILABLE_FEATURES}")
    return [k for k in keys if k in AVAILABLE_FEATURES]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Unified edge probing benchmark for cell tracking"
    )
    p.add_argument("--data-root", default="../../data/vanvliet",
                   help="Path to vanvliet data root")
    p.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL",
                   help="Comma-separated conditions to scan")
    p.add_argument("--train-conditions", default="rpsM,recA,pheA,metA",
                   help="Conditions for training")
    p.add_argument("--val-conditions", default="cib,trpL",
                   help="Conditions for validation")
    p.add_argument("--features", default="rp",
                   help=f"Features to test. Choices: {AVAILABLE_FEATURES}, "
                        f"or aliases: all, shallow, deep")
    p.add_argument("--probe", default="both", choices=["linear", "mlp", "both"],
                   help="Probe architecture to use")
    p.add_argument("--max-pairs", type=int, default=30,
                   help="Max consecutive frame pairs per condition")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per probe")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for linear probe (MLP uses lr/10)")
    p.add_argument("--batch-size-linear", type=int, default=256,
                   help="Batch size for linear probe")
    p.add_argument("--batch-size-mlp", type=int, default=128,
                   help="Batch size for MLP probe")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Evaluate every N epochs")
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(
                       os.path.dirname(__file__),
                       "../cnn_encoder/probe/cnn_ntxent_large.pt"
                   ),
                   help="Path to NT-Xent checkpoint for cnn_frozen")
    p.add_argument("--output", type=str, default="",
                   help="Path to save JSON results (default: no save)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--no-cache", action="store_true",
                   help="Skip loading/saving feature cache")
    p.add_argument("--cv-folds", type=int, default=0,
                   help="Number of CV folds (0 = use original train/val split)")
    p.add_argument("--shuffle-baseline", action="store_true",
                   help="Also run on shuffled features as overfitting baseline")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Validation fraction for non-CV mode (used in random split fallback)")
    return p.parse_args(argv)


def main():
    global SEED, TRAIN_CONDITIONS, VAL_CONDITIONS, _DINO_MODEL
    args = parse_args()
    SEED = args.seed
    TRAIN_CONDITIONS = [c.strip() for c in args.train_conditions.split(",")]
    VAL_CONDITIONS = [c.strip() for c in args.val_conditions.split(",")]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    conditions = [c.strip() for c in args.conditions.split(",")]
    features_to_run = parse_features(args.features)

    logger.info("=" * 60)
    logger.info("Unified Edge Probing Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Data root:      {args.data_root}")
    logger.info(f"  Conditions:     {conditions}")
    logger.info(f"  Train:          {TRAIN_CONDITIONS}")
    logger.info(f"  Val:            {VAL_CONDITIONS}")
    logger.info(f"  Features:       {features_to_run}")
    logger.info(f"  Probe:          {args.probe}")
    logger.info(f"  Epochs:         {args.epochs}")
    logger.info(f"  LR:             {args.lr}")
    logger.info(f"  Batch linear:   {args.batch_size_linear}")
    logger.info(f"  Batch MLP:      {args.batch_size_mlp}")
    logger.info(f"  Device:         {device}")
    logger.info(f"  Cache dir:      {CACHE_DIR}")
    logger.info("=" * 60)

    # ── Scan pairs ────────────────────────────────────────────────────────
    logger.info("Scanning consecutive frame pairs...")
    all_pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Found {len(all_pairs)} consecutive frame pairs")

    # Log condition breakdown
    from collections import Counter
    cond_counts = Counter(p[5] for p in all_pairs)
    for cond, count in sorted(cond_counts.items()):
        logger.info(f"  {cond}: {count} pairs")

    if len(all_pairs) < 3:
        logger.error("Need at least 3 consecutive frame pairs. Check --data-root.")
        sys.exit(1)

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_path = None

    # ── Run each feature type ─────────────────────────────────────────────
    all_results = {}
    for fkey in features_to_run:
        try:
            result = evaluate_feature(fkey, all_pairs, args, checkpoint_path)
            all_results[fkey] = result
        except Exception as e:
            logger.error(f"Error running {fkey}: {e}", exc_info=True)
            all_results[fkey] = None

        # Free DINO memory between runs
        if fkey == 'dino' and _DINO_MODEL is not None:
            logger.info("  Clearing DINOv2 from GPU...")
            _DINO_MODEL.cpu()
            del _DINO_MODEL
            _DINO_MODEL = None
            torch.cuda.empty_cache()

    # ── Print results ─────────────────────────────────────────────────────
    rows = print_results_table(all_results)

    # ── Save results ──────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for fkey, result in all_results.items():
            if result is None:
                serializable[fkey] = None
            else:
                serializable[fkey] = {}
                for ptype, metrics in result.items():
                    if ptype in ("linear", "mlp"):
                        serializable[fkey][ptype] = {
                            k: v for k, v in metrics.items() if k != "history"
                        }

        with open(output_path, "w") as f:
            json.dump({
                "args": vars(args),
                "results": serializable,
                "rows": rows,
            }, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
