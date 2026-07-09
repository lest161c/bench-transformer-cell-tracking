#!/usr/bin/env python3
"""Edge probing benchmark: which features best predict cell-cell edges?

Tests 3 feature sets × 2 edge types × 2 probe architectures = 12 conditions.

Feature sets:
  1. Regionprops 7D only (mask-based shape/intensity descriptors)
  2. Hu moments 20D only (patch-based moment descriptors from 32x32 crops)
  3. Both combined 27D

Edge types:
  - All edges: same-cell persistence + parent→child division edges
  - Division edges only: only parent→child divisions (harder task)

Probes:
  - Linear: concat(f₁, f₂) -> Linear(2D, 1)
  - MLP:    concat(f₁, f₂) -> Linear(2D, 128) -> ReLU -> Linear(128, 1)

Usage:
  cd benchmark_ssl/probe
  ../.venv/bin/python edge_probe.py --max-pairs 5 --steps 20    # quick smoke test
  ../.venv/bin/python edge_probe.py --max-pairs 30 --steps 200  # full run
"""

import argparse
import logging
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
from skimage.measure import moments_hu, moments, moments_normalized, moments_central
from tifffile import imread

# Suppress sklearn warnings about single-label predictions in imbalanced tasks
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="A single label was found")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edge_probe")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PATCH_SIZE_32 = 32  # patch size for Hu moments


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Load a single mask+image frame.

    Returns (coords, labels, img) or None if < 2 cells.
    Mirrors end_to_end.py's load_frame for compatibility.
    """
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
    """Parse man_track.txt into a dict: label -> {t1, t2, parent}.

    Format: label start_frame end_frame parent_id
      parent_id == 0  → founder cell (no division parent)
      parent_id == X  → this tracklet is a daughter of tracklet X
    """
    df = pd.read_csv(
        man_track_path, delimiter=' ', header=None,
        names=['label', 't1', 't2', 'parent']
    )
    tracklets = {}
    for _, row in df.iterrows():
        label = int(row['label'])
        tracklets[label] = {
            't1': int(row['t1']),
            't2': int(row['t2']),
            'parent': int(row['parent']),
        }
    return tracklets


def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame PAIRS from the same experiment.

    Returns list of (mask_t, mask_n, img_t, img_n, man_track_path).
    """
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
                logger.warning(f"  No man_track.txt in {tra_dir}, skipping")
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
                        ))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
#  Patch extraction (32x32 for Hu moments)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_patches_32(img, centroids):
    """Extract 32×32 patches centered on each centroid from a raw image.

    Handles edge padding via reflection.
    """
    h, w = img.shape[-2:]
    half = PATCH_SIZE_32 // 2
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
        if crop.shape != (PATCH_SIZE_32, PATCH_SIZE_32):
            crop = np.pad(
                crop,
                tuple(
                    (0, max(0, t)) for t in [
                        PATCH_SIZE_32 - s for s in crop.shape
                    ]
                ),
                mode="reflect",
            )[:PATCH_SIZE_32, :PATCH_SIZE_32]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, PATCH_SIZE_32, PATCH_SIZE_32), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extractors
# ═══════════════════════════════════════════════════════════════════════════════

def extract_regionprops_7d(mask, img):
    """7D regionprops: area, eccentricity, perimeter, solidity,
    extent, orientation, mean_intensity.

    Returns (coords, labels, features) or (None, None, None).
    """
    props_list = sk_regionprops(mask, intensity_image=img)
    if len(props_list) == 0:
        return None, None, None

    N = len(props_list)
    feats = np.zeros((N, 7), dtype=np.float32)
    coords = np.zeros((N, 2), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int32)

    for i, r in enumerate(props_list):
        feats[i, 0] = r.area
        feats[i, 1] = r.eccentricity
        feats[i, 2] = r.perimeter
        feats[i, 3] = r.solidity if r.solidity is not None else 1.0
        feats[i, 4] = r.extent if r.extent is not None else 1.0
        feats[i, 5] = r.orientation if r.orientation is not None else 0.0
        feats[i, 6] = r.intensity_mean if r.intensity_mean is not None else 0.0
        coords[i] = r.centroid
        labels[i] = r.label

    return coords, labels, feats


def extract_hu_moments_20d(img, centroids):
    """20D patch-based moment features: 7 Hu moments + 13 raw/normalized moments.

    Each feature is extracted from a 32×32 patch around the cell centroid
    on the raw (intensity) image.

    Composition:
      [0:7]   7 Hu moments (log-scaled):  -sign(m)*log10(|m|+1e-10)
      [7:20]  13 raw moments from moments(order=3),
              excluding M00, M01, M10 (total intensity and centroid,
              which are redundant with regionprops or uninformative
              for centered patches).

    Returns (N, 20) float32 array.
    """
    patches = extract_patches_32(img, centroids)
    N = len(patches)
    if N == 0:
        return np.zeros((0, 20), dtype=np.float32)

    feats = np.zeros((N, 20), dtype=np.float32)

    for i in range(N):
        p = patches[i].astype(np.float64)

        # --- 7 Hu moments (log-scaled) ---
        hu = moments_hu(p)
        feats[i, :7] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        # --- 13 raw moments up to order 3 ---
        # moments(p, 3) returns (4, 4) with indices:
        #   row 0: M00, M01, M02, M03
        #   row 1: M10, M11, M12, M13
        #   row 2: M20, M21, M22, M23
        #   row 3: M30, M31, M32, M33
        m_raw = moments(p, order=3)  # (4, 4)
        # Flatten row-major, skip M00(0), M01(1), M10(4)
        # Keep indices: all except 0,1,4
        keep_mask = np.ones(16, dtype=bool)
        keep_mask[[0, 1, 4]] = False
        feats[i, 7:20] = m_raw.flatten()[keep_mask]

    return feats


def extract_combined_27d(mask, img, centroids):
    """27D = 7D regionprops + 20D Hu/patch moments."""
    _, _, rp_feats = extract_regionprops_7d(mask, img)
    hu_feats = extract_hu_moments_20d(img, centroids)
    if rp_feats is None or len(hu_feats) == 0:
        return None, None, None
    combined = np.concatenate([rp_feats, hu_feats], axis=-1)
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  Edge target builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_targets(labels_t, labels_n, tracklets):
    """Build edge targets for a frame pair.

    Returns:
      target_all: (N1, N2) — 1 for same-cell OR parent→child, 0 otherwise
      target_div: (N1, N2) — 1 for parent→child only, 0 otherwise
      n_same: number of same-cell positive pairs
      n_div:  number of division positive pairs
    """
    N1, N2 = len(labels_t), len(labels_n)
    target_all = torch.zeros(N1, N2, dtype=torch.float32)
    target_div = torch.zeros(N1, N2, dtype=torch.float32)
    n_same, n_div = 0, 0

    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_n):
            if lt == ln:
                target_all[i, j] = 1.0
                n_same += 1
            elif ln in tracklets and tracklets[ln]['parent'] == lt:
                target_all[i, j] = 1.0
                target_div[i, j] = 1.0
                n_div += 1

    return target_all, target_div, n_same, n_div


# ═══════════════════════════════════════════════════════════════════════════════
#  Edge data builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_edge_data(pairs, max_pairs=30):
    """Build edge prediction data from consecutive frame pairs.

    Uses ALL cells in each frame (not just shared labels), so both
    same-cell persistence edges AND parent→child division edges are captured.

    For each pair:
      - Extract 7D regionprops and 20D Hu moments for both frames
      - Build all-edges and division-only target matrices
      - Track positive pair counts

    Returns list of dicts:
      feat_7d_t, feat_7d_n     : (N, 7)  regionprops
      feat_20d_t, feat_20d_n   : (N, 20) Hu/patch moments
      target_all               : (N1, N2) all positive edges
      target_div               : (N1, N2) division-only edges
      n_same, n_div, n_neg_all, n_neg_div
    """
    edge_data = []
    total_processed = 0

    for mt, mn, it_, in_, man_txt in pairs[:max_pairs]:
        # Load frames
        rt = load_frame(mt, it_)
        rn = load_frame(mn, in_)
        if rt is None or rn is None:
            continue
        coords_t, labels_t, imgt, mask_t = rt
        coords_n, labels_n, imgn, mask_n = rn

        # Minimum cells per frame
        if len(labels_t) < 3 or len(labels_n) < 3:
            continue

        # Load tracklet database for this experiment
        tracklets = load_tracklets(man_txt)

        # --- Extract 7D regionprops for ALL cells ---
        _, labels_rp_t, feats_7d_t = extract_regionprops_7d(mask_t, imgt)
        _, labels_rp_n, feats_7d_n = extract_regionprops_7d(mask_n, imgn)
        if feats_7d_t is None or feats_7d_n is None:
            continue

        # Map label -> index for regionprops (sorted by label in both frames)
        label_to_idx_t = {l: i for i, l in enumerate(labels_rp_t)}
        label_to_idx_n = {l: i for i, l in enumerate(labels_rp_n)}

        # Get features for ALL cells in the order of labels_t / labels_n
        idx_7d_t = [label_to_idx_t[l] for l in labels_t if l in label_to_idx_t]
        idx_7d_n = [label_to_idx_n[l] for l in labels_n if l in label_to_idx_n]
        if len(idx_7d_t) < 3 or len(idx_7d_n) < 3:
            continue

        feats_7d_t_s = feats_7d_t[idx_7d_t]
        feats_7d_n_s = feats_7d_n[idx_7d_n]

        # --- Extract 20D Hu moments for ALL cells ---
        # (coords_t/labels_t are aligned from load_frame's regionprops_table)
        feats_20d_t = extract_hu_moments_20d(imgt, coords_t)
        feats_20d_n = extract_hu_moments_20d(imgn, coords_n)
        if len(feats_20d_t) < 3 or len(feats_20d_n) < 3:
            continue

        # --- Build targets using ALL pairs (N1 × N2) ---
        target_all, target_div, n_same, n_div = build_targets(
            labels_t, labels_n, tracklets
        )

        N1, N2 = target_all.shape
        n_neg_all = N1 * N2 - target_all.sum().item()
        n_neg_div = N1 * N2 - target_div.sum().item()

        # Need at least one positive in all-edges target
        if target_all.sum() < 1:
            continue

        edge_data.append({
            "feat_7d_t": torch.from_numpy(feats_7d_t_s).float(),
            "feat_7d_n": torch.from_numpy(feats_7d_n_s).float(),
            "feat_20d_t": torch.from_numpy(feats_20d_t).float(),
            "feat_20d_n": torch.from_numpy(feats_20d_n).float(),
            "target_all": target_all,
            "target_div": target_div,
            "n_same": n_same,
            "n_div": n_div,
            "n_neg_all": n_neg_all,
            "n_neg_div": n_neg_div,
            "N1": N1,
            "N2": N2,
        })
        total_processed += 1

    logger.info(f"  Built {total_processed} frame pairs")
    logger.info(f"    Total division edges across all pairs: "
                f"{sum(d['n_div'] for d in edge_data)}")
    return edge_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Probe models
# ═══════════════════════════════════════════════════════════════════════════════

class LinearEdgeProbe(nn.Module):
    """Linear probe: concat(feat_t[i], feat_n[j]) → score."""

    def __init__(self, feat_dim):
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, 1)

    def forward(self, emb_t, emb_n):
        """emb_t: (N1, D), emb_n: (N2, D) → scores: (N1, N2)"""
        N1, D = emb_t.shape
        N2 = emb_n.shape[0]
        emb_t_exp = emb_t.unsqueeze(1).expand(-1, N2, -1)   # (N1, N2, D)
        emb_n_exp = emb_n.unsqueeze(0).expand(N1, -1, -1)   # (N1, N2, D)
        pairs = torch.cat([emb_t_exp, emb_n_exp], dim=-1)   # (N1, N2, 2D)
        return self.fc(pairs.view(-1, 2 * D)).view(N1, N2)  # (N1, N2)


class MLPEdgeProbe(nn.Module):
    """2-layer MLP probe: concat(feat_t[i], feat_n[j]) → hidden → score."""

    def __init__(self, feat_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, emb_t, emb_n):
        """emb_t: (N1, D), emb_n: (N2, D) → scores: (N1, N2)"""
        N1, D = emb_t.shape
        N2 = emb_n.shape[0]
        emb_t_exp = emb_t.unsqueeze(1).expand(-1, N2, -1)
        emb_n_exp = emb_n.unsqueeze(0).expand(N1, -1, -1)
        pairs = torch.cat([emb_t_exp, emb_n_exp], dim=-1)
        return self.net(pairs.view(-1, 2 * D)).view(N1, N2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores, target):
    """Compute balanced accuracy, F1, precision, recall on CPU.

    scores: logits (NOT sigmoided)
    target: binary targets
    """
    pred = (scores > 0.0).float()  # logits > 0 => positive class
    target_np = target.cpu().numpy().ravel()
    pred_np = pred.cpu().numpy().ravel()

    bal_acc = balanced_accuracy_score(target_np, pred_np)
    f1 = f1_score(target_np, pred_np, zero_division=0)
    prec = precision_score(target_np, pred_np, zero_division=0)
    rec = recall_score(target_np, pred_np, zero_division=0)
    return bal_acc, f1, prec, rec


def train_probe(probe, train_data, val_data, edge_key, steps=200, lr=1e-3,
                patience=20, eval_every=20):
    """Train an edge probe.

    Args:
        probe: nn.Module (LinearEdgeProbe or MLPEdgeProbe)
        train_data: list of dicts with 'feat_t', 'feat_n', edge_key target
        val_data: same
        edge_key: 'target_all' or 'target_div' — which target to use
        steps: training steps
        lr: learning rate
        patience: early stopping patience
        eval_every: evaluate every N steps

    Returns dict of final metrics.
    """
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_bal_acc = 0.0
    best_state = None
    patience_counter = 0

    for step in range(steps):
        probe.train()
        train_losses = []
        for item in train_data:
            feat_t = item["feat_t"].to(device)
            feat_n = item["feat_n"].to(device)
            target = item[edge_key].to(device)

            scores = probe(feat_t, feat_n)
            loss = criterion(scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate periodically
        if step % eval_every == 0 or step == steps - 1:
            probe.eval()
            val_losses = []
            val_metrics_list = []
            with torch.no_grad():
                for item in val_data:
                    feat_t = item["feat_t"].to(device)
                    feat_n = item["feat_n"].to(device)
                    target = item[edge_key].to(device)
                    scores = probe(feat_t, feat_n)
                    loss = criterion(scores, target)
                    val_losses.append(loss.item())
                    bal_acc, f1, prec, rec = compute_metrics(scores, target)
                    val_metrics_list.append({
                        "bal_acc": bal_acc, "f1": f1,
                        "precision": prec, "recall": rec,
                    })

            avg_val_bal_acc = float(np.mean([m["bal_acc"] for m in val_metrics_list]))

            # Early stopping
            if avg_val_bal_acc > best_val_bal_acc:
                best_val_bal_acc = avg_val_bal_acc
                patience_counter = 0
                best_state = probe.state_dict()
            else:
                patience_counter += eval_every
                if patience_counter >= patience:
                    logger.info(f"      Early stopping at step {step}")
                    break

    # Restore best state
    if best_state is not None:
        probe.load_state_dict(best_state)

    # Final evaluation
    probe.eval()
    final_metrics_list = []
    with torch.no_grad():
        for item in val_data:
            feat_t = item["feat_t"].to(device)
            feat_n = item["feat_n"].to(device)
            target = item[edge_key].to(device)
            scores = probe(feat_t, feat_n)
            bal_acc, f1, prec, rec = compute_metrics(scores, target)
            final_metrics_list.append({
                "bal_acc": bal_acc, "f1": f1,
                "precision": prec, "recall": rec,
            })

    result = {
        "final_bal_acc": float(np.mean([m["bal_acc"] for m in final_metrics_list])),
        "final_f1": float(np.mean([m["f1"] for m in final_metrics_list])),
        "final_precision": float(np.mean([m["precision"] for m in final_metrics_list])),
        "final_recall": float(np.mean([m["recall"] for m in final_metrics_list])),
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation for one (feature_set, edge_type) combination
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_combo(edge_data, feat_key, edge_key, probe_cls, probe_lr,
                   steps=200, patience=20):
    """Train and evaluate a single (features, edge_type, probe) combination.

    Args:
        edge_data: list of dicts from build_edge_data
        feat_key: 'feat_7d' or 'feat_20d' or 'feat_27d'
        edge_key: 'target_all' or 'target_div'
        probe_cls: LinearEdgeProbe or MLPEdgeProbe
        probe_lr: learning rate for the probe
        steps: training steps
        patience: early stopping patience

    Returns dict of metrics.
    """
    # Build feature-specific data
    if feat_key == "feat_7d":
        feat_dim = 7
    elif feat_key == "feat_20d":
        feat_dim = 20
    elif feat_key == "feat_27d":
        feat_dim = 27
    else:
        raise ValueError(f"Unknown feat_key: {feat_key}")

    feat_data = []
    for item in edge_data:
        # Build combined features if needed
        if feat_key == "feat_27d":
            feat_t = torch.cat([item["feat_7d_t"], item["feat_20d_t"]], dim=-1)
            feat_n = torch.cat([item["feat_7d_n"], item["feat_20d_n"]], dim=-1)
        else:
            feat_t = item[f"{feat_key}_t"]
            feat_n = item[f"{feat_key}_n"]

        feat_data.append({
            "feat_t": feat_t,
            "feat_n": feat_n,
            "target_all": item["target_all"],
            "target_div": item["target_div"],
            "N1": item["N1"],
            "N2": item["N2"],
        })

    # Split train/val (80/20)
    np.random.seed(SEED)
    idx = np.random.permutation(len(feat_data))
    n_val = max(1, int(len(feat_data) * 0.2))
    train_data = [feat_data[i] for i in idx[:-n_val]]
    val_data = [feat_data[i] for i in idx[-n_val:]]

    logger.info(
        f"    Train: {len(train_data)} pairs, Val: {len(val_data)} pairs"
    )

    # Check that there are at least some positives in the training data
    total_pos = sum(item[edge_key].sum().item() for item in train_data)
    total_pairs = sum(item["N1"] * item["N2"] for item in train_data)
    pos_ratio = total_pos / max(1, total_pairs)
    logger.info(
        f"      Pos ratio: {total_pos}/{total_pairs} = {pos_ratio:.6f}"
        f"  (train pairs: {len(train_data)}, val: {len(val_data)})"
    )
    if total_pos < 1:
        logger.warning("      No positive edges in training data, skipping")
        return None

    # Create probe and train
    probe = probe_cls(feat_dim)
    result = train_probe(
        probe, train_data, val_data, edge_key,
        steps=steps, lr=probe_lr, patience=patience, eval_every=20,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Main benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(edge_data, args):
    """Run all 12 (feature × edge × probe) combinations.

    Returns list of dicts for table rows.
    """
    # Define all combinations
    combos = [
        # (feat_key, feat_label, edge_key, edge_label, ProbeClass, lr)
        ("feat_7d",  "Regionprops 7D",   "target_all", "All edges",    LinearEdgeProbe, 1e-3),
        ("feat_7d",  "Regionprops 7D",   "target_all", "All edges",    MLPEdgeProbe,    1e-4),
        ("feat_7d",  "Regionprops 7D",   "target_div", "Divisions",    LinearEdgeProbe, 1e-3),
        ("feat_7d",  "Regionprops 7D",   "target_div", "Divisions",    MLPEdgeProbe,    1e-4),

        ("feat_20d", "Hu Moments 20D",   "target_all", "All edges",    LinearEdgeProbe, 1e-3),
        ("feat_20d", "Hu Moments 20D",   "target_all", "All edges",    MLPEdgeProbe,    1e-4),
        ("feat_20d", "Hu Moments 20D",   "target_div", "Divisions",    LinearEdgeProbe, 1e-3),
        ("feat_20d", "Hu Moments 20D",   "target_div", "Divisions",    MLPEdgeProbe,    1e-4),

        ("feat_27d", "Both 27D",         "target_all", "All edges",    LinearEdgeProbe, 1e-3),
        ("feat_27d", "Both 27D",         "target_all", "All edges",    MLPEdgeProbe,    1e-4),
        ("feat_27d", "Both 27D",         "target_div", "Divisions",    LinearEdgeProbe, 1e-3),
        ("feat_27d", "Both 27D",         "target_div", "Divisions",    MLPEdgeProbe,    1e-4),
    ]

    rows = []
    for feat_key, feat_label, edge_key, edge_label, probe_cls, lr in combos:
        probe_name = "Linear" if probe_cls == LinearEdgeProbe else "MLP"
        logger.info(
            f"\n  {feat_label} | {edge_label} | {probe_name} "
            f"(lr={lr}, steps={args.steps})"
        )
        t0 = time.time()
        result = evaluate_combo(
            edge_data, feat_key, edge_key, probe_cls, lr,
            steps=args.steps, patience=args.patience,
        )
        elapsed = time.time() - t0

        if result is None:
            rows.append({
                "Feature Set": feat_label,
                "Edge Type": edge_label,
                "Probe": probe_name,
                "BalAcc": None,
                "F1": None,
                "Precision": None,
                "Recall": None,
            })
            logger.info(f"    → FAILED ({elapsed:.1f}s)")
        else:
            rows.append({
                "Feature Set": feat_label,
                "Edge Type": edge_label,
                "Probe": probe_name,
                "BalAcc": result["final_bal_acc"],
                "F1": result["final_f1"],
                "Precision": result["final_precision"],
                "Recall": result["final_recall"],
            })
            logger.info(
                f"    → bal_acc={result['final_bal_acc']:.4f}  "
                f"f1={result['final_f1']:.4f}  "
                f"({elapsed:.1f}s)"
            )

    return rows


def print_table(rows):
    """Print the comparison table and verdict."""
    print()
    print("=" * 90)
    print("Edge Probing Benchmark — Results")
    print("=" * 90)
    print(
        f"{'Feature Set':<20} {'Edge Type':<15} {'Probe':<8} "
        f"{'BalAcc':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}"
    )
    print("-" * 90)

    for r in rows:
        if r["BalAcc"] is None:
            print(
                f"{r['Feature Set']:<20} {r['Edge Type']:<15} {r['Probe']:<8} "
                f"{'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10}"
            )
        else:
            print(
                f"{r['Feature Set']:<20} {r['Edge Type']:<15} {r['Probe']:<8} "
                f"{r['BalAcc']:<10.4f} {r['F1']:<10.4f} "
                f"{r['Precision']:<10.4f} {r['Recall']:<10.4f}"
            )

    print("-" * 90)
    print()

    # ─── Verdict ─────────────────────────────────────────────────────────
    # Extract results by group
    def get_best(rows, feat_label, edge_label):
        """Get best bal_acc across probe types for a (feat, edge) combo."""
        vals = [
            r["BalAcc"] for r in rows
            if r["Feature Set"] == feat_label
            and r["Edge Type"] == edge_label
            and r["BalAcc"] is not None
        ]
        return max(vals) if vals else 0.0

    all_7d = get_best(rows, "Regionprops 7D", "All edges")
    all_20d = get_best(rows, "Hu Moments 20D", "All edges")
    all_27d = get_best(rows, "Both 27D", "All edges")
    div_7d = get_best(rows, "Regionprops 7D", "Divisions")
    div_20d = get_best(rows, "Hu Moments 20D", "Divisions")
    div_27d = get_best(rows, "Both 27D", "Divisions")

    delta_all = all_20d - all_7d
    delta_div = div_20d - div_7d
    delta_all_27 = all_27d - all_20d
    delta_div_27 = div_27d - div_20d

    lines = []
    lines.append("=" * 90)
    lines.append("VERDICT")
    lines.append("=" * 90)
    lines.append("")

    if delta_all > 0.01:
        lines.append(
            f"All edges: Hu moments BETTER than Regionprops 7D (Δ=+{delta_all:.4f})"
        )
    elif delta_all < -0.01:
        lines.append(
            f"All edges: Hu moments WORSE than Regionprops 7D (Δ={delta_all:.4f})"
        )
    else:
        lines.append(
            f"All edges: Hu moments similar to Regionprops 7D (Δ={delta_all:.4f})"
        )

    if delta_div > 0.01:
        lines.append(
            f"Division edges: Hu moments BETTER than Regionprops 7D (Δ=+{delta_div:.4f})"
        )
    elif delta_div < -0.01:
        lines.append(
            f"Division edges: Hu moments WORSE than Regionprops 7D (Δ={delta_div:.4f})"
        )
    else:
        lines.append(
            f"Division edges: Hu moments similar to Regionprops 7D (Δ={delta_div:.4f})"
        )
    lines.append("")

    if delta_all_27 > 0.01:
        lines.append(
            f"Both features HELP over Hu alone for ALL edges (Δ=+{delta_all_27:.4f})"
        )
    elif delta_all_27 < -0.01:
        lines.append(
            f"Both features HURT over Hu alone for ALL edges (Δ={delta_all_27:.4f})"
        )
    else:
        lines.append(
            f"Both features don't change much over Hu alone for ALL edges "
            f"(Δ={delta_all_27:.4f})"
        )

    if delta_div_27 > 0.01:
        lines.append(
            f"Both features HELP over Hu alone for DIVISIONS (Δ=+{delta_div_27:.4f})"
        )
    elif delta_div_27 < -0.01:
        lines.append(
            f"Both features HURT over Hu alone for DIVISIONS (Δ={delta_div_27:.4f})"
        )
    else:
        lines.append(
            f"Both features don't change much over Hu alone for DIVISIONS "
            f"(Δ={delta_div_27:.4f})"
        )
    lines.append("")

    # Recommendation
    best_all_feat = max(
        [("7D", all_7d), ("Hu", all_20d), ("Both", all_27d)],
        key=lambda x: x[1],
    )
    best_div_feat = max(
        [("7D", div_7d), ("Hu", div_20d), ("Both", div_27d)],
        key=lambda x: x[1],
    )

    if best_all_feat[0] == best_div_feat[0]:
        lines.append(
            f"Recommendation: {best_all_feat[0]} works best for both edge types "
            f"(all={best_all_feat[1]:.4f}, div={best_div_feat[1]:.4f})"
        )
    else:
        lines.append(
            f"Recommendation: depends on edge type — "
            f"{best_all_feat[0]} for ALL edges ({best_all_feat[1]:.4f}), "
            f"{best_div_feat[0]} for DIVISIONS ({best_div_feat[1]:.4f})"
        )

    lines.append("")
    lines.append("=" * 90)

    verdict = "\n".join(lines)
    print(verdict)

    return verdict


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Edge probing benchmark: feature × edge-type × probe"
    )
    p.add_argument("--data-root", default="../../data/vanvliet")
    p.add_argument(
        "--conditions", default="rpsM,recA,pheA,metA",
        help="Comma-separated conditions"
    )
    p.add_argument(
        "--max-pairs", type=int, default=30,
        help="Max consecutive frame pairs to use"
    )
    p.add_argument(
        "--steps", type=int, default=200,
        help="Training steps per probe"
    )
    p.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience (in steps, checked every 20)"
    )
    return p.parse_args(argv)


def main():
    args = parse_args()
    conditions = [c.strip() for c in args.conditions.split(",")]

    logger.info(
        f"Edge Probing Benchmark\n"
        f"  Data root: {args.data_root}\n"
        f"  Conditions: {conditions}\n"
        f"  Max pairs: {args.max_pairs}\n"
        f"  Steps: {args.steps}\n"
        f"  Patience: {args.patience}\n"
        f"  Device: {device}"
    )

    # ─── Step 1: Scan pairs ──────────────────────────────────────────────
    logger.info("Scanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Found {len(pairs)} consecutive frame pairs")

    if len(pairs) < 3:
        logger.error(
            "Need at least 3 consecutive frame pairs. Check --data-root."
        )
        sys.exit(1)

    # ─── Step 2: Build edge data ─────────────────────────────────────────
    logger.info("Building edge data with features...")
    edge_data = build_edge_data(pairs, max_pairs=args.max_pairs)
    logger.info(f"Edge data: {len(edge_data)} frame pairs")

    if len(edge_data) < 2:
        logger.error("Need at least 2 usable frame pairs. Check data.")
        sys.exit(1)

    # ─── Step 3: Run all combinations ────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Running all 12 probe combinations")
    logger.info("=" * 60)
    rows = run_benchmark(edge_data, args)

    # ─── Step 4: Print table and verdict ─────────────────────────────────
    verdict = print_table(rows)

    # Save results
    out_path = Path("results_edge_probe.txt")
    with open(out_path, "w") as f:
        f.write("Edge Probing Benchmark Results\n")
        f.write("=" * 90 + "\n")
        for r in rows:
            f.write(
                f"{r['Feature Set']:20} {r['Edge Type']:15} {r['Probe']:8} "
                f"{str(r['BalAcc']):10} {str(r['F1']):10} "
                f"{str(r['Precision']):10} {str(r['Recall']):10}\n"
            )
        f.write("\n" + verdict + "\n")
    logger.info(f"Results saved to {out_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
