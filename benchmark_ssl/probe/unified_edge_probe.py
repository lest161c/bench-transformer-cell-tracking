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
  hoct19    — HOCT-style 12D (2D-adapted from 19D): centroid(2),
              eq_diam, intensity(4), inertia(4), border  [12 dim]
              (see extract_hoct19() for the exact 19D -> 12D mapping
              and the deliberate deviations from the HOCT paper)

Normalization:
  All frozen (non-learned) feature vectors are z-score standardized before
  probing, following the HOCT paper ("All features are standardized per
  dataset"). Statistics are fit on the training split of each CV fold only
  and applied to the validation split (leakage-safe standardization).
  Disable with --no-standardize. cnn_e2e is exempt (learned features).

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
import copy
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
from scipy.ndimage import distance_transform_edt
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

EM_DASH = '\u2014'

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
    height, width = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i = int(round(float(cy)))
        cx_i = int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, height - 1)
        cx_i = np.clip(cx_i, 0, width - 1)
        y1 = cy_i - half
        x1 = cx_i - half
        y2 = cy_i + half
        x2 = cx_i + half
        pt = max(0, -y1)
        pb = max(0, y2 - height)
        pl = max(0, -x1)
        pr = max(0, x2 - width)
        y1c = max(0, y1)
        x1c = max(0, x1)
        y2c = min(height, y2)
        x2c = min(width, x2)
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
                tuple((0, max(0, pad_amount)) for pad_amount in [patch_size - dim_size for dim_size in crop.shape]),
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
    return tuple(region.intensity_max for region in sk_regionprops(mask, intensity_image=dist))


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


# ── HOCT 19D (adapted to 2D → 12D) ───────────────────────────────────────────
#
# Reference — "Higher Order Cell Tracking" (HOCT) paper, appendix "Input
# features" (papers/higher_order_cell_tracking.pdf):
#
#   "Each node i carries a d=19-dimensional feature vector xi derived from the
#    segmentation mask: spatiotemporal position (t, z, y, x), equivalent
#    diameter, intensity statistics (min, max, mean, standard deviation), the
#    3x3 inertia tensor (9 values), and the distance to the nearest
#    field-of-view border. All features are standardized per dataset."
#
# Cross-checked against the official implementation (hoct/hoct/src/hoct):
# node_feats = [t, z, y, x, eq_diam, int_min, int_max, int_mean, int_std,
#               inertia_tensor(9), border_dist]  →  19 dims, matching the
# 19-element per-dataset standardization vectors (_MEAN/_STD in _api.py).
#
# 2D adaptation used here — 19D → 12D:
#
#   group       HOCT 3D (19D)     this code (12D)   delta   reason
#   ────────────────────────────────────────────────────────────────────────
#   position    t, z, y, x  (4)   y, x        (2)    −2     see notes 1+2
#   size        eq_diam     (1)   eq_diam     (1)     0
#   intensity   min/max/    (4)   min/max/    (4)     0
#               mean/std          mean/std
#   inertia     3x3 tensor  (9)   2x2 tensor  (4)    −5     see note 3
#   border      border_dist (1)   border_dist (1)     0     see note 4
#   ────────────────────────────────────────────────────────────────────────
#   total                   19                  12   −7
#
# Notes:
#   1. t is dropped DELIBERATELY. The paper includes the absolute time point
#      as an input feature because HOCT reasons over multi-frame windows. Our
#      probe evaluates single frame pairs (t, t+1), where t is constant per
#      frame and the global frame index is arbitrary — it carries no
#      discriminative signal for edge identity and would only encode dataset
#      sampling artifacts. Dropping it is an intentional deviation.
#   2. z is dropped because the data is 2D (the official code instead keeps a
#      constant pseudo-z=0 plane for 2D data, which adds only dead weight).
#   3. The 2D inertia tensor is 2x2 (4 values, of which 3 are unique due to
#      symmetry) instead of 3x3 (9 values, 6 unique). We keep all 4 entries,
#      mirroring the paper's choice to keep redundant symmetric entries.
#   4. border_dist semantics differ from the official implementation, which
#      computes a clipped inverse distance at the CENTROID:
#      `1 - min(1, dist/5)` (0 if >= 5 px from border, 1 at the border).
#      Here we compute the Euclidean distance of the closest REGION PIXEL to
#      the FoV border, in pixels, non-inverted and unclipped (larger = farther
#      from border). The two statistics are highly correlated; since probes
#      standardize features and learn signed weights, the directionality
#      difference is absorbed. Kept for consistency with earlier experiments.
#   5. Standardization: the paper standardizes all features per dataset. This
#      is handled downstream (leakage-safe per-fold z-score), NOT inside this
#      extractor — extract_hoct19 returns RAW features. See
#      fit_feature_standardizer() / apply_feature_standardizer().

#: Ordered names of the 12 features returned by extract_hoct19().
#: Shared with analysis/visualization tools (e.g. visualize_props.py).
HOCT2D_FEATURE_NAMES = [
    # position (2) — y, x centroid in pixels
    "centroid_y", "centroid_x",
    # size (1) — diameter of a circle with the same area, in pixels
    "eq_diam",
    # intensity (4) — within-region statistics of the normalized image [0, 1]
    "intensity_min", "intensity_max", "intensity_mean", "intensity_std",
    # inertia (4) — 2x2 inertia tensor, row-major (I01 == I10 by symmetry)
    "inertia_00", "inertia_01", "inertia_10", "inertia_11",
    # border (1) — min Euclidean distance of any region pixel to the FoV edge
    "border_dist",
]


def _compute_border_dist(mask):
    """Per-cell minimum distance to the nearest field-of-view edge, in pixels.

    Builds a binary map of the image border, computes its Euclidean distance
    transform, and takes the minimum distance over each region's pixels
    (vectorized via regionprops ``intensity_min``).

    Result: 0 if the cell touches the FoV border; larger = farther away.
    NOTE: direction and statistic differ from the official HOCT
    ``border_dist`` (clipped inverse distance at the centroid) — see note 4
    in the section header above.

    Returns array of distances (one per region, in regionprops label order).
    """
    ndim = mask.ndim
    edge_mask = np.zeros(mask.shape, dtype=bool)
    for axis in range(ndim):
        sl = [slice(None)] * ndim
        sl[axis] = 0
        edge_mask[tuple(sl)] = True
        sl[axis] = -1
        edge_mask[tuple(sl)] = True

    # Distance from every pixel to nearest image border
    dist_to_border = distance_transform_edt(~edge_mask)
    # For each region, the minimum distance to border (vectorized via regionprops)
    return np.array([
        region.intensity_min
        for region in sk_regionprops(mask, intensity_image=dist_to_border)
    ], dtype=np.float32)


def extract_hoct19(mask, img):
    """Extract the 12D HOCT-style node features for one 2D frame.

    2D adaptation of the HOCT paper's 19D (3D+t) node features — see the
    section header above for the exact 19D → 12D mapping (drop t: −1,
    drop z: −1, inertia 3×3→2×2: −5) and the deliberate deviations.

    Layout (order matches HOCT2D_FEATURE_NAMES):
      Position  (2): y, x centroid [pixels]
      Size      (1): equivalent_diameter_area [pixels]
      Intensity (4): min, max, mean, std of the normalized image [0, 1]
      Inertia   (4): 2×2 inertia tensor, flattened row-major
      Border    (1): min distance of any region pixel to the FoV edge [pixels]

    Features are returned RAW (not standardized). Standardization is applied
    downstream, fit on the training split only (see module docstring).

    Args:
        mask: (H, W) integer label image.
        img:  (H, W) float intensity image (already normalized to [0, 1]
              by load_frame).

    Returns:
        (coords, labels, features) with coords (N, 2), labels (N,),
        features (N, 12) float32 — or (None, None, None) if mask is empty.
    """
    ndim = mask.ndim
    props = ("equivalent_diameter_area", "intensity_min", "intensity_max",
             "intensity_mean", "intensity_std", "inertia_tensor")
    df = pd.DataFrame(
        regionprops_table(mask, intensity_image=img,
                          properties=("label", "centroid", *props))
    )
    if len(df) == 0:
        return None, None, None

    coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    # 1. Position (2): y, x centroid
    position = df[["centroid-0", "centroid-1"]].values.astype(np.float32)

    # 2. Size (1): equivalent_diameter_area
    eq_diam = df["equivalent_diameter_area"].values.astype(np.float32)[:, None]

    # 3. Intensity (4): min, max, mean, std
    intensity = np.column_stack([
        df["intensity_min"].values,
        df["intensity_max"].values,
        df["intensity_mean"].values,
        df["intensity_std"].values,
    ]).astype(np.float32)

    # 4. Inertia (4 for 2D): 2×2 tensor flattened
    inertias = np.stack(
        [np.column_stack([df[f"inertia_tensor-{i}-{j}"] for j in range(ndim)])
         for i in range(ndim)], axis=-1
    ).reshape(len(df), -1).astype(np.float32)

    # 5. Border (1): min distance to nearest FoV edge
    border_dists = _compute_border_dist(mask)[:, None]

    features = np.concatenate(
        [position, eq_diam, intensity, inertias, border_dists], axis=-1
    ).astype(np.float32)

    assert features.shape[1] == len(HOCT2D_FEATURE_NAMES), (
        f"HOCT 2D feature count drifted: got {features.shape[1]}, "
        f"expected {len(HOCT2D_FEATURE_NAMES)} (see HOCT2D_FEATURE_NAMES)"
    )

    return coords, labels, features


# ── ScaledCNN (same arch as cnn_ssl.py) ──────────────────────────────────────

class ScaledCNN(nn.Module):
    """ConvNet for 64x64 grayscale patches. Output: 128-dim embedding."""
    def __init__(self, scale='large', out_dim=128):
        """Initialize the ScaledCNN feature extractor.

        Args:
            scale: architecture size — 'small', 'medium', or 'large'
                (controls the list of per-convolution channel widths).
            out_dim: dimensionality of the output embedding per patch.
        """
        super().__init__()
        if scale == 'small':
            channel_widths = [8, 16, 32]
        elif scale == 'medium':
            channel_widths = [16, 32, 64]
        else:  # large
            channel_widths = [32, 64, 128, 256]

        layers = []
        in_channels = 1
        for out_channels in channel_widths:
            layers += [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        spatial = PATCH_SIZE // (2 ** len(channel_widths))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_widths[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, patches):
        """Compute an embedding for a batch of patches.

        Args:
            patches: (N, 1, 64, 64) float tensor of grayscale patches.

        Returns:
            (N, out_dim) float tensor of patch embeddings.
        """
        return self.fc(self.conv(patches))


# ── DINOv2 (lazy-loaded, cached) ────────────────────────────────────────────

_DINO_MODEL = None

def load_dino():
    """Load the DINOv2 (dinov2_vits14) model once and cache it globally.

    The model is downloaded via torch.hub on first use and kept in the
    module-level _DINO_MODEL cache to avoid repeated loading.

    Returns:
        The eval-mode DINOv2 model on the current device.
    """
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
    patches_norm = (patches_np - pmin) / (pmax - pmin + 1e-8)
    patches_tensor = torch.from_numpy(patches_norm).float().unsqueeze(1).to(device)  # (N, 1, 64, 64)
    patches_tensor = F.interpolate(patches_tensor, size=(224, 224), mode="bilinear", align_corners=False)
    patches_tensor = patches_tensor.expand(-1, 3, -1, -1)  # -> (N, 3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    patches_tensor = (patches_tensor - mean) / std
    return model(patches_tensor).cpu().numpy()


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
        """Flatten an (n_cells_t, n_cells_n) target matrix into one sample per pair.

        Args:
            feat_t: (n_cells_t, D) features of cells in frame t.
            feat_n: (n_cells_n, D) features of cells in frame t+1.
            target: (n_cells_t, n_cells_n) binary edge matrix; each entry yields
                one training sample.
        """
        n_cells_t, n_cells_n = target.shape
        pairs_t, pairs_n, labels = [], [], []
        for i in range(n_cells_t):
            for j in range(n_cells_n):
                pairs_t.append(feat_t[i])
                pairs_n.append(feat_n[j])
                labels.append(int(target[i, j].item()))
        self.pairs_t = torch.stack(pairs_t) if pairs_t else torch.empty(0, feat_t.shape[-1])
        self.pairs_n = torch.stack(pairs_n) if pairs_n else torch.empty(0, feat_n.shape[-1])
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Return the number of cell-cell pair samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return the (feat_t, feat_n, label) sample at index idx."""
        return self.pairs_t[idx], self.pairs_n[idx], self.labels[idx]


class EdgePairDatasetPatches(Dataset):
    """
    Dataset of individual cell-cell pairs where features are patches
    (for CNN end-to-end training).
    """
    def __init__(self, patches_t, patches_n, target):
        """Flatten an (n_cells_t, n_cells_n) target matrix into one sample per pair.

        Unlike EdgePairDataset, the per-cell features are raw image patches
        (used for CNN end-to-end training).

        Args:
            patches_t: (n_cells_t, 1, H, W) patches of cells in frame t.
            patches_n: (n_cells_n, 1, H, W) patches of cells in frame t+1.
            target: (n_cells_t, n_cells_n) binary edge matrix.
        """
        n_cells_t, n_cells_n = target.shape
        p_t, p_n, labels = [], [], []
        for i in range(n_cells_t):
            for j in range(n_cells_n):
                p_t.append(patches_t[i])
                p_n.append(patches_n[j])
                labels.append(int(target[i, j].item()))
        self.patches_t = torch.stack(p_t) if p_t else torch.empty(0, *patches_t.shape[1:])
        self.patches_n = torch.stack(p_n) if p_n else torch.empty(0, *patches_n.shape[1:])
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Return the number of cell-cell pair samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return the (patches_t, patches_n, label) sample at index idx."""
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
    feat_t = torch.stack([sample[0] for sample in batch])
    feat_n = torch.stack([sample[1] for sample in batch])
    labels = torch.stack([sample[2] for sample in batch])
    return feat_t, feat_n, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature standardization (leakage-safe, per training split)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_feature_standardizer(datasets):
    """Fit per-feature z-score statistics (mean, std) on training datasets.

    The HOCT paper standardizes all node features per dataset; since raw
    handcrafted features mix wildly different scales (centroids ~10² px vs
    intensities ~[0, 1] vs border distances ~10⁰–10¹ px), unregularized
    probes would otherwise weight features by their numeric magnitude.

    Statistics are pooled over all node features from BOTH sides of every
    training pair (pairs_t and pairs_n) — the exact distribution the probe
    is trained on. Note the stats are computed on pair-expanded rows, so
    cells are implicitly weighted by their pair multiplicity; the probe sees
    the same weighting, so this is consistent.

    IMPORTANT: fit only on the training split of each fold, never on the
    full dataset — otherwise validation statistics leak into training.

    Args:
        datasets: list of EdgePairDataset (training split only).

    Returns:
        (mean, std): (D,) float32 tensors. std is clamped to >= 1e-6 so
        constant features (e.g. a degenerate inertia entry) stay finite and
        map to 0 after transformation.
    """
    feats = torch.cat(
        [ds.pairs_t for ds in datasets] + [ds.pairs_n for ds in datasets],
        dim=0,
    )
    mean = feats.mean(dim=0)
    std = feats.std(dim=0).clamp(min=1e-6)
    return mean, std


def apply_feature_standardizer(datasets, mean, std):
    """Apply a fitted standardizer to datasets, returning transformed copies.

    Applies z = (x - mean) / std to both sides of every pair. The fitted
    statistics must come from fit_feature_standardizer() on the training
    split; validation data is only transformed, never refit.

    Datasets without pair features (EdgePairDatasetPatches for cnn_e2e —
    learned features are exempt from standardization) pass through unchanged.

    Args:
        datasets: list of EdgePairDataset / EdgePairDatasetPatches.
        mean, std: (D,) tensors from fit_feature_standardizer().

    Returns:
        New list of shallow copies with standardized pair features; the
        input datasets are not modified.
    """
    out = []
    for dataset in datasets:
        if not hasattr(dataset, "pairs_t"):
            out.append(dataset)  # e.g. EdgePairDatasetPatches (cnn_e2e): skip
            continue
        dataset_copy = copy.copy(dataset)
        dataset_copy.pairs_t = (dataset.pairs_t - mean) / std
        dataset_copy.pairs_n = (dataset.pairs_n - mean) / std
        out.append(dataset_copy)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Probe architectures
# ═══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    """Linear probe: concat(feat_t[i], feat_n[j]) -> score."""
    def __init__(self, feat_dim):
        """Initialize the linear probe.

        Args:
            feat_dim: dimensionality of each cell's feature vector; the layer
                maps the concatenation of the two cells' features (2*feat_dim)
                to a single logit.
        """
        super().__init__()
        self.fc = nn.Linear(2 * feat_dim, 1)

    def forward(self, feat_t, feat_n):
        """feat_t: (N, D), feat_n: (N, D) -> scores: (N,)"""
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.fc(pairs).squeeze(-1)  # (N,)


class MLPProbe(nn.Module):
    """2-layer MLP probe: concat(feat_t[i], feat_n[j]) -> hidden -> score."""
    def __init__(self, feat_dim, hidden=128):
        """Initialize the 2-layer MLP probe.

        Args:
            feat_dim: dimensionality of each cell's feature vector; the input
                layer maps the concatenation of the two cells' features
                (2*feat_dim) to `hidden` units.
            hidden: number of hidden units between the two linear layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_t, feat_n):
        """Score a batch of cell-cell pairs.

        Args:
            feat_t: (N, feat_dim) features of cells in frame t.
            feat_n: (N, feat_dim) features of cells in frame t+1.

        Returns:
            (N,) logit scores, one per pair.
        """
        pairs = torch.cat([feat_t, feat_n], dim=-1)  # (N, 2D)
        return self.net(pairs).squeeze(-1)  # (N,)


class CNNProbeE2E(nn.Module):
    """ScaledCNN + probe, trained end-to-end."""
    def __init__(self, scale='large', out_dim=128, probe_type='linear'):
        """Initialize the end-to-end CNN + probe model.

        Args:
            scale: ScaledCNN architecture size ('small'/'medium'/'large').
            out_dim: dimensionality of the CNN patch embedding.
            probe_type: 'linear' for a single linear layer or 'mlp' for a
                2-layer MLP on the concatenated embedding pair.
        """
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
            avg_val_bal_acc = float(np.mean([metrics["bal_acc"] for metrics in val_metrics_list]))
            avg_val_f1 = float(np.mean([metrics["f1"] for metrics in val_metrics_list]))

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
                best_state = {key: value.cpu().clone() for key, value in probe.state_dict().items()}
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
        "final_bal_acc": float(np.mean([metrics["bal_acc"] for metrics in final_metrics_list])),
        "final_f1": float(np.mean([metrics["f1"] for metrics in final_metrics_list])),
        "final_precision": float(np.mean([metrics["precision"] for metrics in final_metrics_list])),
        "final_recall": float(np.mean([metrics["recall"] for metrics in final_metrics_list])),
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
      - feat_t: (n_cells_t, D) or patches_t: (n_cells_t, 1, 64, 64)
      - feat_n: (n_cells_n, D) or patches_n: (n_cells_n, 1, 64, 64)
      - target: (n_cells_t, n_cells_n) binary matrix
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
            lt_map = {label: i for i, label in enumerate(labels_rp_t)}
            ln_map = {label: i for i, label in enumerate(labels_rp_n)}
            idx_t = [lt_map[label] for label in labels_t if label in lt_map]
            idx_n = [ln_map[label] for label in labels_n if label in ln_map]
            if len(idx_t) < 2 or len(idx_n) < 2:
                continue
            feats_t = torch.from_numpy(feats_t[idx_t]).float()
            feats_n = torch.from_numpy(feats_n[idx_n]).float()
            labels_t_use = labels_t[[label in lt_map for label in labels_t]]
            labels_n_use = labels_n[[label in ln_map for label in labels_n]]

        elif feature_type == 'hoct19':
            # HOCT 12D (adapted from 19D)
            _, labels_h_t, feats_t = extract_hoct19(mask_t, imgt)
            _, labels_h_n, feats_n = extract_hoct19(mask_n, imgn)
            if feats_t is None or feats_n is None:
                continue
            # Align by label
            lt_map = {label: i for i, label in enumerate(labels_h_t)}
            ln_map = {label: i for i, label in enumerate(labels_h_n)}
            idx_t = [lt_map[label] for label in labels_t if label in lt_map]
            idx_n = [ln_map[label] for label in labels_n if label in ln_map]
            if len(idx_t) < 2 or len(idx_n) < 2:
                continue
            feats_t = torch.from_numpy(feats_t[idx_t]).float()
            feats_n = torch.from_numpy(feats_n[idx_n]).float()
            labels_t_use = labels_t[[label in lt_map for label in labels_t]]
            labels_n_use = labels_n[[label in ln_map for label in labels_n]]

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
            n_cells_t, n_cells_n = len(labels_t), len(labels_n)
            target = torch.zeros(n_cells_t, n_cells_n, dtype=torch.float32)
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
                "n_neg": n_cells_t * n_cells_n - n_pos,
                "condition": cond,
                "experiment": exp,
            })
            total += 1
            continue
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        # Build target matrix
        if feature_type != 'cnn_e2e':
            n_cells_t, n_cells_n = len(labels_t_use), len(labels_n_use)
            target = torch.zeros(n_cells_t, n_cells_n, dtype=torch.float32)
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
                "n_neg": n_cells_t * n_cells_n - n_pos,
                "condition": cond,
                "experiment": exp,
            })
            total += 1

    logger.info(f"  Built {total} frame pairs for {feature_type}")
    return edge_data, cnn_frozen_model


def split_by_condition(edge_data, train_conditions, val_conditions):
    """Split edge data by condition (experimental group)."""
    train_data = [item for item in edge_data if item["condition"] in train_conditions]
    val_data = [item for item in edge_data if item["condition"] in val_conditions]
    return train_data, val_data


def flatten_to_pairs(edge_data):
    """
    Convert list of per-frame-pair dicts into a flat list of datasets,
    one per frame pair (for later concatenation into a single dataset).
    """
    datasets = []
    for item in edge_data:
        if "feat_t" in item:
            dataset = EdgePairDataset(item["feat_t"], item["feat_n"], item["target"])
        else:
            # For cnn_e2e, patches are stored
            dataset = EdgePairDatasetPatches(item["patches_t"], item["patches_n"],
                                             item.get("target"))
        if len(dataset) > 0:
            datasets.append(dataset)
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

        # Leakage-safe standardization: fit on train folds, transform val fold
        if not is_e2e and getattr(args, "standardize", True):
            feat_mean, feat_std = fit_feature_standardizer(train_datasets)
            train_datasets = apply_feature_standardizer(train_datasets, feat_mean, feat_std)
            val_datasets = apply_feature_standardizer(val_datasets, feat_mean, feat_std)

        train_dataset = concatenate_datasets(train_datasets)
        val_dataset = concatenate_datasets(val_datasets)

        lr = args.lr if probe_name == "Linear" else args.lr / 10
        batch_size = args.batch_size_linear if probe_name == "Linear" else args.batch_size_mlp

        if is_e2e:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
                )
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
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

    bal_accs = [result["final_bal_acc"] for result in all_fold_results]
    f1s = [result["final_f1"] for result in all_fold_results]

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
        "standardized": bool(not is_e2e and getattr(args, "standardize", True)),
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
    'hoct19': {
        'feat_dim': 12,
        'display': 'HOCT 19D (2D \u2192 12D)',
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
        if feature_type != 'cnn_e2e' and getattr(args, "standardize", True):
            logger.info("  Features z-score standardized per fold (fit on train folds)")
        all_datasets = flatten_to_pairs(edge_data)
        logger.info(f"  Total frame-pair datasets: {len(all_datasets)}")
        if not all_datasets:
            logger.warning("  No pairs after flattening, skipping")
            return None

        results = {}

        def _run_cv(probe_name, feat_dim, is_e2e):
            """Run k-fold cross-validation for one probe type on all frame-pair datasets."""
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

    # ── Leakage-safe standardization: fit on train split, transform val ───
    if feature_type != 'cnn_e2e' and getattr(args, "standardize", True):
        feat_mean, feat_std = fit_feature_standardizer(train_datasets)
        train_datasets = apply_feature_standardizer(train_datasets, feat_mean, feat_std)
        val_datasets = apply_feature_standardizer(val_datasets, feat_mean, feat_std)
        logger.info("  Features z-score standardized (fit on train split)")

    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)

    n_train_pos = sum(int(dataset.labels.sum()) for dataset in train_datasets)
    n_train_neg = sum(len(dataset) - int(dataset.labels.sum()) for dataset in train_datasets)
    logger.info(f"  Train pairs: {len(train_dataset)} ({n_train_pos} pos / {n_train_neg} neg)")
    n_val_pos = sum(int(dataset.labels.sum()) for dataset in val_datasets)
    n_val_neg = sum(len(dataset) - int(dataset.labels.sum()) for dataset in val_datasets)
    logger.info(f"  Val pairs:   {len(val_dataset)} ({n_val_pos} pos / {n_val_neg} neg)")

    results = {}

    # ── Helper to run one probe type ──────────────────────────────────────
    def _run_probe(probe_name, probe_lr, batch_size):
        """Train and evaluate one probe architecture on this feature type.

        Args:
            probe_name: 'Linear' or 'MLP'.
            probe_lr: learning rate for the probe.
            batch_size: batch size for training and validation loaders.

        Returns:
            Metrics dict from train_probe, or None.
        """
        logger.info(f"  Training {probe_name} probe (lr={probe_lr}, bs={batch_size})...")

        if feature_type == 'cnn_e2e':
            # End-to-end: patches stored in datasets, build loaders specially
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
                )
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: (
                    torch.stack([sample[0] for sample in batch]),
                    torch.stack([sample[1] for sample in batch]),
                    torch.stack([sample[2] for sample in batch]),
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
        result is not None
        and any(isinstance(metrics, dict) and metrics.get("is_cv", False) for metrics in result.values())
        for result in all_results.values()
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
                print(f"{cfg['display']:<28} {EM_DASH:<8} {EM_DASH:<22} {EM_DASH:<22} {'SKIP':<10}")
            else:
                print(f"{cfg['display']:<25} {EM_DASH:<8} {EM_DASH:<10} {EM_DASH:<10} "
                      f"{EM_DASH:<10} {EM_DASH:<10} {'SKIP':<10}")
            continue

        result = all_results[fkey]
        for ptype in ["linear", "mlp"]:
            if ptype not in result or result[ptype] is None:
                continue
            metrics = result[ptype]

            if metrics.get("is_cv", False):
                print(f"{cfg['display']:<28} {ptype:<8} "
                      f"{metrics['bal_acc_mean']:.4f}\u00b1{metrics['bal_acc_std']:<.4f}        "
                      f"{metrics['f1_mean']:.4f}\u00b1{metrics['f1_std']:<.4f}        "
                      f"{'OK':<10}")
                rows.append({
                    "feature": cfg['display'],
                    "probe": ptype,
                    "bal_acc": metrics['bal_acc_mean'],
                    "bal_acc_std": metrics['bal_acc_std'],
                    "f1": metrics['f1_mean'],
                    "f1_std": metrics['f1_std'],
                    "is_cv": True,
                })
            else:
                print(f"{cfg['display']:<25} {ptype:<8} "
                      f"{metrics['final_bal_acc']:<10.4f} {metrics['final_f1']:<10.4f} "
                      f"{metrics['final_precision']:<10.4f} {metrics['final_recall']:<10.4f} "
                      f"{'OK':<10}")
                rows.append({
                    "feature": cfg['display'],
                    "probe": ptype,
                    "bal_acc": metrics['final_bal_acc'],
                    "f1": metrics['final_f1'],
                    "precision": metrics['final_precision'],
                    "recall": metrics['final_recall'],
                })

        # Check for shuffled results
        for ptype in ["linear_shuffled", "mlp_shuffled"]:
            if ptype in result and result[ptype] is not None:
                has_shuffled = True

    # Print shuffled baseline section
    if has_shuffled:
        print("--- shuffled baseline ---")
        for fkey, cfg in FEATURE_CONFIGS.items():
            if fkey not in all_results or all_results[fkey] is None:
                continue
            result = all_results[fkey]
            for ptype, label in [("linear_shuffled", "linear"), ("mlp_shuffled", "mlp")]:
                if ptype not in result or result[ptype] is None:
                    continue
                metrics = result[ptype]
                display_name = f"{cfg['display']} (shuffled)"
                print(f"{display_name:<28} {label:<8} "
                      f"{metrics['bal_acc_mean']:.4f}\u00b1{metrics['bal_acc_std']:<.4f}        "
                      f"{metrics['f1_mean']:.4f}\u00b1{metrics['f1_std']:<.4f}        "
                      f"{'SHUF':<10}")

    print("-" * 120)
    print()

    # Best performer
    if rows:
        if is_cv_mode:
            best = max(rows, key=lambda row: row["bal_acc"])
            print(f"Best performer: {best['feature']} + {best['probe']} "
                  f"(bal_acc={best['bal_acc']:.4f}\u00b1{best['bal_acc_std']:.4f}, "
                  f"f1={best['f1']:.4f}\u00b1{best['f1_std']:.4f})")
        else:
            best = max(rows, key=lambda row: row["bal_acc"])
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
    keys = [feature_key.strip() for feature_key in features_arg.split(",")]
    for key in keys:
        if key not in AVAILABLE_FEATURES:
            logger.warning(f"Unknown feature: {key}. Available: {AVAILABLE_FEATURES}")
    return [key for key in keys if key in AVAILABLE_FEATURES]


def parse_args(argv=None):
    """Parse command-line arguments for the unified edge probing benchmark."""
    parser = argparse.ArgumentParser(
        description="Unified edge probing benchmark for cell tracking"
    )
    parser.add_argument("--data-root", default="../../data/vanvliet",
                   help="Path to vanvliet data root")
    parser.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL",
                   help="Comma-separated conditions to scan")
    parser.add_argument("--train-conditions", default="rpsM,recA,pheA,metA",
                   help="Conditions for training")
    parser.add_argument("--val-conditions", default="cib,trpL",
                   help="Conditions for validation")
    parser.add_argument("--features", default="rp",
                   help=f"Features to test. Choices: {AVAILABLE_FEATURES}, "
                        f"or aliases: all, shallow, deep")
    parser.add_argument("--probe", default="both", choices=["linear", "mlp", "both"],
                   help="Probe architecture to use")
    parser.add_argument("--max-pairs", type=int, default=30,
                   help="Max consecutive frame pairs per condition")
    parser.add_argument("--epochs", type=int, default=200,
                   help="Training epochs per probe")
    parser.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for linear probe (MLP uses lr/10)")
    parser.add_argument("--batch-size-linear", type=int, default=256,
                   help="Batch size for linear probe")
    parser.add_argument("--batch-size-mlp", type=int, default=128,
                   help="Batch size for MLP probe")
    parser.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs)")
    parser.add_argument("--eval-every", type=int, default=1,
                   help="Evaluate every N epochs")
    parser.add_argument("--checkpoint", type=str,
                   default=os.path.join(
                       os.path.dirname(__file__),
                       "../cnn_encoder/probe/cnn_ntxent_large.pt"
                   ),
                   help="Path to NT-Xent checkpoint for cnn_frozen")
    parser.add_argument("--output", type=str, default="",
                   help="Path to save JSON results (default: no save)")
    parser.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    parser.add_argument("--no-cache", action="store_true",
                   help="Skip loading/saving feature cache")
    parser.add_argument("--cv-folds", type=int, default=0,
                   help="Number of CV folds (0 = use original train/val split)")
    parser.add_argument("--shuffle-baseline", action="store_true",
                   help="Also run on shuffled features as overfitting baseline")
    parser.add_argument("--val-frac", type=float, default=0.2,
                   help="Validation fraction for non-CV mode (used in random split fallback)")
    parser.add_argument("--standardize", dest="standardize", action="store_true",
                   default=True,
                   help="Z-score standardize frozen features per feature dim, "
                        "fit on the training split only (HOCT-paper default: on)")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false",
                   help="Disable feature standardization (legacy behavior)")
    return parser.parse_args(argv)


def main():
    """Run the unified edge probing benchmark for the requested feature types."""
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
    logger.info(f"  Standardize:    {args.standardize}")
    logger.info("=" * 60)

    # ── Scan pairs ────────────────────────────────────────────────────────
    logger.info("Scanning consecutive frame pairs...")
    all_pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Found {len(all_pairs)} consecutive frame pairs")

    # Log condition breakdown
    from collections import Counter
    cond_counts = Counter(pair[5] for pair in all_pairs)
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
            logger.error(f"Error running {fkey}: {exc}", exc_info=True)
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
                            key: value for key, value in metrics.items() if key != "history"
                        }

        with open(output_path, "w") as file_handle:
            json.dump({
                "args": vars(args),
                "results": serializable,
                "rows": rows,
            }, file_handle, indent=2)
        logger.info(f"Results saved to {output_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
