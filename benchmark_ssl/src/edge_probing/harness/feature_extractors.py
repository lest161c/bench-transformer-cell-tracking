"""Feature extractors for edge probing.

Provides per-cell feature extraction for all registered feature types
(regionprops 7D, HOCT 13D, ScaledCNN NT-Xent frozen, DINOv2 frozen,
ScaledCNN end-to-end). Heavy models (DINOv2) are lazy-loaded and
cached on disk per frame under ``results/feature_cache/``.
"""

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table, regionprops as sk_regionprops
from tifffile import imread

from src.data.feature_extraction import _border_dist_fast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("edge_probing.features")
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = PROJECT_ROOT / "results" / "feature_cache"
PATCH_SIZE = 64


# ── Regionprops 7D (wrfeat-style) ────────────────────────────────────────────

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


# ── HOCT 19D (adapted to 2D → 13D) ───────────────────────────────────────────
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
# 2D adaptation used here — 19D → 13D:
#
#   group       HOCT 3D (19D)     this code (13D)   delta   reason
#   ────────────────────────────────────────────────────────────────────────
#   position    t, z, y, x  (4)   t, y, x     (3)    −1     see note 1
#   size        eq_diam     (1)   eq_diam     (1)     0
#   intensity   min/max/    (4)   min/max/    (4)     0
#               mean/std          mean/std
#   inertia     3x3 tensor  (9)   2x2 tensor  (4)    −5     see note 2
#   border      border_dist (1)   border_dist (1)     0     see note 3
#   ────────────────────────────────────────────────────────────────────────
#   total                   19                  13   −6
#
# Notes:
#   1. t is KEPT (not dropped). The HOCT paper includes the absolute time
#      point as an explicit node feature. The model uses it alongside the
#      3D Rotary Position Embedding (RoPE) applied to spatial coordinates
#      (z, y, x) — t is a separate scalar feature, not a positional encoding.
#      In our 2D adaptation we drop z (data is planar) but retain t because
#      it provides a temporal ordering signal that helps the probe
#      distinguish cells in different frames, which is essential for
#      edge prediction between consecutive time points.
#   2. The 2D inertia tensor is 2x2 (4 values, of which 3 are unique due to
#      symmetry) instead of 3x3 (9 values, 6 unique). We keep all 4 entries,
#      mirroring the paper's choice to keep redundant symmetric entries.
#   3. border_dist semantics differ from the official implementation, which
#      computes a clipped inverse distance at the CENTROID:
#      `1 - min(1, dist/5)` (0 if >= 5 px from border, 1 at the border).
#      Here we compute the Euclidean distance of the closest REGION PIXEL to
#      the FoV border, in pixels, non-inverted and unclipped (larger = farther
#      from border). The two statistics are highly correlated; since probes
#      standardize features and learn signed weights, the directionality
#      difference is absorbed. Kept for consistency with earlier experiments.
#   4. Standardization: the paper standardizes all features per dataset. This
#      is handled downstream (leakage-safe per-fold z-score), NOT inside this
#      extractor — extract_hoct19 returns RAW features. See
#      fit_feature_standardizer() / apply_feature_standardizer().
#   5. The HOCT paper uses a multi-frame window for inference; our probe
#      evaluates single frame pairs (t, t+1). The t feature is the absolute
#      frame index within the experiment, which is meaningful for temporal
#      ordering even in pairwise evaluation.

#: Ordered names of the 13 features returned by extract_hoct19().
#: Shared with analysis/visualization tools (e.g. visualize_props.py).
HOCT2D_FEATURE_NAMES = [
    # position (3) — t (frame index), y, x centroid in pixels
    "time", "centroid_y", "centroid_x",
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


def extract_hoct19(mask, img, frame_idx=0):
    """Extract the 13D HOCT-style node features for one 2D frame.

    2D adaptation of the HOCT paper's 19D (3D+t) node features — see the
    section header above for the exact 19D → 13D mapping (drop z: −1,
    inertia 3×3→2×2: −5) and the deliberate deviations.

    Layout (order matches HOCT2D_FEATURE_NAMES):
      Position  (3): t (frame index), y, x centroid [pixels]
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
        frame_idx: absolute frame index within the experiment.
            Used as the time feature (t) in the HOCT-style node
            representation. The HOCT paper includes t as an explicit
            scalar feature alongside the 3D RoPE applied to spatial
            coordinates.

    Returns:
        (coords, labels, features) with coords (N, 2), labels (N,),
        features (N, 13) float32 — or (None, None, None) if mask is empty.
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

    # 1. Position (3): t (frame index), y, x centroid
    time_col = np.full((len(df), 1), frame_idx, dtype=np.float32)
    position = np.column_stack([
        time_col,
        df["centroid-0"].values.astype(np.float32),
        df["centroid-1"].values.astype(np.float32),
    ])

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
