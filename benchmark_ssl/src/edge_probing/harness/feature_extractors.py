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
from src.models.fourier_pe import FourierPE

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


def extract_regionprops_7d_fourier(mask, img, frame_idx=0, n_freqs=8, cutoff=128.0):
    """Extract 7D regionprops features with Fourier positional encoding.

    Same base features as ``extract_regionprops_7d`` plus Fourier PE
    of the spatial position (t, y, x).  The Fourier PE replaces the
    raw centroid coordinates with a smooth sin/cos encoding at
    geometrically decaying frequencies (see
    ``_init_fourier_frequencies`` in ``fourier_pe.py``).

    Layout (order matches HOCT2D_FEATURE_NAMES):
      fourier_pe (3 * n_freqs * 2): sin/cos of (t, centroid_y, centroid_x)
      eq_diam           (1): equivalent diameter area
      intensity_mean    (1): mean intensity within region
      inertia           (4): 2x2 inertia tensor, flattened row-major
      border_dist       (1): min distance of any region pixel to FoV edge

    Args:
        mask: (H, W) integer label image.
        img:  (H, W) float intensity image (already normalized to [0, 1]).
        frame_idx: absolute frame index within the experiment.
        n_freqs: number of frequency components per coordinate dimension.
        cutoff: controls the decay rate of the geometric frequency schedule.

    Returns:
        (coords, labels, features) with coords (N, 2), labels (N,),
        features (N, 3 * n_freqs * 2 + 7) float32 — or (None, None, None)
        if mask is empty.
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

    # Fourier PE of positions (t, y, x)
    pe = FourierPE(coord_dim=ndim + 1, n_freqs=n_freqs, cutoff=cutoff)
    time_col = np.full((len(df), 1), frame_idx, dtype=np.float32)
    position = np.column_stack([
        time_col,
        df["centroid-0"].values.astype(np.float32),
        df["centroid-1"].values.astype(np.float32),
    ])
    with torch.no_grad():
        fourier_pe = pe(torch.from_numpy(position)).numpy().astype(np.float32)

    # Inertia (4 for 2D): 2x2 tensor flattened
    inertias = np.stack(
        [np.column_stack([df[f"inertia_tensor-{i}-{j}"] for j in range(ndim)])
         for i in range(ndim)], axis=-1
    ).reshape(len(df), -1).astype(np.float32)

    features = OrderedDict()
    features["fourier_pe"] = fourier_pe
    features["eq_diam"] = df["equivalent_diameter_area"].values.astype(np.float32)[:, None]
    features["intensity"] = df["intensity_mean"].values.astype(np.float32)[:, None]
    features["inertia"] = inertias
    features["border"] = np.array(list(_border_dist_fast(mask)), dtype=np.float32)[:, None]

    feats_combined = np.concatenate(list(features.values()), axis=-1).astype(np.float32)
    assert feats_combined.shape[1] == fourier_pe.shape[1] + 7
    return coords, labels, feats_combined


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

# ── HOCT 2D adaptation (13D node features) ───────────────────────────
#
# HOCT paper ("Higher Order Cell Tracking", appendix "Input features"):
#   19D = [t, z, y, x, eq_diam, int_min, int_max, int_mean, int_std,
#          inertia_3x3(9), border_dist].
#
# 2D adaptation (19 → 13): drop z (planar data), replace 3×3 inertia
# with 2×2 (4 entries). t is kept as an explicit temporal feature
# (not a positional encoding); the 3D RoPE handles spatial positions.
# border_dist is Euclidean distance of the closest region pixel to the
# FoV edge (unclipped, non-inverted) — differs from the paper's
# centroid-based clipped inverse distance but is highly correlated.
#
# Standardization is handled downstream (per-fold z-score); this
# extractor returns RAW features.
#
# Feature order (HOCT2D_FEATURE_NAMES):
#   position (3):  time, centroid_y, centroid_x
#   size     (1):  eq_diam
#   intensity(4):  intensity_min, intensity_max, intensity_mean, intensity_std
#   inertia  (4):  inertia_00, inertia_01, inertia_10, inertia_11
#   border   (1):  border_dist

#: Ordered names of the 13 features returned by extract_hoct2d().
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


def extract_hoct2d(mask, img, frame_idx=0):
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


# ── HOCT 2D + Fourier PE (spatial positions encoded) ──────────

#: Ordered names of the features returned by extract_hoct2d_fourier().
#: time (1) + fourier_pe (2 * n_freqs * 2) + eq_diam (1) +
#: intensity (4) + inertia (4) + border (1).
HOCT2D_FOURIER_FEATURE_NAMES = [
    # time (1): frame index, kept as scalar (not Fourier-encoded)
    "time",
    # fourier_pe (32): sin/cos of (centroid_y, centroid_x) at 8 freqs
    *[f"fourier_pe_{i}" for i in range(32)],
    # size (1): diameter of a circle with the same area, in pixels
    "eq_diam",
    # intensity (4): within-region statistics of the normalized image [0, 1]
    "intensity_min", "intensity_max", "intensity_mean", "intensity_std",
    # inertia (4): 2x2 inertia tensor, row-major (I01 == I10 by symmetry)
    "inertia_00", "inertia_01", "inertia_10", "inertia_11",
    # border (1): min Euclidean distance of any region pixel to FoV edge
    "border_dist",
]


def extract_hoct2d_fourier(mask, img, frame_idx=0, n_freqs=8, cutoff=128.0):
    """Extract HOCT-style 13D node features with Fourier PE for spatial
    positions, replacing raw centroid coordinates.

    Same base features as ``extract_hoct2d`` but the spatial positions
    (centroid_y, centroid_x) are replaced with Fourier PE instead of
    raw coordinates.  The time feature ``t`` is kept as a scalar
    (not encoded with Fourier PE) because it provides a temporal
    ordering signal independent of spatial proximity.

    Layout:
      time       (1):  frame index
      fourier_pe (32): sin/cos of (centroid_y, centroid_x) at
                       8 geometrically decaying frequencies each
      eq_diam    (1):  equivalent diameter area
      intensity  (4):  intensity min, max, mean, std
      inertia    (4):  2x2 inertia tensor, row-major
      border     (1):  min distance to FoV edge

    Features are returned RAW (not standardized). Standardization is
    applied downstream, fit on the training split only.

    Args:
        mask: (H, W) integer label image.
        img:  (H, W) float intensity image (already normalized to [0, 1]).
        frame_idx: absolute frame index within the experiment.
        n_freqs: number of frequency components per spatial dimension.
        cutoff: controls the decay rate of the geometric frequency
            schedule.

    Returns:
        (coords, labels, features) with coords (N, 2), labels (N,),
        features (N, 43) float32 — or (None, None, None) if mask is empty.
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

    # 1. Time feature (1): scalar frame index, kept as-is
    time_col = np.full((len(df), 1), frame_idx, dtype=np.float32)

    # 2. Fourier PE of spatial positions (32): sin/cos of (y, x)
    spatial_pos = np.column_stack([
        df["centroid-0"].values.astype(np.float32),
        df["centroid-1"].values.astype(np.float32),
    ])
    pe = FourierPE(coord_dim=ndim, n_freqs=n_freqs, cutoff=cutoff)
    with torch.no_grad():
        fourier_pe = pe(torch.from_numpy(spatial_pos)).numpy().astype(np.float32)

    # 3. Size (1): equivalent_diameter_area
    eq_diam = df["equivalent_diameter_area"].values.astype(np.float32)[:, None]

    # 4. Intensity (4): min, max, mean, std
    intensity = np.column_stack([
        df["intensity_min"].values,
        df["intensity_max"].values,
        df["intensity_mean"].values,
        df["intensity_std"].values,
    ]).astype(np.float32)

    # 5. Inertia (4 for 2D): 2x2 tensor flattened
    inertias = np.stack(
        [np.column_stack([df[f"inertia_tensor-{i}-{j}"] for j in range(ndim)])
         for i in range(ndim)], axis=-1
    ).reshape(len(df), -1).astype(np.float32)

    # 6. Border (1): min distance to nearest FoV edge
    border_dists = _compute_border_dist(mask)[:, None]

    features = np.concatenate(
        [time_col, fourier_pe, eq_diam, intensity, inertias, border_dists],
        axis=-1,
    ).astype(np.float32)

    assert features.shape[1] == len(HOCT2D_FOURIER_FEATURE_NAMES), (
        f"HOCT Fourier feature count drifted: got {features.shape[1]}, "
        f"expected {len(HOCT2D_FOURIER_FEATURE_NAMES)}"
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


def load_cached_features(cond, exp, frame, feat_type, labels):
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


def save_cached_features(cond, exp, frame, feat_type, features, labels):
    """Save features to cache."""
    path = _cache_path(cond, exp, frame, feat_type)
    np.save(path, {"features": features, "labels": labels})
    return path
