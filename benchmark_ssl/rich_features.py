"""Richer feature extraction for contrastive SSL signal analysis.

Proposal §4.4: Adds shape descriptors, Hu moments, and local image patch
features to the 7D regionprops baseline.

Feature sets:
  - basic:   [equivalent_diameter_area, intensity_mean, inertia_tensor(4), border_dist] = 7D
  - shape:   basic + eccentricity, perimeter, solidity, extent, axis_lengths, orientation = +6 → 13D
  - hu:      shape + 7 Hu moments = +7 → 20D
  - patch:   hu + local image patch (16-dim PCA projection) = +16 → 36D
"""

import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, regionprops as sk_regionprops
from skimage.transform import resize

logger = logging.getLogger(__name__)


# ─── regionprops property lists ────────────────────────────────────────────────


BASIC_PROPS = (
    "equivalent_diameter_area", "intensity_mean", "inertia_tensor",
)

SHAPE_PROPS = (
    "eccentricity", "perimeter", "solidity", "extent",
    "major_axis_length", "minor_axis_length", "orientation",
)

# Hu moments are returned as mu_01...mu_07 by regionprops when called
# with properties=("moments_hu",)


# ─── border distance (reused from ssl_pipeline) ───────────────────────────────


def _border_dist_fast(mask, cutoff=5):
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


# ─── feature extraction ────────────────────────────────────────────────────────


def extract_basic(mask, img, ndim):
    """7D regionprops2 baseline."""
    df = pd.DataFrame(
        regionprops_table(mask, intensity_image=img, properties=("label", "centroid", *BASIC_PROPS))
    )
    if len(df) == 0:
        return None
    coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)
    # inertia_tensor is 2x2 → 4 values
    inertias = np.stack(
        [np.column_stack([df[f"inertia_tensor-{i}-{j}"] for j in range(ndim)])
         for i in range(ndim)], axis=-1
    ).reshape(len(df), -1).astype(np.float32)
    features = OrderedDict()
    features["eq_diam"] = df["equivalent_diameter_area"].values.astype(np.float32)[:, None]
    features["intensity"] = df["intensity_mean"].values.astype(np.float32)[:, None]
    features["inertia"] = inertias
    features["border"] = np.array(list(_border_dist_fast(mask)), dtype=np.float32)[:, None]
    return coords, labels, features


def extract_shape(mask, img, ndim):
    """Basic + shape descriptors (13D)."""
    result = extract_basic(mask, img, ndim)
    if result is None:
        return None
    coords, labels, features = result
    df = pd.DataFrame(regionprops_table(mask, properties=("label", *SHAPE_PROPS)))
    features["eccentricity"] = df["eccentricity"].values.astype(np.float32)[:, None]
    features["perimeter"] = df["perimeter"].values.astype(np.float32)[:, None]
    features["solidity"] = df["solidity"].values.astype(np.float32)[:, None]
    features["extent"] = df["extent"].values.astype(np.float32)[:, None]
    features["axis_major"] = df["major_axis_length"].values.astype(np.float32)[:, None]
    features["axis_minor"] = df["minor_axis_length"].values.astype(np.float32)[:, None]
    features["orientation"] = df["orientation"].values.astype(np.float32)[:, None]
    return coords, labels, features


def extract_hu(mask, img, ndim):
    """Basic + shape + Hu moments (20D)."""
    result = extract_shape(mask, img, ndim)
    if result is None:
        return None
    coords, labels, features = result
    props_list = sk_regionprops(mask, intensity_image=img)
    hu_moments = np.array([r.moments_hu for r in props_list], dtype=np.float32)
    # log transform to make scale-invariant (moments hu are very small)
    hu_log = np.sign(hu_moments) * np.log1p(np.abs(hu_moments))
    for i in range(7):
        features[f"hu_{i}"] = hu_log[:, i:i + 1]
    return coords, labels, features


def extract_patch(mask, img, ndim, patch_size=32, n_pca=16):
    """Hu + local image patch features (20 + 16 = 36D after PCA).

    For each cell, extract a square crop centered on the centroid,
    resize to patch_size×patch_size, flatten, reduce via PCA.
    """
    result = extract_hu(mask, img, ndim)
    if result is None:
        return None
    coords, labels, features = result

    centroids = coords  # (N, ndim)
    h, w = mask.shape[-2:]
    patches = np.zeros((len(labels), patch_size * patch_size), dtype=np.float32)

    for i, (cy, cx) in enumerate(centroids):
        cy_int, cx_int = int(round(cy)), int(round(cx))
        half = patch_size // 2
        # crop from image
        y1 = max(0, cy_int - half)
        y2 = min(h, cy_int + half)
        x1 = max(0, cx_int - half)
        x2 = min(w, cx_int + half)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # resize to patch_size×patch_size
        crop_resized = resize(crop, (patch_size, patch_size), mode="reflect", anti_aliasing=True)
        patches[i] = crop_resized.ravel()

    # PCA projection to n_pca dims
    if len(labels) >= n_pca + 1:
        patch_mean = patches.mean(axis=0, keepdims=True)
        patches_centered = patches - patch_mean
        U, S, Vt = np.linalg.svd(patches_centered, full_matrices=False)
        patches_pca = patches_centered @ Vt[:n_pca].T
    else:
        patches_pca = patches[:, :n_pca]

    for i in range(n_pca):
        features[f"patch_{i}"] = patches_pca[:, i:i + 1]

    return coords, labels, features


# ─── dispatch ──────────────────────────────────────────────────────────────────


EXTRACTORS = {
    "basic": extract_basic,
    "shape": extract_shape,
    "hu": extract_hu,
    "patch": extract_patch,
}

FEATURE_DIMS = {
    "basic": 7,
    "shape": 13,
    "hu": 20,
    "patch": 36,
}

FEATURE_NAMES = {
    "basic": ["eq_diam", "intensity", "inertia_00", "inertia_01", "inertia_10", "inertia_11", "border_dist"],
    "shape": ["eq_diam", "intensity", "inertia_00", "inertia_01", "inertia_10", "inertia_11",
              "border_dist", "eccentricity", "perimeter", "solidity", "extent",
              "axis_major", "axis_minor", "orientation"],
    "hu": ["eq_diam", "intensity", "inertia_00", "inertia_01", "inertia_10", "inertia_11",
           "border_dist", "eccentricity", "perimeter", "solidity", "extent",
           "axis_major", "axis_minor", "orientation",
           "hu_0", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6"],
    "patch": None,  # dynamic, determined at runtime
}


def extract(level: str, mask, img, ndim=2, patch_size=32, n_pca=16):
    """Extract features at the specified richness level.

    Returns (coords, labels, features: OrderedDict) or None.
    """
    if level not in EXTRACTORS:
        raise ValueError(f"Unknown feature level: {level}. Choose from {list(EXTRACTORS.keys())}")
    if level == "patch":
        return extract_patch(mask, img, ndim, patch_size, n_pca)
    return EXTRACTORS[level](mask, img, ndim)
