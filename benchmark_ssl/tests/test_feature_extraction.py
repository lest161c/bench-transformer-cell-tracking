"""Smoke tests for feature extraction."""

import numpy as np
import pytest
from skimage.draw import disk

from src.data.feature_extraction import (
    EXTRACTORS,
    FEATURE_DIMS,
    extract,
    extract_basic,
    extract_hu,
    extract_patch,
    extract_shape,
    features_from_frame,
)


def _make_synthetic_mask(n_cells=5, img_size=128):
    """Create a synthetic mask with circular cells."""
    mask = np.zeros((img_size, img_size), dtype=np.int32)
    rng = np.random.RandomState(42)
    for i in range(1, n_cells + 1):
        cy, cx = rng.randint(20, img_size - 20, size=2)
        rr, cc = disk((cy, cx), radius=8, shape=mask.shape)
        mask[rr, cc] = i
    img = rng.rand(img_size, img_size).astype(np.float32) * 200 + 50
    return mask, img


def test_features_from_frame_regionprops2():
    mask, img = _make_synthetic_mask()
    result = features_from_frame(mask, img, properties="regionprops2")
    assert result is not None
    coords, labels, features = result
    assert coords.shape[0] == labels.shape[0]
    assert "equivalent_diameter_area" in features
    assert "border_dist" in features


def test_extract_basic():
    mask, img = _make_synthetic_mask()
    result = extract_basic(mask, img, ndim=2)
    assert result is not None
    coords, labels, features = result
    assert "eq_diam" in features
    assert "inertia" in features
    assert "border" in features


def test_extract_shape_adds_shape_descriptors():
    mask, img = _make_synthetic_mask()
    result = extract_shape(mask, img, ndim=2)
    assert result is not None
    coords, labels, features = result
    assert "eccentricity" in features
    assert "perimeter" in features
    assert "solidity" in features


def test_extract_hu_adds_hu_moments():
    mask, img = _make_synthetic_mask()
    result = extract_hu(mask, img, ndim=2)
    assert result is not None
    coords, labels, features = result
    for i in range(7):
        assert f"hu_{i}" in features


def test_extract_patch_adds_pca_features():
    mask, img = _make_synthetic_mask(n_cells=10)
    result = extract_patch(mask, img, ndim=2, patch_size=16, n_pca=4)
    assert result is not None
    coords, labels, features = result
    for i in range(4):
        assert f"patch_{i}" in features


def test_extract_dispatch_unknown_level_raises():
    mask, img = _make_synthetic_mask()
    with pytest.raises(ValueError, match="Unknown feature level"):
        extract("nonexistent", mask, img, ndim=2)


def test_feature_dims_consistent():
    for level, expected_dim in FEATURE_DIMS.items():
        assert level in EXTRACTORS
        assert isinstance(expected_dim, int)
