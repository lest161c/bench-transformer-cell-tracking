"""Tests for the standalone SAM2.1 feature row used by the KNN probe.

The pure helper tests run without the optional ``sam2`` dependency: they use
synthetic feature maps and check centroid mapping, clamping, nearest-patch
selection, and L2 normalization.  The module-import and checkpoint tests skip
when ``sam2`` is unavailable, so the default benchmark virtual environment
reports them as skipped rather than failing.
"""

import importlib.util

import numpy as np
import pytest

from src.analysis.sam21_probe_features import (
    SAM21_FEATURE_DIM,
    SAM21_MODEL_NAME,
    Sam21ProbeFeatureExtractor,
    l2_normalize_rows,
    map_centroids_to_feature_grid,
    pool_feature_map_at_centroids,
)


def _sam2_is_installed() -> bool:
    """Return whether the optional official ``sam2`` package can be located."""
    try:
        return importlib.util.find_spec("sam2") is not None
    except (ImportError, ValueError):
        return False


@pytest.mark.skipif(
    not _sam2_is_installed(),
    reason="optional SAM2.1 dependency is not installed",
)
def test_module_imports_and_reports_standalone_metadata() -> None:
    """The extractor metadata names the official model and no fork import."""
    metadata = Sam21ProbeFeatureExtractor.metadata()

    assert metadata["feature"] == "sam21"
    assert metadata["model"] == SAM21_MODEL_NAME
    assert metadata["feature_dim"] == SAM21_FEATURE_DIM
    assert "standalone official sam2" in metadata["implementation"]
    assert "no Trackastra fork import" in metadata["implementation"]


def test_map_centroids_to_feature_grid_maps_and_clamps() -> None:
    """Centroids scale into the model frame, map to patches, and clamp."""
    image_shape = (32, 64)
    feature_grid_shape = (2, 4)
    centroids = np.array(
        [
            [0.0, 0.0],
            [16.0, 32.0],  # exact patch boundary -> higher-index patch
            [31.9, 63.9],
            [100.0, -5.0],  # outside the frame -> clamped
        ],
        dtype=np.float32,
    )

    grid_indices = map_centroids_to_feature_grid(
        centroids, image_shape, feature_grid_shape
    )

    assert grid_indices.dtype == np.int64
    assert grid_indices.tolist() == [[0, 0], [1, 2], [1, 3], [1, 0]]


def test_pool_feature_map_selects_nearest_patch_without_averaging() -> None:
    """Pooling returns the exact feature vector at each mapped grid cell."""
    feature_map = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4)
    image_shape = (32, 64)
    # Cell centers in original-frame coordinates: (0, 0) and row 1 / column 2.
    centroids = np.array([[0.0, 0.0], [24.0, 40.0]], dtype=np.float32)

    pooled_features = pool_feature_map_at_centroids(
        feature_map, centroids, image_shape
    )

    assert pooled_features.shape == (2, 2)
    assert pooled_features.dtype == np.float32
    assert np.array_equal(pooled_features[0], feature_map[:, 0, 0])
    assert np.array_equal(pooled_features[1], feature_map[:, 1, 2])


def test_l2_normalize_rows_normalizes_and_keeps_zero_rows() -> None:
    """Non-zero rows become unit length and zero rows stay zero."""
    feature_matrix = np.array(
        [[3.0, 4.0], [0.0, 0.0], [1.0, -1.0]], dtype=np.float32
    )

    normalized_features = l2_normalize_rows(feature_matrix)

    assert normalized_features.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(normalized_features[:1], axis=1), [1.0], rtol=1e-6
    )
    assert np.array_equal(normalized_features[1], np.zeros(2, dtype=np.float32))
    np.testing.assert_allclose(
        normalized_features[2], [np.sqrt(0.5), -np.sqrt(0.5)], rtol=1e-6
    )


def test_extract_frame_is_deterministic_with_expected_shape() -> None:
    """A tiny synthetic image yields ``(N, 256)`` unit rows on repeated calls."""
    pytest.importorskip(
        "sam2", reason="optional SAM2.1 dependency is not installed"
    )

    frame = np.zeros((48, 64), dtype=np.float32)
    frame[16:32, 24:40] = 1.0
    centroids = np.array([[24.0, 32.0], [8.0, 8.0]], dtype=np.float32)
    extractor = Sam21ProbeFeatureExtractor(device="cpu")
    try:
        first_features = extractor.extract_frame(frame, centroids)
        second_features = extractor.extract_frame(frame, centroids)
    finally:
        extractor.close()

    assert first_features.shape == (2, SAM21_FEATURE_DIM)
    assert first_features.dtype == np.float32
    assert np.isfinite(first_features).all()
    np.testing.assert_allclose(
        np.linalg.norm(first_features, axis=1), np.ones(2), rtol=1e-5
    )
    assert np.array_equal(first_features, second_features)
