"""Tests for the position-only Fourier PE feature extractor (``posenc``)."""

import numpy as np

from src.edge_probing.harness.feature_extractors import (
    POSENC_DIM,
    extract_posenc,
)


def test_posenc_shape_and_dtype():
    """Default encoding has shape (N, 32) and float32 dtype."""
    coords = np.array([[10.0, 20.0], [30.5, 40.25], [100.0, 128.0]], dtype=np.float32)
    features = extract_posenc(coords)
    assert features.shape == (3, POSENC_DIM)
    assert features.shape == (3, 32)
    assert features.dtype == np.float32


def test_posenc_is_deterministic():
    """Repeated calls on the same input produce identical encodings."""
    coords = np.array([[5.0, 7.0], [64.0, 96.0]], dtype=np.float32)
    first = extract_posenc(coords)
    second = extract_posenc(coords)
    np.testing.assert_array_equal(first, second)


def test_posenc_values_are_finite():
    """Encodings of finite coordinates contain no NaN or infinity."""
    coords = np.array([[0.0, 0.0], [1e4, -1e4], [63.5, 127.5]], dtype=np.float32)
    features = extract_posenc(coords)
    assert np.isfinite(features).all()


def test_posenc_empty_input():
    """Empty (0, 2) input returns an empty (0, 32) encoding."""
    features = extract_posenc(np.zeros((0, 2), dtype=np.float32))
    assert features.shape == (0, POSENC_DIM)
    assert features.shape == (0, 32)
    assert features.dtype == np.float32


def test_posenc_distinguishes_different_coordinates():
    """Different coordinates produce different position encodings."""
    coords = np.array([[10.0, 20.0], [11.0, 20.0]], dtype=np.float32)
    features = extract_posenc(coords)
    assert not np.allclose(features[0], features[1])
