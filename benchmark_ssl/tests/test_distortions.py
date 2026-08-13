"""Smoke tests for distortion families and DistortionPipeline."""

import numpy as np
import pytest

from src.distortions import (
    AffineDistortion,
    DropoutDistortion,
    DistortionPipeline,
    ElasticDistortion,
    FeatureNoise,
    JitterDistortion,
    PhotometricDistortion,
)


def _make_frame(n_cells=20, ndim=2, seed=42):
    rng = np.random.RandomState(seed)
    coords = rng.randn(n_cells, ndim).astype(np.float32) * 50 + 128
    features = {
        "equivalent_diameter_area": rng.rand(n_cells).astype(np.float32) * 10 + 5,
        "intensity_mean": rng.rand(n_cells).astype(np.float32) * 200 + 50,
        "inertia_tensor": rng.rand(n_cells, 4).astype(np.float32),
        "border_dist": rng.rand(n_cells).astype(np.float32) * 5,
    }
    labels = np.arange(1, n_cells + 1, dtype=np.int32)
    return coords, features, labels


def test_affine_distortion_preserves_cell_count():
    coords, features, labels = _make_frame()
    dist = AffineDistortion(rng=np.random.RandomState(0))
    coords_t, feats_t = dist(coords, features, labels)
    assert coords_t.shape == coords.shape
    assert len(feats_t) == len(features)


def test_jitter_distortion_preserves_cell_count():
    coords, features, labels = _make_frame()
    dist = JitterDistortion(rng=np.random.RandomState(0))
    coords_t, feats_t = dist(coords, features, labels)
    assert coords_t.shape == coords.shape


def test_dropout_distortion_reduces_cell_count():
    coords, features, labels = _make_frame()
    dist = DropoutDistortion(p_drop=(0.5, 0.5), rng=np.random.RandomState(0))
    coords_t, feats_t = dist(coords, features, labels)
    assert coords_t.shape[0] < coords.shape[0]


def test_distortion_pipeline_produces_two_views():
    coords, features, labels = _make_frame()
    pipeline = DistortionPipeline.from_config({
        "distortions": ["jitter"],
        "jitter": {"std": [2, 8], "p_cell_jitter": 0.8},
        "seed": 42,
    })
    coords1, feats1, labels1, coords2, feats2, labels2 = pipeline(coords, features, labels)
    assert coords1.shape == coords.shape
    assert coords2.shape == coords.shape


def test_distortion_pipeline_unknown_distortion_raises():
    with pytest.raises(ValueError, match="Unknown distortion"):
        DistortionPipeline.from_config({"distortions": ["nonexistent"]})
