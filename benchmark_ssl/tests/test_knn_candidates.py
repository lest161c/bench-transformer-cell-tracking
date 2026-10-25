"""Unit tests for coordinate-based K-NN candidate construction (SPEC 0046 F02).

The module is loaded directly from its file instead of via
``src.analysis.knn_candidates`` because ``src/analysis/__init__.py`` eagerly
imports diagnostic scripts that load DINOv2 at import time.  Loading by path
keeps these tests offline and independent of unrelated analysis modules.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "analysis" / "knn_candidates.py"
)


def _load_knn_candidates_module():
    """Load ``knn_candidates.py`` by path, bypassing the analysis package init."""
    module_spec = importlib.util.spec_from_file_location(
        "knn_candidates_under_test", MODULE_PATH
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


knn_candidates = _load_knn_candidates_module()


def _make_synthetic_candidates():
    """Return anchors and candidates whose true matches are within the top 3."""
    anchor_coords = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 5.0]])
    candidate_coords = np.array(
        [
            [1.0, 0.0],
            [10.0, 11.0],
            [19.0, 5.0],
            [0.5, 0.0],
            [10.0, 10.5],
            [20.0, 4.5],
        ]
    )
    anchor_labels = np.array([7, 8, 9], dtype=np.int64)
    candidate_labels = np.array([7, 8, 9, 99, 98, 97], dtype=np.int64)
    return anchor_coords, candidate_coords, anchor_labels, candidate_labels


def test_target_index_points_to_true_match():
    """The true match is retained and target_index selects the correct row."""
    anchor_coords, candidate_coords, anchor_labels, candidate_labels = (
        _make_synthetic_candidates()
    )
    anchor_features = np.arange(12, dtype=np.float32).reshape(3, 4)
    candidate_features = np.arange(24, dtype=np.float32).reshape(6, 4)

    result = knn_candidates.build_knn_candidate_set(
        anchor_features,
        candidate_features,
        anchor_coords,
        candidate_coords,
        anchor_labels,
        candidate_labels,
        k=3,
    )

    assert result["target_index"].tolist() == [1, 1, 1]
    assert result["target_index"].dtype == np.int64
    for anchor_position in range(3):
        target_index = result["target_index"][anchor_position]
        assert (
            result["candidate_labels"][anchor_position, target_index]
            == result["anchor_labels"][anchor_position]
        )
        np.testing.assert_array_equal(
            result["anchor_features"][anchor_position],
            np.repeat(
                anchor_features[anchor_position][np.newaxis, :], 3, axis=0
            ),
        )


def test_equal_distances_break_ties_by_label_then_index():
    """Exact distance ties order candidates by label, then original index."""
    anchor_coords = np.array([[0.0, 0.0]])
    candidate_coords = np.array(
        [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    anchor_labels = np.array([3], dtype=np.int64)
    candidate_labels = np.array([3, 9, 3, 3], dtype=np.int64)

    kwargs = dict(
        anchor_features=np.zeros((1, 2), dtype=np.float32),
        candidate_features=np.zeros((4, 2), dtype=np.float32),
        anchor_coords=anchor_coords,
        candidate_coords=candidate_coords,
        anchor_labels=anchor_labels,
        candidate_labels=candidate_labels,
        k=4,
    )
    first_result = knn_candidates.build_knn_candidate_set(**kwargs)
    second_result = knn_candidates.build_knn_candidate_set(**kwargs)

    assert first_result["candidate_labels"].tolist() == [[3, 3, 3, 9]]
    np.testing.assert_array_equal(
        first_result["candidate_coords"][0],
        np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]),
    )
    assert first_result["target_index"].tolist() == [0]
    np.testing.assert_array_equal(
        first_result["candidate_coords"], second_result["candidate_coords"]
    )
    np.testing.assert_array_equal(
        first_result["candidate_labels"], second_result["candidate_labels"]
    )
    np.testing.assert_array_equal(
        first_result["target_index"], second_result["target_index"]
    )


def test_features_never_influence_candidate_selection():
    """Different appearance features leave the coordinate-based ranking unchanged."""
    anchor_coords = np.array([[0.0, 0.0], [5.0, 5.0]])
    candidate_coords = np.array(
        [[0.2, 0.2], [5.2, 5.2], [2.0, 2.0]]
    )
    anchor_labels = np.array([1, 2], dtype=np.int64)
    candidate_labels = np.array([1, 2, 9], dtype=np.int64)

    first_features = np.random.default_rng(11).normal(size=(3, 5)).astype(np.float32)
    second_features = np.random.default_rng(29).normal(size=(3, 5)).astype(np.float32)
    first_result = knn_candidates.build_knn_candidate_set(
        np.random.default_rng(3).normal(size=(2, 5)).astype(np.float32),
        first_features,
        anchor_coords,
        candidate_coords,
        anchor_labels,
        candidate_labels,
        k=2,
    )
    second_result = knn_candidates.build_knn_candidate_set(
        np.random.default_rng(4).normal(size=(2, 5)).astype(np.float32),
        second_features,
        anchor_coords,
        candidate_coords,
        anchor_labels,
        candidate_labels,
        k=2,
    )

    for key in (
        "candidate_coords",
        "candidate_labels",
        "target_index",
        "anchor_coords",
        "anchor_labels",
    ):
        np.testing.assert_array_equal(first_result[key], second_result[key])
    assert not np.array_equal(
        first_result["candidate_features"], second_result["candidate_features"]
    )


def test_anchor_without_true_match_is_dropped_and_counted():
    """An anchor whose label is absent from the candidates is omitted."""
    anchor_coords = np.array([[0.0, 0.0], [50.0, 50.0]])
    candidate_coords = np.array([[0.0, 0.0], [100.0, 100.0]])
    anchor_labels = np.array([1, 2], dtype=np.int64)
    candidate_labels = np.array([1, 3], dtype=np.int64)

    result = knn_candidates.build_knn_candidate_set(
        np.ones((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        anchor_coords,
        candidate_coords,
        anchor_labels,
        candidate_labels,
        k=2,
    )

    assert result["n_anchors"] == 1
    assert result["dropped_missing_match"] == 1
    assert result["dropped_short_candidate_list"] == 0
    assert result["anchor_labels"].tolist() == [1]
    assert result["target_index"].tolist() == [0]
    assert result["candidate_labels"].tolist() == [[1, 3]]
    assert result["anchor_features"].shape == (1, 2, 3)


def test_k_larger_than_candidate_count_drops_anchors_without_padding():
    """When k exceeds the frame size, anchors are dropped rather than padded."""
    anchor_coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    candidate_coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    anchor_labels = np.array([1, 2], dtype=np.int64)
    candidate_labels = np.array([1, 2], dtype=np.int64)

    result = knn_candidates.build_knn_candidate_set(
        np.ones((2, 5), dtype=np.float32),
        np.ones((2, 5), dtype=np.float32),
        anchor_coords,
        candidate_coords,
        anchor_labels,
        candidate_labels,
        k=3,
    )

    assert result["k"] == 3
    assert result["n_anchors"] == 0
    assert result["dropped_missing_match"] == 0
    assert result["dropped_short_candidate_list"] == 2
    assert result["anchor_features"].shape == (0, 3, 5)
    assert result["candidate_features"].shape == (0, 3, 5)
    assert result["target_index"].shape == (0,)
    assert result["target_index"].dtype == np.int64
    assert result["anchor_labels"].shape == (0,)
    assert result["candidate_labels"].shape == (0, 3)
    assert result["anchor_coords"].shape == (0, 2)
    assert result["candidate_coords"].shape == (0, 3, 2)


def test_coverage_reaches_one_at_known_k_and_choose_k_returns_it():
    """Coverage rises to 1.0 at K=3 and choose_k selects that smallest K."""
    anchor_coords = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    candidate_coords = np.array(
        [
            [0.0, 1.0],
            [100.0, 2.0],
            [100.0, 1.0],
            [200.0, 3.0],
            [200.0, 2.0],
            [200.0, 1.0],
        ]
    )
    anchor_labels = np.array([10, 20, 30], dtype=np.int64)
    candidate_labels = np.array([10, 20, 99, 30, 98, 97], dtype=np.int64)

    coverage_rows = knn_candidates.coverage_vs_k(
        anchor_coords,
        candidate_coords,
        anchor_labels,
        candidate_labels,
        k_max=6,
    )

    assert [row["k"] for row in coverage_rows] == [1, 2, 3, 4, 5, 6]
    assert [row["coverage"] for row in coverage_rows[:3]] == [
        pytest.approx(1.0 / 3.0),
        pytest.approx(2.0 / 3.0),
        1.0,
    ]
    assert all(row["n_anchors"] == 3 for row in coverage_rows)
    assert all(row["n_missing_match"] == 0 for row in coverage_rows)
    assert knn_candidates.choose_k(coverage_rows) == 3
    assert knn_candidates.choose_k(coverage_rows[:2]) == 2


def test_empty_inputs_return_documented_empty_shapes():
    """Empty anchor or candidate inputs return zero-length arrays with k retained."""
    no_anchor_result = knn_candidates.build_knn_candidate_set(
        np.empty((0, 3), dtype=np.float32),
        np.ones((4, 3), dtype=np.float32),
        np.empty((0, 2), dtype=np.float64),
        np.arange(8, dtype=np.float64).reshape(4, 2),
        np.empty((0,), dtype=np.int64),
        np.array([1, 2, 3, 4], dtype=np.int64),
        k=2,
    )

    assert no_anchor_result["anchor_features"].shape == (0, 2, 3)
    assert no_anchor_result["candidate_features"].shape == (0, 2, 3)
    assert no_anchor_result["target_index"].shape == (0,)
    assert no_anchor_result["anchor_labels"].shape == (0,)
    assert no_anchor_result["candidate_labels"].shape == (0, 2)
    assert no_anchor_result["anchor_coords"].shape == (0, 2)
    assert no_anchor_result["candidate_coords"].shape == (0, 2, 2)
    assert no_anchor_result["k"] == 2
    assert no_anchor_result["n_anchors"] == 0
    assert no_anchor_result["dropped_missing_match"] == 0
    assert no_anchor_result["dropped_short_candidate_list"] == 0

    no_candidate_result = knn_candidates.build_knn_candidate_set(
        np.ones((2, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.empty((0, 2), dtype=np.float64),
        np.array([1, 2], dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        k=2,
    )

    assert no_candidate_result["anchor_features"].shape == (0, 2, 3)
    assert no_candidate_result["candidate_labels"].shape == (0, 2)
    assert no_candidate_result["candidate_coords"].shape == (0, 2, 2)
    assert no_candidate_result["dropped_short_candidate_list"] == 2
    assert (
        knn_candidates.coverage_vs_k(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.empty((0, 2), dtype=np.float64),
            np.array([1, 2], dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
        == []
    )


def test_torch_inputs_are_accepted_and_return_numpy_arrays():
    """Torch tensors are converted to NumPy, including int64 target indices."""
    anchor_coords, candidate_coords, anchor_labels, candidate_labels = (
        _make_synthetic_candidates()
    )

    result = knn_candidates.build_knn_candidate_set(
        torch.zeros((3, 4), dtype=torch.float32),
        torch.zeros((6, 4), dtype=torch.float32),
        torch.as_tensor(anchor_coords, dtype=torch.float64),
        torch.as_tensor(candidate_coords, dtype=torch.float64),
        torch.as_tensor(anchor_labels, dtype=torch.int64),
        torch.as_tensor(candidate_labels, dtype=torch.int64),
        k=3,
    )

    for key in (
        "anchor_features",
        "candidate_features",
        "target_index",
        "anchor_labels",
        "candidate_labels",
        "anchor_coords",
        "candidate_coords",
    ):
        assert isinstance(result[key], np.ndarray), key
    assert result["target_index"].dtype == np.int64
    assert result["target_index"].tolist() == [1, 1, 1]
