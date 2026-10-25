"""K-nearest-neighbour candidate construction and coverage diagnostics.

This module implements the candidate-building step of the Phase-1 K-NN probe
in SPEC 0046.  Candidates are selected with cell-centroid geometry only:
features must never influence which cells enter a candidate set.  Keeping the
geometry and appearance paths separate prevents feature leakage into candidate
construction and lets the coverage-vs-K diagnostic run without extracting any
appearance features.

All results are returned as NumPy arrays, even when the inputs are PyTorch
tensors.  Candidate construction is data preparation rather than training, so
keeping it framework-neutral avoids an unnecessary dependency on a particular
device or autograd state.  Downstream probe code can convert the arrays to
torch tensors once, after normalization.

Terminology:
    ``D``: feature dimensionality, i.e. the second axis of a feature matrix.
    ``k``: number of nearest candidates retained per anchor.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

NumpyOrTorchArray = np.ndarray | torch.Tensor


def _to_numpy(array: NumpyOrTorchArray, *, name: str) -> np.ndarray:
    """Convert a NumPy array or PyTorch tensor to a detached CPU NumPy array.

    Called by the public functions so their contracts can accept either
    framework without duplicating conversion logic.  The original object is
    not modified.

    Args:
        array: NumPy array, PyTorch tensor, or array-like object accepted by
            ``numpy.asarray``.
        name: Argument name used in error messages.

    Returns:
        A NumPy array sharing or copying the input data as required by the
        conversion.

    Raises:
        TypeError: If the converted object is not an array.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    try:
        return np.asarray(array)
    except Exception as conversion_error:
        raise TypeError(f"{name} could not be converted to a NumPy array") from conversion_error


def _as_feature_matrix(features: NumpyOrTorchArray, *, name: str) -> np.ndarray:
    """Validate and return a two-dimensional feature matrix.

    Args:
        features: Array-like of shape ``(N, D)`` in NumPy or torch form.
        name: Argument name used in error messages.

    Returns:
        The feature matrix as a NumPy array with shape ``(N, D)``.

    Raises:
        ValueError: If the array is not two-dimensional.
    """
    feature_array = _to_numpy(features, name=name)
    if feature_array.ndim != 2:
        raise ValueError(
            f"{name} must have shape (N, D), got shape {feature_array.shape}"
        )
    return feature_array


def _as_coordinate_matrix(
    coordinates: NumpyOrTorchArray,
    *,
    name: str,
    expected_rows: int,
) -> np.ndarray:
    """Validate and return a two-dimensional ``(y, x)`` coordinate matrix.

    The Euclidean distance used here is rotation-invariant, so the ``(y, x)``
    order does not change ranking; it is documented for callers because the
    rest of the pipeline uses image coordinates.

    Args:
        coordinates: Array-like of shape ``(N, 2)`` in NumPy or torch form.
        name: Argument name used in error messages.
        expected_rows: Required number of rows, determined by the matching
            feature or label array.

    Returns:
        Coordinate array with shape ``(N, 2)`` and the original numeric dtype.

    Raises:
        ValueError: If the shape is not ``(expected_rows, 2)``.
    """
    coordinate_array = _to_numpy(coordinates, name=name)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (N, 2), got shape {coordinate_array.shape}"
        )
    if coordinate_array.shape[0] != expected_rows:
        raise ValueError(
            f"{name} has {coordinate_array.shape[0]} rows but expected {expected_rows}"
        )
    return coordinate_array


def _as_label_vector(
    labels: NumpyOrTorchArray,
    *,
    name: str,
    expected_rows: int | None,
) -> np.ndarray:
    """Validate and return integer labels as an ``int64`` vector.

    Args:
        labels: Array-like of shape ``(N,)`` in NumPy or torch form.
        name: Argument name used in error messages.
        expected_rows: Required number of labels, or ``None`` when the caller
            only needs validation and the row count.

    Returns:
        One-dimensional ``int64`` label array.

    Raises:
        TypeError: If the labels are not integer-valued.
        ValueError: If the shape or row count is wrong.
    """
    label_array = _to_numpy(labels, name=name)
    if label_array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got shape {label_array.shape}")
    if expected_rows is not None and label_array.shape[0] != expected_rows:
        raise ValueError(
            f"{name} has {label_array.shape[0]} entries but expected {expected_rows}"
        )
    if not np.issubdtype(label_array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer labels, got {label_array.dtype}")
    return label_array.astype(np.int64, copy=False)


def _compute_coordinate_distances(
    anchor_coordinates: np.ndarray,
    candidate_coordinates: np.ndarray,
) -> np.ndarray:
    """Compute the Euclidean distance from every anchor to every candidate.

    Coordinates are promoted to ``float64`` for the distance computation so
    integer centroid coordinates cannot overflow and ties are reproducible on
    ordinary hardware.  ``(y, x)`` axis order is irrelevant to the norm.

    Args:
        anchor_coordinates: ``(N_anchor, 2)`` anchor centroids.
        candidate_coordinates: ``(N_candidate, 2)`` candidate centroids.

    Returns:
        Distance matrix of shape ``(N_anchor, N_candidate)``.
    """
    anchor_as_float = anchor_coordinates.astype(np.float64, copy=False)
    candidate_as_float = candidate_coordinates.astype(np.float64, copy=False)
    differences = anchor_as_float[:, np.newaxis, :] - candidate_as_float[np.newaxis, :, :]
    return np.sqrt(np.sum(differences * differences, axis=-1))


def _rank_candidates_by_coordinate_distance(
    distances_to_candidates: np.ndarray,
    candidate_labels: np.ndarray,
) -> np.ndarray:
    """Return candidate indices ordered by distance, label, then index.

    Called by ``build_knn_candidate_set`` and ``coverage_vs_k`` so both use
    exactly the same deterministic candidate order.  The label and index keys
    only break exact distance ties, which makes the ordering independent of the
    input candidate order for distinct labels and stable for duplicate labels.

    Args:
        distances_to_candidates: ``(N_candidate,)`` distances from one anchor.
        candidate_labels: ``(N_candidate,)`` integer candidate labels.

    Returns:
        ``int64`` array of original candidate indices in ranked order.
    """
    candidate_indices = np.arange(candidate_labels.shape[0], dtype=np.int64)
    return np.lexsort(
        (candidate_indices, candidate_labels, distances_to_candidates)
    ).astype(np.int64, copy=False)


def _find_true_match_rank(
    ordered_candidate_labels: np.ndarray,
    anchor_label: int,
) -> int | None:
    """Find the rank of the true match in an ordered candidate list.

    An anchor matches the first candidate in ranked order whose label equals
    the anchor label.  Tracklet labels are unique within a frame in normal
    data, so the first match is the true match.  For malformed duplicate labels
    the earliest ranked match is still chosen deterministically.

    Args:
        ordered_candidate_labels: Candidate labels in ranked order.
        anchor_label: Tracklet label of the anchor.

    Returns:
        Zero-based rank of the true match, or ``None`` when no candidate has
        that label.
    """
    matching_ranks = np.flatnonzero(ordered_candidate_labels == anchor_label)
    if matching_ranks.size == 0:
        return None
    return int(matching_ranks[0])


def _empty_candidate_set(
    *,
    k: int,
    feature_dimension: int,
    anchor_feature_dtype: np.dtype[Any],
    candidate_feature_dtype: np.dtype[Any],
    anchor_coordinate_dtype: np.dtype[Any],
    candidate_coordinate_dtype: np.dtype[Any],
    n_anchors: int,
    dropped_missing_match: int,
    dropped_short_candidate_list: int,
) -> dict[str, Any]:
    """Build the empty result dictionary for ``build_knn_candidate_set``.

    Called when no anchor survives candidate construction.  Keeping this in one
    place guarantees that every empty output uses the same trailing shapes
    ``(0, k, D)``, ``(0, k)`` and ``(0, k, 2)``.

    Args:
        k: Requested number of candidates per anchor.
        feature_dimension: Feature dimensionality ``D``.
        anchor_feature_dtype: NumPy dtype for the empty anchor features.
        candidate_feature_dtype: NumPy dtype for the empty candidate features.
        anchor_coordinate_dtype: NumPy dtype for the empty anchor coordinates.
        candidate_coordinate_dtype: NumPy dtype for the empty candidate coordinates.
        n_anchors: Number of retained anchors, which is zero.
        dropped_missing_match: Number of anchors dropped because their true
            match was absent from the selected candidates.
        dropped_short_candidate_list: Number of anchors dropped because fewer
            than ``k`` candidates existed.

    Returns:
        Result dictionary with the same keys as a non-empty result.
    """
    return {
        "anchor_features": np.empty(
            (0, k, feature_dimension), dtype=anchor_feature_dtype
        ),
        "candidate_features": np.empty(
            (0, k, feature_dimension), dtype=candidate_feature_dtype
        ),
        "target_index": np.empty((0,), dtype=np.int64),
        "anchor_labels": np.empty((0,), dtype=np.int64),
        "candidate_labels": np.empty((0, k), dtype=np.int64),
        "anchor_coords": np.empty((0, 2), dtype=anchor_coordinate_dtype),
        "candidate_coords": np.empty(
            (0, k, 2), dtype=candidate_coordinate_dtype
        ),
        "k": int(k),
        "n_anchors": int(n_anchors),
        "dropped_missing_match": int(dropped_missing_match),
        "dropped_short_candidate_list": int(dropped_short_candidate_list),
    }


def build_knn_candidate_set(
    anchor_features: NumpyOrTorchArray,
    candidate_features: NumpyOrTorchArray,
    anchor_coords: NumpyOrTorchArray,
    candidate_coords: NumpyOrTorchArray,
    anchor_labels: NumpyOrTorchArray,
    candidate_labels: NumpyOrTorchArray,
    k: int,
) -> dict[str, Any]:
    """Build per-anchor K-nearest candidate tensors for the K-NN probe.

    Candidate selection is strictly coordinate-based: for each anchor all
    candidates are ranked by Euclidean centroid distance with deterministic
    tie-breaks on ``(distance, candidate_label, candidate_index)``.  Feature
    values are copied into the output but never inspected while choosing
    candidates.

    The true match is the candidate whose label equals the anchor label.  An
    anchor is dropped and counted in ``dropped_missing_match`` when that label
    is absent from the candidate frame or when the true match does not fall in
    the selected K nearest.  The latter case can only occur when ``k`` is below
    the coverage-optimal K; the probe should choose ``k`` with
    ``coverage_vs_k`` first.  Dropping rather than forcing the match preserves
    the meaning of candidate coverage at the residual-miss ``K_max`` described
    in SPEC 0046.

    If ``k`` exceeds the number of candidates, every anchor is dropped and
    counted in ``dropped_short_candidate_list`` instead of being padded.  This
    lets the K coverage diagnostic handle small frames without fabricating
    candidates.

    Args:
        anchor_features: ``(N_anchor, D)`` anchor appearance features.
        candidate_features: ``(N_candidate, D)`` candidate appearance features.
        anchor_coords: ``(N_anchor, 2)`` anchor centroids in ``(y, x)`` order.
        candidate_coords: ``(N_candidate, 2)`` candidate centroids in
            ``(y, x)`` order.
        anchor_labels: ``(N_anchor,)`` integer anchor tracklet labels.
        candidate_labels: ``(N_candidate,)`` integer candidate tracklet labels.
        k: Positive number of nearest candidates to retain per anchor.

    Returns:
        Dictionary with NumPy arrays:
            ``anchor_features``: ``(N_kept, k, D)`` anchor features tiled along
            the candidate axis.
            ``candidate_features``: ``(N_kept, k, D)`` selected candidate
            features in deterministic rank order.
            ``target_index``: ``(N_kept,)`` ``int64`` true-match rank in
            ``[0, k)``.
            ``anchor_labels``: ``(N_kept,)`` ``int64`` labels of retained
            anchors.
            ``candidate_labels``: ``(N_kept, k)`` ``int64`` selected labels.
            ``anchor_coords``: ``(N_kept, 2)`` retained anchor coordinates.
            ``candidate_coords``: ``(N_kept, k, 2)`` selected candidate
            coordinates.
            ``k``: requested candidate count.
            ``n_anchors``: number of retained anchors, equal to ``N_kept``.
            ``dropped_missing_match``: anchors without a selected true match.
            ``dropped_short_candidate_list``: anchors dropped because
            ``k > N_candidate``.
        If no anchor survives, arrays have leading dimension zero with the
        documented trailing shapes and ``k`` is unchanged.

    Raises:
        TypeError: If labels are not integer-valued.
        ValueError: If input shapes are inconsistent or ``k`` is not positive.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be a positive integer, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = int(k)

    anchor_feature_array = _as_feature_matrix(anchor_features, name="anchor_features")
    candidate_feature_array = _as_feature_matrix(
        candidate_features, name="candidate_features"
    )
    if candidate_feature_array.shape[1] != anchor_feature_array.shape[1]:
        raise ValueError(
            "anchor_features and candidate_features must have the same D, got "
            f"{anchor_feature_array.shape[1]} and {candidate_feature_array.shape[1]}"
        )

    n_anchors_input = anchor_feature_array.shape[0]
    n_candidates = candidate_feature_array.shape[0]
    feature_dimension = anchor_feature_array.shape[1]

    anchor_coordinate_array = _as_coordinate_matrix(
        anchor_coords, name="anchor_coords", expected_rows=n_anchors_input
    )
    candidate_coordinate_array = _as_coordinate_matrix(
        candidate_coords, name="candidate_coords", expected_rows=n_candidates
    )
    anchor_label_array = _as_label_vector(
        anchor_labels, name="anchor_labels", expected_rows=n_anchors_input
    )
    candidate_label_array = _as_label_vector(
        candidate_labels, name="candidate_labels", expected_rows=n_candidates
    )

    distances = _compute_coordinate_distances(
        anchor_coordinate_array, candidate_coordinate_array
    )

    retained_anchor_features: list[np.ndarray] = []
    retained_candidate_features: list[np.ndarray] = []
    retained_target_indices: list[int] = []
    retained_anchor_labels: list[int] = []
    retained_candidate_labels: list[np.ndarray] = []
    retained_anchor_coordinates: list[np.ndarray] = []
    retained_candidate_coordinates: list[np.ndarray] = []
    dropped_missing_match = 0
    dropped_short_candidate_list = 0

    for anchor_position in range(n_anchors_input):
        if n_candidates < k:
            dropped_short_candidate_list += 1
            continue

        ranked_candidate_indices = _rank_candidates_by_coordinate_distance(
            distances[anchor_position], candidate_label_array
        )
        selected_candidate_indices = ranked_candidate_indices[:k]
        target_index = _find_true_match_rank(
            candidate_label_array[selected_candidate_indices],
            int(anchor_label_array[anchor_position]),
        )
        if target_index is None:
            dropped_missing_match += 1
            continue

        retained_anchor_features.append(
            np.repeat(
                anchor_feature_array[anchor_position][np.newaxis, :],
                k,
                axis=0,
            )
        )
        retained_candidate_features.append(
            candidate_feature_array[selected_candidate_indices]
        )
        retained_target_indices.append(target_index)
        retained_anchor_labels.append(int(anchor_label_array[anchor_position]))
        retained_candidate_labels.append(
            candidate_label_array[selected_candidate_indices]
        )
        retained_anchor_coordinates.append(anchor_coordinate_array[anchor_position])
        retained_candidate_coordinates.append(
            candidate_coordinate_array[selected_candidate_indices]
        )

    if not retained_anchor_features:
        return _empty_candidate_set(
            k=k,
            feature_dimension=feature_dimension,
            anchor_feature_dtype=anchor_feature_array.dtype,
            candidate_feature_dtype=candidate_feature_array.dtype,
            anchor_coordinate_dtype=anchor_coordinate_array.dtype,
            candidate_coordinate_dtype=candidate_coordinate_array.dtype,
            n_anchors=0,
            dropped_missing_match=dropped_missing_match,
            dropped_short_candidate_list=dropped_short_candidate_list,
        )

    n_anchors_retained = len(retained_anchor_features)
    return {
        "anchor_features": np.stack(retained_anchor_features, axis=0),
        "candidate_features": np.stack(retained_candidate_features, axis=0),
        "target_index": np.asarray(retained_target_indices, dtype=np.int64),
        "anchor_labels": np.asarray(retained_anchor_labels, dtype=np.int64),
        "candidate_labels": np.stack(retained_candidate_labels, axis=0),
        "anchor_coords": np.stack(retained_anchor_coordinates, axis=0),
        "candidate_coords": np.stack(retained_candidate_coordinates, axis=0),
        "k": k,
        "n_anchors": int(n_anchors_retained),
        "dropped_missing_match": int(dropped_missing_match),
        "dropped_short_candidate_list": int(dropped_short_candidate_list),
    }


def coverage_vs_k(
    anchor_coords: NumpyOrTorchArray,
    candidate_coords: NumpyOrTorchArray,
    anchor_labels: NumpyOrTorchArray,
    candidate_labels: NumpyOrTorchArray,
    k_max: int = 64,
) -> list[dict[str, int | float]]:
    """Measure true-match coverage for every candidate count up to ``k_max``.

    The diagnostic ranks candidates by coordinate distance with the same
    deterministic ``(distance, candidate_label, candidate_index)`` order as
    ``build_knn_candidate_set``.  For each ``k`` it reports the fraction of
    anchors whose true match is among the unforced K nearest.  An anchor whose
    true-match label is absent from the candidate frame is excluded from the
    coverage denominator and counted separately in ``n_missing_match``.

    Feature similarity is never used.  This function therefore permits K
    selection on a training split before appearance features are extracted.

    Args:
        anchor_coords: ``(N_anchor, 2)`` anchor centroids in ``(y, x)`` order.
        candidate_coords: ``(N_candidate, 2)`` candidate centroids in
            ``(y, x)`` order.
        anchor_labels: ``(N_anchor,)`` integer anchor tracklet labels.
        candidate_labels: ``(N_candidate,)`` integer candidate tracklet labels.
        k_max: Largest candidate count to evaluate.  The returned rows stop at
            ``min(k_max, N_candidate)``.

    Returns:
        Rows for ``k = 1 .. min(k_max, N_candidate)``.  Each row contains:
            ``k``: candidate count.
            ``coverage``: ``n_covered / n_anchors``, or ``0.0`` when no anchor
            has a true match.
            ``n_anchors``: anchors whose true-match label exists among the
            candidates.
            ``n_covered``: such anchors whose true match is within the K
            nearest.
            ``n_missing_match``: anchors with no candidate label equal to their
            own label.
        If there are no candidates, the list is empty.

    Raises:
        TypeError: If labels are not integer-valued.
        ValueError: If input shapes are inconsistent or ``k_max`` is not
            positive.
    """
    if isinstance(k_max, bool) or not isinstance(k_max, (int, np.integer)):
        raise TypeError(
            f"k_max must be a positive integer, got {type(k_max).__name__}"
        )
    if k_max <= 0:
        raise ValueError(f"k_max must be positive, got {k_max}")
    k_max = int(k_max)

    anchor_label_array = _as_label_vector(
        anchor_labels,
        name="anchor_labels",
        expected_rows=None,
    )
    n_anchors = anchor_label_array.shape[0]
    candidate_label_array = _as_label_vector(
        candidate_labels,
        name="candidate_labels",
        expected_rows=None,
    )
    n_candidates = candidate_label_array.shape[0]

    anchor_coordinate_array = _as_coordinate_matrix(
        anchor_coords, name="anchor_coords", expected_rows=n_anchors
    )
    candidate_coordinate_array = _as_coordinate_matrix(
        candidate_coords, name="candidate_coords", expected_rows=n_candidates
    )

    max_k = min(k_max, n_candidates)
    if max_k < 1:
        return []

    distances = _compute_coordinate_distances(
        anchor_coordinate_array, candidate_coordinate_array
    )

    true_match_ranks: list[int] = []
    n_missing_match = 0
    for anchor_position in range(n_anchors):
        ranked_candidate_indices = _rank_candidates_by_coordinate_distance(
            distances[anchor_position], candidate_label_array
        )
        true_match_rank = _find_true_match_rank(
            candidate_label_array[ranked_candidate_indices],
            int(anchor_label_array[anchor_position]),
        )
        if true_match_rank is None:
            n_missing_match += 1
        else:
            true_match_ranks.append(true_match_rank)

    n_anchors_with_match = len(true_match_ranks)
    rank_array = np.asarray(true_match_ranks, dtype=np.int64)

    coverage_rows: list[dict[str, int | float]] = []
    for k in range(1, max_k + 1):
        if n_anchors_with_match == 0:
            n_covered = 0
            coverage = 0.0
        else:
            n_covered = int(np.count_nonzero(rank_array < k))
            coverage = float(n_covered) / float(n_anchors_with_match)
        coverage_rows.append(
            {
                "k": int(k),
                "coverage": coverage,
                "n_anchors": int(n_anchors_with_match),
                "n_covered": int(n_covered),
                "n_missing_match": int(n_missing_match),
            }
        )
    return coverage_rows


def choose_k(coverage_rows: Sequence[dict[str, int | float]]) -> int:
    """Choose the smallest K that reaches full true-match coverage.

    The caller must escalate to the owner if the returned K still has coverage
    below ``1.0``.  SPEC 0046 requires using the largest evaluated K and
    recording the residual miss rate in that case rather than silently
    changing the protocol.

    Args:
        coverage_rows: Non-empty rows from ``coverage_vs_k``.  Each row must
            contain ``"k"`` and ``"coverage"``.

    Returns:
        The smallest ``k`` whose coverage is at least ``1.0``.  If no row
        reaches full coverage, the largest ``k`` in the rows is returned.

    Raises:
        ValueError: If ``coverage_rows`` is empty.
    """
    if not coverage_rows:
        raise ValueError("coverage_rows must contain at least one row")

    fully_covered_k_values = [
        int(row["k"])
        for row in coverage_rows
        if float(row["coverage"]) >= 1.0
    ]
    if fully_covered_k_values:
        return min(fully_covered_k_values)
    return max(int(row["k"]) for row in coverage_rows)
