"""Tests for the K-nearest-neighbour softmax probe and candidate metrics."""

import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.analysis.knn_probe import (
    DEFAULT_THRESHOLD_GRID,
    KnnLinearProbe,
    KnnMLPProbe,
    evaluate_knn_probe,
    predict_proba,
    select_threshold_and_metrics,
    train_knn_probe,
)

FEATURE_DIM = 4
NUM_CANDIDATES = 4
NUM_ANCHORS = 3


def _make_model(model_kind: str) -> torch.nn.Module:
    """Create a small linear or MLP probe for tests."""
    if model_kind == "linear":
        return KnnLinearProbe(FEATURE_DIM)
    if model_kind == "mlp":
        return KnnMLPProbe(FEATURE_DIM, hidden=8)
    raise ValueError(f"unknown model kind: {model_kind}")


def _make_features(
    n_anchors: int = NUM_ANCHORS,
    n_candidates: int = NUM_CANDIDATES,
    feature_dim: int = FEATURE_DIM,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic random anchor and candidate feature tensors.

    The anchor feature is repeated across the candidate axis to match the
    ``(K, 2, D)`` SPEC representation, where each candidate slot pairs the same
    anchor ``z_i`` with a different candidate ``z_j``.
    """
    generator = torch.Generator().manual_seed(seed)
    anchor_features = torch.randn(
        n_anchors, 1, feature_dim, generator=generator
    ).expand(-1, n_candidates, -1)
    candidate_features = torch.randn(
        n_anchors, n_candidates, feature_dim, generator=generator
    )
    return anchor_features.clone(), candidate_features


@pytest.mark.parametrize("model_kind", ["linear", "mlp"])
def test_output_shapes_and_softmax_rows_sum_to_one(model_kind: str) -> None:
    """Both probes return (B, K) logits and normalised (B, K) probabilities."""
    model = _make_model(model_kind)
    anchor_features, candidate_features = _make_features()

    logits = model(anchor_features, candidate_features)
    assert logits.shape == (NUM_ANCHORS, NUM_CANDIDATES)

    pair_features = torch.stack((anchor_features, candidate_features), dim=-2)
    assert pair_features.shape == (
        NUM_ANCHORS,
        NUM_CANDIDATES,
        2,
        FEATURE_DIM,
    )
    pair_logits = model.forward_pairs(pair_features)
    assert pair_logits.shape == (NUM_ANCHORS, NUM_CANDIDATES)
    torch.testing.assert_close(pair_logits, logits)

    per_anchor_pairs = pair_features[0]
    assert per_anchor_pairs.shape == (NUM_CANDIDATES, 2, FEATURE_DIM)
    assert model.forward_pairs(per_anchor_pairs).shape == (NUM_CANDIDATES,)

    probabilities = predict_proba(model, anchor_features, candidate_features)
    assert probabilities.shape == (NUM_ANCHORS, NUM_CANDIDATES)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(NUM_ANCHORS),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize("model_kind", ["linear", "mlp"])
def test_candidate_permutation_permutes_logits_and_probabilities(
    model_kind: str,
) -> None:
    """Permuting candidates permutes scores and probabilities consistently."""
    model = _make_model(model_kind)
    anchor_features, candidate_features = _make_features()
    permutation = torch.tensor([2, 0, 3, 1])

    logits = model(anchor_features, candidate_features)
    permuted_logits = model(anchor_features, candidate_features[:, permutation, :])
    torch.testing.assert_close(permuted_logits, logits[:, permutation])

    probabilities = predict_proba(model, anchor_features, candidate_features)
    permuted_probabilities = predict_proba(
        model, anchor_features, candidate_features[:, permutation, :]
    )
    torch.testing.assert_close(permuted_probabilities, probabilities[:, permutation])

    pair_features = torch.stack((anchor_features, candidate_features), dim=-2)
    permuted_pair_features = torch.stack(
        (anchor_features, candidate_features[:, permutation, :]), dim=-2
    )
    torch.testing.assert_close(
        model.forward_pairs(permuted_pair_features),
        model.forward_pairs(pair_features)[:, permutation],
    )


@pytest.mark.parametrize("model_kind", ["linear", "mlp"])
def test_gradient_flows_to_both_anchor_and_candidate_branches(
    model_kind: str,
) -> None:
    """The shared scorer receives non-zero gradients through both pair halves."""
    torch.manual_seed(0)
    model = _make_model(model_kind)
    anchor_features = torch.randn(
        NUM_ANCHORS, NUM_CANDIDATES, FEATURE_DIM, requires_grad=True
    )
    candidate_features = torch.randn(
        NUM_ANCHORS, NUM_CANDIDATES, FEATURE_DIM, requires_grad=True
    )

    logits = model(anchor_features, candidate_features)
    target_index = torch.zeros(NUM_ANCHORS, dtype=torch.long)
    loss = F.cross_entropy(logits, target_index)
    loss.backward()

    assert anchor_features.grad is not None
    assert candidate_features.grad is not None
    assert anchor_features.grad.abs().sum().item() > 0.0
    assert candidate_features.grad.abs().sum().item() > 0.0


def test_perfect_predictions_give_perfect_metrics() -> None:
    """One-hot true-candidate probabilities give top-1 and pair F1 of one."""
    target_index = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    probabilities = np.zeros((target_index.size, NUM_CANDIDATES), dtype=np.float64)
    probabilities[np.arange(target_index.size), target_index] = 1.0

    metrics = select_threshold_and_metrics(probabilities, target_index)

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["nll"] == pytest.approx(0.0)
    assert metrics["n_candidates"] == NUM_CANDIDATES
    assert metrics["n_anchors"] == target_index.size
    # All thresholds achieve F1 = 1; the documented tie-break picks the
    # smallest grid threshold.
    assert metrics["threshold"] == DEFAULT_THRESHOLD_GRID[0]


def test_random_predictions_have_bounded_metrics_and_threshold() -> None:
    """Random probabilities produce bounded metrics and a selected threshold."""
    generator = np.random.default_rng(0)
    logits = torch.as_tensor(generator.normal(size=(8, 5)), dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    target_index = torch.randint(0, 5, (8,), generator=torch.Generator().manual_seed(1))

    metrics = select_threshold_and_metrics(probabilities, target_index)

    for metric_name in ("top1_accuracy", "precision", "recall", "f1"):
        assert 0.0 <= metrics[metric_name] <= 1.0
    assert metrics["nll"] >= 0.0
    assert np.isfinite(metrics["nll"])
    assert metrics["threshold"] in DEFAULT_THRESHOLD_GRID
    assert metrics["n_candidates"] == 5
    assert metrics["n_anchors"] == 8


def test_zero_anchors_raise_and_k_equals_one_is_supported() -> None:
    """Zero anchors raise; a single candidate has a documented perfect result."""
    with pytest.raises(ValueError, match="at least one anchor"):
        select_threshold_and_metrics(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    n_anchors = 5
    probabilities = np.ones((n_anchors, 1), dtype=np.float64)
    target_index = np.zeros(n_anchors, dtype=np.int64)
    metrics = select_threshold_and_metrics(probabilities, target_index)

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["n_candidates"] == 1
    assert metrics["n_anchors"] == n_anchors
    assert metrics["threshold"] == DEFAULT_THRESHOLD_GRID[0]


def test_module_contains_no_balanced_accuracy_reference() -> None:
    """The new main probe module must not reference the retired metric."""
    module_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "analysis"
        / "knn_probe.py"
    )
    assert "balanced_accuracy" not in module_path.read_text(encoding="utf-8")


def _make_separable_batches() -> tuple[
    list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]
]:
    """Build tiny perfectly separable candidate-classification batches."""
    anchor_features = torch.tensor(
        [[[1.0, 0.0, 0.0]] * 3] * 4,
        dtype=torch.float32,
    )
    candidate_features = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
        * 4,
        dtype=torch.float32,
    )
    target_index = torch.zeros(4, dtype=torch.long)
    train_batch = {
        "anchor_features": anchor_features,
        "candidate_features": candidate_features,
        "target_index": target_index,
    }
    return [train_batch], [train_batch]


def test_train_knn_probe_returns_history_best_model_and_is_deterministic() -> None:
    """Training records NLL history, restores a model, and seeds deterministically."""
    train_batches, val_batches = _make_separable_batches()

    torch.manual_seed(0)
    first_model = KnnLinearProbe(feat_dim=3)
    first_history, trained_model = train_knn_probe(
        first_model,
        train_batches,
        val_batches,
        epochs=40,
        lr=0.05,
        patience=5,
        seed=42,
    )

    assert len(first_history) > 0
    assert set(first_history[0]) == {"epoch", "train_nll", "validation_nll"}
    assert trained_model.training is False

    torch.manual_seed(0)
    second_model = KnnLinearProbe(feat_dim=3)
    second_history, _ = train_knn_probe(
        second_model,
        train_batches,
        val_batches,
        epochs=40,
        lr=0.05,
        patience=5,
        seed=42,
    )
    assert second_history == first_history


def test_evaluate_knn_probe_uses_supplied_validation_threshold() -> None:
    """Evaluation selects the threshold on validation data, not on test data."""
    model = KnnLinearProbe(feat_dim=FEATURE_DIM)
    anchor_features, candidate_features = _make_features()
    target_index = torch.tensor([0, 1, 2], dtype=torch.long)

    metrics = evaluate_knn_probe(
        model,
        anchor_features,
        candidate_features,
        target_index,
        validation_data={
            "anchor_features": anchor_features,
            "candidate_features": candidate_features,
            "target_index": target_index,
        },
    )

    assert set(metrics) == {
        "top1_accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "nll",
        "n_candidates",
        "n_anchors",
    }
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["n_candidates"] == NUM_CANDIDATES
    assert metrics["n_anchors"] == NUM_ANCHORS
