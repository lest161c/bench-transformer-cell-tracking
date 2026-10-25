"""Tests for population-based z-normalization of KNN candidate features."""

import math

import numpy as np
import pytest
import torch

from src.analysis.knn_normalization import (
    MINIMUM_STANDARD_DEVIATION,
    apply_population_standardizer,
    assert_population_normalized,
    fit_population_standardizer,
)


def test_fit_and_apply_training_population_has_zero_mean_and_unit_std():
    """Fitting on the training pool yields zero mean and unit std after apply."""
    anchor_features = torch.tensor(
        [
            [[0.0, 10.0], [2.0, 20.0]],
            [[4.0, 30.0], [6.0, 40.0]],
        ]
    )
    candidate_features = torch.tensor(
        [
            [[8.0, 50.0], [10.0, 60.0]],
            [[12.0, 70.0], [14.0, 80.0]],
        ]
    )

    mean, std = fit_population_standardizer(anchor_features, candidate_features)

    # Pooled feature 0 is 0, 2, ..., 14; feature 1 is 10, 20, ..., 80.
    torch.testing.assert_close(
        mean, torch.tensor([7.0, 45.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        std,
        torch.tensor([math.sqrt(24.0), math.sqrt(600.0)]),
        atol=1e-6,
        rtol=0.0,
    )

    standardized_anchor, standardized_candidate = apply_population_standardizer(
        anchor_features, candidate_features, mean, std
    )
    pooled = torch.cat(
        [
            standardized_anchor.reshape(-1, 2),
            standardized_candidate.reshape(-1, 2),
        ],
        dim=0,
    )

    torch.testing.assert_close(
        pooled.mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        pooled.std(dim=0), torch.ones(2), atol=1e-6, rtol=0.0
    )
    assert_population_normalized(anchor_features, candidate_features, mean, std)


def test_apply_uses_training_statistics_for_validation_data():
    """Validation data is transformed by training stats and is not refitted."""
    base_values = torch.arange(8, dtype=torch.float32)
    train_anchor_features = torch.stack(
        [base_values[:4], 2.0 * base_values[:4]], dim=-1
    ).reshape(2, 2, 2)
    train_candidate_features = torch.stack(
        [base_values[4:], 2.0 * base_values[4:]], dim=-1
    ).reshape(2, 2, 2)

    train_mean, train_std = fit_population_standardizer(
        train_anchor_features, train_candidate_features
    )

    validation_values = base_values + 100.0
    validation_anchor_features = torch.stack(
        [validation_values[:4], 2.0 * validation_values[:4]], dim=-1
    ).reshape(2, 2, 2)
    validation_candidate_features = torch.stack(
        [validation_values[4:], 2.0 * validation_values[4:]], dim=-1
    ).reshape(2, 2, 2)

    validation_pooled = torch.cat(
        [
            validation_anchor_features.reshape(-1, 2),
            validation_candidate_features.reshape(-1, 2),
        ],
        dim=0,
    )
    validation_mean = validation_pooled.mean(dim=0)
    expected_transformed_mean = (validation_mean - train_mean) / train_std

    standardized_validation_anchor, standardized_validation_candidate = (
        apply_population_standardizer(
            validation_anchor_features,
            validation_candidate_features,
            train_mean,
            train_std,
        )
    )
    transformed_validation_pooled = torch.cat(
        [
            standardized_validation_anchor.reshape(-1, 2),
            standardized_validation_candidate.reshape(-1, 2),
        ],
        dim=0,
    )
    transformed_validation_mean = transformed_validation_pooled.mean(dim=0)

    torch.testing.assert_close(
        transformed_validation_mean,
        expected_transformed_mean,
        atol=1e-5,
        rtol=0.0,
    )
    # A refit would force this mean to zero; it must stay non-zero instead.
    assert torch.all(transformed_validation_mean.abs() > 1.0)


def test_constant_feature_std_is_clamped_to_minimum():
    """A constant feature gets the clamp floor and maps to zero after apply."""
    anchor_features = torch.tensor([[[5.0, 0.0], [5.0, 1.0]]])
    candidate_features = torch.tensor([[[5.0, 2.0], [5.0, 3.0]]])

    mean, std = fit_population_standardizer(anchor_features, candidate_features)

    assert mean[0].item() == pytest.approx(5.0)
    assert std[0].item() == pytest.approx(MINIMUM_STANDARD_DEVIATION)
    assert std[0].item() == pytest.approx(1e-6)
    assert torch.isfinite(std).all()

    standardized_anchor, standardized_candidate = apply_population_standardizer(
        anchor_features, candidate_features, mean, std
    )
    pooled_feature_zero = torch.cat(
        [
            standardized_anchor[..., 0].reshape(-1),
            standardized_candidate[..., 0].reshape(-1),
        ],
        dim=0,
    )
    torch.testing.assert_close(
        pooled_feature_zero, torch.zeros_like(pooled_feature_zero), atol=1e-6, rtol=0.0
    )


@pytest.mark.parametrize(
    "anchor_shape, candidate_shape",
    [
        ((0, 3, 4), (2, 3, 4)),
        ((2, 0, 4), (2, 3, 4)),
        ((2, 3, 0), (2, 3, 4)),
    ],
)
def test_empty_input_raises_value_error(anchor_shape, candidate_shape):
    """Empty anchor or candidate inputs raise a clear ValueError."""
    with pytest.raises(ValueError, match="empty"):
        fit_population_standardizer(
            torch.empty(anchor_shape),
            torch.empty(candidate_shape),
        )


def test_numpy_and_torch_inputs_are_equivalent():
    """Numpy and torch inputs produce identical statistics and transforms."""
    random_generator = np.random.default_rng(42)
    anchor_numpy = random_generator.normal(size=(3, 4, 5)).astype(np.float32)
    candidate_numpy = random_generator.normal(
        loc=2.0, size=(3, 4, 5)
    ).astype(np.float32)

    numpy_mean, numpy_std = fit_population_standardizer(
        anchor_numpy, candidate_numpy
    )
    torch_mean, torch_std = fit_population_standardizer(
        torch.from_numpy(anchor_numpy), torch.from_numpy(candidate_numpy)
    )

    torch.testing.assert_close(numpy_mean, torch_mean)
    torch.testing.assert_close(numpy_std, torch_std)

    numpy_anchor_standardized, numpy_candidate_standardized = (
        apply_population_standardizer(
            anchor_numpy, candidate_numpy, numpy_mean, numpy_std
        )
    )
    torch_anchor_standardized, torch_candidate_standardized = (
        apply_population_standardizer(
            torch.from_numpy(anchor_numpy),
            torch.from_numpy(candidate_numpy),
            torch_mean,
            torch_std,
        )
    )

    torch.testing.assert_close(numpy_anchor_standardized, torch_anchor_standardized)
    torch.testing.assert_close(
        numpy_candidate_standardized, torch_candidate_standardized
    )


def test_shapes_and_dtype_are_preserved():
    """Fit returns (D,) float32 stats; apply preserves shapes with float32."""
    anchor_features = torch.randn(3, 4, 7, dtype=torch.float64)
    candidate_features = torch.randn(3, 4, 7, dtype=torch.float64)

    mean, std = fit_population_standardizer(anchor_features, candidate_features)

    assert mean.shape == (7,)
    assert std.shape == (7,)
    assert mean.dtype == torch.float32
    assert std.dtype == torch.float32

    standardized_anchor, standardized_candidate = apply_population_standardizer(
        anchor_features, candidate_features, mean, std
    )

    assert standardized_anchor.shape == anchor_features.shape
    assert standardized_candidate.shape == candidate_features.shape
    assert standardized_anchor.dtype == torch.float32
    assert standardized_candidate.dtype == torch.float32
