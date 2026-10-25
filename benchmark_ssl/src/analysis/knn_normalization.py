"""Population-based, leakage-safe z-normalization for KNN candidate probes.

The KNN probe consumes ``(N, K, D)`` tensors: for each anchor in frame ``t``,
the ``K`` nearest candidates in frame ``t+1`` are described by ``D`` features
on both the anchor and candidate sides.  Raw handcrafted features mix very
different numeric scales (centroids in pixels, intensities in ``[0, 1]``,
distances in pixels), so the probe needs standardized inputs.

Why population-based rather than per-sample, per-batch, or per-pair
statistics: the standardization must be a single fixed affine map shared by
the training, validation, and held-out data.  Per-sample or per-pair
normalization would change each row based on the other rows that happen to be
in the same batch or candidate set, leaking information across the split
boundary and making the train and test distributions inconsistent.
Population statistics are estimated once from the training split's candidate
population and then frozen.

Leakage safety: :func:`fit_population_standardizer` must only see the training
split.  :func:`apply_population_standardizer` never refits; validation and
held-out features are transformed with the training statistics.  This mirrors
``fit_feature_standardizer`` / ``apply_feature_standardizer`` in
``src.edge_probing.harness.edge_datasets``, including their use of
``torch.std(dim=0)`` with the default Bessel-corrected estimator and a
standard-deviation floor of ``1e-6``.

Numerics: inputs are converted to ``float32`` before statistics are computed,
and all returned tensors are ``float32``.  This matches the existing harness
feature dtype and keeps numpy and torch inputs equivalent.  The standard
deviation is Bessel-corrected (``torch.std`` default, denominator ``n - 1``),
which is the existing harness convention; the population here is the pooled
set of all anchor and candidate rows, not the sample size of a single pair.
"""

import numpy as np
import torch

# Smallest standard deviation used for the affine transform.  A constant
# feature has zero variance; clamping keeps the division finite and maps the
# constant value to zero after centering.  This is the same floor used by the
# existing harness standardizer.
MINIMUM_STANDARD_DEVIATION = 1e-6

FeatureArray = np.ndarray | torch.Tensor


def _to_float32_tensor(value: FeatureArray, argument_name: str) -> torch.Tensor:
    """Convert a numpy array or torch tensor to a ``float32`` torch tensor.

    Args:
        value: Feature array of shape ``(..., D)``.  numpy arrays and torch
            tensors are accepted; other types are rejected so a mistaken list
            or scalar cannot silently change the statistics.
        argument_name: Name of the caller's argument, used in error messages
            so failures identify the offending input.

    Returns:
        A ``float32`` torch tensor with the same shape as ``value``.

    Raises:
        TypeError: If ``value`` is neither a numpy array nor a torch tensor.
        ValueError: If ``value`` has no feature dimension (zero-dimensional).
    """
    if isinstance(value, torch.Tensor):
        tensor = value.to(dtype=torch.float32)
    elif isinstance(value, np.ndarray):
        tensor = torch.as_tensor(value, dtype=torch.float32)
    else:
        raise TypeError(
            f"{argument_name} must be a numpy array or torch tensor, "
            f"got {type(value).__name__}"
        )

    if tensor.ndim < 1:
        raise ValueError(
            f"{argument_name} must have at least one dimension (the feature "
            f"axis), got shape {tuple(tensor.shape)}"
        )
    return tensor


def _pool_anchor_and_candidate_features(
    anchor_features: torch.Tensor,
    candidate_features: torch.Tensor,
) -> torch.Tensor:
    """Flatten and concatenate anchor and candidate features into ``(M, D)``.

    Called by :func:`fit_population_standardizer` and
    :func:`assert_population_normalized` so both use exactly the same pooling
    order.  Pooling every ``(N, K, D)`` position treats each candidate row as
    one population entry, which is the distribution the probe is trained on.

    Args:
        anchor_features: ``float32`` tensor of shape ``(N, K, D)``.
        candidate_features: ``float32`` tensor of shape ``(N, K, D)`` with the
            same feature dimension as ``anchor_features``.

    Returns:
        A ``float32`` tensor of shape ``(2 * N * K, D)``.

    Raises:
        ValueError: If either input does not have shape ``(N, K, D)``, the
            feature dimensions differ, or either input is empty.
    """
    if anchor_features.ndim != 3 or candidate_features.ndim != 3:
        raise ValueError(
            "anchor_features and candidate_features must have shape "
            f"(N, K, D); got {tuple(anchor_features.shape)} and "
            f"{tuple(candidate_features.shape)}"
        )
    if anchor_features.numel() == 0 or candidate_features.numel() == 0:
        raise ValueError(
            "Cannot fit population statistics on empty anchor or candidate "
            f"features; got shapes {tuple(anchor_features.shape)} and "
            f"{tuple(candidate_features.shape)}"
        )
    if anchor_features.shape[-1] != candidate_features.shape[-1]:
        raise ValueError(
            "anchor_features and candidate_features must have the same "
            f"feature dimension, got {anchor_features.shape[-1]} and "
            f"{candidate_features.shape[-1]}"
        )

    feature_dim = anchor_features.shape[-1]
    return torch.cat(
        [
            anchor_features.reshape(-1, feature_dim),
            candidate_features.reshape(-1, feature_dim),
        ],
        dim=0,
    )


def fit_population_standardizer(
    anchor_features: FeatureArray,
    candidate_features: FeatureArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit per-feature mean and standard deviation on the training population.

    Pool all anchor and candidate entries across both sides and all ``K``
    candidate slots, then compute one mean and one standard deviation per
    feature.  This is the normalization step from SPEC 0046 section 4.5 and
    matches the existing harness semantics: ``torch.std(dim=0)`` with its
    default Bessel correction, clamped to at least
    :data:`MINIMUM_STANDARD_DEVIATION`.

    Fit this only on the training split.  Validation and held-out features
    must be transformed with the returned statistics through
    :func:`apply_population_standardizer`; refitting there would leak
    validation information into the preprocessing.

    Args:
        anchor_features: Training anchor features of shape ``(N, K, D)`` as a
            numpy array or torch tensor.  They are flattened to ``(N * K, D)``
            before pooling.
        candidate_features: Training candidate features of shape ``(N, K, D)``
            with the same feature dimension.  Both sides contribute equally
            to the pooled population.

    Returns:
        ``(mean, std)``: two ``float32`` tensors of shape ``(D,)``.  ``std`` is
        clamped to be at least ``1e-6`` so a constant feature stays finite and
        maps to zero after centering.

    Raises:
        TypeError: If an input is neither a numpy array nor a torch tensor.
        ValueError: If an input is empty, has no feature axis, or the two
            feature dimensions differ.
    """
    anchor_tensor = _to_float32_tensor(anchor_features, "anchor_features")
    candidate_tensor = _to_float32_tensor(candidate_features, "candidate_features")
    pooled_features = _pool_anchor_and_candidate_features(
        anchor_tensor, candidate_tensor
    )

    mean = pooled_features.mean(dim=0)
    std = pooled_features.std(dim=0).clamp(min=MINIMUM_STANDARD_DEVIATION)
    return mean, std


def apply_population_standardizer(
    anchor_features: FeatureArray,
    candidate_features: FeatureArray,
    mean: FeatureArray,
    std: FeatureArray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fixed training statistics to anchor and candidate features.

    Computes ``(x - mean) / std`` for both sides using the supplied
    statistics.  This function never estimates or updates ``mean`` or ``std``,
    so passing validation or held-out features here guarantees they are
    transformed with the training split's affine map.

    Args:
        anchor_features: Anchor features to transform, shape ``(N, K, D)``.
        candidate_features: Candidate features to transform, shape
            ``(N, K, D)``.  Their ``N`` and ``K`` may differ from the anchor
            shape, but both must have the same trailing feature dimension as
            ``mean`` and ``std``.
        mean: Per-feature means of shape ``(D,)`` returned by
            :func:`fit_population_standardizer`.
        std: Per-feature standard deviations of shape ``(D,)`` returned by
            :func:`fit_population_standardizer`.  The caller is responsible
            for passing fitted (already clamped) statistics.

    Returns:
        ``(standardized_anchor_features, standardized_candidate_features)``:
        two ``float32`` tensors with the same shapes as the respective inputs.

    Raises:
        TypeError: If an input is neither a numpy array nor a torch tensor.
        ValueError: If a feature input does not have shape ``(N, K, D)``, if
            ``mean`` or ``std`` is not one-dimensional, if their shapes
            differ, or if a feature dimension does not match the statistics.
    """
    anchor_tensor = _to_float32_tensor(anchor_features, "anchor_features")
    candidate_tensor = _to_float32_tensor(candidate_features, "candidate_features")
    mean_tensor = _to_float32_tensor(mean, "mean")
    std_tensor = _to_float32_tensor(std, "std")

    if anchor_tensor.ndim != 3 or candidate_tensor.ndim != 3:
        raise ValueError(
            "anchor_features and candidate_features must have shape "
            f"(N, K, D); got {tuple(anchor_tensor.shape)} and "
            f"{tuple(candidate_tensor.shape)}"
        )
    if mean_tensor.ndim != 1 or std_tensor.ndim != 1:
        raise ValueError(
            "mean and std must be one-dimensional per-feature tensors, got "
            f"shapes {tuple(mean_tensor.shape)} and {tuple(std_tensor.shape)}"
        )
    if mean_tensor.shape != std_tensor.shape:
        raise ValueError(
            f"mean and std must have the same shape, got "
            f"{tuple(mean_tensor.shape)} and {tuple(std_tensor.shape)}"
        )
    feature_dim = mean_tensor.shape[0]
    if (
        anchor_tensor.shape[-1] != feature_dim
        or candidate_tensor.shape[-1] != feature_dim
    ):
        raise ValueError(
            f"Feature dimensions do not match the statistics: anchor "
            f"{anchor_tensor.shape[-1]}, candidate "
            f"{candidate_tensor.shape[-1]}, statistics {feature_dim}"
        )

    standardized_anchor_features = (anchor_tensor - mean_tensor) / std_tensor
    standardized_candidate_features = (
        candidate_tensor - mean_tensor
    ) / std_tensor
    return standardized_anchor_features, standardized_candidate_features


def assert_population_normalized(
    anchor_features: FeatureArray,
    candidate_features: FeatureArray,
    mean: FeatureArray,
    std: FeatureArray,
    atol: float = 1e-4,
) -> None:
    """Check that the training population is standardized to mean 0 and std 1.

    This is a diagnostic helper for unit tests and debugging, not a check for
    validation or held-out data.  After applying the supplied training
    statistics to the given population, it verifies that each feature has a
    pooled mean close to zero and a pooled Bessel-corrected standard deviation
    close to one.  Features whose supplied ``std`` is at the clamp floor (a
    constant training feature) are excluded from the standard-deviation check
    because centering necessarily turns them into a zero-variance feature.

    Do not call this on validation or held-out data: their transformed mean is
    intentionally not zero when their distribution differs from training.

    Args:
        anchor_features: Anchor features, normally ``(N, K, D)``, from the
            population used to fit ``mean`` and ``std``.
        candidate_features: Candidate features from the same training
            population.
        mean: Per-feature means of shape ``(D,)``.
        std: Per-feature standard deviations of shape ``(D,)``.
        atol: Absolute tolerance for both the mean and standard-deviation
            checks.

    Returns:
        ``None``.  The function is silent on success.

    Raises:
        AssertionError: If the pooled mean is not within ``atol`` of zero or a
            non-constant feature's pooled standard deviation is not within
            ``atol`` of one.
        TypeError: If an input is neither a numpy array nor a torch tensor.
        ValueError: If the inputs or statistics have incompatible shapes.
    """
    standardized_anchor_features, standardized_candidate_features = (
        apply_population_standardizer(
            anchor_features,
            candidate_features,
            mean,
            std,
        )
    )
    feature_dim = standardized_anchor_features.shape[-1]
    pooled_features = torch.cat(
        [
            standardized_anchor_features.reshape(-1, feature_dim),
            standardized_candidate_features.reshape(-1, feature_dim),
        ],
        dim=0,
    )

    population_mean = pooled_features.mean(dim=0)
    population_std = pooled_features.std(dim=0)
    std_tensor = _to_float32_tensor(std, "std")
    non_constant_feature_mask = std_tensor > MINIMUM_STANDARD_DEVIATION

    if not torch.allclose(
        population_mean,
        torch.zeros_like(population_mean),
        atol=atol,
        rtol=0.0,
    ):
        raise AssertionError(
            "Standardized training population is not centered at zero; "
            f"per-feature means: {population_mean.tolist()}"
        )

    non_constant_std = population_std[non_constant_feature_mask]
    if not torch.allclose(
        non_constant_std,
        torch.ones_like(non_constant_std),
        atol=atol,
        rtol=0.0,
    ):
        raise AssertionError(
            "Standardized training population does not have unit standard "
            f"deviation for non-constant features; values: "
            f"{non_constant_std.tolist()}"
        )
