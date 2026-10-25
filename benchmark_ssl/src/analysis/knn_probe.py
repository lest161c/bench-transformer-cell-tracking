"""Softmax probes for K-nearest-neighbour candidate classification.

Implements the SPEC 0046 candidate-classification probe: a shared scorer
``s(z_i, z_j)`` is applied independently to each of ``K`` candidates of an
anchor, and a softmax over the ``K`` scores turns the scores into a candidate
distribution. The module also provides cross-entropy training with validation
NLL early stopping, probability prediction, and candidate-level threshold
metrics (top-1, precision, recall, F1, and negative log-likelihood).

Notation used in docstrings:

- ``B``: number of anchors in a batch.
- ``K``: number of candidates per anchor.
- ``D``: feature dimensionality of one cell.
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
import torch
import torch.nn as nn

# Fixed validation grid from SPEC 0046 section 4.4: inclusive 0.01 to 0.99.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(
    float(threshold) for threshold in np.round(np.arange(0.01, 1.0, 0.01), 2)
)

ProbeBatch: TypeAlias = Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
ProbeBatches: TypeAlias = Sequence[ProbeBatch]


def _score_candidates(
    scorer: nn.Module,
    anchor_features: torch.Tensor,
    candidate_features: torch.Tensor,
) -> torch.Tensor:
    """Apply a shared pair scorer to every anchor-candidate combination.

    The scorer is deliberately applied to the flattened anchor-candidate pairs
    so that both probes share all parameters across the ``K`` candidates. Only
    the softmax that follows in the training loss couples the candidates.

    Args:
        scorer: Module mapping ``(..., 2 * D)`` pair features to ``(..., 1)``
            logits.
        anchor_features: ``(..., K, D)`` anchor features.
        candidate_features: ``(..., K, D)`` candidate features with the same
            leading shape as ``anchor_features``.

    Returns:
        ``(..., K)`` logits, with one score per anchor-candidate pair.

    Raises:
        ValueError: If the tensors do not have matching leading shapes or do
            not contain at least one candidate axis.
    """
    if anchor_features.ndim < 2 or candidate_features.ndim < 2:
        raise ValueError(
            "anchor_features and candidate_features must have shape (..., K, D)"
        )
    if anchor_features.shape[:-1] != candidate_features.shape[:-1]:
        raise ValueError(
            "anchor_features and candidate_features must share their leading "
            f"shape, got {tuple(anchor_features.shape[:-1])} and "
            f"{tuple(candidate_features.shape[:-1])}"
        )

    leading_shape = anchor_features.shape[:-1]
    pair_features = torch.cat((anchor_features, candidate_features), dim=-1)
    flat_logits = scorer(pair_features.reshape(-1, pair_features.shape[-1]))
    return flat_logits.reshape(*leading_shape)


class KnnLinearProbe(nn.Module):
    """Shared linear scorer ``w^T [z_i ; z_j]`` over ``K`` candidates.

    The same weight vector is applied to every anchor-candidate pair before the
    softmax over the candidate axis, so the model has one global notion of pair
    compatibility rather than one weight per candidate slot.
    """

    def __init__(self, feat_dim: int) -> None:
        """Initialize the shared linear scorer.

        Args:
            feat_dim: Feature dimensionality ``D`` of one cell. The scorer maps
                the concatenation of two cells, ``2 * D``, to one logit.
        """
        super().__init__()
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {feat_dim}")

        self.feat_dim = int(feat_dim)
        self.scorer = nn.Linear(2 * self.feat_dim, 1)

    def forward(
        self,
        anchor_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score every candidate of every anchor.

        Args:
            anchor_features: ``(B, K, D)`` anchor features.
            candidate_features: ``(B, K, D)`` candidate features.

        Returns:
            ``(B, K)`` logits. The softmax over the ``K`` axis is applied by
            the training loss or by :func:`predict_proba`.
        """
        return _score_candidates(self.scorer, anchor_features, candidate_features)

    def forward_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        """Score the SPEC's ``(..., K, 2, D)`` pair representation.

        Args:
            pair_features: ``(..., K, 2, D)`` tensor whose axis ``-2`` index
                ``0`` is the anchor feature and index ``1`` is the candidate
                feature. The SPEC's per-anchor ``(K, 2, D)`` representation is
                accepted by treating ``K`` as the only leading axis.

        Returns:
            ``(..., K)`` logits.

        Raises:
            ValueError: If the last two axes are not ``(2, D)``.
        """
        if pair_features.ndim < 3 or pair_features.shape[-2] != 2:
            raise ValueError(
                "pair_features must have shape (..., K, 2, D), got "
                f"{tuple(pair_features.shape)}"
            )

        anchor_features = pair_features[..., 0, :]
        candidate_features = pair_features[..., 1, :]
        return self.forward(anchor_features, candidate_features)


class KnnMLPProbe(nn.Module):
    """Shared 2-layer MLP scorer over ``K`` candidates.

    The architecture is ``Linear(2D, hidden) -> ReLU -> Linear(hidden, 1)`` and
    is applied to each anchor-candidate pair before the softmax over the
    candidate axis.
    """

    def __init__(self, feat_dim: int, hidden: int = 128) -> None:
        """Initialize the shared MLP scorer.

        Args:
            feat_dim: Feature dimensionality ``D`` of one cell.
            hidden: Number of hidden units in the single hidden layer.
        """
        super().__init__()
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be positive, got {feat_dim}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")

        self.feat_dim = int(feat_dim)
        self.hidden = int(hidden)
        self.scorer = nn.Sequential(
            nn.Linear(2 * self.feat_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        anchor_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score every candidate of every anchor.

        Args:
            anchor_features: ``(B, K, D)`` anchor features.
            candidate_features: ``(B, K, D)`` candidate features.

        Returns:
            ``(B, K)`` logits. The softmax over the ``K`` axis is applied by
            the training loss or by :func:`predict_proba`.
        """
        return _score_candidates(self.scorer, anchor_features, candidate_features)

    def forward_pairs(self, pair_features: torch.Tensor) -> torch.Tensor:
        """Score the SPEC's ``(..., K, 2, D)`` pair representation.

        Args:
            pair_features: ``(..., K, 2, D)`` tensor whose axis ``-2`` index
                ``0`` is the anchor feature and index ``1`` is the candidate
                feature. The SPEC's per-anchor ``(K, 2, D)`` representation is
                accepted by treating ``K`` as the only leading axis.

        Returns:
            ``(..., K)`` logits.

        Raises:
            ValueError: If the last two axes are not ``(2, D)``.
        """
        if pair_features.ndim < 3 or pair_features.shape[-2] != 2:
            raise ValueError(
                "pair_features must have shape (..., K, 2, D), got "
                f"{tuple(pair_features.shape)}"
            )

        anchor_features = pair_features[..., 0, :]
        candidate_features = pair_features[..., 1, :]
        return self.forward(anchor_features, candidate_features)


def _seed_torch(seed: int) -> None:
    """Seed the PyTorch random generators used by probe training.

    Args:
        seed: Non-negative integer seed. CPU and, when available, all CUDA
            generators are seeded so two runs with identical model
            initialization and data follow the same optimization path.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unpack_batch(batch: ProbeBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract anchor, candidate, and target tensors from a batch container.

    Args:
        batch: Either a mapping with keys ``anchor_features``,
            ``candidate_features``, and ``target_index``, or a length-3
            sequence ordered ``(anchor_features, candidate_features,
            target_index)``.

    Returns:
        The three tensors in anchor, candidate, target order.

    Raises:
        ValueError: If a mapping is missing a required key or a sequence does
            not have exactly three entries.
    """
    if isinstance(batch, Mapping):
        required_keys = ("anchor_features", "candidate_features", "target_index")
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise ValueError(
                f"batch is missing required key(s): {', '.join(missing_keys)}"
            )
        return (
            batch["anchor_features"],
            batch["candidate_features"],
            batch["target_index"],
        )

    if len(batch) != 3:
        raise ValueError(
            "a sequence batch must contain exactly three entries: "
            "(anchor_features, candidate_features, target_index)"
        )
    return batch[0], batch[1], batch[2]


def _prepare_batch_tensors(
    batch: ProbeBatch,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move one batch to a device and validate its shapes and labels.

    Args:
        batch: Mapping or sequence accepted by :func:`_unpack_batch`.
        device: Device on which training or evaluation runs.

    Returns:
        ``(anchor_features, candidate_features, target_index)`` tensors with
        float32 features and int64 targets on ``device``.

    Raises:
        ValueError: If the features are not three-dimensional with matching
            shapes, or the target is not a one-dimensional valid candidate
            index per anchor.
    """
    anchor_features, candidate_features, target_index = _unpack_batch(batch)
    anchor_features = torch.as_tensor(anchor_features, dtype=torch.float32, device=device)
    candidate_features = torch.as_tensor(
        candidate_features, dtype=torch.float32, device=device
    )
    target_index = torch.as_tensor(target_index, dtype=torch.long, device=device)

    if anchor_features.ndim != 3 or candidate_features.ndim != 3:
        raise ValueError("features must have shape (B, K, D)")
    if anchor_features.shape != candidate_features.shape:
        raise ValueError(
            "anchor_features and candidate_features must have identical shapes"
        )
    if target_index.ndim != 1 or target_index.shape[0] != anchor_features.shape[0]:
        raise ValueError("target_index must have shape (B,)")
    if anchor_features.shape[0] == 0:
        raise ValueError("batches must contain at least one anchor")
    if ((target_index < 0) | (target_index >= anchor_features.shape[1])).any():
        raise ValueError("target_index values must lie in [0, K)")

    return anchor_features, candidate_features, target_index


def _mean_batch_nll(
    model: nn.Module,
    batches: Sequence[ProbeBatch],
    criterion: nn.Module,
    device: torch.device | str,
) -> float:
    """Compute the anchor-weighted mean cross-entropy over a batch sequence.

    Args:
        model: Probe in evaluation mode.
        batches: Non-empty sequence of probe batches.
        criterion: Cross-entropy criterion with mean reduction.
        device: Device on which to evaluate.

    Returns:
        Mean negative log-likelihood per anchor.

    Raises:
        ValueError: If ``batches`` is empty.
    """
    if len(batches) == 0:
        raise ValueError("cannot compute NLL over an empty batch sequence")

    model.eval()
    total_weighted_nll = 0.0
    total_anchors = 0
    with torch.no_grad():
        for batch in batches:
            anchor_features, candidate_features, target_index = _prepare_batch_tensors(
                batch, device
            )
            logits = model(anchor_features, candidate_features)
            nll = criterion(logits, target_index)
            n_anchors = int(target_index.shape[0])
            total_weighted_nll += float(nll.item()) * n_anchors
            total_anchors += n_anchors

    return total_weighted_nll / total_anchors


def train_knn_probe(
    model: nn.Module,
    train_batches: ProbeBatches,
    val_batches: ProbeBatches,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[list[dict[str, float | int]], nn.Module]:
    """Train a K-way softmax probe with validation-NLL early stopping.

    Each batch contains ``anchor_features`` ``(B, K, D)``, ``candidate_features``
    ``(B, K, D)``, and ``target_index`` ``(B,)``. The loss is
    ``nn.CrossEntropyLoss`` over the ``K`` candidate axis, which makes the true
    match the positive class and the other candidates implicit negatives.

    Early stopping uses validation negative log-likelihood rather than top-1 or
    thresholded F1 because NLL is a proper scoring rule that uses the full
    softmax distribution and is less noisy on small validation splits. The
    validation split must never be the held-out evaluation split. The best
    validation-NLL state is restored before return.

    Args:
        model: A :class:`KnnLinearProbe`, :class:`KnnMLPProbe`, or compatible
            module consuming ``(B, K, D)`` anchors and candidates.
        train_batches: Non-empty sequence of training batches as mappings or
            length-3 sequences.
        val_batches: Non-empty sequence of validation batches used for early
            stopping.
        epochs: Maximum number of full passes over ``train_batches``.
        lr: Adam learning rate.
        patience: Number of epochs without validation-NLL improvement before
            stopping.
        seed: Seed passed to :func:`_seed_torch` before optimization starts.
        device: Device string, for example ``"cpu"`` or ``"cuda"``.

    Returns:
        ``(history, best_model)`` where ``history`` is a list with one dict per
        completed epoch containing ``epoch``, ``train_nll``, and
        ``validation_nll``. ``best_model`` is ``model`` with its best
        validation-NLL parameters loaded and in evaluation mode.

    Raises:
        ValueError: If the batch sequences are empty, ``epochs`` or
            ``patience`` is not positive, or the learning rate is not positive.
    """
    train_batches = list(train_batches)
    val_batches = list(val_batches)
    if len(train_batches) == 0:
        raise ValueError("train_batches must not be empty")
    if len(val_batches) == 0:
        raise ValueError("val_batches must not be empty for NLL early stopping")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if patience <= 0:
        raise ValueError(f"patience must be positive, got {patience}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    _seed_torch(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_validation_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        model.train()
        total_weighted_train_nll = 0.0
        total_train_anchors = 0

        for batch in train_batches:
            anchor_features, candidate_features, target_index = _prepare_batch_tensors(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(anchor_features, candidate_features)
            loss = criterion(logits, target_index)
            loss.backward()
            optimizer.step()

            n_anchors = int(target_index.shape[0])
            total_weighted_train_nll += float(loss.item()) * n_anchors
            total_train_anchors += n_anchors

        train_nll = total_weighted_train_nll / total_train_anchors
        validation_nll = _mean_batch_nll(model, val_batches, criterion, device)
        history.append(
            {
                "epoch": int(epoch),
                "train_nll": float(train_nll),
                "validation_nll": float(validation_nll),
            }
        )

        if validation_nll < best_validation_nll:
            best_validation_nll = float(validation_nll)
            best_state = {
                parameter_name: parameter.detach().cpu().clone()
                for parameter_name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history, model


def predict_proba(
    model: nn.Module,
    anchor_features: torch.Tensor,
    candidate_features: torch.Tensor,
) -> torch.Tensor:
    """Predict candidate probabilities for every anchor.

    Args:
        model: Trained or untrained K-nearest-neighbour probe.
        anchor_features: ``(N, K, D)`` anchor features.
        candidate_features: ``(N, K, D)`` candidate features.

    Returns:
        ``(N, K)`` CPU tensor of softmax probabilities whose rows sum to one.
        The model is set to evaluation mode and gradients are disabled.
    """
    model_device = next(model.parameters()).device
    anchor_tensor = torch.as_tensor(
        anchor_features, dtype=torch.float32, device=model_device
    )
    candidate_tensor = torch.as_tensor(
        candidate_features, dtype=torch.float32, device=model_device
    )

    model.eval()
    with torch.no_grad():
        logits = model(anchor_tensor, candidate_tensor)
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.detach().cpu()


def _as_probability_array(
    probabilities: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Convert probabilities to a validated two-dimensional float64 array.

    Args:
        probabilities: ``(N, K)`` tensor or array of candidate probabilities.

    Returns:
        Float64 array of shape ``(N, K)``.

    Raises:
        ValueError: If the input is not two-dimensional, contains non-finite
            values, or contains values outside ``[0, 1]``.
    """
    if isinstance(probabilities, torch.Tensor):
        probability_array = probabilities.detach().cpu().numpy()
    else:
        probability_array = np.asarray(probabilities)

    probability_array = np.asarray(probability_array, dtype=np.float64)
    if probability_array.ndim != 2:
        raise ValueError(
            "probabilities must have shape (N, K), got "
            f"{tuple(probability_array.shape)}"
        )
    if not np.isfinite(probability_array).all():
        raise ValueError("probabilities must be finite")
    if probability_array.size and (
        (probability_array < 0.0).any() or (probability_array > 1.0).any()
    ):
        raise ValueError("probabilities must lie in [0, 1]")
    return probability_array


def _as_target_indices(
    target_index: torch.Tensor | np.ndarray | Sequence[int],
    n_anchors: int,
    n_candidates: int,
) -> np.ndarray:
    """Validate target candidate indices for a probability matrix.

    Args:
        target_index: ``(N,)`` true-match index per anchor.
        n_anchors: Number of probability rows ``N``.
        n_candidates: Number of probability columns ``K``.

    Returns:
        One-dimensional int64 array of valid candidate indices.

    Raises:
        ValueError: If the target is not one-dimensional, its length does not
            match ``n_anchors``, its dtype is not integral, or a value lies
            outside ``[0, n_candidates)``.
    """
    if isinstance(target_index, torch.Tensor):
        target_array = target_index.detach().cpu().numpy()
    else:
        target_array = np.asarray(target_index)

    if target_array.ndim != 1:
        raise ValueError(
            "target_index must have shape (N,), got "
            f"{tuple(target_array.shape)}"
        )
    if target_array.shape[0] != n_anchors:
        raise ValueError(
            f"target_index has {target_array.shape[0]} entries but there are "
            f"{n_anchors} anchors"
        )
    if not np.issubdtype(target_array.dtype, np.integer):
        raise ValueError("target_index must contain integer candidate indices")
    if target_array.size and (
        (target_array < 0).any() or (target_array >= n_candidates).any()
    ):
        raise ValueError("target_index values must lie in [0, K)")
    return target_array.astype(np.int64)


def _prepare_threshold_grid(
    threshold_grid: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    """Validate and sort the candidate decision thresholds.

    Args:
        threshold_grid: Optional thresholds in ``[0, 1]``. ``None`` uses
            :data:`DEFAULT_THRESHOLD_GRID`. The returned grid is sorted
            ascending, so a later strict-improvement comparison keeps the
            smaller threshold on ties. The grid boundaries have direct
            meanings: ``0.0`` marks every candidate positive, while ``1.0``
            marks only candidates whose probability is exactly one.

    Returns:
        One-dimensional sorted float64 threshold array.

    Raises:
        ValueError: If the grid is empty, not one-dimensional, non-finite, or
            outside ``[0, 1]``.
    """
    if threshold_grid is None:
        threshold_array = np.asarray(DEFAULT_THRESHOLD_GRID, dtype=np.float64)
    else:
        threshold_array = np.asarray(threshold_grid, dtype=np.float64)

    if threshold_array.ndim != 1 or threshold_array.size == 0:
        raise ValueError("threshold_grid must be a non-empty one-dimensional grid")
    if not np.isfinite(threshold_array).all():
        raise ValueError("threshold_grid must contain only finite values")
    if ((threshold_array < 0.0) | (threshold_array > 1.0)).any():
        raise ValueError("threshold_grid values must lie in [0, 1]")
    return np.sort(threshold_array)


def _candidate_level_metrics(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Compute candidate-level precision, recall, and F1 at one threshold.

    Each anchor contributes exactly one positive pair (its true match) and
    ``K - 1`` negative pairs. A pair is predicted positive when its
    probability is greater than or equal to the threshold.

    Args:
        probabilities: ``(N, K)`` validated probability array.
        target_indices: ``(N,)`` validated true-match indices.
        threshold: Decision threshold in ``[0, 1]``.

    Returns:
        ``(precision, recall, f1)``, each in ``[0, 1]``. Precision is zero when
        nothing is predicted positive; F1 is zero when precision plus recall is
        zero.
    """
    n_anchors, n_candidates = probabilities.shape
    true_positive_mask = np.zeros((n_anchors, n_candidates), dtype=bool)
    true_positive_mask[np.arange(n_anchors), target_indices] = True
    predicted_positive_mask = probabilities >= threshold

    true_positives = int(np.count_nonzero(predicted_positive_mask & true_positive_mask))
    predicted_positives = int(np.count_nonzero(predicted_positive_mask))

    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / n_anchors
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return float(precision), float(recall), float(f1)


def select_threshold_and_metrics(
    probabilities: torch.Tensor | np.ndarray,
    target_index: torch.Tensor | np.ndarray | Sequence[int],
    threshold_grid: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Select a candidate-level threshold by F1 and compute all metrics.

    The threshold is chosen on the supplied probabilities by maximising F1 over
    the fixed validation grid. Ties are resolved in favour of the smaller
    threshold because the sorted grid is scanned in ascending order and only a
    strict F1 improvement replaces the current choice. The same probabilities
    are then evaluated at the selected threshold.

    Candidate-level evaluation treats each ``(anchor, candidate)`` pair as one
    binary example: the true match is positive and every other candidate is
    negative. Consequently ``n_candidates`` is the per-anchor ``K``, not the
    total number of pairs ``n_anchors * K``.

    Args:
        probabilities: ``(N, K)`` softmax probabilities. Rows need not be
            exactly normalised, but every entry must lie in ``[0, 1]``.
        target_index: ``(N,)`` true-match candidate index per anchor.
        threshold_grid: Thresholds in ``[0, 1]`` to search. ``None`` uses
            :data:`DEFAULT_THRESHOLD_GRID`.

    Returns:
        Dict with keys ``top1_accuracy``, ``precision``, ``recall``, ``f1``,
        ``threshold``, ``nll``, ``n_candidates``, and ``n_anchors``.
        ``top1_accuracy`` is the fraction of anchors whose modal candidate is
        the true match. ``nll`` is the mean negative log-likelihood of the true
        candidate per anchor; true probabilities are clamped at the smallest
        positive float64 value to keep underflowed softmax values finite.

    Raises:
        ValueError: If there are zero anchors, ``K`` is zero, shapes disagree,
            probabilities or thresholds are invalid, or target indices are out
            of range. A zero-anchor call cannot define top-1 accuracy,
            precision, recall, or F1, so it raises rather than returning
            misleading sentinel values. ``K = 1`` is valid: the single
            candidate is always the true match and, for the default grid, the
            metrics are all one.
    """
    probability_array = _as_probability_array(probabilities)
    n_anchors, n_candidates = probability_array.shape
    if n_anchors == 0:
        raise ValueError("at least one anchor is required to compute metrics")
    if n_candidates == 0:
        raise ValueError("at least one candidate per anchor is required")

    target_indices = _as_target_indices(target_index, n_anchors, n_candidates)
    threshold_array = _prepare_threshold_grid(threshold_grid)

    best_threshold = float(threshold_array[0])
    best_f1 = -1.0
    for threshold in threshold_array:
        _, _, f1 = _candidate_level_metrics(
            probability_array, target_indices, float(threshold)
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    precision, recall, f1 = _candidate_level_metrics(
        probability_array, target_indices, best_threshold
    )
    top1_matches = np.argmax(probability_array, axis=1) == target_indices
    top1_accuracy = float(np.mean(top1_matches))

    true_probabilities = probability_array[np.arange(n_anchors), target_indices]
    clipped_true_probabilities = np.clip(
        true_probabilities, np.finfo(np.float64).tiny, 1.0
    )
    nll = float(-np.mean(np.log(clipped_true_probabilities)))

    return {
        "top1_accuracy": top1_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_threshold,
        "nll": nll,
        "n_candidates": int(n_candidates),
        "n_anchors": int(n_anchors),
    }


def evaluate_knn_probe(
    model: nn.Module,
    anchor_features: torch.Tensor,
    candidate_features: torch.Tensor,
    target_index: torch.Tensor,
    threshold: float | None = None,
    validation_data: ProbeBatch | None = None,
) -> dict[str, float | int]:
    """Evaluate a probe, selecting the threshold on validation data if needed.

    Args:
        model: Trained K-nearest-neighbour probe.
        anchor_features: ``(N, K, D)`` evaluation anchor features.
        candidate_features: ``(N, K, D)`` evaluation candidate features.
        target_index: ``(N,)`` evaluation true-match indices.
        threshold: Optional pre-selected decision threshold. When given, it is
            applied directly to the evaluation probabilities.
        validation_data: Optional validation batch as a mapping with keys
            ``anchor_features``, ``candidate_features``, and ``target_index``,
            or a length-3 sequence in that order. Used only when ``threshold``
            is ``None``.

    Returns:
        The metric dict from :func:`select_threshold_and_metrics` evaluated on
        the supplied evaluation data at either the supplied or selected
        threshold.

    Warns:
        UserWarning: If ``threshold`` is ``None`` and ``validation_data`` is
            also ``None``, the threshold is selected on the evaluation data
            itself. That leaks evaluation labels into threshold selection and
            must only be used for smoke tests or debugging, never for reported
            held-out results.
    """
    if threshold is None:
        if validation_data is None:
            warnings.warn(
                "threshold=None without validation_data: selecting the "
                "threshold on the evaluation data leaks labels; use this only "
                "for smoke tests or debugging, never for held-out results",
                UserWarning,
                stacklevel=2,
            )
            validation_anchor_features = anchor_features
            validation_candidate_features = candidate_features
            validation_target_index = target_index
        else:
            (
                validation_anchor_features,
                validation_candidate_features,
                validation_target_index,
            ) = _unpack_batch(validation_data)

        validation_probabilities = predict_proba(
            model, validation_anchor_features, validation_candidate_features
        )
        validation_metrics = select_threshold_and_metrics(
            validation_probabilities, validation_target_index
        )
        threshold = float(validation_metrics["threshold"])

    evaluation_probabilities = predict_proba(
        model, anchor_features, candidate_features
    )
    return select_threshold_and_metrics(
        evaluation_probabilities,
        target_index,
        threshold_grid=[threshold],
    )
