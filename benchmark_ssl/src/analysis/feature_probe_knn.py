"""CLI for the Phase-1 K-NN feature probe (SPEC 0046, ticket F05).

This module implements both Phase-1 protocols:

* ``frame_cv`` — within-experiment consecutive-frame-pair cross-validation.
* ``loco`` — leave-one-condition-out; for every held-out condition the probe
  trains on all loaded pairs from the other conditions and evaluates on the
  held-out condition.

The shallow/handcrafted rows ``regionprops_7d``, ``regionprops_13d``,
``hu_20d`` and ``posenc`` are available, together with the frozen appearance
rows ``dino`` (DINOv2 ``dinov2_vits14``), ``cnn_frozen`` (DynaCLR ScaledCNN),
``cnn_ntxent`` (NT-Xent ScaledCNN) and ``sam21``
(SAM2.1-Hiera-B+, ``facebook/sam2.1-hiera-base-plus``).  The ``sam21`` row uses
the optional ``sam2`` package through
:class:`src.analysis.sam21_features.Sam21FeatureExtractor`; when that package is
missing the CLI exits with the adapter's install hint instead of an import
traceback.

Design summary
--------------
For every consecutive frame pair the anchor cells of frame ``t`` are matched
against the cells of frame ``t + 1``:

1. Candidate sets are built from centroids only with
   :func:`src.analysis.knn_candidates.build_knn_candidate_set`.
2. K is selected from a coverage-vs-K diagnostic on each fold's *training*
   pairs.  For ``frame_cv`` the folds are frame-pair folds; for ``loco`` each
   fold holds out one condition and its training pairs are all loaded pairs
   from the other conditions.  The protocol K is the maximum of the per-fold
   chosen K values so the same K is shared by every feature row and probe.  If
   automatic selection cannot reach full coverage at the largest searched K,
   the run stops with :class:`InsufficientCoverageError` instead of silently
   weakening the protocol.
3. Each fold's training pairs are split deterministically (80/20 by pair id by
   default) into train-internal and validation-internal parts.  The population
   z-normalizer is fitted on the train-internal candidate population only,
   early stopping uses validation-internal NLL, and the decision threshold is
   selected on validation-internal F1.  The held-out fold (or held-out
   condition) is used *only* for final evaluation, never for K, normalization,
   early stopping, or threshold selection.

The exact internal split is recorded in every per-run JSON so a reader can
audit the leakage boundary.  Fold assignment and the internal split use fixed
seeds that are independent of the probe seed, keeping the folds comparable
across seeds 42, 43 and 44.

Frozen appearance features are loaded once per run.  ``dino`` reuses
``extract_patches`` and ``compute_dino_embs`` from
``src.edge_probing.harness``; the CNN rows load their ScaledCNN checkpoint
once, run it in eval mode under ``torch.no_grad()``, and record the checkpoint
path and MD5 in every per-run JSON.

CLI
---
Run from ``benchmark_ssl/``.  A shallow frame-CV smoke run::

    python -m src.analysis.feature_probe_knn \\
      --data-root ../data/vanvliet \\
      --conditions rpsM \\
      --max-pairs-per-condition 6 \\
      --protocol frame_cv \\
      --features regionprops_7d,posenc \\
      --probes linear,mlp \\
      --seeds 42 \\
      --cv-folds 3 \\
      --epochs 2 \\
      --k auto \\
      --out results/probes/knn/smoke

A tiny local DINOv2 smoke run on the 4 GB A500::

    .venv/bin/python -m src.analysis.feature_probe_knn \\
      --data-root ../data/vanvliet \\
      --conditions rpsM \\
      --max-pairs-per-condition 4 \\
      --protocol frame_cv \\
      --features dino \\
      --probes linear \\
      --seeds 42 \\
      --cv-folds 2 \\
      --epochs 2 \\
      --k auto \\
      --out results/probes/knn/smoke_dino

A leave-one-condition-out run; by default every loaded condition is held out
once, and ``--heldout-conditions`` can restrict the evaluation to a subset::

    python -m src.analysis.feature_probe_knn \\
      --data-root ../data/vanvliet \\
      --protocol loco \\
      --conditions rpsM,recA \\
      --heldout-conditions recA \\
      --max-pairs-per-condition 3 \\
      --features regionprops_7d \\
      --probes linear \\
      --seeds 42 \\
      --epochs 2 \\
      --k auto \\
      --out results/probes/knn/smoke_loco

Outputs
-------
``<out>/coverage_vs_k.json``
    Feature-agnostic coverage diagnostic with one entry per fold: the chosen
    K, the coverage rows for ``K = 1 .. K_max``, and the held-out and training
    pair ids used by that fold.
``<out>/runs/<feature>_<probe>_s<seed>.json``
    One run artifact with per-fold metrics, chosen K, thresholds, arguments,
    versions, environment, data scope, checkpoint identity, and the
    leakage-boundary pair ids.
``<out>/summary.csv``
    One row per feature/probe/seed with the mean and standard deviation of
    top-1 accuracy, precision, recall, F1 and NLL across folds.  ``protocol``
    distinguishes ``frame_cv`` from ``loco``; ``heldout_condition`` names the
    LOCO condition (``overall`` for the all-condition summary row and an empty
    value for ``frame_cv``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from src.analysis.knn_candidates import (
    build_knn_candidate_set,
    choose_k,
    coverage_vs_k,
)
from src.analysis.knn_normalization import (
    apply_population_standardizer,
    fit_population_standardizer,
)
from src.analysis.knn_probe import (
    KnnLinearProbe,
    KnnMLPProbe,
    predict_proba,
    select_threshold_and_metrics,
    train_knn_probe,
)
from src.edge_probing.harness.feature_extractors import (
    POSENC_DIM,
    POSENC_N_FREQS,
    ScaledCNN,
    compute_dino_embs,
    extract_posenc,
    extract_regionprops_7d,
)
from src.edge_probing.harness.frame_data import (
    PATCH_SIZE as FRAME_PATCH_SIZE,
    extract_patches,
    load_frame,
    scan_consecutive_pairs,
)
from src.models.scaled_cnn import checkpoint_uses_batch_norm

logger = logging.getLogger("feature_probe_knn")

#: Feature rows implemented by this module, in the order reported in outputs.
SUPPORTED_FEATURES: tuple[str, ...] = (
    "regionprops_7d",
    "regionprops_13d",
    "hu_20d",
    "posenc",
    "dino",
    "cnn_frozen",
    "cnn_ntxent",
    "sam21",
)
#: Feature rows reserved for a later ticket.  None remain after F05c; the tuple
#: is kept so the CLI can still fail with a precise message if one is added.
DEFERRED_FEATURES: tuple[str, ...] = ()
SUPPORTED_PROBES: tuple[str, ...] = ("linear", "mlp")
IMPLEMENTED_PROTOCOLS: tuple[str, ...] = ("frame_cv", "loco")
DEFERRED_PROTOCOLS: tuple[str, ...] = ()

#: Repository root of the ``benchmark_ssl`` project, used to resolve the
#: frozen CNN checkpoints independently of the caller's working directory.
BENCHMARK_SSL_ROOT = Path(__file__).resolve().parents[2]
#: Frozen ScaledCNN checkpoints for the two CNN feature rows.  The NT-Xent
#: path is the one already used by ``unified_edge_probe`` and the LOCO slurm
#: scripts (``results/checkpoints/cnn/cnn_ntxent_large.pt``).
CNN_CHECKPOINT_PATHS: dict[str, Path] = {
    "cnn_frozen": BENCHMARK_SSL_ROOT
    / "results"
    / "checkpoints"
    / "cnn_bn100k_step100000.pt",
    "cnn_ntxent": BENCHMARK_SSL_ROOT
    / "results"
    / "checkpoints"
    / "cnn"
    / "cnn_ntxent_large.pt",
}
#: Feature rows that require a frozen local CNN checkpoint.
CNN_FEATURES: frozenset[str] = frozenset(CNN_CHECKPOINT_PATHS)
#: DINOv2 hub identity recorded in the run JSON.  No local checkpoint file is
#: owned by this repository; the weights live in the torch hub cache.
DINO_HUB_PATH = "torch.hub:facebookresearch/dinov2:dinov2_vits14"
#: Hugging Face SAM2.1 hub identity recorded in the run JSON.  The weights are
#: downloaded by the optional ``sam2`` package, not owned by this repository.
SAM21_HUB_PATH = "facebook/sam2.1-hiera-base-plus"
#: Per-patch CNN embedding dimensionality and DINOv2 embedding dimensionality.
CNN_FEATURE_DIM = 128
DINO_FEATURE_DIM = 384
#: Inference batch sizes for the frozen appearance extractors.  DINOv2 uses a
#: small batch because the 64x64 patches are resized to 224x224 on the 4 GB
#: local A500.
DINO_INFERENCE_BATCH_SIZE = 16
CNN_INFERENCE_BATCH_SIZE = 64

#: Default conditions for each protocol.  The frame-CV default is unchanged
#: from the original F05 slice; LOCO intentionally defaults to all six
#: vanvliet conditions.
DEFAULT_FRAME_CV_CONDITIONS: tuple[str, ...] = ("rpsM",)
DEFAULT_LOCO_CONDITIONS: tuple[str, ...] = (
    "rpsM",
    "recA",
    "pheA",
    "metA",
    "cib",
    "trpL",
)

#: Experiments excluded by the existing harness (report Section 2.11).  The
#: exclusion is inherited here and recorded in every run's data scope.
EXCLUDED_EXPERIMENTS: frozenset[str] = frozenset({"150303-08_BAD"})

#: Largest K searched by the coverage diagnostic, matching SPEC 0046 section
#: 4.2 ("capped at 64 if larger").
DEFAULT_K_MAX = 64
#: Global cap passed to :func:`scan_consecutive_pairs` when locating the full
#: consecutive-pair list.  It is deliberately far above any vanvliet sequence
#: length so the per-condition sampling sees the whole sequence; the effective
#: number of loaded pairs is still governed by ``max_pairs_per_condition``.
MAX_SCANNED_CONSECUTIVE_PAIRS = 1_000_000
#: Oversampling factor used when building the candidate pool for each
#: condition.  Up to three times the requested pair count is inspected so
#: sparse early frames can be dropped without losing the target sample size.
PAIR_SAMPLE_OVERSAMPLE_FACTOR = 3
#: Held-out fraction of each fold's *training* pairs used for early stopping
#: and threshold selection.  The final held-out fold is never used here.
DEFAULT_VALIDATION_FRACTION = 0.2
#: Fixed seed for fold assignment; independent of the probe seed so K and the
#: leakage boundary are identical across seeds.
FOLD_ASSIGNMENT_SEED = 42
#: Fixed seed for the train-internal / validation-internal split.
INTERNAL_SPLIT_SEED = 42
#: Cutoff passed to the existing ``extract_posenc`` helper.  Recorded in run
#: artifacts because it defines the position-only encoding.
POSENC_CUTOFF = 128.0
#: Artifact schema version for per-run JSON files.  Version 2 adds the
#: protocol/heldout-condition summary columns and checkpoint identity.
RUN_SCHEMA_VERSION = 2

#: Column order for ``summary.csv``.  Every metric has a mean and a standard
#: deviation across folds.  ``heldout_condition`` is empty for ``frame_cv``,
#: names the condition for per-condition LOCO rows, and is ``"overall"`` for
#: the LOCO all-condition row.
SUMMARY_FIELDNAMES: tuple[str, ...] = (
    "protocol",
    "feature",
    "probe",
    "seed",
    "heldout_condition",
    "n_folds",
    "protocol_k",
    "threshold_mean",
    "threshold_std",
    "top1_accuracy_mean",
    "top1_accuracy_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "nll_mean",
    "nll_std",
)

#: Metric keys reported per fold and aggregated in the summary.
METRIC_KEYS: tuple[str, ...] = (
    "top1_accuracy",
    "precision",
    "recall",
    "f1",
    "nll",
)

PairRecord = dict[str, Any]
CandidateDataset = dict[str, Any]
FeatureExtractor = Callable[[np.ndarray, np.ndarray], tuple[Any, Any, Any]]


def checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, str]:
    """Return the path and content MD5 of a frozen checkpoint file.

    The MD5 is used as a content identity for reproducibility, not as a
    security mechanism.  Hashing is streamed so a multi-gigabyte checkpoint
    does not need to be loaded into memory.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Returns:
        ``{"path": str(path), "md5": hexdigest}``.

    Raises:
        FileNotFoundError: If the path does not exist or is not a file.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {path}")

    digest = hashlib.md5()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "md5": digest.hexdigest()}


class FeatureExtractorContext:
    """Load frozen appearance models once per run and reuse them for every pair.

    The DINOv2 model is cached globally inside
    :func:`src.edge_probing.harness.feature_extractors.compute_dino_embs`; this
    context owns the two ScaledCNN checkpoints and the optional SAM2.1
    extractor so switching between feature rows never reloads a model per
    frame pair.

    Args:
        device: Device used for ScaledCNN and SAM2.1 inference.
    """

    def __init__(self, device: torch.device | str) -> None:
        """Store the inference device and initialise empty model caches."""
        self.device = torch.device(device)
        self._cnn_models: dict[str, torch.nn.Module] = {}
        self._checkpoint_records: dict[str, dict[str, Any]] = {}
        self._sam21_extractor: Any | None = None

    def cnn_model(self, feature_name: str) -> torch.nn.Module:
        """Load and cache the ScaledCNN checkpoint for one CNN feature row.

        Args:
            feature_name: ``"cnn_frozen"`` or ``"cnn_ntxent"``.

        Returns:
            The checkpointed model in eval mode on :attr:`device`.

        Raises:
            ValueError: If the feature row has no configured CNN checkpoint.
            FileNotFoundError: If the checkpoint file is missing.
        """
        if feature_name not in CNN_CHECKPOINT_PATHS:
            raise ValueError(f"{feature_name!r} is not a CNN feature row")
        cached_model = self._cnn_models.get(feature_name)
        if cached_model is not None:
            return cached_model

        checkpoint_path = CNN_CHECKPOINT_PATHS[feature_name]
        checkpoint_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint_state, dict) or "model_state_dict" not in (
            checkpoint_state
        ):
            raise ValueError(
                f"checkpoint {checkpoint_path} does not contain 'model_state_dict'"
            )
        state_dict = checkpoint_state["model_state_dict"]
        model_scale = str(checkpoint_state.get("scale", "large"))
        model_out_dim = int(checkpoint_state.get("out_dim", CNN_FEATURE_DIM))
        if model_out_dim != CNN_FEATURE_DIM:
            raise ValueError(
                f"checkpoint {checkpoint_path} has out_dim={model_out_dim}, "
                f"expected {CNN_FEATURE_DIM}"
            )
        uses_batch_norm = bool(
            checkpoint_state.get(
                "batch_norm", checkpoint_uses_batch_norm(state_dict)
            )
        )
        model = ScaledCNN(
            scale=model_scale,
            out_dim=CNN_FEATURE_DIM,
            batch_norm=uses_batch_norm,
        )
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        self._cnn_models[feature_name] = model
        self._checkpoint_records[feature_name] = {
            **checkpoint_metadata(checkpoint_path),
            "status": "loaded",
            "feature": feature_name,
            "scale": model_scale,
            "out_dim": int(CNN_FEATURE_DIM),
            "batch_norm": uses_batch_norm,
        }
        logger.info(
            "loaded %s checkpoint %s (md5=%s)",
            feature_name,
            checkpoint_path,
            self._checkpoint_records[feature_name]["md5"],
        )
        return model

    def sam21_extractor(self) -> Any:
        """Load and cache the optional SAM2.1 feature extractor.

        The adapter module is imported lazily so a missing ``sam2`` package
        only matters when the ``sam21`` row is actually requested.  One
        extractor is reused for every frame pair, mirroring the CNN cache.

        Returns:
            The constructed
            :class:`src.analysis.sam21_features.Sam21FeatureExtractor`.

        Raises:
            ImportError: If ``sam2`` or the vendored Trackastra fork is
                unavailable.  The message contains the adapter's install hint.
        """
        if self._sam21_extractor is not None:
            return self._sam21_extractor

        from src.analysis.sam21_features import Sam21FeatureExtractor

        self._sam21_extractor = Sam21FeatureExtractor(device=str(self.device))
        return self._sam21_extractor

    def checkpoint_record(self, feature_name: str) -> dict[str, Any]:
        """Return the checkpoint identity block for one feature row.

        Shallow and position-only rows get a ``not_applicable`` block, DINOv2
        and SAM2.1 record their pretrained-hub identity (no repository-owned
        file to hash), and CNN rows record the actual checkpoint path and MD5.
        Calling this for a CNN row loads the model if it is not loaded yet, so
        the hash refers to the exact file inference will use.

        Args:
            feature_name: Feature row name.

        Returns:
            A JSON-serialisable checkpoint mapping.
        """
        if feature_name in CNN_FEATURES:
            if feature_name not in self._checkpoint_records:
                self.cnn_model(feature_name)
            return dict(self._checkpoint_records[feature_name])
        if feature_name == "dino":
            return {
                "status": "pretrained_hub",
                "feature": feature_name,
                "path": DINO_HUB_PATH,
                "md5": None,
                "note": (
                    "weights are cached by torch.hub; no repository-owned "
                    "checkpoint file is hashed"
                ),
            }
        if feature_name == "sam21":
            return {
                "status": "pretrained_hub",
                "feature": feature_name,
                "path": SAM21_HUB_PATH,
                "md5": None,
                "note": (
                    "weights are downloaded from the Hugging Face hub; no "
                    "repository-owned checkpoint file is hashed"
                ),
            }
        return {
            "status": "not_applicable",
            "feature": feature_name,
            "path": None,
            "md5": None,
            "note": "shallow/position feature row; no frozen checkpoint is used",
        }

    def close(self) -> None:
        """Release every cached model and the GPU memory it occupies.

        The SAM2.1 adapter's own :meth:`close` is called when it was
        constructed, then the CNN model cache is dropped.  Safe to call when no
        model was ever loaded.
        """
        if self._sam21_extractor is not None:
            self._sam21_extractor.close()
            self._sam21_extractor = None
        self._cnn_models.clear()
        self._checkpoint_records.clear()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


#: Registry for feature rows whose extractor is importable without the legacy
#: analysis package.  ``posenc`` is handled separately because it consumes
#: coordinates, and the legacy 13D/Hu extractors are imported lazily inside
#: ``_extract_feature_frame`` to avoid the package's import-time DINOv2 load.
_FEATURE_EXTRACTORS: dict[str, FeatureExtractor] = {
    "regionprops_7d": extract_regionprops_7d,
}


class InsufficientCoverageError(RuntimeError):
    """Raised when automatic K selection cannot reach full candidate coverage.

    The exception carries the fold name, the largest searched K, the achieved
    coverage, and the residual miss rate so the CLI can escalate to the owner
    with an actionable message rather than silently changing the protocol.
    """


def _parse_k(value: str) -> str | int:
    """Parse the ``--k`` argument as ``"auto"`` or a positive integer.

    Args:
        value: Raw command-line value.

    Returns:
        The string ``"auto"`` or a positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is neither ``"auto"`` nor a
            positive integer.
    """
    if value == "auto":
        return "auto"
    try:
        parsed_value = int(value)
    except ValueError as conversion_error:
        raise argparse.ArgumentTypeError(
            f"--k must be 'auto' or a positive integer, got {value!r}"
        ) from conversion_error
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            f"--k must be positive, got {parsed_value}"
        )
    return parsed_value


def _parse_comma_separated(value: str) -> list[str]:
    """Split a comma-separated CLI value and drop empty entries.

    Args:
        value: Raw comma-separated string.

    Returns:
        The non-empty, whitespace-stripped tokens in input order.
    """
    return [token.strip() for token in value.split(",") if token.strip()]


def validate_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Validate and de-duplicate requested feature rows in input order.

    Args:
        feature_names: Feature row names from the CLI.

    Returns:
        The validated feature names without duplicates, preserving the first
        occurrence order.

    Raises:
        NotImplementedError: If a row is a known deferred row.
        ValueError: If the request is empty or contains an unknown row.
    """
    if not feature_names:
        raise ValueError("at least one feature row must be requested")

    validated_names: list[str] = []
    for feature_name in feature_names:
        if feature_name in validated_names:
            continue
        if feature_name in SUPPORTED_FEATURES:
            validated_names.append(feature_name)
            continue
        if feature_name in DEFERRED_FEATURES:
            raise NotImplementedError(
                f"feature row {feature_name!r} is not implemented in this F05 "
                "slice; the SAM2.1 row belongs to the next slice (F06)"
            )
        raise ValueError(
            f"unknown feature row {feature_name!r}; supported rows are "
            f"{', '.join(SUPPORTED_FEATURES)}"
        )
    return validated_names


def validate_protocol(protocol: str) -> str:
    """Validate that the requested protocol is implemented.

    Args:
        protocol: Protocol name from the CLI.

    Returns:
        The validated protocol name.

    Raises:
        NotImplementedError: If the protocol is a known deferred protocol.
        ValueError: If the protocol name is unknown.
    """
    if protocol in IMPLEMENTED_PROTOCOLS:
        return protocol
    if protocol in DEFERRED_PROTOCOLS:
        raise NotImplementedError(
            f"protocol {protocol!r} is not implemented in this slice"
        )
    raise ValueError(
        f"unknown protocol {protocol!r}; implemented protocols are "
        f"{', '.join(IMPLEMENTED_PROTOCOLS)}"
    )


def validate_probe_names(probe_names: Sequence[str]) -> list[str]:
    """Validate and de-duplicate requested probe families.

    Args:
        probe_names: Probe names from the CLI.

    Returns:
        The validated probe names without duplicates.

    Raises:
        ValueError: If the request is empty or contains an unknown probe.
    """
    if not probe_names:
        raise ValueError("at least one probe must be requested")

    validated_names: list[str] = []
    for probe_name in probe_names:
        if probe_name in validated_names:
            continue
        if probe_name not in SUPPORTED_PROBES:
            raise ValueError(
                f"unknown probe {probe_name!r}; supported probes are "
                f"{', '.join(SUPPORTED_PROBES)}"
            )
        validated_names.append(probe_name)
    return validated_names


def resolve_device(device_name: str) -> torch.device:
    """Resolve the CLI device string, including the ``"auto"`` shorthand.

    Args:
        device_name: ``"auto"``, ``"cpu"``, ``"cuda"`` or another torch device
            string.

    Returns:
        The resolved :class:`torch.device`.

    Raises:
        RuntimeError: If a CUDA device is requested but CUDA is unavailable.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_device = torch.device(device_name)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {device_name!r} was requested but CUDA is not available"
        )
    return resolved_device


def _json_default(value: Any) -> Any:
    """Convert NumPy, torch and path objects into JSON-serialisable values.

    Args:
        value: Object encountered by :func:`json.dump`.

    Returns:
        A JSON-serialisable representation.

    Raises:
        TypeError: If the object has no documented conversion.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic, strict JSON artifact.

    Args:
        path: Destination path.  Parent directories are created.
        payload: JSON-serialisable mapping.  Non-finite floats raise instead of
            producing a file that reads back as ``NaN``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        output_file.write("\n")


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the cross-run metric summary.

    Args:
        path: Destination CSV path.  Parent directories are created.
        rows: One mapping per feature/probe/seed run.  Every mapping must use
            exactly :data:`SUMMARY_FIELDNAMES`; extra keys are an error so a
            silent schema drift cannot happen.

    Raises:
        ValueError: If a row is missing a required field or has an extra one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(SUMMARY_FIELDNAMES),
            extrasaction="raise",
        )
        writer.writeheader()
        for row in rows:
            missing_fields = [field for field in SUMMARY_FIELDNAMES if field not in row]
            if missing_fields:
                raise ValueError(
                    f"summary row is missing field(s): {', '.join(missing_fields)}"
                )
            writer.writerow(row)


def _environment_record(resolved_device: torch.device) -> dict[str, Any]:
    """Collect the runtime environment fields needed to reproduce a run.

    Args:
        resolved_device: Device actually used by the probe.

    Returns:
        A JSON-serialisable environment mapping.
    """
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(resolved_device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        environment["cuda_device_name"] = torch.cuda.get_device_name(
            resolved_device
        )
    return environment


def _pair_identifier(
    condition: str,
    experiment: str,
    anchor_mask_path: str,
    candidate_mask_path: str,
) -> str:
    """Build a stable, human-readable identifier for one frame pair.

    Args:
        condition: Vanvliet condition name.
        experiment: Experiment directory name.
        anchor_mask_path: Path to frame ``t``'s mask.
        candidate_mask_path: Path to frame ``t + 1``'s mask.

    Returns:
        An identifier such as ``rpsM/151029_E1-1/man_track000002->man_track000003``.
    """
    anchor_stem = Path(anchor_mask_path).stem
    candidate_stem = Path(candidate_mask_path).stem
    return f"{condition}/{experiment}/{anchor_stem}->{candidate_stem}"


def _evenly_spaced_indices(n_items: int, n_select: int) -> list[int]:
    """Return up to ``n_select`` evenly spaced indices over ``n_items`` items.

    Called by :func:`load_base_pairs` to spread the sampled frame pairs across
    each condition's full consecutive-pair sequence instead of taking the first
    N pairs.  The first and last item are selected whenever at least two
    indices are requested, so the sample covers the whole sequence.

    Args:
        n_items: Number of items in the full sequence.
        n_select: Desired number of indices.

    Returns:
        Strictly increasing unique indices in ``[0, n_items)``.  All indices
        are returned when ``n_select >= n_items``; an empty list is returned
        when either count is not positive.
    """
    if n_items <= 0 or n_select <= 0:
        return []
    if n_select >= n_items:
        return list(range(n_items))

    raw_indices = np.linspace(0, n_items - 1, num=n_select).round().astype(int)
    return sorted({int(index) for index in raw_indices})


def load_base_pairs(
    data_root: str | Path,
    conditions: Sequence[str],
    max_pairs_per_condition: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan and load consecutive frame pairs without extracting features.

    The returned records hold only coordinates, labels, images and masks.  They
    are the feature-agnostic basis for fold assignment and the coverage-vs-K
    diagnostic, so K selection never depends on appearance features.

    Pairs are sampled evenly across each condition's full consecutive-pair
    list rather than taking the first N pairs, because early frames are sparser
    and do not represent later candidate density.  A candidate pool of up to
    three times the requested count is inspected so sparse pairs can be dropped
    without shrinking the usable sample when denser pairs are available.

    Args:
        data_root: Root containing the vanvliet condition directories.
        conditions: Condition names to scan, in order.
        max_pairs_per_condition: Target maximum number of usable pairs per
            condition.

    Returns:
        ``(base_pairs, scan_statistics)`` where each base pair has keys
        ``pair_id``, ``condition``, ``experiment``, ``anchor`` and ``candidate``
        (each frame has ``coords``, ``labels``, ``img`` and ``mask``).

    Raises:
        ValueError: If the root or a requested condition directory is missing,
            or if no usable pair is found.
    """
    root = Path(data_root)
    if not root.is_dir():
        raise ValueError(f"data root does not exist or is not a directory: {root}")

    for condition in conditions:
        if not (root / condition).is_dir():
            raise ValueError(
                f"condition directory not found under {root}: {condition}"
            )

    scanned_pairs = scan_consecutive_pairs(
        str(root),
        list(conditions),
        MAX_SCANNED_CONSECUTIVE_PAIRS,
        max_pairs_per_condition=0,
    )

    pairs_by_condition: dict[str, list[tuple[str, ...]]] = {
        condition: [] for condition in conditions
    }
    excluded_pair_count = 0
    for scanned_pair in scanned_pairs:
        condition = scanned_pair[5]
        experiment = scanned_pair[6]
        if experiment in EXCLUDED_EXPERIMENTS:
            excluded_pair_count += 1
            continue
        pairs_by_condition[condition].append(scanned_pair)

    target_pairs_per_condition = max(1, int(max_pairs_per_condition))
    base_pairs: list[dict[str, Any]] = []
    pair_counts_by_condition: dict[str, int] = {}
    skipped_pairs: list[dict[str, str]] = []
    for condition in conditions:
        condition_pairs = pairs_by_condition[condition]
        candidate_pool_size = min(
            len(condition_pairs),
            target_pairs_per_condition * PAIR_SAMPLE_OVERSAMPLE_FACTOR,
        )
        candidate_indices = _evenly_spaced_indices(
            len(condition_pairs), candidate_pool_size
        )

        usable_condition_pairs: list[dict[str, Any]] = []
        for pair_index in candidate_indices:
            (
                anchor_mask_path,
                candidate_mask_path,
                anchor_image_path,
                candidate_image_path,
                _man_track_path,
                _pair_condition,
                experiment,
            ) = condition_pairs[pair_index]
            pair_id = _pair_identifier(
                condition, experiment, anchor_mask_path, candidate_mask_path
            )
            try:
                anchor_frame = load_frame(anchor_mask_path, anchor_image_path)
                candidate_frame = load_frame(
                    candidate_mask_path, candidate_image_path
                )
            except (OSError, ValueError) as loading_error:
                skipped_pairs.append(
                    {"pair_id": pair_id, "reason": str(loading_error)}
                )
                logger.warning("skipping pair %s: %s", pair_id, loading_error)
                continue

            if anchor_frame is None or candidate_frame is None:
                skip_reason = "fewer than two cells in at least one frame"
                skipped_pairs.append({"pair_id": pair_id, "reason": skip_reason})
                logger.warning("skipping pair %s: %s", pair_id, skip_reason)
                continue

            anchor_coords, anchor_labels, anchor_image, anchor_mask = anchor_frame
            candidate_coords, candidate_labels, candidate_image, candidate_mask = (
                candidate_frame
            )
            if min(anchor_coords.shape[0], candidate_coords.shape[0]) < 3:
                skip_reason = "fewer than three cells in at least one frame"
                skipped_pairs.append({"pair_id": pair_id, "reason": skip_reason})
                logger.warning("skipping pair %s: %s", pair_id, skip_reason)
                continue

            usable_condition_pairs.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "experiment": experiment,
                    "anchor": {
                        "coords": anchor_coords,
                        "labels": anchor_labels,
                        "img": anchor_image,
                        "mask": anchor_mask,
                    },
                    "candidate": {
                        "coords": candidate_coords,
                        "labels": candidate_labels,
                        "img": candidate_image,
                        "mask": candidate_mask,
                    },
                }
            )

        selected_positions = _evenly_spaced_indices(
            len(usable_condition_pairs),
            min(
                len(usable_condition_pairs),
                target_pairs_per_condition,
            ),
        )
        for selected_position in selected_positions:
            base_pairs.append(usable_condition_pairs[selected_position])
            pair_counts_by_condition[condition] = (
                pair_counts_by_condition.get(condition, 0) + 1
            )

    if not base_pairs:
        raise ValueError(
            "no usable consecutive frame pairs were found for conditions "
            f"{', '.join(conditions)} under {root}"
        )

    scan_statistics = {
        "requested_condition_names": list(conditions),
        "scanned_pair_count": len(scanned_pairs),
        "excluded_pair_count": excluded_pair_count,
        "excluded_experiments": sorted(EXCLUDED_EXPERIMENTS),
        "loaded_pair_count": len(base_pairs),
        "skipped_pair_count": len(skipped_pairs),
        "pair_skips": skipped_pairs,
        "pair_counts_by_condition": pair_counts_by_condition,
    }
    logger.info(
        "loaded %d consecutive pairs from %d scanned pairs (%d excluded, %d skipped)",
        len(base_pairs),
        len(scanned_pairs),
        excluded_pair_count,
        len(skipped_pairs),
    )
    return base_pairs, scan_statistics


def _extract_feature_frame(
    feature_name: str,
    frame: Mapping[str, Any],
    context: FeatureExtractorContext,
) -> tuple[Any, Any, Any] | None:
    """Extract one feature row for one frame.

    Args:
        feature_name: Validated feature row name.
        frame: Frame mapping with ``coords``, ``labels``, ``img`` and ``mask``.
        context: Per-run extractor context that caches the frozen CNN models
            and exposes their checkpoint identity.

    Returns:
        ``(coords, labels, features)`` or ``None`` when the extractor reports
        an empty frame.  For ``posenc`` the coordinates and labels come from
        the frame and the features are the Fourier encoding of the centroids.
        For ``dino`` and the CNN rows the features are the frozen appearance
        embeddings of the patches centred on the frame's centroids.  For
        ``sam21`` the features are the frozen SAM2.1-Hiera-B+ vectors selected
        at the frame's centroids.

    Raises:
        ValueError: If the frame produced fewer labels than coordinates.
        ImportError: If ``sam21`` is requested and the optional ``sam2``
            package is unavailable.
    """
    if feature_name == "sam21":
        features = context.sam21_extractor().extract_frame(
            frame["img"], frame["coords"]
        )
        return frame["coords"], frame["labels"], features

    if feature_name == "posenc":
        return (
            frame["coords"],
            frame["labels"],
            extract_posenc(frame["coords"]),
        )

    if feature_name == "regionprops_13d":
        # Imported lazily: ``src.analysis.feature_probe`` sits in a package
        # whose __init__ eagerly loads DINOv2, which the shallow smoke run does
        # not need and which would otherwise consume accelerator memory.
        from src.analysis.feature_probe import extract_regionprops_13d

        return extract_regionprops_13d(frame["mask"], frame["img"])

    if feature_name == "hu_20d":
        # Imported lazily for the same reason as ``regionprops_13d``.
        from src.analysis.feature_probe import extract_hu_20d

        return extract_hu_20d(frame["mask"], frame["img"])

    if feature_name == "dino":
        patches = extract_patches(frame["img"], frame["coords"])
        features = compute_dino_embs(patches)
        return frame["coords"], frame["labels"], features

    if feature_name in CNN_FEATURES:
        patches = extract_patches(frame["img"], frame["coords"])
        patch_tensor = (
            torch.from_numpy(np.asarray(patches, dtype=np.float32))
            .unsqueeze(1)
            .to(context.device)
        )
        model = context.cnn_model(feature_name)
        with torch.no_grad():
            features = model(patch_tensor).cpu().numpy()
        return frame["coords"], frame["labels"], features

    extractor = _FEATURE_EXTRACTORS[feature_name]
    return extractor(frame["mask"], frame["img"])


def _validate_feature_frame(
    pair_id: str,
    role: str,
    coords: Any,
    labels: Any,
    features: Any,
) -> None:
    """Check that one frame's coordinates, labels and features agree.

    Args:
        pair_id: Pair identifier used in error messages.
        role: ``"anchor"`` or ``"candidate"``.
        coords: Extracted centroid array.
        labels: Extracted label array.
        features: Extracted feature matrix.

    Raises:
        ValueError: If shapes are not ``(N, 2)``, ``(N,)`` and ``(N, D)`` with a
            consistent ``N`` and ``D > 0``.
    """
    coordinate_array = np.asarray(coords)
    label_array = np.asarray(labels)
    feature_array = np.asarray(features)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 2:
        raise ValueError(
            f"{pair_id} {role} coords must have shape (N, 2), got "
            f"{coordinate_array.shape}"
        )
    if label_array.ndim != 1:
        raise ValueError(
            f"{pair_id} {role} labels must have shape (N,), got "
            f"{label_array.shape}"
        )
    if feature_array.ndim != 2 or feature_array.shape[1] == 0:
        raise ValueError(
            f"{pair_id} {role} features must have shape (N, D) with D > 0, got "
            f"{feature_array.shape}"
        )
    row_count = coordinate_array.shape[0]
    if label_array.shape[0] != row_count or feature_array.shape[0] != row_count:
        raise ValueError(
            f"{pair_id} {role} has mismatched rows: coords {row_count}, labels "
            f"{label_array.shape[0]}, features {feature_array.shape[0]}"
        )


def extract_feature_pairs(
    base_pairs: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    extractor_context: FeatureExtractorContext | None = None,
) -> tuple[dict[str, list[PairRecord]], dict[str, dict[str, Any]]]:
    """Extract every requested feature row for every loaded pair.

    One :class:`FeatureExtractorContext` is created for the call when the
    caller does not supply one, so a frozen CNN checkpoint is loaded at most
    once per extraction run rather than once per frame pair.

    Args:
        base_pairs: Feature-agnostic pair records from :func:`load_base_pairs`.
        feature_names: Validated feature row names.
        extractor_context: Optional per-run extractor context.  When omitted,
            a new context is created on the automatically resolved device.

    Returns:
        ``(pairs_by_feature, statistics_by_feature)``.  Each pair record has
        ``feature``, ``condition``, ``experiment``, ``pair_id``,
        ``anchor_features``, ``candidate_features``, ``anchor_coords``,
        ``candidate_coords``, ``anchor_labels`` and ``candidate_labels``.  Each
        statistics mapping records the number of extracted pairs and every
        skipped pair with its reason.

    Raises:
        ValueError: If an extractor returns inconsistent shapes.
    """
    if extractor_context is None:
        extractor_context = FeatureExtractorContext(resolve_device("auto"))

    pairs_by_feature: dict[str, list[PairRecord]] = {
        feature_name: [] for feature_name in feature_names
    }
    statistics_by_feature: dict[str, dict[str, Any]] = {}

    for feature_name in feature_names:
        skipped_pairs: list[dict[str, str]] = []
        for base_pair in base_pairs:
            pair_id = str(base_pair["pair_id"])
            try:
                anchor_extraction = _extract_feature_frame(
                    feature_name, base_pair["anchor"], extractor_context
                )
                candidate_extraction = _extract_feature_frame(
                    feature_name, base_pair["candidate"], extractor_context
                )
            except (OSError, ValueError) as extraction_error:
                skipped_pairs.append(
                    {"pair_id": pair_id, "reason": str(extraction_error)}
                )
                logger.warning(
                    "skipping feature %s pair %s: %s",
                    feature_name,
                    pair_id,
                    extraction_error,
                )
                continue

            if anchor_extraction is None or candidate_extraction is None:
                skip_reason = "feature extractor returned no cells"
                skipped_pairs.append({"pair_id": pair_id, "reason": skip_reason})
                logger.warning(
                    "skipping feature %s pair %s: %s",
                    feature_name,
                    pair_id,
                    skip_reason,
                )
                continue

            anchor_coords, anchor_labels, anchor_features = anchor_extraction
            candidate_coords, candidate_labels, candidate_features = (
                candidate_extraction
            )
            _validate_feature_frame(
                pair_id, "anchor", anchor_coords, anchor_labels, anchor_features
            )
            _validate_feature_frame(
                pair_id,
                "candidate",
                candidate_coords,
                candidate_labels,
                candidate_features,
            )

            pairs_by_feature[feature_name].append(
                {
                    "feature": feature_name,
                    "condition": base_pair["condition"],
                    "experiment": base_pair["experiment"],
                    "pair_id": pair_id,
                    "anchor_features": np.asarray(anchor_features, dtype=np.float32),
                    "candidate_features": np.asarray(
                        candidate_features, dtype=np.float32
                    ),
                    "anchor_coords": np.asarray(anchor_coords, dtype=np.float32),
                    "candidate_coords": np.asarray(
                        candidate_coords, dtype=np.float32
                    ),
                    "anchor_labels": np.asarray(anchor_labels, dtype=np.int64),
                    "candidate_labels": np.asarray(
                        candidate_labels, dtype=np.int64
                    ),
                }
            )

        statistics_by_feature[feature_name] = {
            "pair_count": len(pairs_by_feature[feature_name]),
            "skipped_pair_count": len(skipped_pairs),
            "pair_skips": skipped_pairs,
        }
        logger.info(
            "extracted feature %s for %d/%d pairs (%d skipped)",
            feature_name,
            len(pairs_by_feature[feature_name]),
            len(base_pairs),
            len(skipped_pairs),
        )

    return pairs_by_feature, statistics_by_feature


def assign_pair_folds(
    pair_ids: Sequence[str],
    requested_folds: int,
) -> tuple[dict[str, int], int]:
    """Assign pair ids to deterministic cross-validation folds.

    Fold assignment uses a fixed-seed permutation of the sorted pair ids,
    independent of the probe seed, so every feature/seed combination sees the
    same held-out pairs.  When fewer pairs than requested folds exist, the
    effective fold count is reduced and reported to the caller.

    Args:
        pair_ids: Pair identifiers to partition.
        requested_folds: Requested number of folds.

    Returns:
        ``(fold_by_pair_id, effective_fold_count)``.

    Raises:
        ValueError: If there are fewer than two unique pair ids or the
            requested fold count is not positive.
    """
    unique_pair_ids = sorted(set(pair_ids))
    if requested_folds <= 0:
        raise ValueError(f"cv_folds must be positive, got {requested_folds}")
    if len(unique_pair_ids) < 2:
        raise ValueError(
            "frame_cv needs at least two distinct frame pairs, got "
            f"{len(unique_pair_ids)}"
        )

    effective_fold_count = min(int(requested_folds), len(unique_pair_ids))
    if effective_fold_count != requested_folds:
        logger.warning(
            "requested %d CV folds but only %d pairs are available; using %d folds",
            requested_folds,
            len(unique_pair_ids),
            effective_fold_count,
        )

    generator = np.random.default_rng(FOLD_ASSIGNMENT_SEED)
    shuffled_positions = generator.permutation(len(unique_pair_ids))
    fold_by_pair_id = {
        unique_pair_ids[position]: int(shuffled_position % effective_fold_count)
        for shuffled_position, position in enumerate(shuffled_positions)
    }
    return fold_by_pair_id, effective_fold_count


def split_internal_training_pairs(
    training_pair_ids: Sequence[str],
    validation_fraction: float,
) -> tuple[list[str], list[str]]:
    """Split one fold's training pairs into train-internal and validation-internal.

    The split is by pair id, not by anchor, so no anchor from a validation pair
    leaks into training.  It uses a fixed seed independent of the probe seed.

    Args:
        training_pair_ids: Pair ids in the fold's training pool (the held-out
            fold must not be present).
        validation_fraction: Fraction assigned to validation-internal.

    Returns:
        ``(train_internal_pair_ids, validation_internal_pair_ids)``, both
        sorted for deterministic output.

    Raises:
        ValueError: If there are fewer than two training pairs or the fraction
            is not in ``(0, 1)``.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be strictly between 0 and 1, got "
            f"{validation_fraction}"
        )

    unique_pair_ids = sorted(set(training_pair_ids))
    if len(unique_pair_ids) < 2:
        raise ValueError(
            "each fold needs at least two training pairs for an internal "
            f"train/validation split, got {len(unique_pair_ids)}"
        )

    validation_count = max(1, int(round(validation_fraction * len(unique_pair_ids))))
    validation_count = min(validation_count, len(unique_pair_ids) - 1)
    generator = np.random.default_rng(INTERNAL_SPLIT_SEED)
    shuffled_positions = generator.permutation(len(unique_pair_ids))
    validation_positions = set(shuffled_positions[:validation_count].tolist())
    train_internal_ids = [
        pair_id
        for position, pair_id in enumerate(unique_pair_ids)
        if position not in validation_positions
    ]
    validation_internal_ids = [
        pair_id
        for position, pair_id in enumerate(unique_pair_ids)
        if position in validation_positions
    ]
    return sorted(train_internal_ids), sorted(validation_internal_ids)


def aggregate_coverage_rows(
    per_pair_coverage: Sequence[Sequence[Mapping[str, int | float]]],
    candidate_counts: Sequence[int],
    k_max: int,
) -> list[dict[str, int | float]]:
    """Aggregate per-pair coverage rows into one diagnostic curve.

    For each K the aggregate counts only pairs that actually have at least K
    candidates; a pair with fewer candidates cannot contribute a K-nearest
    candidate set at that K.  Anchors are therefore pooled over the pairs that
    remain evaluable, and ``coverage`` is anchor-weighted.

    Args:
        per_pair_coverage: One row sequence per pair, as returned by
            :func:`src.analysis.knn_candidates.coverage_vs_k`.
        candidate_counts: Number of candidates in the candidate frame of each
            pair, aligned with ``per_pair_coverage``.
        k_max: Largest K to report.  Rows are emitted for every integer from 1
            through ``k_max`` so the artifact has a fixed schema even when some
            pairs are smaller.

    Returns:
        Coverage rows with keys ``k``, ``coverage``, ``n_anchors``,
        ``n_covered``, ``n_missing_match`` and ``n_pairs``.

    Raises:
        ValueError: If the two input sequences differ in length or ``k_max`` is
            not positive.
    """
    if len(per_pair_coverage) != len(candidate_counts):
        raise ValueError(
            "per_pair_coverage and candidate_counts must have the same length"
        )
    if k_max <= 0:
        raise ValueError(f"k_max must be positive, got {k_max}")

    aggregated_rows: list[dict[str, int | float]] = []
    for k in range(1, int(k_max) + 1):
        total_covered = 0
        total_anchors = 0
        total_missing_match = 0
        contributing_pairs = 0
        for pair_rows, n_candidates in zip(per_pair_coverage, candidate_counts):
            if int(n_candidates) < k or len(pair_rows) < k:
                continue
            row = pair_rows[k - 1]
            total_covered += int(row["n_covered"])
            total_anchors += int(row["n_anchors"])
            total_missing_match += int(row["n_missing_match"])
            contributing_pairs += 1

        coverage = (
            float(total_covered) / float(total_anchors)
            if total_anchors > 0
            else 0.0
        )
        aggregated_rows.append(
            {
                "k": int(k),
                "coverage": coverage,
                "n_anchors": int(total_anchors),
                "n_covered": int(total_covered),
                "n_missing_match": int(total_missing_match),
                "n_pairs": int(contributing_pairs),
            }
        )
    return aggregated_rows


def select_protocol_k(
    fold_coverage_rows: Sequence[Sequence[Mapping[str, int | float]]],
    fold_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select one protocol K from per-fold coverage curves.

    Each fold's K is the smallest K reaching full coverage on that fold's
    training pairs.  The protocol K is the maximum over folds, which keeps a
    single controlled K for all feature rows.  Full coverage is re-checked at
    the protocol K in every fold.

    Args:
        fold_coverage_rows: One aggregated coverage curve per fold.
        fold_names: Optional fold labels for error messages and output.

    Returns:
        ``{"protocol_k": int, "folds": [...]}`` where each fold entry contains
        its ``chosen_k``, ``chosen_coverage``, ``coverage_at_protocol_k`` and
        the full ``coverage`` rows.

    Raises:
        ValueError: If no fold coverage curve is supplied.
        InsufficientCoverageError: If any fold cannot reach full coverage at
            its largest searched K, or if the shared protocol K is not fully
            covered in a fold.
    """
    if not fold_coverage_rows:
        raise ValueError("at least one fold coverage curve is required")

    resolved_fold_names = (
        list(fold_names)
        if fold_names is not None
        else [f"fold_{fold_index}" for fold_index in range(len(fold_coverage_rows))]
    )
    if len(resolved_fold_names) != len(fold_coverage_rows):
        raise ValueError("fold_names and fold_coverage_rows must have the same length")

    fold_reports: list[dict[str, Any]] = []
    chosen_k_values: list[int] = []
    for fold_name, coverage_rows in zip(resolved_fold_names, fold_coverage_rows):
        if not coverage_rows:
            raise ValueError(f"fold {fold_name} has no coverage rows")

        chosen_k = choose_k(coverage_rows)
        rows_by_k = {int(row["k"]): row for row in coverage_rows}
        chosen_row = rows_by_k.get(chosen_k)
        if chosen_row is None:
            raise InsufficientCoverageError(
                f"fold {fold_name}: chosen K={chosen_k} is absent from the "
                "coverage grid"
            )
        chosen_coverage = float(chosen_row["coverage"])
        if chosen_coverage < 1.0:
            largest_k = max(rows_by_k)
            raise InsufficientCoverageError(
                f"fold {fold_name}: coverage did not reach 1.0 at the largest "
                f"searched K={largest_k} (best coverage {chosen_coverage:.6f}, "
                f"residual miss rate {1.0 - chosen_coverage:.6f}, "
                f"{chosen_row['n_covered']}/{chosen_row['n_anchors']} anchors "
                "covered); escalate to the owner instead of weakening the "
                "protocol"
            )

        chosen_k_values.append(chosen_k)
        fold_reports.append(
            {
                "fold": fold_name,
                "chosen_k": int(chosen_k),
                "chosen_coverage": chosen_coverage,
                "coverage": list(coverage_rows),
            }
        )

    protocol_k = max(chosen_k_values)
    for fold_report in fold_reports:
        rows_by_k = {
            int(row["k"]): row for row in fold_report["coverage"]
        }
        protocol_row = rows_by_k.get(protocol_k)
        if protocol_row is None:
            raise InsufficientCoverageError(
                f"fold {fold_report['fold']}: protocol K={protocol_k} exceeds "
                "the largest searched K in this fold"
            )
        coverage_at_protocol_k = float(protocol_row["coverage"])
        if coverage_at_protocol_k < 1.0:
            raise InsufficientCoverageError(
                f"fold {fold_report['fold']}: coverage at protocol K="
                f"{protocol_k} is {coverage_at_protocol_k:.6f}, below 1.0; "
                "escalate to the owner instead of weakening the protocol"
            )
        fold_report["coverage_at_protocol_k"] = coverage_at_protocol_k

    return {"protocol_k": int(protocol_k), "folds": fold_reports}


def build_candidate_dataset(
    pair_records: Sequence[Mapping[str, Any]],
    k: int,
) -> CandidateDataset:
    """Build concatenated candidate tensors for a set of feature pairs.

    Args:
        pair_records: Feature pair records containing both frames' features,
            coordinates and labels.
        k: Candidate count selected by the coverage diagnostic.

    Returns:
        A mapping with concatenated ``anchor_features`` ``(N, K, D)``,
        ``candidate_features`` ``(N, K, D)``, ``target_index`` ``(N,)``,
        labels, coordinates, ``n_anchors``, pair counts and skip counters.

    Raises:
        ValueError: If ``pair_records`` is empty or ``k`` is not positive.
    """
    if not pair_records:
        raise ValueError("cannot build a candidate dataset from no pair records")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    dataset_parts: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, str]] = []
    dropped_missing_match = 0
    dropped_short_candidate_list = 0

    for pair_record in pair_records:
        pair_id = str(pair_record["pair_id"])
        candidate_set = build_knn_candidate_set(
            pair_record["anchor_features"],
            pair_record["candidate_features"],
            pair_record["anchor_coords"],
            pair_record["candidate_coords"],
            pair_record["anchor_labels"],
            pair_record["candidate_labels"],
            k=k,
        )
        dropped_missing_match += int(candidate_set["dropped_missing_match"])
        dropped_short_candidate_list += int(
            candidate_set["dropped_short_candidate_list"]
        )
        if candidate_set["n_anchors"] == 0:
            if candidate_set["dropped_short_candidate_list"] > 0:
                reason = (
                    f"fewer than K={k} candidates in the candidate frame"
                )
            else:
                reason = "no anchor had its true match among the selected candidates"
            skipped_pairs.append({"pair_id": pair_id, "reason": reason})
            logger.warning("skipping pair %s during candidate building: %s", pair_id, reason)
            continue
        dataset_parts.append(candidate_set)

    if not dataset_parts:
        raise ValueError(
            f"no usable anchors remained after candidate building with K={k}"
        )

    concatenation_keys = (
        "anchor_features",
        "candidate_features",
        "target_index",
        "anchor_labels",
        "candidate_labels",
        "anchor_coords",
        "candidate_coords",
    )
    concatenated: dict[str, Any] = {}
    for key in concatenation_keys:
        concatenated[key] = np.concatenate(
            [part[key] for part in dataset_parts], axis=0
        )
    concatenated["anchor_features"] = concatenated["anchor_features"].astype(
        np.float32, copy=False
    )
    concatenated["candidate_features"] = concatenated["candidate_features"].astype(
        np.float32, copy=False
    )
    concatenated["anchor_coords"] = concatenated["anchor_coords"].astype(
        np.float32, copy=False
    )
    concatenated["candidate_coords"] = concatenated["candidate_coords"].astype(
        np.float32, copy=False
    )
    concatenated["target_index"] = concatenated["target_index"].astype(
        np.int64, copy=False
    )
    concatenated["anchor_labels"] = concatenated["anchor_labels"].astype(
        np.int64, copy=False
    )
    concatenated["candidate_labels"] = concatenated["candidate_labels"].astype(
        np.int64, copy=False
    )

    return {
        **concatenated,
        "k": int(k),
        "n_anchors": int(concatenated["anchor_features"].shape[0]),
        "pair_count": len(pair_records),
        "used_pair_count": int(
            sum(1 for part in dataset_parts if part["n_anchors"] > 0)
        ),
        "skipped_pair_count": len(skipped_pairs),
        "pair_skips": skipped_pairs,
        "dropped_missing_match": int(dropped_missing_match),
        "dropped_short_candidate_list": int(dropped_short_candidate_list),
    }


def standardize_datasets(
    train_dataset: Mapping[str, Any],
    validation_dataset: Mapping[str, Any],
    evaluation_dataset: Mapping[str, Any],
) -> tuple[CandidateDataset, CandidateDataset, CandidateDataset, torch.Tensor, torch.Tensor]:
    """Fit normalization on training data and apply it to all three splits.

    The standardizer is fitted exactly once, on ``train_dataset``.  Validation
    and evaluation data are transformed with those frozen statistics; this
    function never refits on either split.

    Args:
        train_dataset: Train-internal candidate dataset.
        validation_dataset: Validation-internal candidate dataset.
        evaluation_dataset: Held-out-fold candidate dataset.

    Returns:
        ``(standardized_train, standardized_validation,
        standardized_evaluation, mean, std)``.  Feature tensors are
        ``float32``; target indices stay one-dimensional ``int64`` arrays.

    Raises:
        ValueError: If the training population is empty.
    """
    mean, std = fit_population_standardizer(
        np.asarray(train_dataset["anchor_features"]),
        np.asarray(train_dataset["candidate_features"]),
    )

    def _apply(dataset: Mapping[str, Any]) -> CandidateDataset:
        """Transform one candidate dataset with the fitted training statistics."""
        standardized_anchor, standardized_candidate = apply_population_standardizer(
            np.asarray(dataset["anchor_features"]),
            np.asarray(dataset["candidate_features"]),
            mean,
            std,
        )
        return {
            "anchor_features": standardized_anchor,
            "candidate_features": standardized_candidate,
            "target_index": np.asarray(dataset["target_index"], dtype=np.int64),
        }

    return _apply(train_dataset), _apply(validation_dataset), _apply(
        evaluation_dataset
    ), mean, std


def _chunk_dataset(
    dataset: Mapping[str, Any],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Split a candidate dataset into anchor batches for the training loop.

    Args:
        dataset: Mapping with ``anchor_features``, ``candidate_features`` and
            ``target_index``.
        batch_size: Maximum anchors per batch.

    Returns:
        A list of batch mappings.  An empty dataset returns an empty list.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    anchor_features = np.asarray(dataset["anchor_features"])
    candidate_features = np.asarray(dataset["candidate_features"])
    target_index = np.asarray(dataset["target_index"], dtype=np.int64)
    n_anchors = anchor_features.shape[0]
    if n_anchors == 0:
        return []

    batches: list[dict[str, Any]] = []
    for start in range(0, n_anchors, batch_size):
        stop = min(start + batch_size, n_anchors)
        batches.append(
            {
                "anchor_features": anchor_features[start:stop],
                "candidate_features": candidate_features[start:stop],
                "target_index": target_index[start:stop],
            }
        )
    return batches


def _make_probe(probe_name: str, feature_dimension: int) -> torch.nn.Module:
    """Construct the requested probe family.

    Args:
        probe_name: ``"linear"`` or ``"mlp"``.
        feature_dimension: Per-cell feature dimensionality ``D``.

    Returns:
        A fresh :class:`KnnLinearProbe` or :class:`KnnMLPProbe`.

    Raises:
        ValueError: If the probe name is not supported.
    """
    if probe_name == "linear":
        return KnnLinearProbe(feature_dimension)
    if probe_name == "mlp":
        return KnnMLPProbe(feature_dimension)
    raise ValueError(f"unknown probe {probe_name!r}")


def train_and_evaluate_probe(
    train_dataset: Mapping[str, Any],
    validation_dataset: Mapping[str, Any],
    evaluation_dataset: Mapping[str, Any],
    probe_name: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    batch_size: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Train one probe and evaluate it on a held-out fold.

    The validation dataset selects the threshold by F1 and drives early
    stopping by NLL.  The evaluation dataset is used only after the threshold
    is fixed, so its labels never influence training or threshold selection.

    Args:
        train_dataset: Standardized train-internal candidate dataset.
        validation_dataset: Standardized validation-internal candidate dataset.
        evaluation_dataset: Standardized held-out-fold candidate dataset.
        probe_name: ``"linear"`` or ``"mlp"``.
        seed: Probe seed used for initialization and optimization.
        epochs: Maximum training epochs.
        learning_rate: Adam learning rate.
        patience: Early-stopping patience in epochs.
        batch_size: Maximum anchors per training batch.
        device: Resolved torch device.

    Returns:
        A mapping with the selected ``threshold``, held-out ``metrics``,
        validation ``validation_metrics``, the training ``history`` and anchor
        counts for all three splits.

    Raises:
        ValueError: If any split has no anchors or the probe name is unknown.
    """
    for split_name, dataset in (
        ("train-internal", train_dataset),
        ("validation-internal", validation_dataset),
        ("held-out", evaluation_dataset),
    ):
        if int(np.asarray(dataset["target_index"]).shape[0]) == 0:
            raise ValueError(f"the {split_name} split has no anchors")

    feature_dimension = int(np.asarray(train_dataset["anchor_features"]).shape[-1])
    train_batches = _chunk_dataset(train_dataset, batch_size)
    validation_batches = _chunk_dataset(validation_dataset, batch_size)

    # Seed before model construction so parameter initialization is also
    # deterministic; train_knn_probe reseeds for the optimization path.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _make_probe(probe_name, feature_dimension)

    history, trained_model = train_knn_probe(
        model,
        train_batches,
        validation_batches,
        epochs=epochs,
        lr=learning_rate,
        patience=patience,
        seed=seed,
        device=str(device),
    )

    validation_probabilities = predict_proba(
        trained_model,
        np.asarray(validation_dataset["anchor_features"]),
        np.asarray(validation_dataset["candidate_features"]),
    )
    validation_metrics = select_threshold_and_metrics(
        validation_probabilities,
        np.asarray(validation_dataset["target_index"], dtype=np.int64),
    )
    selected_threshold = float(validation_metrics["threshold"])

    evaluation_probabilities = predict_proba(
        trained_model,
        np.asarray(evaluation_dataset["anchor_features"]),
        np.asarray(evaluation_dataset["candidate_features"]),
    )
    evaluation_metrics = select_threshold_and_metrics(
        evaluation_probabilities,
        np.asarray(evaluation_dataset["target_index"], dtype=np.int64),
        threshold_grid=[selected_threshold],
    )

    return {
        "threshold": selected_threshold,
        "metrics": evaluation_metrics,
        "validation_metrics": validation_metrics,
        "history": history,
        "epochs_trained": len(history),
        "n_train_anchors": int(
            np.asarray(train_dataset["target_index"]).shape[0]
        ),
        "n_validation_anchors": int(
            np.asarray(validation_dataset["target_index"]).shape[0]
        ),
        "n_evaluation_anchors": int(
            np.asarray(evaluation_dataset["target_index"]).shape[0]
        ),
    }


def _evaluate_split(
    *,
    feature_pair_records: Sequence[Mapping[str, Any]],
    feature_name: str,
    probe_name: str,
    seed: int,
    protocol_k: int,
    training_pool_pair_ids: Sequence[str],
    evaluation_pair_ids: Sequence[str],
    coverage_fold: Mapping[str, Any],
    fold_label: int | str,
    feature_dimension: int,
    args: argparse.Namespace,
    resolved_device: torch.device,
    heldout_condition: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one protocol split.

    This helper is shared by ``frame_cv`` and ``loco``.  It splits the supplied
    training pool into train-internal and validation-internal pairs, fits the
    normalizer on the train-internal population only, trains the probe with
    validation-internal early stopping, selects the threshold on the
    validation-internal split, and finally evaluates on the held-out pairs.
    The held-out pairs must be absent from ``training_pool_pair_ids``; this is
    the caller's responsibility and is what keeps LOCO leakage-free.

    Args:
        feature_pair_records: Feature pair records for the current feature row.
        feature_name: Feature row name, used in error messages and output.
        probe_name: ``"linear"`` or ``"mlp"``.
        seed: Probe seed.
        protocol_k: Shared protocol K.
        training_pool_pair_ids: Pair ids available for training and validation.
        evaluation_pair_ids: Pair ids used only for final held-out evaluation.
        coverage_fold: Coverage report for this split, containing ``chosen_k``,
            ``chosen_coverage`` and ``coverage_at_protocol_k``.
        fold_label: Frame-fold index for ``frame_cv`` or condition name for
            ``loco``.
        feature_dimension: Per-cell feature dimensionality ``D``.
        args: Parsed CLI arguments.
        resolved_device: Resolved torch device.
        heldout_condition: LOCO condition name, or ``None`` for ``frame_cv`` so
            that the frame-CV result schema stays unchanged.

    Returns:
        A per-split result mapping with the leakage-boundary pair ids, chosen K,
        thresholds, held-out metrics and anchor counts.

    Raises:
        ValueError: If any of the three internal splits has no usable pairs.
    """
    train_internal_ids, validation_internal_ids = split_internal_training_pairs(
        training_pool_pair_ids,
        validation_fraction=args.val_fraction,
    )

    train_records = _records_for_pair_ids(feature_pair_records, train_internal_ids)
    validation_records = _records_for_pair_ids(
        feature_pair_records, validation_internal_ids
    )
    evaluation_records = _records_for_pair_ids(
        feature_pair_records, evaluation_pair_ids
    )
    if not train_records:
        raise ValueError(
            f"fold {fold_label} feature {feature_name!r}: the train-internal "
            "split has no usable pairs"
        )
    if not validation_records:
        raise ValueError(
            f"fold {fold_label} feature {feature_name!r}: the validation-internal "
            "split has no usable pairs; increase the pair count or --val-fraction"
        )
    if not evaluation_records:
        raise ValueError(
            f"fold {fold_label} feature {feature_name!r}: the held-out split "
            "has no usable pairs"
        )

    train_dataset = build_candidate_dataset(train_records, protocol_k)
    validation_dataset = build_candidate_dataset(validation_records, protocol_k)
    evaluation_dataset = build_candidate_dataset(evaluation_records, protocol_k)
    (
        standardized_train,
        standardized_validation,
        standardized_evaluation,
        _mean,
        _std,
    ) = standardize_datasets(
        train_dataset, validation_dataset, evaluation_dataset
    )

    probe_result = train_and_evaluate_probe(
        standardized_train,
        standardized_validation,
        standardized_evaluation,
        probe_name=probe_name,
        seed=seed,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        device=resolved_device,
    )

    fold_result: dict[str, Any] = {
        "fold": fold_label,
        "feature_dimension": feature_dimension,
        "held_out_pair_ids": sorted(evaluation_pair_ids),
        "train_internal_pair_ids": train_internal_ids,
        "validation_internal_pair_ids": validation_internal_ids,
        "chosen_k": int(coverage_fold["chosen_k"]),
        "coverage_at_chosen_k": float(coverage_fold["chosen_coverage"]),
        "coverage_at_protocol_k": float(coverage_fold["coverage_at_protocol_k"]),
        "threshold": float(probe_result["threshold"]),
        "metrics": dict(probe_result["metrics"]),
        "validation_metrics": dict(probe_result["validation_metrics"]),
        "epochs_trained": int(probe_result["epochs_trained"]),
        "n_train_anchors": int(probe_result["n_train_anchors"]),
        "n_validation_anchors": int(probe_result["n_validation_anchors"]),
        "n_evaluation_anchors": int(probe_result["n_evaluation_anchors"]),
        "train_candidate_skips": train_dataset["skipped_pair_count"],
        "validation_candidate_skips": validation_dataset["skipped_pair_count"],
        "evaluation_candidate_skips": evaluation_dataset["skipped_pair_count"],
        "train_dropped_missing_match": train_dataset["dropped_missing_match"],
        "train_dropped_short_candidate_list": train_dataset[
            "dropped_short_candidate_list"
        ],
        "validation_dropped_missing_match": validation_dataset[
            "dropped_missing_match"
        ],
        "validation_dropped_short_candidate_list": validation_dataset[
            "dropped_short_candidate_list"
        ],
        "evaluation_dropped_missing_match": evaluation_dataset[
            "dropped_missing_match"
        ],
        "evaluation_dropped_short_candidate_list": evaluation_dataset[
            "dropped_short_candidate_list"
        ],
    }
    if heldout_condition is not None:
        fold_result["heldout_condition"] = heldout_condition

    logger.info(
        "fold %s %s/%s seed=%d: threshold=%.4f top1=%.4f f1=%.4f",
        fold_label,
        feature_name,
        probe_name,
        seed,
        probe_result["threshold"],
        probe_result["metrics"]["top1_accuracy"],
        probe_result["metrics"]["f1"],
    )
    return fold_result


def _aggregate_fold_metrics(
    fold_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate held-out metrics across folds.

    The standard deviation is the sample standard deviation (``ddof=1``) when
    more than one fold is present; a single fold reports ``0.0`` so the schema
    stays fixed.

    Args:
        fold_results: One mapping per fold containing a ``metrics`` mapping.

    Returns:
        Mapping from metric name to ``{"mean": float, "std": float}``.

    Raises:
        ValueError: If no fold results are supplied.
    """
    if not fold_results:
        raise ValueError("at least one fold result is required")

    aggregated: dict[str, dict[str, float]] = {}
    for metric_name in METRIC_KEYS:
        values = np.asarray(
            [float(fold["metrics"][metric_name]) for fold in fold_results],
            dtype=np.float64,
        )
        standard_deviation = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
        aggregated[metric_name] = {
            "mean": float(np.mean(values)),
            "std": standard_deviation,
        }
    return aggregated


def _summary_row(
    protocol: str,
    feature_name: str,
    probe_name: str,
    seed: int,
    protocol_k: int,
    fold_results: Sequence[Mapping[str, Any]],
    heldout_condition: str = "",
) -> dict[str, Any]:
    """Build one ``summary.csv`` row from per-fold results.

    Args:
        protocol: Protocol name (``frame_cv`` in this slice).
        feature_name: Feature row name.
        probe_name: Probe name.
        seed: Probe seed.
        protocol_k: Shared protocol K.
        fold_results: Per-fold result mappings.
        heldout_condition: LOCO condition name; empty for ``frame_cv``.

    Returns:
        A mapping with exactly :data:`SUMMARY_FIELDNAMES`.
    """
    aggregated_metrics = _aggregate_fold_metrics(fold_results)
    thresholds = np.asarray(
        [float(fold["threshold"]) for fold in fold_results], dtype=np.float64
    )
    threshold_standard_deviation = (
        float(np.std(thresholds, ddof=1)) if thresholds.size > 1 else 0.0
    )

    summary_row: dict[str, Any] = {
        "protocol": protocol,
        "feature": feature_name,
        "probe": probe_name,
        "seed": int(seed),
        "heldout_condition": heldout_condition,
        "n_folds": len(fold_results),
        "protocol_k": int(protocol_k),
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": threshold_standard_deviation,
    }
    for metric_name in METRIC_KEYS:
        summary_row[f"{metric_name}_mean"] = aggregated_metrics[metric_name]["mean"]
        summary_row[f"{metric_name}_std"] = aggregated_metrics[metric_name]["std"]
    return summary_row


def _records_for_pair_ids(
    pair_records: Sequence[Mapping[str, Any]],
    pair_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Select feature pair records whose ids are in ``pair_ids``.

    Args:
        pair_records: Feature pair records.
        pair_ids: Pair identifiers to retain.

    Returns:
        The retained records sorted by pair id for deterministic ordering.
    """
    selected_pair_ids = set(pair_ids)
    return sorted(
        (
            pair_record
            for pair_record in pair_records
            if str(pair_record["pair_id"]) in selected_pair_ids
        ),
        key=lambda pair_record: str(pair_record["pair_id"]),
    )


def _loco_split_specs(
    base_pairs: Sequence[Mapping[str, Any]],
    heldout_conditions: Sequence[str],
) -> list[dict[str, Any]]:
    """Build the leave-one-condition-out training and evaluation split.

    Every held-out condition contributes one split whose training pool is all
    loaded pairs from the other conditions.  Pairs are selected by condition,
    so the held-out condition can never leak into its own training pool even
    when pair identifiers happen to overlap.

    Args:
        base_pairs: Feature-agnostic pair records.
        heldout_conditions: Conditions to hold out, in output order.

    Returns:
        One mapping per held-out condition with ``fold`` (the condition name),
        ``training_pair_ids`` and ``held_out_pair_ids``.

    Raises:
        ValueError: If a held-out condition has no loaded pairs, or if no
            pairs from other conditions remain to train on.
    """
    pair_ids_by_condition: dict[str, list[str]] = {}
    for base_pair in base_pairs:
        condition = str(base_pair["condition"])
        pair_ids_by_condition.setdefault(condition, []).append(
            str(base_pair["pair_id"])
        )

    split_specs: list[dict[str, Any]] = []
    for heldout_condition in heldout_conditions:
        held_out_pair_ids = sorted(pair_ids_by_condition.get(heldout_condition, []))
        if not held_out_pair_ids:
            raise ValueError(
                f"held-out condition {heldout_condition!r} has no loaded pairs"
            )
        training_pair_ids = sorted(
            pair_id
            for condition, pair_ids in pair_ids_by_condition.items()
            if condition != heldout_condition
            for pair_id in pair_ids
        )
        if not training_pair_ids:
            raise ValueError(
                f"held-out condition {heldout_condition!r}: no training pairs "
                "remain from the other conditions"
            )
        split_specs.append(
            {
                "fold": heldout_condition,
                "training_pair_ids": training_pair_ids,
                "held_out_pair_ids": held_out_pair_ids,
            }
        )
    return split_specs


def _coverage_fold_reports_for_splits(
    base_pairs: Sequence[Mapping[str, Any]],
    split_specs: Sequence[Mapping[str, Any]],
    k_max: int,
) -> list[dict[str, Any]]:
    """Compute the aggregate coverage curve for explicit train/held-out splits.

    This is shared by ``frame_cv`` (frame-pair folds) and ``loco`` (condition
    folds).  Only ``training_pair_ids`` from each split enter the curve; the
    corresponding held-out ids are recorded for auditability but never used.

    Args:
        base_pairs: Feature-agnostic pair records.
        split_specs: One mapping per split with ``fold``,
            ``training_pair_ids`` and ``held_out_pair_ids``.
        k_max: Largest K to report.

    Returns:
        One mapping per split with ``fold``, ``training_pair_ids``,
        ``held_out_pair_ids`` and aggregated ``coverage`` rows.

    Raises:
        ValueError: If a split has no training pairs.
    """
    fold_reports: list[dict[str, Any]] = []
    for split_spec in split_specs:
        fold_name = str(split_spec["fold"])
        held_out_pair_ids = sorted(split_spec["held_out_pair_ids"])
        training_pair_ids = sorted(split_spec["training_pair_ids"])
        if not training_pair_ids:
            raise ValueError(
                f"fold {fold_name} has no training pairs; increase the pair "
                "count or reduce --cv-folds"
            )

        training_pairs = _records_for_pair_ids(base_pairs, training_pair_ids)
        per_pair_coverage: list[list[dict[str, int | float]]] = []
        candidate_counts: list[int] = []
        for pair in training_pairs:
            per_pair_coverage.append(
                coverage_vs_k(
                    pair["anchor"]["coords"],
                    pair["candidate"]["coords"],
                    pair["anchor"]["labels"],
                    pair["candidate"]["labels"],
                    k_max=k_max,
                )
            )
            candidate_counts.append(
                int(np.asarray(pair["candidate"]["labels"]).shape[0])
            )

        fold_reports.append(
            {
                "fold": fold_name,
                "training_pair_ids": training_pair_ids,
                "held_out_pair_ids": held_out_pair_ids,
                "coverage": aggregate_coverage_rows(
                    per_pair_coverage, candidate_counts, k_max
                ),
            }
        )
    return fold_reports


def _coverage_fold_reports(
    base_pairs: Sequence[Mapping[str, Any]],
    fold_by_pair_id: Mapping[str, int],
    effective_fold_count: int,
    k_max: int,
) -> list[dict[str, Any]]:
    """Compute the aggregate coverage curve for every frame-CV fold.

    Args:
        base_pairs: Feature-agnostic pair records.
        fold_by_pair_id: Fold assignment for each pair id.
        effective_fold_count: Number of folds in use.
        k_max: Largest K to report.

    Returns:
        One mapping per fold with ``fold``, ``training_pair_ids``,
        ``held_out_pair_ids`` and aggregated ``coverage`` rows.  Held-out pairs
        are excluded from every curve.

    Raises:
        ValueError: If a fold has no training pairs.
    """
    split_specs: list[dict[str, Any]] = []
    for fold_index in range(effective_fold_count):
        held_out_pair_ids = sorted(
            pair_id
            for pair_id, assigned_fold in fold_by_pair_id.items()
            if assigned_fold == fold_index
        )
        training_pair_ids = sorted(
            pair_id
            for pair_id, assigned_fold in fold_by_pair_id.items()
            if assigned_fold != fold_index
        )
        split_specs.append(
            {
                "fold": f"fold_{fold_index}",
                "training_pair_ids": training_pair_ids,
                "held_out_pair_ids": held_out_pair_ids,
            }
        )
    return _coverage_fold_reports_for_splits(base_pairs, split_specs, k_max)


def _build_coverage_payload(
    args: argparse.Namespace,
    protocol_k: int,
    k_max: int,
    selection_mode: str,
    fold_reports: Sequence[Mapping[str, Any]],
    scan_statistics: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the ``coverage_vs_k.json`` payload.

    Args:
        args: Parsed CLI arguments.
        protocol_k: Shared protocol K.
        k_max: Largest searched K.
        selection_mode: ``"auto"`` or ``"fixed"``.
        fold_reports: Per-fold coverage reports from :func:`select_protocol_k`
            or :func:`_coverage_fold_reports`.
        scan_statistics: Data-scope statistics.

    Returns:
        The JSON-serialisable coverage payload.
    """
    payload = {
        "protocol": args.protocol,
        "selection_mode": selection_mode,
        "feature_agnostic": True,
        "conditions": list(args.conditions),
        "k_max": int(k_max),
        "protocol_k": int(protocol_k),
        # Alias for readers that only look for "the chosen K" at the top level;
        # the authoritative per-fold values remain in ``folds[*].chosen_k``.
        "chosen_k": int(protocol_k),
        "pair_counts_by_condition": scan_statistics["pair_counts_by_condition"],
        "folds": list(fold_reports),
    }
    if args.protocol == "loco":
        payload["heldout_conditions"] = list(args.heldout_conditions)
    return payload


def _build_run_payload(
    args: argparse.Namespace,
    feature_name: str,
    probe_name: str,
    seed: int,
    protocol_k: int,
    fold_results: Sequence[Mapping[str, Any]],
    coverage_folds: Sequence[Mapping[str, Any]],
    scan_statistics: Mapping[str, Any],
    feature_statistics: Mapping[str, Any],
    resolved_device: torch.device,
    environment: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one per-run JSON payload.

    Args:
        args: Parsed CLI arguments.
        feature_name: Feature row name.
        probe_name: Probe name.
        seed: Probe seed.
        protocol_k: Shared protocol K.
        fold_results: Per-fold held-out results.
        coverage_folds: Per-fold coverage reports.
        scan_statistics: Data-scope statistics.
        feature_statistics: Feature-specific extraction statistics.
        resolved_device: Device used for training.
        environment: Runtime environment mapping.
        checkpoint_record: Checkpoint identity for this feature row, as returned
            by :meth:`FeatureExtractorContext.checkpoint_record`.

    Returns:
        The JSON-serialisable run payload.  It contains every key required by
        the SPEC verification schema (``protocol``, ``feature``, ``probe``,
        ``seed``, ``k``, ``coverage``, ``threshold``, ``metrics``,
        ``data_scope``, ``checkpoint``, ``environment``) plus versions and the
        leakage-boundary pair ids.
    """
    aggregated_metrics = _aggregate_fold_metrics(fold_results)
    thresholds = np.asarray(
        [float(fold["threshold"]) for fold in fold_results], dtype=np.float64
    )

    feature_configuration: dict[str, Any]
    if feature_name == "posenc":
        feature_configuration = {
            "encoding": "Fourier positional encoding of absolute (y, x) centroids",
            "n_freqs": int(POSENC_N_FREQS),
            "cutoff": float(POSENC_CUTOFF),
            "output_dim": int(POSENC_DIM),
            "displacement_definition": (
                "none; the existing extractor encodes absolute centroids, not "
                "relative displacement"
            ),
        }
    else:
        feature_configuration = {
            "encoding": feature_name,
            "output_dim": int(fold_results[0].get("feature_dimension", 0)),
        }

    data_scope: dict[str, Any] = {
        **scan_statistics,
        "feature_statistics": dict(feature_statistics),
        "requested_cv_folds": int(args.cv_folds),
        "effective_cv_folds": len(fold_results),
        "validation_fraction": float(args.val_fraction),
        "fold_assignment_seed": int(FOLD_ASSIGNMENT_SEED),
        "internal_split_seed": int(INTERNAL_SPLIT_SEED),
        "data_root": str(args.data_root),
        "max_pairs_per_condition": int(args.max_pairs_per_condition),
    }
    args_record: dict[str, Any] = {
        "data_root": str(args.data_root),
        "conditions": list(args.conditions),
        "max_pairs_per_condition": int(args.max_pairs_per_condition),
        "protocol": args.protocol,
        "features": list(args.features),
        "probes": list(args.probes),
        "seeds": list(args.seeds),
        "cv_folds": int(args.cv_folds),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "device": str(args.device),
        "resolved_device": str(resolved_device),
        "k": args.k,
        "k_max": int(args.k_max),
        "val_fraction": float(args.val_fraction),
        "out": str(args.out),
    }
    if args.protocol == "loco":
        data_scope["heldout_conditions"] = list(args.heldout_conditions)
        args_record["heldout_conditions"] = list(args.heldout_conditions)

    return {
        "schema_version": int(RUN_SCHEMA_VERSION),
        "protocol": args.protocol,
        "feature": feature_name,
        "probe": probe_name,
        "seed": int(seed),
        "k": int(protocol_k),
        "coverage": {
            "selection_mode": args.k if isinstance(args.k, str) else "fixed",
            "protocol_k": int(protocol_k),
            "per_fold": [
                {
                    "fold": fold_report.get("fold"),
                    "chosen_k": fold_report.get("chosen_k", protocol_k),
                    "coverage_at_protocol_k": fold_report.get(
                        "coverage_at_protocol_k"
                    ),
                }
                for fold_report in coverage_folds
            ],
        },
        "threshold": float(np.mean(thresholds)),
        "metrics": aggregated_metrics,
        "per_fold": list(fold_results),
        "feature_configuration": feature_configuration,
        "checkpoint": dict(checkpoint_record),
        "data_scope": data_scope,
        "environment": dict(environment),
        "versions": {
            "run_schema": int(RUN_SCHEMA_VERSION),
            "python": environment["python"],
            "numpy": environment["numpy"],
            "torch": environment["torch"],
        },
        "args": args_record,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Phase-1 K-NN probe CLI.

    Returns:
        A parser whose ``--help`` text documents every supported argument.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run protocol frame_cv (P1, within-experiment frame-pair CV) or "
            "loco (P2, leave-one-condition-out) of the SPEC 0046 "
            "K-nearest-neighbour feature probe.  Feature rows: "
            "regionprops_7d, regionprops_13d, hu_20d, posenc, dino, "
            "cnn_frozen, cnn_ntxent, sam21.  The sam21 row requires the "
            "optional sam2 package (pip install git+https://github.com/"
            "facebookresearch/sam2.git)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default="../data/vanvliet",
        help="root directory containing the vanvliet condition folders",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help=(
            "comma-separated condition names, for example rpsM,recA; empty "
            "uses rpsM for frame_cv and all six conditions for loco"
        ),
    )
    parser.add_argument(
        "--heldout-conditions",
        default="",
        help=(
            "comma-separated loco conditions to hold out; empty holds out every "
            "loaded condition.  Ignored by frame_cv"
        ),
    )
    parser.add_argument(
        "--max-pairs-per-condition",
        type=int,
        default=15,
        help="maximum number of usable consecutive frame pairs per condition",
    )
    parser.add_argument(
        "--protocol",
        default="frame_cv",
        help="protocol name: frame_cv (P1) or loco (P2)",
    )
    parser.add_argument(
        "--features",
        default="regionprops_7d,regionprops_13d,hu_20d,posenc",
        help=(
            "comma-separated feature rows: regionprops_7d, regionprops_13d, "
            "hu_20d, posenc, dino, cnn_frozen, cnn_ntxent, sam21 (sam21 "
            "requires the optional sam2 package)"
        ),
    )
    parser.add_argument(
        "--probes",
        default="linear,mlp",
        help="comma-separated probe families (linear, mlp)",
    )
    parser.add_argument(
        "--seeds",
        "--seed",
        dest="seeds",
        default="42",
        help="comma-separated probe seeds, for example 42,43,44",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="requested number of frame-pair CV folds",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="maximum training epochs per probe",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early-stopping patience in epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="maximum anchors per training batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device string, or auto to prefer CUDA when available",
    )
    parser.add_argument(
        "--k",
        type=_parse_k,
        default="auto",
        help="candidate count; use auto for coverage-vs-K selection",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=DEFAULT_K_MAX,
        help="largest K searched by the automatic coverage diagnostic",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help=(
            "fraction of each fold's training pairs used for internal early "
            "stopping and threshold selection"
        ),
    )
    parser.add_argument(
        "--out",
        default="results/probes/knn",
        help="output directory for coverage, per-run JSON and summary.csv",
    )
    return parser


def _parse_seed_list(seed_text: str) -> list[int]:
    """Parse and validate the comma-separated seed list.

    Args:
        seed_text: Raw ``--seeds`` value.

    Returns:
        A non-empty list of non-negative integer seeds.

    Raises:
        ValueError: If a seed is not a non-negative integer.
    """
    seeds: list[int] = []
    for token in _parse_comma_separated(seed_text):
        try:
            seed = int(token)
        except ValueError as conversion_error:
            raise ValueError(
                f"invalid seed {token!r}; seeds must be integers"
            ) from conversion_error
        if seed < 0:
            raise ValueError(f"invalid seed {seed}; seeds must be non-negative")
        seeds.append(seed)
    if not seeds:
        raise ValueError("at least one seed must be supplied")
    return seeds


def run_experiment(args: argparse.Namespace) -> int:
    """Run the requested feature/probe/seed combinations and write artifacts.

    Args:
        args: Parsed CLI namespace.  ``args.features``, ``args.probes``,
            ``args.conditions`` and ``args.heldout_conditions`` are
            comma-separated strings at this boundary; they are parsed and
            validated here.

    Returns:
        ``0`` on success.

    Raises:
        NotImplementedError: For deferred feature rows.
        InsufficientCoverageError: If automatic K selection cannot reach full
            coverage.
        ValueError: For invalid arguments, missing data, or empty splits.
    """
    protocol = validate_protocol(args.protocol)
    feature_names = validate_feature_names(_parse_comma_separated(args.features))
    probe_names = validate_probe_names(_parse_comma_separated(args.probes))
    requested_condition_names = _parse_comma_separated(args.conditions)
    if requested_condition_names:
        condition_names = requested_condition_names
    elif protocol == "loco":
        condition_names = list(DEFAULT_LOCO_CONDITIONS)
    else:
        condition_names = list(DEFAULT_FRAME_CV_CONDITIONS)
    if not condition_names:
        raise ValueError("at least one condition must be requested")
    requested_heldout_conditions = _parse_comma_separated(
        args.heldout_conditions
    )
    if protocol == "loco":
        heldout_condition_names = (
            requested_heldout_conditions
            if requested_heldout_conditions
            else list(condition_names)
        )
        unknown_heldout_conditions = [
            heldout_condition
            for heldout_condition in heldout_condition_names
            if heldout_condition not in condition_names
        ]
        if unknown_heldout_conditions:
            raise ValueError(
                "held-out condition(s) not present in --conditions: "
                f"{', '.join(unknown_heldout_conditions)}"
            )
    else:
        if requested_heldout_conditions:
            raise ValueError(
                "--heldout-conditions is only valid with --protocol loco"
            )
        heldout_condition_names = []
    seeds = _parse_seed_list(args.seeds)
    # Replace the raw comma-separated CLI strings with their validated forms so
    # every downstream payload records the same values the run actually used.
    args.features = feature_names
    args.probes = probe_names
    args.conditions = condition_names
    args.heldout_conditions = heldout_condition_names
    args.seeds = seeds
    if args.max_pairs_per_condition <= 0:
        raise ValueError(
            "--max-pairs-per-condition must be positive, got "
            f"{args.max_pairs_per_condition}"
        )
    if args.cv_folds <= 0:
        raise ValueError(f"--cv-folds must be positive, got {args.cv_folds}")
    if args.k_max <= 0:
        raise ValueError(f"--k-max must be positive, got {args.k_max}")
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs, --patience and --batch-size must be positive")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError(
            f"--val-fraction must be strictly between 0 and 1, got {args.val_fraction}"
        )

    resolved_device = resolve_device(args.device)
    environment = _environment_record(resolved_device)
    extractor_context = FeatureExtractorContext(resolved_device)
    output_directory = Path(args.out)

    base_pairs, scan_statistics = load_base_pairs(
        args.data_root,
        condition_names,
        max_pairs_per_condition=args.max_pairs_per_condition,
    )
    selection_mode = "auto" if args.k == "auto" else "fixed"
    coverage_k_max = args.k_max if args.k == "auto" else max(args.k_max, int(args.k))
    if protocol == "loco":
        coverage_fold_reports = _coverage_fold_reports_for_splits(
            base_pairs,
            _loco_split_specs(base_pairs, heldout_condition_names),
            coverage_k_max,
        )
        fold_by_pair_id: dict[str, int] = {}
        effective_fold_count = 0
    else:
        base_pair_ids = [str(base_pair["pair_id"]) for base_pair in base_pairs]
        fold_by_pair_id, effective_fold_count = assign_pair_folds(
            base_pair_ids, args.cv_folds
        )
        coverage_fold_reports = _coverage_fold_reports(
            base_pairs,
            fold_by_pair_id,
            effective_fold_count,
            coverage_k_max,
        )

    if args.k == "auto":
        selection_result = select_protocol_k(
            [fold_report["coverage"] for fold_report in coverage_fold_reports],
            [str(fold_report["fold"]) for fold_report in coverage_fold_reports],
        )
        protocol_k = int(selection_result["protocol_k"])
        protocol_fold_reports = selection_result["folds"]
        for fold_report, coverage_fold_report in zip(
            protocol_fold_reports, coverage_fold_reports
        ):
            fold_report["training_pair_ids"] = coverage_fold_report[
                "training_pair_ids"
            ]
            fold_report["held_out_pair_ids"] = coverage_fold_report[
                "held_out_pair_ids"
            ]
    else:
        protocol_k = int(args.k)
        protocol_fold_reports = []
        for coverage_fold_report in coverage_fold_reports:
            coverage_rows_by_k = {
                int(row["k"]): row for row in coverage_fold_report["coverage"]
            }
            protocol_row = coverage_rows_by_k.get(protocol_k)
            if protocol_row is None:
                raise InsufficientCoverageError(
                    f"fold {coverage_fold_report['fold']}: fixed K={protocol_k} "
                    f"exceeds the searched K_max={coverage_k_max}"
                )
            coverage_at_protocol_k = float(protocol_row["coverage"])
            if coverage_at_protocol_k < 1.0:
                logger.warning(
                    "fixed K=%d has coverage %.6f in fold %s (residual miss "
                    "rate %.6f); proceeding because --k was supplied explicitly",
                    protocol_k,
                    coverage_at_protocol_k,
                    1.0 - coverage_at_protocol_k,
                    coverage_fold_report["fold"],
                )
            protocol_fold_reports.append(
                {
                    "fold": coverage_fold_report["fold"],
                    "chosen_k": protocol_k,
                    "chosen_coverage": coverage_at_protocol_k,
                    "coverage_at_protocol_k": coverage_at_protocol_k,
                    "coverage": coverage_fold_report["coverage"],
                    "training_pair_ids": coverage_fold_report["training_pair_ids"],
                    "held_out_pair_ids": coverage_fold_report["held_out_pair_ids"],
                }
            )

    coverage_payload = _build_coverage_payload(
        args,
        protocol_k=protocol_k,
        k_max=coverage_k_max,
        selection_mode=selection_mode,
        fold_reports=protocol_fold_reports,
        scan_statistics=scan_statistics,
    )
    coverage_path = output_directory / "coverage_vs_k.json"
    write_json(coverage_path, coverage_payload)
    logger.info(
        "selected protocol K=%d (%s) and wrote %s",
        protocol_k,
        selection_mode,
        coverage_path,
    )

    evaluation_folds: list[dict[str, Any]] = []
    if protocol == "loco":
        for coverage_fold in protocol_fold_reports:
            heldout_condition = str(coverage_fold["fold"])
            evaluation_folds.append(
                {
                    "fold_label": heldout_condition,
                    "heldout_condition": heldout_condition,
                    "training_pool_pair_ids": coverage_fold["training_pair_ids"],
                    "held_out_pair_ids": coverage_fold["held_out_pair_ids"],
                    "coverage_fold": coverage_fold,
                }
            )
    else:
        for fold_index in range(effective_fold_count):
            coverage_fold = next(
                fold_report
                for fold_report in protocol_fold_reports
                if fold_report["fold"] == f"fold_{fold_index}"
            )
            evaluation_folds.append(
                {
                    "fold_label": int(fold_index),
                    "heldout_condition": None,
                    "training_pool_pair_ids": coverage_fold["training_pair_ids"],
                    "held_out_pair_ids": coverage_fold["held_out_pair_ids"],
                    "coverage_fold": coverage_fold,
                }
            )

    try:
        pairs_by_feature, statistics_by_feature = extract_feature_pairs(
            base_pairs, feature_names, extractor_context
        )
        runs_directory = output_directory / "runs"
        summary_rows: list[dict[str, Any]] = []

        for feature_name in feature_names:
            feature_pair_records = pairs_by_feature[feature_name]
            if not feature_pair_records:
                raise ValueError(
                    f"feature {feature_name!r} has no usable pairs after extraction"
                )
            feature_dimension = int(
                np.asarray(feature_pair_records[0]["anchor_features"]).shape[1]
            )

            for probe_name in probe_names:
                for seed in seeds:
                    logger.info(
                        "running feature=%s probe=%s seed=%d",
                        feature_name,
                        probe_name,
                        seed,
                    )
                    fold_results: list[dict[str, Any]] = []

                    for evaluation_fold in evaluation_folds:
                        fold_results.append(
                            _evaluate_split(
                                feature_pair_records=feature_pair_records,
                                feature_name=feature_name,
                                probe_name=probe_name,
                                seed=seed,
                                protocol_k=protocol_k,
                                training_pool_pair_ids=evaluation_fold[
                                    "training_pool_pair_ids"
                                ],
                                evaluation_pair_ids=evaluation_fold[
                                    "held_out_pair_ids"
                                ],
                                coverage_fold=evaluation_fold["coverage_fold"],
                                fold_label=evaluation_fold["fold_label"],
                                feature_dimension=feature_dimension,
                                args=args,
                                resolved_device=resolved_device,
                                heldout_condition=evaluation_fold[
                                    "heldout_condition"
                                ],
                            )
                        )

                    run_payload = _build_run_payload(
                        args,
                        feature_name=feature_name,
                        probe_name=probe_name,
                        seed=seed,
                        protocol_k=protocol_k,
                        fold_results=fold_results,
                        coverage_folds=protocol_fold_reports,
                        scan_statistics=scan_statistics,
                        feature_statistics=statistics_by_feature[feature_name],
                        resolved_device=resolved_device,
                        environment=environment,
                        checkpoint_record=extractor_context.checkpoint_record(
                            feature_name
                        ),
                    )
                    run_path = (
                        runs_directory
                        / f"{feature_name}_{probe_name}_s{seed}.json"
                    )
                    write_json(run_path, run_payload)
                    if protocol == "loco":
                        for evaluation_fold, fold_result in zip(
                            evaluation_folds, fold_results
                        ):
                            summary_rows.append(
                                _summary_row(
                                    args.protocol,
                                    feature_name,
                                    probe_name,
                                    seed,
                                    protocol_k,
                                    [fold_result],
                                    heldout_condition=str(
                                        evaluation_fold["heldout_condition"]
                                    ),
                                )
                            )
                        summary_rows.append(
                            _summary_row(
                                args.protocol,
                                feature_name,
                                probe_name,
                                seed,
                                protocol_k,
                                fold_results,
                                heldout_condition="overall",
                            )
                        )
                    else:
                        summary_rows.append(
                            _summary_row(
                                args.protocol,
                                feature_name,
                                probe_name,
                                seed,
                                protocol_k,
                                fold_results,
                            )
                        )
                    logger.info("wrote %s", run_path)

        summary_path = output_directory / "summary.csv"
        write_summary_csv(summary_path, summary_rows)
        logger.info("wrote %s with %d run rows", summary_path, len(summary_rows))
        return 0
    finally:
        extractor_context.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entry point.

    Args:
        argv: Optional argument list; ``None`` uses :data:`sys.argv`.

    Returns:
        Process exit code: ``0`` on success, ``2`` for a reported protocol,
        coverage, validation, or optional-dependency error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        return run_experiment(args)
    except (
        ImportError,
        NotImplementedError,
        InsufficientCoverageError,
        ValueError,
    ) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
