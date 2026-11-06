#!/usr/bin/env python3
"""Full-model inference grid for SPEC 0062.

F1 loads the complete Trackastra transformer from a local checkpoint,
generates the synthetic full-model inputs, and swaps one of four attention
implementations into every encoder and decoder layer. F2 adds the fp32 CPU
sanity/equivalence checks. F3 adds the timed forward chain, the total-peak
VRAM memory columns (allocated and reserved per stage plus the incremental
delta; SPEC 0062 §12), environment JSON, and raw CSV writing. F4 adds linker
timing for ``B=1`` using a read-only
consecutive-frame weight lookup over the normalized association matrix.

The memory protocol matches the isolated attention benchmark
(``src/bench/timing.py::measure``): every repeat synchronises, runs
``gc.collect()`` plus ``empty_cache()`` before the peak-memory baseline is
captured, and the repeat's ``association``/``normalized_association``
tensors are released and the cache cleared once after each repeat,
outside the timed region.

A fresh model is loaded for every variant so a swap cannot contaminate the
next variant. The swap copies ``q_pro``/``k_pro``/``v_pro``/``proj`` and
``rot_pos_enc`` weights with a strict state-dict load and raises a descriptive
error if any key is missing or unexpected.

The linker runs ``build_graph`` and then ``track_greedy`` (or the optional
``motile`` ILP) on CPU. The normalized association matrix is copied to CPU
outside the timed linker region and exposed lazily, so no ``N x N`` Python
dictionary is materialised. On ``ok`` rows,
``total_ms = full_forward_ms + normalize_ms + linker_ms`` for ``B=1``; linker
timing is skipped for ``B>1``.

Usage (from ``benchmark_attn``)::

    python scripts/benchmarks/benchmark_full_model_inference_grid.py \
        --checkpoint ../checkpoints/k16 --smoke

    python scripts/benchmarks/benchmark_full_model_inference_grid.py \
        --checkpoint ../checkpoints/k16 --sanity

The smoke path forces ``B=1``, ``N=128``, ``repeats=1``, ``warmup=1``, all
four variants, CPU unless ``--device cuda`` is explicit, and writes
``results/full_model_inference_grid_smoke.csv``. The sanity path runs the fp32
CPU equivalence checks and exits without constructing the grid.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_ATTN_ROOT = _PROJECT_ROOT / "benchmark_attn"
_TRACKASTRA_PACKAGE_ROOT = _PROJECT_ROOT / "trackastra"

for _import_root in (
    _PROJECT_ROOT,
    _BENCHMARK_ATTN_ROOT,
    _TRACKASTRA_PACKAGE_ROOT,
):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from src.bench.cli import parse_int_list  # noqa: E402
from src.bench.data import make_full_model_inputs  # noqa: E402
from src.bench.registry import (  # noqa: E402
    FULL_MODEL_VARIANT_REGISTRY,
    FullModelVariantEntry,
    resolve_class,
)
from src.bench.timing import ComponentTimer  # noqa: E402
from trackastra.model.model_api import Trackastra  # noqa: E402
from trackastra.tracking import build_graph, track_greedy  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_VARIANTS = (
    "relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16,flex_attn"
)
_DEFAULT_NS = "512,1024,2048,4096,8192,16384,32768"
_SMOKE_SEQ_LEN = 128
_SMOKE_PRIMARY_CSV = "results/full_model_inference_grid_smoke.csv"
_SMOKE_ENV_JSON = "results/full_model_inference_grid_smoke_env.json"
_SMOKE_SUMMARY_CSV = "results/full_model_inference_grid_smoke_summary.csv"
_LINKER_STATUS_OK = "ok"
_LINKER_STATUS_SKIPPED_B_GT_1 = "skipped_b_gt_1"
_LINKER_STATUS_ILP_UNAVAILABLE = "ilp_unavailable"
_LINKER_MAX_DISTANCE = 256
_LINKER_MAX_NEIGHBORS = 10
_LINKER_DELTA_T = 1
_LINKER_EDGE_THRESHOLD = 0.05
_LINKER_THRESHOLD = 0.5
_LINKER_ALLOW_DIVISIONS = True
_KNN_NEIGHBORS = 16
RAW_CSV_HEADER = [
    "variant",
    "N",
    "B",
    "repeat_idx",
    "status",
    "checkpoint",
    "dtype",
    "seed",
    "knn_neighbors",
    "coord_layout",
    "linker_mode",
    "linker_status",
    "full_forward_ms",
    "embed_ms",
    "encoder_ms",
    "encoder_attn_ms",
    "decoder_ms",
    "decoder_attn_ms",
    "head_logits_ms",
    "head_x_ms",
    "head_y_ms",
    "einsum_ms",
    "embed_proj_ms",
    "normalize_ms",
    "linker_build_graph_ms",
    "linker_greedy_ms",
    "linker_ms",
    "total_ms",
    "peak_mem_mb",
    "baseline_mem_mb",
    "peak_mem_total_mb",
    "peak_mem_reserved_mb",
    "forward_peak_mem_total_mb",
    "forward_peak_mem_reserved_mb",
    "normalize_peak_mem_total_mb",
    "normalize_peak_mem_reserved_mb",
    "dist_2d_standalone_ms",
    "forward_closure_error_ms",
    "error",
]
_TIMING_COLUMNS = (
    "full_forward_ms",
    "embed_ms",
    "encoder_ms",
    "encoder_attn_ms",
    "decoder_ms",
    "decoder_attn_ms",
    "head_logits_ms",
    "head_x_ms",
    "head_y_ms",
    "einsum_ms",
    "embed_proj_ms",
    "normalize_ms",
    "total_ms",
    "peak_mem_mb",
    "baseline_mem_mb",
    "peak_mem_total_mb",
    "peak_mem_reserved_mb",
    "forward_peak_mem_total_mb",
    "forward_peak_mem_reserved_mb",
    "normalize_peak_mem_total_mb",
    "normalize_peak_mem_reserved_mb",
    "dist_2d_standalone_ms",
    "forward_closure_error_ms",
)
_LINKER_TIMING_COLUMNS = (
    "linker_build_graph_ms",
    "linker_greedy_ms",
    "linker_ms",
)
_DTYPE_BY_NAME = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
_REQUIRED_WEIGHT_PREFIXES = (
    "q_pro.",
    "k_pro.",
    "v_pro.",
    "proj.",
    "rot_pos_enc.",
)
_SANITY_SEQ_LEN = 128
_SANITY_BATCH_SIZE = 1
_SANITY_ATOL = 1e-4
_SANITY_RTOL = 1e-3


@dataclass(frozen=True)
class AttentionSwapReport:
    """Summary of one variant swap across all encoder/decoder layers.

    Attributes:
        variant_key: Registry key of the swapped variant.
        target_class_path: Dotted import path of the target attention class.
        encoder_layers_swapped: Number of encoder layers whose ``attn`` was
            replaced.
        decoder_layers_swapped: Number of decoder layers whose ``attn`` was
            replaced.
        weight_keys_loaded: Total source state-dict entries loaded across all
            swapped layers.
        missing_weight_keys: Source keys absent from the target module, if
            any. Strict loading must leave this empty.
        unexpected_weight_keys: Target keys absent from the source module, if
            any. Strict loading must leave this empty.
    """

    variant_key: str
    target_class_path: str
    encoder_layers_swapped: int
    decoder_layers_swapped: int
    weight_keys_loaded: int
    missing_weight_keys: tuple[str, ...]
    unexpected_weight_keys: tuple[str, ...]


def parse_string_list(value: str) -> list[str]:
    """Parse a comma-separated string into stripped, non-empty tokens.

    Args:
        value: Comma-separated string such as ``"relative_pos,cached_dist"``.

    Returns:
        List of non-empty tokens.

    Raises:
        ValueError: If the input contains no tokens.
    """
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Expected at least one comma-separated value.")
    return tokens


def resolve_device(device_argument: str) -> torch.device:
    """Resolve the CLI device choice to a concrete torch device.

    Args:
        device_argument: ``"auto"``, ``"cpu"``, or ``"cuda"``.

    Returns:
        Concrete ``torch.device``.

    Raises:
        ValueError: If the name is unknown.
        RuntimeError: If CUDA is requested but unavailable.
    """
    if device_argument == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(device_argument)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_dtype(dtype_argument: str, device: torch.device) -> torch.dtype:
    """Resolve the CLI dtype choice for the selected device.

    Args:
        dtype_argument: ``"auto"`` or one of the accepted dtype names.
        device: Concrete torch device used for the ``"auto"`` default.

    Returns:
        Floating torch dtype.

    Raises:
        ValueError: If the dtype name is unknown.
    """
    if dtype_argument == "auto":
        return torch.float32 if device.type == "cpu" else torch.float16
    if dtype_argument not in _DTYPE_BY_NAME:
        raise ValueError(
            f"Unknown dtype {dtype_argument!r}; expected 'auto' or one of "
            f"{sorted(_DTYPE_BY_NAME)}."
        )
    return _DTYPE_BY_NAME[dtype_argument]


def resolve_checkpoint_path(checkpoint_argument: str) -> Path:
    """Resolve a checkpoint path from the CWD, project, or suffix.

    The script can be launched from the repository root or from
    ``benchmark_attn``. Candidate paths are tried in order; a ``checkpoints``
    suffix from the supplied path is also tried under the project root so the
    documented ``../../checkpoints/k16`` smoke invocation works from
    ``benchmark_attn``.

    Args:
        checkpoint_argument: Checkpoint path supplied on the CLI.

    Returns:
        Existing checkpoint directory.

    Raises:
        FileNotFoundError: If no candidate directory exists.
    """
    raw_path = Path(checkpoint_argument).expanduser()
    candidates = [raw_path]

    if not raw_path.is_absolute():
        candidates.append(_BENCHMARK_ATTN_ROOT / raw_path)
        candidates.append(_PROJECT_ROOT / raw_path)

    if "checkpoints" in raw_path.parts:
        checkpoint_index = raw_path.parts.index("checkpoints")
        candidates.append(
            _PROJECT_ROOT.joinpath(*raw_path.parts[checkpoint_index:])
        )

    seen_candidates: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen_candidates:
            continue
        seen_candidates.add(resolved_candidate)
        if resolved_candidate.is_dir():
            return resolved_candidate

    tried_paths = ", ".join(
        str(candidate) for candidate in seen_candidates
    )
    raise FileNotFoundError(
        f"Checkpoint directory {checkpoint_argument!r} not found; "
        f"tried: {tried_paths}"
    )


def load_full_trackastra_model(
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Trackastra:
    """Load a full Trackastra model and put its transformer in eval mode.

    Args:
        checkpoint_path: Directory containing ``config.yaml``,
            ``train_config.yaml``, and ``model.pt``.
        device: Device for the loaded transformer.
        dtype: Optional dtype conversion applied after loading. ``None``
            keeps the checkpoint dtype.

    Returns:
        Loaded :class:`Trackastra` instance.
    """
    model = Trackastra.from_folder(Path(checkpoint_path), device=str(device))
    transformer = model.transformer
    if dtype is not None:
        transformer.to(dtype=dtype)
    transformer.eval()
    return model


def _attention_constructor_kwargs(
    transformer_config: dict[str, object],
    variant_entry: FullModelVariantEntry,
) -> dict[str, object]:
    """Build target-attention constructor kwargs from checkpoint config.

    Args:
        transformer_config: ``TrackingTransformer.config`` loaded from the
            checkpoint.
        variant_entry: Registry metadata for the target attention class.

    Returns:
        Keyword arguments accepted by all four target attention classes.

    Raises:
        ValueError: If a sparse variant has no ``knn_neighbors`` value.
    """
    constructor_kwargs: dict[str, object] = {
        "coord_dim": int(transformer_config["coord_dim"]),
        "embed_dim": int(transformer_config["d_model"]),
        "n_head": int(transformer_config["nhead"]),
        "cutoff_spatial": transformer_config["spatial_pos_cutoff"],
        "cutoff_temporal": transformer_config["window"],
        "n_spatial": int(
            transformer_config["attn_positional_bias_n_spatial"]
        ),
        "n_temporal": transformer_config["window"],
        "dropout": transformer_config["dropout"],
        "mode": transformer_config["attn_positional_bias"],
        "attn_dist_mode": transformer_config["attn_dist_mode"],
    }
    if variant_entry.needs_knn:
        if variant_entry.knn_neighbors is None:
            raise ValueError(
                f"Sparse variant {variant_entry.registry_key!r} must define "
                "knn_neighbors."
            )
        constructor_kwargs["knn_neighbors"] = variant_entry.knn_neighbors
    return constructor_kwargs


def _copy_attention_weights(
    source_attention: torch.nn.Module,
    target_attention: torch.nn.Module,
) -> int:
    """Strictly copy all source attention weights into the target module.

    Args:
        source_attention: Checkpoint attention module being replaced.
        target_attention: Fresh target module that receives the weights.

    Returns:
        Number of state-dict entries loaded.

    Raises:
        RuntimeError: If the strict load fails. The message lists missing and
            unexpected keys so a required projection is never dropped
            silently.
    """
    source_state = source_attention.state_dict()
    target_keys = set(target_attention.state_dict())
    source_keys = set(source_state)
    missing_keys = sorted(source_keys - target_keys)
    unexpected_keys = sorted(target_keys - source_keys)

    try:
        target_attention.load_state_dict(source_state, strict=True)
    except RuntimeError as load_error:
        raise RuntimeError(
            "Strict attention weight load failed for "
            f"{type(target_attention).__name__}. "
            f"missing_keys={missing_keys}, "
            f"unexpected_keys={unexpected_keys}. Original error: "
            f"{load_error}"
        ) from load_error

    return len(source_state)


def swap_full_model_attention(
    transformer: torch.nn.Module,
    variant_key: str,
) -> AttentionSwapReport:
    """Replace every encoder/decoder attention with one variant.

    The target modules are built from ``transformer.config`` and receive the
    checkpoint weights through :func:`_copy_attention_weights`. Each layer
    gets a fresh target instance; the layer's ``positional_bias`` flag is
    kept consistent with the checkpoint mode so coordinates continue to be
    passed to the attention class.

    Args:
        transformer: Loaded ``TrackingTransformer`` whose attention modules
            are replaced in place.
        variant_key: Key in :data:`FULL_MODEL_VARIANT_REGISTRY`.

    Returns:
        :class:`AttentionSwapReport` describing the completed swap.

    Raises:
        KeyError: If ``variant_key`` is not registered.
        RuntimeError: If a strict weight copy fails.
    """
    variant_entry = FULL_MODEL_VARIANT_REGISTRY[variant_key]
    target_class = resolve_class(variant_entry.class_path)
    transformer_config = transformer.config
    constructor_kwargs = _attention_constructor_kwargs(
        transformer_config, variant_entry
    )

    weight_keys_loaded = 0
    missing_weight_keys: list[str] = []
    unexpected_weight_keys: list[str] = []

    for layer in list(transformer.encoder) + list(transformer.decoder):
        source_attention = layer.attn
        target_attention = target_class(**constructor_kwargs)
        source_parameter = next(source_attention.parameters())
        target_attention = target_attention.to(
            device=source_parameter.device,
            dtype=source_parameter.dtype,
        )
        weight_keys_loaded += _copy_attention_weights(
            source_attention, target_attention
        )
        target_attention.eval()
        layer.attn = target_attention
        layer.positional_bias = transformer_config[
            "attn_positional_bias"
        ]

    return AttentionSwapReport(
        variant_key=variant_key,
        target_class_path=variant_entry.class_path,
        encoder_layers_swapped=len(transformer.encoder),
        decoder_layers_swapped=len(transformer.decoder),
        weight_keys_loaded=weight_keys_loaded,
        missing_weight_keys=tuple(missing_weight_keys),
        unexpected_weight_keys=tuple(unexpected_weight_keys),
    )


def audit_required_weight_keys(
    attention_module: torch.nn.Module,
) -> tuple[str, ...]:
    """Return the required projection/rope keys present on a module.

    Args:
        attention_module: Swapped attention module to inspect.

    Returns:
        Sorted state-dict keys beginning with a required prefix
        (``q_pro.``, ``k_pro.``, ``v_pro.``, ``proj.``, ``rot_pos_enc.``).
    """
    return tuple(
        sorted(
            key
            for key in attention_module.state_dict()
            if key.startswith(_REQUIRED_WEIGHT_PREFIXES)
        )
    )


@dataclass(frozen=True)
class TensorComparison:
    """One pass/fail numerical comparison between two variant outputs.

    Attributes:
        label: Human-readable identifier such as
            ``"relative_pos_vs_cached_dist.raw_A"``.
        max_absolute_difference: Largest elementwise
            ``|candidate - reference|``.
        max_relative_difference: Largest elementwise
            ``|candidate - reference| / max(|reference|, eps)`` where ``eps``
            is ``torch.finfo(dtype).eps``. Near-zero reference entries are
            therefore floored rather than producing unbounded ratios.
        atol: Absolute tolerance used by ``torch.allclose``.
        rtol: Relative tolerance used by ``torch.allclose``.
        passed: Whether ``torch.allclose`` accepted the pair.
    """

    label: str
    max_absolute_difference: float
    max_relative_difference: float
    atol: float
    rtol: float
    passed: bool


@dataclass(frozen=True)
class SanityReport:
    """Results of the full-model fp32 equivalence checks.

    Attributes:
        checkpoint_path: Resolved checkpoint directory.
        device: Device the checks ran on (CPU for the required protocol).
        seq_len: Sequence length of the shared synthetic inputs.
        batch_size: Batch size of the shared synthetic inputs.
        comparisons: One entry per compared tensor pair.
    """

    checkpoint_path: Path
    device: torch.device
    seq_len: int
    batch_size: int
    comparisons: tuple[TensorComparison, ...]


def _max_tensor_differences(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    """Return the maximum absolute and relative elementwise differences.

    Args:
        candidate: Tensor produced by the variant under test.
        reference: Tensor produced by the reference variant.

    Returns:
        Tuple ``(max_absolute_difference, max_relative_difference)``. The
        relative difference divides by ``max(|reference|, eps)`` so entries
        whose reference value is zero contribute a finite, floored ratio
        instead of an unbounded one.

    Raises:
        ValueError: If the two tensors do not have the same shape.
    """
    if candidate.shape != reference.shape:
        raise ValueError(
            f"Cannot compare tensors with shapes {tuple(candidate.shape)} and "
            f"{tuple(reference.shape)}."
        )

    absolute_difference = (candidate - reference).abs()
    relative_floor = torch.finfo(reference.dtype).eps
    relative_difference = absolute_difference / reference.abs().clamp_min(
        relative_floor
    )
    return (
        float(absolute_difference.max()),
        float(relative_difference.max()),
    )


def _compare_tensors(
    label: str,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    atol: float,
    rtol: float,
) -> TensorComparison:
    """Compare two tensors with ``torch.allclose`` and measure their diffs.

    Args:
        label: Human-readable comparison identifier.
        candidate: Tensor produced by the variant under test.
        reference: Tensor produced by the reference variant.
        atol: Absolute tolerance for ``torch.allclose``.
        rtol: Relative tolerance for ``torch.allclose``.

    Returns:
        :class:`TensorComparison` with measured differences and the
        ``torch.allclose`` verdict.
    """
    max_absolute_difference, max_relative_difference = (
        _max_tensor_differences(candidate, reference)
    )
    passed = bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol))
    return TensorComparison(
        label=label,
        max_absolute_difference=max_absolute_difference,
        max_relative_difference=max_relative_difference,
        atol=atol,
        rtol=rtol,
        passed=passed,
    )


def _run_variant_forward(
    checkpoint_path: Path,
    variant_key: str,
    coords: torch.Tensor,
    features: torch.Tensor,
    timepoints: torch.Tensor,
    knn_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a fresh model, swap one variant, and return its outputs.

    Each call reloads the checkpoint so the two variants being compared never
    share modules or weights. The transformer is set to eval mode and the
    forward chain runs under ``torch.no_grad()`` in fp32.

    Args:
        checkpoint_path: Resolved checkpoint directory.
        variant_key: Key in :data:`FULL_MODEL_VARIANT_REGISTRY`.
        coords: Shared synthetic coordinates ``(B, N, 3)``.
        features: Shared synthetic features ``(B, N, 7)``.
        timepoints: Time frame per token ``(B, N)``.
        knn_indices: Shared KNN indices ``(B, N, 16)``.
        device: Device on which to load the model.

    Returns:
        Tuple ``(raw_association, normalized_association)`` where
        ``raw_association`` is the direct ``transformer(...)`` output and
        ``normalized_association`` is
        ``transformer.normalize_output(raw_association, timepoints, coords)``.
    """
    model = load_full_trackastra_model(
        checkpoint_path, device=device, dtype=torch.float32
    )
    transformer = model.transformer
    transformer.eval()
    swap_full_model_attention(transformer, variant_key)

    with torch.no_grad():
        raw_association = transformer(
            coords,
            features,
            padding_mask=None,
            knn_indices=knn_indices,
        )
        normalized_association = transformer.normalize_output(
            raw_association, timepoints, coords
        )
    return raw_association, normalized_association


def run_sanity_checks(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
    seq_len: int = _SANITY_SEQ_LEN,
    batch_size: int = _SANITY_BATCH_SIZE,
    seed: int = 42,
    coord_layout: str = "uniform",
    atol: float = _SANITY_ATOL,
    rtol: float = _SANITY_RTOL,
) -> SanityReport:
    """Run the SPEC 0062 fp32 full-model equivalence checks.

    The checks compare ``relative_pos`` against ``cached_dist`` on the raw
    association matrix and after ``normalize_output``, and compare
    ``sparse_mask_k16`` against ``sparse_gather_k16`` on the raw association
    matrix using the same KNN indices. Every variant is loaded independently
    from the checkpoint, so no module or weight is shared between the two
    sides of a comparison. Inputs are generated once and reused.

    The required protocol is CPU fp32 with TF32 disabled, but the helper
    accepts a device for explicit CUDA runs. TF32 backend flags are restored
    after the checks even when a comparison or forward pass raises.

    Args:
        checkpoint_path: Resolved checkpoint directory.
        device: Device for inputs and models; the required protocol is
            ``"cpu"``.
        seq_len: Sequence length of the shared synthetic inputs.
        batch_size: Batch size of the shared synthetic inputs.
        seed: Synthetic-input seed.
        coord_layout: ``"uniform"`` or ``"tiled"`` spatial layout.
        atol: Absolute tolerance for every comparison.
        rtol: Relative tolerance for every comparison.

    Returns:
        :class:`SanityReport` with one :class:`TensorComparison` per checked
        tensor pair. Each measured difference is printed as it is computed.

    Raises:
        RuntimeError: If any comparison fails. The message contains the
            measured maximum absolute and relative differences; tolerances are
            never loosened implicitly.
    """
    resolved_device = torch.device(device)
    coords, features, timepoints, knn_indices = make_full_model_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        coord_layout=coord_layout,
        device=resolved_device,
        dtype=torch.float32,
        seed=seed,
    )

    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        relative_raw, relative_normalized = _run_variant_forward(
            checkpoint_path,
            "relative_pos",
            coords,
            features,
            timepoints,
            knn_indices,
            resolved_device,
        )
        cached_raw, cached_normalized = _run_variant_forward(
            checkpoint_path,
            "cached_dist",
            coords,
            features,
            timepoints,
            knn_indices,
            resolved_device,
        )
        sparse_mask_raw, _ = _run_variant_forward(
            checkpoint_path,
            "sparse_mask_k16",
            coords,
            features,
            timepoints,
            knn_indices,
            resolved_device,
        )
        sparse_gather_raw, _ = _run_variant_forward(
            checkpoint_path,
            "sparse_gather_k16",
            coords,
            features,
            timepoints,
            knn_indices,
            resolved_device,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    comparison_inputs = (
        (
            "relative_pos_vs_cached_dist.raw_A",
            cached_raw,
            relative_raw,
        ),
        (
            "relative_pos_vs_cached_dist.normalized_A",
            cached_normalized,
            relative_normalized,
        ),
        (
            "sparse_mask_k16_vs_sparse_gather_k16.raw_A",
            sparse_gather_raw,
            sparse_mask_raw,
        ),
    )

    comparisons: list[TensorComparison] = []
    for label, candidate, reference in comparison_inputs:
        comparison = _compare_tensors(label, candidate, reference, atol, rtol)
        print(
            f"sanity {label}: "
            f"max_abs_diff={comparison.max_absolute_difference:.6e} "
            f"max_rel_diff={comparison.max_relative_difference:.6e} "
            f"atol={comparison.atol:g} rtol={comparison.rtol:g} "
            f"{'PASS' if comparison.passed else 'FAIL'}"
        )
        comparisons.append(comparison)

    report = SanityReport(
        checkpoint_path=checkpoint_path,
        device=resolved_device,
        seq_len=seq_len,
        batch_size=batch_size,
        comparisons=tuple(comparisons),
    )

    failed_comparisons = [
        comparison for comparison in comparisons if not comparison.passed
    ]
    if failed_comparisons:
        failure_details = "; ".join(
            f"{comparison.label}: "
            f"max_abs_diff={comparison.max_absolute_difference:.6e}, "
            f"max_rel_diff={comparison.max_relative_difference:.6e} "
            f"(atol={comparison.atol:g}, rtol={comparison.rtol:g})"
            for comparison in failed_comparisons
        )
        raise RuntimeError(
            "Full-model fp32 sanity check failed (tolerances are not "
            f"loosened): {failure_details}"
        )

    return report


def _dtype_name(dtype: torch.dtype) -> str:
    """Return the short CSV/JSON name for a floating torch dtype.

    Args:
        dtype: Torch dtype used by the model and inputs.

    Returns:
        ``"fp32"``, ``"fp16"``, ``"bf16"``, or the torch dtype string with
        the ``torch.`` prefix removed for an unexpected dtype.
    """
    names_by_dtype = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }
    return names_by_dtype.get(dtype, str(dtype).removeprefix("torch."))


def _ensure_parent_directory(path: Path) -> None:
    """Create the parent directory of an output path when needed.

    Args:
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _checkpoint_sha256(checkpoint_path: Path) -> str:
    """Compute the SHA-256 of the checkpoint ``model.pt`` file.

    Args:
        checkpoint_path: Directory containing the checkpoint files.

    Returns:
        Lowercase hexadecimal digest.

    Raises:
        FileNotFoundError: If ``model.pt`` is absent.
    """
    model_path = checkpoint_path / "model.pt"
    digest = hashlib.sha256()
    with open(model_path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> tuple[str | None, bool | None]:
    """Read the current git commit and dirty-tree flag without raising.

    Returns:
        Tuple ``(commit, dirty)``. Both are ``None`` when git is unavailable
        or the project is not inside a git work tree.
    """
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None, None

    if commit_result.returncode != 0 or status_result.returncode != 0:
        return None, None
    return commit_result.stdout.strip(), bool(status_result.stdout.strip())


def _motile_available() -> bool:
    """Return whether the optional ``motile`` ILP package is importable."""
    return importlib.util.find_spec("motile") is not None


def _write_environment_json(
    env_json_path: Path,
    *,
    command: str,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint_path: Path,
    model_config: dict[str, object],
    variant_keys: list[str],
    sequence_lengths: list[int],
    batch_sizes: list[int],
    repeats: int,
    warmup: int,
    seed: int,
    coord_layout: str,
    knn_neighbors: int,
    linker_mode: str,
) -> None:
    """Write the SPEC 0062 §4.3 environment record.

    Only the named provenance fields are written; no environment dump or
    secret-bearing value is included.

    Args:
        env_json_path: Destination JSON path.
        command: Command line used to launch the run.
        device: Concrete device used for the grid.
        dtype: Model/input dtype.
        checkpoint_path: Resolved checkpoint directory.
        model_config: ``transformer.config`` read from the checkpoint.
        variant_keys: Variants run in the grid.
        sequence_lengths: Sequence lengths run in the grid.
        batch_sizes: Batch sizes run in the grid.
        repeats: Timed repeats per configuration.
        warmup: Warmup iterations per configuration.
        seed: Synthetic-input seed.
        coord_layout: Synthetic spatial layout.
        knn_neighbors: Number of KNN neighbours used for inputs.
        linker_mode: Requested linker mode.
    """
    git_commit, git_dirty = _git_metadata()
    gpu_name = (
        torch.cuda.get_device_name(0)
        if device.type == "cuda" and torch.cuda.is_available()
        else None
    )
    environment_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "device": str(device),
        "dtype": _dtype_name(dtype),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "model_config": model_config,
        "variants": list(variant_keys),
        "Ns": list(sequence_lengths),
        "batch_sizes": list(batch_sizes),
        "repeats": repeats,
        "warmup": warmup,
        "seed": seed,
        "coord_layout": coord_layout,
        "knn_neighbors": knn_neighbors,
        "linker_mode": linker_mode,
        "motile_available": _motile_available(),
    }
    _ensure_parent_directory(env_json_path)
    with open(env_json_path, "w") as env_file:
        json.dump(environment_record, env_file, indent=2, default=str)
        env_file.write("\n")


def _failure_status(exception: BaseException) -> str:
    """Map an exception to the SPEC 0062 raw-CSV status string.

    Args:
        exception: Exception raised by setup, forward, or normalization.

    Returns:
        ``"oom"`` for an out-of-memory error, otherwise ``"err: <message>"``.
    """
    message = str(exception)
    if "out of memory" in message.lower():
        return "oom"
    return f"err: {message}"


def _clear_device_cache(device: torch.device) -> None:
    """Run ``gc.collect()`` and release the CUDA caching-allocator cache.

    Called before every repeat's baseline capture (the same
    ``gc.collect()`` + ``empty_cache()`` protocol as the isolated
    attention benchmark, ``src/bench/timing.py::measure``) and once in
    each repeat's ``finally`` block so no large tensor survives into the
    next measurement or a subsequent configuration.

    Args:
        device: Device whose cache should be cleared. ``empty_cache()``
            runs only when ``device`` is a CUDA device and CUDA is
            available; otherwise only ``gc.collect()`` runs.
    """
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _measure_standalone_dist_2d(
    coords: torch.Tensor,
    device: torch.device,
) -> float:
    """Time one standalone ``torch.cdist`` on the spatial coordinates.

    This estimates a cost already included in the forward chain; it is never
    subtracted from any CSV column.

    Args:
        coords: Full coordinate tensor ``(B, N, 3)``.
        device: Device holding ``coords``; selects CUDA or CPU timing.

    Returns:
        Elapsed milliseconds for the standalone distance matrix.
    """
    spatial_coords = coords[..., 1:].float()
    timer = ComponentTimer(use_cuda=device.type == "cuda")
    timer.start("dist_2d_standalone_start")
    torch.cdist(spatial_coords, spatial_coords)
    timer.stop("dist_2d_standalone_end")
    return timer.elapsed_ms(
        "dist_2d_standalone_start", "dist_2d_standalone_end"
    )


@dataclass(frozen=True)
class PreparedInputs:
    """Synthetic inputs and the standalone distance cost for one ``(N, B)``.

    Attributes:
        coords: Coordinate tensor ``(B, N, 3)``, or ``None`` if setup failed.
        features: Feature tensor ``(B, N, 7)``, or ``None`` if setup failed.
        timepoints: Timepoint tensor ``(B, N)``, or ``None`` if setup failed.
        knn_indices: KNN index tensor ``(B, N, 16)``, or ``None`` if setup
            failed.
        dist_2d_standalone_ms: Standalone distance-matrix time, or ``None`` if
            setup failed.
        failure_status: Raw-CSV status for a failed setup, or ``None``.
        failure_error: Exception type/message for a failed setup, or ``None``.
    """

    coords: torch.Tensor | None
    features: torch.Tensor | None
    timepoints: torch.Tensor | None
    knn_indices: torch.Tensor | None
    dist_2d_standalone_ms: float | None
    failure_status: str | None
    failure_error: str | None


def _prepare_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    coord_layout: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    knn_neighbors: int,
) -> PreparedInputs:
    """Build one ``(N, B)`` input bundle and measure its standalone ``dist_2d``.

    Args:
        batch_size: Batch size.
        sequence_length: Number of tokens per batch element.
        coord_layout: ``"uniform"`` or ``"tiled"``.
        device: Device for the returned tensors.
        dtype: Floating dtype for coordinates and features.
        seed: Synthetic-input seed.
        knn_neighbors: Number of KNN neighbours.

    Returns:
        :class:`PreparedInputs`; setup failures are captured instead of
        raised so the grid can write placeholder failure rows and continue.
    """
    try:
        coords, features, timepoints, knn_indices = make_full_model_inputs(
            batch_size=batch_size,
            seq_len=sequence_length,
            knn_neighbors=knn_neighbors,
            coord_layout=coord_layout,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        dist_2d_standalone_ms = _measure_standalone_dist_2d(coords, device)
    except Exception as setup_error:
        _clear_device_cache(device)
        return PreparedInputs(
            coords=None,
            features=None,
            timepoints=None,
            knn_indices=None,
            dist_2d_standalone_ms=None,
            failure_status=_failure_status(setup_error),
            failure_error=f"{type(setup_error).__name__}: {setup_error}",
        )
    return PreparedInputs(
        coords=coords,
        features=features,
        timepoints=timepoints,
        knn_indices=knn_indices,
        dist_2d_standalone_ms=dist_2d_standalone_ms,
        failure_status=None,
        failure_error=None,
    )


def _make_start_hook(
    timer: ComponentTimer,
    event_name: str,
) -> Callable[..., None]:
    """Create a forward-pre hook that records ``event_name`` on the timer.

    Args:
        timer: Timer shared by all hooks in one repeat.
        event_name: Boundary name recorded when the module forward starts.

    Returns:
        Hook callable accepted by ``register_forward_pre_hook``.
    """

    def start_hook(module, inputs) -> None:
        """Record the module's entry boundary."""
        timer.start(event_name)

    return start_hook


def _make_stop_hook(
    timer: ComponentTimer,
    event_name: str,
) -> Callable[..., None]:
    """Create a forward hook that records ``event_name`` on the timer.

    Args:
        timer: Timer shared by all hooks in one repeat.
        event_name: Boundary name recorded when the module forward ends.

    Returns:
        Hook callable accepted by ``register_forward_hook``.
    """

    def stop_hook(module, inputs, output) -> None:
        """Record the module's exit boundary."""
        timer.stop(event_name)

    return stop_hook


def _register_component_timing_hooks(
    transformer: torch.nn.Module,
    timer: ComponentTimer,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register the SPEC 0062 component boundary hooks on a transformer.

    Args:
        transformer: Loaded ``TrackingTransformer`` with swapped attention.
        timer: Timer that receives every boundary event.

    Returns:
        Removable hook handles; the caller must remove them after the repeat.
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def add_both(module: torch.nn.Module, name: str) -> None:
        """Attach start/stop hooks named ``name`` to one module."""
        handles.append(
            module.register_forward_pre_hook(
                _make_start_hook(timer, f"{name}_start")
            )
        )
        handles.append(
            module.register_forward_hook(
                _make_stop_hook(timer, f"{name}_end")
            )
        )

    add_both(transformer, "forward")
    add_both(transformer.proj, "embed_proj")

    for layer_index, encoder_layer in enumerate(transformer.encoder):
        add_both(encoder_layer, f"encoder_{layer_index}")
        add_both(encoder_layer.attn, f"encoder_attn_{layer_index}")

    for layer_index, decoder_layer in enumerate(transformer.decoder):
        add_both(decoder_layer, f"decoder_{layer_index}")
        add_both(decoder_layer.attn, f"decoder_attn_{layer_index}")

    add_both(transformer.head_x, "head_x")
    add_both(transformer.head_y, "head_y")
    return handles


def _remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove every hook handle registered for one repeat.

    Args:
        handles: Handles returned by
            :func:`_register_component_timing_hooks`.
    """
    for handle in handles:
        handle.remove()


def _base_raw_row(
    *,
    variant_key: str,
    sequence_length: int,
    batch_size: int,
    repeat_index: int,
    status: str,
    checkpoint_path: Path,
    dtype: torch.dtype,
    seed: int,
    knn_neighbors: int,
    coord_layout: str,
    linker_mode: str,
    error: str,
) -> dict[str, object]:
    """Build one raw-CSV row with all SPEC 0062 §4.1 columns.

    Numeric cells start empty so failed rows keep their place without
    inventing timings; successful rows are filled by
    :func:`_fill_ok_timings`.

    Args:
        variant_key: Variant registry key.
        sequence_length: Grid ``N``.
        batch_size: Grid ``B``.
        repeat_index: Zero-based repeat index.
        status: ``ok``, ``oom``, or ``err: <message>``.
        checkpoint_path: Resolved checkpoint path recorded in the CSV.
        dtype: Model/input dtype.
        seed: Synthetic-input seed.
        knn_neighbors: Number of KNN neighbours.
        coord_layout: Synthetic spatial layout.
        linker_mode: Requested linker mode.
        error: Empty on success, otherwise exception type/message.

    Returns:
        Row dictionary keyed by every raw-CSV header column.
    """
    row: dict[str, object] = {
        "variant": variant_key,
        "N": sequence_length,
        "B": batch_size,
        "repeat_idx": repeat_index,
        "status": status,
        "checkpoint": str(checkpoint_path),
        "dtype": _dtype_name(dtype),
        "seed": seed,
        "knn_neighbors": knn_neighbors,
        "coord_layout": coord_layout,
        "linker_mode": linker_mode,
        "linker_status": (
            _LINKER_STATUS_SKIPPED_B_GT_1 if batch_size > 1 else ""
        ),
        "error": error,
    }
    for timing_column in _TIMING_COLUMNS + _LINKER_TIMING_COLUMNS:
        row[timing_column] = ""
    return row


def _fill_ok_timings(
    row: dict[str, object],
    *,
    full_forward_ms: float,
    embed_ms: float,
    encoder_ms: float,
    encoder_attn_ms: float,
    decoder_ms: float,
    decoder_attn_ms: float,
    head_logits_ms: float,
    head_x_ms: float,
    head_y_ms: float,
    einsum_ms: float,
    embed_proj_ms: float,
    normalize_ms: float,
    peak_mem_mb: float,
    baseline_mem_mb: float,
    peak_mem_total_mb: float,
    peak_mem_reserved_mb: float,
    forward_peak_mem_total_mb: float,
    forward_peak_mem_reserved_mb: float,
    normalize_peak_mem_total_mb: float,
    normalize_peak_mem_reserved_mb: float,
    dist_2d_standalone_ms: float,
    linker_ms: float | None = None,
) -> None:
    """Fill every timing column on an ``ok`` raw row.

    ``total_ms`` includes the linker only when ``linker_ms`` is provided. This
    keeps the SPEC's ``B>1`` formula (forward plus normalization) and the
    ``B=1`` formula (forward plus normalization plus linker) in one place.

    The memory columns report the total peak VRAM (allocated and reserved)
    per repeat and per stage, plus the pre-repeat allocated baseline;
    ``peak_mem_mb`` keeps its incremental-delta meaning
    (``max(0, peak_mem_total_mb - baseline_mem_mb)``) for compatibility with
    earlier runs (SPEC 0062 §12).

    Args:
        row: Mutable raw-CSV row.
        full_forward_ms: Event pair around the transformer forward.
        embed_ms: Forward start to encoder[0] entry.
        encoder_ms: Encoder[0] entry to encoder[-1] exit.
        encoder_attn_ms: Sum of encoder attention intervals.
        decoder_ms: Decoder[0] entry to decoder[-1] exit.
        decoder_attn_ms: Sum of decoder attention intervals.
        head_logits_ms: Decoder[-1] exit to forward exit.
        head_x_ms: ``head_x`` hook interval.
        head_y_ms: ``head_y`` hook interval.
        einsum_ms: ``head_y`` exit to forward exit.
        embed_proj_ms: ``transformer.proj`` hook interval.
        normalize_ms: Event pair around ``normalize_output``.
        peak_mem_mb: Incremental CUDA peak (total peak minus baseline, floored
            at 0), or ``0.0`` on CPU.
        baseline_mem_mb: CUDA memory allocated before the repeat, or ``0.0``
            on CPU.
        peak_mem_total_mb: Total peak VRAM allocated across forward plus
            normalization, or ``0.0`` on CPU.
        peak_mem_reserved_mb: Total peak VRAM reserved by the caching
            allocator across forward plus normalization, or ``0.0`` on CPU.
        forward_peak_mem_total_mb: Peak allocated captured after the forward,
            or ``0.0`` on CPU.
        forward_peak_mem_reserved_mb: Peak reserved captured after the
            forward, or ``0.0`` on CPU.
        normalize_peak_mem_total_mb: Peak allocated captured after
            ``normalize_output``, or ``0.0`` on CPU.
        normalize_peak_mem_reserved_mb: Peak reserved captured after
            ``normalize_output``, or ``0.0`` on CPU.
        dist_2d_standalone_ms: Standalone model-level distance-matrix time.
        linker_ms: Total CPU linker time, or ``None`` when the linker was
            skipped or failed. A failed linker leaves its cells empty but
            does not change the successful forward timings.
    """
    row["full_forward_ms"] = full_forward_ms
    row["embed_ms"] = embed_ms
    row["encoder_ms"] = encoder_ms
    row["encoder_attn_ms"] = encoder_attn_ms
    row["decoder_ms"] = decoder_ms
    row["decoder_attn_ms"] = decoder_attn_ms
    row["head_logits_ms"] = head_logits_ms
    row["head_x_ms"] = head_x_ms
    row["head_y_ms"] = head_y_ms
    row["einsum_ms"] = einsum_ms
    row["embed_proj_ms"] = embed_proj_ms
    row["normalize_ms"] = normalize_ms
    row["total_ms"] = (
        full_forward_ms + normalize_ms + (linker_ms or 0.0)
    )
    row["peak_mem_mb"] = peak_mem_mb
    row["baseline_mem_mb"] = baseline_mem_mb
    row["peak_mem_total_mb"] = peak_mem_total_mb
    row["peak_mem_reserved_mb"] = peak_mem_reserved_mb
    row["forward_peak_mem_total_mb"] = forward_peak_mem_total_mb
    row["forward_peak_mem_reserved_mb"] = forward_peak_mem_reserved_mb
    row["normalize_peak_mem_total_mb"] = normalize_peak_mem_total_mb
    row["normalize_peak_mem_reserved_mb"] = normalize_peak_mem_reserved_mb
    row["dist_2d_standalone_ms"] = dist_2d_standalone_ms
    row["forward_closure_error_ms"] = full_forward_ms - (
        embed_ms + encoder_ms + decoder_ms + head_logits_ms
    )


def _sum_hook_intervals(
    timer: ComponentTimer,
    event_prefix: str,
    layer_count: int,
) -> float:
    """Sum one hook interval per layer.

    Args:
        timer: Timer holding the recorded boundary events.
        event_prefix: Prefix before ``_<index>_start``/``_end``.
        layer_count: Number of layers whose intervals are summed.

    Returns:
        Sum of the per-layer elapsed milliseconds.
    """
    return float(
        sum(
            timer.elapsed_ms(
                f"{event_prefix}_{layer_index}_start",
                f"{event_prefix}_{layer_index}_end",
            )
            for layer_index in range(layer_count)
        )
    )


class ConsecutiveFrameWeightLookup:
    """Read-only consecutive-frame view over a normalized association matrix.

    ``build_graph`` asks this callable for a candidate ``(node_in, node_out)``
    weight. The view returns ``association_matrix[node_in, node_out]`` only
    when ``time_by_node[node_out] == time_by_node[node_in] + 1`` and the
    weight reaches the Trackastra edge threshold; otherwise it returns
    ``None``. It therefore exposes only the edges the linker can use and
    never builds the full ``N x N`` Python dictionary.

    Attributes:
        _association_matrix: CPU tensor of shape ``(N, N)`` holding the
            normalized association weights.
        _time_by_node: CPU integer tensor of shape ``(N,)`` holding one frame
            per linker node.
    """

    def __init__(
        self,
        association_matrix: torch.Tensor,
        time_by_node: torch.Tensor,
    ) -> None:
        """Store the read-only matrix and frame labels.

        Args:
            association_matrix: Normalized association matrix ``(N, N)`` on
                CPU. It is referenced, not copied.
            time_by_node: Frame index per linker node ``(N,)`` on CPU.

        Raises:
            ValueError: If the matrix is not square or the frame labels do not
                match the matrix size.
        """
        if association_matrix.ndim != 2:
            raise ValueError(
                "association_matrix must be 2-D, got shape "
                f"{tuple(association_matrix.shape)}."
            )
        if association_matrix.shape[0] != association_matrix.shape[1]:
            raise ValueError(
                "association_matrix must be square, got shape "
                f"{tuple(association_matrix.shape)}."
            )
        if time_by_node.ndim != 1:
            raise ValueError(
                "time_by_node must be 1-D, got shape "
                f"{tuple(time_by_node.shape)}."
            )
        if time_by_node.shape[0] != association_matrix.shape[0]:
            raise ValueError(
                "time_by_node length "
                f"{time_by_node.shape[0]} does not match matrix size "
                f"{association_matrix.shape[0]}."
            )
        self._association_matrix = association_matrix
        self._time_by_node = time_by_node

    def __call__(
        self, node_in: Hashable, node_out: Hashable
    ) -> float | None:
        """Return the weight for a consecutive-frame edge, else ``None``.

        Args:
            node_in: Source linker node id (an integer index).
            node_out: Target linker node id (an integer index).

        Returns:
            ``association_matrix[node_in, node_out]`` as a Python float when
            ``node_out`` is exactly one frame after ``node_in`` and the
            weight is at least ``_LINKER_EDGE_THRESHOLD``; otherwise
            ``None``.
        """
        node_in_index = int(node_in)
        node_out_index = int(node_out)
        if (
            int(self._time_by_node[node_out_index])
            != int(self._time_by_node[node_in_index]) + 1
        ):
            return None
        weight = float(
            self._association_matrix[node_in_index, node_out_index]
        )
        if weight < _LINKER_EDGE_THRESHOLD:
            return None
        return weight


@dataclass(frozen=True)
class LinkerTiming:
    """Outcome of one linker timing attempt.

    Attributes:
        linker_status: ``ok``, ``skipped_b_gt_1``, ``ilp_unavailable``,
            ``oom``, or ``err: <message>``.
        build_graph_ms: CPU wall time of ``build_graph``, or ``None`` when the
            linker was skipped or failed.
        tracker_ms: CPU wall time of the tracker stage (greedy or ILP), or
            ``None`` when the linker was skipped or failed.
        linker_ms: Sum of ``build_graph_ms`` and ``tracker_ms``, or ``None``
            when the linker was skipped or failed.
    """

    linker_status: str
    build_graph_ms: float | None
    tracker_ms: float | None
    linker_ms: float | None


def _build_linker_nodes(
    timepoints_for_sample: torch.Tensor,
    coords_for_sample: torch.Tensor,
) -> list[dict[str, object]]:
    """Build the linker node dictionaries for one batch element.

    Args:
        timepoints_for_sample: Frame per token ``(N,)`` on CPU.
        coords_for_sample: Coordinates ``(N, 3)`` on CPU; columns 1-2 are the
            spatial ``y``/``x`` values passed to the linker.

    Returns:
        List of ``{"id", "time", "label", "coords"}`` dictionaries in token
        order.
    """
    time_values = [int(time_value) for time_value in timepoints_for_sample]
    spatial_coords = coords_for_sample[:, 1:].tolist()
    return [
        {
            "id": node_index,
            "time": time_values[node_index],
            "label": node_index,
            "coords": [
                float(spatial_coords[node_index][0]),
                float(spatial_coords[node_index][1]),
            ],
        }
        for node_index in range(len(time_values))
    ]


def _time_greedy_linker(
    nodes: list[dict[str, object]],
    weight_lookup: ConsecutiveFrameWeightLookup,
) -> LinkerTiming:
    """Time graph construction and greedy tracking on the CPU.

    Args:
        nodes: Linker node dictionaries.
        weight_lookup: Lazy consecutive-frame weight view over the normalized
            association matrix.

    Returns:
        :class:`LinkerTiming` with the ``ok`` status and stage timings, or an
        ``oom``/``err`` status with empty timings when either stage raises.
    """
    try:
        build_start = time.perf_counter()
        candidate_graph = build_graph(
            nodes,
            weight_lookup=weight_lookup,
            use_distance=False,
            max_distance=_LINKER_MAX_DISTANCE,
            max_neighbors=_LINKER_MAX_NEIGHBORS,
            delta_t=_LINKER_DELTA_T,
        )
        build_end = time.perf_counter()
        tracker_start = time.perf_counter()
        track_greedy(
            candidate_graph,
            allow_divisions=_LINKER_ALLOW_DIVISIONS,
            threshold=_LINKER_THRESHOLD,
        )
        tracker_end = time.perf_counter()
    except Exception as linker_error:
        return LinkerTiming(
            linker_status=_failure_status(linker_error),
            build_graph_ms=None,
            tracker_ms=None,
            linker_ms=None,
        )

    build_graph_ms = (build_end - build_start) * 1000.0
    tracker_ms = (tracker_end - tracker_start) * 1000.0
    return LinkerTiming(
        linker_status=_LINKER_STATUS_OK,
        build_graph_ms=build_graph_ms,
        tracker_ms=tracker_ms,
        linker_ms=build_graph_ms + tracker_ms,
    )


def _time_ilp_linker(
    nodes: list[dict[str, object]],
    weight_lookup: ConsecutiveFrameWeightLookup,
) -> LinkerTiming:
    """Time graph construction and the optional motile ILP tracker on CPU.

    ``track_ilp`` is imported lazily because importing it raises when the
    optional ``motile`` package is absent. The ILP solver time is stored in
    the ``linker_greedy_ms`` column, which holds the second linker stage for
    either mode.

    Args:
        nodes: Linker node dictionaries.
        weight_lookup: Lazy consecutive-frame weight view over the normalized
            association matrix.

    Returns:
        :class:`LinkerTiming` with the ``ok`` status and stage timings,
        ``ilp_unavailable`` when ``motile`` is absent, or an ``oom``/``err``
        status with empty timings when the linker raises.
    """
    try:
        from trackastra.tracking.ilp import track_ilp
    except ImportError:
        return LinkerTiming(
            linker_status=_LINKER_STATUS_ILP_UNAVAILABLE,
            build_graph_ms=None,
            tracker_ms=None,
            linker_ms=None,
        )

    try:
        build_start = time.perf_counter()
        candidate_graph = build_graph(
            nodes,
            weight_lookup=weight_lookup,
            use_distance=False,
            max_distance=_LINKER_MAX_DISTANCE,
            max_neighbors=_LINKER_MAX_NEIGHBORS,
            delta_t=_LINKER_DELTA_T,
        )
        build_end = time.perf_counter()
        tracker_start = time.perf_counter()
        track_ilp(candidate_graph, ilp_config="gt")
        tracker_end = time.perf_counter()
    except Exception as linker_error:
        return LinkerTiming(
            linker_status=_failure_status(linker_error),
            build_graph_ms=None,
            tracker_ms=None,
            linker_ms=None,
        )

    build_graph_ms = (build_end - build_start) * 1000.0
    tracker_ms = (tracker_end - tracker_start) * 1000.0
    return LinkerTiming(
        linker_status=_LINKER_STATUS_OK,
        build_graph_ms=build_graph_ms,
        tracker_ms=tracker_ms,
        linker_ms=build_graph_ms + tracker_ms,
    )


def _prepare_and_time_linker(
    normalized_association: torch.Tensor,
    timepoints: torch.Tensor,
    coords: torch.Tensor,
    *,
    batch_size: int,
    linker_mode: str,
) -> LinkerTiming:
    """Copy the linker inputs to CPU and time the requested linker.

    The normalized association matrix, timepoints, and coordinates are moved
    to CPU before any ``perf_counter`` call, so copies are excluded from
    ``linker_build_graph_ms``/``linker_greedy_ms``. Linker failures are
    captured and reported instead of raised, which keeps a successful forward
    row valid.

    Args:
        normalized_association: ``normalize_output`` result ``(B, N, N)``.
        timepoints: Frame per token ``(B, N)``.
        coords: Coordinates ``(B, N, 3)``.
        batch_size: Current grid batch size. Linker timing is skipped for
            ``B > 1``.
        linker_mode: ``"greedy"`` or ``"ilp"``.

    Returns:
        :class:`LinkerTiming`; skipped, unavailable, successful, or failed.
    """
    if batch_size > 1:
        return LinkerTiming(
            linker_status=_LINKER_STATUS_SKIPPED_B_GT_1,
            build_graph_ms=None,
            tracker_ms=None,
            linker_ms=None,
        )

    try:
        association_cpu = normalized_association.detach().to("cpu")
        timepoints_cpu = timepoints.detach().to("cpu")
        coords_cpu = coords.detach().to("cpu")
        timepoints_for_sample = timepoints_cpu[0]
        nodes = _build_linker_nodes(
            timepoints_for_sample, coords_cpu[0]
        )
        weight_lookup = ConsecutiveFrameWeightLookup(
            association_cpu[0], timepoints_for_sample
        )
        if linker_mode == "ilp":
            return _time_ilp_linker(nodes, weight_lookup)
        return _time_greedy_linker(nodes, weight_lookup)
    except Exception as linker_error:
        return LinkerTiming(
            linker_status=_failure_status(linker_error),
            build_graph_ms=None,
            tracker_ms=None,
            linker_ms=None,
        )


def _apply_linker_timing(
    row: dict[str, object], linker_timing: LinkerTiming
) -> None:
    """Write the linker status and available numeric cells to a raw row.

    Args:
        row: Mutable raw-CSV row.
        linker_timing: Outcome for this repeat.
    """
    row["linker_status"] = linker_timing.linker_status
    if linker_timing.build_graph_ms is not None:
        row["linker_build_graph_ms"] = linker_timing.build_graph_ms
    if linker_timing.tracker_ms is not None:
        row["linker_greedy_ms"] = linker_timing.tracker_ms
    if linker_timing.linker_ms is not None:
        row["linker_ms"] = linker_timing.linker_ms


def _write_raw_row(writer: csv.writer, row: dict[str, object]) -> None:
    """Write one raw-CSV row in the exact SPEC §4.1 column order.

    Args:
        writer: Open CSV writer.
        row: Row dictionary with every header key.
    """
    writer.writerow([row[column] for column in RAW_CSV_HEADER])


def _write_summary_placeholder(summary_path: Path) -> None:
    """Write an explicit F3 placeholder where F6 will put the summary table.

    Args:
        summary_path: Destination summary path from ``--summary``.
    """
    _ensure_parent_directory(summary_path)
    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["placeholder"])
        writer.writerow(
            [
                "F3 placeholder; F6 owns the summary table. "
                "Use the raw CSV as the source of truth."
            ]
        )


@dataclass(frozen=True)
class GridRunResult:
    """Artifacts and rows produced by one :func:`run_grid` invocation.

    Attributes:
        out_path: Raw CSV path written by the run.
        env_json_path: Environment JSON path written by the run.
        summary_path: Placeholder summary path written by the run.
        rows: Raw-CSV rows in write order.
    """

    out_path: Path
    env_json_path: Path
    summary_path: Path
    rows: tuple[dict[str, object], ...]


def run_grid(
    checkpoint_path: Path,
    variant_keys: list[str],
    sequence_lengths: list[int],
    batch_sizes: list[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    coord_layout: str,
    repeats: int,
    warmup: int,
    linker_mode: str,
    out_path: Path,
    env_json_path: Path,
    summary_path: Path,
    knn_neighbors: int = _KNN_NEIGHBORS,
    command: str | None = None,
) -> GridRunResult:
    """Run the SPEC 0062 timed grid and write raw CSV, env JSON, and summary.

    A fresh model is loaded per variant and reused across the requested
    ``(N, B)`` shapes. Inputs and the standalone ``dist_2d`` estimate are
    cached per ``(N, B)`` and shared across variants. Each repeat
    synchronises, clears the cache through :func:`_clear_device_cache`
    (``gc.collect()`` plus ``empty_cache()``, as in the isolated
    attention benchmark ``src/bench/timing.py::measure``), synchronises
    again, resets peak-memory statistics, and captures the pre-repeat
    baseline before registering component hooks and running the forward
    and ``normalize_output`` under ``torch.no_grad()``. It records the
    total peak VRAM (allocated and reserved) per stage plus the
    pre-repeat baseline (``0.0`` on CPU; SPEC 0062 §12) and writes one raw
    row. After every repeat the association tensors are released and the
    cache is cleared once outside the timed region, so no ``N x N`` tensor
    survives into the next repeat. Failures are recorded as ``oom``/``err``
    rows and the grid continues.

    Args:
        checkpoint_path: Resolved checkpoint directory.
        variant_keys: Registered variant keys to run.
        sequence_lengths: Sequence lengths to run.
        batch_sizes: Batch sizes to run.
        seed: Synthetic-input seed.
        device: Concrete device for models and inputs.
        dtype: Model/input dtype.
        coord_layout: ``"uniform"`` or ``"tiled"``.
        repeats: Timed repeats per configuration.
        warmup: Untimed warmup forwards per configuration.
        linker_mode: Requested linker mode; ``"greedy"`` runs
            ``build_graph`` plus ``track_greedy`` and ``"ilp"`` runs
            ``build_graph`` plus the optional ``motile`` ILP.
        out_path: Raw CSV destination.
        env_json_path: Environment JSON destination.
        summary_path: Placeholder summary destination.
        knn_neighbors: Number of KNN neighbours for synthetic inputs.
        command: Command line recorded in the environment JSON. Defaults to
            ``sys.argv``.

    Returns:
        :class:`GridRunResult` with every written row.

    Raises:
        KeyError: If ``variant_keys`` contains an unregistered key.
        ValueError: If ``repeats`` or ``warmup`` is negative.
    """
    unknown_variants = [
        key for key in variant_keys if key not in FULL_MODEL_VARIANT_REGISTRY
    ]
    if unknown_variants:
        raise KeyError(
            f"Unknown full-model variants {unknown_variants}; available: "
            f"{sorted(FULL_MODEL_VARIANT_REGISTRY)}"
        )
    if repeats < 0 or warmup < 0:
        raise ValueError(
            f"repeats and warmup must be non-negative, got {repeats} and "
            f"{warmup}."
        )

    resolved_command = command or " ".join(sys.argv)
    rows: list[dict[str, object]] = []
    prepared_inputs_by_shape: dict[tuple[int, int], PreparedInputs] = {}
    model_config: dict[str, object] | None = None

    _ensure_parent_directory(out_path)
    with open(out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(RAW_CSV_HEADER)
        csv_file.flush()

        for variant_key in variant_keys:
            try:
                model = load_full_trackastra_model(
                    checkpoint_path, device=device, dtype=dtype
                )
                transformer = model.transformer
                transformer.eval()
                swap_full_model_attention(transformer, variant_key)
            except Exception as variant_error:
                failure_status = _failure_status(variant_error)
                failure_error = (
                    f"{type(variant_error).__name__}: {variant_error}"
                )
                for batch_size in batch_sizes:
                    for sequence_length in sequence_lengths:
                        for repeat_index in range(repeats):
                            row = _base_raw_row(
                                variant_key=variant_key,
                                sequence_length=sequence_length,
                                batch_size=batch_size,
                                repeat_index=repeat_index,
                                status=failure_status,
                                checkpoint_path=checkpoint_path,
                                dtype=dtype,
                                seed=seed,
                                knn_neighbors=knn_neighbors,
                                coord_layout=coord_layout,
                                linker_mode=linker_mode,
                                error=failure_error,
                            )
                            _write_raw_row(writer, row)
                            csv_file.flush()
                            rows.append(row)
                _clear_device_cache(device)
                continue

            if model_config is None:
                model_config = dict(transformer.config)
                _write_environment_json(
                    env_json_path,
                    command=resolved_command,
                    device=device,
                    dtype=dtype,
                    checkpoint_path=checkpoint_path,
                    model_config=model_config,
                    variant_keys=variant_keys,
                    sequence_lengths=sequence_lengths,
                    batch_sizes=batch_sizes,
                    repeats=repeats,
                    warmup=warmup,
                    seed=seed,
                    coord_layout=coord_layout,
                    knn_neighbors=knn_neighbors,
                    linker_mode=linker_mode,
                )

            for batch_size in batch_sizes:
                for sequence_length in sequence_lengths:
                    shape_key = (batch_size, sequence_length)
                    if shape_key not in prepared_inputs_by_shape:
                        prepared_inputs_by_shape[shape_key] = _prepare_inputs(
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            coord_layout=coord_layout,
                            device=device,
                            dtype=dtype,
                            seed=seed,
                            knn_neighbors=knn_neighbors,
                        )
                    prepared_inputs = prepared_inputs_by_shape[shape_key]

                    if prepared_inputs.failure_status is not None:
                        for repeat_index in range(repeats):
                            row = _base_raw_row(
                                variant_key=variant_key,
                                sequence_length=sequence_length,
                                batch_size=batch_size,
                                repeat_index=repeat_index,
                                status=prepared_inputs.failure_status,
                                checkpoint_path=checkpoint_path,
                                dtype=dtype,
                                seed=seed,
                                knn_neighbors=knn_neighbors,
                                coord_layout=coord_layout,
                                linker_mode=linker_mode,
                                error=prepared_inputs.failure_error or "",
                            )
                            _write_raw_row(writer, row)
                            csv_file.flush()
                            rows.append(row)
                        continue

                    assert prepared_inputs.coords is not None
                    assert prepared_inputs.features is not None
                    assert prepared_inputs.timepoints is not None
                    assert prepared_inputs.knn_indices is not None
                    assert prepared_inputs.dist_2d_standalone_ms is not None

                    coords = prepared_inputs.coords
                    features = prepared_inputs.features
                    timepoints = prepared_inputs.timepoints
                    knn_indices = prepared_inputs.knn_indices

                    with torch.no_grad():
                        try:
                            for _ in range(warmup):
                                transformer(
                                    coords,
                                    features,
                                    padding_mask=None,
                                    knn_indices=knn_indices,
                                )
                        except Exception:
                            _clear_device_cache(device)

                    for repeat_index in range(repeats):
                        row = _base_raw_row(
                            variant_key=variant_key,
                            sequence_length=sequence_length,
                            batch_size=batch_size,
                            repeat_index=repeat_index,
                            status="ok",
                            checkpoint_path=checkpoint_path,
                            dtype=dtype,
                            seed=seed,
                            knn_neighbors=knn_neighbors,
                            coord_layout=coord_layout,
                            linker_mode=linker_mode,
                            error="",
                        )
                        timer = ComponentTimer(
                            use_cuda=device.type == "cuda"
                        )
                        hook_handles: list[
                            torch.utils.hooks.RemovableHandle
                        ] = []
                        association = None
                        normalized_association = None
                        try:
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                                _clear_device_cache(device)
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()
                                baseline_allocated_bytes = (
                                    torch.cuda.memory_allocated()
                                )
                                baseline_reserved_bytes = (
                                    torch.cuda.memory_reserved()
                                )
                            else:
                                baseline_allocated_bytes = 0
                                baseline_reserved_bytes = 0

                            timer.clear()
                            hook_handles = _register_component_timing_hooks(
                                transformer, timer
                            )

                            with torch.no_grad():
                                association = transformer(
                                    coords,
                                    features,
                                    padding_mask=None,
                                    knn_indices=knn_indices,
                                )
                                if device.type == "cuda":
                                    torch.cuda.synchronize()
                                    forward_peak_allocated_bytes = (
                                        torch.cuda.max_memory_allocated()
                                    )
                                    forward_peak_reserved_bytes = (
                                        torch.cuda.max_memory_reserved()
                                    )
                                    torch.cuda.reset_peak_memory_stats()
                                    normalize_baseline_allocated_bytes = (
                                        torch.cuda.memory_allocated()
                                    )
                                timer.start("normalize_start")
                                normalized_association = (
                                    transformer.normalize_output(
                                        association, timepoints, coords
                                    )
                                )
                                timer.stop("normalize_end")

                            encoder_layer_count = len(transformer.encoder)
                            decoder_layer_count = len(transformer.decoder)
                            full_forward_ms = timer.elapsed_ms(
                                "forward_start", "forward_end"
                            )
                            embed_ms = timer.elapsed_ms(
                                "forward_start", "encoder_0_start"
                            )
                            encoder_ms = timer.elapsed_ms(
                                "encoder_0_start",
                                f"encoder_{encoder_layer_count - 1}_end",
                            )
                            encoder_attn_ms = _sum_hook_intervals(
                                timer, "encoder_attn", encoder_layer_count
                            )
                            decoder_ms = timer.elapsed_ms(
                                "decoder_0_start",
                                f"decoder_{decoder_layer_count - 1}_end",
                            )
                            decoder_attn_ms = _sum_hook_intervals(
                                timer, "decoder_attn", decoder_layer_count
                            )
                            head_logits_ms = timer.elapsed_ms(
                                f"decoder_{decoder_layer_count - 1}_end",
                                "forward_end",
                            )
                            head_x_ms = timer.elapsed_ms(
                                "head_x_start", "head_x_end"
                            )
                            head_y_ms = timer.elapsed_ms(
                                "head_y_start", "head_y_end"
                            )
                            einsum_ms = timer.elapsed_ms(
                                "head_y_end", "forward_end"
                            )
                            embed_proj_ms = timer.elapsed_ms(
                                "embed_proj_start", "embed_proj_end"
                            )
                            normalize_ms = timer.elapsed_ms(
                                "normalize_start", "normalize_end"
                            )

                            if device.type == "cuda":
                                torch.cuda.synchronize()
                                normalize_peak_allocated_bytes = (
                                    torch.cuda.max_memory_allocated()
                                )
                                normalize_peak_reserved_bytes = (
                                    torch.cuda.max_memory_reserved()
                                )
                                peak_allocated_bytes = max(
                                    forward_peak_allocated_bytes,
                                    normalize_peak_allocated_bytes,
                                )
                                peak_reserved_bytes = max(
                                    forward_peak_reserved_bytes,
                                    normalize_peak_reserved_bytes,
                                )
                                bytes_per_mib = 1024 ** 2
                                baseline_mem_mb = (
                                    baseline_allocated_bytes / bytes_per_mib
                                )
                                peak_mem_total_mb = (
                                    peak_allocated_bytes / bytes_per_mib
                                )
                                peak_mem_reserved_mb = (
                                    peak_reserved_bytes / bytes_per_mib
                                )
                                forward_peak_mem_total_mb = (
                                    forward_peak_allocated_bytes / bytes_per_mib
                                )
                                forward_peak_mem_reserved_mb = (
                                    forward_peak_reserved_bytes / bytes_per_mib
                                )
                                normalize_peak_mem_total_mb = (
                                    normalize_peak_allocated_bytes
                                    / bytes_per_mib
                                )
                                normalize_peak_mem_reserved_mb = (
                                    normalize_peak_reserved_bytes
                                    / bytes_per_mib
                                )
                                peak_mem_mb = max(
                                    0.0, peak_mem_total_mb - baseline_mem_mb
                                )
                            else:
                                baseline_mem_mb = 0.0
                                peak_mem_total_mb = 0.0
                                peak_mem_reserved_mb = 0.0
                                forward_peak_mem_total_mb = 0.0
                                forward_peak_mem_reserved_mb = 0.0
                                normalize_peak_mem_total_mb = 0.0
                                normalize_peak_mem_reserved_mb = 0.0
                                peak_mem_mb = 0.0

                            linker_timing = _prepare_and_time_linker(
                                normalized_association,
                                timepoints,
                                coords,
                                batch_size=batch_size,
                                linker_mode=linker_mode,
                            )
                            _fill_ok_timings(
                                row,
                                full_forward_ms=full_forward_ms,
                                embed_ms=embed_ms,
                                encoder_ms=encoder_ms,
                                encoder_attn_ms=encoder_attn_ms,
                                decoder_ms=decoder_ms,
                                decoder_attn_ms=decoder_attn_ms,
                                head_logits_ms=head_logits_ms,
                                head_x_ms=head_x_ms,
                                head_y_ms=head_y_ms,
                                einsum_ms=einsum_ms,
                                embed_proj_ms=embed_proj_ms,
                                normalize_ms=normalize_ms,
                                peak_mem_mb=peak_mem_mb,
                                baseline_mem_mb=baseline_mem_mb,
                                peak_mem_total_mb=peak_mem_total_mb,
                                peak_mem_reserved_mb=peak_mem_reserved_mb,
                                forward_peak_mem_total_mb=(
                                    forward_peak_mem_total_mb
                                ),
                                forward_peak_mem_reserved_mb=(
                                    forward_peak_mem_reserved_mb
                                ),
                                normalize_peak_mem_total_mb=(
                                    normalize_peak_mem_total_mb
                                ),
                                normalize_peak_mem_reserved_mb=(
                                    normalize_peak_mem_reserved_mb
                                ),
                                dist_2d_standalone_ms=(
                                    prepared_inputs.dist_2d_standalone_ms
                                ),
                                linker_ms=linker_timing.linker_ms,
                            )
                            _apply_linker_timing(row, linker_timing)
                            closure_error_ms = float(
                                row["forward_closure_error_ms"]
                            )
                            closure_limit_ms = max(
                                0.05 * full_forward_ms, 5.0
                            )
                            if abs(closure_error_ms) > closure_limit_ms:
                                logger.warning(
                                    "forward closure error %.3f ms exceeds "
                                    "the %.3f ms threshold for variant=%s "
                                    "N=%d B=%d repeat=%d; the row is not "
                                    "corrected.",
                                    closure_error_ms,
                                    closure_limit_ms,
                                    variant_key,
                                    sequence_length,
                                    batch_size,
                                    repeat_index,
                                )
                        except Exception as repeat_error:
                            failure_status = _failure_status(repeat_error)
                            row["status"] = failure_status
                            row["error"] = (
                                f"{type(repeat_error).__name__}: "
                                f"{repeat_error}"
                            )
                            for timing_column in (
                                _TIMING_COLUMNS + _LINKER_TIMING_COLUMNS
                            ):
                                row[timing_column] = ""
                        finally:
                            _remove_hooks(hook_handles)
                            association = None
                            normalized_association = None
                            _clear_device_cache(device)

                        _write_raw_row(writer, row)
                        csv_file.flush()
                        rows.append(row)

            del model
            _clear_device_cache(device)

    _write_summary_placeholder(summary_path)
    return GridRunResult(
        out_path=out_path,
        env_json_path=env_json_path,
        summary_path=summary_path,
        rows=tuple(rows),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the full-model grid script.

    Returns:
        Parser with the SPEC 0062 flags. ``--linker-mode`` selects the
        greedy tracker or the optional motile ILP linker.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Trackastra inference timing grid with swapped "
            "attention variants (SPEC 0062), including CPU linker timing."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/k16",
        help="Checkpoint directory containing config.yaml and model.pt.",
    )
    parser.add_argument(
        "--variants",
        default=_DEFAULT_VARIANTS,
        help="Comma-separated full-model variant keys.",
    )
    parser.add_argument(
        "--Ns",
        default=_DEFAULT_NS,
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repeats per configuration."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warmup forwards per configuration.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Synthetic-input seed."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help=(
            "Dtype: 'auto' (fp16 on CUDA, fp32 on CPU), 'fp32', 'fp16', "
            "or 'bf16'."
        ),
    )
    parser.add_argument(
        "--coord-layout",
        default="uniform",
        choices=("uniform", "tiled"),
        help="Synthetic spatial layout.",
    )
    parser.add_argument(
        "--linker-mode",
        default="greedy",
        choices=("greedy", "ilp"),
        help=(
            "Linker mode: 'greedy' runs build_graph plus track_greedy; "
            "'ilp' uses motile when available."
        ),
    )
    parser.add_argument(
        "--out",
        default="results/full_model_inference_grid.csv",
        help="Raw CSV output path.",
    )
    parser.add_argument(
        "--env-json",
        default="results/full_model_inference_grid_env.json",
        help="Environment JSON output path.",
    )
    parser.add_argument(
        "--summary",
        default="results/full_model_inference_grid_summary.csv",
        help="Summary CSV path; F3 writes an explicit placeholder (F6 owns it).",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional summary PNG path (parsed only).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Force B=1, N=128, repeats=1, warmup=1, and CPU unless CUDA is "
            "explicitly requested. Never writes the primary CSV."
        ),
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help=(
            "Run the fp32 CPU equivalence checks for all four variants and "
            "exit without running the timing grid or writing CSV files."
        ),
    )
    return parser


def _apply_smoke_overrides(
    args: argparse.Namespace,
) -> tuple[list[str], list[int], list[int]]:
    """Apply the smoke-mode grid overrides and return parsed lists.

    Args:
        args: Parsed CLI namespace, modified in place for smoke mode.

    Returns:
        Tuple ``(variant_keys, sequence_lengths, batch_sizes)``.

    Raises:
        KeyError: If an unknown variant key is requested.
    """
    if args.smoke:
        args.batch_sizes = "1"
        args.Ns = str(_SMOKE_SEQ_LEN)
        args.repeats = 1
        args.warmup = 1
        if args.device != "cuda":
            args.device = "cpu"
        args.out = _SMOKE_PRIMARY_CSV
        args.env_json = _SMOKE_ENV_JSON
        args.summary = _SMOKE_SUMMARY_CSV
        variant_keys = list(FULL_MODEL_VARIANT_REGISTRY)
    else:
        variant_keys = parse_string_list(args.variants)

    unknown_variants = [
        key for key in variant_keys if key not in FULL_MODEL_VARIANT_REGISTRY
    ]
    if unknown_variants:
        raise KeyError(
            f"Unknown full-model variants {unknown_variants}; available: "
            f"{sorted(FULL_MODEL_VARIANT_REGISTRY)}"
        )

    sequence_lengths = parse_int_list(args.Ns)
    batch_sizes = parse_int_list(args.batch_sizes)
    return variant_keys, sequence_lengths, batch_sizes


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the sanity or timing-grid path.

    ``--sanity`` takes precedence over the grid and ``--smoke``: it runs the
    fp32 CPU equivalence checks and exits without writing CSV/JSON files.

    Args:
        argv: Optional argument list for tests; defaults to ``sys.argv``.

    Returns:
        Process exit code (always 0 on success).
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.sanity:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        print(f"checkpoint={checkpoint_path}")
        print("sanity: fp32 CPU, TF32 disabled, eval() + no_grad()")
        sanity_report = run_sanity_checks(
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu"),
            seq_len=_SANITY_SEQ_LEN,
            batch_size=_SANITY_BATCH_SIZE,
            seed=args.seed,
            coord_layout=args.coord_layout,
        )
        print(
            f"sanity checks passed: {len(sanity_report.comparisons)} "
            "comparisons"
        )
        return 0

    variant_keys, sequence_lengths, batch_sizes = _apply_smoke_overrides(
        args
    )

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    command_arguments = (
        sys.argv if argv is None else [sys.argv[0], *argv]
    )

    print(f"checkpoint={checkpoint_path}")
    print(f"device={device.type} dtype={dtype}")
    print(
        f"grid variants={variant_keys} Ns={sequence_lengths} "
        f"batch_sizes={batch_sizes} repeats={args.repeats} "
        f"warmup={args.warmup} linker_mode={args.linker_mode}"
    )
    print(f"raw_csv={args.out}")

    result = run_grid(
        checkpoint_path=checkpoint_path,
        variant_keys=variant_keys,
        sequence_lengths=sequence_lengths,
        batch_sizes=batch_sizes,
        seed=args.seed,
        device=device,
        dtype=dtype,
        coord_layout=args.coord_layout,
        repeats=args.repeats,
        warmup=args.warmup,
        linker_mode=args.linker_mode,
        out_path=Path(args.out),
        env_json_path=Path(args.env_json),
        summary_path=Path(args.summary),
        knn_neighbors=_KNN_NEIGHBORS,
        command=" ".join(command_arguments),
    )
    print(f"wrote {len(result.rows)} raw rows to {result.out_path}")
    print(f"environment JSON: {result.env_json_path}")
    print(f"summary placeholder: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
