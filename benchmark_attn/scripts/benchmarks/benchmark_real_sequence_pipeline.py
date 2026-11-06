#!/usr/bin/env python3
"""Real-sequence aggregate inference benchmark for SPEC 0062.

WHY this script exists
----------------------
The synthetic full-model grid packs ``N`` tokens into a fixed 10-frame
window. At ``N=32768`` that is ~3277 cells/frame, far outside every real
frame in vanvliet (max 460) and DeepCell (max 235). Candidate-graph
construction then scales with ``(N / window) ** 2`` and the linker appears to
dominate (~80--95% of the synthetic total).

This script measures the same **total detection count** as an aggregate over
real sequences instead. It uses Trackastra's actual inference path:

    load frames -> normalize -> regionprops features -> build_windows
    -> predict_windows (forward + averaging + 0.05 edge filter)
    -> build_graph (max_distance=256, max_neighbors=10)
    -> track_greedy (threshold=0.5)

Each sequence keeps its own time axis, so no artificial cross-sequence edges
are created. The target ``N`` is reached by taking deterministic real
sequence prefixes; the final frame may be truncated to hit ``N`` exactly
(timing-only, no accuracy claim).

DECISIONS
---------
* No new dataset: ``data/vanvliet`` is enough (33 sequences, 1,690 frames,
  75,828 detections). DeepCell can be selected with ``--data-root``.
* N=32768 is total detections across the aggregate, not tokens in one model
  window. This is the natural end-to-end throughput question and avoids the
  fixed-10-frame artifact.
* Sequence order is a seeded shuffle, so conditions are mixed rather than
  taking the alphabetically first sequences only.
* ``predict_windows`` already applies the paper's 0.05 association
  threshold; the sparse ``predictions["weights"]`` tuple is then passed to
  ``build_graph`` exactly as in production.
* The final partial frame truncation is explicit in the CSV and env JSON.

Usage (from ``benchmark_attn/``)::

    python scripts/benchmarks/benchmark_real_sequence_pipeline.py \
        --checkpoint ../checkpoints/k16 \
        --data-root ../data/vanvliet \
        --target-detections 32768 \
        --device cuda --dtype fp16 --batch-size 4 \
        --out results/real_sequence_pipeline.csv

A small local smoke test (CPU, ~100 detections)::

    python scripts/benchmarks/benchmark_real_sequence_pipeline.py \
        --checkpoint ../checkpoints/k16 \
        --data-root ../data/vanvliet \
        --target-detections 100 --max-sequences 1 \
        --device cpu --dtype fp32 --batch-size 1 \
        --out /tmp/real_sequence_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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

from trackastra.data import (  # noqa: E402
    WRFeatures,
    build_windows,
    get_features,
    load_tiff_timeseries,
)
from trackastra.data.utils import load_tracklet_links  # noqa: E402
from trackastra.model.model_api import Trackastra  # noqa: E402
from trackastra.model.predict import predict_windows  # noqa: E402
from trackastra.tracking import build_graph, track_greedy  # noqa: E402
from trackastra.utils import normalize  # noqa: E402

logger = logging.getLogger(__name__)
_DTYPE_BY_NAME = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
_RAW_COLUMNS = (
    "condition",
    "sequence",
    "take_frames",
    "detections",
    "partial_last_frame",
    "load_ms",
    "normalize_ms",
    "feature_ms",
    "windows_ms",
    "predict_ms",
    "build_graph_ms",
    "greedy_ms",
    "compute_ms",
    "total_ms",
    "candidate_edges",
    "weight_edges",
)
_SUMMARY_COLUMNS = (
    "target_detections",
    "total_detections",
    "total_frames",
    "num_sequences",
    "mean_cells_per_frame",
    "median_cells_per_frame",
    "max_cells_per_frame",
    "load_ms",
    "normalize_ms",
    "feature_ms",
    "windows_ms",
    "predict_ms",
    "build_graph_ms",
    "greedy_ms",
    "compute_ms",
    "total_ms",
    "feature_share",
    "predict_share",
    "build_graph_share",
    "linker_share",
    "throughput_detections_per_s",
    "throughput_frames_per_s",
)


@dataclass(frozen=True)
class SequenceInfo:
    """One discoverable Trackastra sequence."""

    condition: str
    name: str
    path: Path
    frame_count: int
    frame_detections: np.ndarray


@dataclass(frozen=True)
class SequencePlan:
    """A real-sequence prefix selected to contribute to the target N."""

    info: SequenceInfo
    take_frames: int
    partial_last_frame: int | None
    detections: int


def _frame_counts_from_track_df(
    track_df, frame_count: int
) -> np.ndarray:
    """Count active tracks per frame from a CTC ``man_track.txt`` table."""
    counts = np.zeros(frame_count, dtype=np.int64)
    for row in track_df.itertuples(index=False):
        start = max(0, int(row.t1))
        end = min(frame_count - 1, int(row.t2))
        if end >= start:
            counts[start : end + 1] += 1
    return counts


def discover_sequences(
    data_root: Path,
    conditions: list[str] | None = None,
    include_bad: bool = False,
) -> list[SequenceInfo]:
    """Discover image/TRA sequence folders and their per-frame counts."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    selected_conditions = set(conditions) if conditions else None
    infos: list[SequenceInfo] = []
    for condition_dir in sorted(data_root.iterdir()):
        if not condition_dir.is_dir():
            continue
        if selected_conditions and condition_dir.name not in selected_conditions:
            continue
        for sequence_dir in sorted(condition_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue
            if not include_bad and sequence_dir.name.endswith("_BAD"):
                logger.info("Skipping flagged sequence %s", sequence_dir.name)
                continue
            image_dir = sequence_dir / "img"
            tra_dir = sequence_dir / "TRA"
            if not image_dir.is_dir() or not tra_dir.is_dir():
                continue
            mask_files = sorted(tra_dir.glob("man_track*.tif"))
            if not mask_files:
                continue
            try:
                track_df = load_tracklet_links(tra_dir)
            except FileNotFoundError:
                logger.warning("No track table in %s; skipping", tra_dir)
                continue
            frame_count = min(len(mask_files), int(track_df.t2.max()) + 1)
            if frame_count <= 0:
                continue
            counts = _frame_counts_from_track_df(track_df, frame_count)
            infos.append(
                SequenceInfo(
                    condition=condition_dir.name,
                    name=sequence_dir.name,
                    path=sequence_dir,
                    frame_count=frame_count,
                    frame_detections=counts,
                )
            )
    if not infos:
        raise ValueError(f"No usable sequences found under {data_root}")
    return infos


def plan_sequence_prefix(
    frame_detections: np.ndarray, remaining: int
) -> tuple[int, int | None, int]:
    """Take the longest prefix that does not exceed ``remaining``.

    Returns ``(take_frames, partial_last_frame, detections)``. If the next
    full frame would overshoot, the last included frame is truncated to
    ``partial_last_frame`` detections so the aggregate hits ``remaining``
    exactly. A partial count is only a timing device; it is recorded in the
    CSV and env JSON.

    DECISION: truncating the last frame is preferable to overshooting the
    requested N or silently dropping a whole sequence. The effect is a
    handful of detections in one frame and is timing-only.
    """
    if remaining <= 0:
        return 0, None, 0
    cumulative = 0
    for index, count in enumerate(frame_detections):
        count = int(count)
        if cumulative + count <= remaining:
            cumulative += count
            if cumulative == remaining:
                return index + 1, None, cumulative
        else:
            partial = remaining - cumulative
            if partial <= 0:
                return index, None, cumulative
            return index + 1, partial, remaining
    if cumulative == 0:
        return 0, None, 0
    return len(frame_detections), None, cumulative


def select_sequence_plans(
    infos: list[SequenceInfo],
    target_detections: int,
    seed: int = 42,
    max_sequences: int | None = None,
) -> list[SequencePlan]:
    """Select seeded real-sequence prefixes whose detections sum to target."""
    if target_detections <= 0:
        raise ValueError("target_detections must be positive")
    shuffled = list(infos)
    random.Random(seed).shuffle(shuffled)

    plans: list[SequencePlan] = []
    remaining = target_detections
    for info in shuffled:
        if remaining <= 0:
            break
        if max_sequences is not None and len(plans) >= max_sequences:
            break
        take_frames, partial, detections = plan_sequence_prefix(
            info.frame_detections, remaining
        )
        if take_frames <= 0 or detections <= 0:
            continue
        plans.append(
            SequencePlan(
                info=info,
                take_frames=take_frames,
                partial_last_frame=partial,
                detections=detections,
            )
        )
        remaining -= detections

    if remaining > 0:
        raise ValueError(
            f"Could not reach target_detections={target_detections}; "
            f"short by {remaining} detections with "
            f"{len(plans)} selected sequence prefixes."
        )
    return plans


def _subset_wrfeatures(features: WRFeatures, count: int) -> WRFeatures:
    """Return the first ``count`` detections of one frame's features."""
    return WRFeatures(
        coords=features.coords[:count],
        labels=features.labels[:count],
        timepoints=features.timepoints[:count],
        features=OrderedDict(
            (key, value[:count])
            for key, value in features.features.items()
        ),
    )


def _cast_windows_to_dtype(
    windows: list[dict], dtype: torch.dtype
) -> None:
    """Cast tensor inputs in-place to the model dtype."""
    for window in windows:
        for key in ("coords", "features", "pretrained_feats", "patches_cnn"):
            value = window.get(key)
            if torch.is_tensor(value):
                window[key] = value.to(dtype=dtype)


def run_sequence(
    plan: SequencePlan,
    model: Trackastra,
    *,
    dtype: torch.dtype,
    batch_size: int,
    n_workers: int,
    edge_threshold: float,
    max_distance: int,
    max_neighbors: int,
    delta_t: int,
    greedy_threshold: float,
    window: int | None = None,
    normalize_imgs: bool = True,
    use_cnn: bool = False,
) -> dict[str, object]:
    """Run and time Trackastra's real inference pipeline on one prefix."""
    image_dir = plan.info.path / "img"
    tra_dir = plan.info.path / "TRA"

    start = time.perf_counter()
    imgs = load_tiff_timeseries(
        image_dir, dtype=np.float32, end_frame=plan.take_frames
    )
    masks = load_tiff_timeseries(
        tra_dir, dtype=np.int32, end_frame=plan.take_frames
    )
    load_ms = (time.perf_counter() - start) * 1000.0

    if len(imgs) != plan.take_frames or len(masks) != plan.take_frames:
        raise RuntimeError(
            f"{plan.info.name}: loaded {len(imgs)} images / {len(masks)} "
            f"masks for take_frames={plan.take_frames}"
        )

    start = time.perf_counter()
    if normalize_imgs:
        imgs = normalize(imgs)
    normalize_ms = (time.perf_counter() - start) * 1000.0

    feature_mode = model.train_args.get("features", "wrfeat")
    if feature_mode not in ("wrfeat", "none"):
        raise ValueError(
            "This aggregate runner supports the regionprops path only; "
            f"checkpoint train_args['features']={feature_mode!r}"
        )

    start = time.perf_counter()
    features = list(
        get_features(
            detections=masks,
            imgs=imgs,
            features=feature_mode,
            ndim=2,
            n_workers=n_workers,
            progbar_class=tqdm,
        )
    )
    if plan.partial_last_frame is not None:
        features[-1] = _subset_wrfeatures(
            features[-1], plan.partial_last_frame
        )
    feature_ms = (time.perf_counter() - start) * 1000.0

    actual_detections = sum(len(frame_features) for frame_features in features)
    if actual_detections != plan.detections:
        raise RuntimeError(
            f"{plan.info.name}: planned {plan.detections} detections but "
            f"features contain {actual_detections}; the TRA masks and "
            "man_track.txt disagree."
        )

    transformer = model.transformer
    model_window = int(window or transformer.config["window"])
    start = time.perf_counter()
    windows = build_windows(
        features,
        window_size=model_window,
        as_torch=True,
        imgs=imgs,
        use_cnn=use_cnn,
    )
    _cast_windows_to_dtype(windows, dtype)
    windows_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    predictions = predict_windows(
        windows=windows,
        features=features,
        model=transformer,
        intra_window_weight=0,
        delta_t=1,
        edge_threshold=edge_threshold,
        spatial_dim=masks.ndim - 1,
        batch_size=batch_size,
        progbar_class=tqdm,
    )
    predict_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    candidate_graph = build_graph(
        nodes=predictions["nodes"],
        weights=predictions["weights"],
        use_distance=False,
        max_distance=max_distance,
        max_neighbors=max_neighbors,
        delta_t=delta_t,
    )
    build_graph_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    track_greedy(
        candidate_graph,
        allow_divisions=True,
        threshold=greedy_threshold,
    )
    greedy_ms = (time.perf_counter() - start) * 1000.0

    compute_ms = (
        normalize_ms
        + feature_ms
        + windows_ms
        + predict_ms
        + build_graph_ms
        + greedy_ms
    )
    return {
        "condition": plan.info.condition,
        "sequence": plan.info.name,
        "take_frames": plan.take_frames,
        "detections": plan.detections,
        "partial_last_frame": (
            "" if plan.partial_last_frame is None
            else plan.partial_last_frame
        ),
        "load_ms": load_ms,
        "normalize_ms": normalize_ms,
        "feature_ms": feature_ms,
        "windows_ms": windows_ms,
        "predict_ms": predict_ms,
        "build_graph_ms": build_graph_ms,
        "greedy_ms": greedy_ms,
        "compute_ms": compute_ms,
        "total_ms": load_ms + compute_ms,
        "candidate_edges": candidate_graph.number_of_edges(),
        "weight_edges": len(predictions["weights"]),
    }


def summarize_rows(
    rows: list[dict[str, object]],
    target_detections: int,
    plans: list[SequencePlan],
) -> dict[str, object]:
    """Aggregate per-sequence rows into one throughput summary."""
    def total(key: str) -> float:
        return float(sum(float(row[key]) for row in rows))

    per_frame_counts: list[int] = []
    for plan in plans:
        counts = plan.info.frame_detections[: plan.take_frames].copy()
        if plan.partial_last_frame is not None:
            counts[-1] = plan.partial_last_frame
        per_frame_counts.extend(int(count) for count in counts)

    detections = [float(row["detections"]) for row in rows]
    frames = [int(row["take_frames"]) for row in rows]
    compute_ms = total("compute_ms")
    build_graph_ms = total("build_graph_ms")
    greedy_ms = total("greedy_ms")
    summary = {
        "target_detections": target_detections,
        "total_detections": int(sum(detections)),
        "total_frames": int(sum(frames)),
        "num_sequences": len(rows),
        "mean_cells_per_frame": float(np.mean(per_frame_counts)),
        "median_cells_per_frame": float(np.median(per_frame_counts)),
        "max_cells_per_frame": float(np.max(per_frame_counts)),
        "load_ms": total("load_ms"),
        "normalize_ms": total("normalize_ms"),
        "feature_ms": total("feature_ms"),
        "windows_ms": total("windows_ms"),
        "predict_ms": total("predict_ms"),
        "build_graph_ms": build_graph_ms,
        "greedy_ms": greedy_ms,
        "compute_ms": compute_ms,
        "total_ms": total("total_ms"),
        "feature_share": total("feature_ms") / compute_ms,
        "predict_share": total("predict_ms") / compute_ms,
        "build_graph_share": build_graph_ms / compute_ms,
        "linker_share": (build_graph_ms + greedy_ms) / compute_ms,
        "throughput_detections_per_s": (
            sum(detections) / (compute_ms / 1000.0)
        ),
        "throughput_frames_per_s": (
            sum(frames) / (compute_ms / 1000.0)
        ),
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], columns: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def write_env_json(
    path: Path,
    *,
    args: argparse.Namespace,
    plans: list[SequencePlan],
    summary: dict[str, object],
    dtype: torch.dtype,
) -> None:
    env = {
        "command": " ".join(sys.argv),
        "script": str(Path(__file__).relative_to(_PROJECT_ROOT)),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "target_detections": args.target_detections,
        "seed": args.seed,
        "device": args.device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "n_workers": args.n_workers,
        "edge_threshold": args.edge_threshold,
        "max_distance": args.max_distance,
        "max_neighbors": args.max_neighbors,
        "delta_t": args.delta_t,
        "greedy_threshold": args.greedy_threshold,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "plans": [
            {
                "condition": plan.info.condition,
                "sequence": plan.info.name,
                "take_frames": plan.take_frames,
                "partial_last_frame": plan.partial_last_frame,
                "detections": plan.detections,
            }
            for plan in plans
        ],
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(env, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time Trackastra's real inference pipeline over real sequence "
            "prefixes until the aggregate detection count reaches N."
        )
    )
    parser.add_argument("--checkpoint", default="../checkpoints/k16")
    parser.add_argument("--data-root", default="../data/vanvliet")
    parser.add_argument("--target-detections", type=int, default=32768)
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated condition folders (default: all).",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=sorted(_DTYPE_BY_NAME))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    parser.add_argument("--max-distance", type=int, default=256)
    parser.add_argument("--max-neighbors", type=int, default=10)
    parser.add_argument("--delta-t", type=int, default=1)
    parser.add_argument("--greedy-threshold", type=float, default=0.5)
    parser.add_argument("--include-bad", action="store_true")
    parser.add_argument(
        "--out", default="results/real_sequence_pipeline.csv"
    )
    parser.add_argument(
        "--env-json", default="results/real_sequence_pipeline_env.json"
    )
    parser.add_argument(
        "--summary", default="results/real_sequence_pipeline_summary.csv"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    dtype = _DTYPE_BY_NAME[args.dtype]
    conditions = (
        [item.strip() for item in args.conditions.split(",") if item.strip()]
        if args.conditions
        else None
    )

    infos = discover_sequences(
        Path(args.data_root),
        conditions=conditions,
        include_bad=args.include_bad,
    )
    plans = select_sequence_plans(
        infos,
        target_detections=args.target_detections,
        seed=args.seed,
        max_sequences=args.max_sequences,
    )
    logger.info(
        "Selected %d sequences for %d detections",
        len(plans),
        args.target_detections,
    )

    model = Trackastra.from_folder(Path(args.checkpoint), device=args.device)
    transformer = model.transformer
    transformer.eval()
    transformer.to(dtype=dtype)

    rows: list[dict[str, object]] = []
    for plan in plans:
        logger.info(
            "Running %s/%s: %d frames, %d detections",
            plan.info.condition,
            plan.info.name,
            plan.take_frames,
            plan.detections,
        )
        rows.append(
            run_sequence(
                plan,
                model,
                dtype=dtype,
                batch_size=args.batch_size,
                n_workers=args.n_workers,
                edge_threshold=args.edge_threshold,
                max_distance=args.max_distance,
                max_neighbors=args.max_neighbors,
                delta_t=args.delta_t,
                greedy_threshold=args.greedy_threshold,
                window=args.window,
                use_cnn=bool(transformer.config.get("use_cnn", False)),
            )
        )

    summary = summarize_rows(rows, args.target_detections, plans)
    write_csv(Path(args.out), rows, _RAW_COLUMNS)
    write_csv(Path(args.summary), [summary], _SUMMARY_COLUMNS)
    write_env_json(
        Path(args.env_json),
        args=args,
        plans=plans,
        summary=summary,
        dtype=dtype,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
