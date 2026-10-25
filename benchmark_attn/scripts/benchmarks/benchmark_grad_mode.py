"""Grad-enabled vs no-grad forward ablation for the attention benchmark.

Purpose
-------
The main sweep times forward passes inside :func:`src.bench.timing.measure`,
which does not disable autograd.  When the attention modules' parameters have
``requires_grad=True`` (the default), the forward pass also builds the autograd
graph and keeps intermediate activations alive.  This script measures how much
extra time and peak memory that graph costs by running the *same* closure twice
per representative configuration:

1. first wrapped in ``torch.no_grad()`` (inference-style forward),
2. then with autograd enabled (training-style forward, no backward).

Why one shape per method
------------------------
A full grid is not needed to answer "does autograd change the cost of the
forward pass?".  One representative shape per method is enough to expose the
activation-memory overhead, keeps the run short (30 min Slurm limit), and
avoids an ablation grid that would mostly repeat the same ratio.  The shapes
are chosen at the largest sequence length each method is expected to survive
on an H100:

- ``dense_flash`` / ``flex_attn``: N=32768, no KNN (K=0)
- ``dense_masked`` / ``sparse_mask``: N=4096, KNN K=16 for ``sparse_mask``
- ``sparse_gather_matmul``: N=32768, KNN K=16

The closure is built once per configuration by ``build_closure`` from
``benchmark_sweep.py``, so this script inherits contract B: one shared spatial
graph per batch, chunked exact KNN, and no unused ``dist_2d`` allocation.
Inputs, layers, and seed match the production sweep settings; the dtype is
fp16 on CUDA and fp32 on CPU, matching ``benchmark_sweep.py`` so CPU-only
smoke runs do not fail on fp16 kernels that require a GPU.

Why no-grad first
-----------------
The no-grad run is cheap and rarely OOMs, so recording it first guarantees a
baseline result even when the grad-enabled run exhausts GPU memory.  A failed
grad run is recorded as ``oom``/``err``, the CUDA cache is cleared, and the
next configuration continues; the script never aborts on a single failed
configuration.

How to run
----------
From ``benchmark_attn/``::

    .venv/bin/python scripts/benchmarks/benchmark_grad_mode.py \\
        --batch-size 64 --seed 42 --warmup 5 --rep 300 --repeats 3 \\
        --device cuda --out results/grad_mode_ablation.csv

Local A500 validation (small shapes, short timing)::

    .venv/bin/python scripts/benchmarks/benchmark_grad_mode.py \\
        --smoke --out /tmp/grad_mode_smoke.csv

The script writes ``--out`` with the header::

    method,N,K,batch,no_grad_time_ms,no_grad_time_std_ms,no_grad_mem_mb,
    no_grad_status,grad_time_ms,grad_time_std_ms,grad_mem_mb,grad_status

Rows with a failed measurement leave the corresponding numeric cells empty and
record ``oom`` or ``err`` in the status column.  A compact table and a one-line
grad/no-grad time and memory ratio summary (over rows where both modes are
``ok``) are printed to stdout.
"""

import sys
from pathlib import Path

BENCHMARK_ATTN_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_ATTN_ROOT))
sys.path.insert(0, str(BENCHMARKS_DIR))

import argparse
import gc
import statistics
from dataclasses import dataclass

import torch

from src.bench import measure, write_results_csv
from benchmark_sweep import build_closure


CSV_HEADER = [
    "method",
    "N",
    "K",
    "batch",
    "no_grad_time_ms",
    "no_grad_time_std_ms",
    "no_grad_mem_mb",
    "no_grad_status",
    "grad_time_ms",
    "grad_time_std_ms",
    "grad_mem_mb",
    "grad_status",
]


@dataclass(frozen=True)
class AblationConfig:
    """One representative method/shape measured in both autograd modes.

    Attributes:
        method_key: Registry key understood by ``build_closure``.
        knn_neighbors: KNN neighbourhood size, or 0 for methods that do not
            consume KNN indices.
        seq_len: Sequence length N for this method's representative shape.
    """

    method_key: str
    knn_neighbors: int
    seq_len: int


@dataclass(frozen=True)
class MeasurementResult:
    """Outcome of one ``measure`` call, successful or failed.

    Attributes:
        time_s: Mean wall time in seconds, or -1.0 on failure.
        time_std_s: Standard deviation across timing repeats in seconds, or
            -1.0 on failure.
        memory_mb: Incremental peak device memory in MiB, or -1.0 on failure.
        status: ``"ok"``, ``"oom"``, or ``"err"``.
        detail: Human-readable error detail for stdout; empty when ``ok``.
    """

    time_s: float
    time_std_s: float
    memory_mb: float
    status: str
    detail: str


REPRESENTATIVE_CONFIGS = (
    AblationConfig(method_key="dense_flash", knn_neighbors=0, seq_len=32768),
    AblationConfig(method_key="dense_masked", knn_neighbors=0, seq_len=4096),
    AblationConfig(method_key="sparse_mask", knn_neighbors=16, seq_len=4096),
    AblationConfig(method_key="sparse_gather_matmul", knn_neighbors=16, seq_len=32768),
    AblationConfig(method_key="flex_attn", knn_neighbors=0, seq_len=32768),
)

# ---------------------------------------------------------------------------
# Fixed ablation settings (SPEC 0051 contract-B harness).
# ---------------------------------------------------------------------------
LAYER_COUNT = 1
D_MODEL = 320
N_HEAD = 8
COORD_DIM = 2
MODE = "none"
DIST_MODE = "v1"
WITH_KNN = False
DIST_CACHE = "on"


def _recover_after_failure() -> None:
    """Release failed-allocation state so the next configuration starts clean.

    A CUDA OOM can leave the caching allocator holding fragmented or reserved
    blocks.  Clearing them after every failed measurement prevents one failed
    grad run from cascading into the following configurations (same policy as
    ``src.bench.timing.try_bench``).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _measure_mode(closure, use_no_grad, warmup, min_run_time, repeats):
    """Measure ``closure`` once with or without autograd enabled.

    Args:
        closure: Zero-argument callable running the chained attention layers.
        use_no_grad: When ``True``, wrap the call in ``torch.no_grad()``;
            otherwise call it normally so autograd builds the forward graph.
        warmup: Number of untimed warmup iterations passed to ``measure``.
        min_run_time: Minimum autorange duration in seconds.
        repeats: Number of timing repetitions passed to ``measure``.

    Returns:
        A :class:`MeasurementResult`; failures are captured rather than
        raised so the ablation continues with the next configuration.
    """
    if use_no_grad:
        def closure_call():
            """Run the closure under ``torch.no_grad()`` for inference timing."""
            with torch.no_grad():
                return closure()
    else:
        closure_call = closure

    try:
        time_s, time_std_s, memory_mb = measure(
            closure_call,
            warmup=warmup,
            min_run_time=min_run_time,
            repeats=repeats,
        )
        return MeasurementResult(time_s, time_std_s, memory_mb, "ok", "")
    except RuntimeError as error:
        _recover_after_failure()
        if "out of memory" in str(error).lower():
            return MeasurementResult(-1.0, -1.0, -1.0, "oom", str(error))
        return MeasurementResult(
            -1.0, -1.0, -1.0, "err", f"{type(error).__name__}: {error}"
        )
    except Exception as error:  # noqa: BLE001 — recorded as a soft outcome
        _recover_after_failure()
        return MeasurementResult(
            -1.0, -1.0, -1.0, "err", f"{type(error).__name__}: {error}"
        )


def _format_csv_row(config, batch_size, no_grad_result, grad_result):
    """Convert one configuration's two measurements into a CSV row.

    Args:
        config: The :class:`AblationConfig` that was measured.
        batch_size: Batch dimension B used for both measurements.
        no_grad_result: :class:`MeasurementResult` for the no-grad run.
        grad_result: :class:`MeasurementResult` for the grad-enabled run.

    Returns:
        A row matching :data:`CSV_HEADER`; numeric cells are empty when the
        corresponding status is not ``"ok"``.
    """
    row = [config.method_key, config.seq_len, config.knn_neighbors, batch_size]
    for result in (no_grad_result, grad_result):
        if result.status == "ok":
            row.extend([
                f"{result.time_s * 1000:.3f}",
                f"{result.time_std_s * 1000:.3f}",
                f"{result.memory_mb:.1f}",
            ])
        else:
            row.extend(["", "", ""])
        row.append(result.status)
    return row


def _print_table(rows):
    """Print a compact per-configuration table of the measured results.

    Args:
        rows: CSV-style rows produced by :func:`_format_csv_row`.
    """
    print(
        f"\n{'method':<22s} {'N':>6s} {'K':>3s} {'B':>3s} "
        f"{'no_grad ms':>11s} {'no_grad MB':>10s}  "
        f"{'grad ms':>9s} {'grad MB':>9s}  "
        f"{'no_grad':>7s} {'grad':>7s}",
        flush=True,
    )
    for row in rows:
        (method_key, seq_len, knn_neighbors, batch_size,
         no_grad_time_ms, no_grad_std_ms, no_grad_mem_mb, no_grad_status,
         grad_time_ms, grad_std_ms, grad_mem_mb, grad_status) = row
        print(
            f"{method_key:<22s} {seq_len:>6d} {knn_neighbors:>3d} {batch_size:>3d} "
            f"{(no_grad_time_ms or '-'):>11s} {(no_grad_mem_mb or '-'):>10s}  "
            f"{(grad_time_ms or '-'):>9s} {(grad_mem_mb or '-'):>9s}  "
            f"{no_grad_status:>7s} {grad_status:>7s}",
            flush=True,
        )


def _print_ratio_summary(no_grad_results, grad_results):
    """Print one line of mean grad/no-grad time and memory ratios.

    Only pairs where both measurements succeeded are included, so the ratios
    are not distorted by OOM rows.  Ratios are averaged per configuration
    (mean of per-config ratios), not computed from summed times/memory.

    Args:
        no_grad_results: List of :class:`MeasurementResult` in config order.
        grad_results: List of :class:`MeasurementResult` in config order.
    """
    time_ratios = []
    memory_ratios = []
    for no_grad_result, grad_result in zip(no_grad_results, grad_results):
        if no_grad_result.status != "ok" or grad_result.status != "ok":
            continue
        if no_grad_result.time_s > 0:
            time_ratios.append(grad_result.time_s / no_grad_result.time_s)
        if no_grad_result.memory_mb > 0:
            memory_ratios.append(grad_result.memory_mb / no_grad_result.memory_mb)

    if not time_ratios:
        print("\n[Summary] no configuration completed both modes; ratios n/a.")
        return

    time_text = (
        f"time={statistics.fmean(time_ratios):.3f}x"
        if time_ratios else "time=n/a"
    )
    memory_text = (
        f"memory={statistics.fmean(memory_ratios):.3f}x"
        if memory_ratios else "memory=n/a"
    )
    print(
        f"\n[Summary] grad/no-grad ratios over {len(time_ratios)} ok config(s): "
        f"{time_text}, {memory_text} (mean of per-config ratios)."
    )


def _resolve_device(device_argument):
    """Resolve the CLI device string into a ``torch.device``.

    Args:
        device_argument: ``"auto"``, ``"cuda"``, ``"cpu"``, or a concrete
            PyTorch device string.

    Returns:
        The selected ``torch.device`` (CUDA when available for ``"auto"``).
    """
    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_argument)


def run_ablation(configs, batch_size, device, dtype, seed, warmup,
                 min_run_time, repeats):
    """Run every config under no-grad first, then grad-enabled.

    Each configuration builds its closure once with ``build_closure`` and
    measures it in both modes.  Build failures, OOMs, and kernel errors are
    recorded per row; they do not terminate the ablation.

    Args:
        configs: Sequence of :class:`AblationConfig` to measure.
        batch_size: Batch dimension B for every configuration.
        device: Torch device for inputs and layers.
        dtype: Torch dtype for inputs and layers.
        seed: Random seed passed to ``build_closure``.
        warmup: Warmup iterations per measurement.
        min_run_time: Minimum autorange duration in seconds.
        repeats: Timing repetitions per measurement.

    Returns:
        Tuple ``(rows, no_grad_results, grad_results)`` where ``rows`` are
        CSV-style rows and the result lists follow ``configs`` order.
    """
    rows = []
    no_grad_results = []
    grad_results = []

    for config in configs:
        print(
            f"\n[Config] method={config.method_key} N={config.seq_len} "
            f"K={config.knn_neighbors} B={batch_size}",
            flush=True,
        )
        try:
            closure = build_closure(
                method_key=config.method_key,
                layer_count=LAYER_COUNT,
                seq_len=config.seq_len,
                knn_neighbors=config.knn_neighbors,
                batch_size=batch_size,
                d_model=D_MODEL,
                n_head=N_HEAD,
                coord_dim=COORD_DIM,
                mode=MODE,
                dist_mode=DIST_MODE,
                device=device,
                dtype=dtype,
                seed=seed,
                with_knn=WITH_KNN,
                dist_cache=DIST_CACHE,
            )
        except RuntimeError as error:
            _recover_after_failure()
            status = "oom" if "out of memory" in str(error).lower() else "err"
            detail = f"{type(error).__name__}: {error}"
            print(f"  [build {status}] {detail}", flush=True)
            no_grad_result = MeasurementResult(-1.0, -1.0, -1.0, status, detail)
            grad_result = MeasurementResult(-1.0, -1.0, -1.0, status, detail)
            no_grad_results.append(no_grad_result)
            grad_results.append(grad_result)
            rows.append(
                _format_csv_row(config, batch_size, no_grad_result, grad_result)
            )
            continue
        except Exception as error:  # noqa: BLE001 — recorded as a soft outcome
            _recover_after_failure()
            detail = f"{type(error).__name__}: {error}"
            print(f"  [build err] {detail}", flush=True)
            no_grad_result = MeasurementResult(-1.0, -1.0, -1.0, "err", detail)
            grad_result = MeasurementResult(-1.0, -1.0, -1.0, "err", detail)
            no_grad_results.append(no_grad_result)
            grad_results.append(grad_result)
            rows.append(
                _format_csv_row(config, batch_size, no_grad_result, grad_result)
            )
            continue

        no_grad_result = _measure_mode(
            closure,
            use_no_grad=True,
            warmup=warmup,
            min_run_time=min_run_time,
            repeats=repeats,
        )
        no_grad_results.append(no_grad_result)
        if no_grad_result.status != "ok":
            print(
                f"  [no_grad {no_grad_result.status}] {no_grad_result.detail}",
                flush=True,
            )

        grad_result = _measure_mode(
            closure,
            use_no_grad=False,
            warmup=warmup,
            min_run_time=min_run_time,
            repeats=repeats,
        )
        grad_results.append(grad_result)
        if grad_result.status != "ok":
            print(
                f"  [grad {grad_result.status}] {grad_result.detail}",
                flush=True,
            )

        rows.append(
            _format_csv_row(config, batch_size, no_grad_result, grad_result)
        )
        del closure
        _recover_after_failure()

    return rows, no_grad_results, grad_results


def build_argument_parser():
    """Build the CLI parser for the grad-mode ablation.

    Returns:
        A configured ``argparse.ArgumentParser`` whose defaults reproduce the
        main ablation settings (``measure`` warmup=5, min_run_time=0.3 s,
        repeats=3) unless ``--smoke`` overrides them.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare no-grad and grad-enabled forward time/memory for one "
            "representative shape per attention method (SPEC 0051 H5)."
        ),
    )
    parser.add_argument(
        "--out", default="results/grad_mode_ablation.csv",
        help="Output CSV path (default: results/grad_mode_ablation.csv).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch dimension B for every configuration (default: 64).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed passed to build_closure (default: 42).",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations per measurement (default: 5).",
    )
    parser.add_argument(
        "--rep", type=int, default=300,
        help="Minimum autorange run time in milliseconds (default: 300).",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timing repetitions per measurement (default: 3).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Local validation mode: override every config to B=4, N=256, "
            "warmup=1, rep=5 ms, repeats=1."
        ),
    )
    return parser


def main():
    """Parse CLI arguments, run the ablation, write the CSV, print summaries."""
    parser = build_argument_parser()
    args = parser.parse_args()

    device = _resolve_device(args.device)
    # Match the main sweep: fp16 on CUDA; fp32 on CPU so CPU-only smoke runs
    # do not fail on fp16 kernels that require a GPU.
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if args.smoke:
        batch_size = 4
        configs = tuple(
            AblationConfig(config.method_key, config.knn_neighbors, 256)
            for config in REPRESENTATIVE_CONFIGS
        )
        warmup = 1
        min_run_time = 5 / 1000.0
        repeats = 1
        print("[Smoke] B=4, N=256, warmup=1, rep=5 ms, repeats=1")
    else:
        batch_size = args.batch_size
        configs = REPRESENTATIVE_CONFIGS
        warmup = args.warmup
        min_run_time = args.rep / 1000.0
        repeats = args.repeats

    print(f"Device: {device}  dtype: {dtype}")
    print(
        f"Fixed: L={LAYER_COUNT} d={D_MODEL} nhead={N_HEAD} coord_dim={COORD_DIM} "
        f"mode={MODE} dist_mode={DIST_MODE} with_knn={WITH_KNN} "
        f"dist_cache={DIST_CACHE} seed={args.seed}"
    )
    print(
        f"Timing: warmup={warmup} min_run_time={min_run_time:.3f}s "
        f"repeats={repeats}"
    )

    rows, no_grad_results, grad_results = run_ablation(
        configs=configs,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        seed=args.seed,
        warmup=warmup,
        min_run_time=min_run_time,
        repeats=repeats,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_csv(rows, out_path, CSV_HEADER)
    print(f"\nWrote {len(rows)} rows -> {out_path}")

    _print_table(rows)
    _print_ratio_summary(no_grad_results, grad_results)


if __name__ == "__main__":
    main()
