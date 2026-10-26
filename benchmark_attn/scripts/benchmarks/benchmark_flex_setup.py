"""Measure FlexAttention setup cost separately from steady-state forward time.

Why setup is separated
----------------------
The production sweep warms up every method before timing.  For FlexAttention
that warmup silently absorbs two expensive one-time steps:

1. ``module.prepare(coords)`` builds the shared block mask with
   ``create_block_mask`` (``B=1``, ``H=1``) and creates the
   ``torch.compile(flex_attention, dynamic=False)`` wrapper.
2. The first forward call compiles the wrapped ``flex_attention`` for this
   exact ``(batch, seq_len)`` shape and executes it once.

Because ``src.bench.timing.measure`` discards its warmup iterations, the
existing CSV reports only kernel time and has no setup number.  This script
measures setup and steady state per ``(B, N)`` so the report can state both
without changing the meaning of the existing steady-state columns.

What ``prepare_ms`` and ``first_call_ms`` include
-------------------------------------------------
``prepare_ms`` times ``module.prepare(coords)``: block-mask construction plus
creation of the compiled callable.  ``first_call_ms`` times one forward call on
a freshly prepared module, so it contains the Inductor compilation, the
block-mask reuse, and exactly one forward execution.  ``steady_ms`` then uses
the production ``measure`` helper on the same module, so its warmup absorbs
nothing that is still pending.

``compile_overhead_ms`` is the signed difference
``first_call_ms - steady_ms``.  Because the first call also performs one
execution, the signed value isolates compilation cost: a first call that is
slower than steady state yields the compile overhead; a first call that is not
slower yields a negative value.  ``total_setup_ms`` is
``prepare_ms + max(0.0, compile_overhead_ms)``, i.e. setup excluding the one
execution already represented in the steady-state time.

Fresh Inductor cache caveat
---------------------------
``torch._dynamo.reset()`` clears only the *in-process* Dynamo cache.  It does
not delete the on-disk Inductor cache (``TORCHINDUCTOR_CACHE_DIR``).  A truly
cold compile therefore requires a fresh process and a fresh cache directory;
the Slurm runner creates one per batch size.  Within a single process this
script resets Dynamo before every shape so no compiled graph from a previous
shape is reused, but a repeated ``(B, N)`` in the same process could still hit
the on-disk cache.

How to run
----------
Local smoke on a GPU (small shapes, short timing, writes
``results/flex_setup_smoke.csv``)::

    cd benchmark_attn
    .venv/bin/python scripts/benchmarks/benchmark_flex_setup.py \\
        --smoke --device cuda

Full run (normally through ``slurm/run_flex_setup.slurm``)::

    .venv/bin/python scripts/benchmarks/benchmark_flex_setup.py \\
        --batch-sizes 64 --Ns 128,256,512,1024,2048,4096,8192,16384,32768 \\
        --warmup 5 --rep 300 --repeats 3 --device cuda --no-grad \\
        --out results/flex_setup_h100_b64.csv

The CSV header is::

    batch,N,prepare_ms,first_call_ms,compile_overhead_ms,total_setup_ms,
    prepare_peak_mem_mb,first_call_peak_mem_mb,steady_ms,steady_std_ms,
    steady_mem_mb,status

Rows with a failed measurement leave every numeric cell empty and record
``oom`` or ``err`` in ``status``; the run continues with the next shape.
"""

import sys
from pathlib import Path

BENCHMARK_ATTN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARK_ATTN_ROOT))

import argparse
import gc
import time
from dataclasses import dataclass

import torch

from src.attention_modules import FlexAttentionModule
from src.bench.cli import parse_int_list
from src.bench.data import make_inputs
from src.bench.io import write_results_csv
from src.bench.timing import measure


# Fixed harness settings required by SPEC 0051 H2.  ``make_inputs`` is called
# with ``coord_dim + 1`` exactly as in ``benchmark_sweep.py``.
D_MODEL = 320
N_HEAD = 8
COORD_DIM = 3
CUTOFF_SPATIAL = 256.0
MODE = "none"
DIST_MODE = "v1"
SEED = 42

DEFAULT_NS = "128,256,512,1024,2048,4096,8192,16384,32768"

CSV_HEADER = [
    "batch",
    "N",
    "prepare_ms",
    "first_call_ms",
    "compile_overhead_ms",
    "total_setup_ms",
    "prepare_peak_mem_mb",
    "first_call_peak_mem_mb",
    "steady_ms",
    "steady_std_ms",
    "steady_mem_mb",
    "status",
]


@dataclass(frozen=True)
class ShapeMeasurement:
    """Successful setup and steady-state measurements for one ``(B, N)``.

    Attributes:
        prepare_ms: Wall time of ``module.prepare(coords)`` in milliseconds.
        first_call_ms: Wall time of the first forward call in milliseconds;
            includes ``torch.compile`` and one forward execution.
        compile_overhead_ms: Signed ``first_call_ms - steady_ms``.
        total_setup_ms: ``prepare_ms + max(0.0, compile_overhead_ms)``.
        prepare_peak_mem_mb: Incremental peak device memory during
            ``prepare`` in MiB.
        first_call_peak_mem_mb: Incremental peak device memory during the
            first forward call in MiB.
        steady_ms: Mean steady-state forward time in milliseconds.
        steady_std_ms: Standard deviation across timing repeats in
            milliseconds (0.0 when ``repeats == 1``).
        steady_mem_mb: Incremental peak device memory of one steady-state
            call in MiB.
    """

    prepare_ms: float
    first_call_ms: float
    compile_overhead_ms: float
    total_setup_ms: float
    prepare_peak_mem_mb: float
    first_call_peak_mem_mb: float
    steady_ms: float
    steady_std_ms: float
    steady_mem_mb: float


def _is_cuda(device: torch.device) -> bool:
    """Return whether ``device`` selects an available CUDA device.

    Args:
        device: Torch device resolved from the CLI.

    Returns:
        ``True`` when CUDA timing and memory calls are valid.
    """
    return device.type == "cuda" and torch.cuda.is_available()


def _synchronize(device: torch.device) -> None:
    """Block until queued CUDA work on ``device`` has completed.

    Args:
        device: Torch device used for the measurement.
    """
    if _is_cuda(device):
        torch.cuda.synchronize()


def _cleanup_after_shape() -> None:
    """Release cached CUDA memory after a finished or failed shape.

    A CUDA OOM or compile failure can leave the caching allocator holding
    fragmented blocks.  Collecting Python garbage and emptying the cache keeps
    one failed shape from cascading into the next configuration.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_delta_mb(baseline_bytes: int, device: torch.device) -> float:
    """Return peak-minus-baseline device memory in MiB.

    Args:
        baseline_bytes: ``torch.cuda.memory_allocated()`` recorded before the
            measured region.
        device: Torch device used for the measurement.

    Returns:
        Incremental peak memory in MiB, or ``0.0`` on non-CUDA devices.
    """
    if not _is_cuda(device):
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated()
    return (peak_bytes - baseline_bytes) / (1024 ** 2)


def _time_prepare(
    module: FlexAttentionModule,
    coords: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Time ``module.prepare(coords)`` and its incremental peak memory.

    The measured region includes ``create_block_mask`` and creation of the
    ``torch.compile`` wrapper.  It excludes module construction and input
    generation, which happen before this function is called.

    Args:
        module: Freshly constructed FlexAttention module.
        coords: Coordinates ``(batch_size, seq_len, coord_dim + 1)``.
        device: Torch device used for the measurement.

    Returns:
        Tuple ``(prepare_ms, prepare_peak_mem_mb)``.
    """
    _synchronize(device)
    gc.collect()
    if _is_cuda(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
    else:
        baseline_bytes = 0

    start_time = time.perf_counter()
    module.prepare(coords)
    _synchronize(device)
    prepare_ms = (time.perf_counter() - start_time) * 1000.0
    prepare_peak_mem_mb = _peak_delta_mb(baseline_bytes, device)
    return prepare_ms, prepare_peak_mem_mb


def _time_first_call(
    module: FlexAttentionModule,
    query: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
    no_grad: bool,
) -> tuple[float, float]:
    """Time one fresh forward call, which triggers ``torch.compile``.

    ``module.prepare`` already ran, so this call reuses the block mask and the
    compiled wrapper but performs the actual Inductor compilation for the
    current shape.  The measured region contains that compilation *and* one
    forward execution.

    Args:
        module: Prepared FlexAttention module.
        query: Query tensor ``(batch_size, seq_len, d_model)``.
        coords: Coordinates ``(batch_size, seq_len, coord_dim + 1)``.
        device: Torch device used for the measurement.
        no_grad: When ``True``, run the call under ``torch.no_grad()``.

    Returns:
        Tuple ``(first_call_ms, first_call_peak_mem_mb)``.
    """
    _synchronize(device)
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
    else:
        baseline_bytes = 0

    start_time = time.perf_counter()
    if no_grad:
        with torch.no_grad():
            module(query, query, query, coords=coords)
    else:
        module(query, query, query, coords=coords)
    _synchronize(device)
    first_call_ms = (time.perf_counter() - start_time) * 1000.0
    first_call_peak_mem_mb = _peak_delta_mb(baseline_bytes, device)
    return first_call_ms, first_call_peak_mem_mb


def _measure_steady_state(
    module: FlexAttentionModule,
    query: torch.Tensor,
    coords: torch.Tensor,
    warmup: int,
    min_run_time: float,
    repeats: int,
    no_grad: bool,
) -> tuple[float, float, float]:
    """Measure steady-state forward time and memory with the harness helper.

    Args:
        module: Prepared FlexAttention module whose compile is already warm.
        query: Query tensor ``(batch_size, seq_len, d_model)``.
        coords: Coordinates ``(batch_size, seq_len, coord_dim + 1)``.
        warmup: Warmup iterations passed to ``measure``.
        min_run_time: Minimum autorange duration in seconds.
        repeats: Number of timing repetitions passed to ``measure``.
        no_grad: When ``True``, run every call under ``torch.no_grad()``.

    Returns:
        Tuple ``(steady_ms, steady_std_ms, steady_mem_mb)``.
    """

    def forward_call():
        """Run one FlexAttention forward pass on the prepared module."""
        return module(query, query, query, coords=coords)

    if no_grad:
        def timed_call():
            """Run one forward pass in inference mode for timing."""
            with torch.no_grad():
                return forward_call()
    else:
        timed_call = forward_call

    time_s, time_std_s, memory_mb = measure(
        timed_call,
        warmup=warmup,
        min_run_time=min_run_time,
        repeats=repeats,
    )
    return time_s * 1000.0, time_std_s * 1000.0, memory_mb


def _measure_shape(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    min_run_time: float,
    repeats: int,
    no_grad: bool,
) -> ShapeMeasurement:
    """Measure setup and steady state for one ``(batch_size, seq_len)``.

    Dynamo state is reset before the module is built so no in-memory compiled
    graph from a previous shape is reused.  The on-disk Inductor cache is not
    reset here; see the module docstring for that caveat.

    Args:
        batch_size: Batch dimension B.
        seq_len: Sequence length N.
        device: Torch device for inputs and module.
        dtype: Torch dtype for inputs and module.
        warmup: Warmup iterations for the steady-state measurement.
        min_run_time: Minimum autorange duration in seconds.
        repeats: Number of steady-state timing repetitions.
        no_grad: When ``True``, run forward calls under ``torch.no_grad()``.

    Returns:
        A :class:`ShapeMeasurement` with all times in milliseconds and all
        memory values in MiB.
    """
    torch._dynamo.reset()
    query, coords = make_inputs(
        batch_size,
        seq_len,
        D_MODEL,
        COORD_DIM,
        device,
        dtype,
        seed=SEED,
    )
    module = FlexAttentionModule(
        embed_dim=D_MODEL,
        n_head=N_HEAD,
        cutoff_spatial=CUTOFF_SPATIAL,
        mode=MODE,
        dist_mode=DIST_MODE,
    ).to(device, dtype)

    prepare_ms, prepare_peak_mem_mb = _time_prepare(module, coords, device)
    first_call_ms, first_call_peak_mem_mb = _time_first_call(
        module, query, coords, device, no_grad
    )
    steady_ms, steady_std_ms, steady_mem_mb = _measure_steady_state(
        module,
        query,
        coords,
        warmup=warmup,
        min_run_time=min_run_time,
        repeats=repeats,
        no_grad=no_grad,
    )

    compile_overhead_ms = first_call_ms - steady_ms
    total_setup_ms = prepare_ms + max(0.0, compile_overhead_ms)
    return ShapeMeasurement(
        prepare_ms=prepare_ms,
        first_call_ms=first_call_ms,
        compile_overhead_ms=compile_overhead_ms,
        total_setup_ms=total_setup_ms,
        prepare_peak_mem_mb=prepare_peak_mem_mb,
        first_call_peak_mem_mb=first_call_peak_mem_mb,
        steady_ms=steady_ms,
        steady_std_ms=steady_std_ms,
        steady_mem_mb=steady_mem_mb,
    )


def _classify_error(error: Exception) -> str:
    """Map an exception to the CSV status vocabulary.

    Args:
        error: Exception raised while building or measuring a shape.

    Returns:
        ``"oom"`` for out-of-memory failures, otherwise ``"err"``.
    """
    if "out of memory" in str(error).lower():
        return "oom"
    return "err"


def _format_row(
    batch_size: int,
    seq_len: int,
    measurement: ShapeMeasurement | None,
    status: str,
) -> list:
    """Convert one shape outcome into a CSV row.

    Args:
        batch_size: Batch dimension B.
        seq_len: Sequence length N.
        measurement: Successful measurement, or ``None`` on failure.
        status: ``"ok"``, ``"oom"``, or ``"err"``.

    Returns:
        A row matching :data:`CSV_HEADER`.  Every numeric cell is empty when
        ``measurement`` is ``None``.
    """
    if measurement is None:
        numeric_cells = [""] * 9
    else:
        numeric_cells = [
            f"{measurement.prepare_ms:.3f}",
            f"{measurement.first_call_ms:.3f}",
            f"{measurement.compile_overhead_ms:.3f}",
            f"{measurement.total_setup_ms:.3f}",
            f"{measurement.prepare_peak_mem_mb:.1f}",
            f"{measurement.first_call_peak_mem_mb:.1f}",
            f"{measurement.steady_ms:.3f}",
            f"{measurement.steady_std_ms:.3f}",
            f"{measurement.steady_mem_mb:.1f}",
        ]
    return [batch_size, seq_len, *numeric_cells, status]


def run_measurements(
    batch_sizes: list[int],
    seq_lens: list[int],
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    min_run_time: float,
    repeats: int,
    no_grad: bool,
) -> list[list]:
    """Measure every ``(batch_size, seq_len)`` pair and collect CSV rows.

    Failures are recorded with blank numeric fields and a ``status`` of
    ``oom`` or ``err``; the loop continues with the next shape.

    Args:
        batch_sizes: Batch dimensions B to measure.
        seq_lens: Sequence lengths N to measure.
        device: Torch device for inputs and module.
        dtype: Torch dtype for inputs and module.
        warmup: Warmup iterations for steady-state timing.
        min_run_time: Minimum autorange duration in seconds.
        repeats: Number of steady-state timing repetitions.
        no_grad: When ``True``, run forward calls under ``torch.no_grad()``.

    Returns:
        CSV-style rows in measurement order.
    """
    rows = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(
                f"\n[Shape] B={batch_size} N={seq_len}",
                flush=True,
            )
            status = "ok"
            measurement = None
            try:
                measurement = _measure_shape(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    dtype=dtype,
                    warmup=warmup,
                    min_run_time=min_run_time,
                    repeats=repeats,
                    no_grad=no_grad,
                )
            except RuntimeError as error:
                status = _classify_error(error)
                print(f"  [{status}] {type(error).__name__}: {error}", flush=True)
            except Exception as error:  # noqa: BLE001 — recorded as a soft outcome
                status = _classify_error(error)
                print(f"  [{status}] {type(error).__name__}: {error}", flush=True)
            finally:
                _cleanup_after_shape()

            if measurement is not None:
                print(
                    f"  prepare={measurement.prepare_ms:.3f} ms "
                    f"first_call={measurement.first_call_ms:.3f} ms "
                    f"compile_overhead={measurement.compile_overhead_ms:.3f} ms "
                    f"total_setup={measurement.total_setup_ms:.3f} ms "
                    f"steady={measurement.steady_ms:.3f} ms "
                    f"status={status}",
                    flush=True,
                )
            rows.append(
                _format_row(batch_size, seq_len, measurement, status)
            )
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the FlexAttention setup benchmark.

    Returns:
        A configured ``argparse.ArgumentParser`` whose defaults reproduce the
        H100 run: batch 64, the full N grid, warmup 5, 300 ms minimum run
        time, and three timing repeats.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure FlexAttention block-mask and torch.compile setup cost "
            "separately from steady-state forward time (SPEC 0051 H2)."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        default="64",
        help="Comma-separated batch sizes B (default: 64).",
    )
    parser.add_argument(
        "--Ns",
        default=DEFAULT_NS,
        help=(
            "Comma-separated sequence lengths N "
            f"(default: {DEFAULT_NS})."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations for steady-state timing (default: 5).",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=300,
        help="Minimum autorange run time in milliseconds (default: 300).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Steady-state timing repetitions (default: 3).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device: 'cuda' or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--out",
        default="results/flex_setup_h100.csv",
        help=(
            "Output CSV path (default: results/flex_setup_h100.csv). "
            "Ignored in --smoke mode, which always writes "
            "results/flex_setup_smoke.csv."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Local validation: override to B=2, N in {128,256}, warmup=1, "
            "rep=50 ms, repeats=1, and write results/flex_setup_smoke.csv."
        ),
    )
    parser.add_argument(
        "--no-grad",
        dest="no_grad",
        action="store_true",
        default=True,
        help=(
            "Run first and steady forward calls under torch.no_grad() "
            "(default: true; matches the report's B=64 inference regime)."
        ),
    )
    return parser


def main() -> None:
    """Parse CLI arguments, run the measurements, and write the CSV."""
    parser = build_argument_parser()
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if args.smoke:
        batch_sizes = [2]
        seq_lens = [128, 256]
        warmup = 1
        min_run_time = 50 / 1000.0
        repeats = 1
        out_path = Path("results/flex_setup_smoke.csv")
        print(
            "[Smoke] B=2, N=128,256, warmup=1, rep=50 ms, repeats=1; "
            "output results/flex_setup_smoke.csv"
        )
    else:
        batch_sizes = parse_int_list(args.batch_sizes)
        seq_lens = parse_int_list(args.Ns)
        warmup = args.warmup
        min_run_time = args.rep / 1000.0
        repeats = args.repeats
        out_path = Path(args.out)

    print(f"Device: {device}  dtype: {dtype}")
    print(
        f"Fixed: d={D_MODEL} nhead={N_HEAD} coord_dim={COORD_DIM} "
        f"cutoff_spatial={CUTOFF_SPATIAL} mode={MODE} dist_mode={DIST_MODE} "
        f"seed={SEED} no_grad={args.no_grad}"
    )
    print(
        f"Timing: warmup={warmup} min_run_time={min_run_time:.3f}s "
        f"repeats={repeats}"
    )
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lens}")

    rows = run_measurements(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        device=device,
        dtype=dtype,
        warmup=warmup,
        min_run_time=min_run_time,
        repeats=repeats,
        no_grad=args.no_grad,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_csv(rows, out_path, CSV_HEADER)
    print(f"\nWrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
