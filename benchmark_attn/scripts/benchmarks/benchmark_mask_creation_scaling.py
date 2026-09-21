"""Creation-only scaling benchmark for the O(N^2) KNN attention mask.

Why this exists
---------------
The KNN masked-attention path (``KNNMaskSparseAttention`` in
``src/attention_modules.py``) spends part of every forward pass building a
``(batch_size, n_head, seq_len, seq_len)`` float mask via ``torch.full(-inf)``
followed by ``mask.scatter_(3, indices, 0.0)`` at
``src/attention_modules.py:889`` (``KNNMaskSparseAttention._make_mask``).
The construction is a single float mask (``-inf`` background, ``0.0`` at
neighbour positions); this attention path has no separate Boolean-to-float
conversion step, so ``_make_mask`` is the complete creation cost.  The
existing H100 ``mask_knn`` timings (``benchmark_mask_vs_gather.csv`` at the
repo root and the H100 sweep rows in ``benchmark_attn/results/``) measure
the full attention forward, so the mask-construction cost is entangled with
the rest of the path.  The A500 (4 GB) historically OOM'd above N=512, so
this O(N^2) cost has never been measured in isolation for the N >= 2048
training regime.  This benchmark times ONLY the construction
(``KNNMaskSparseAttention._make_mask`` is imported and reused unchanged --
``attention_modules.py`` is not modified) for N in {2048, 4096, 8192, 16384}
and K in {16, 64} to validate the K-sweep training framing.  An H100 (80 GB)
fits the mask trivially at these sizes (at the default shapes
batch_size=1, n_head=8, float32 the mask is ``32*N^2`` bytes: ~134 MB at
N=2048 up to ~8.6 GB at N=16384); any (N, K) combination that does not fit
the device is skipped with a logged ``[SKIP OOM]`` message so the script
also runs on the 4 GB A500 or on CPU for a small-N smoke test.

What is measured
----------------
Per (N, K) combination:
  * the median wall time of ``_make_mask`` over ``--rep`` (>= 20) repetitions,
    each bracketed by ``torch.cuda.synchronize()`` (CPU calls are already
    synchronous), after ``--warmup`` warmup builds;
  * the incremental peak memory of one build.  On CUDA this is exact
    (``torch.cuda.max_memory_allocated`` minus the pre-build allocated
    baseline, the same recipe as ``src.bench.timing.measure``); on CPU it is
    the ``getrusage`` ``ru_maxrss`` high-water delta and therefore
    under-reports when the process watermark already exceeds the new
    allocation (the OS never returns memory to the baseline after a free).

Output
------
A CSV with the schema of the repo-root ``benchmark_mask_vs_gather.csv``
(``method,N,K,time_s,mem_mb,speedup_vs_gather``).  Creation-only rows use
``method=mask_creation`` so they are distinguishable from the
full-attention ``mask_knn`` rows in the same schema; ``speedup_vs_gather``
is left empty because a creation-only timing has no gather-attention
counterpart.  KNN indices are generated with the same recipe as the sweep
harness (seeded Gaussian coords -> cdist -> topk via
``src.bench.data.compute_knn_indices``), so timings are reproducible for a
given ``--seed``.

How to run
----------
Local CPU smoke test (from the ``benchmark_attn`` directory):
    python3 scripts/benchmarks/benchmark_mask_creation_scaling.py \\
        --device cpu --seq-lens 512 --k-values 16,64 \\
        --out results/mask_creation_scaling.csv

H100 production run (minutes; from a login node of the cluster):
    srun --account=p_scads_celltracking --partition=gpu-h100 \\
        --gres=gpu:1 --cpus-per-task=8 --mem=90G --time=00:30:00 \\
        bash -c 'source "$HOME/.bench.env" && \\
                 source "$REPO_ROOT/slurm/load_cluster_env.sh" && \\
                 cd "$BENCH/benchmark_attn" && \\
                 .venv/bin/python scripts/benchmarks/benchmark_mask_creation_scaling.py \\
                     --device cuda --seq-lens 2048,4096,8192,16384 \\
                     --k-values 16,64 --warmup 5 --rep 30 \\
                     --out results/mask_creation_scaling.csv'
    The partition name is ``gpu-h100`` per C11 in ``HANDOVER.md``; a
    separate handover document uses ``capella``.  Verify with
    ``sinfo -p gpu-h100,gpu-capella`` before launching.

Demonstrating the OOM-skip path:
    On the 4 GB A500 (or any small GPU), simply request a combo that cannot
    fit and the real CUDA out-of-memory is caught and skipped:
        python3 scripts/benchmarks/benchmark_mask_creation_scaling.py \\
            --device cuda --seq-lens 65536 --k-values 16 \\
            --out /tmp/mask_creation_skip_demo.csv
    On a CPU-only host the same path can be exercised with an address-space
    cap, which makes the cdist allocation fail inside torch (a
    ``RuntimeError`` from the CPU allocator) instead of relying on kernel
    overcommit -- which on many hosts grants the mapping and only kills the
    process later when the pages are touched, an OOM kill rather than a
    catchable Python exception:
        bash -c 'ulimit -v 8000000; python3 \\
            scripts/benchmarks/benchmark_mask_creation_scaling.py \\
            --device cpu --seq-lens 65536 --k-values 16 \\
            --out /tmp/mask_creation_skip_demo.csv'
    In both cases the combo is skipped with a ``[SKIP OOM]`` message and
    the CSV is still written (header only when every combo is skipped).
    Caveat: a GPU host with a broken NVML driver (``nvidia-smi`` failing
    with a driver/library version mismatch) turns the failing allocation
    into a ``RuntimeError ... INTERNAL ASSERT FAILED ... CUDACachingAllocator``
    instead of a clean CUDA OOM; that error is deliberately NOT classified
    as OOM (it is a driver defect, not a capacity limit) and propagates.
"""

import argparse
import gc
import resource
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

# Make ``src`` importable regardless of CWD (same shim as the sibling
# benchmark scripts in this directory).
_SCRIPT_PARENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_PARENT.parents[1]))

from src.attention_modules import KNNMaskSparseAttention
from src.bench.cli import parse_int_list
from src.bench.data import compute_knn_indices
from src.bench.io import write_results_csv

_CSV_HEADER = ["method", "N", "K", "time_s", "mem_mb", "speedup_vs_gather"]
_DTYPE_CHOICES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_MIN_REPS = 20  # SPEC 0002 T3: median over at least 20 synchronize-bracketed reps.


def is_out_of_memory_error(exception: BaseException) -> bool:
    """Return True if ``exception`` looks like a device allocation failure.

    Why this exists: both CUDA out-of-memory (a ``torch.cuda.OutOfMemoryError``
    whose message contains "out of memory") and CPU allocation failures
    (the PyTorch CPU allocator raises ``RuntimeError`` with messages such
    as "DefaultCPUAllocator: can't allocate memory ... Error code 12") arrive
    as ``RuntimeError`` subclasses but with device-specific wording.  Both
    must trigger the skip path rather than aborting the sweep.

    Args:
        exception: The exception raised while building or timing a combo.

    Returns:
        True when the message indicates an allocation failure, False
        otherwise.  All non-allocation failures are left to propagate.
    """
    message = str(exception).lower()
    return (
        "out of memory" in message
        or "bad_alloc" in message
        or "cannot allocate memory" in message
        or ("alloc" in message and "memory" in message)
    )


def synchronize_device(device: torch.device) -> None:
    """Block until pending work on ``device`` finishes (CUDA only).

    Why this exists: CUDA kernel launches are asynchronous, so a plain
    ``time.perf_counter`` around the Python call would only measure the
    launch latency, not the device-side construction.  CPU ops are already
    synchronous at the Python boundary, so this is a no-op for CPU.

    Args:
        device: Target device for the synchronization.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_mask_creation_closure(
    batch_size: int,
    seq_len: int,
    knn_neighbors: int,
    d_model: int,
    n_head: int,
    coord_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Callable[[], torch.Tensor]:
    """Return a zero-arg closure that runs the mask-construction path once.

    Why this exists: isolates the timed operation from the (untimed)
    input-preparation work.  The closure invokes
    ``KNNMaskSparseAttention._make_mask`` -- the exact method used by the
    attention forward at ``src/attention_modules.py:852-890``, with
    ``scatter_`` at line ``889`` -- so the benchmark reuses the production
    code path without modifying ``attention_modules.py``.  The module's
    parameters are constructed but never moved to ``device`` because
    ``_make_mask`` does not read them; this keeps CPU smoke tests free of
    needless GPU allocations and makes the intent of the benchmark clear.

    Args:
        batch_size: Batch dimension.
        seq_len: Sequence length (N).
        knn_neighbors: Number of KNN neighbours per query (K).
        d_model: Embedding dimension (used only to size the module's
            projection layers, which are unused here).
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns (the harness adds
            a leading non-spatial column that is dropped by
            ``compute_knn_indices``).
        device: Device on which to allocate the KNN index tensor and the
            mask.
        dtype: Dtype of the mask tensor (matches the attention module's
            query/key dtype).
        seed: Seed for ``torch.manual_seed`` before the coordinate draw,
            so identical arguments produce identical KNN indices.

    Returns:
        A zero-arg callable that returns the new mask tensor.
    """
    torch.manual_seed(seed)
    coords_with_lead = torch.randn(
        batch_size, seq_len, coord_dim + 1, device=device, dtype=torch.float32
    )
    coords_with_lead[..., 0] *= 4  # leading non-spatial column; dropped by compute_knn_indices (mirrors src/bench/data.make_inputs)
    knn_indices = compute_knn_indices(coords_with_lead, knn_neighbors)
    del coords_with_lead  # coords are inputs only; indices are the only KNN artefact the closure needs.

    attention_module = KNNMaskSparseAttention(
        embed_dim=d_model,
        n_head=n_head,
        knn_neighbors=knn_neighbors,
        coord_dim=coord_dim,
        mode="none",
    )

    def build_mask() -> torch.Tensor:
        return attention_module._make_mask(
            batch_size=batch_size,
            n_head=n_head,
            seq_len=seq_len,
            knn_neighbors=knn_neighbors,
            knn_indices=knn_indices,
            device=device,
            dtype=dtype,
        )

    return build_mask


def time_mask_creation_median_s(
    build_mask: Callable[[], torch.Tensor],
    warmup_iterations: int,
    measured_reps: int,
    device: torch.device,
) -> float:
    """Median wall time of one mask build across ``measured_reps`` reps.

    Why this exists: the SPEC requires synchronize-bracketed median timing
    (not mean), with at least ``_MIN_REPS`` reps, after a warmup.  Each
    measured rep is bracketed by ``synchronize_device`` so the interval
    covers the full device-side construction; warmup primes the caching
    allocator and any first-touch CUDA kernel-cache effects.

    Args:
        build_mask: Zero-arg callable that builds one mask (the closure
            returned by :func:`build_mask_creation_closure`).
        warmup_iterations: Number of un-timed builds before measurement.
        measured_reps: Number of timed builds (>= ``_MIN_REPS``).
        device: Device on which to synchronize.

    Returns:
        Median wall time per build, in seconds.
    """
    for _ in range(warmup_iterations):
        build_mask()
    synchronize_device(device)

    sample_times_s: list[float] = []
    for _ in range(measured_reps):
        start_s = time.perf_counter()
        build_mask()
        synchronize_device(device)
        sample_times_s.append(time.perf_counter() - start_s)
    return statistics.median(sample_times_s)


def measure_incremental_peak_memory_mb(
    build_mask: Callable[[], torch.Tensor],
    device: torch.device,
) -> float:
    """Incremental peak memory of one mask build, in MiB.

    Why this exists: peak memory is reported separately from wall time
    because the two use different instrumentation (CUDA exact vs. CPU
    high-water-mark approximation), and the warmup loop above already
    primes the allocator so the measured build is steady-state.

    On CUDA the number is exact (``torch.cuda.max_memory_allocated`` delta
    over the allocated baseline after a cache flush, mirroring
    ``src.bench.timing.measure``).  On CPU the number is the
    ``resource.getrusage`` ``ru_maxrss`` delta in MiB -- a high-water mark
    that does not decrease after ``free``, so it under-reports when the
    process watermark already exceeds the new allocation.  This caveat is
    acceptable for the present purpose (the H100 numbers are the
    deliverable; the CPU number is a smoke-test sanity check).

    Args:
        build_mask: Zero-arg callable that builds one mask.
        device: Device whose memory is being measured.

    Returns:
        Incremental peak memory of the build, in MiB.
    """
    if device.type == "cuda":
        synchronize_device(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        build_mask()
        synchronize_device(device)
        return (torch.cuda.max_memory_allocated() - baseline_bytes) / (1024 ** 2)

    watermark_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    build_mask()
    watermark_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max(0.0, (watermark_after_kb - watermark_before_kb) / 1024.0)


def run_scaling_sweep(
    seq_lens: list[int],
    k_values: list[int],
    batch_size: int,
    d_model: int,
    n_head: int,
    coord_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    warmup_iterations: int,
    measured_reps: int,
) -> tuple[list[list[str]], int, int]:
    """Run every (N, K) combination, time mask creation, collect CSV rows.

    Why this exists: isolates the per-combo OOM-skip guard from the CSV
    writer and the CLI parser so each piece is unit-readable.  ``K >= N``
    is caught early (topk would refuse); allocation failures (CUDA OOM
    or CPU allocator refusal) are caught and logged as ``[SKIP OOM]`` so
    the sweep continues -- e.g. the 4 GB A500 finishes the small-N combos
    and logs skips for the rest.  Any other exception propagates, which is
    the honest failure mode.

    Args:
        seq_lens: Sequence lengths to sweep.
        k_values: KNN neighbour counts to sweep.
        batch_size: Batch dimension.
        d_model: Embedding dimension (sizes unused module parameters).
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns.
        device: Target device.
        dtype: Mask dtype.
        seed: RNG seed for KNN index generation.
        warmup_iterations: Warmup builds per combo.
        measured_reps: Measured reps per combo (>= ``_MIN_REPS``).

    Returns:
        Tuple ``(rows, ok_count, skipped_count)`` where ``rows`` is a list
        of 6-column CSV rows (header excluded) for successful combos.
    """
    rows: list[list[str]] = []
    ok_count = 0
    skipped_count = 0

    for seq_len in seq_lens:
        for knn_neighbors in k_values:
            if knn_neighbors >= seq_len:
                print(
                    f"  [SKIP] N={seq_len} K={knn_neighbors}: "
                    f"K >= N (topk would fail)",
                    flush=True,
                )
                skipped_count += 1
                continue
            try:
                build_mask = build_mask_creation_closure(
                    batch_size, seq_len, knn_neighbors,
                    d_model, n_head, coord_dim,
                    device, dtype, seed,
                )
                median_time_s = time_mask_creation_median_s(
                    build_mask, warmup_iterations, measured_reps, device
                )
                peak_memory_mb = measure_incremental_peak_memory_mb(
                    build_mask, device
                )
            except (RuntimeError, MemoryError) as error:
                if is_out_of_memory_error(error):
                    print(
                        f"  [SKIP OOM] N={seq_len} K={knn_neighbors}: "
                        f"{str(error)[:140]}",
                        flush=True,
                    )
                    skipped_count += 1
                    continue
                raise

            rows.append([
                "mask_creation",
                str(seq_len),
                str(knn_neighbors),
                f"{median_time_s:.9f}",
                f"{peak_memory_mb:.1f}",
                "",  # speedup_vs_gather has no counterpart for creation-only timing.
            ])
            ok_count += 1
            print(
                f"  N={seq_len:<6d} K={knn_neighbors:<3d} -> ok "
                f"{median_time_s * 1000:9.3f} ms  {peak_memory_mb:8.1f} MB",
                flush=True,
            )

    return rows, ok_count, skipped_count


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the ``--device`` argument to a concrete ``torch.device``.

    Why this exists: mirrors the harness ``--device auto`` convention
    (``benchmark_attn/scripts/benchmarks/benchmark_sweep.py``) so the
    CPU smoke and the H100 run use the same resolution rule.

    Args:
        device_arg: One of ``"auto"``, ``"cuda"``, ``"cpu"``.

    Returns:
        A ``torch.device`` of the resolved kind.
    """
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    """Parse CLI args, run the scaling sweep, and write the CSV.

    Why this exists: the entry point is small and delegates to named
    helpers (``build_mask_creation_closure``, ``time_mask_creation_median_s``,
    ``measure_incremental_peak_memory_mb``, ``run_scaling_sweep``,
    ``resolve_device``) so each piece can be inspected independently and
    the OOM-skip path is visible from the sweep driver alone.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Creation-only scaling benchmark for the O(N^2) KNN attention "
            "mask at attention_modules.py:889 (SPEC 0002 T3)."
        ),
    )
    parser.add_argument(
        "--seq-lens", default="2048,4096,8192,16384",
        help="Comma-separated sequence lengths to sweep.",
    )
    parser.add_argument(
        "--k-values", default="16,64",
        help="Comma-separated KNN neighbour counts to sweep.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch dimension (mask shape is batch_size * n_head * N * N).",
    )
    parser.add_argument(
        "--d", type=int, default=320,
        help="Embedding dimension (sizes unused module parameters).",
    )
    parser.add_argument(
        "--nhead", type=int, default=8,
        help="Number of attention heads (mask shape uses this).",
    )
    parser.add_argument(
        "--coord-dim", type=int, default=2,
        help="Number of spatial coordinate columns for KNN.",
    )
    parser.add_argument(
        "--dtype", default="float32", choices=sorted(_DTYPE_CHOICES),
        help="Mask dtype. Default float32 matches the SPEC memory "
             "estimates; --dtype float16 mirrors the old benchmark_sweep "
             "CUDA default.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Target device: 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup builds per combo (un-timed).",
    )
    parser.add_argument(
        "--rep", type=int, default=30,
        help=(
            f"Measured reps per combo (>= {_MIN_REPS}); median is taken "
            "over synchronize-bracketed wall times."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for KNN index generation.",
    )
    parser.add_argument(
        "--out", default="results/mask_creation_scaling.csv",
        help="Output CSV path (resolved relative to CWD).",
    )
    args = parser.parse_args()

    if args.rep < _MIN_REPS:
        parser.error(
            f"--rep must be >= {_MIN_REPS} (SPEC 0002 T3 requires median "
            "over at least 20 synchronize-bracketed reps)."
        )

    seq_lens = parse_int_list(args.seq_lens)
    k_values = parse_int_list(args.k_values)
    device = resolve_device(args.device)
    dtype = _DTYPE_CHOICES[args.dtype]

    print(f"Device: {device} (resolved from --device {args.device!r})")
    print(f"dtype: {dtype}  batch_size={args.batch_size}  "
          f"d={args.d}  nhead={args.nhead}  coord_dim={args.coord_dim}  "
          f"seed={args.seed}")
    print(f"seq_lens={seq_lens}  k_values={k_values}  "
          f"warmup={args.warmup}  rep={args.rep}")
    print(f"out: {args.out}\n", flush=True)

    rows, ok_count, skipped_count = run_scaling_sweep(
        seq_lens, k_values,
        batch_size=args.batch_size, d_model=args.d, n_head=args.nhead,
        coord_dim=args.coord_dim,
        device=device, dtype=dtype, seed=args.seed,
        warmup_iterations=args.warmup, measured_reps=args.rep,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_csv(rows, out_path, _CSV_HEADER)

    print(
        f"\nWrote {ok_count} ok + {skipped_count} skipped rows -> {out_path}"
    )


if __name__ == "__main__":
    main()
