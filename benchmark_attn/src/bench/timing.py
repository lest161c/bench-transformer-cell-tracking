"""Shared timing helpers for the benchmark harness.

Extracted from the near-identical ``measure()`` and ``try_bench()``
implementations that previously lived in each benchmark script under
``benchmarks/`` (e.g. ``benchmark_sparse.py``, ``benchmark_full.py``,
``benchmark_small_n.py``, ``benchmark_gather_v3.py``,
``benchmark_cached_dist.py``).

The ``measure()`` variant guarded by ``torch.cuda.is_available()`` (from the
sparse/small-n scripts) is kept as canonical so that both helpers degrade
gracefully on CPU-only machines, returning 0.0 for peak memory instead of
crashing on CUDA API calls.
"""

import gc

import torch
import torch.utils.benchmark as benchmark


def measure(fn, warmup=5, min_run_time=0.3) -> tuple[float, float]:
    """Benchmark a callable, returning (mean_time_s, incremental_peak_mem_mb).

    Runs ``warmup`` warmup iterations, then measures the incremental peak GPU
    memory of a single call (peak minus baseline, in MiB) and the mean wall
    time via ``torch.utils.benchmark.Timer.blocked_autorange`` with a minimum
    run time of ``min_run_time`` seconds.

    Args:
        fn: Callable to benchmark; each invocation runs one forward pass.
        warmup: Number of warmup iterations executed before timing starts.
        min_run_time: Minimum run time in seconds for the autorange timer.

    Returns:
        Tuple (mean_time_s, incremental_peak_mem_mb). On systems without CUDA
        the memory figure is 0.0 and timing falls back to the CPU timer.
    """
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        _ = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        memory_mb = (peak - baseline) / (1024 ** 2)
    else:
        memory_mb = 0.0

    timer = benchmark.Timer("fn()", globals={"fn": fn}, num_threads=1)
    result = timer.blocked_autorange(min_run_time=min_run_time)
    return result.mean, memory_mb


def try_bench(bench_fn, *args, **kwargs) -> tuple[float, float, str]:
    """Try to build and benchmark, catching OOM/errors. Returns (time_s, memory_mb, status).

    Calls ``bench_fn(*args, **kwargs)`` to build a callable and benchmarks it
    with :func:`measure`. CUDA out-of-memory errors are reported with the
    dedicated status ``"oom"``; any other error is reported as
    ``"err: <message>"``. Builders that raise before benchmarking yield the
    same failure tuples.

    Args:
        bench_fn: Function that builds and returns the callable to benchmark.
        *args: Positional arguments passed to bench_fn.
        **kwargs: Keyword arguments passed to bench_fn.

    Returns:
        Tuple (time_s, memory_mb, status). On success time_s and memory_mb
        come from :func:`measure` and status is ``"ok"``. On failure time_s
        and memory_mb are -1 and status is ``"oom"`` or ``"err: ..."``.
    """
    try:
        fn = bench_fn(*args, **kwargs)
        time_s, memory_mb = measure(fn)
        return time_s, memory_mb, "ok"
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or ("cuda" in msg and ("memory" in msg or "alloc" in msg)):
            return -1, -1, "oom"
        return -1, -1, f"err: {e}"
    except Exception as e:
        return -1, -1, f"err: {e}"
