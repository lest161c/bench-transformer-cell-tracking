"""Shared timing helpers for the benchmark harness.

The ``measure()`` variant guarded by ``torch.cuda.is_available()`` (from the
sparse/small-n scripts) is kept as canonical so that both helpers degrade
gracefully on CPU-only machines, returning 0.0 for peak memory instead of
crashing on CUDA API calls.
"""

import gc
import statistics

import torch
import torch.utils.benchmark as benchmark


# Pin the thread count once per process so benchmark.Timer's implicit
# torch.set_num_threads(1) per measurement is a value-level no-op; a
# changing num_threads trips dynamo's GLOBAL_STATE guard and forces a
# recompile, and after torch._dynamo.config.recompile_limit (8)
# recompiles dynamo SILENTLY falls back to eager, which materializes the
# N x N matrix (quadratic memory, 20-50x slowdown) instead of the
# compiled kernel. Pinning once at import makes the guard stable.
torch.set_num_threads(1)

# Belt and braces: allow more recompiles before dynamo gives up, so a
# stray thread-count change cannot silently force an eager fallback
# mid-sweep.
if hasattr(torch._dynamo, "config") and hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 32


def measure(fn, warmup=5, min_run_time=0.3, repeats=1) -> tuple[float, float, float]:
    """Benchmark a callable, returning (mean_time_s, std_time_s, incremental_peak_mem_mb).

    When ``repeats > 1`` the timing measurement is repeated ``repeats`` times
    and the mean and sample standard deviation (Bessel-corrected, ``n-1``
    denominator) of the per-repeat mean wall times are returned; when
    ``repeats == 1`` the existing single-run behaviour is preserved and
    ``std_time_s`` is 0.0 (a single sample has no variance).

    Memory is measured on one representative call (single ``fn()``
    invocation), NOT maxed across repeats — the incremental peak memory of
    a single call is reported.

    Args:
        fn: Callable to benchmark; each invocation runs one forward pass.
        warmup: Number of warmup iterations executed before timing starts.
        min_run_time: Minimum run time in seconds for the autorange timer.
        repeats: Number of timing repetitions per configuration.

    Returns:
        Tuple (mean_time_s, std_time_s, incremental_peak_mem_mb).
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
    all_mean_times = [timer.blocked_autorange(min_run_time=min_run_time).mean
                      for _ in range(repeats)]
    mean_time_s = statistics.fmean(all_mean_times)
    if len(all_mean_times) > 1:
        std_time_s = statistics.stdev(all_mean_times)
    else:
        std_time_s = 0.0
    return mean_time_s, std_time_s, memory_mb


def _recover_after_failure() -> None:
    """Release failed-allocation state so the next config starts clean.

    A CUDA OOM (or any failed kernel launch) can leave the caching
    allocator fragmented or holding reserved blocks; without an explicit
    release the NEXT configuration can spuriously OOM too. Called after
    every failed benchmark attempt — OOM is a soft, recorded outcome
    (status ``'oom'``), never a run-terminating condition.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def try_bench(bench_fn, *args, warmup=5, min_run_time=0.3, repeats=1, **kwargs) -> tuple[float, float, float, str]:
    """Try to build and benchmark, catching OOM/errors as soft outcomes.

    Calls ``bench_fn(*args, **kwargs)`` to build a callable and benchmarks it
    with :func:`measure`, passing ``warmup``/``min_run_time``/``repeats``
    through. CUDA out-of-memory errors are reported with the dedicated
    status ``"oom"``; any other error is reported as ``"err: <message>"``.
    Either way the failure is RECORDED, not raised: the sweep continues
    with the next configuration, and :func:`_recover_after_failure`
    releases the failed allocation state first so one OOM cannot cascade
    into the following configurations.

    Args:
        bench_fn: Function that builds and returns the callable to benchmark.
        *args: Positional arguments passed to bench_fn.
        warmup: Warmup iterations passed to :func:`measure`.
        min_run_time: Minimum autorange run time in seconds.
        repeats: Timing repetitions passed to :func:`measure`.
        **kwargs: Keyword arguments passed to bench_fn.

    Returns:
        Tuple (time_s_mean, time_s_std, memory_mb, status). On success the
        numeric fields come from :func:`measure` and status is ``"ok"``. On
        failure the numeric fields are -1 and status is ``"oom"`` or
        ``"err: ..."``.
    """
    try:
        fn = bench_fn(*args, **kwargs)
        time_s_mean, time_s_std, memory_mb = measure(fn, warmup=warmup, min_run_time=min_run_time, repeats=repeats)
        return time_s_mean, time_s_std, memory_mb, "ok"
    except RuntimeError as e:
        _recover_after_failure()
        msg = str(e).lower()
        if "out of memory" in msg:
            return -1, -1, -1, "oom"
        return -1, -1, -1, f"err: {e}"
    except Exception as e:
        _recover_after_failure()
        return -1, -1, -1, f"err: {e}"
