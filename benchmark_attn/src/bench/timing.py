"""Shared timing helpers for the benchmark harness.

The ``measure()`` variant guarded by ``torch.cuda.is_available()`` (from the
sparse/small-n scripts) is kept as canonical so that both helpers degrade
gracefully on CPU-only machines, returning 0.0 for peak memory instead of
crashing on CUDA API calls.
"""

import gc
import statistics
import time

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


class ComponentTimer:
    """Record named boundary events for component-level timings.

    On CUDA the boundaries are ``torch.cuda.Event(enable_timing=True)``
    objects on the current stream; on CPU they are ``time.perf_counter``
    readings. The class is additive: :func:`measure` and :func:`try_bench`
    keep their existing behaviour and do not use it.

    Callers assign a distinct name to every boundary, for example
    ``timer.start("forward_start")``, ``timer.start("encoder_0_start")``,
    ``timer.stop("encoder_0_end")``, and then
    ``timer.elapsed_ms("forward_start", "encoder_0_start")``. Because start
    and stop boundaries have different names, an interval can be measured
    from a start event to another start event (for ``embed_ms``), from a stop
    event to another stop event (for ``head_logits_ms``), or from a start to
    a stop event (for ``full_forward_ms``).

    The timer is reusable across repeats: :meth:`clear` discards the previous
    boundary events, and recording a name again replaces its old event.
    """

    def __init__(self, use_cuda: bool | None = None) -> None:
        """Create a timer that uses CUDA events or CPU wall-clock counters.

        Args:
            use_cuda: Force CUDA-event timing when ``True`` or CPU
                ``time.perf_counter`` timing when ``False``. ``None`` selects
                CUDA when :func:`torch.cuda.is_available` reports True.

        Raises:
            RuntimeError: If CUDA timing is requested but CUDA is unavailable.
        """
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA event timing was requested but CUDA is not available."
            )
        self._use_cuda = bool(use_cuda)
        self._events: dict[str, torch.cuda.Event | float] = {}

    @property
    def use_cuda(self) -> bool:
        """Whether this timer records CUDA events instead of CPU counters."""
        return self._use_cuda

    def start(self, event_name: str) -> None:
        """Record the start boundary called ``event_name``.

        Args:
            event_name: Unique name for this boundary; recording the same name
                again replaces the previous event.
        """
        self._record(event_name)

    def stop(self, event_name: str) -> None:
        """Record the stop boundary called ``event_name``.

        Args:
            event_name: Unique name for this boundary; recording the same name
                again replaces the previous event.
        """
        self._record(event_name)

    def _record(self, event_name: str) -> None:
        """Record one boundary event using the timer's backend.

        Args:
            event_name: Name under which the event is stored.
        """
        if self._use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._events[event_name] = event
        else:
            self._events[event_name] = time.perf_counter()

    def elapsed_ms(self, start_event_name: str, end_event_name: str) -> float:
        """Return the elapsed milliseconds between two recorded boundaries.

        On CUDA the current stream is synchronised before the event pair is
        read, so the result includes all work queued between the two
        boundaries. On CPU the stored ``time.perf_counter`` readings are
        subtracted and scaled to milliseconds.

        Args:
            start_event_name: Name passed to :meth:`start` for the interval
                start.
            end_event_name: Name passed to :meth:`stop` (or :meth:`start`)
                for the interval end.

        Returns:
            Elapsed time in milliseconds as a float.

        Raises:
            KeyError: If either boundary name has not been recorded.
        """
        start_event = self._events[start_event_name]
        end_event = self._events[end_event_name]
        if self._use_cuda:
            torch.cuda.synchronize()
            return float(start_event.elapsed_time(end_event))
        return float((end_event - start_event) * 1000.0)

    def clear(self) -> None:
        """Discard all recorded boundary events so the timer can be reused."""
        self._events.clear()

    def synchronize(self) -> None:
        """Wait for queued CUDA work; a no-op on CPU."""
        if self._use_cuda:
            torch.cuda.synchronize()
