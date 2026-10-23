"""cuDNN flash vs FA2 mask-overhead decomposition benchmark.

Why this exists
---------------
The H100 sweep rows for ``dense_flash`` (plain SDPA, no mask) and
``dense_masked`` (``CachedDistAttention``, N x N additive mask) bundle three
separable effects into one delta: (1) which fused kernel the unmasked call
dispatches to (cuDNN attention vs the PyTorch flash/FA2 kernel — different
kernels), (2) the cost of *constructing* the additive mask (zeros canvas +
``masked_fill_`` + ``cdist`` + ``exp`` decay), and (3) the cost of *carrying*
a mask into a kernel call (backend acceptance, layout conversion, kernel
swap).  The owner question is whether the observed
``dense_flash`` vs ``dense_masked`` gap is kernel-intrinsic or attributable
to mask construction.  This benchmark separates the three effects and
closes the decomposition against the end-to-end delta.

What is measured (stages)
-------------------------
A — ``backend_census``: for each forced SDPA backend
    (``FLASH_ATTENTION`` = the vendored FA2 kernels, ``CUDNN_ATTENTION``,
    ``EFFICIENT_ATTENTION``) one unmasked and one masked probe call is run
    under ``torch.profiler`` and the dispatched CUDA kernel names are
    recorded.  Backend forcing is a *probe* here — every later timed row
    also carries the kernel names captured for its exact configuration, so
    no number is ever attributed to a kernel that did not actually run.
B — ``kernel_intrinsic_nomask``: identical seeded Q/K/V, ``attn_mask=None``,
    forced ``FLASH_ATTENTION`` vs forced ``CUDNN_ATTENTION``.  This is the
    kernel-intrinsic difference; at ``d_head=40`` (Trackastra's real head
    dim) the flash/FA2 kernel is expected to refuse dispatch (flash kernels
    exist only for d in {16, 32, 64, 128, 256}), which the row records as an
    error status rather than a timing.
C — ``mask_build_isolated``: ONLY the additive mask construction of
    ``CachedDistAttention._build_mask`` (imported unchanged — no mirrored
    copy), timed per ``dist_mode`` (v0/v1) x ``dist_cache`` (on/off).  For
    ``dist_cache=on`` the fp32->dtype ``dist_2d`` matrix is pre-computed
    outside the timed closure with the same recipe as the sweep harness.
D — ``mask_carriage`` and ``end_to_end_anchor``: (i) cuDNN forced with the
    pre-built mask, (ii) mem-efficient forced with the same mask (the
    kernel ``dense_masked`` actually uses), (iii) unforced auto-dispatch
    with the mask (kernel names recorded), plus unmodified
    ``dense_flash`` / ``dense_masked`` module forwards as the end-to-end
    anchor.  Closure check: t(dense_masked) - t(dense_flash) should
    approximately equal mask-build (C) + mask-carriage (D-i/D-ii) +
    kernel-swap (D vs B) contributions.
E — ``causal_vs_mask_dtype``: diagnostic arms bounding the mask-carriage
    overhead.  ``is_causal=True`` needs no N x N mask input at all (the
    pattern is generated inside the kernel), timed for FLASH and CUDNN;
    a prebuilt bool mask pairs with stage D's prebuilt float mask to
    isolate the dispatcher's dtype handling.  The causal pattern is NOT
    the production spatial cutoff (which is distance-based) — these arms
    bound the overhead, they are not a production configuration.  Prebuilt
    masks are allocated before the harness baseline, so incremental peak
    memory isolates kernel-side allocations from caller-owned mask storage.

The ``sdpa_kernel`` context manager is entered inside every timed closure.
Its CPU overhead is identical across the compared arms, so within-stage
deltas are unbiased; absolute row times include a small constant context
overhead (documented alongside the experiment plan).

How to run
----------
Local CPU smoke test (plumbing only; forced-backend rows report ``err``
statuses on CPU by design), from the ``benchmark_attn`` directory::

    .venv/bin/python scripts/benchmarks/benchmark_cudnn_vs_fa2_mask.py \\
        --device cpu --seq-lens 128,256 --d-heads 40,64 \\
        --out results/cudnn_vs_fa2_mask_smoke.csv

H100 production run (via slurm; see ``slurm/run_cudnn_vs_fa2_mask.slurm``)::

    sbatch slurm/run_cudnn_vs_fa2_mask.slurm

Output
------
A CSV with the decomposition schema (``stage, method, N, d_head, time_ms,
time_std_ms, mem_mb, dist_mode, dist_cache, backend, kernel_name, error``)
plus an environment manifest JSON next to it (``*_env.json``) recording
torch/cuDNN versions, GPU identity, backend availability flags, and the
full CLI configuration, so the numbers are reproducible for a given
``--seed``.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from src.bench import (
    add_common_args, make_inputs, measure, parse_int_list, try_bench,
    write_results_csv,
)
from src.attention_modules import CachedDistAttention, DenseFlashAttention

# Backends probed in the census and compared in the kernel-intrinsic pair.
# FLASH_ATTENTION hosts the vendored FA2 kernels; CUDNN_ATTENTION is the
# cuDNN fused attention backend; EFFICIENT_ATTENTION is the mem-efficient
# (cutlass fmha) kernel that masked dense SDPA falls back to.
CENSUS_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
]

# Column order mirrors the sweep schema conventions (timing columns, then
# configuration columns, then dispatch evidence, then error status last).
RESULT_HEADER = [
    "stage", "method", "N", "d_head", "time_ms", "time_std_ms", "mem_mb",
    "dist_mode", "dist_cache", "backend", "kernel_name", "error",
]


def make_head_split_inputs(batch_size, seq_len, head_dim, n_head, coord_dim,
                           device, dtype, seed):
    """Create head-split Q/K/V tensors plus coordinates for kernel timing.

    Draws are seeded via the harness ``make_inputs`` recipe (three calls with
    ``seed``, ``seed + 1``, ``seed + 2`` for query/key/value so all arms of a
    comparison see bit-identical inputs) and the ``(batch_size, seq_len,
    d_model)`` tokens are split into ``(batch_size, n_head, seq_len,
    head_dim)`` without any projection, matching the kernel-level isolation
    used by the SDPA backend dispatch benchmark.

    Args:
        batch_size: Batch dimension.
        seq_len: Number of tokens.
        head_dim: Per-head embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns (a leading
            non-spatial column is prepended, as in ``make_inputs``).
        device: torch device for the tensors.
        dtype: torch dtype of the tensors.
        seed: Base random seed for the query draw.

    Returns:
        Tuple ``(query, key, value, coords)`` with Q/K/V shaped
        ``(batch_size, n_head, seq_len, head_dim)`` and coords shaped
        ``(batch_size, seq_len, coord_dim + 1)``.
    """
    d_model = head_dim * n_head
    query_tokens, coords = make_inputs(
        batch_size, seq_len, d_model, coord_dim + 1, device, dtype, seed,
    )
    key_tokens, _ = make_inputs(
        batch_size, seq_len, d_model, coord_dim + 1, device, dtype, seed + 1,
    )
    value_tokens, _ = make_inputs(
        batch_size, seq_len, d_model, coord_dim + 1, device, dtype, seed + 2,
    )

    def split_heads(tokens):
        """Reshape (B, N, H*d) tokens to (B, H, N, d) without projection."""
        return tokens.view(
            batch_size, seq_len, n_head, head_dim,
        ).transpose(1, 2)

    return (
        split_heads(query_tokens), split_heads(key_tokens),
        split_heads(value_tokens), coords,
    )


def make_cached_dist_module(coord_dim, head_dim, n_head, dist_mode,
                            positional_mode, device, dtype):
    """Instantiate a ``CachedDistAttention`` layer for mask construction.

    The embedding dimension is irrelevant to ``_build_mask`` (the mask shape
    depends only on ``n_head`` and the sequence length); it is set to
    ``head_dim * n_head`` so the module also works as the end-to-end
    ``dense_masked`` anchor at the requested head dimension.

    Args:
        coord_dim: Number of spatial coordinate columns.
        head_dim: Per-head embedding dimension.
        n_head: Number of attention heads.
        dist_mode: Distance decay mode ("v0" or "v1").
        positional_mode: Positional encoding mode ("none", "bias", "rope").
        device: torch device for the module.
        dtype: torch dtype for the module parameters.

    Returns:
        The instantiated and moved module (eval mode is not required; the
        mask construction is dropout-free).
    """
    return CachedDistAttention(
        coord_dim, head_dim * n_head, n_head,
        mode=positional_mode, attn_dist_mode=dist_mode,
    ).to(device, dtype)


def compute_cached_dist_2d(coords, dtype):
    """Pre-compute the 2D spatial distance matrix with the sweep recipe.

    Mirrors ``build_closure`` in ``benchmark_sweep.py``: fp32 ``cdist`` over
    the trailing spatial coordinate columns, cast to the working dtype, so
    ``dist_cache=on`` stages see exactly the tensor production passes to
    ``CachedDistAttention.forward``.

    Args:
        coords: Coordinate tensor ``(batch_size, seq_len, coord_dim + 1)``.
        dtype: Working dtype the matrix is cast to.

    Returns:
        Distance matrix ``(batch_size, seq_len, seq_len)`` in ``dtype``.
    """
    return torch.cdist(
        coords[..., 1:].float(), coords[..., 1:].float(),
    ).to(dtype)


def make_forced_sdpa_callable(backend, query, key, value, attn_mask=None,
                              is_causal=False):
    """Build a zero-argument closure running one SDPA call under ``backend``.

    The ``sdpa_kernel`` context is entered inside the closure so the harness
    timer times the forced-backend call itself; the context-manager overhead
    is identical across compared arms (see module docstring).

    Args:
        backend: ``SDPBackend`` enum member to force.
        query: Query tensor ``(batch_size, n_head, seq_len, head_dim)``.
        key: Key tensor, same shape as ``query``.
        value: Value tensor, same shape as ``query``.
        attn_mask: Optional additive attention mask passed through to SDPA.
        is_causal: Whether to pass ``is_causal=True`` (procedural
            lower-triangular mask generated inside the kernel; mutually
            exclusive with a meaningful ``attn_mask``).

    Returns:
        A closure executing the forced-backend SDPA call once.
    """
    def forced_sdpa():
        with sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, is_causal=is_causal,
            )
    return forced_sdpa


def make_auto_sdpa_callable(query, key, value, attn_mask=None):
    """Build a zero-argument closure running one unforced SDPA call.

    The default dispatcher regime (no backend forcing) is preserved, matching
    production; the profiler capture on this closure records which backend
    the dispatcher actually selected.

    Args:
        query: Query tensor ``(batch_size, n_head, seq_len, head_dim)``.
        key: Key tensor, same shape as ``query``.
        value: Value tensor, same shape as ``query``.
        attn_mask: Optional additive attention mask passed through to SDPA.

    Returns:
        A closure executing the auto-dispatch SDPA call once.
    """
    def auto_sdpa():
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask,
        )
    return auto_sdpa


def capture_cuda_kernel_names(timed_callable, device):
    """Run ``timed_callable`` once under the CUDA profiler; return kernel names.

    Args:
        timed_callable: Zero-argument callable executing the GPU work.
        device: torch device the run executes on; a CPU run cannot launch
            CUDA kernels, so a placeholder list is returned instead of
            profiling an empty window.

    Returns:
        List of CUDA kernel-name strings observed during the call; a
        one-element placeholder list when the run device is not CUDA.
    """
    if device.type != "cuda":
        return ["(cpu run: no CUDA kernels)"]
    with profile(activities=[ProfilerActivity.CUDA]) as profiler_run:
        timed_callable()
        torch.cuda.synchronize()
    return [
        event.key for event in profiler_run.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


def summarize_kernel_names(kernel_names, max_names=3):
    """Join the first ``max_names`` kernel names into one CSV cell string.

    Args:
        kernel_names: List of kernel-name strings from the profiler capture.
        max_names: Maximum number of names kept (cells stay readable).

    Returns:
        Semicolon-joined kernel names, or an empty string for no captures.
    """
    return ";".join(kernel_names[:max_names])


def build_result_row(stage, method, seq_len, head_dim, time_s, time_std_s,
                     memory_mb, status, dist_mode="", dist_cache="",
                     backend_label="", kernel_name=""):
    """Build one CSV row in the decomposition schema.

    Successful rows carry formatted timing/memory columns and an empty
    ``error`` cell; failed rows leave the timing columns empty and carry the
    harness status string (``"oom"`` or ``"err: ..."``) in ``error``,
    mirroring the sweep row conventions.

    Args:
        stage: Stage label ("A_backend_census", "B_kernel_intrinsic_nomask",
            "C_mask_build_isolated", "D_mask_carriage",
            "D_end_to_end_anchor").
        method: Method label within the stage.
        seq_len: Sequence length of the configuration.
        head_dim: Per-head dimension of the configuration.
        time_s: Mean time in seconds (unused when ``status != "ok"``).
        time_std_s: Time std across repeats in seconds.
        memory_mb: Incremental peak memory in MiB.
        status: Harness status ("ok", "oom", or "err: ...").
        dist_mode: Distance decay mode for mask-building stages.
        dist_cache: ``"on"``/``"off"`` for mask-building stages.
        backend_label: Forced backend enum name, or ``"auto(unforced)"``.
        kernel_name: Semicolon-joined profiler kernel names.

    Returns:
        List of CSV cell values ordered per ``RESULT_HEADER``.
    """
    if status == "ok":
        return [
            stage, method, seq_len, head_dim,
            f"{time_s * 1000:.3f}", f"{time_std_s * 1000:.3f}",
            f"{memory_mb:.1f}", dist_mode, dist_cache, backend_label,
            kernel_name, "",
        ]
    return [
        stage, method, seq_len, head_dim, "", "", "",
        dist_mode, dist_cache, backend_label, kernel_name, status,
    ]


def run_backend_census(batch_size, seq_len, head_dims, n_head, coord_dim,
                       device, dtype, seed, dist_mode, dist_cache):
    """Stage A: record dispatched kernel names per forced backend.

    For each head dimension, one unmasked and one additive-masked SDPA call
    is probed per census backend.  A backend that refuses the configuration
    (e.g. flash/FA2 at ``d_head=40``, or any backend with a mask it cannot
    consume) is recorded with its error instead of silently disappearing.

    Args:
        batch_size: Batch dimension.
        seq_len: Census sequence length (one representative shape).
        head_dims: Per-head dimensions to probe.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns.
        device: torch device.
        dtype: Working dtype.
        seed: Base random seed.
        dist_mode: Distance decay mode for the probe mask.
        dist_cache: ``"on"``/``"off"`` mask regime for the probe mask.

    Returns:
        List of census CSV rows.
    """
    rows = []
    for head_dim in head_dims:
        query, key, value, coords = make_head_split_inputs(
            batch_size, seq_len, head_dim, n_head, coord_dim, device, dtype,
            seed,
        )
        cached_dist_module = make_cached_dist_module(
            coord_dim, head_dim, n_head, dist_mode, "none", device, dtype,
        )
        dist_2d = (
            compute_cached_dist_2d(coords, dtype) if dist_cache == "on"
            else None
        )
        probe_mask = cached_dist_module._build_mask(
            batch_size, seq_len, coords, None, dist_2d,
            device=device, dtype=dtype,
        )
        for backend in CENSUS_BACKENDS:
            for method_name, attn_mask in (
                ("unmasked_probe", None), ("masked_probe", probe_mask),
            ):
                probe_callable = make_forced_sdpa_callable(
                    backend, query, key, value, attn_mask=attn_mask,
                )
                try:
                    probe_callable()
                except Exception as probe_error:
                    rows.append(build_result_row(
                        "A_backend_census", method_name, seq_len, head_dim,
                        0.0, 0.0, 0.0, f"err: {probe_error}",
                        dist_mode, dist_cache, backend.name,
                    ))
                    print(
                        f"  [census] {backend.name:<18s} {method_name:<14s}"
                        f" d={head_dim:<3d} -> refused: {probe_error}"
                    )
                    continue
                kernel_names = capture_cuda_kernel_names(probe_callable, device)
                rows.append(build_result_row(
                    "A_backend_census", method_name, seq_len, head_dim,
                    0.0, 0.0, 0.0, "ok",
                    dist_mode, dist_cache, backend.name,
                    summarize_kernel_names(kernel_names),
                ))
                print(
                    f"  [census] {backend.name:<18s} {method_name:<14s}"
                    f" d={head_dim:<3d} -> {summarize_kernel_names(kernel_names)}"
                )
    return rows


def run_kernel_intrinsic_pair(batch_size, seq_lens, head_dims, n_head,
                              coord_dim, device, dtype, seed, warmup,
                              min_run_time_s, repeats):
    """Stage B: time the unmasked FLASH vs CUDNN kernel pair.

    Identical seeded Q/K/V per configuration; ``attn_mask=None``; each arm
    forced to its backend.  Successful rows additionally record the profiler
    kernel names, so the kernel-intrinsic comparison is backed by dispatch
    evidence.

    Args:
        batch_size: Batch dimension.
        seq_lens: Sequence lengths to sweep.
        head_dims: Per-head dimensions to sweep.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns (unused by this
            stage's SDPA calls; kept for the shared input recipe).
        device: torch device.
        dtype: Working dtype.
        seed: Base random seed.
        warmup: Warmup iterations for the harness timer.
        min_run_time_s: Minimum autorange run time in seconds.
        repeats: Timing repetitions per configuration.

    Returns:
        List of stage-B CSV rows.
    """
    rows = []
    timed_arms = [
        (SDPBackend.FLASH_ATTENTION, "flash_nomask"),
        (SDPBackend.CUDNN_ATTENTION, "cudnn_nomask"),
    ]
    for head_dim in head_dims:
        for seq_len in seq_lens:
            query, key, value, _ = make_head_split_inputs(
                batch_size, seq_len, head_dim, n_head, coord_dim,
                device, dtype, seed,
            )
            for backend, method_name in timed_arms:
                time_s, time_std_s, memory_mb, status = try_bench(
                    make_forced_sdpa_callable, backend, query, key, value,
                    warmup=warmup, min_run_time=min_run_time_s,
                    repeats=repeats,
                )
                kernel_name = ""
                if status == "ok":
                    kernel_name = summarize_kernel_names(
                        capture_cuda_kernel_names(
                            make_forced_sdpa_callable(
                                backend, query, key, value,
                            ), device,
                        )
                    )
                rows.append(build_result_row(
                    "B_kernel_intrinsic_nomask", method_name, seq_len,
                    head_dim, time_s, time_std_s, memory_mb, status,
                    backend_label=backend.name, kernel_name=kernel_name,
                ))
                time_ms_display = f"{time_s * 1000:.3f}" if time_s >= 0 else "-"
                print(
                    f"  [B] {method_name:<14s} N={seq_len:<5d}"
                    f" d={head_dim:<3d} {time_ms_display:>9s} ms"
                    f"  [{status if status == 'ok' else 'FAILED'}]"
                )
    return rows


def make_mask_build_callable(cached_dist_module, batch_size, seq_len,
                             coords, dist_2d, device, dtype):
    """Build a closure constructing the additive mask via ``_build_mask``.

    Args:
        cached_dist_module: ``CachedDistAttention`` instance owning the
            cutoff/decay configuration.
        batch_size: Batch dimension.
        seq_len: Number of tokens.
        coords: Coordinate tensor ``(batch_size, seq_len, coord_dim + 1)``.
        dist_2d: Pre-computed distance matrix for ``dist_cache=on``, else
            ``None`` (the construction then computes distances itself).
        device: Device for the returned mask.
        dtype: Dtype for the returned mask.

    Returns:
        A closure whose call builds one fresh mask tensor.
    """
    def build_mask():
        return cached_dist_module._build_mask(
            batch_size, seq_len, coords, None, dist_2d,
            device=device, dtype=dtype,
        )
    return build_mask


def run_mask_build_isolation(batch_size, seq_lens, n_head, coord_dim,
                             head_dim, device, dtype, seed, warmup,
                             min_run_time_s, repeats):
    """Stage C: time ONLY the additive mask construction.

    Runs all four ``(dist_mode, dist_cache)`` combinations; the head
    dimension is irrelevant to the mask shape, so the stage runs once per
    sequence length rather than once per ``(N, d_head)`` pair.

    Args:
        batch_size: Batch dimension.
        seq_lens: Sequence lengths to sweep.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns.
        head_dim: Head dimension used to size the module (mask-independent).
        device: torch device.
        dtype: Working dtype.
        seed: Base random seed.
        warmup: Warmup iterations for the harness timer.
        min_run_time_s: Minimum autorange run time in seconds.
        repeats: Timing repetitions per configuration.

    Returns:
        List of stage-C CSV rows.
    """
    rows = []
    for dist_mode in ("v0", "v1"):
        for dist_cache in ("on", "off"):
            cached_dist_module = make_cached_dist_module(
                coord_dim, head_dim, n_head, dist_mode, "none", device, dtype,
            )
            for seq_len in seq_lens:
                _, coords = make_inputs(
                    batch_size, seq_len, head_dim * n_head, coord_dim + 1,
                    device, dtype, seed,
                )
                dist_2d = (
                    compute_cached_dist_2d(coords, dtype)
                    if dist_cache == "on" else None
                )
                time_s, time_std_s, memory_mb, status = try_bench(
                    make_mask_build_callable, cached_dist_module, batch_size,
                    seq_len, coords, dist_2d, device, dtype,
                    warmup=warmup, min_run_time=min_run_time_s,
                    repeats=repeats,
                )
                rows.append(build_result_row(
                    "C_mask_build_isolated", "cached_dist_mask_build",
                    seq_len, head_dim, time_s, time_std_s, memory_mb, status,
                    dist_mode, dist_cache, backend_label="n/a (construction only)",
                ))
                time_ms_display = f"{time_s * 1000:.3f}" if time_s >= 0 else "-"
                print(
                    f"  [C] dist={dist_mode} cache={dist_cache:<3s}"
                    f" N={seq_len:<5d} {time_ms_display:>9s} ms"
                    f"  [{status if status == 'ok' else 'FAILED'}]"
                )
    return rows


def run_mask_carriage_and_end_to_end(batch_size, seq_lens, head_dims, n_head,
                                     coord_dim, device, dtype, seed,
                                     positional_mode, dist_mode, dist_cache,
                                     d_model_end_to_end, warmup,
                                     min_run_time_s, repeats):
    """Stage D: mask-carriage timings plus the end-to-end anchor rows.

    Kernel-level arms pass a pre-built additive mask (construction outside
    the timed region) to (i) forced cuDNN, (ii) forced mem-efficient, and
    (iii) the unforced auto dispatcher.  The end-to-end arms time the
    unmodified ``DenseFlashAttention`` and ``CachedDistAttention`` modules
    exactly as the sweep harness invokes them (``layer(x, x, x, ...)``),
    providing the anchor the decomposition is closed against.

    Args:
        batch_size: Batch dimension.
        seq_lens: Sequence lengths to sweep.
        head_dims: Per-head dimensions for the kernel-level arms.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns.
        device: torch device.
        dtype: Working dtype.
        seed: Base random seed.
        positional_mode: Positional encoding mode for the end-to-end
            ``dense_masked`` module.
        dist_mode: Distance decay mode.
        dist_cache: ``"on"``/``"off"`` regime for mask construction and the
            end-to-end ``dense_masked`` arm.
        d_model_end_to_end: Embedding dimension of the end-to-end modules
            (``--d``; its ``d_model / n_head`` head dimension is recorded in
            the ``d_head`` column).
        warmup: Warmup iterations for the harness timer.
        min_run_time_s: Minimum autorange run time in seconds.
        repeats: Timing repetitions per configuration.

    Returns:
        List of stage-D CSV rows.
    """
    rows = []
    cached_dist_module = make_cached_dist_module(
        coord_dim, head_dims[0], n_head, dist_mode, positional_mode,
        device, dtype,
    )
    carriage_arms = [
        (SDPBackend.CUDNN_ATTENTION, "cudnn_masked_prebuilt"),
        (SDPBackend.EFFICIENT_ATTENTION, "efficient_masked_prebuilt"),
        (None, "auto_masked_dispatch"),
    ]
    for head_dim in head_dims:
        for seq_len in seq_lens:
            query, key, value, coords = make_head_split_inputs(
                batch_size, seq_len, head_dim, n_head, coord_dim,
                device, dtype, seed,
            )
            dist_2d = (
                compute_cached_dist_2d(coords, dtype) if dist_cache == "on"
                else None
            )
            prebuilt_mask = cached_dist_module._build_mask(
                batch_size, seq_len, coords, None, dist_2d,
                device=device, dtype=dtype,
            )
            for backend, method_name in carriage_arms:
                if backend is None:
                    time_s, time_std_s, memory_mb, status = try_bench(
                        make_auto_sdpa_callable, query, key, value,
                        attn_mask=prebuilt_mask,
                        warmup=warmup, min_run_time=min_run_time_s,
                        repeats=repeats,
                    )
                    backend_label = "auto(unforced)"
                else:
                    time_s, time_std_s, memory_mb, status = try_bench(
                        make_forced_sdpa_callable, backend, query, key, value,
                        attn_mask=prebuilt_mask,
                        warmup=warmup, min_run_time=min_run_time_s,
                        repeats=repeats,
                    )
                    backend_label = backend.name
                kernel_name = ""
                if status == "ok":
                    auto_callable = (
                        make_auto_sdpa_callable(
                            query, key, value, attn_mask=prebuilt_mask,
                        )
                        if backend is None
                        else make_forced_sdpa_callable(
                            backend, query, key, value, attn_mask=prebuilt_mask,
                        )
                    )
                    kernel_name = summarize_kernel_names(
                        capture_cuda_kernel_names(auto_callable, device)
                    )
                rows.append(build_result_row(
                    "D_mask_carriage", method_name, seq_len, head_dim,
                    time_s, time_std_s, memory_mb, status,
                    dist_mode, dist_cache, backend_label, kernel_name,
                ))
                time_ms_display = f"{time_s * 1000:.3f}" if time_s >= 0 else "-"
                print(
                    f"  [D] {method_name:<26s} N={seq_len:<5d}"
                    f" d={head_dim:<3d} {time_ms_display:>9s} ms"
                    f"  [{status if status == 'ok' else 'FAILED'}]"
                )

    end_to_end_head_dim = d_model_end_to_end // n_head
    dense_flash_module = DenseFlashAttention(
        d_model_end_to_end, n_head,
    ).to(device, dtype)
    cached_dist_end_to_end = make_cached_dist_module(
        coord_dim, end_to_end_head_dim, n_head, dist_mode, positional_mode,
        device, dtype,
    )
    for seq_len in seq_lens:
        query_tokens, coords = make_inputs(
            batch_size, seq_len, d_model_end_to_end, coord_dim + 1,
            device, dtype, seed,
        )
        dist_2d = (
            compute_cached_dist_2d(coords, dtype) if dist_cache == "on"
            else None
        )
        for method_name, end_to_end_callable in (
            ("dense_flash_e2e",
             lambda: dense_flash_module(query_tokens, query_tokens, query_tokens)),
            ("dense_masked_e2e",
             lambda: cached_dist_end_to_end(
                 query_tokens, query_tokens, query_tokens,
                 coords=coords, dist_2d=dist_2d,
             )),
        ):
            time_s, time_std_s, memory_mb, status = try_bench(
                lambda: end_to_end_callable,
                warmup=warmup, min_run_time=min_run_time_s, repeats=repeats,
            )
            kernel_name = ""
            if status == "ok":
                kernel_name = summarize_kernel_names(
                    capture_cuda_kernel_names(end_to_end_callable, device)
                )
            rows.append(build_result_row(
                "D_end_to_end_anchor", method_name, seq_len,
                end_to_end_head_dim, time_s, time_std_s, memory_mb, status,
                dist_mode, dist_cache, backend_label="auto(unforced)",
                kernel_name=kernel_name,
            ))
            time_ms_display = f"{time_s * 1000:.3f}" if time_s >= 0 else "-"
            print(
                f"  [D] {method_name:<26s} N={seq_len:<5d}"
                f" d={end_to_end_head_dim:<3d} {time_ms_display:>9s} ms"
                f"  [{status if status == 'ok' else 'FAILED'}]"
            )
    return rows


def run_causal_vs_mask_dtype(batch_size, seq_lens, head_dims, n_head,
                             coord_dim, device, dtype, seed,
                             dist_mode, dist_cache, warmup,
                             min_run_time_s, repeats):
    """Stage E: procedural causal masks vs prebuilt float/bool masks.

    Diagnostic arms probing where the masked-dense cost enters, complementing
    stage D: (i) ``is_causal=True`` needs no N x N mask input at all (the
    mask pattern is generated inside the kernel), timed for both FLASH and
    CUDNN; (ii) a prebuilt bool mask pairs with stage D's prebuilt float
    mask to isolate the dispatcher's dtype handling for bool masks.  The
    causal pattern is NOT the production spatial cutoff (which is
    distance-based): these arms bound the mask-carriage overhead, they are
    not a production configuration.  Prebuilt masks are allocated before the
    harness baseline, so the incremental peak memory isolates kernel-side
    allocations from caller-owned mask storage.

    Args:
        batch_size: Batch dimension.
        seq_lens: Sequence lengths to sweep.
        head_dims: Per-head dimensions for the kernel-level arms.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate columns.
        device: torch device.
        dtype: Working dtype.
        seed: Base random seed.
        dist_mode: Distance decay mode for mask construction.
        dist_cache: ``"on"``/``"off"`` regime for mask construction.
        warmup: Warmup iterations for the harness timer.
        min_run_time_s: Minimum autorange run time in seconds.
        repeats: Timing repetitions per configuration.

    Returns:
        List of stage-E CSV rows.
    """
    rows = []
    cached_dist_module = make_cached_dist_module(
        coord_dim, head_dims[0], n_head, dist_mode, "none",
        device, dtype,
    )
    for head_dim in head_dims:
        for seq_len in seq_lens:
            query, key, value, coords = make_head_split_inputs(
                batch_size, seq_len, head_dim, n_head, coord_dim,
                device, dtype, seed,
            )
            dist_2d = (
                compute_cached_dist_2d(coords, dtype) if dist_cache == "on"
                else None
            )
            prebuilt_float = cached_dist_module._build_mask(
                batch_size, seq_len, coords, None, dist_2d,
                device=device, dtype=dtype,
            )
            prebuilt_bool = torch.isfinite(prebuilt_float)
            arms = (
                ("flash_causal_procedural", SDPBackend.FLASH_ATTENTION,
                 None, True),
                ("cudnn_causal_procedural", SDPBackend.CUDNN_ATTENTION,
                 None, True),
                ("cudnn_bool_prebuilt", SDPBackend.CUDNN_ATTENTION,
                 prebuilt_bool, False),
            )
            for method_name, backend, attn_mask, is_causal in arms:
                time_s, time_std_s, memory_mb, status = try_bench(
                    make_forced_sdpa_callable, backend, query, key, value,
                    attn_mask=attn_mask, is_causal=is_causal,
                    warmup=warmup, min_run_time=min_run_time_s,
                    repeats=repeats,
                )
                kernel_name = ""
                if status == "ok":
                    kernel_name = summarize_kernel_names(
                        capture_cuda_kernel_names(
                            make_forced_sdpa_callable(
                                backend, query, key, value,
                                attn_mask=attn_mask, is_causal=is_causal,
                            ),
                            device,
                        )
                    )
                rows.append(build_result_row(
                    "E_causal_vs_mask_dtype", method_name, seq_len, head_dim,
                    time_s, time_std_s, memory_mb, status,
                    dist_mode, dist_cache, backend.name, kernel_name,
                ))
                time_ms_display = f"{time_s * 1000:.3f}" if time_s >= 0 else "-"
                print(
                    f"  [E] {method_name:<26s} N={seq_len:<5d}"
                    f" d={head_dim:<3d} {time_ms_display:>9s} ms"
                    f"  [{status if status == 'ok' else 'FAILED'}]"
                )
    return rows


def write_env_manifest(out_path, args, device, dtype):
    """Write the environment/CLI manifest JSON next to the results CSV.

    Records everything needed to reproduce the run: library versions, GPU
    identity, SDPA backend availability flags, and the parsed CLI
    configuration.

    Args:
        out_path: Results CSV path; the manifest is written as
            ``<stem>_env.json`` beside it.
        args: Parsed CLI namespace.
        device: torch device the run executed on.
        dtype: Working dtype of the run.
    """
    backend_flags = {}
    if torch.cuda.is_available():
        backend_flags = {
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled":
                torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        }
    manifest = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None
        ),
        "device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"
        ),
        "device_capability": (
            str(torch.cuda.get_device_capability())
            if torch.cuda.is_available() else None
        ),
        "backend_flags": backend_flags,
        "dtype": str(dtype),
        "device": str(device),
        "cli_config": {key: value for key, value in vars(args).items()},
    }
    manifest_path = out_path.with_name(f"{out_path.stem}_env.json")
    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    print(f"Wrote environment manifest -> {manifest_path}")


def main():
    """Run the mask-overhead decomposition benchmark and write CSV + manifest."""
    parser = argparse.ArgumentParser(
        description=(
            "cuDNN flash vs FA2 mask-overhead decomposition: "
            "kernel census, unmasked kernel pair, isolated mask construction, "
            "mask carriage, causal-vs-mask-dtype diagnostic, and end-to-end "
            "anchor."
        ),
    )
    add_common_args(parser)
    parser.add_argument(
        "--seq-lens", default="128,256,512,1024,2048,4096,8192",
        help="Comma-separated sequence lengths to sweep",
    )
    parser.add_argument(
        "--d-heads", default="40,64",
        help=(
            "Comma-separated per-head dimensions for the kernel-level "
            "stages (40 = Trackastra's real head dim, where flash/FA2 "
            "cannot dispatch; 64 = both kernels dispatchable)"
        ),
    )
    parser.add_argument(
        "--dist-mode", default="v1", choices=["v0", "v1"],
        help="Distance decay mode for mask construction and dense_masked",
    )
    parser.add_argument(
        "--dist-cache", default="on", choices=["on", "off"],
        help="Whether dist_2d is pre-computed (on) or rebuilt per call (off)",
    )
    parser.add_argument(
        "--mode", default="none", choices=["none", "bias", "rope"],
        help="Positional encoding mode for the end-to-end dense_masked module",
    )
    parser.add_argument(
        "--coord-dim", type=int, default=2,
        help="Number of spatial coordinate columns",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    seq_lens = parse_int_list(args.seq_lens)
    head_dims = parse_int_list(args.d_heads)
    min_run_time_s = args.rep / 1000.0

    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"Head dims (kernel stages): {head_dims}  N = {seq_lens}")
    print(
        f"dist-mode={args.dist_mode}  dist-cache={args.dist_cache}"
        f"  mode={args.mode}  coord-dim={args.coord_dim}  seed={args.seed}"
    )
    print(f"[Timing] warmup={args.warmup} rep={args.rep} repeats={args.repeats}")

    out_path = Path(args.out)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print("\n=== Stage A: backend census (kernel-name evidence) ===")
    rows += run_backend_census(
        batch_size=1, seq_len=1024, head_dims=head_dims, n_head=args.nhead,
        coord_dim=args.coord_dim, device=device, dtype=dtype, seed=args.seed,
        dist_mode=args.dist_mode, dist_cache=args.dist_cache,
    )

    print("\n=== Stage B: kernel-intrinsic pair (unmasked) ===")
    rows += run_kernel_intrinsic_pair(
        batch_size=1, seq_lens=seq_lens, head_dims=head_dims,
        n_head=args.nhead, coord_dim=args.coord_dim, device=device,
        dtype=dtype, seed=args.seed, warmup=args.warmup,
        min_run_time_s=min_run_time_s, repeats=args.repeats,
    )

    print("\n=== Stage C: isolated mask construction ===")
    rows += run_mask_build_isolation(
        batch_size=1, seq_lens=seq_lens, n_head=args.nhead,
        coord_dim=args.coord_dim, head_dim=head_dims[0], device=device,
        dtype=dtype, seed=args.seed, warmup=args.warmup,
        min_run_time_s=min_run_time_s, repeats=args.repeats,
    )

    print("\n=== Stage D: mask carriage + end-to-end anchor ===")
    rows += run_mask_carriage_and_end_to_end(
        batch_size=1, seq_lens=seq_lens, head_dims=head_dims,
        n_head=args.nhead, coord_dim=args.coord_dim, device=device,
        dtype=dtype, seed=args.seed, positional_mode=args.mode,
        dist_mode=args.dist_mode, dist_cache=args.dist_cache,
        d_model_end_to_end=args.d, warmup=args.warmup,
        min_run_time_s=min_run_time_s, repeats=args.repeats,
    )

    print("\n=== Stage E: causal vs prebuilt float/bool masks ===")
    rows += run_causal_vs_mask_dtype(
        batch_size=1, seq_lens=seq_lens, head_dims=head_dims,
        n_head=args.nhead, coord_dim=args.coord_dim, device=device,
        dtype=dtype, seed=args.seed, dist_mode=args.dist_mode,
        dist_cache=args.dist_cache, warmup=args.warmup,
        min_run_time_s=min_run_time_s, repeats=args.repeats,
    )

    write_results_csv(rows, out_path, RESULT_HEADER)
    print(f"\nWrote {len(rows)} rows -> {out_path}")
    write_env_manifest(out_path, args, device, dtype)


if __name__ == "__main__":
    main()
