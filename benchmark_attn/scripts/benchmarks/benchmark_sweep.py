"""Consolidated attention-method benchmark sweep: methods × layers × N × K → CSV.

For every combination of (method, L, N, K) the sweep:
  1. builds synthetic inputs with ``make_inputs``,
  2. computes KNN indices with ``compute_knn_indices`` when the method needs
     them (``MethodEntry.needs_knn`` from the harness registry),
  3. builds a closure that chains ``L`` freshly-instantiated layers of the
     method's class (resolved from the registry via ``resolve_class``),
  4. times the closure through the harness ``try_bench`` (which wraps
     ``measure`` and translates OOM / other failures into a status string),
  5. appends a CSV row ``[method, L, N, K, time_ms, memory_mb, error]``.

Methods are selected with ``--methods`` (registry keys such as
``sparse_gather_sdpa``, ``sparse_gather_fused``, ``sparse_gather_matmul``,
``sparse_mask``, ``dense_flash``, ``dense_masked``, ``nsa``,
``sparse_gather_relpos``, ``minimax``) and
default to every registered method (``list_methods``).  K is only meaningful
for KNN-based methods; all other methods are emitted with ``K=0``.

Usage:
    python scripts/benchmarks/benchmark_sweep.py \\
        --methods sparse_gather_sdpa,sparse_gather_fused,sparse_gather_matmul,sparse_mask,dense_flash,dense_masked,nsa,sparse_gather_relpos,minimax \\
        --Ns 128,256,512,1024,2048,4096,8192 --Ks 4,16,64 --layers 1,4 \\
        --mode none --dist-mode v1 --warmup 5 --rep 30 \\
        --d 320 --nhead 8 --coord-dim 2 --seed 42 \\
        --out results/sweep_results.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import gc
import torch

from src.bench import (
    measure, try_bench, make_inputs, compute_knn_indices,
    METHOD_REGISTRY, resolve_class, list_methods,
    write_results_csv, add_common_args, parse_int_list,
)
from src.attention_modules import (
    RelativePositionalAttention, SparseGatherRelativePositionalAttention,
    SparseGatherAttention, SparseGatherFusedAttention, SparseGatherMatmulAttention,
    SparseMaskAttention, DenseFlashAttention, NSASparseAttention, MiniMaxSparseAttention,
    SparseSoftmaxAttention,
)

# How each registered method's ``forward`` is invoked.  The gather/mask
# variants take ``(query, key, value, knn_indices, coords)``, the relative
# positional variant takes ``(query, key, value, coords, knn_indices)``, the
# dense masked variant takes ``(query, key, value, coords)``, and everything
# else (dense_flash, nsa, minimax) is called as ``(query, key, value)``.
_METHOD_FORWARD_STYLES = {
    "sparse_gather_sdpa": "knn",
    "sparse_gather_fused": "knn",
    "sparse_gather_matmul": "knn",
    "sparse_mask": "knn",
    "sparse_softmax": "knn",
    "sparse_gather_relpos": "relpos",
    "dense_masked": "masked",
    "dense_flash": "qkv",
    "nsa": "qkv",
    "minimax": "qkv",
    "flex_attn": "knn",
}


def _make_layer(method_key, attention_class, seq_len, d_model, n_head, knn_neighbors,
                coord_dim, mode, dist_mode, device, dtype):
    """Instantiate a single attention layer of ``method_key`` on ``device``.

    The constructor signature differs per method family, so the argument
    layout is dispatched on ``method_key`` here.  The class
    ``attention_class`` comes from the harness registry
    (``resolve_class``) so it always matches the registry key.

    Args:
        method_key: Registry key of the method.
        attention_class: Attention module class to instantiate.
        seq_len: Sequence length (used to bound MiniMax block selection).
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        knn_neighbors: Number of KNN neighbours (KNN-based methods only).
        coord_dim: Number of spatial coordinate dimensions.
        mode: Positional encoding mode ("none", "bias", "rope").
        dist_mode: Distance decay mode ("v0" or "v1") for dense_masked.
        device: torch device for the layer.
        dtype: torch dtype for the layer.

    Returns:
        An instantiated layer moved to ``(device, dtype)``.
    """
    if method_key == "dense_masked":
        return attention_class(coord_dim, d_model, n_head, mode=mode,
                    attn_dist_mode=dist_mode).to(device, dtype)
    if method_key == "sparse_gather_relpos":
        return attention_class(coord_dim, d_model, n_head, mode=mode,
                    knn_neighbors=knn_neighbors).to(device, dtype)
    if METHOD_REGISTRY[method_key].needs_knn:
        # sparse_gather_sdpa / sparse_gather_fused / sparse_gather_matmul /
        # sparse_mask / sparse_softmax
        return attention_class(d_model, n_head, knn_neighbors=knn_neighbors,
                    coord_dim=coord_dim, mode=mode).to(device, dtype)
    if method_key == "flex_attn":
        # FlexAttention: lazy block-mask build + compiled flex_attention.
        # The module builds the mask on first forward (coords fixed per
        # config, cached); the measure() warmup absorbs the build + the
        # torch.compile cost so the first config is slow, later ones fast.
        return attention_class(d_model, n_head, knn_neighbors=knn_neighbors,
                    coord_dim=coord_dim, mode=mode).to(device, dtype)
    if method_key == "minimax":
        # Mirror benchmark_full.py: cap the selected block count at the number
        # of blocks actually present, so small N never exceeds topk's k.
        block_size = 128
        num_selected_blocks = max(1, min(4, (seq_len + block_size - 1) // block_size - 1))
        return attention_class(d_model, n_head, block_size=block_size,
                    num_selected_blocks=num_selected_blocks, mode="none").to(device, dtype)
    # dense_flash / nsa: constructed from (embed_dim, n_head) only.
    return attention_class(d_model, n_head).to(device, dtype)


def _forward_layer(layer, x, coords, knn_idx, forward_style, dist_2d=None):
    """Invoke one attention layer with the argument pattern it expects.

    Args:
        layer: An instantiated attention module.
        x: Token tensor of shape (batch_size, seq_len, d_model).
        coords: Coordinate tensor of shape (batch_size, seq_len, coord_dim + 1)
            or None for methods that do not use coordinates.
        knn_idx: KNN index tensor (batch_size, seq_len, K) or None.
        forward_style: One of the keys of ``_METHOD_FORWARD_STYLES``.
        dist_2d: Optional pre-computed 2D spatial distance matrix
            ``(batch_size, seq_len, seq_len)``.  Passed to layers that
            accept it (e.g. :class:`CachedDistAttention`).

    Returns:
        The layer's output tensor.
    """
    if forward_style == "knn":
        return layer(x, x, x, knn_indices=knn_idx, coords=coords)
    if forward_style == "relpos":
        return layer(x, x, x, coords=coords, knn_indices=knn_idx)
    if forward_style == "masked":
        return layer(x, x, x, coords=coords, dist_2d=dist_2d)
    if forward_style == "flex":
        # FlexAttention: module owns the block mask (lazy build, cached);
        # coords fixed per config so the mask build lands in warmup.
        return layer(x, x, x, coords=coords)
    return layer(x, x, x)


def build_closure(method_key, layer_count, seq_len, knn_neighbors,
                  batch_size, d_model, n_head, coord_dim, mode, dist_mode,
                  device, dtype, seed, with_knn=False, dist_cache="on"):
    """Build a zero-argument closure that runs ``layer_count`` chained layers.

    Creates the synthetic inputs with ``make_inputs``, computes the KNN
    indices with ``compute_knn_indices`` when the method requires them, and
    instantiates ``layer_count`` fresh layers chained over the same query
    tensor.  Intended to be passed to the harness ``try_bench``, which calls
    this builder and times the returned closure.

    When ``with_knn`` is ``True`` the KNN index computation is moved
    *inside* the timed closure so that the measurement reflects the
    per-forward-pass cost incurred during training (where
    :class:`trackastra.model.TrackingTransformer` recomputes
    ``cdist`` + ``topk`` on every call to ``forward``).  When
    ``with_knn`` is ``False`` (default) the indices are pre-computed
    once before timing, isolating the attention-kernel cost.

    Args:
        method_key: Registry key of the method.
        layer_count: Number of chained layers.
        seq_len: Sequence length.
        knn_neighbors: Number of KNN neighbours (ignored for non-KNN methods).
        batch_size: Batch dimension.
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate dimensions.
        mode: Positional encoding mode ("none", "bias", "rope").
        dist_mode: Distance decay mode ("v0" or "v1").
        device: torch device for inputs and layers.
        dtype: torch dtype for inputs and layers.
        seed: Random seed for input and parameter generation.
        with_knn: If ``True``, include KNN index computation inside the
            timed closure (training-realistic mode).
        dist_cache: "on" or "off". When "off", the 2D distance matrix
            (``dist_2d``) is computed inside the timed closure so every
            call pays the ``torch.cdist`` cost; when "on" it is computed
            once and amortised across the chained layers (current default).

    Returns:
        A closure whose return value is the output of the last layer.
    """
    query, coords = make_inputs(
        batch_size, seq_len, d_model, coord_dim + 1, device, dtype, seed
    )
    registry_entry = METHOD_REGISTRY[method_key]
    attention_class = resolve_class(registry_entry.class_path)

    # Pre-compute KNN indices unless with_knn is requested.
    layers = torch.nn.ModuleList([
        _make_layer(method_key, attention_class, seq_len, d_model, n_head, knn_neighbors,
                    coord_dim, mode, dist_mode, device, dtype)
        for _ in range(layer_count)
    ])
    forward_style = _METHOD_FORWARD_STYLES[method_key]

    # Pre-compute the 2D spatial distance matrix once, matching
    # TrackingTransformer.forward() which computes dist_2d once and
    # shares it across all L layers.  This avoids per-layer cdist.
    # Pre-compute KNN indices unless with_knn is requested or dist_cache=off
    # folds the KNN selection inside the timed closure (KNN methods only).
    knn_inside = with_knn or (dist_cache == "off" and registry_entry.needs_knn)
    if knn_inside:
        knn_idx = None  # Will be computed inside the closure.
    else:
        knn_idx = compute_knn_indices(coords, knn_neighbors) if registry_entry.needs_knn else None

    if dist_cache == "on":
        dist_2d = torch.cdist(coords[..., 1:].float(), coords[..., 1:].float()).to(dtype)
    else:
        dist_2d = None  # Each layer's forward computes distances from coords.

    if knn_inside:
        def closure():
            # Training-realistic: recompute KNN indices every forward pass.
            knn_idx_local = compute_knn_indices(coords, knn_neighbors)
            output = query
            for layer in layers:
                output = _forward_layer(layer, output, coords, knn_idx_local, forward_style, dist_2d)
            return output
    else:
        def closure():
            output = query
            for layer in layers:
                output = _forward_layer(layer, output, coords, knn_idx, forward_style, dist_2d)
            return output

    return closure


def _resolve_backend(method_key, mode):
    """Resolve the declared SDPA dispatch class for ``method_key``.

    Derivation rule: query ``torch.backends.cuda.flash_sdp_enabled()`` /
    ``mem_efficient_sdp_enabled()`` / ``cudnn_sdp_enabled()`` at run
    time, combined with the method's known dispatch behaviour per
    SPEC 0024.  This is a DECLARED dispatch class derived from backend
    availability flags + 0024 evidence, NOT a per-row profiler
    measurement (profiler evidence lives in FLEX_ENV_REPORT.md).

    Per-method mapping (mode-aware where the dispatch class depends on
    the run's ``--mode``):
      - sparse_gather_matmul: "matmul_cublas" (manual torch.matmul +
        F.softmax, no SDPA call at all — dispatches to cuBLAS,
        mode-independent)
      - sparse_gather_relpos: mode-dependent — mode=="bias" passes
        attn_mask=pos_bias(...) to SDPA (masked path → "cudnn_flash"
        if cudnn_sdp_enabled else "mem_efficient");
        mode=="none"/"rope" calls SDPA without a mask (unmasked →
        "flash" if flash_sdp_enabled else "mem_efficient")
      - sparse_gather_sdpa, sparse_gather_fused: same mode-dependence as
        sparse_gather_relpos (attn_mask=pos_bias(...) passed to SDPA
        when mode=="bias")
      - dense_flash: mode-independent — always calls SDPA without a
        mask → "flash" if flash_sdp_enabled else "mem_efficient"
      - dense_masked, sparse_mask: always pass an N×N mask to SDPA →
        "cudnn_flash" if cudnn_sdp_enabled else "mem_efficient"
      - flex_attn: "flex_compiled_triton"
      - nsa: "nsa_kernel"
      - minimax: "minimax_kernel"

    Args:
        method_key: Registry key of the method.
        mode: The sweep's ``--mode`` value ("none", "bias", or "rope").
            Methods whose forward passes ``attn_mask`` to SDPA when
            ``mode=="bias"`` dispatch to the masked class under that
            mode; under ``mode=="none"`` or ``mode=="rope"`` they call
            SDPA without a mask and dispatch to flash/mem_efficient.

    Returns:
        The declared dispatch class string.
    """
    flash = torch.backends.cuda.flash_sdp_enabled()
    mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
    cudnn = torch.backends.cuda.cudnn_sdp_enabled()

    # sparse_gather_matmul: manual torch.matmul + F.softmax, no SDPA call
    # at all — dispatches to cuBLAS.  Never claims an SDPA class.
    if method_key == "sparse_gather_matmul":
        return "matmul_cublas"

    # Mode-dependent gather methods: sparse_gather_sdpa, sparse_gather_fused,
    # sparse_gather_relpos all pass attn_mask=pos_bias(...) to SDPA when
    # mode=="bias" (same pattern); under mode=="none"/"rope" they call SDPA
    # without a mask (unmasked → flash).  Mode-aware because the dispatch
    # class depends on the run's --mode.
    if method_key in ("sparse_gather_sdpa", "sparse_gather_fused",
                      "sparse_gather_relpos"):
        if mode == "bias":
            return "cudnn_flash" if cudnn else "mem_efficient"
        return "flash" if flash else "mem_efficient"

    # dense_flash / sparse_gather_sdpa / sparse_gather_fused: unmasked SDPA
    # on gathered tensors (mode-independent — DenseFlashAttention has no
    # mode parameter, always calls SDPA without a mask).
    if method_key in ("dense_flash", "sparse_gather_sdpa", "sparse_gather_fused"):
        return "flash" if flash else "mem_efficient"

    # dense_masked / sparse_mask: N×N mask passed to SDPA (masked path).
    # On A500 (sm86)/torch 2.12 the masked path dispatches to the
    # mem_efficient (cutlass fmha) kernel; on H100 (sm90) it dispatches
    # to cudnn_flash.  Declare what actually runs so the probe checks
    # the right kernel family.
    if method_key in ("dense_masked", "sparse_mask"):
        if torch.cuda.get_device_capability() >= (9, 0):
            return "cudnn_flash" if cudnn else "mem_efficient"
        return "mem_efficient"

    if method_key == "flex_attn":
        return "flex_compiled_triton"
    if method_key == "nsa":
        return "nsa_kernel"
    if method_key == "minimax":
        return "minimax_kernel"
    return "unknown"


# ---------------------------------------------------------------------------
# Backend verification probe (SPEC 0022 §11 / ticket S22-T7).
#
# One-shot, per-method kernel-family verification executed before any timing
# row is produced, so a silent dispatcher fallback is caught instead of
# silently mislabelling rows.  The kernel families and their expected
# substrings come from ``benchmark_attn/docs/FLEX_ENV_REPORT.md`` (SPEC 0024
# H100 session body + A500 appendix):
#
#   flash
#     F.scaled_dot_product_attention with the flash backend active
#     (kernel family: "flash" — anything containing "flash" in the
#     CUDA kernel name, no competing mem-efficient family substring).
#   cudnn_flash
#     Masked SDPA dispatched via cuDNN on H100/torch ≥2.13 (kernel family:
#     "cudnn" AND ("flash" OR "sdpa"); FLEX_ENV_REPORT.md §3 names a
#     representative kernel
#     "cudnn_generated_forte_native_sdpa_sm90_flash_fprop_...").
#   mem_efficient
#     Masked SDPA dispatched to the mem-efficient backend on A500/torch 2.12
#     (kernel family: "cutlass" OR "fmha" OR "mem_efficient";
#     FLEX_ENV_REPORT.md §8 names the A500 path).
#   flex_compiled_triton
#     Dedicated flex API (create_block_mask + compiled flex_attention);
#     kernel family contains "flex_attention" or "_to_copy_flex_attention"
#     (FLEX_ENV_REPORT.md §3, §5: Triton kernels "triton_tem_fused__to_copy_
#     flex_attention_*").
#   matmul_cublas, nsa_kernel, minimax_kernel
#     Self-implemented paths (manual torch.matmul + F.softmax, or third-
#     party kernels with distinctive names).  These do NOT route through
#     F.scaled_dot_product_attention, so SDPA-family kernel matching would
#     give false negatives; the probe logs those as ``self_impl OK`` without
#     kernel matching.  Decision documented in SPEC 0022 §11 / TASKS.md
#     S22-T7.
_BACKEND_KERNEL_EXPECTATIONS: dict[str, dict] = {
    # NOTE: "cutlass" must NOT be excluded here — the genuine flash kernel
    # name embeds cutlass::half_t in its template args, so excluding
    # "cutlass" would false-negative the real flash kernel.  Flash is
    # identified by its pytorch_flash::/flash name; mem_efficient kernels
    # are excluded via their own names.
    "flash": {
        "include": ["flash"],
        "exclude": ["mem_efficient", "fmha"],
    },
    "cudnn_flash": {
        "include": ["cudnn"],
        "include_any": ["flash", "sdpa"],
    },
    "mem_efficient": {
        "include_any": ["cutlass", "mem_efficient", "fmha"],
        # Harden: flash kernels embed cutlass::half_t in template args,
        # so a bare "cutlass" include would false-positive against
        # pytorch_flash::flash_fwd_kernel.  Exclude pytorch_flash to
        # reject flash kernels while accepting genuine mem_efficient
        # (cutlass fmha) kernels.
        "exclude": ["pytorch_flash"],
    },
    "flex_compiled_triton": {
        "include_any": ["flex_attention", "_to_copy_flex_attention"],
    },
    "matmul_cublas":   {"self_impl": True},
    "nsa_kernel":      {"self_impl": True},
    "minimax_kernel":  {"self_impl": True},
}


def _match_kernel_expectations(kernel_names, expectation):
    """Match observed CUDA kernel names against a backend expectation.

    Args:
        kernel_names: List of kernel-name strings observed under profiler.
        expectation: Dict with optional keys ``include`` (list of
            substrings that must ALL appear in a single kernel name),
            ``include_any`` (list of substrings where at least ONE must
            appear), ``exclude`` (list of substrings that must NOT appear).

    Returns:
        Tuple ``(matched, observed)``. ``matched`` is True if at least
        one kernel name satisfies the expectation. ``observed`` is the
        first matching kernel name, or "none" if no kernel matched.
    """
    include = expectation.get("include", [])
    exclude = expectation.get("exclude", [])
    include_any = expectation.get("include_any", [])

    for kernel_name in kernel_names:
        name_lc = kernel_name.lower()
        # ALL include substrings must appear
        if not all(sub.lower() in name_lc for sub in include):
            continue
        # AT LEAST ONE include_any substring must appear
        if include_any and not any(sub.lower() in name_lc for sub in include_any):
            continue
        # NO exclude substring may appear
        if any(sub.lower() in name_lc for sub in exclude):
            continue
        return True, kernel_name
    return False, "none"


def _verify_backends(methods, mode, d_model, n_head, coord_dim,
                     batch_size, device, dtype, seed,
                     dist_mode="v1", seq_len=256, knn_neighbors=16):
    """One-shot backend verification probe (SPEC 0022 §11, ticket S22-T7).

    For each method in ``methods``, runs ONE forward pass at a
    representative shape (default ``(seq_len=256, L=1, fp16)``) under
    ``torch.profiler`` (CUDA only) and verifies that the dispatched
    kernel-name family matches the declared label from
    :func:`_resolve_backend`.  Emits one ``[BackendVerify]`` log line per
    method and raises ``RuntimeError`` on any MISMATCH (aborts the whole
    run BEFORE any timing row is produced — owner rule: no silent
    fallbacks).

    The probe is a *verification*, not an *enforcement* — it does NOT
    wrap any call in ``torch.nn.attention.sdpa_kernel`` context, so the
    default-dispatcher regime that Trackastra uses in production is
    preserved.

    Args:
        methods: List of registry keys selected for the sweep.
        mode: The sweep's ``--mode`` value.
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate dimensions.
        batch_size: Batch dimension for the probe inputs.
        device: torch device.
        dtype: torch dtype.
        seed: Random seed for the probe inputs.
        dist_mode: Distance decay mode ("v0" or "v1").
        seq_len: Sequence length of the probe forward pass (default 256,
            per SPEC §11).
        knn_neighbors: KNN-K used by KNN methods during the probe
            (default 16).

    Raises:
        RuntimeError: On any MISMATCH between declared label and observed
            CUDA kernel names (owner rule: no silent fallbacks).  The
            message names the method, the declared label, the observed
            kernel names, and the rule.
    """
    print(f"[BackendVerify] probing {len(methods)} method(s) at N={seq_len} "
          f"L=1 fp16 (one-shot, before sweep loop)...")
    for method_key in methods:
        declared = _resolve_backend(method_key, mode)
        expectation = _BACKEND_KERNEL_EXPECTATIONS.get(declared)
        if expectation is None:
            # No documented expectation — treat as MISMATCH (no silent
            # fallback to a permissive "match anything" default).
            raise RuntimeError(
                f"[BackendVerify] MISMATCH for method={method_key}: "
                f"declared={declared}, no kernel expectation documented. "
                f"Owner rule: no silent fallbacks — aborting before any "
                f"timing row."
            )
        if expectation.get("self_impl"):
            # Self-implemented path: manual torch.matmul + F.softmax (or
            # third-party kernels with distinctive names) — no SDPA call,
            # so no kernel-family matching.  Run the forward once and log.
            try:
                query, coords = make_inputs(
                    batch_size, seq_len, d_model, coord_dim + 1,
                    device, dtype, seed,
                )
                registry_entry = METHOD_REGISTRY[method_key]
                attention_class = resolve_class(registry_entry.class_path)
                layer = _make_layer(
                    method_key, attention_class, seq_len, d_model, n_head, knn_neighbors,
                    coord_dim, mode, dist_mode, device, dtype,
                )
                forward_style = _METHOD_FORWARD_STYLES[method_key]
                knn_idx = (
                    compute_knn_indices(coords, knn_neighbors)
                    if METHOD_REGISTRY[method_key].needs_knn else None
                )
                with torch.no_grad():
                    _forward_layer(layer, query, coords, knn_idx, forward_style)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                del query, coords, layer
            except Exception as exc:  # noqa: BLE001 — surfaced as MISMATCH
                raise RuntimeError(
                    f"[BackendVerify] MISMATCH for method={method_key}: "
                    f"declared={declared}, observed=probe_error: "
                    f"{type(exc).__name__}: {exc}. Owner rule: no silent "
                    f"fallbacks — aborting before any timing row."
                ) from exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[BackendVerify] method={method_key} "
                  f"expected={declared} observed=self_impl self_impl")
            continue

        # Build inputs + layer for this method.
        try:
            query, coords = make_inputs(
                batch_size, seq_len, d_model, coord_dim + 1,
                device, dtype, seed,
            )
            registry_entry = METHOD_REGISTRY[method_key]
            attention_class = resolve_class(registry_entry.class_path)
            layer = _make_layer(
                method_key, attention_class, seq_len, d_model, n_head, knn_neighbors,
                coord_dim, mode, dist_mode, device, dtype,
            )
            knn_idx = (
                compute_knn_indices(coords, knn_neighbors)
                if METHOD_REGISTRY[method_key].needs_knn else None
            )
            forward_style = _METHOD_FORWARD_STYLES[method_key]
        except Exception as exc:  # noqa: BLE001 — surfaced as MISMATCH
            raise RuntimeError(
                f"[BackendVerify] MISMATCH for method={method_key}: "
                f"declared={declared}, observed=build_error: "
                f"{type(exc).__name__}: {exc}. Owner rule: no silent "
                f"fallbacks — aborting before any timing row."
            ) from exc

        # Warmup forward (no profiler) so flex_attn triggers its lazy
        # torch.compile + block-mask build BEFORE we profile, and so
        # cuBLAS handles etc. settle.  Without this, the profiled
        # forward can capture compile-time Triton kernels that look
        # like flex kernels but aren't the steady-state dispatch.
        try:
            with torch.no_grad():
                _forward_layer(layer, query, coords, knn_idx, forward_style)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — surfaced as MISMATCH
            raise RuntimeError(
                f"[BackendVerify] MISMATCH for method={method_key}: "
                f"declared={declared}, observed=warmup_error: "
                f"{type(exc).__name__}: {exc}. Owner rule: no silent "
                f"fallbacks — aborting before any timing row."
            ) from exc

        # Profiled forward — extract CUDA kernel names.
        try:
            with torch.no_grad():
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA],
                    with_stack=False,
                ) as prof:
                    _forward_layer(layer, query, coords, knn_idx, forward_style)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — surfaced as MISMATCH
            raise RuntimeError(
                f"[BackendVerify] MISMATCH for method={method_key}: "
                f"declared={declared}, observed=profile_error: "
                f"{type(exc).__name__}: {exc}. Owner rule: no silent "
                f"fallbacks — aborting before any timing row."
            ) from exc

        kernel_names = []
        for event in prof.key_averages():
            key = getattr(event, "key", None) or getattr(event, "name", "<unnamed>")
            dev_time = getattr(event, "self_device_time_total", 0) or 0
            if dev_time > 0:
                kernel_names.append(key)

        # Now match observed kernel names against the expectation.
        # (Delegates to ``_match_kernel_expectations`` so the same
        # matching logic is used for the negative test.)
        matched, observed = _match_kernel_expectations(kernel_names, expectation)
        if not matched:
            observed_str = (
                observed if observed == "none"
                else ",".join(kernel_names)
            )
            raise RuntimeError(
                f"[BackendVerify] MISMATCH for method={method_key}: "
                f"declared={declared}, observed CUDA kernels={observed_str}. "
                f"Owner rule: no silent fallbacks (SPEC 0024 evidence + "
                f"SPEC 0022 §11). Aborting before any timing row is "
                f"produced."
            )
        print(f"[BackendVerify] method={method_key} "
              f"expected={declared} observed={observed} OK")


def _format_row(method_key, layer_count, seq_len, knn_neighbors, batch_size,
                time_s, time_std_s, memory_mb, status, dist_cache, backend):
    """Convert one timing result into a CSV row, blanking numbers on failure.

    Args:
        method_key: Registry key of the method.
        layer_count: Number of chained layers.
        seq_len: Sequence length.
        knn_neighbors: KNN neighbour count (0 for non-KNN methods).
        batch_size: Batch dimension B of the synthetic inputs.
        time_s: Mean wall time in seconds, or -1 on failure.
        time_std_s: Standard deviation across repeats in seconds, or -1 on failure.
        memory_mb: Incremental peak memory in MiB, or -1 on failure.
        status: Harness status string ("ok", "oom", or "err: ...").
        dist_cache: "on" or "off" (distance-cache regime for this row).
        backend: Declared dispatch class (from _resolve_backend).

    Returns:
        A row list matching the CSV header
        [method, L, N, K, batch, time_ms, memory_mb, error, time_std_ms,
        dist_cache, backend].
    """
    if status == "ok":
        return [method_key, layer_count, seq_len, knn_neighbors, batch_size,
                f"{time_s * 1000:.3f}", f"{memory_mb:.1f}", "",
                f"{time_std_s * 1000:.3f}", dist_cache, backend]
    return [method_key, layer_count, seq_len, knn_neighbors, batch_size,
            "", "", status, "", dist_cache, backend]


def run_sweep(methods, layer_counts, seq_lens, knn_choices, mode, dist_mode,
              d_model, n_head, coord_dim, batch_size, device, dtype, seed,
              warmup=5, min_run_time=0.3, repeats=1, with_knn=False,
              dist_cache="on"):
    """Run the full method × L × N × K sweep and collect the result rows.

    Non-KNN methods are swept once per (method, L, N) with ``K=0``; KNN-based
    methods are swept over every K in ``knn_choices`` (skipping ``K >= N``).
    Each configuration is measured through the harness ``try_bench`` so OOM
    and other errors are recorded instead of aborting the sweep.

    Args:
        methods: List of registry keys to benchmark.
        layer_counts: Layer counts to sweep.
        seq_lens: Sequence lengths to sweep.
        knn_choices: KNN neighbour counts to sweep (KNN methods only).
        mode: Positional encoding mode ("none", "bias", "rope").
        dist_mode: Distance decay mode ("v0" or "v1").
        d_model: Embedding dimension.
        n_head: Number of attention heads.
        coord_dim: Number of spatial coordinate dimensions.
        batch_size: Batch dimension for the synthetic inputs.
        device: torch device.
        dtype: torch dtype.
        seed: Random seed for reproducibility.
        with_knn: If ``True``, include KNN index computation inside the
            timed closure (training-realistic mode).
        dist_cache: "on" or "off". When "on", the 2D distance matrix is
            computed once outside the timed closure and amortised across
            all chained layers; when "off", it is recomputed inside the
            closure on every timed call.

    Returns:
        List of CSV rows [method, L, N, K, batch, time_ms, memory_mb, error,
        time_std_ms, dist_cache].
    """
    rows = []
    for layer_count in layer_counts:
        for seq_len in seq_lens:
            print(f"\nL={layer_count}  N={seq_len}", flush=True)
            for method_key in methods:
                registry_entry = METHOD_REGISTRY[method_key]
                # Per-method dist_cache column value: distance-consuming
                # methods (needs_knn or needs_coords) record the regime;
                # methods that consume no distances record "na".
                if registry_entry.needs_knn or registry_entry.needs_coords:
                    method_dist_cache = dist_cache
                else:
                    method_dist_cache = "na"
                knn_values = knn_choices if registry_entry.needs_knn else [0]
                for knn_neighbors in knn_values:
                    # K >= N guard: a KNN over N tokens can select at most N-1
                    # other tokens (a token is not its own neighbour), so any
                    # K >= N is invalid. Record it as a skip row instead of
                    # measuring, so the grid is complete and auditable.
                    if registry_entry.needs_knn and knn_neighbors >= seq_len:
                        rows.append(_format_row(
                            method_key, layer_count, seq_len, knn_neighbors,
                            batch_size, -1, -1, -1, "skip_k_ge_n",
                            method_dist_cache,
                            _resolve_backend(method_key, mode),
                        ))
                        continue
                    time_s, time_std_s, memory_mb, status = try_bench(
                        build_closure,
                        method_key, layer_count, seq_len, knn_neighbors,
                        batch_size, d_model, n_head, coord_dim, mode, dist_mode,
                        device, dtype, seed, with_knn,
                        warmup=warmup, min_run_time=min_run_time, repeats=repeats,
                        dist_cache=dist_cache,
                    )
                    rows.append(_format_row(
                        method_key, layer_count, seq_len, knn_neighbors,
                        batch_size, time_s, time_std_s, memory_mb, status,
                        method_dist_cache,
                        _resolve_backend(method_key, mode),
                    ))
                    time_ms = time_s * 1000 if time_s >= 0 else 0.0
                    time_std_ms = time_std_s * 1000 if time_std_s >= 0 else 0.0
                    print(f"  {method_key:<14s} K={knn_neighbors:<3d} -> "
                          f"{status:>6s}  {time_ms:8.3f} ms ±{time_std_ms:6.3f} ms  {memory_mb:8.1f} MB",
                          flush=True)
            # Keep peak-memory stats from bleeding across (L, N) groups.
            gc.collect()
            torch.cuda.empty_cache()
    return rows


def _select_methods(methods_arg, parser):
    """Resolve the ``--methods`` argument into a validated list of keys.

    Args:
        methods_arg: Raw ``--methods`` value (None selects every method).
        parser: argparse parser used for ``parser.error`` on unknown keys.

    Returns:
        List of registry keys to benchmark.
    """
    if methods_arg is None:
        return list_methods()
    selected = [name.strip() for name in methods_arg.split(",")]
    unknown = [name for name in selected if name not in METHOD_REGISTRY]
    if unknown:
        parser.error(f"Unknown method(s): {unknown}. Available: {list_methods()}")
    return selected


def _print_summary(out_path):
    """Print a readable per-row summary of the written results CSV.

    Args:
        out_path: Path of the CSV written by :func:`write_results_csv`.
    """
    print(f"\nSummary from {out_path}:")
    with open(out_path, newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            if row["error"]:
                print(f"  {row['method']:<14s} L={row['L']} N={int(row['N']):>5d} "
                      f"K={row['K']:>4s} -> {row['error']}")
            else:
                print(f"  {row['method']:<14s} L={row['L']} N={int(row['N']):>5d} "
                      f"K={row['K']:>4s} -> {row['time_ms']:>9s} ms  "
                      f"{row['memory_mb']:>8s} MB")


def main():
    """Parse CLI arguments, run the sweep, write the CSV, and print a summary.

    Sweeps over the methods × layers × N × K grid described in the module
    docstring and writes ``--out`` with the header
    ``["method", "L", "N", "K", "time_ms", "memory_mb", "error"]``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Consolidated attention benchmark sweep over methods × layers × "
            "N × K (replaces benchmark_sparse/full/gather_v3/small_n)."
        ),
    )
    parser.add_argument(
        "--methods", default=None,
        help="Comma-separated registry keys to benchmark; "
             "defaults to all registered methods.",
    )
    parser.add_argument(
        "--Ns", default="128,256,512,1024,2048,4096,8192",
        help="Comma-separated sequence lengths to sweep.",
    )
    parser.add_argument(
        "--Ks", default="4,16,64",
        help="Comma-separated KNN neighbour counts (used by KNN methods only).",
    )
    parser.add_argument(
        "--layers", default="1",
        help="Comma-separated layer counts; each count instantiates that many "
             "chained layers.",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch dimension B for the synthetic inputs. The v2 H100 grid "
             "ran B=1 (hardcoded); production Trackastra trains at B=48, so "
             "e.g. --batch 64 probes the kernel-saturated regime where "
             "launch overhead no longer dominates. Recorded per row in the "
             "CSV 'batch' column.",
    )
    parser.add_argument(
        "--mode", default="none", choices=["none", "bias", "rope"],
        help="Positional encoding mode for methods that support it.",
    )
    parser.add_argument(
        "--dist-mode", default="v1", choices=["v0", "v1"],
        help="Distance decay mode for dense_masked.",
    )
    parser.add_argument(
        "--coord-dim", type=int, default=2,
        help="Number of spatial coordinate dimensions.",
    )
    parser.add_argument(
        "--dist-cache", choices=["on", "off"], default="on",
        help="Distance cache regime: 'on' computes the 2D distance matrix "
             "once outside the timed closure (amortised, current default); "
             "'off' recomputes it inside the closure on every timed call.",
    )
    parser.add_argument(
        "--with-knn", action="store_true",
        help="Include KNN index computation (cdist+topk) inside the timed "
             "closure. This reflects the training-realistic cost where "
             "TrackingTransformer.forward() recomputes KNN indices every "
             "forward pass. Without this flag, KNN indices are pre-computed "
             "once before timing, isolating the attention-kernel cost.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    methods = _select_methods(args.methods, parser)
    layer_counts = parse_int_list(args.layers)
    seq_lens = parse_int_list(args.Ns)
    knn_choices = parse_int_list(args.Ks)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}  dtype: {dtype}  d={args.d}  nhead={args.nhead}")
    print(f"Methods: {methods}")
    print(f"L = {layer_counts}  N = {seq_lens}  K = {knn_choices}  B = {args.batch}")
    print(f"with_knn: {args.with_knn}\n")
    print(f"[Timing] warmup={args.warmup} rep={args.rep} repeats={args.repeats}")
    print(f"[DistCache] dist_cache={args.dist_cache}")

    # The backend probe profiles CUDA kernel families; on a CPU device there
    # are no CUDA kernels to observe and the declared classes ("flash",
    # "cudnn_flash", ...) are CUDA-only concepts, so verification is skipped
    # rather than guaranteed-failed.  Every cluster (CUDA) run still probes.
    if device.type == "cuda":
        _verify_backends(
            methods,
            mode=args.mode,
            d_model=args.d,
            n_head=args.nhead,
            coord_dim=args.coord_dim,
            batch_size=args.batch,
            device=device,
            dtype=dtype,
            seed=args.seed,
            dist_mode=args.dist_mode,
            seq_len=seq_lens[0],
            knn_neighbors=knn_choices[0],
        )
    else:
        print("[BackendVerify] skipped (CPU device — dispatch classes are CUDA-only)")

    rows = run_sweep(
        methods, layer_counts, seq_lens, knn_choices,
        args.mode, args.dist_mode, args.d, args.nhead, args.coord_dim,
        batch_size=args.batch, device=device, dtype=dtype, seed=args.seed,
        warmup=args.warmup, min_run_time=args.rep / 1000.0, repeats=args.repeats,
        with_knn=args.with_knn, dist_cache=args.dist_cache,
    )

    header = ["method", "L", "N", "K", "batch", "time_ms", "memory_mb", "error", "time_std_ms", "dist_cache", "backend"]
    write_results_csv(rows, args.out, header)
    print(f"\nWrote {len(rows)} rows -> {args.out}")

    _print_summary(args.out)


if __name__ == "__main__":
    main()
