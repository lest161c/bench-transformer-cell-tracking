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
``gather_sdpa``, ``gather_fused``, ``gather_matmul``, ``mask_knn``,
``dense_flash``, ``dense_masked``, ``nsa``, ``knn_relpos``, ``minimax``) and
default to every registered method (``list_methods``).  K is only meaningful
for KNN-based methods; all other methods are emitted with ``K=0``.

Usage:
    python scripts/benchmarks/benchmark_sweep.py \\
        --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,dense_flash,dense_masked,nsa,knn_relpos,minimax \\
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
    RelativePositionalAttention, KNNRelativePositionalAttention,
    GatherSparseAttention, GatherSparseFusedAttention, GatherSparseMatmulAttention,
    KNNMaskSparseAttention, DenseFlashAttention, NSASparseAttention, MiniMaxSparseAttention,
)

# How each registered method's ``forward`` is invoked.  The gather/mask
# variants take ``(query, key, value, knn_indices, coords)``, the relative
# positional variant takes ``(query, key, value, coords, knn_indices)``, the
# dense masked variant takes ``(query, key, value, coords)``, and everything
# else (dense_flash, nsa, minimax) is called as ``(query, key, value)``.
_METHOD_FORWARD_STYLES = {
    "gather_sdpa": "knn",
    "gather_fused": "knn",
    "gather_matmul": "knn",
    "mask_knn": "knn",
    "knn_relpos": "relpos",
    "dense_masked": "masked",
    "dense_flash": "qkv",
    "nsa": "qkv",
    "minimax": "qkv",
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
    if method_key == "knn_relpos":
        return attention_class(coord_dim, d_model, n_head, mode=mode,
                    knn_neighbors=knn_neighbors).to(device, dtype)
    if METHOD_REGISTRY[method_key].needs_knn:
        # gather_sdpa / gather_fused / gather_matmul / mask_knn
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
    return layer(x, x, x)


def build_closure(method_key, layer_count, seq_len, knn_neighbors,
                  batch_size, d_model, n_head, coord_dim, mode, dist_mode,
                  device, dtype, seed, with_knn=False):
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
        dist_mode: Distance decay mode ("v0" or "v1") for dense_masked.
        device: torch device for inputs and layers.
        dtype: torch dtype for inputs and layers.
        seed: Random seed for input and parameter generation.
        with_knn: If ``True``, include KNN index computation inside the
            timed closure (training-realistic mode).

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
    if not with_knn:
        knn_idx = compute_knn_indices(coords, knn_neighbors) if registry_entry.needs_knn else None
    else:
        knn_idx = None  # Will be computed inside the closure.

    dist_2d = torch.cdist(coords[..., 1:].float(), coords[..., 1:].float()).to(dtype)

    if with_knn and registry_entry.needs_knn:
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


def _format_row(method_key, layer_count, seq_len, knn_neighbors,
                time_s, memory_mb, status):
    """Convert one timing result into a CSV row, blanking numbers on failure.

    Args:
        method_key: Registry key of the method.
        layer_count: Number of chained layers.
        seq_len: Sequence length.
        knn_neighbors: KNN neighbour count (0 for non-KNN methods).
        time_s: Mean wall time in seconds, or -1 on failure.
        memory_mb: Incremental peak memory in MiB, or -1 on failure.
        status: Harness status string ("ok", "oom", or "err: ...").

    Returns:
        A row list matching the CSV header
        [method, L, N, K, time_ms, memory_mb, error].
    """
    if status == "ok":
        return [method_key, layer_count, seq_len, knn_neighbors,
                f"{time_s * 1000:.3f}", f"{memory_mb:.1f}", ""]
    return [method_key, layer_count, seq_len, knn_neighbors, "", "", status]


def run_sweep(methods, layer_counts, seq_lens, knn_choices, mode, dist_mode,
              d_model, n_head, coord_dim, batch_size, device, dtype, seed,
              with_knn=False):
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

    Returns:
        List of CSV rows [method, L, N, K, time_ms, memory_mb, error].
    """
    rows = []
    for layer_count in layer_counts:
        for seq_len in seq_lens:
            print(f"\nL={layer_count}  N={seq_len}", flush=True)
            for method_key in methods:
                registry_entry = METHOD_REGISTRY[method_key]
                knn_values = knn_choices if registry_entry.needs_knn else [0]
                for knn_neighbors in knn_values:
                    if registry_entry.needs_knn and knn_neighbors >= seq_len:
                        continue
                    time_s, memory_mb, status = try_bench(
                        build_closure,
                        method_key, layer_count, seq_len, knn_neighbors,
                        batch_size, d_model, n_head, coord_dim, mode, dist_mode,
                        device, dtype, seed, with_knn,
                    )
                    rows.append(_format_row(
                        method_key, layer_count, seq_len, knn_neighbors,
                        time_s, memory_mb, status,
                    ))
                    time_ms = time_s * 1000 if time_s >= 0 else 0.0
                    print(f"  {method_key:<14s} K={knn_neighbors:<3d} -> "
                          f"{status:>6s}  {time_ms:8.3f} ms  {memory_mb:8.1f} MB",
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
    print(f"L = {layer_counts}  N = {seq_lens}  K = {knn_choices}")
    print(f"with_knn: {args.with_knn}\n")

    rows = run_sweep(
        methods, layer_counts, seq_lens, knn_choices,
        args.mode, args.dist_mode, args.d, args.nhead, args.coord_dim,
        batch_size=1, device=device, dtype=dtype, seed=args.seed,
        with_knn=args.with_knn,
    )

    header = ["method", "L", "N", "K", "time_ms", "memory_mb", "error"]
    write_results_csv(rows, args.out, header)
    print(f"\nWrote {len(rows)} rows -> {args.out}")

    _print_summary(args.out)


if __name__ == "__main__":
    main()
