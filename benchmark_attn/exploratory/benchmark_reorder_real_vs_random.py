"""Compare token reorder speedup: real cell centroids vs random points.

Benchmarks :class:`GatherSparseAttention` with and without spatial token
reordering (:class:`SpatialReorder`). Real centroids from vanvliet
microscopy data have spatial structure (clusters) that may make reordering
more effective than with uniform random data.

Timing methodology
------------------
Only the attention forward pass is timed (projections, K/V gather, SDPA,
output projection), averaged over ``n_trials`` iterations with CUDA
synchronization. ``SpatialReorder.compute_idx`` and the reorder/unreorder
gathers are *excluded* from the timed region: reorder indices are computed
once and shared across all layers (the deployment intent), so the
measurement isolates the memory-locality effect and is an upper bound on
the end-to-end gain.

With ``--n-layers > 1`` each timed iteration runs that many sequential
attention forwards with the same module and the same reorder indices,
mirroring the per-frame cost of an ``L``-layer encoder--decoder.

Reproducibility
---------------
``--seed`` (default 0) seeds both numpy and torch before data sampling and
module creation. Note that the historical runs behind the report numbers
were unseeded; re-running yields slightly different sampled data.

Data provenance
---------------
Real centroids are pre-extracted from the vanvliet dataset
``rpsM/151101_E4-12`` (see ``load_real_centroids``); no resampling or
alteration is applied beyond min-max normalization to the unit square.

Usage
-----
Legacy sweep (N ∈ {128, 512, 2048, 4096}, K ∈ {8, 32}, single layer)::

    uv run python exploratory/benchmark_reorder_real_vs_random.py

Report configuration (real centroids, L=12 layers, as cited in
``report/methodology.tex``)::

    uv run python exploratory/benchmark_reorder_real_vs_random.py \
        --n-points all --n-layers 12 --seed 0

Single custom configuration::

    uv run python exploratory/benchmark_reorder_real_vs_random.py \
        --n-points 2202 --n-layers 12 --knn-neighbors 8 --trials 100
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, '.')

from src.attention_modules import GatherSparseAttention, SpatialReorder


def load_real_centroids() -> np.ndarray:
    """Load pre-extracted cell centroids from the vanvliet dataset.

    Returns:
        Array of shape (N, 2) with centroid coordinates in pixels.

    Raises:
        FileNotFoundError: If the centroids file does not exist.
    """
    path = '../data/vanvliet/rpsM/151101_E4-12/centroids.npy'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run centroid extraction first: {path}")
    centroids = np.load(path).astype(np.float32)
    print(f"  Loaded {len(centroids)} real centroids from {path}")
    return centroids


def sample_points(n_points: int, real_pool=None) -> torch.Tensor:
    """Sample n_points in [0,1]^2 from real data or uniform random.

    Args:
        n_points: Number of points to sample. When sampling from
            ``real_pool`` with ``n_points > len(real_pool)`` points are
            drawn with replacement (duplicated coordinates).
        real_pool: Optional pool of real centroid coordinates.

    Returns:
        Tensor of shape (n_points, 2) with coordinates in [0, 1].
    """
    if real_pool is not None:
        idx = np.random.choice(len(real_pool), n_points, replace=(n_points > len(real_pool)))
        pts = real_pool[idx].copy()
        pts -= pts.min(axis=0, keepdims=True)
        pts /= pts.max(axis=0, keepdims=True) + 1e-8
    else:
        pts = np.random.rand(n_points, 2).astype(np.float32)
    return torch.from_numpy(pts)


def bench_n(n_points: int, knn_neighbors: int, real_pool, device,
            dtype=torch.float16, n_trials: int = 50,
            batch_size: int = 1, n_head: int = 4, head_dim: int = 64,
            n_layers: int = 1, seed: int = 0):
    """Benchmark GatherSparseAttention with and without spatial reorder.

    Both branches share the same module, weights, and dtype. The KNN graph
    is remapped correctly for the reordered space: recomputing ``cdist`` on
    the reordered points yields identical neighbour sets expressed in
    reordered indices (``cdist`` is permutation-equivariant). Reorder
    indices are computed once and reused for all ``n_layers`` timed
    forwards; ``compute_idx`` and the reorder/unreorder gathers are outside
    the timed region.

    Args:
        n_points: Number of points (tokens).
        knn_neighbors: Number of nearest neighbors for sparse attention.
        real_pool: Optional pool of real centroid coordinates.
        device: torch.device to run on.
        dtype: torch dtype for attention computation.
        n_trials: Number of timing trials.
        batch_size: Batch dimension.
        n_head: Number of attention heads.
        head_dim: Dimension per head.
        n_layers: Number of sequential attention forwards per timed
            iteration (reorder indices shared across layers).
        seed: Seed for numpy and torch (data sampling and weights).

    Returns:
        Dict mapping label to timing results. ``time_s`` and
        ``time_reorder_s`` are per-iteration times covering ``n_layers``
        attention forwards.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    embed_dim = n_head * head_dim
    attn = GatherSparseAttention(
        embed_dim=embed_dim, n_head=n_head,
        knn_neighbors=knn_neighbors, mode='none'
    ).to(device, dtype)

    results = {}
    for label, pts_fn in [("random", lambda: sample_points(n_points)),
                          ("real",   lambda: sample_points(n_points, real_pool))]:
        pts = pts_fn().to(device)
        dist = torch.cdist(pts, pts)
        _, knn_idx = torch.topk(dist, k=knn_neighbors, dim=-1, largest=False)
        tokens = torch.randn(batch_size, n_points, embed_dim, dtype=dtype, device=device)

        knn_idx_flat = knn_idx[None].expand(batch_size, n_points, knn_neighbors)
        for _ in range(5):
            for _ in range(n_layers):
                attn(tokens, tokens, tokens, knn_idx_flat)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(n_trials):
            for _ in range(n_layers):
                attn(tokens, tokens, tokens, knn_idx_flat)
        torch.cuda.synchronize()
        time_base = (time.perf_counter() - start_time) / n_trials

        # reorder
        sr = SpatialReorder(n_bins=32)
        reorder_idx, unreorder_idx = sr.compute_idx(pts[None])
        tokens_reordered = sr.reorder(tokens, reorder_idx)
        pts_re = pts[reorder_idx[0]]
        dist_re = torch.cdist(pts_re, pts_re)
        _, knn_idx_re = torch.topk(dist_re, k=knn_neighbors, dim=-1, largest=False)
        knn_idx_re = knn_idx_re[None].contiguous()
        for _ in range(5):
            for _ in range(n_layers):
                attn(tokens_reordered, tokens_reordered, tokens_reordered, knn_idx_re)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(n_trials):
            for _ in range(n_layers):
                attn(tokens_reordered, tokens_reordered, tokens_reordered, knn_idx_re)
        torch.cuda.synchronize()
        time_reorder = (time.perf_counter() - start_time) / n_trials

        results[label] = {"time_s": float(time_base), "time_reorder_s": float(time_reorder),
                          "speedup": float(time_base / time_reorder)}

        knn_dists = dist.gather(1, knn_idx)
        results[label]["mean_knn_dist"] = float(knn_dists.mean())

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with ``n_points`` (None for the legacy sweep, an int, or
        "all" for the full real centroid pool), ``n_layers``, default 1;
        ``knn_neighbors``, default 8 (single-configuration mode only);
        ``trials``, default None (auto: 100 for N <= 2048, else 30); and
        ``seed``, default 0.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark GatherSparseAttention with/without spatial token reorder"
    )
    parser.add_argument("--n-points", type=str, default=None,
                        help="Number of points: an integer, or 'all' for the full real "
                             "centroid pool (2202). Omit to run the legacy sweep "
                             "N in {128, 512, 2048, 4096}.")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="Sequential attention forwards per timed iteration "
                             "(default 1; the report's real-centroid run used 12)")
    parser.add_argument("--knn-neighbors", type=int, default=8,
                        help="KNN neighbours for single-configuration runs "
                             "(legacy sweep uses 8 and 32)")
    parser.add_argument("--trials", type=int, default=None,
                        help="Timing trials per configuration (default: auto, "
                             "100 for N<=2048 else 30)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for numpy and torch (default 0)")
    args = parser.parse_args()
    if args.n_points is not None and args.n_points != "all":
        try:
            args.n_points = int(args.n_points)
        except ValueError:
            parser.error("--n-points must be an integer or 'all'")
    return args


def main():
    """Run the reorder benchmark and print results."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required")
        return
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    real_pool = load_real_centroids()
    print()

    if args.n_points is None:
        n_points_list = [128, 512, 2048, 4096]
        knn_neighbors_list = [8, 32]
    else:
        n_points_list = [len(real_pool) if args.n_points == "all" else args.n_points]
        knn_neighbors_list = [args.knn_neighbors]

    print(f"{'N':>6} {'K':>3} {'L':>3} {'data':>8} {'time':>10} {'time+re':>10} "
          f"{'speedup':>8} {'meanKNNdist':>12}")
    print("-" * 76)

    for n_points in n_points_list:
        for knn_neighbors in knn_neighbors_list:
            n_trials = args.trials if args.trials is not None else (
                100 if n_points <= 2048 else 30)
            for label in ["random", "real"]:
                bench_results = bench_n(
                    n_points, knn_neighbors, real_pool if label == "real" else None,
                    device, n_trials=n_trials, n_layers=args.n_layers, seed=args.seed)
                result = bench_results[label]
                print(f"{n_points:6d} {knn_neighbors:3d} {args.n_layers:3d} {label:>8} "
                      f"{result['time_s']*1000:8.3f}ms "
                      f"{result['time_reorder_s']*1000:8.3f}ms "
                      f"{result['speedup']:7.3f}x "
                      f"{result['mean_knn_dist']:11.4f}")
            print()


if __name__ == "__main__":
    main()
