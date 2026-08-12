"""Compare token reorder speedup: real cell centroids vs random points.

Real centroids from vanvliet microscopy data have spatial structure (clusters)
that may make reordering more effective than with uniform random data.
"""

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
        n_points: Number of points to sample.
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
            batch_size: int = 1, n_head: int = 4, head_dim: int = 64):
    """Benchmark GatherSparseAttention with and without spatial reorder.

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

    Returns:
        Dict mapping label to timing results.
    """
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
            attn(tokens, tokens, tokens, knn_idx_flat)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(n_trials):
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
            attn(tokens_reordered, tokens_reordered, tokens_reordered, knn_idx_re)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(n_trials):
            attn(tokens_reordered, tokens_reordered, tokens_reordered, knn_idx_re)
        torch.cuda.synchronize()
        time_reorder = (time.perf_counter() - start_time) / n_trials

        results[label] = {"time_s": float(time_base), "time_reorder_s": float(time_reorder),
                          "speedup": float(time_base / time_reorder)}

        knn_dists = dist.gather(1, knn_idx)
        results[label]["mean_knn_dist"] = float(knn_dists.mean())

    return results


def main():
    """Run the reorder benchmark and print results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required")
        return
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    real_pool = load_real_centroids()
    print()

    Ns = [128, 512, 2048, 4096]
    Ks = [8, 32]

    print(f"{'N':>6} {'K':>3} {'data':>8} {'time':>10} {'time+re':>10} {'speedup':>8} {'meanKNNdist':>12}")
    print("-" * 72)

    for N in Ns:
        for K in Ks:
            for label in ["random", "real"]:
                bench_results = bench_n(N, K, real_pool if label == "real" else None, device,
                            n_trials=100 if N <= 2048 else 30)
                result = bench_results[label]
                print(f"{N:6d} {K:3d} {label:>8} "
                      f"{result['time_s']*1000:8.3f}ms "
                      f"{result['time_reorder_s']*1000:8.3f}ms "
                      f"{result['speedup']:7.3f}x "
                      f"{result['mean_knn_dist']:11.4f}")
            print()


if __name__ == "__main__":
    main()
