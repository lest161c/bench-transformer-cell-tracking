"""Compare token reorder speedup: real cell centroids vs random points.
Real centroids from vanvliet microscopy data have spatial structure (clusters)
that may make reordering more effective than with uniform random data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sys, os, time
import numpy as np
import torch

sys.path.insert(0, '.')
from model_parts import GatherSparseAttention, SpatialReorder

def load_real_centroids():
    p = '../data/vanvliet/rpsM/151101_E4-12/centroids.npy'
    if not os.path.exists(p):
        raise FileNotFoundError(f"Run centroid extraction first: {p}")
    c = np.load(p).astype(np.float32)
    print(f"  Loaded {len(c)} real centroids from {p}")
    return c

def sample_points(n, real_pool=None):
    """Sample n points in [0,1]^2 from real data or random."""
    if real_pool is not None:
        idx = np.random.choice(len(real_pool), n, replace=(n > len(real_pool)))
        pts = real_pool[idx].copy()
        pts -= pts.min(axis=0, keepdims=True)
        pts /= pts.max(axis=0, keepdims=True) + 1e-8
    else:
        pts = np.random.rand(n, 2).astype(np.float32)
    return torch.from_numpy(pts)

def bench_n(n, k, real_pool, device, dtype=torch.float16, n_trials=50, B=1, H=4, D_head=64):
    d = H * D_head
    attn = GatherSparseAttention(embed_dim=d, n_head=H, knn_neighbors=k, mode='none').to(device, dtype)

    results = {}
    for label, pts_fn in [("random", lambda: sample_points(n)),
                          ("real",   lambda: sample_points(n, real_pool))]:
        pts = pts_fn().to(device)
        dist = torch.cdist(pts, pts)
        _, knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)
        x = torch.randn(B, n, d, dtype=dtype, device=device)

        knn_idx_flat = knn_idx[None].expand(B, n, k)
        for _ in range(5):
            attn(x, x, x, knn_idx_flat)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            attn(x, x, x, knn_idx_flat)
        torch.cuda.synchronize()
        t_base = (time.perf_counter() - t0) / n_trials

        # reorder
        sr = SpatialReorder(n_bins=32)
        reorder_idx, unreorder_idx = sr.compute_idx(pts[None])
        x_re = sr.reorder(x, reorder_idx)
        pts_re = pts[reorder_idx[0]]
        dist_re = torch.cdist(pts_re, pts_re)
        _, knn_idx_re = torch.topk(dist_re, k=k, dim=-1, largest=False)
        knn_idx_re = knn_idx_re[None].contiguous()
        for _ in range(5):
            attn(x_re, x_re, x_re, knn_idx_re)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            attn(x_re, x_re, x_re, knn_idx_re)
        torch.cuda.synchronize()
        t_re = (time.perf_counter() - t0) / n_trials

        results[label] = {"time_s": float(t_base), "time_reorder_s": float(t_re),
                          "speedup": float(t_base / t_re)}

        knn_dists = dist.gather(1, knn_idx)
        results[label]["mean_knn_dist"] = float(knn_dists.mean())

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA required"); return
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
                r = bench_n(N, K, real_pool if label == "real" else None, device,
                            n_trials=100 if N <= 2048 else 30)
                res = r[label]
                print(f"{N:6d} {K:3d} {label:>8} "
                      f"{res['time_s']*1000:8.3f}ms "
                      f"{res['time_reorder_s']*1000:8.3f}ms "
                      f"{res['speedup']:7.3f}x "
                      f"{res['mean_knn_dist']:11.4f}")
            print()

if __name__ == "__main__":
    main()
