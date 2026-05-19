# Gather-Sparse Attention Benchmark

Benchmarking gather-based sparse attention as a replacement for dense N×N masked attention in transformer-based cell tracking.

**Core idea:** Replace N×N attention with KNN-gathered N×K attention. Precompute KNN indices, gather K/V, run SDPA on N×K tensors. Complexity drops from O(N²) to O(NK).

## Key Results

Tested on NVIDIA RTX 4090 (fp16, B=2, d=256, h=4). All numbers single-layer forward pass.

| N | Method | Time (s) | GPU Mem (MB) | vs Dense |
|---|--------|----------|-------------|----------|
| 2048 | Dense | 0.0085 | 142 | 1× |
| 2048 | Sparse K=16 | 0.0089 | 102 | 0.96× speed, 1.4× mem savings |
| 8192 | Dense | 0.1391 | 2200 | 1× |
| 8192 | Sparse K=4 | 0.0237 | 120 | **5.9× faster**, 18× less mem |
| 8192 | Sparse K=16 | 0.0361 | 408 | **3.9× faster**, 5.4× less mem |
| 8192 L=4 | Dense | OOM | OOM | — |
| 8192 L=4 | Sparse K=4 | 0.0935 | 385 | Fits when dense OOMs |

**Spatial reorder** (Hassani et al. 2024): negligible speedup on random data (~1-3%). Real cell centroid data may differ.

**Attention backends:** FlashAttention/CuDNN vs Math backend ~1.5-2× throughput difference at N≥512.

## Setup

```bash
uv sync
```

Requires Python ≥3.14, PyTorch ≥2.11 with CUDA.

## Usage

```bash
# Main sparse vs dense benchmark
python benchmark_sparse.py

# Reorder ablation with real centroids (requires data/vanvliet)
python benchmark_reorder_real_vs_random.py

# Plot results
python plot_sparse.py  # → benchmark_sparse.html

# Verify CUDA attention backends
python verify_cudnn.py
```

## Structure

```
├── model_parts.py                # GatherSparseAttention, RelativePositionalAttention, SpatialReorder
├── benchmark_sparse.py           # Dense vs sparse benchmark (N up to 8192, K 4-64)
├── benchmark_reorder_real_vs_random.py  # Reorder speedup: real centroids vs random
├── plot_sparse.py                # Seaborn → HTML visualization
├── verify_cudnn.py               # CUDA backend verification
├── config.yaml                   # Default hyperparameters
├── benchmark_sparse_results.csv  # Benchmark data
├── pyproject.toml                # Dependencies
└── uv.lock                       # Lockfile
```

## Reference

- Gallusser & Weigert, *Trackastra*, ECCV 2024
- Hassani et al., *Neighborhood Attention*, ECCV 2024
