# Token Reordering for KNN Sparse Attention — Analysis

**Date:** 2026-05-12  
**Author:** caveman  
**Objective:** 4.1 — Token-reordering variant for memory locality (Hassani et al. 2024)

## Hypothesis

Reordering token sequence by spatial proximity should coalesce KNN gather accesses,
improving cache locality and throughput. Originally shown for sliding-window
neighborhood attention on dense 2D grids (Hassani et al. 2024).

## Implementation

- **SpatialReorder** (`benchmark_attn/model_parts.py:558-595`): quantizes coords to
  grid, computes linearized Z-order-like index, sorts by it.
- **token_reordering** (`sparse_attention_benchmark.py:15-25`): argsort by X
  coordinate, remap indices.

## Benchmark Data

### Experiment 1: Dense vs Sparse sweep (`benchmark_sparse_results.csv`)

GPU: unknown (A100-class). `L=1`, `B=2`, `d=256`, `h=4`. CSV at
`benchmark_attn/benchmark_sparse_results.csv`.

```
N=8192 K=4   sparse  0.023685s  sparse+r  0.023976s  speedup 0.99x
N=8192 K=16  sparse  0.036100s  sparse+r  0.036170s  speedup 1.00x
N=8192 K=64  sparse  0.082885s  sparse+r  0.080730s  speedup 1.03x
N=2048 K=64  sparse  0.020608s  sparse+r  0.020400s  speedup 1.01x
```

Maximum speedup observed: **1.03×** (N=8192, K=64). No meaningful gain at smaller K.

### Experiment 2: Real vs Random centroids (4GB GPU)

**Device:** NVIDIA RTX A500 Laptop GPU (4.0 GB VRAM)  
**Config:** `B=1, H=4, D_head=64, L=12 layers, float16`  
**Real data source:** `data/vanvliet/rpsM/151101_E4-12` — 2202 cell centroids from
63 frames of bacteria microscopy. Cells form dense clusters (typical of colony growth).  
**Script:** `benchmark_attn/benchmark_reorder_real_vs_random.py`

```
     N   K     data       time    time+re  speedup  meanKNNdist
------------------------------------------------------------------------
   128   8   random    0.261ms    0.266ms   0.981x      0.0824
   128   8     real    0.246ms    0.241ms   1.023x      0.0617

   128  32   random    0.406ms    0.421ms   0.966x      0.2181
   128  32     real    0.418ms    0.422ms   0.992x      0.1560

   512   8   random    0.838ms    0.859ms   0.975x      0.0421
   512   8     real    0.866ms    0.861ms   1.006x      0.0259

   512  32   random    1.573ms    1.564ms   1.005x      0.0986
   512  32     real    1.576ms    1.567ms   1.006x      0.0613

  2048   8   random    3.396ms    3.420ms   0.993x      0.0207
  2048   8     real    3.435ms    3.421ms   1.004x      0.0121

  2048  32   random    6.318ms    6.259ms   1.009x      0.0469
  2048  32     real    6.318ms    6.276ms   1.007x      0.0277

  4096   8   random    6.910ms    6.916ms   0.999x      0.0144
  4096   8     real    6.923ms    6.923ms   1.000x      0.0061

  4096  32   random   12.827ms   12.600ms   1.018x      0.0329
  4096  32     real   12.806ms   12.594ms   1.017x      0.0179
```

Key observations:
1. **Random vs Real are indistinguishable** — both cluster around 1.00±0.02×.
2. Real data has ~40% smaller mean KNN distance (more clustered), but this does not
   improve reorder benefit.
3. Speedup never exceeds **1.02×** even at largest N/K fitting 4GB VRAM.

## Root Cause Analysis

**Why reorder works for sliding window attention (Hassani et al.):**

In dense 2D grid + sliding window, sorting by spatial coordinate makes window
neighbors sequence-adjacent. The gather becomes a contiguous block read → optimal
cache line utilization.

**Why reorder fails for KNN sparse attention:**

1. **KNN graph is non-local in sequence space.** Even after spatial sort, a cell's K
   nearest neighbors are not necessarily adjacent. The KNN adjacency graph is
   irregular — it respects Euclidean distance, not sequence order.

2. **GPU cache absorbs KNN gather.** Each query gathers K=8-32 neighbors. At
   `D_head=64` fp16, that's 1-4 KB per query — fits in L1 cache. The irregular
   access pattern doesn't bottleneck DRAM bandwidth.

3. **Reorder overhead dominates.** Sorting (O(N log N)) plus two gather/scatter
   passes adds latency that cancels any marginal locality gain.

4. **Hassani et al. target unfused custom CUDA kernels** with explicit
   scatter/gather to global memory. PyTorch's `F.scaled_dot_product_attention`
   uses fused FlashAttention kernels that keep attention weights on-chip. The
   bottleneck they addressed doesn't exist in this stack.

## Conclusion

**Token reordering for KNN sparse attention provides no measurable benefit on
modern GPU architectures (< 1.03×).** The technique does not transfer from
sliding-window neighborhood attention to the KNN-graph attention used in
cell tracking. Recommend dropping this optimization from 4.1.

## Files Referenced

| Path | Description |
|------|-------------|
| `benchmark_attn/model_parts.py` | `SpatialReorder` (L558-595), `GatherSparseAttention` (L271-370) |
| `benchmark_attn/benchmark_sparse.py` | Full sweep benchmark (N, K, L, reorder) |
| `benchmark_attn/benchmark_sparse_results.csv` | Raw timing/memory data |
| `benchmark_attn/benchmark_reorder_real_vs_random.py` | Real vs random comparison script |
| `sparse_attention_benchmark.py` | Standalone prototype with `token_reordering()` |
| `data/vanvliet/rpsM/151101_E4-12/centroids.npy` | 2202 extracted cell centroids |
