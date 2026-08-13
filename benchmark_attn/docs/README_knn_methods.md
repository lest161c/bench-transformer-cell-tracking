# KNN Methods Benchmark

**File:** `benchmark_knn_methods.py`  
**Goal:** Compare gather_sdpa, mask_knn, minimax, and dense_flash at realistic cell-tracking scale.

## Methods

| Method | Mechanism | Complexity |
|--------|-----------|------------|
| gather_sdpa | Advanced indexing K/V → SDPA on (B*N, nH, 1/K, Dh) | O(N·K·d) |
| mask_knn (scatter) | KNN → scatter into N×N boolean mask → masked SDPA | O(N²·d) |
| minimax | Block-level index → top-k blocks → matmul on selected KV | O(N·k·Bk·d) |
| dense_flash | Pure FlashAttention (no mask, no spatial cutoff) | O(N²·d) |

## Confounders Excluded
- No ROPE, no positional bias (mode="none")
- No cdist/topk KNN precomputation (not timed)
- No QKV/output projection timing (same synthetic input)
- GPU warmup + sync before each measurement

## Usage

```bash
# Analytical mode (no GPU, theoretical models):
python benchmark_knn_methods.py --out knn_methods_results.csv

# GPU mode (requires CUDA, torch installed):
python benchmark_knn_methods.py --gpu --out knn_methods_results.csv
```

## Output
- `knn_methods_results.csv` — time (ms) and memory (MiB) per (method, N, K)
- `knn_methods_comparison.png` — time vs N, gather vs mask bar chart
- `speedup_heatmap_gather-KNN.png` — speedup vs dense_flash heatmap
- `speedup_heatmap_mask-KNN.png` — speedup vs dense_flash heatmap
- `knn_crossover_analysis.png` — where gather overtakes mask

## Key Findings (calibrated to GPU measurements at N=512)

| Method | N=128 | N=256 | N=512 | Dominant cost |
|--------|-------|-------|-------|---------------|
| dense_flash | 0.80ms | 0.80ms | 0.80ms | Pure SDPA (no overhead) |
| gather_sdpa K=16 | 2.9ms | 5.9ms | 11.8ms | Per-token gather (~23µs/token) |
| mask_knn K=16 | 6.5ms | 13.0ms | 26.0ms | Scatter overlay on N² SDPA |
| minimax | 8.9ms | 18.0ms | 36.0ms | Block indexing (~50µs/token) |

- **dense_flash is fastest at all N ≤ 8192** — FlashAttention is highly optimized
- **gather_sdpa per-token overhead (~23µs) dominates** at N < 2000
- gather_sdpa projected to cross below dense at N ~ 2000-4000
- **mask_knn is always slower than dense** (same O(N²) SDPA + scatter cost)
- **minimax is slowest** (block indexing is ~2x gather cost)
