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
- No QKV/output projection timing (same synthetic input)
- GPU warmup + sync before each measurement

**Updated 2026-08-17:** the KNN `cdist + topk` index computation is now
**included** in timing via the `--with-knn` flag (matching training behavior,
where `TrackingTransformer.forward()` recomputes indices every pass).

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

## Key Findings — Corrected H100 benchmark (job 3920627, `--with-knn`)

Corrected measurements on NVIDIA H100 (fp16, L=1, KNN cost included in timing):

| N | dense_masked (ms) | mask_knn K=4 (ms) | Winner |
|---|-------------------|-------------------|--------|
| 128 | 0.199 | 0.335 | dense_masked |
| 2048 | 0.344 | 0.396 | dense_masked |
| 4096 | 1.311 | 0.895 | mask_knn |
| 8192 | 5.113 | 2.914 | mask_knn |

- **`dense_flash` is NOT a realistic baseline** — it uses no mask and gets the
  FlashAttention-2 kernel, but real training always enforces the spatial cutoff
  mask → EfficientAttention. It is an upper bound on speed, not a reachable
  training configuration.
- **Realistic comparison: `dense_masked` vs `mask_knn`** — both use
  EfficientAttention. At N≤2048, `dense_masked` is **faster** than `mask_knn`
  (0.344 vs 0.396 ms at N=2048). At N≥4096, `mask_knn` is **faster** (2.914 vs
  5.113 ms at N=8192). Crossover at ~N=4000.
- **Two benchmark bugs were fixed (2026-08-17):** `dense_masked` was previously
  mapped to `RelativePositionalAttention` (per-layer cdist) instead of
  `CachedDistAttention`; KNN index computation was previously excluded from
  `mask_knn` timing.
- **Vanvliet training regime (N≈140) is well below the crossover** — sparse
  attention is NOT faster at cell-tracking scale.
