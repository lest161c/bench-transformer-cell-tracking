# H100 Benchmark Results with --with-knn

**Date:** 2026-08-17
**Job:** 3920627 (COMPLETED, 7:15 elapsed)
**Data:** `full_bench_h100_with_knn.csv`

## Key Results

### Speed Ranking at N=2048 (training-realistic)

| Method | K | Time (ms) | Memory (MB) |
|--------|---|-----------|-------------|
| dense_flash | 0 | 0.111 | 6.8 |
| mask_knn | 4 | 0.396 | 70.9 |
| mask_knn | 16 | 0.394 | 71.1 |
| mask_knn | 32 | 0.393 | 71.3 |
| mask_knn | 64 | 0.391 | 71.8 |
| dense_masked | 0 | 0.344 | 143.8 |
| knn_relpos | 4 | 0.480 | 19.2 |
| knn_relpos | 16 | 0.476 | 51.0 |
| gather_sdpa | 4 | 0.480 | 19.9 |
| gather_sdpa | 16 | 0.474 | 51.0 |
| gather_sdpa | 32 | 0.617 | 96.3 |
| gather_sdpa | 64 | 0.965 | 188.8 |
| knn_relpos | 32 | 0.617 | 96.3 |
| knn_relpos | 64 | 0.964 | 188.8 |
| gather_fused | 4 | 0.485 | 29.1 |
| gather_fused | 16 | 0.537 | 88.3 |
| gather_fused | 32 | 0.741 | 168.6 |
| gather_fused | 64 | 1.196 | 329.1 |
| gather_matmul | 4 | 0.547 | 19.2 |
| gather_matmul | 16 | 0.561 | 51.0 |
| gather_matmul | 32 | 0.645 | 96.3 |
| gather_matmul | 64 | 0.966 | 188.8 |
| minimax | 0 | 22.472 | 2757.0 |
| nsa | 0 | 2.506 | 398.9 |

### Speed Ranking at N=8192 (high-N)

| Method | K | Time (ms) | Memory (MB) |
|--------|---|-----------|-------------|
| dense_flash | 0 | 0.351 | 25.3 |
| dense_masked | 0 | 5.113 | 2255.0 |
| mask_knn | 4 | 2.914 | 1049.5 |
| mask_knn | 16 | 2.999 | 1050.3 |
| mask_knn | 32 | 3.105 | 1051.3 |
| mask_knn | 64 | 3.293 | 1053.3 |
| knn_relpos | 4 | 2.270 | 268.7 |
| knn_relpos | 16 | 2.736 | 269.8 |
| knn_relpos | 32 | 3.376 | 385.0 |
| knn_relpos | 64 | 4.768 | 755.0 |
| gather_sdpa | 4 | 2.268 | 268.7 |
| gather_sdpa | 16 | 2.734 | 269.8 |
| gather_sdpa | 32 | 3.377 | 385.0 |
| gather_sdpa | 64 | 4.751 | 755.0 |
| nsa | 0 | 5.861 | 1714.0 |
| minimax | 0 | 294.878 | 44052.0 |

### Scaling Analysis

**dense_flash** (no mask, FlashAttention-2):
- N=128: 0.112 ms
- N=2048: 0.111 ms
- N=8192: 0.351 ms
- **Stays flat until N=4096, then scales. Dominant at all N.**

**mask_knn** (KNN mask + EfficientAttention):
- N=128: 0.335 ms (K=4)
- N=2048: 0.396 ms (K=4)
- N=8192: 2.914 ms (K=4)
- **3x slower than dense_flash at N=2048. Never competitive.**

**gather_sdpa** (gather K/V + SDPA):
- N=128: 0.416 ms (K=4)
- N=2048: 0.480 ms (K=4)
- N=8192: 2.268 ms (K=4)
- **Close to mask_knn at low N, slightly faster at high N.**

## Critical Finding

**Dense FlashAttention (dense_flash) dominates at ALL tested N values (128-8192).**

At N=2048 (vanvliet training regime):
- dense_flash: 0.111 ms (1.0x baseline)
- mask_knn: 0.396 ms (3.6x slower)
- gather_sdpa: 0.480 ms (4.3x slower)
- dense_masked: 0.344 ms (3.1x slower)

The mask-based KNN sparse attention (mask_knn) is NOT faster than dense FlashAttention at any tested N. The overhead of building the N×N mask and the fallback to EfficientAttention (not FlashAttention-2) makes it slower than the no-mask dense path.

## Implications

1. **Sparse attention is moot for speed at N≤8192.** Dense FlashAttention is always faster.
2. **The K-sweep should focus on TRA accuracy, not speed.** If mask_knn achieves similar TRA to dense_masked, it's a memory win (70 MB vs 143 MB at N=2048).
3. **The dense_flash ablation (job 3920471) is critical.** If removing the spatial cutoff (dense_flash) degrades TRA, then we need the mask — but the mask forces EfficientAttention, not FlashAttention-2. This is the fundamental tension.
4. **gather_sdpa is competitive with mask_knn** at high N (2.268 ms vs 2.914 ms at N=8192, K=4), and uses less memory (268 MB vs 1049 MB). But both are slower than dense_flash.
