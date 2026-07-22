# Attention Optimization Landscape — Correct Baseline, Real Limits, and Where Gains Could Exist

**Date:** 2026-06-08
**Context:** Analysis of whether a method faster than dense attention with spatial cutoffs can be found, what the correct baseline is, and where further optimization is possible.
**Depends on:** `2026-05-29_gather_sparse_bottleneck.md`, `2026-05-29_flash_attn_integration.md`, `2026-05-29_attention_not_bottleneck.md`, `2026-06-08_nsa_benchmark_negative.md`

---

## 1. The Wrong Baseline Problem

`DenseFlashAttention` (zero mask, pure FlashAttention) is the wrong baseline for our work because Trackastra **requires** a spatial cutoff mask per Equation 3:

```
A(Q,K,V) = softmax(QK^T/√d + M) V
M_ij = 0 if ||p_i - p_j||_2 ≤ d_max, else M_ij = -∞
```

This mask is architectural — it encodes the biological prior that cells cannot teleport beyond `d_max` between frames. Removing it gives faster benchmarks but breaks the model's correctness. The correct baseline is `CachedDistAttention` (computes `cdist(yx, yx)` once in `forward()`, shares across all L layers).

## 2. What Approaches Enforce Spatial Cutoffs

| Method | Speed vs cached-dense (N=256) | Enforces d_max? | Mechanism |
|---|---|---|---|
| `RelativePositionalAttention` | 1.00× (baseline) | Yes | cdist per layer + masked SDPA |
| `CachedDistAttention` | ~1.5× (L=12) | Yes | cdist once, mask per layer, masked SDPA |
| `DenseFlashAttention` | 4.1× | **No** | Zero mask, pure FlashAttention |
| `KNNMaskSparseAttention` | 3.1× | Yes (if K covers d_max) | KNN scatter mask + masked SDPA |
| `GatherSparseAttention` | 0.30× | Yes | KNN gather + SDPA on q_len=1 |
| NSA (Native Sparse Attn) | 0.04× | Partial (sliding window only) | 3-path sparse attention |

**`KNNMaskSparseAttention` is the best method that enforces spatial cutoffs.** It builds a sparse N×N mask from KNN neighbors via `scatter_` and passes it to `F.scaled_dot_product_attention`. At N=256, this is 3.1× faster than the masked baseline because it:

1. **Avoids per-layer cdist** — KNN indices precomputed once (like CachedDist, but with topK)
2. **Builds mask via scatter** — O(N·K) instead of O(N²) threshold
3. **Uses EfficientAttention** — SDPA with mask dispatches to the best available masked kernel

The trade-off: KNN must have K large enough to cover all tokens within `d_max` to preserve correctness.

## 3. Is Finding a Faster Method Realistic?

### 3.1 The Fundamental Limits

Spatial cutoffs create an irreducible requirement: for each query token, you must determine which key tokens are within distance `d_max`. This is a **geometric range query** — at minimum O(N log N + output_size) using spatial data structures. The output size is O(N²) for dense configurations (where most tokens are within `d_max`), or O(N·K) for sparse configurations.

At our dataset N ≈ 140: the entire N² matrix is 19,600 entries — 20 KB in fp16. The attention computation itself is an order of magnitude smaller than the Python function call overhead. The → notation in Trackastra is correct: attention was never the bottleneck, not even close.

### 3.2 Where Gains Could Exist (Hypothetically)

Gains from further attention optimization would only be visible at **N > 500**, where attention dominates step time (>30%). At those sizes:

| Optimization target | Approach | Max theoretical gain | Practical estimate |
|---|---|---|---|
| Mask construction | Cached dist + scatter (done) | O(N²) → O(N·K) | 2–3× at N=2048 |
| Masked SDPA → Flash | `flex_attention` + spatial score_mod | Flash kernel speed | 1.5–2× |
| Token reordering + window | Sort by yx, apply window mask | O(N²) → O(N·W) | 2–4× at large N |
| Block-sparse attention | Spatial block partition + flex_attention | Near-flash speed | 3–5× at N ≫ 2048 |

### 3.3 Memory Access Pattern Optimizations

**Current bottlenecks are NOT memory-access-limited.** The profiler data shows:

- **Masked dense SDPA** at N=256: 33 µs CUDA time, 29 µs for EfficientAttention. The mask is 256×256×2 = 128 KB — fits entirely in L2 cache.
- **Gather Sparse**: 271 µs for `index_elementwise_kernel`. This IS a memory access bottleneck — random reads from scattered K/V locations. But this method is discarded for other reasons.
- **KNN scatter mask**: The `scatter_` operation is the remaining overhead. At N=2048 with K=16, scatter builds a 2048×2048 mask = 8 MB in fp16. Still fits in L2. Not memory-bound.

**Where memory access could matter (hypothetically at N ≫ 5000):**
- Scatter on N×N mask exceeds L2 cache → bank conflicts
- Global memory reads for spatial distance computation
- If using block-sparse: gathering non-contiguous K/V blocks from global memory

**The practical answer:** at N < 2000, all tensors fit in L2 cache. Memory access patterns are not the bottleneck — **kernel launch overhead and algorithmic complexity are.**

## 4. What Would Be Genuinely Novel

Current landscape (all explored or bounded):

```
Approach                     Status
─────────────────────────────────────────
cdist amortization          ✓ Done (CachedDist)
KNN scatter mask            ✓ Done (KNNMaskSparse)
Gather-based KNN sparse     ✗ Slower at all N (bottleneck docs exist)
NSA (DeepSeek 2025)         ✗ Slower at N < 16K (Yuan et al. Fig. 6 + our benchmarks)
FlexAttention spatial       ? Not tested — only promising unexplored direction
Block-sparse attention      ? Not tested — only relevant for N > 500
Token reordering + window   ? Partially tested (SpatialReorder) — overhead > benefit at small N
```

The only unexplored direction with potential is **FlexAttention with spatial score_mod**: PyTorch 2.5+ supports `torch.nn.attention.flex_attention.flex_attention` with a `score_mod` function. A `score_mod` that implements the spatial cutoff using cached distances could:

1. **Avoid explicit N×N mask construction** — no scatter, no threshold, no mask tensor
2. **Dispatch to a compiled Triton kernel** — via `torch.compile(flex_attention)`
3. **Potentially achieve FlashAttention-speed** for masked attention

The `score_mod` would look like:
```python
def spatial_cutoff_mod(score, batch, head, q_idx, kv_idx):
    dist = dist_matrix[batch, q_idx, kv_idx]
    return torch.where(dist <= d_max, score, -inf)
```

However, there are caveats:
- `flex_attention` requires `torch.compile` which incurs compilation overhead on first call
- `score_mod` runs in Python (unless compiled) — overhead might kill small-N benefits
- Not supported on all GPU architectures

**Realistic assessment:** `flex_attention` with spatial score_mod is the only remaining approach that could **both enforce spatial cutoffs AND beat `KNNMaskSparseAttention`**. Expected gain: 0–30% at N=256, 50–200% at N=2048. Worth a focused benchmark but unlikely to change the conclusion for the vanvliet dataset.

## 5. Summary

1. **`dense_flash` is the wrong baseline.** Trackastra requires spatial cutoffs. The correct baseline is `CachedDistAttention`.

2. **`KNNMaskSparseAttention` is the current best method enforcing spatial cutoffs** — 3.1× faster than the masked baseline at N=256.

3. **Finding a significantly better method is unlikely.** The fundamental limit is the geometric range query for spatial cutoffs. At dataset N ≈ 140, attention is <5% of training step time, so even a 10× attention speedup yields <5% throughput improvement. This was already established (`2026-05-29_attention_not_bottleneck.md`).

4. **Memory access patterns are not a bottleneck** at N < 2000 (all tensors fit in L2). Algorithmic complexity (O(N²) vs O(N·K)) and kernel launch overhead dominate.

5. **The only unexplored promising direction is `flex_attention` with spatial score_mod** — avoids explicit N×N mask tensor, potentially faster than EfficientAttention with mask. Could provide modest gains at N > 500 but requires benchmarking.

6. **For the thesis:** The attention benchmark work (§4.1) is a complete investigation that identified the root cause (mask construction, not N² compute), found the correct optimization (cdist amortization), established limits (gather overhead, q_len=1 dispatch), and evaluated a novel published method (NSA). The exploration has reached diminishing returns.
