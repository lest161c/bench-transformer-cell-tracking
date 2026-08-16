# Open Questions — Attention Mechanism Design

**Date:** 2026-08-17
**Status:** OPEN — needs ablation experiments before sparse attention experiment can be designed

---

## 1. Do we need the spatial cutoff for optimal TRA?

The spatial cutoff mask (`||p_i - p_j|| > d_max → -inf`) is a structural
component of Trackastra's attention (Eq. 3 in the Trackastra paper).

**The tension:**

- `dense_flash` (no mask, pure FlashAttention) is the **fastest method at all N**
  on H100: 0.11ms at N=2048 vs 0.15ms for `mask_knn` and 0.41ms for `dense_masked`.
- If the spatial cutoff is unnecessary for tracking accuracy, then
  `dense_flash` is optimal and sparse attention is moot.
- If the spatial cutoff IS necessary, then the question becomes:
  can a KNN mask (`O(NK)`) replace the full spatial cutoff mask (`O(N²)`)
  while maintaining TRA?

**Sub-questions:**

1. Does removing the spatial cutoff degrade TRA?
   (Ablation: `dense_flash` vs `dense_masked`)
2. Does the spatial cutoff regularize training,
   or is it just a compute optimization?
3. Is the spatial cutoff more important for inference (longer sequences)
   than training?

**What the H100 benchmark tells us:**

| Method | N=2048 time | N=2048 mem | Has spatial cutoff? |
|--------|:-----------:|:----------:|:-------------------:|
| `dense_flash` | 0.110 ms | 6.3 MB | No |
| `mask_knn` K=4 | 0.153 ms | 70.3 MB | Yes (KNN-based) |
| `dense_masked` | 0.409 ms | 95.8 MB | Yes (cdist-based) |

`dense_flash` is 3.7× faster than `dense_masked` and uses 15× less memory.
The spatial cutoff costs 3.7× in time and 15× in memory at N=2048.

**The experiment that answers this:**

Train two models with identical hyperparameters:
- Model A: `dense_flash` (no spatial cutoff)
- Model B: `dense_masked` (with spatial cutoff, `cutoff_spatial=256`)

Compare TRA, AOGM, and val_loss curves. If TRA is within ±0.001,
the spatial cutoff is unnecessary for tracking accuracy.

---

## 2. Which K is optimal for TRA/AOGM?

**Status:** OPEN — the current K-sweep uses the wrong attention mechanism
(`GatherSparseAttention` instead of `KNNMaskSparseAttention`)

The H100 benchmark shows `mask_knn` time is **K-independent at N≤2048**
(K=4/16/32/64 all take 0.153ms). So the optimal K is purely an accuracy question.

**The experiment that answers this:**

Train models with `KNNMaskSparseAttention` at K=4, 16, 32, 64 vs
`dense_masked` baseline. 3 seeds each. Measure TRA, AOGM, memory, inference time.

**Prerequisite:** Verify that FlashAttention works with additive float masks
(write a verification script). The `CachedDistAttention` and
`KNNMaskSparseAttention` classes already use this pattern, but we need
to confirm the SDPA backend dispatch.

---

## 3. What N allows measurable sparse gains in full-scale training?

**Status:** OPEN — needs window size sweep or larger-cell-count dataset

From the H100 benchmark, the crossover where `mask_knn` beats `dense_masked`:

| N | `dense_masked` | `mask_knn` K=4 | Speedup |
|---|:-----------:|:-----------:|:-------:|
| 128 | 0.344 ms | 0.152 ms | 2.3× |
| 256 | 0.341 ms | 0.154 ms | 2.2× |
| 512 | 0.345 ms | 0.155 ms | 2.2× |
| 1024 | 0.347 ms | 0.155 ms | 2.2× |
| 2048 | 0.409 ms | 0.153 ms | 2.7× |
| 4096 | 1.422 ms | 0.342 ms | 4.2× |
| 8192 | 5.604 ms | 1.180 ms | 4.7× |

`mask_knn` is faster than `dense_masked` at ALL N, with the advantage
growing from 2.2× (N=128) to 4.7× (N=8192).

But `dense_flash` (no mask) is faster than both at all N.
The sparse advantage over `dense_flash` only appears at N≥8192.

**The window size question:**

Trackastra default: `window=10`
Vanvliet dataset: ~35 cells/frame

| window | N (approx) | Sparse advantage over dense_masked? |
|--------|:----------:|:------------------------------------:|
| 4 | 140-350 | Yes (2.2×) |
| 10 | 350-875 | Yes (2.2×) |
| 20 | 700-1750 | Yes (2.5×) |
| 40 | 1400-3500 | Yes (3×) |

The sparse advantage is measurable at all N, but the absolute time savings
grow with N. At N=140 (window=4), the savings are ~0.2ms per layer —
negligible compared to data loading (~40ms) and other compute.

**Alternative:** Use a dataset with more cells per frame.
DeepCell has ~100-200 cells/frame, so `window=10` → N ≈ 1000-2000.

---

## 4. Do the Trackastra authors account for window size influence on TRA/AOGM?

**Status:** OPEN — needs literature review

The Trackastra paper uses `window=10` as the default but does not report
a sensitivity analysis for window size. The window parameter controls:

1. **Temporal context:** How many consecutive frames are stacked
2. **Sequence length N:** More frames = more cells = larger N
3. **Attention cost:** O(N²) for dense, O(NK) for sparse
4. **Tracking accuracy:** Longer windows may improve temporal association

**Sub-questions:**

1. Does TRA improve with larger window sizes?
2. Is there a point of diminishing returns?
3. Does the optimal window size depend on cell density?
4. Does the spatial cutoff interact with window size?
   (Larger window → more cells → larger N → spatial cutoff more important)

---

## 5. Do the Trackastra authors account for spatial cutoff influence on TRA/AOGM?

**Status:** OPEN — needs literature review

The Trackastra paper uses `spatial_pos_cutoff=256` (pixels) as the default.
This means cells farther than 256 pixels apart are masked out of attention.

**Sub-questions:**

1. Does TRA depend on the spatial cutoff value?
2. Is 256 pixels optimal for all datasets, or dataset-specific?
3. Does removing the spatial cutoff (dense_flash) degrade TRA?
4. Does the spatial cutoff interact with window size?
   (Larger window → cells can drift farther → larger cutoff needed)

---

## Summary of what needs to happen before the K-sweep

1. **Verify FlashAttention + mask** — write a script in `benchmark_attn/analysis/`
   that forces `SDPBackend.FLASH_ATTENTION` with an additive float mask and
   confirms it runs (not falling back to `EFFICIENT_ATTENTION` or `MATH`).

2. **Ablation: spatial cutoff necessity** — train `dense_flash` (no mask)
   vs `dense_masked` (with cutoff). If TRA is within ±0.001,
   the spatial cutoff is unnecessary and `dense_flash` is optimal.

3. **Window size sweep** — train with `window=4, 10, 20, 40` to find
   the N where sparse attention provides measurable gains over `dense_masked`.

4. **K-sweep with `KNNMaskSparseAttention`** — the correct sparse mechanism.
   K=4, 16, 32, 64 vs `dense_masked` baseline. 3 seeds each.
   Measure TRA, AOGM, memory, inference time.

5. **Literature review** — check if Trackastra authors report
   window size or spatial cutoff sensitivity analysis.

---

## Dependencies

- `2026-05-29_gather_sparse_bottleneck.md` — gather overhead analysis
- `2026-05-29_attention_not_bottleneck.md` — attention fraction of training
- `2026-08-14_h100_benchmark_results.md` — H100 attention benchmark
- `2026-08-15_h100_edge_probe_results.md` — edge probe results
