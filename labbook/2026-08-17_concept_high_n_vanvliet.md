# Concept: Achieving High N on Vanvliet Without Degrading Training Quality

**Date:** 2026-08-17
**Status:** DRAFT — concept for achieving high-N benchmarks on vanvliet dataset
**Depends on:** `2026-08-17_open_questions_attention_design.md`

---

## 1. The Problem

The H100 attention benchmark shows that sparse attention (`mask_knn`,
`gather_sdpa`) only provides measurable speedup over dense attention at
N ≥ 2048. At lower N (the actual training regime), dense FlashAttention
is faster.

However, the vanvliet dataset has only ~35 cells per frame. With the
Trackastra default `window=10`, this gives N ≈ 350 — well below the
2048 threshold where sparse attention becomes competitive.

**Naive approach:** Increase `window` to 60 → N ≈ 2100.

**Why this fails:**

1. **Sequence length:** Most vanvliet sequences are 10-30 frames long.
   `window=60` exceeds most sequences, producing padded or synthetic data.
2. **Loss supervision limited:** `delta_cutoff` (default 2) limits *loss
   supervision* to `0 < dt <= delta_cutoff` (train.py:278-284). The attention
   itself has **no temporal cutoff** — only a spatial cutoff
   (model_parts.py:534-535). With `window=60` the model *does* attend ±59
   frames, but only ±2 frames are supervised. The extra context is wasted
   for learning, though not masked at the attention level.
3. **Association sparsity:** With N=2100, the association matrix is
   2100×2100 — extremely sparse (most cells don't divide). This changes
   the training dynamics.
4. **Sliding window reduction:** Current code creates overlapping windows
   (`for t1, t2 in zip(range(0, len), range(window, len+1))`). Dynamic
   window = no sliding = fewer training samples.

---

## 2. The Correct Approach: Multi-Condition Batching

**Key insight:** The `window` parameter controls temporal context, not
batch composition. Instead of increasing `window`, we can **batch multiple
conditions together** to increase the total number of tokens per training
step.

**Current data loading:**

```python
# CTCData loads one condition at a time
CTCData(root=Path(args.input_train[0]), ...)
```

Each condition is a separate sequence. The current approach loads one
condition per sample.

**Proposed approach: Multi-condition batching**

Instead of increasing `window`, we can:

1. **Keep `window=10`** (Trackastra default, gives N≈350 per condition)
2. **Load multiple conditions per sample** (e.g., 6 conditions × 350 = 2100 tokens)
3. **Use condition-aware attention masking** (cells in different conditions
   should not attend to each other)

This gives high N (2100) without degrading training quality:

- Temporal context is preserved (10 frames per condition)
- Training dynamics are preserved (same number of associations per condition)
- The model sees more data per step (6 conditions vs 1)

**Implementation:**

1. Modify `CTCData` to load multiple conditions per sample
2. Add condition-aware attention mask (block-diagonal mask across conditions)
3. Ensure `dist_2d` and `knn_indices` are computed per-condition, not globally

**Pros:**
- High N (2100+) achieved naturally
- Training quality preserved (10-frame window per condition)
- No padding waste (all conditions have similar cell counts)
- More data per step (better gradient estimates)

**Cons:**
- Requires condition-aware attention masking (block-diagonal)
- KNN indices must be computed per-condition (different cell layouts)
- Increased memory per sample (6× more tokens)

---

## 3. Alternative: Length-Grouped Batching

Instead of changing the data pipeline, we can change the batching strategy:

1. **Sort samples by N** (number of cells in the window)
2. **Batch similar-length samples together**
3. **Pad to max N in batch** (minimizes padding waste)

This doesn't increase N itself, but it ensures that the model processes
batches efficiently, without excessive padding.

**Implementation:**

1. Add a `SortByLengthSampler` to the data loader
2. Set `batch_size=1` for memory efficiency (each sample is N≈350)
3. Process one condition at a time (current approach)

**Pros:**
- Simple implementation (just a sampler change)
- No data pipeline changes needed
- Preserves current training dynamics

**Cons:**
- Doesn't actually increase N (still N≈350 per sample)
- Doesn't help with sparse attention threshold (N≥2048)

---

## 4. Alternative: Synthetic High-N Data

For benchmarking purposes only (not training), we can generate synthetic
data with high N:

1. Sample N cell positions from a Poisson point process
2. Assign random features (d_model dimensions)
3. Compute KNN indices, dist_2d, etc.

This is what the current `benchmark_attn` does. It's sufficient for
measuring attention kernel performance but doesn't capture training-
specific overhead (KNN index recomputation, gradient checkpointing, etc.).

---

## 5. Recommendation

For the **benchmark** (measuring attention kernel performance):

- Use the existing `benchmark_attn` synthetic data
- Ensure KNN indices are pre-computed (current behavior)
- Add a "training-realistic" benchmark that includes KNN index computation in timing

For the **K-sweep training experiment** (measuring TRA/AOGM):

- Use `window=10` on vanvliet (Trackastra default)
- Keep `window=10` (N≈350) as the primary training regime
- This is below the sparse advantage threshold (N≥2048)
- But it's the actual training regime, so the benchmark is relevant

For **achieving high N on vanvliet** (if needed):

- Multi-condition batching (Section 2) is the correct approach
- It gives high N without degrading training quality
- Requires condition-aware attention masking

---

## 6. Open Questions

1. **What N does the Trackastra paper actually train at?**
   - Default `window=10`, ~35 cells/frame → N≈350
   - But Trackastra uses `max_tokens` to cap N
   - Paper uses `max_tokens=2048`, `window=6` (Section 2.5)

2. **Does the spatial cutoff (`spatial_pos_cutoff=256`) interact with window size?**
   - Larger window → cells can drift farther → larger cutoff needed
   - But attention has no temporal cutoff — only spatial cutoff
   - `delta_cutoff` limits loss supervision, not attention range

3. **Is N=350 the right benchmark point?**
   - It's the actual training regime
   - But sparse attention doesn't show advantage at N=350
   - The paper may need to acknowledge this limitation

---

## 7. Correction Log

**2026-08-17:** §1 point 2 corrected. Original text claimed `delta_cutoff`
limits temporal attention range and that `window=60` is "masked out anyway."
This is incorrect. `delta_cutoff` only masks the *loss* (train.py:278-284).
The attention itself has **no temporal cutoff** — only a spatial cutoff
(model_parts.py:534-535). With `window=60` the model *does* attend ±59
frames, but only ±2 frames are supervised. The extra context is wasted
for learning, though not masked at the attention level.

---

## 7. Decision Matrix

| Approach | N achieved | Training quality | Implementation effort |
|----------|:----------:|:----------------:|:---------------------:|
| window=60 | ~2100 | Degraded (exceeds sequence length) | Low |
| Multi-condition batching | ~2100 | Preserved (10-frame window per condition) | Medium |
| Length-grouped batching | ~350 (unchanged) | Preserved | Low |
| Synthetic high-N | ~8192 | N/A (benchmark only) | None (existing) |

**Recommendation:** Multi-condition batching for training experiments.
Synthetic high-N for kernel benchmarks. Do NOT inflate window.

---

## 8. Next Steps

1. **Implement multi-condition batching** in `CTCData`
2. **Add condition-aware attention mask** (block-diagonal across conditions)
3. **Cache KNN indices** per-condition in the data pipeline
4. **Port `KNNMaskSparseAttention`** from `benchmark_attn` to `trackastra`
5. **Run K-sweep** with `KNNMaskSparseAttention` at N≈2100 (multi-condition)
6. **Compare** to dense baseline at same N
