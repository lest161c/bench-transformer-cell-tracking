# Inflating N via Batch Size — Why It Doesn't Help K16

**Date:** 2026-05-28

---

## 1. batch_size increases parallelism, not N

In Trackastra, each batch element is an **independent** temporal-spatial window. Attention is computed per-window, independently. batch_size=B runs B attention calls in parallel — each on the same N=~350 cells.

```
batch_size=8:  8 × attention(N=350)
batch_size=48: 48 × attention(N=350)
```

Both dense and KNN benefit equally from more batch elements. The ratio dense_vs_KNN depends on N per call, not B.

---

## 2. Actual N in training is far below max_tokens

| Factor | Value | Result |
|--------|-------|--------|
| Cells/frame (vanvliet avg) | ~35 | — |
| Window | 4 (baseline) | N ≈ 140 |
| Window | 10 (K16) | N ≈ 350 |
| max_tokens | 2048/4096 | Cap, never hit |

The speedup bench shows KNN crossover at N≈2000-3000 (see §3 below). Actual training N=140-350 is in the **slowdown zone** where KNN overhead > dense cost.

---

## 3. KNN speedup depends on N per attention call

From `benchmark_attn/results_analysis.md` (4090, L=1):

| N | Dense (ms) | KNN K=16 (ms) | Ratio |
|---|-----------|-------------------|-------|
| 128 | 6.2 | 17.2 | **0.36×** (slower) |
| 256 | 16.1 | 37.4 | **0.43×** |
| 512 | 53.5 | 88.7 | **0.60×** |
| 1024 | 182 | 208 | **0.88×** |
| 2048 | 8.5 | 8.9 | **0.95×** (breakeven) |
| 8192 | 139 | 36.1 | **3.85×** (fast) |

Crossover ≈2000-3000. H100 with FlashAttention-2 pushes crossover lower (~1500-2000, estimated), but still above N=350.

**KNN overhead (per attention call):**
1. `cdist(N, 2)`: pairwise spatial distance → O(N²·d), d=2 → cheap
2. `topk(N², K)`: find K nearest neighbor indices → O(N²) on GPU
3. `gather(K, N, d_model)`: collect neighbor features → O(K·N·d_model)
4. `SDPA(Q, K_gathered, V_gathered)`: attention on K·N pairs → O(K·N·d_model)

Total sparse: O(N²) [cdist+topk] + O(K·N) [gather+SDPA]. At low N, N² term is small but KNN constant factors dominate. At high N, KNN's linear-K·N term wins over dense's N².

---

## 4. Why dense baseline isn't fully N² either

Baseline uses `spatial_pos_cutoff=256` and `attn_mask` to prevent distant cells attending. This is **spatial sparsity via mask** — still computes QK^T for all pairs, but zeros out distant ones. FlashAttention-2 cannot run with custom attn_mask (memory-bandwidth-limited on H100 — see §2A in speedup analysis).

K16 enables FlashAttention-2 by not using attn_mask (KNN neighbors are the mask). This is the theoretical H100 advantage, distinct from the N-scaling advantage.

---

## 5. How to actually inflate N

| Method | Increases N? | Breaks semantics? |
|--------|-------------|-------------------|
| Larger window | Yes (more frames × cells/frame) | No (temporal tracking improves) |
| Larger max_tokens | Removes cap | No |
| Higher batch_size | No (parallel, independent) | — |
| Merge batch dim into N | Yes (concatenate cells) | **Yes** (unrelated cells attend) |
| Larger cells/frame datasets | Yes | No |
| Synthetic scaling bench | Yes (any N) | N/A (not real tracking) |

**Larger window** is the only viable path to inflate N without breaking tracking semantics. window=10 → N≈350. window=50 → N≈1750. window=100 → N≈3500. At window=100 (≈10 min of video), KNN would show 2-3× speedup.

Trade-off: longer windows → more GPU memory (FFN, cross-attention scale linearly), more temporal context, potentially harder optimization. But tracks across longer time spans are more biologically meaningful.

---

## 6. Strategic recommendation

**Clean separation of concerns for the report:**

1. **Per-step speedup claim:** Synthetic bench (4090/H100), N=8192, K=16 → 3.85-5.88×. Clean, controlled, no data interplay. Already done.

2. **K16 as structural regularizer:** Identical config comparison (same window, max_tokens, dropout, input_train). K16 vs dense both at N≈350. Measured: val_loss vs wall-clock. K16 wins via fewer training steps (regularization), not via per-step speed. This is the "fair" comparison.

3. **Larger-N experiment (optional, high-impact):** Increase window to 50-100 for both. Baseline will slow down ~25-100× (N²). K16 will slow down ~5-10× (linear-K). Difference proves scaling advantage on real data.

Do NOT inflate N via batch_size — it's a category error. Batch parallelism benefits both models equally. The speedup lives in the O(N²) vs O(K·N) scaling of a SINGLE attention call.
