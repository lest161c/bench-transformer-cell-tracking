# Why Trackastra Implements the Spatial Cutoff

**Date:** 2026-08-17
**Status:** COMPLETE — re-read of Gallusser & Weigert (2024), arXiv:2405.15700v2

---

## The spatial cutoff in the paper

The paper introduces the spatial cutoff in **Section 2.2** (Eq. 3):

> 𝒜(Q, K, V) = softmax(QKᵀ/√d + M) V
>
> where M is a mask disabling attention for all token pairs whose distance
> is larger than a user defined threshold d_max, i.e. M_ij = 0 if
> ||p_i − p_j||₂ ≤ d_max and M_ij = −∞ otherwise.

This is the *only* place the paper describes the spatial cutoff. It is
presented as a structural design choice — not benchmarked, not ablated,
not justified with experiments. The ablations (Section 3.6) sweep
parental softmax, transformer size, and window size — **but never test
removing or varying the spatial cutoff**.

## Why they did it (from the paper text)

1. **Biological prior.** Cells only interact within a biologically
   meaningful radius d_max. The paper states this implicitly: the mask
   encodes the prior that "cells should only interact within d_max"
   (our argumentation.tex, derived from the paper's Eq. 3).

2. **Computational efficiency at high N.** The paper notes that
   "Trackastra association prediction scales to videos with thousands
   of objects" (Section 2.5). Without the spatial cutoff, dense
   attention is O(N²) in both compute and memory — at N=2048 this
   would be prohibitive. The spatial cutoff makes the *effective*
   receptive field O(k) per query, where k is the number of cells
   within d_max.

3. **Preventing global shortcuts.** Without the spatial cutoff, every
   cell attends to every other cell. The model could learn global
   position shortcuts (e.g., "cells at position X always associate
   with cells at position Y") rather than local cell-identity features.
   The spatial cutoff forces the model to learn local associations,
   which is the inductive bias the authors want.

4. **No sensitivity analysis.** The paper does not report any
   sensitivity analysis for the spatial cutoff value (d_max=256
   pixels in the code). This is a gap in the paper that our research
   addresses.

## What the paper does NOT say

- The paper does **not** claim the spatial cutoff is necessary for
  optimal TRA. It is presented as a design choice, not a hyperparameter
  sweep.
- The paper does **not** benchmark the spatial cutoff against no-cutoff
  (dense_flash). This is our ablation (job 3920745).
- The paper does **not** report the interaction between spatial cutoff
  and window size.

## Implications for our research

Our dense_flash ablation (job 3920745) is the first test of whether
removing the spatial cutoff degrades TRA. If TRA is within ±0.001 of
the masked baseline, the spatial cutoff is unnecessary for tracking
accuracy — and dense_flash (no mask) is optimal, even though it uses
EfficientAttention (not FlashAttention-2) because the mask is gone but
the attention is still dense.

However, even if the spatial cutoff is unnecessary for TRA, it may
still be beneficial for:
- **Memory efficiency** at high N (dense attention is O(N²) memory)
- **Training speed** at high N (dense attention is O(N²) compute)
- **Generalization** (preventing global position shortcuts)

The spatial cutoff is a **regularizer**, not just a compute optimization.
This is consistent with the paper's framing of the mask as a structural
prior, not a performance optimization.

## Summary

| Question | Answer |
|----------|--------|
| Do the authors justify the spatial cutoff? | No — it's a design choice, not ablated |
| Do the authors sweep d_max? | No — d_max=256 is fixed throughout |
| Do the authors test removing the cutoff? | No — this is our ablation |
| Is the cutoff for compute or accuracy? | Both — but the paper frames it as a structural prior |

The spatial cutoff is **not a hyperparameter** in the paper — it is a
**structural design choice** that encodes the biological prior of local
cell interactions. Our research tests whether this prior is necessary
for optimal TRA, or whether it can be removed (dense_flash) without
degrading tracking accuracy.
