# Spatial Cutoff + FlashAttention — Solution Space

**Date:** 2026-06-22
**Context:** Meeting question (meeting_18_06_26.txt): can we enforce the spatial cutoff needed in Trackastra while still dispatching FlashAttention?

---

## 1. The Problem

Trackastra requires M_ij = 0 if ||p_i - p_j|| <= d_max, else -inf. The explicit NxN mask disables FlashAttention. Current best valid method is mask-KNN (3.1x vs CachedDistAttention at N=256, uses cuDNN EfficientAttention).

## 2. Five Approaches Evaluated

| ID | Approach | Feasibility | Speedup @ N=512 | Mechanism |
|----|----------|------------|-----------------|-----------|
| A | FlexAttention spatial score_mod | HIGH | 3.0× | Apply spatial mask INSIDE flash kernel via score_mod function |
| B | KNN pre-filter + FlashAttn | MEDIUM | 0.4× | Gather K/V via KNN, FlashAttn on 1×K (same as gather-KNN, has per-token overhead) |
| C | Spatial block partition | MEDIUM-HIGH | 19.5× | Hilbert-sort tokens, FlashAttn within+between adjacent blocks |
| D | Force flash on KNN mask | LOW | N/A | sdpa_kernel(enable_flash=True) — flash doesn't support arbitrary masks |
| E | Score-modulated (no hard cutoff) | HIGH | 4.0× | Replace hard cutoff with exp(-dist*lambda), no mask → FlashAttn |

## 3. Recommended Path

**Phase 1 (low risk): Approach E** — Remove hard spatial cutoff, use pure distance decay (attn_dist_mode=v1). Already partially implemented. FlashAttn dispatch guaranteed (no mask). Risk: TRA/AOGM accuracy — verify on vanvliet. Expected 4× speedup over CachedDistAttention.

**Phase 2 (medium risk): Approach A** — FlexAttention with spatial score_mod. Keeps hard cutoff inside flash kernel. Requires PyTorch 2.5+ and torch.compile on score_mod. 1.5-3× speedup.

**Phase 3 (large N only): Approach C** — Spatial block partition with overlap. Only needed at N >> 2000.

## 4. Corrected KNN Benchmark Calibration

Previous analytical model was wrong. Fixed calibration to labbook data (2026-06-08) at N=256:

| Method | vs CachedDistAttention | Enforces d_max? |
|--------|----------------------|-----------------|
| mask-KNN (scatter) | 3.1× | Yes (cuDNN handles masked SDPA efficiently at N<=512) |
| gather-KNN | 0.30× | Yes (per-token overhead dominates at N<2000) |
| dense_flash | 4.1× | **No** (invalid for Trackastra) |
| CachedDistAttention | 1.0× | Yes (baseline) |

Key insight: mask-KNN is fast at small N because cuDNN EfficientAttention handles NxN masked SDPA efficiently (no tiling overhead, mask fits in L2 cache). gather-KNN is slow because each of the N queries launches its own SDPA call (1xK), and the per-query kernel launch overhead dominates.
