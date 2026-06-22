# Soft Distance Decay — Origin and Verification

**Date:** 2026-06-22
**Context:** Approach E: remove hard spatial mask, keep soft distance decay, enable FlashAttention.

---

## 1. Origin

Not from an external paper. It's an ablation of Trackastra's own attention design.

Trackastra's attention score (model_parts.py, CachedDistAttention.forward):
```
score_ij = QK^T / sqrt(d) + bias_ij + decay_ij + mask_ij
```
where:
- `bias_ij` = learned positional bias (optional, mode="bias")
- `decay_ij` = exp(-5 * spatial_dist / cutoff)  ← attn_dist_mode=v1
- `mask_ij` = -inf if spatial_dist > cutoff, else 0  ← hard spatial cutoff

**Insight:** The v1 decay already suppresses cells beyond d_max to e^{-5} ≈ 0.0067.
The hard mask may be redundant. Removing it enables FlashAttention.

Supporting references:
- Trackastra (Gallusser & Weigert, ECCV 2024): origin of v1 distance decay
- FlexAttention (PyTorch 2.5+, `torch.nn.attention.flex_attention`): mechanism to apply score_mod inside flash kernel without materializing N×N bias

---

## 2. Verification (3 phases)

**Phase 0 — Weight analysis:**
- λ=5, d_max=256: weight at d_max = 0.0067 (nearly zero)
- weight at 1.5×d_max = 0.00055 (effectively zero)
- 0.0% of far cells get attention weight > 0.01
- ✓ PASS

**Phase 1 — Forward pass equivalence:**
- Synthetic Q/K/V/coords, compare hard mask output vs soft decay output
- Cosine similarity > 0.9999 across N=32..1024
- Max absolute difference < 0.07 across all trials
- ✓ PASS — outputs are functionally identical

**Phase 2 — TRA/AOGM on vanvliet:**
- Full training with soft decay only (no hard mask)
- Config: `attn_dist_mode=v1`, remove hard spatial mask, enable `sdpa_kernel(enable_flash=True)`
- Compare TRA against baseline 0.9972
- TBD — needs Capella run

---

## 3. Speed vs dense_flash

Soft decay is NOT as fast as pure dense_flash (no cutoff at all).
10% overhead from FlexAttention score_mod function applied inside flash kernel.

| N=256 | Time | Mechanism |
|-------|------|-----------|
| dense_flash | 0.20ms | Pure FlashAttn, no modifications |
| soft decay | 0.22ms | FlashAttn + FlexAttention score_mod |
| hard mask | 0.80ms | cuDNN masked SDPA (current) |

---

## 4. Files

| File | Change |
|------|--------|
| `benchmark_attn/benchmark_soft_decay.py` | Phase 0 + Phase 1 verification benchmark |
| `benchmark_attn/soft_decay_forward_equivalence.csv` | Forward pass cos_sim per N |
| `labbook/2026-06-22_soft_decay_origin.md` | This entry |
