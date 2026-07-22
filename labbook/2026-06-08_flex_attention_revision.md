# The Mask Indirection Principle — Revised with flex_attention

**Date:** 2026-06-08 (revised after literature/API review)
**Depends on:** `2026-06-08_mask_indirection_principle.md`
**Supersedes:** Section 4 of `2026-06-08_mask_indirection_principle.md` (the escape route is real and available)

---

## 1. Restated Principle

The SDPA API offers exactly two mechanisms for heterogeneous per-query neighbor sets \(\mathcal{N}(b,h,i)\): **mask** (N² tensor → single kernel launch, forced Off-FlashAttention dispatch) or **gather** (N kernel launches, no mask, forced q_len=1 dispatch). Our earlier analysis concluded these were exhaustive. They are not.

**A third mechanism exists in PyTorch 2.5+ prior work and implementations confirmed:**

\[\text{FlexAttention score\_mod} \; \Longrightarrow \; \text{1 kernel launch, 0 N² materialization, heterogeneous neighbors}\]

The key sources:
- PyTorch `flex_attention` blog post (Oct 2024): https://pytorch.org/blog/flexattention/ — demonstrates fused score modification without materializing N² masks
- PyTorch `__init__.py`: `from torch.nn.attention.flex_attention import flex_attention, create_block_mask`
- PyTorch source: `torch/nn/attention/flex_attention.py` ~2500 lines — Triton-kernel implementation
- `xformers` `BlockSparseAttention` — predecessor approach, now superseded by FlexAttention

---

## 2. How flex_attention Escapes the Tradeoff

**The conventional SDPA interface** couples attention compute and mask semantics:

```python
y = F.scaled_dot_product_attention(q, k, v, attn_mask=M)
# M ∈ R^(B,H,N,N)  →  pre-materialized, blocks FlashAttention dispatch
```

**FlexAttention decouples them** — the masking logic is a Python function `score_mod` that PyTorch compiles into the Triton attention kernel itself. No intermediate N² tensor ever exists in HBM:

```python
import torch
from torch.nn.attention.flex_attention import flex_attention

dist_2d = torch.cdist(yx, yx)  # (B, N, N), precomputed ONCE, cached

def spatial_cutoff_mod(score, b, h, q_idx, kv_idx):
    d = dist_2d[b, q_idx, kv_idx]
    return torch.where(d <= d_max, score, float('-inf'))

# SINGLE kernel launch. ZERO N² materialization. Per-element scoring fused in-kernel.
y = flex_attention(q, k, v, score_mod=spatial_cutoff_mod)
```

The `score_mod` function runs **inside the GPU Triton kernel** at every (b, h, q_idx, kv_idx) position. PyTorch's `torch.compile` lifts captured tensors (`dist_2d`, `d_max`) into kernel arguments — no recompilation needed when tensor *values* change, only when the code of `score_mod` changes.

### Block sparsity for additional speed

When `d_max` ≪ total spatial extent, most query-key pairs are masked out. `create_block_mask` pre-computes full-skip blocks:

```python
from torch.nn.attention.flex_attention import create_block_mask

def spatial_block_mask(b, h, q_idx, kv_idx):
    return dist_2d[b, q_idx, kv_idx] <= d_max

block_mask = create_block_mask(spatial_block_mask, B=B, H=None,
                               Q_LEN=N, KV_LEN=N, _compile=True)

y = flex_attention(q, k, v, block_mask=block_mask)
# Entire blocks where ALL pairs exceed d_max are skipped → further speedup
```

`create_block_mask` costs hundreds of µs (run once per sample, amortized across all L layers). For our static-coordinate problem, the `BlockMask` can be computed once.

---

## 3. Performance Expectations

Published benchmarks (PyTorch flex_attention blog, Oct 2024):

| Configuration | Backend | Relative speed |
|---|---|---|
| No mask, causal | FlashAttention-2 | 1.00× (baseline) |
| Causal, sliding window via flex_attention | Triton (compile) | ~0.90× FA2 |
| Causal, sliding window via FA2 `window_size` | FlashAttention-2 | ~1.00× FA2 |
| Causal, sliding window via explicit N² mask | EfficientAttention | ~0.50× FA2 |
| **Our case:** dense + spatial cutoff via flex_attention | Triton (compile) | ~0.85–0.90× FA2 (estimated) |
| **Our case:** dense + spatial cutoff via explicit N² mask | EfficientAttention | ~0.35× FA2 (measured: 33 µs vs 12 µs at N=256) |

For our spatial cutoff at N=256, we estimate:
- Current best (KNNMask + EfficientAttention): 33 µs
- Expected (flex_attention + score_mod): **~14–17 µs**
- The unmasked floor (pure FA2, incorrect): 12 µs

The mask dispatch penalty of 21 µs reduces to a compiled-in-kernel overhead of ~2–5 µs.

---

## 4. Caveats

1. **Compilation overhead.** First call triggers `torch.compile` — expect 5–30s compilation on first bench run. Subsequent calls (same `score_mod` code) are instant.

2. **GPU architecture support.** FlexAttention requires Triton — supports NVIDIA Ampere+ (SM80+). Our A500 (SM86) is supported. CUDA 11.8+ required.

3. **Explicit spatial cutoff in score_mod, not mask.** Trackastra Equation 3 uses an additive mask \(M_{ij}\) that includes both the cutoff \(-\infty \cdot \mathbb{1}[d > d_{\text{max}}]\) and a distance decay term \(\exp(-0.1 \cdot \|p_i - p_j\|)\). The spatial cutoff goes into `flex_attention`'s `score_mod`. The distance decay term can also be fused into the same `score_mod`:
   ```python
   def spatial_mod(score, b, h, q_idx, kv_idx):
       d = dist_2d[b, q_idx, kv_idx]
       score = torch.where(d <= d_max, score, float('-inf'))
       score = score + torch.exp(-5 * d / d_max)  # Trackastra distance decay
       return score
   ```

4. **BlockMask cost.** `create_block_mask` is relatively expensive (~hundreds of µs). Only worth it when block-level sparsity exceeds ~50%. At our dataset N ≈ 140 with high cell density, most blocks are partially active — `score_mod` alone is sufficient.

5. **Not yet tested.** The benchmark numbers above are estimates from published data, not our own measurements. A focused benchmark under `benchmark_attn/` is required to confirm.

---

## 5. Relation to Our Earlier Work

| Date | Finding | Status after this |
|---|---|---|
| 2026-05-29 | "Mask tax" of 21 µs is irreducible under SDPA API | **Revised.** flex_attention avoids it. |
| 2026-05-29 | Gather fragments attention into N launches → worse | **Unchanged.** Still true, fixed by flex_attention differently. |
| 2026-06-08 | Mask indirection principle — only mask and gather exist | **Revised.** Third mechanism: score_mod, one launch, no materialization. |
| 2026-06-08 | Attention is not the training bottleneck at N ≈ 140 | **Unchanged.** Even at 14 µs, attention is 0.004% of a training step. |

---

## 6. Implementation Path

For `benchmark_attn/`:

1. Add `FlexAttentionSpatial` to `model_parts.py` — wraps `flex_attention` with spatial `score_mod` using a precomputed `dist_2d`
2. Add `bench_flex_spatial()` to `benchmark_sparse.py`
3. Compare against `KNNMaskSparseAttention` and `CachedDistAttention` at N = 128–8192
4. Measure compilation overhead separately (first-call cost)

For training pipeline (if attention ever becomes the bottleneck):
- Replace `CachedDistAttention`'s `F.scaled_dot_product_attention(q, k, v, attn_mask=M)` with `flex_attention(q, k, v, score_mod=spatial_cutoff_mod)` — identical semantics, faster dispatch.

---

## 7. Broader Context

The FlexAttention approach represents the consensus direction in the PyTorch ecosystem:

- **xFormers BlockSparseAttention** (Massa et al.) → predecessor, now integrated into FlexAttention as `BlockMask`
- **HazyResearch flash-attention** → supports only `causal`, `window_size`, `alibi_slopes` natively; general sparsity deferred to FlexAttention
- **PyTorch 2.5+ FlexAttention** → the canonical API for custom attention patterns in PyTorch

The community recognized that enumerating every sparse pattern in C++/CUDA (FlashAttention's approach) doesn't scale. Instead, the solution is a Python-level DSL (`score_mod`, `mask_mod`) that compiles to fused Triton kernels. Our spatial cutoff is a textbook use case for this pattern.
