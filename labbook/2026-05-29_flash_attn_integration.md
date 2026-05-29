# CachedDistAttention — Amortized distance matrix for spatial-cutoff attention

**Date:** 2026-05-29 (revised)  
**Branch:** `cached-dist-attn`  
**Depends on:** `2026-05-29_gather_sparse_bottleneck.md`, `2026-05-29_knn_regularizer_critique.md`

---

## 1. What changed

Replaced the original `RelativePositionalAttention` with `CachedDistAttention` — a functionally identical attention layer that accepts a pre-computed 2D pairwise distance matrix instead of re-computing `cdist` internally.

**Single change:** `cdist(yx, yx)` moved from inside each attention layer (computed L× per forward pass) to `TrackingTransformer.forward()` (computed once, shared across all L layers).

## 2. Mechanism

```
Original (per layer):    CachedDistAttention (once + share):
                         
  cdist(yx, yx)  56us    dist_2d = cdist(yx, yx)  ← in forward(), once
  threshold       5us     ───────── pass to all layers ─────────
  masked_fill     24us    
  distance_decay  24us    threshold        5us      }
  SDPA            33us    masked_fill     24us      } per layer
  ────────────────────    distance_decay   24us      }
  Total/layer:   142us    SDPA             33us      }
                          ─────────────────────
                          Total/layer:    86us
                          
  L=6: 6 × 142 = 852us    L=6: 56 + 6×86 = 572us  → 1.5× faster
  L=12:12×142 = 1704us   L=12:56 + 12×86 = 1088us → 1.6× faster
```

At training scale (L=12, Trackastra default): ~1.5× per-step speedup from this change alone. Additional gains from removing the 3D cdist variant (v0 distance decay uses 3D coords — still per-layer, but the main win is from the 2D cdist amortization).

**Key insight:** No KNN, no FlashAttention, no scatter mask. The win is purely from recognizing that the 2D pairwise distances between N cells are identical across all attention layers within one forward pass.

## 3. Class: `CachedDistAttention`

```python
class CachedDistAttention(nn.Module):
    # forward(query, key, value, coords, padding_mask, dist_2d=None)
    # If dist_2d provided: uses it for spatial cutoff threshold
    # If dist_2d is None: falls back to per-layer cdist (backward compat)
    # Otherwise identical to RelativePositionalAttention
```

File: `trackastra/trackastra/model/model_parts.py:440`

## 4. No configuration needed

The distance caching is automatic — `TrackingTransformer.forward()` computes `dist_2d` once and passes it through the layer chain. No new flags, no config options.

## 5. Files modified

| File | Change |
|---|---|
| `model_parts.py` | Added `CachedDistAttention` (accepts optional `dist_2d` kwarg) |
| `model.py` | `dist_2d = cdist(yx, yx)` in `forward()` and `encode()`; passed to EncoderLayer/DecoderLayer |
| (EncoderLayer, DecoderLayer) | Added `dist_2d` parameter, passed through to `self.attn(dist_2d=dist_2d)` |
