# KNNMaskSparseAttention Integration into Trackastra

**Date:** 2026-05-29 (revised)  
**Branch:** `flash-attention`  
**Depends on:** `2026-05-29_gather_sparse_bottleneck.md` (rationale), `2026-05-29_knn_regularizer_critique.md`

---

## 1. What changed

Replaced the original `RelativePositionalAttention` (cdist + masked_fill + distance decay) with `KNNMaskSparseAttention` — a KNN-derived N×N scatter mask that preserves the spatial cutoff while avoiding the expensive cdist mask construction.

**Rationale:** The Trackastra paper (Eq. 3) defines `M_ij = -∞ if ||p_i - p_j|| > d_max`. The spatial cutoff is structural, not an optional ablation. We cannot remove it and claim validity. `KNNMaskSparseAttention` preserves it via KNN indices computed from 2D spatial coordinates, then builds the N×N mask via a cheap `scatter_` operation instead of the original `cdist` + `masked_fill` + `add_` pipeline (82% of CUDA time).

**Measured speedup:** 2.5–3.1× faster than the original dense_masked at N=128..512.

## 2. Performance comparison (N=256, K=16, fp16, A500)

| Method | Time | Memory | Spatial Cutoff |
|---|---|---|---|
| dense_masked (original) | 360 us | 10.4 MB | mask via cdist+fill+add |
| gather_sparse (KNN) | 1061 us | 12.8 MB | via KNN gather (slow) |
| **mask_knn (NEW)** | **131 us** | **10.4 MB** | via KNN scatter (fast) |

## 3. New class: `KNNMaskSparseAttention`

```python
class KNNMaskSparseAttention(nn.Module):
    # Builds NxN mask from knn_indices via scatter_ (O(NK))
    # Runs F.scaled_dot_product_attention(q,k,v, attn_mask=mask)
    # Native (B, nH, N, Dh) shape — avoids q_len=1 problem
    # Uses EfficientAttention (masked SDPA backend)
```

File: `trackastra/trackastra/model/model_parts.py:440`

## 4. Configuration

**`TrackingTransformer.__init__`** parameter:

```python
knn_size: int = 16   # K in KNN — spatial neighbours to attend to
```

Train script flag: `--knn_size 16` (default). Controls the sparsity of the spatial mask.

## 5. Files modified

| File | Change |
|---|---|
| `model_parts.py` | Replaced `DenseFlashAttention` with `KNNMaskSparseAttention` |
| `model.py` | Factory uses `KNNMaskSparseAttention`; restored KNN index computation; `knn_size` parameter |
| `train.py` | `--knn_size` flag (default 16); removed `--flash_attn` |
| `ssl_trainer.py` | Replaced `--flash_attn` with `--knn_size` |

## 6. Why not DenseFlashAttention (no mask)?

The Trackastra paper §2.2 Eq. 3 defines the spatial cutoff mask `M` as a core architectural component. Removing it changes the model's inductive bias (all cells attend to all cells, not just spatial neighbors). RoPE alone may compensate, but this is unverified. `KNNMaskSparseAttention` preserves the spatial cutoff and is 3× faster than the original — no reason to drop the mask.
