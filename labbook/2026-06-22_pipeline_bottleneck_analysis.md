# Trackastra Pipeline — Compute & Memory Bottleneck Analysis

**Date:** 2026-06-22
**Context:** Meeting question (meeting_18_06_26.txt line 19): "What is the most memory and compute intensive task in trackastra (training and inference) if its likely not attention?"

---

## 1. TL;DR — Attention is NOT the bottleneck

The meeting's suspicion is correct. The most intensive operations are:
- **Compute (training):** Serial `blockwise_causal_norm` loop in loss + redundant per-layer 3D cdist
- **Compute (inference):** CPU-bound `regionprops_table` + serial `blockwise_causal_norm` in `normalize_output`
- **Memory (training):** Stored N×N attention masks + FFN activations across 12 layers for backprop

---

## 2. Training Pipeline Breakdown (per step)

| Stage | Op | Compute | Memory |
|-------|----|---------|--------|
| Data | `regionprops_table` (CPU) | **Dominant** for small N | ~1-5 GB RAM |
| Data | LZ4 decompress, aug, crop | Medium | ~500 MB |
| Forward | Pos embed (Fourier sin/cos) | O(N·D), negligible | ~N·D·4B |
| Forward | 2D cdist (once) | O(N²·2) | B×N²×4B (~128MB) |
| Forward | Encoder attn × 6 | O(N²·dH) per layer | N×N masks stored |
| Forward | Decoder attn × 6 | O(N²·dH) per layer | N×N masks stored |
| Forward | FFN × 12 | O(N·4D²) per layer | ~384MB total |
| Forward | Einsum (outer product) | O(N²·D) | B×N²×2B |
| Loss | `blockwise_causal_norm` | **Serial loop over batch**, O(N²)·4×scatter_reduce per sample | Intermediate N×N |
| Loss | BCE/Loss weights | O(N²) | Negligible |
| Backward | All activations retained | — | **3-10× forward mem** |

---

## 3. Key Bottleneck #1: Serial `blockwise_causal_norm`

Located at `scripts/train.py:228` — operates per-sample in Python loop:
```python
for b in range(B):
    A_pred_soft_norm[b] = blockwise_causal_norm(A_pred_soft[b], ...)
```

Inside: 4× `blockwise_sum` calls using `torch.scatter_reduce` (amax + sum × 2 dimensions each). For B=8, N=2000, this loop is the dominant GPU compute in the loss. The code has a TODO: "I could softmax only the part of the matrix."

Same issue in inference: `normalize_output()` at `model.py:505` — serial blockwise_causal_norm loop.

## 4. Key Bottleneck #2: CPU Feature Extraction

`WRFeatures.from_mask_img()` calls `skimage.measure.regionprops_table()` — a CPU-only operation. Even with 8 joblib workers, this dominates wall time when the GPU is fast (e.g., H100). For inference, `get_features()` can be 50-80% of total time at small N.

## 5. Key Bottleneck #3: Redundant per-layer 3D cdist (v0 mode)

In `CachedDistAttention.forward` (v0 mode), `torch.cdist(coords, coords)` in full 3D coordinates is called **per layer**. With 12 layers, this is 12× O(N²·3). CachedDistAttention pre-computes the 2D distance once but the 3D distance decay in v0 mode is still recomputed.

Switching to v1 (`attn_dist_mode: v1`) eliminates this: uses `exp(-5 * spatial_dist / cutoff_spatial)` where spatial_dist is the pre-computed 2D cdist.

## 6. Key Bottleneck #4: Memory from Stored Attention Masks

For training with N=2000, B=8, nhead=4, fp16:
- Per-layer attention mask: 8×4×2000×2000×2B = 256 MB
- ×12 layers = ~3 GB just for masks
- FFN activations across 12 layers = ~384 MB
- Total activation memory: 4-12 GB

For inference (no_grad), activations are NOT stored — memory drops to ~1-3 GB.

---

## 7. Comparison: Attention vs Other Ops (N=500, B=4)

| Operation | Approx FLOPs | Notes |
|-----------|-------------|-------|
| 6 encoder attentions | 6 × N² × d_head × 2 | ~6 × 250K × 40 × 2 = 120M |
| 6 decoder attentions | 6 × N² × d_head × 2 | ~120M |
| 12 FFN layers | 12 × N × 4D² | 12 × 500 × 4 × 320² = ~2.5G |
| Final einsum | N² × D | 250K × 320 = 80M |
| **Total attention** | | **~320M FLOPs** |
| **Total FFN** | | **~2.5G FLOPs** |
| blockwise_causal_norm | 4 × scatter_reduce × B | **Serial, hard to FLOP-count** |

FFN is 8× more FLOPs than attention at N=500!

---

## 8. Recommendations

1. **Vectorize `blockwise_causal_norm`**: Batch the per-sample normalization instead of serial loop
2. **Use v1 dist mode**: Eliminates 12× redundant 3D cdist
3. **Consider gradient checkpointing for FFN**: FFN activations dominate memory
4. **Profile CPU feature extraction**: `regionprops_table` may be the bottleneck for fast GPUs
5. **For large N**: Gather-KNN attention reduces O(N²) mask memory; FFN still dominates compute
