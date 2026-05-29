# Honest Assessment: What Is Proven vs Hypothized

**Date:** 2026-05-23

---

## The 5x convergence claim — provenance and limits

### Source
`benchmark_combined/` ablation: synthetic identity-association task, tiny model (d=128, L=4, 647K params), N=512 cells, B=2.

| Variant | Val Loss (epoch 15) | Epochs to reach 0.15 |
|---------|---------------------|-----------------------|
| dense+rand | 0.504 | never (best ~0.35) |
| sparseK4+rand | 0.108 | ~4 |
| sparseK16+rand | 0.106 | ~4 |

The "5x" comes from: sparse reaches 0.15 in 4 epochs; dense needs 15+ (extrapolated ~20) to reach even 0.35. Not a clean multiplier — it is "dense never reaches sparse quality."

### What makes this unreliable for the real system

1. **Tiny model.** d=128 vs Trackastra's d=320. At d=128, the model capacity is small enough that attending to distant cells actively hurts (wastes limited representational budget on noise). At d=320, the model may have enough capacity to use spatial information productively.

2. **Synthetic task.** The ablation uses identity matching (same distortion pipeline as SSL). On real tracking data (consecutive-frame linking with biological motion), the spatial prior matters more — cells don't teleport, so distant cells ARE irrelevant. Trackastra's `delta_cutoff` already encodes this.

3. **No Trackastra decoder.** The ablation model is encoder-only. Trackastra has encoder+decoder (cross-attention across time). The decoder's cross-attention already provides time awareness. KNN's benefit might be absorbed by the decoder.

4. **Trackastra already masks.** `attn_mask` with `delta_cutoff: 1` zeroes attention to cells beyond a spatial radius. It's already sparse in effect, just not in implementation. KNN skips the materialization overhead, but the effective computation (softmax over valid entries) is similar.

### What IS proven

- **Per-step attention throughput**: KNN K=4 is 1.4x faster at N=2048, 5.9x at N=8192 (measured on 4090, single layer). This is a hardware measurement, not a model claim.
- **Memory**: Sparse K=32 uses 1.8x less memory at N=2048, fits where dense OOMs at N=8192. Hardware measurement.
- **Enabling FlashAttention**: KNN gather (no attn_mask) lets PyTorch dispatch to FlashAttention-2. Measurable, reproducible.

### What must be measured before claiming 5x

Run Trackastra-dense (base-config, d=320, L=12, 500ep) vs Trackastra-KNN (same config, K=32 gather) on H100. Plot TRA vs epoch. If KNN converges faster or to higher TRA, report the actual ratio. If they're identical, the "5x" claim is dead and KNN's contribution is limited to per-step speedup + FlashAttention enablement.

**Hypothesis before measurement**: KNN gives maybe 1.3-2x convergence speedup on real Trackastra, not 5x. The toy model exaggerates because at low capacity, restricting attention is a much stronger regularizer.

---

## Why wouldn't Trackastra authors use KNN gather?

1. **FlashAttention didn't exist** when Trackastra was written (2023). Without FlashAttention, gather-based sparse attention was actually SLOWER than masked dense SDPA at N<2048 because gather overhead + kernel launch latency dominates O(NK) savings. The benchmark shows: at N=512, sparse is 0.6x (slower).

2. **The mask is not binary.** Trackastra's `attn_mask` encodes a **distance-weighted bias** (`delta_cutoff` controls the radius, `attn_dist_mode: v1` controls the falloff). It's not "attend to K neighbors" — it's "attend to all cells, but weight by spatial distance." KNN throws away distance information beyond K. Retaining distance weighting with KNN requires additional engineering.

3. **Not the bottleneck.** At N≈2000, attention is <20% of total step time. Feature extraction (regionprops + wrfeat), mask I/O, data loading dominate. Optimizing attention is a mis-targeted optimization at current scale.

---

## Why `crop_size` in config?

`crop_size: [320, 320]` limits the image tile extracted around each cell for wrfeat computation. It controls the per-cell pixel budget (320×320 = 102K pixels). At 500 cells/window × 102K pixels = 51M pixels → 100MB+ in float32. The image I/O and feature extraction dominate runtime. Reducing `crop_size` speeds up training. It is unrelated to attention N.

The config also has `max_tokens: 2048` which is a soft packing limit — sample windows until total tokens exceed this, then truncate. It saves memory by not processing windows with unusually many cells.

---

## Why low-label regime is the key contribution

Without low-label benefit, the entire project delivers:
- Sparse KNN attention: 1.4x per-step speedup, FlashAttention, lower memory. Engineering improvement.
- SSL pretraining: contrastive embeddings that don't help at full labels (see ablation: sparseK4+SSL = sparseK4+rand = 0.108).

These are not enough for a conference paper. They are incremental Trackastra optimizations.

With low-label benefit (10% labels, SSL+KNN > KNN alone > dense baseline):
- **Scientific contribution**: structural inductive bias (KNN) + self-supervised representation learning (SSL) jointly enable tracking from minimal annotation. This addresses the proposal's core motivation: "annotated tracking data is scarce and expensive."
- The finding is: at 100% labels, supervised training dominates and SSL adds nothing; at ≤10% labels, the pretrained representations compensate for missing labels. This is consistent with SimCLR, BYOL, etc. in vision — self-supervised pretraining helps most when downstream labels are scarce.

The label fraction sweep is the **make-or-break experiment**. If SSL shows zero benefit even at 1% labels, the contrastive objective or augmentation strength needs reworking, or the KNN structural prior already provides all the regularization the model needs.
