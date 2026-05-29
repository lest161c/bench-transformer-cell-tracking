# SSL Memorization Analysis: Identity BCE on Full TrackingTransformer

**Date:** 2026-05-29  
**Context:** Fix SSL pipeline from contrastive NT-Xent (benchmark_ssl) to proposal §4.2 identity BCE on full model. Primary concern: model will memorize inverse of geometric distortions instead of learning transferable feature-based associations.

---

## The Memorization Problem

With identity BCE pretraining:

```
frame0: cell_i at pos_i, features f_i
frame1: cell_i at T(pos_i), features f_i + noise
A[i,i] = dot(head_x(enc(cell_i)), head_y(dec(cross-attn, cell_i)))

Ground truth: A[i,i] = 1, A[i,j≠i] = 0
```

Model can achieve zero BCE by:
1. Learning to estimate distortion T from coordinate pairs
2. Computing T⁻¹(pos_j) in decoder cross-attention
3. Scoring high for cells at T⁻¹(pos_j) ≈ pos_i

Features contribute nothing. Positional encoding pathway dominates.

### Shortcut pathway details

| Component | Learned behavior during SSL | During real tracking |
|-----------|---------------------------|---------------------|
| Encoder | Ignore features, pass through coordinates | Must encode feature-based identity |
| Decoder cross-attn | Attend to cell at T⁻¹(query_position) | Attend to same cell at biologically-valid position |
| Association head | score(pos_i, T(pos_i)) ≈ 1 | score(pos_i, pos_i+1) unknown across real frames |

**Worst case**: pretrained weights become adversarial initialization. Fine-tuning must unlearn positional shortcut before learning real associations → convergence slower than from scratch.

### When shortcuts are possible

| Distortion type | Invertibility | Shortcut type | Risk |
|----------------|-------------|---------------|------|
| Affine (rotation/scale) | 6 params, exact inverse | Global parametric | Very high |
| Elastic deformation | Smooth field, grid_size^2 params | Approximate, batch-specific | Medium-high |
| Per-cell jitter | N×2 params, no structure | No inverse possible | Low |
| Feature noise | Independent, high-dim | No inverse possible | Low |
| Dropout | Stochastic removal | 1:1 mapping broken | Actually prevents shortcut |

---

## Mitigation Design

Implemented in `SSLPretrainDataset` via `ssl_pretrain.py` (2026-05-29 revision):

### Distortion chain (applied sequentially)

| Order | Distortion | Parameters | Anti-memorization property |
|-------|-----------|------------|---------------------------|
| 1 | JitterDistortion | std=(5,15), p=0.9 | Per-cell independent noise. No global inverse exists. Model MUST use feature-based identity. |
| 2 | ElasticDistortion | alpha=(30,100), sigma=(8,20) | Non-parametric high-dimensional warp. Decoder cannot learn simple analytic inverse. |
| 3 | FeatureNoise | std=(0.05,0.2), scaled to ~10% of feature std | Prevents exact feature matching. Forces robust identity representation. |
| 4 | DropoutDistortion | p_drop=(0.05,0.2) | tgt-only dropout. Cells disappear → no 1:1 matching guarantee. Matches "modulo dropped cells" from proposal. |
| 5 | PhotometricDistortion | scale=(0.5,2.0), shift=(-0.1,0.1) | Intensity-only features perturbed. Shape features invariant. |

### Omissions

- **AffineDistortion excluded**: 6-parameter global transform trivially invertible. Model would learn rotation/scale inverse in ~3 epochs.
- **WRAugmentationPipeline removed**: Only 5 simple ops (flip, affine, offset, movement, brightness). Insufficient to prevent memorization.

### Parameter selection rationale

Current parameters (jitter std 5-15, elastic alpha 30-100) are substantially stronger than default `DistortionPipeline.from_config` (jitter std 2-8, elastic alpha 10-50). The defaults were calibrated for contrastive NT-Xent on encoder-only embeddings, where the loss is InfoNCE (relative ranking task). Identity BCE is an absolute matching task — the model sees exact ground truth, so distortions must be more destructive to prevent trivial convergence.

---

## Diagnostic Protocol

Before running full fine-tuning experiment:

### Checklist

| Check | Method | Pass criterion |
|-------|--------|----------------|
| SSL loss not → 0 | Log BCE per epoch for 20 epochs | Loss plateaus > 1e-3 at epoch 20 |
| Features matter | Inference with `features=zeros` | BCE loss decreases significantly from zero-feature baseline |
| SSL helps downstream | Fine-tune 10% labels 20 epochs vs from-scratch 40 epochs | SSL init converges faster at matched compute |

### Feature ablation

```python
# Zero-out features to check if model uses coords only
batch_zero_feat = batch.copy()
batch_zero_feat["features"] = torch.zeros_like(batch["features"])
out_zero = model(**batch_zero_feat)
# Compare loss with vs without features → gap should be > 0
```

If gap ≈ 0 → model uses coordinates only → pretraining worthless.

### Warning signs of memorization

- BCE loss < 1e-3 within 5 epochs → model found coordinate shortcut
- Consistency (accuracy on SSL pairs) ≈ 100% at epoch 3 → model learned T⁻¹
- Fine-tuned model converges SLOWER than from-scratch → negative transfer

---

## Implementation Changes

### `ssl_pretrain.py` (2026-05-29)

**Before:** `WRAugmentationPipeline` with 5 weak ops applied to WRFeatures object.
**After:** Distortion chain from `ssl_distortions.py` with 5 strong distortion classes applied to numpy arrays.

Data format (Trackastra-compatible) unchanged. Same BCE loss on full TrackingTransformer via `_common_step`.

Key diff:
```python
# Before: weak augmentations on WRFeatures
self.augment = WRAugmentationPipeline([WRRandomFlip(...), WRRandomAffine(...), ...])
feats_t = self.augment(feats)

# After: strong distortion chain on numpy arrays
self.distortions = [JitterDistortion(std=(5,15), ...), ElasticDistortion(alpha=(30,100), ...), ...]
for dist in self.distortions:
    tgt_coords, tgt_feats, tgt_labels = dist(tgt_coords, tgt_feats, tgt_labels)
```

---

## Status

- [x] Analyze memorization risk — this document
- [x] Implement strong distortion chain in SSLPretrainDataset
- [ ] Run SSL pretraining with strong distortions, log BCE convergence
- [ ] Run feature ablation test
- [ ] Run downstream fine-tuning comparison
