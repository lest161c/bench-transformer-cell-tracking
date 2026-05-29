# SSL Critical Review: Contrastive SSL vs Proposal §4.2

**Date:** 2026-05-28  
**Purpose:** Critical evaluation of current ASCENT-style contrastive SSL implementation against the proposal's stated method. Identify architectural mismatches, correctness issues, and recommend corrective path.

---

## Summary of Critical Issues

### 1. Train objective mismatch: encoder-only contrastive vs full-model association

Current: SSL trains only `model.encode()` (encoder + head_x) via NT-Xent loss. Decoder (6× cross-attention layers, head_y, outer-product association) never sees SSL training.

Proposal: "Train Trackastra's **association head** on these synthetic pairs" — full model, encoder + decoder + association. BCE identity loss, same architecture as downstream.

The decoder and association head account for ~50% of model parameters. They start from random init regardless of SSL pretraining. The part of the model that ACTUALLY produces associations is never pretrained.

### 2. SSL task is trivially solveable: feature poverty + model overcapacity

Input: 7-dim regionprops (inter-cell cosine sim ~0.89 — nearly identical)  
Model: 320-dim, 6-layer encoder (~7M params)  
Result: NT-Xent loss → 0 in 2 epochs

Model memorizes 7-dim → 320-dim mapping trivially. No compression, no generalization, no useful representation learned. ASCENT's success relies on 12288-dim image patches → 256-dim compression; our 7-dim → 320-dim is EXPANSION, the opposite.

### 3. Coordinate shortcut dominates

Trackastra's encoder uses Fourier positional encoding (96-dim) concatenated with feature encoding (56-dim). After distortions (jitter std 2-8px), cells remain in roughly the same spatial arrangement. Model can match cells across views by spatial proximity alone, bypassing feature-based identity learning.

When transferred to real tracking, cells ACTUALLY MOVE between frames. The spatial shortcut learned during SSL is harmful: cells at same position in consecutive frames are different cells.

### 4. Shared-dropout RNG removes challenge

Fix forces both SSL views to drop identical cells → perfect 1:1 positive pairs, zero false negatives. Proposal explicitly says "identity i → i (**modulo dropped/occluded cells**)". The intent was dropped cells simulate segmentation failures. Current implementation removes this challenge entirely.

### 5. Distortions ≠ biological motion

SSL applies random geometric transforms (affine rotation, elastic deformation, jitter) to SINGLE frames. None of these simulate real cell motion between consecutive frames. Model learns invariance to transforms that never occur in real tracking data.

### 6. Zero downstream evidence

No experiment has loaded SSL-pretrained Trackastra encoder and fine-tuned on tracking data with BCE loss. Label-sweep (standalone CellEmbedder, frozen SSL embeddings + Hungarian) shows: SSL (65.1%) ≈ random (64.8%) — zero improvement.

---

## Proposal vs Implementation Gap

| Aspect | Proposal §4.2 | Current Implementation |
|--------|--------------|----------------------|
| Model trained | FULL TrackingTransformer | Encoder only (`encode()`) |
| Loss | BCE identity association | NT-Xent contrastive |
| Window structure | 2+ frames (original + distorted) | Single frame, two views at same time |
| Association head | Trained | Not trained |
| Decoder | Trained | Not trained |
| Transfer | Same model, fine-tune encoder+decoder | Encoder weight init, decoder random |
| Dropout handling | "modulo dropped cells" (one-sided) | Same cells dropped both views |

**The current implementation is a fundamentally different method than what the proposal describes.** The proposal's identity-BCE approach and the current contrastive-NT-Xent approach train different components, use different losses, and have different relationships to downstream tracking.

---

## Recommendation: Return to Proposal's Identity BCE on Full Model

The proposal clearly describes: synthesize a 2-frame tracking window from a single frame via geometric distortion, train the FULL Trackastra model with identity associations (BCE loss). This approach:

1. **Trains the full model** — encoder + decoder + association head all see SSL signal
2. **Uses same loss as downstream** — BCE on association matrix, direct transfer
3. **Preserves tracking structure** — multi-frame window, temporal modeling, cross-attention
4. **Teaches correct inductive biases** — cell consistency across frames, smooth motion, spatial locality
5. **Leverages existing infrastructure** — Trackastra's training loop already handles augmented tracking windows (RandomTemporalAffine etc.)

### Implementation sketch

```
Single frame → extract cells (coords, features, labels)
     │
     ├── Frame 0 (t=0): original cells (optionally mildly augmented)
     ├── Frame 1 (t=1): distorted cells via DistortionPipeline (one view)
     │
     ▼
2-frame tracking window with identity associations
     │
     ▼
TrackingTransformer.forward() → BCE loss
Same training loop as supervised, different data source
```

The contrastive implementation should be **kept for ablation comparison**, but the primary SSL method should be identity BCE on the full model, as proposed.

---

## Downside: Disortion-Real Motion Gap

Even identity BCE on full model faces the fundamental issue that random geometric distortions do not simulate real cell motion. However:

1. **Existing Trackastra augmentations already do this** — `RandomTemporalAffine` warps cells across time in supervised training. This is the same idea, just with real tracking labels instead of identity.
2. **Inductive biases still transfer** — cell identity, feature consistency, spatial locality, temporal smoothness are all useful starting points for downstream.
3. **This is testable** — unlike contrastive encoder-only pretraining, this approach directly plugs into Trackastra's training pipeline. Run 20 epochs of SSL (identity BCE), then fine-tune on labeled data. Compare convergence against random init.

The cross-frame motion gap is real but secondary to the architectural misalignment of the current contrastive approach. Fix the model/loss alignment first, then evaluate whether the distortion-quality gap matters.
