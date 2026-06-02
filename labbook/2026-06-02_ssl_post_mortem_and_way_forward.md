# SSL Post-Mortem: BCE + Image Features + Coordinate-Free Pretraining

**Date:** 2026-06-02

## What We Tried

| Experiment | Model | Loss | Distortions | Features | Result |
|---|---|---|---|---|---|
| NT-Xent contrastive | CellEmbedder (encoder only) | InfoNCE | Full pipeline (6 types) | 7-dim regionprops | Collapse: cos-sim 0.89, TRA 65% (random) |
| BCE identity | Full TrackingTransformer | BCE | WRAugmentation (5 basic ops) | 7-dim regionprops | No improvement: val_loss 0.404 vs 0.416 |

## Why Both Failed: 7-dim Regionprops Bottleneck

Trackastra uses 7 hand-crafted features per cell:
`[area, perimeter, eccentricity, solidity, extent, mean_intensity, std_intensity]`

Two bacteria in the same colony have nearly identical values for all 7. The inter-cell cosine similarity is ~0.89 before any transformation. There is simply **no variance to organize** — any SSL objective on this space is trying to discriminate between near-identical points.

ASCENT (Han & Lu 2025) succeeds because it uses 64×64×3 image patches → 256-dim CNN embeddings. Two neurons that look similar in regionprops are highly distinct in image patches (different local neighborhood, different spatial arrangement). Compression from 12288 → 256 forces the model to learn meaningful invariances. Our setup is the opposite: expansion from 7 → 320, no compression pressure.

## What Remains Untested

**BCE + rich distortions (proposal §4.2 full spec) + image features + coordinate-free pretraining.**

The proposal calls for:
1. Full-model BCE (not contrastive) → decoder trains during SSL, no transfer gap
2. Rich distortion pipeline (affine, elastic, jitter, dropout, photometric) → harder task
3. Image features (CNN crops per cell) → real visual identity
4. **Remove coordinates during SSL** → force feature-based learning

Steps 1-3 alone still fail because of coordinate shortcut. Step 4 is the key enabler.

## How to Remove Coordinates (Technical)

Trackastra's encoder in `model.py:forward()` concatenates positional encoding with feature encoding:

```python
pos = self.pos_embed(coords)            # 96-dim Fourier encoding of (t, y, x)
features = self.feat_embed(features)     # 56-dim from 7-dim regionprops
features = torch.cat((pos, features))    # 152-dim total
```

The positional encoding (96-dim) dominates the feature encoding (56-dim) by nearly 2:1. During SSL, the model learns that `pos(cell_i) ≈ pos(T(cell_i))` and solves BCE without ever looking at features.

**To remove coordinates during SSL only:**

### Option A: Zero out coordinate encoding (simplest)
```python
# During SSL forward, override coords to dummy values
dummy_coords = torch.zeros_like(coords)
x = self.forward(features=features, coords=dummy_coords, ...)
```
Encoder sees only feature information. No spatial shortcut possible. But: the Fourier positional encoding dimension (96) is hard-coded — with zero coords, it produces a constant encoding, effectively reducing model capacity.

### Option B: Drop pos_embed output (cleaner)
```python
# In encode(), skip pos entirely
features = self.feat_embed(features)
features = self.proj(features)  # shape (B, N, d_model) from 56-dim input
```
Requires modifying the forward path to accept a flag `ssl_mode=True`. The `feat_embed` projects 7-dim → 56-dim, then `proj` projects 56-dim → 320-dim. This is a learned expansion with no positional bias.

### Option C: Inject random coordinates (regularization)
```python
# Shuffle coordinates across cells to break position→identity mapping
rand_coords = coords[:, torch.randperm(N), :]
x = self.forward(features=features, coords=rand_coords, ...)
```
Each cell gets a random position. The pos_embed still fires but carries no information about which cell is which. Forces feature-only learning. Less destructive to pretrained pos_embed weights than Option A/B.

### Application:
Option C is safest for pretraining. During finetune, restore real coordinates — the pos_embed weights have been perturbed but the feature pathway has learned real cell identity. Finetune will quickly re-align positional encoding to tracking data.

## Open Questions

| Question | Assessment |
|----------|------------|
| Would BCE + rich distortions + image features + coord-free work? | Plausible. Solves both the feature poverty and the shortcut. But unproven. |
| Is it worth the implementation effort? | ~2-3 weeks: CNN feature extractor, SSL pipeline integration, hyperparameter search. Risky given negative track record. |
| Could we just use image features + contrastive instead? | ASCENT-style on encoder only. Cleaner (proven literature), but leaves decoder untrained. |
| What would a minimally viable test be? | Option C + existing regionprops (no CNN yet). If BCE + coord-free + rich distortions works with 7-dim features, the feature bottleneck hypothesis is wrong and CNN is unnecessary. If it still fails, CNN is the next step. |

## Recommendation

A minimal diagnostics experiment: run BCE pretraining with **Option C (shuffled coordinates)** + existing 7-dim regionprops + rich distortion pipeline. If this improves over baseline (which had real coordinates + basic distortions), the bottleneck was the coordinate shortcut, not feature poverty. If it still fails, the problem is genuinely the 7-dim features and only CNN image features can fix it.

This takes ~1 day to set up (modify `ssl_pretrain.py`, add coordinate shuffling, use `DistortionPipeline` instead of `WRAugmentationPipeline`). Run on cluster.
