# SSL Failure Analysis: Representational Collapse

**Date:** 2026-05-24

## Observed behavior

Contrastive SSL pretraining (InfoNCE, 50 epochs, 6 vanvliet conditions) on H100:

| Metric | Value |
|--------|-------|
| NT-Xent train loss | 3.46 → 2.97 (plateau after epoch 10) |
| NT-Xent val loss | 3.29 → 2.82 |
| Cosine consistency (epoch 1) | 0.974 |
| Cosine consistency (epoch 50) | 0.919 |
| Downstream tracking (knn_SSL frozen) | 65.1% |
| Downstream tracking (knn random) | 64.8% avg |
| Downstream tracking (dense random) | 64.3% avg |

SSL provides **zero improvement** over random embeddings for downstream tracking.

## Root cause: representational collapse

Diagnostic on a real batch (4 frames, 228 real cells):

| Metric | Random init | SSL (epoch 44) |
|--------|-------------|-----------------|
| Inter-cell cosine sim (same frame) | 0.85-0.88 | 0.82-0.90 |
| Pos-pair cosine sim (same cell, two views) | 0.86-0.92 | 0.91-0.94 |
| Global inter-cell similarity | **0.89** | — |

All cells map to nearly identical embeddings (cosine similarity ~0.89). The model cannot distinguish different cells in the same frame. The gap between positive pairs (0.91) and negatives (0.85) is only 0.06 — far too small for reliable tracking.

## Why collapse happens

1. **Feature poverty**: 7-dim regionprops (equivalent_diameter_area, intensity_mean, inertia_tensor, border_dist) have small inter-cell variance. Two bacteria in the same colony have similar area, intensity, shape. The encoder receives nearly identical input for many cells.

2. **Coordinate dominance**: With element-wise addition of feature and coordinate projections, the coordinate signal dominates. Two views of the same cell have similar coordinates (jitter std=2-8 is small), so the encoder trivially maps them to similar embeddings without learning feature-based identity.

3. **Augmentation weakness**: jitter std=[2,8] and feature_noise std=[0.02,0.1] are too mild to break the coordinate shortcut. The model can match views using coordinates alone, bypassing the need to learn feature-based identity.

## Contrast with ASCENT paper

ASCENT (Han & Lu 2025) uses 64×64×3 image patches through a ChannelViT → rich 256-dim visual features per cell. Two cells that look similar in regionprops are highly distinct in image patches (different local neighborhood, different spatial arrangement of nearby cells). The contrastive loss has orders of magnitude more information to organize.

Without image features, our pipeline is learning to contrastively organize a 7-dim space — fundamentally limited.

## Potential fixes to investigate

1. Stronger augmentations (bigger jitter, heavier noise)
2. Remove coordinate encoding to force feature-based learning
3. Smaller model (less capacity = less room to collapse)
4. Feature normalization (standardize per frame)
5. Image features (32×32 crops + small CNN) — matching ASCENT architecture

## Fixes tested locally (2026-05-24)

| Fix | Result |
|-----|--------|
| Stronger augmentations (jitter std 15-40, noise 0.1-0.5) | Collapse WORSE: inter-sim 0.90 → 0.96 |
| Remove coordinate encoding | Crash: dimension mismatch (coord_enc expects d_model) |
| Smaller model (d=32, L=2) | Same collapse, inter-sim ~0.90 |
| 32×32 image patches + shallow CNN | Loss flat at 3.9, inter-sim=0.99. CNN too weak, patches too small. |

## Synthetic verification: architecture is correct

A synthetic test with 16-dim unique cell signatures + 4-dim shared view-noise proved the pipeline works:
- Loss: 1.41 → **0.03** in 20 epochs
- Inter-cell similarity: 0.68 → **0.17**
- Generalization to unseen signatures: pos_sim=0.80 vs neg_sim=0.13 (**gap=0.67**)

The NT-Xent loss and CellEmbedder are correct. Collapse on real data is purely a feature-quality problem.

## Measurement scripts (cluster)

SLURM scripts in `bench-transformer-cell-tracking/benchmark_combined/`:
- `run_phase1_scaling.slurm` — H100 attention scaling (N=128..8192, L=1/6/12)
- `run_phase1_ssl.slurm` — contrastive SSL pretraining 50 epochs
- `run_phase1_sweep.slurm` — label fraction sweep (frozen embeddings + Hungarian)
- `run_diag.slurm` — GPU diagnostics
- `run_diag2.slurm` — embedding collapse diagnostic

Results on cluster at `benchmark_ssl/runs/phase1/`:
- `scaling_h100.csv` — per-step attention timing
- `label_sweep.csv` — tracking accuracy vs label fraction
- `../ssl_phase1/training_log.csv` — SSL NT-Xent loss curve
