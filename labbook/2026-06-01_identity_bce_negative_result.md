# Identity BCE SSL Pretraining: Negative Result

**Date:** 2026-06-01

## Experiment

SSL pretraining via identity BCE on geometrically distorted single frames (Trackastra full model, 20 epochs), followed by finetuning on 10% labeled data (100 epochs). Compared against baseline trained from scratch on 10% labels.

### Setup
- Distortion pipeline: jitter, elastic, feature noise, dropout, photometric (excl. affine)
- Label fraction: 10% (slice_pct=0.1, window=4)
- Training: 66 windows (after 10% slice), ~3125 batches/epoch
- Validation: 100 steps over 100 epochs

### Results

| Metric | Baseline (random init) | SSL-Finetune |
|--------|----------------------|--------------|
| val_loss_epoch (final) | 0.416 | 0.404 |
| val_loss_step (final) | 0.115 | 0.111 |
| Early convergence | Same trajectory | Same trajectory |

Nearly identical convergence curves. SSL provides **zero benefit** over random initialization.

## Root Cause

Identity BCE on single-frame distortions is trivially solvable:
1. Model learns `f(x) ≈ f(T(x))` via spatial shortcuts in the distortion
2. Coordinate encoding dominates (same cell, similar coordinates after warp)
3. No need to learn cell identity or tracking-relevant features
4. 7-dim regionprops are too impoverished for meaningful SSL

## Relation to Prior Contrastive Failure

Previous contrastive (NT-Xent) experiment (2026-05-24) also showed zero improvement:
- Inter-cell cosine sim ~0.89 (representational collapse)
- 7-dim features → insufficient variance to discriminate cells
- Downstream tracking with SSL embeddings: 65.1% vs random: 64.8% (no improvement)

**Both SSL paradigms fail for the same root cause: shallow regionprops bottleneck.**

## References
- Lab: `2026-05-24_ssl_collapse_findings.md` (contrastive failure analysis)
- Lab: `2026-05-24_imgfeat_negative_result.md` (image patch CNN failure)
- Proposal §4.2: proposed identity BCE method (tested, negative)
- Lab: `2026-05-29_ssl_memorization_analysis.md` (anti-memorization distortions)
