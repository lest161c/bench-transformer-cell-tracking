# Edge Accuracy Evaluation — 2026-05-27

Fair comparison on same val data (rpsM+recA+pheA), same eval script, same metric.

| Model | Edge Accuracy | BCE Loss | Pairs Evaluated |
|-------|--------------|----------|-----------------|
| Dense (baseline) | 0.976940 | 0.099242 | 20,665,110 |
| **KNN K=16** | **0.998107** | **0.003643** | 20,665,110 |
| **KNN K=32** | **0.998821** | **0.002062** | 20,665,110 |

Metric: binary edge prediction accuracy over all cell pairs within temporal window (delta=2 frames). Threshold at 0. Same mask for all models — no trivial-negative bias.

## Key finding

KNN attention significantly improves edge prediction accuracy:
- K=32: **+2.3%** accuracy over dense (0.999 vs 0.977)
- K=16: **+2.2%** accuracy over dense (0.998 vs 0.977)
- BCE loss: **48x lower** for K32 (0.002 vs 0.099), **27x lower** for K16

Sparse attention acts as structural regularizer — prevents false-positive edges between cells that are spatially distant.

## Method

1. Load trained model from `model.pt`
2. Load val data (CTCData with ground truth TRA labels)
3. Run model on each batch → assoc_matrix predictions
4. Mask: valid cells × temporal window (delta=2) — SAME mask for all models
5. Compute binary accuracy (pred > 0) vs ground truth

## Caveats

- K4 cancelled (val_loss 7.5x worse than dense at epoch 52)
- Training still running at evaluation time (K16 ep36, K32 ep26 when model.pt copied)
- Only val data evaluated (not test set)
- Full TRA/AOGM tracking metrics not yet computed (need traccuracy installation)

## K4 excluded

K4 job cancelled: val_loss 0.015 vs dense 0.002 at epoch 52. K=4 too sparse — misses real cell connections. Minimum viable K appears to be 16.

## Bugs fixed

1. DropoutDistortion: `l[:len(c)]` → `l[keep]` (labels misaligned after dropout)
2. DistortionPipeline: dropout RNG shared between views (independent dropout broke sorted-collation)
3. Slurm scripts: `--name sparse_k4`/`sparse_k32` (was all `sparse_k16`, causing checkpoint collision)
