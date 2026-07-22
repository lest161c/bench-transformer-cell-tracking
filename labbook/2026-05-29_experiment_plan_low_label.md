# SSL Low-Label Convergence Experiment

**Date:** 2026-05-29

## Objective
Prove that identity BCE SSL pretraining on unlabeled data speeds up convergence downstream, especially in the low-label regime (10% of annotations).

## Implementation
1. Added `--train_fraction` to `train.py` to support low-label subsetting.
2. Slices `CTCData` sequentially (`slice_pct`) across both train and val splits to ensure consistent evaluation.
3. Designed 3-stage experiment:
   - **Run A**: SSL Pretraining (100% unlabeled frames) → 20 epochs.
   - **Run B**: Baseline Random Init (10% labeled tracks) → 100 epochs.
   - **Run C**: Fine-tune SSL Init (10% labeled tracks) → 100 epochs.

## Files Created
- `benchmark_combined/run_low_label_exp.slurm`: SLURM batch job executing A, B, and C sequentially.
- `benchmark_combined/plot_low_label.py`: Plots `val_loss` vs `epoch` comparing Run B and Run C.

## Expected Result
Run C (`SSL Init`) should reach the same `val_loss` as Run B (`Random Init`) in significantly fewer epochs, proving that the identity BCE pretraining successfully initialized the TrackingTransformer weights with transferable geometric priors.
