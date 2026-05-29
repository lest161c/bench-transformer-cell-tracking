# Findings & Fixes — 2026-05-26 Session

---

## Trackastra KNN Results (ongoing)

| Model | K | Best val_loss | vs Dense (0.002) | Status |
|-------|---|---------------|-------------------|--------|
| Dense baseline | — | 0.00200 | baseline | DONE (234ep, 6h51m) |
| KNN K=4 | 4 | 0.01508 | **7.5x worse** | CANCELLED (~ep52) |
| KNN K=16 | 16 | 0.00228 | 14% worse | RUNNING (~ep36) |
| KNN K=32 | 32 | **0.00140** | **30% better** | RUNNING (~ep26) |

**Conclusion:** K=4 too sparse for real tracking. K=16 approaches dense quality. K=32 beats dense — strong evidence KNN attention viable for cell tracking.

---

## SSL Distortion Ablation: Two Bugs Found & Fixed

### Bug 1: DropoutDistortion label truncation

**Symptom:** SSL on regionprops collapsed (loss ~3.0, no improvement). Only `no_dropout` variant converged (val_loss 0.23 vs 2.88).

**Root cause:** In `distortions.py`, DropoutDistortion truncated labels with `l[:len(c)]` instead of filtering with the keep mask. After dropout removes cells, remaining cells get wrong labels → positive pairs misaligned.

```python
# BEFORE (buggy):
l = l[:len(c)]  # Truncates — wrong labels!

# AFTER (fixed):
labels_t = labels[keep]  # Filter — correct labels
```

Also all distortions now return `(coords, features, labels)` consistently — previously only DropoutDistortion returned triples, others returned pairs.

**Fix:** Commit `942c8dc` on `feature/imgfeat-ssl`.

### Bug 2: Independent per-view dropout

**Symptom:** Even with Bug 1 fixed, `full` variant (dropout enabled) still showed loss ~3.0. Only `no_dropout` converged.

**Root cause:** SSLDataset sorts cells by label for index-based positive pair alignment. Dropout applied INDEPENDENTLY to each view removes different cells → sorted indices misalign → `valid_pair` mask rejects nearly all pairs.

Example: cell 5 dropped from view 1 but not view 2:
- View 1 sorted: [1, 2, 3, 4, 6, 7, 8] (7 cells)
- View 2 sorted: [1, 2, 3, 4, 5, 6, 7, 8] (8 cells)
- valid_pair: [1=1, 2=2, 3=3, 4=4, 6≠5, 7≠6, 8≠7, 0≠8] → only 4/8 valid!

**Fix:** DistortionPipeline now saves DropoutDistortion RNG state before view 1 and restores it before view 2. Both views get identical cell drops → sorted indices stay aligned.

**Fix:** Commit `1bde9dc` on `feature/imgfeat-ssl`.

### Re-run

Job 3578124 (`dist_abl2` v3) queued with both fixes. Will run all 7 distortion variants. Expected: ALL variants should now converge, not just `no_dropout`.

---

## Other Fixes

| Issue | Fix | Commit |
|-------|-----|--------|
| K4/K16/K32 all named `sparse_k16` in slurm → checkpoint collision | Changed `--name` to `sparse_k4`, `sparse_k32` per script | `3c00fb6` |

---

## Running Jobs

| Job ID | Name | Status |
|--------|------|--------|
| 3576411 | tr_sparse_k16 | RUNNING |
| 3576412 | tr_sparse_k32 | RUNNING |
| 3578124 | dist_abl2 v3 | QUEUED |

---

## Next Steps

1. Wait for K16/K32 to finish → run TRA/AOGM evaluation
2. Wait for dist_abl2 v3 → verify SSL convergence with fixed dropout
3. Compile all results → figures and report

---

## Files Reference

- `benchmark_ssl/distortions.py` — all distortion implementations (two bugs fixed)
- `benchmark_ssl/train_ssl.py` — NT-Xent loss + training loop
- `benchmark_ssl/ssl_pipeline.py` — SSLDataset + label-sorted collation
- `benchmark_ssl/pretrain.py` — main SSL pretraining entrypoint
- `benchmark_combined/run_dist_abl_fixed.slurm` — dist_abl v3 job script
- `benchmark_combined/run_sparse_k4.slurm` — K4 trackastra (now uses `--name sparse_k4`)
- `benchmark_combined/run_sparse_k16.slurm` — K16 trackastra
- `benchmark_combined/run_sparse_k32.slurm` — K32 trackastra (now uses `--name sparse_k32`)
