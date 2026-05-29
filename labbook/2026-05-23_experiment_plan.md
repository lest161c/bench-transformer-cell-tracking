# Concrete Experiment Plan: Local + HPC

**Date:** 2026-05-23

---

## What "X% labels" means

Trackastra trains on experiment directories (e.g. `data/vanvliet/rpsM/151101_E2-1/TRA/man_track000.tif` etc.). Each directory is one microscopy session with ground-truth tracking annotations stored in the TRA folder. The base-config lists 22 training experiments in `input_train`.

**X% labels** = X% of those 22 training experiments used for fine-tuning.

| Fraction | Experiments | Simulated scenario |
|----------|-------------|--------------------|
| 1% | 1 | Rapid pilot: annotate one video, track the rest |
| 5% | 1-2 | Minimal annotation budget |
| 10% | 2-3 | Realistic: annotate 2-3 videos |
| 25% | 5-6 | Moderate annotation effort |
| 50% | 11 | Half-labeled dataset |
| 100% | 22 | Full supervision (baseline) |

SSL pretraining always uses **all 22 experiments** — only segmentations are needed, no tracking labels. Fine-tuning uses the reduced experiment subset. Validation and test sets are fixed (from `input_val` and `input_test`).

Implementation: modify `input_train` in config to subset the experiment list. Restart fine-tuning from SSL-pretrained checkpoint for each fraction.

---

## Image features: current limit and upside

Current SSL pipeline: 7-dim regionprops (equivalent_diameter_area, intensity_mean, inertia_tensor, border_dist). ASCENT uses a ChannelViT on 64×64×3 image patches — orders of magnitude richer input. Without image features:

- Contrastive learning is organizing a 7-dim space — limited discriminative capacity
- Two cells with similar area + inertia may be indistinguishable
- SSL benefit is capped by feature entropy

With image features (even a small CNN on 32×32 crops):
- 1024-dim feature vector per cell → much richer contrastive space
- Local neighborhood context (other cells in the patch) becomes learnable — ASCENT's key insight
- SSL benefit could be substantially larger (5-10% → 15-25% at low labels)

**Decision:** Run all experiments with current 7-dim regionprops first. If SSL benefit is near zero, the feature bottleneck is the likely cause. Adding image features is a ~1-week engineering task (extract crops, add CNN encoder, retune). Only pursue if regionprops results are negative.

---

## Step-by-step experiment list

### PHASE 0: Local verification (A500 laptop, 4GB)

Goal: verify all scripts run without NaN/crash, produce valid CSVs before shipping to HPC.

| Step | Script | Config | Time | Checks |
|------|--------|--------|------|--------|
| L0 | `python train_ssl.py config_local.yaml` | d=32, L=2, B=2, 5ep, 1 cond | ~1 min | Loss decreases, no NaN, CSV written |
| L1 | `python downstream_compare.py runs/ssl_local/best_model.pt` | 1 cond, 50 pairs | ~1 min | Accuracy printed, pretrained > rand or equal |
| L2 | `python -c "import pretrain_multi"` | — | <1s | No import errors |

**config_local.yaml:**
```yaml
name: ssl_local
data_root: ../data/vanvliet
conditions: [rpsM]
ndim: 2
features: regionprops2
distortions: [jitter, dropout, feature_noise]
jitter: {std: [2, 8], p_cell_jitter: 0.8}
dropout: {p_drop: [0.05, 0.2]}
feature_noise: {std: [0.02, 0.1]}
encoder: {d_model: 32, nhead: 2, num_layers: 2, dim_feedforward: 64, dropout: 0.0}
ssl: {temperature: 0.05}
training: {batch_size: 2, lr: 0.0001, weight_decay: 0.01, epochs: 5, val_split: 0.2}
seed: 42
```

---

### PHASE 1: SSL pretraining (H100, ~30 min)

Goal: produce pretrained checkpoint + convergence CSVs.

| Step | Script | Config | Time | Output |
|------|--------|--------|------|--------|
| H1.1 | `python pretrain.py config_h100.yaml` | d=128, L=4, B=16, 50ep, all conds, FP16 autocast | ~20 min | `runs/ssl_h100/best_model.pt` + `training_log.csv` (columns: epoch, train_loss, train_consistency, val_loss, val_consistency) + TensorBoard |
| H1.2 | `python pretrain_multi.py` (updated to use CellEmbedder+NT-Xent) | Same as H1.1 but sweeps K=None, K=4, K=16 | ~60 min (3 variants × 20min) | `runs/ssl_dense/best_model.pt`, `runs/ssl_K4/best_model.pt`, `runs/ssl_K16/best_model.pt` with per-variant loss CSVs |
| H1.3 | `python plot_ssl.py` | Points at H1.2 output dirs | ~1 min | `benchmark_ssl.html` — NT-Xent convergence curves, consistency curves, per-variant comparison |

**config_h100.yaml:**
```yaml
name: ssl_h100
data_root: ../data/vanvliet
conditions: [rpsM, recA, pheA, metA, cib, trpL]
ndim: 2
features: regionprops2
distortions: [affine, elastic, jitter, dropout, photometric, feature_noise]
affine: {degrees: 15, scale: [0.85, 1.15], shear: [0.1, 0.1]}
elastic: {alpha: [10, 50], sigma: [5, 15]}
jitter: {std: [2, 8], p_cell_jitter: 0.8}
dropout: {p_drop: [0.05, 0.2]}
photometric: {scale: [0.5, 2.0], shift: [-0.1, 0.1]}
feature_noise: {std: [0.02, 0.15]}
encoder: {d_model: 128, nhead: 4, num_layers: 4, dim_feedforward: 256, dropout: 0.1}
ssl: {temperature: 0.05}
training: {batch_size: 16, lr: 0.0003, weight_decay: 0.01, epochs: 50, val_split: 0.1}
seed: 42
```

---

### PHASE 2: Distortion ablation (H100, ~3 hours)

Goal: rank distortion families by downstream impact.

| Step | Config | Distortions | Time | Output |
|------|--------|-------------|------|--------|
| H2.1 | `ssl_abl_full.yaml` | all 6 | 20ep SSL + eval | `runs/dist_abl/full/` — baseline |
| H2.2 | `ssl_abl_nojitter.yaml` | all except jitter | 20ep + eval | `runs/dist_abl/nojitter/` |
| H2.3 | `ssl_abl_nodropout.yaml` | all except dropout | 20ep + eval | `runs/dist_abl/nodropout/` |
| H2.4 | `ssl_abl_nonoise.yaml` | all except feature_noise | 20ep + eval | `runs/dist_abl/nonoise/` |
| H2.5 | `ssl_abl_noelastic.yaml` | all except elastic | 20ep + eval | `runs/dist_abl/noelastic/` |
| H2.6 | `ssl_abl_noaffine.yaml` | all except affine | 20ep + eval | `runs/dist_abl/noaffine/` |
| H2.7 | `ssl_abl_jitteronly.yaml` | jitter only | 20ep + eval | `runs/dist_abl/jitteronly/` |

Each run: 20 epochs SSL pretraining, then evaluate using `downstream_compare.py` on 10% label fraction.

Output CSV per run: `epoch, val_loss, val_consistency` (SSL phase) + `model, val_accuracy, test_accuracy` (downstream).

Aggregate into `dist_ablation.csv`: `distortion_removed, val_acc, test_acc`.

---

### PHASE 3: Label fraction sweep (H100, biggest experiment)

Goal: TRA vs label fraction for {dense, KNN, KNN+SSL}.

This requires running actual Trackastra training, not just the standalone benchmark. Trackastra training on H100 at d=320 L=12 takes significant time. Plan:

| Step | Method | Fraction | Experiments | Epochs | Time (est.) | Output |
|------|--------|----------|-------------|--------|-------------|--------|
| H3.1 | Dense (baseline) | 100% | 22 | 500 | ~10h | `runs/label_sweep/dense_100pct/` |
| H3.2 | Dense | 50% | 11 | 500 | ~5h | `runs/label_sweep/dense_50pct/` |
| H3.3 | Dense | 25% | 5-6 | 500 | ~3h | `runs/label_sweep/dense_25pct/` |
| H3.4 | Dense | 10% | 2-3 | 500 | ~2h | `runs/label_sweep/dense_10pct/` |
| H3.5 | Dense | 5% | 1-2 | 500 | ~1h | `runs/label_sweep/dense_5pct/` |
| H3.6 | Dense | 1% | 1 | 500 | ~30min | `runs/label_sweep/dense_1pct/` |
| H3.7-12 | KNN (same fractions) | 1,5,10,25,50,100% | same | 250-500 (20-40% fewer) | ~0.5-7h each | `runs/label_sweep/knn_{X}pct/` |
| H3.13-18 | KNN+SSL (same) | 1,5,10,25,50,100% | same, pretrained | 250-500 | ~0.5-7h each | `runs/label_sweep/knnssl_{X}pct/` |

Total HPC hours for Phase 3: **~30-50 GPU-hours**. Worth running H3.1 and H3.7/H3.13 first at 100% to check if KNN actually converges faster. If no speedup, reduce KNN epochs to match dense total time (fair comparison). If KNN = dense at 100%, run only {1%, 5%, 10%} fractions for KNN+SSL since benefit only expected at low labels.

**Alternative (faster):** Use the same evaluation as `downstream_compare.py` — frozen embeddings + Hungarian matching — at each label fraction. This skips full Trackastra fine-tuning and evaluates pure embedding quality. ~10min per fraction. Less convincing (doesn't test the full Trackastra fine-tune), but fast enough to run all 18 configs in 2-3 hours.

**Recommendation:** Start with fast frozen-embedding evaluation for all 18 configs. Then run full Trackastra training only for the top 3-4 most interesting configs (e.g. dense 100%, KNN 100%, KNN+SSL 10%).

Output CSV: `label_sweep.csv` with columns: `method, label_fraction, val_TRA, test_TRA, val_cHOTA, test_cHOTA, epoch_time_s, total_time_min`.

---

### PHASE 4: Cross-dataset generalization (H100, ~5 hours)

| Step | Train | Fine-tune | Time | Output |
|------|-------|-----------|------|--------|
| H4.1 | SSL on vanvliet (22 exps) | Fine-tune on bacteria (all labeled exps, Trackastra) | ~3h | `runs/xdataset/v2b/` |
| H4.2 | None | Train bacteria from scratch (Trackastra baseline) | ~3h | `runs/xdataset/b_scratch/` |
| H4.3 | SSL on vanvliet | Fine-tune on bacteria at 10% labels | ~1h | `runs/xdataset/v2b_10pct/` |

Output CSV: `xdataset.csv` with columns: `method, train_data, test_data, label_fraction, TRA, cHOTA`.

---

### Raw data collection summary

All experiments produce CSV files. Final aggregation script reads them all.

| File | Source | Columns |
|------|--------|---------|
| `ssl_convergence.csv` | H1.1, H1.2 | epoch, variant, train_loss, val_loss, consistency |
| `dist_ablation.csv` | H2.1-2.7 | distortion_removed, val_acc, test_acc |
| `label_sweep.csv` | H3.1-3.18 | method, label_fraction, val_TRA, test_TRA, epoch_time |
| `xdataset.csv` | H4.1-4.3 | method, train_data, test_data, label_fraction, TRA |
| `throughput.csv` | benchmark_attn/ (already exists) + new H100 sweep | N, K, L, time_ms, memory_mb, device |

---

### Quick-start minimal path (2 HPC days)

If time is short, the minimal publishable set:

1. **H1.1:** SSL pretrain (20 ep, 15 min)
2. **H3.7, H3.13:** KNN vs KNN+SSL at 1%, 5%, 10% labels (fast frozen-embedding eval, 30 min each)
3. **H3.1:** Dense baseline at 100% labels (1 run, ~10h — can run overnight)
4. **H3.8:** KNN at 100% labels (1 run, confirm parity)
5. **H2.1, H2.2, H2.7:** Distortion ablation: full vs no-jitter vs jitter-only (3 runs)

Total: ~12-15 HPC GPU-hours. Produces the key figure (label fraction sweep) plus distortion ablation.
