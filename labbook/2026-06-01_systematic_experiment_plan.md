# Systematic Experiment Plan & Data Audit

**Date:** 2026-06-01
**Reference:** Trackastra paper (arXiv:2405.15700v2) — evaluation methodology

## 1. Core Research Question

Does replacing dense masked SDPA with gather-based sparse attention (KNN) in Trackastra improve:
- **(a) Throughput/memory** at inference time? (benchmark only, §4.1)
- **(b) Tracking accuracy** via structural regularization? (§4.1 convergence benefit)
- **(c) Can SSL pretraining further improve convergence?** (§4.2 — tested, negative)

## 2. Evaluation Methodology (per Trackastra paper)

Trackastra reports:
- **TRA** (normalized AOGM, primary metric)
- **AOGM** (absolute error count)
- **FP/FN edges** and **FP/FN divisions** (per-video means)
- **3 runs per model** for statistical robustness
- **Test on held-out videos** (6 test videos for Bacteria)

Our setup (vanvliet dataset) uses the same bacteria data from [55] as Trackastra's Bacteria experiments. We must follow the same reporting standards.

## 3. Required Experiments

### 3.1 Core: Dense vs Sparse K=16 (fair comparison)  ← **SUBMITTED**

| Config | File | knn_neighbors | Status |
|--------|------|--------------|--------|
| Baseline (dense) | `vanvliet_baseline_clean.yaml` | -1 | Job 3610842 (pending) |
| Sparse K=16 | `vanvliet_sparse_k16_clean.yaml` | 16 | Job 3610842 (pending) |

**Identical in ALL other parameters**: same seed (42), same train/val split, same window=4, same attn_dist_mode=v1, same dropout=0.05, same max_tokens=2048, same train_samples=32000, same crop_size=[320,320].

Configs differ ONLY in `name` and `knn_neighbors`. Verified by `diff`.

### 3.2 K-value sweep (needed for trend)

| Config | knn_neighbors | Rationale |
|--------|-------------|-----------|
| K=4 | 4 | Minimal neighborhood, max speedup |
| K=16 | 16 | Moderate, matches avg neighbor count |
| K=32 | 32 | Generous neighborhood |
| Dense | -1 | Full attention (baseline) |

Trackastra's distance cutoff typically yields ~10-20 neighbors per cell. K=16 is the natural match to the cutoff graph. K=4 tests aggressive sparsity, K=32 tests whether more neighbors approach dense performance.

### 3.3 Speed benchmark on full model (not just attention)

The existing benchmarks (§4.1) measure pure attention forward pass. We need end-to-end timing:
- Full Trackastra inference (encoder+decoder+head)
- Compare dense vs K=16 on actual model
- Report tokens/sec and memory usage

### 3.4 Ablations (following Trackastra paper §3.6)

| Ablation | Variants | Measures |
|----------|----------|----------|
| Linker | greedy, LAP, ILP | TRA/AOGM for each linker |
| Window size s | 2, 3, 4, 6 | AOGM vs s (cf. Fig 4c) |
| KNN structural regularization | Dense vs K=16 (different seeds) | Edge accuracy (cf. §3.2) |

### 3.5 SSL Pretraining (negative result documentation)

| Experiment | Method | Result |
|------------|--------|--------|
| Contrastive NT-Xent | CellEmbedder, 50 epochs, InfoNCE | 65.1% vs 64.8% (zero improvement) |
| Identity BCE | Full Trackastra, 20 epochs, 10% labels | val_loss 0.404 vs 0.416 (identical) |

Document as valid negative result with root-cause analysis.

## 4. Data Audit — What Exists vs What's Missing

### 4.1 Dense vs Sparse KNN (fair comparison)

| Item | Status | Location |
|------|--------|----------|
| Benchmark: attention-only speed (N vs time) | ✅ Done | `benchmark_sparse_results.csv` (81 rows) |
| Benchmark: mask vs gather | ✅ Done | `benchmark_mask_vs_gather.csv` (12 rows) |
| Benchmark: combined ablation | ✅ Done | `benchmark_combined/results/ablation_full.csv` (205 rows) |
| Token reordering | ✅ Done (negative) | `labbook/2026-05-12_*` |
| **Clean dense vs K=16 (identical configs)** | ❌ **Not done** | Job 3610842 submitted (pending) |
| Clean K=4, K=32, K=64 | ❌ **Not done** | Configs needed |
| Full-model speed benchmark | ❌ **Not done** | Need to profile Trackastra inference |
| TRA/AOGM on test set | ❌ **Not done** | Need traccuracy + eval pipeline |
| 3 runs per config | ❌ **Not done** | Need multi-seed runs |

### 4.2 Contaminated data (DO NOT USE for report)

| Item | Issue | Should NOT cite |
|------|-------|-----------------|
| `checkpoints/k16/` | 12 config diffs vs baseline (data leakage, window=10, attn_dist_mode=v0, ...) | Edge accuracy 0.998 vs 0.977 is invalid |
| `checkpoints/k32/` | Same contamination | Same issue |
| `eval_data/runs/eval_*.csv` | Evaluates contaminated models | Cannot trust |

### 4.3 SSL Pretraining

| Item | Status | Location |
|------|--------|----------|
| Identity BCE (today, June 1) | ✅ Done (negative) | TB logs, `labbook/2026-06-01_identity_bce_*` |
| Contrastive NT-Xent (May 24) | ✅ Done (negative) | `labbook/2026-05-24_ssl_collapse_*` |
| Distortion ablation (buggy) | ❌ Buggy code | `labbook/2026-05-26_findings_*` |
| Distortion ablation (fixed) | ❌ **Never run** | Job 3578124 was queued, no results |
| Image features (CNN patches) | ✅ Done (negative) | `labbook/2026-05-24_imgfeat_*` |

### 4.4 Feature Ablations

| Item | Status | Location |
|------|--------|----------|
| Regionprops 7-dim | ✅ Done | `labbook/2026-05-24_imgfeat_*` |
| Scratch CNN 1.8M | ✅ Done (undertrained) | Same |
| ResNet18 frozen | ✅ Done (no transfer) | Same |
| DINOv2 ViT-S frozen | ✅ Done (no transfer) | Same |

## 5. To-Do Steps (Prioritized)

### [P0] Must-do for minimal publishable result

- [x] Create clean configs (baseline, K=16) — only knn_neighbors differs
- [x] Submit clean run (job 3610842, pending, ~14h sequential)
- [x] Document SSL negative result (labbook entry)
- [ ] **Wait for job 3610842 to complete** (~14h from submission)
- [ ] **Evaluate TRA/AOGM** on test set for both models
- [ ] **Generate comparison plots**: val_loss curves, edge accuracy, TRA/AOGM table

### [P1] Strongly recommended

- [ ] Create K=4, K=32, K=64 configs (identical, only knn_neighbors changes)
- [ ] Submit multi-seed runs (seed=42, 43, 44 for each K)
- [ ] Profile full-model inference speed (dense vs K=16 vs K=32)
- [ ] Run SSL evaluation on TRA metric for completeness

### [P2] Nice-to-have (if time permits)

- [ ] Ablation: linker type (greedy vs LAP vs ILP)
- [ ] Ablation: window size (s=2, 3, 4, 6) for K=16
- [ ] General model trained on both vanvliet + deepcell
- [ ] Out-of-domain evaluation

## 6. Config Generation Checklist

For each new experiment config, verify:
```
diff config_baseline.yaml config_experiment.yaml
```
must show ONLY the intended differences (name, knn_neighbors, seed, etc.)

The `vanvliet_baseline_clean.yaml` is the SOURCE OF TRUTH. Every experiment config must be derived from it with minimal changes.

## 7. Reproducibility Requirements

For each experiment in the final report:
- [ ] Config file committed and pushed
- [ ] SLURM script committed and pushed  
- [ ] Seed recorded (fixed seed=42 for primary runs)
- [ ] Input train/val split recorded (specific sequence directories)
- [ ] Model checkpoint saved
- [ ] TB event file saved
- [ ] Evaluation script and results saved
- [ ] All random seeds fixed (torch, numpy, python)
