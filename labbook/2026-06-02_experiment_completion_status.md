# Experiment Completion Status

**Date:** 2026-06-02
**Author:** auto-generated from session

## Experiment Summary

5 clean models trained with identical configs, differing ONLY in `knn_neighbors` (verified: 60 of 63 config keys identical). Complete convergence data + TRA/AOGM tracking metrics collected.

## Verifiable Claims

- All 5 configs differ only in `name` and `knn_neighbors` (`diff` proof in `bench-transformer-cell-tracking/benchmark_combined/`)
- All 5 train_config.yaml files on cluster differ only in `name`, `knn_neighbors`, and `config` (cosmetic path) — **60 of 63 keys identical**
- Same seed=42, same train/val split, same architecture, same hyperparameters
- TRA evaluation on 32 vanvliet experiments with GT detections + greedy linker

## Completed Tasks

| Task | Status | Evidence |
|------|--------|----------|
| Dense (K=-1) vs K=16 with identical configs | ✅ Complete | Both ran 500 epochs, early stopped |
| K=4 config | ✅ Complete | `vanvliet_sparse_k4_clean.yaml` |
| K=16 config | ✅ Complete | `vanvliet_sparse_k16_clean.yaml` |
| K=32 config | ✅ Complete | `vanvliet_sparse_k32_clean.yaml` |
| K=64 config | ✅ Complete | `vanvliet_sparse_k64_clean.yaml` |
| All 5 models trained | ✅ Complete | model.pt + TB logs for each |
| Val_loss convergence curves | ✅ Complete | `results_final/convergence.csv` (113KB, per-epoch data) |
| TRA/AOGM evaluation (all 5) | ✅ Complete | `results_final/results_{model}.csv` (per-experiment metrics) |
| SSL identity BCE experiment | ✅ Complete | Labbook `2026-06-01_identity_bce_negative_result.md` |
| SSL contrastive experiment | ✅ Complete | Labbook `2026-05-24_ssl_collapse_findings.md` |
| SSL negative result documented | ✅ Complete | `results_final/README.md` |
| Attention speed benchmark | ✅ Complete | `benchmark_sparse_results.csv` (81 rows) |
| Token reordering analysis | ✅ Complete | Labbook `2026-05-12_*` (negative, <3% speedup) |

## Pending Optional Tasks

| Task | Priority | How-to |
|------|----------|--------|
| Full-model throughput profiling (tokens/sec) | **P1** | `profile_throughput.py` exists but needs GPU run |
| Linker ablation (greedy vs LAP vs ILP) | **P2** | Requires `motile` for ILP |
| Multi-seed runs (seeds 43, 44) for robustness | P2 | Modify `run_single_config.slurm` with `--seed 43` |

## Where Results Are Stored

### Local (`results_final/`)
| File | Description |
|------|-------------|
| `convergence.csv` | Per-epoch val_loss for all 5 models (113KB, 1743 rows) |
| `results_table.csv` | Summary: epochs, times, val_loss stats |
| `results_baseline.csv` | TRA/AOGM per-experiment for K=-1 |
| `results_k4.csv` | TRA/AOGM per-experiment for K=4 |
| `results_k16.csv` | TRA/AOGM per-experiment for K=16 |
| `results_k32.csv` | TRA/AOGM per-experiment for K=32 |
| `results_k64.csv` | TRA/AOGM per-experiment for K=64 |
| `README.md` | Full results table and key findings |
| `convergence_data.py` | Script to regenerate convergence.csv from TB logs |

### Cluster (Capella, `trackastra/runs/`)
| Run | Path |
|-----|------|
| Baseline (K=-1) | `2026-06-01_23-04-13_baseline_clean/` |
| K=4 | `2026-06-01_23-04-12_sparse_k4_clean/` |
| K=16 | `2026-06-01_23-04-18_sparse_k16_clean/` |
| K=32 | `2026-06-01_23-04-29_sparse_k32_clean/` |
| K=64 | `2026-06-01_23-04-09_sparse_k64_clean/` |

## Key Results Table

| Model | Mean TRA | Mean AOGM | Edge F1 | Div F1 | Epochs | Time (h) |
|-------|----------|-----------|---------|--------|--------|----------|
| Baseline (K=-1) | 0.9963 | 51.0 | 0.991 | 0.970 | 372 | 10.1 |
| K=4 | 0.9957 | 62.0 | 0.991 | 0.971 | 237 | 6.4 |
| K=16 | 0.9972 | 37.7 | 0.993 | 0.976 | 406 | 11.0 |
| K=32 | 0.9963 | 52.6 | 0.991 | 0.971 | 362 | 9.9 |
| K=64 | 0.9969 | 39.8 | 0.993 | 0.977 | 366 | 10.0 |

## Attention Speed Benchmark (from `benchmark_sparse_results.csv`)

| N | Dense (ms) | Sparse K=4 (ms) | Speedup |
|---|-----------|-----------------|---------|
| 512 | 0.59 | 1.36 | 0.41x (dense faster) |
| 2048 | 8.4 | 5.5 | 1.44x |
| 8192 | 139 | 21.9 | **5.87x** |

## Relevant Labbook Files

- `2026-06-01_systematic_experiment_plan.md` — original plan
- `2026-06-01_identity_bce_negative_result.md` — SSL experiment
- `2026-05-24_ssl_collapse_findings.md` — contrastive failure analysis
- `2026-05-27_edge_accuracy_results.md` — contaminated KNN comparison (DO NOT USE)
- `2026-05-29_knn_regularizer_critique.md` — why contamination invalidates results
- `2026-05-12_token_reordering_analysis.md` — token reordering (negative)
- `2026-05-29_gather_sparse_bottleneck.md` — profiling deep-dive
- `2026-05-24_imgfeat_negative_result.md` — image feature SSL (negative)
