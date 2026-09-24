# REPRODUCTION.md — benchmark_ssl

Reproduction guide for every experiment in `benchmark_ssl/`: exact entrypoints,
required environments, output artifacts, and the measured-results index.

Authoritative sources: the `src/` modules, `slurm/` scripts,
the CSVs under `results/`, and the local venv `benchmark_ssl/.venv`
(Python 3.14.4, torch 2.12.0+cu130).

---

## 1. Hardware & environment split

| Scope | Hardware | GPU VRAM | Venv | Used for |
|---|---|---|---|---|
| Local SSL experiments | NVIDIA RTX A500 Laptop | 4 GiB | `benchmark_ssl/.venv` | All `runs/*` experiments, edge probing |
| Full DINOv2 SSL pretraining | NVIDIA H100 (Capella) | 80 GiB | `$TRK/.venv` (cluster) | **never completed** |

Local data: `data/vanvliet/` (3.1 GB; 6 conditions, 33 experiment dirs).
All local scripts resolve data relative to the repo root.

---

## 2. Prerequisites

### 2.1 Data
`data/vanvliet/` at repo root. Structure per condition:
`data/vanvliet/<condition>/<experiment>/img/tNNNNNN.tif` + `TRA/man_trackNNNNNN.tif`.

### 2.2 Local environment (`benchmark_ssl/.venv`)
Key packages: `torch 2.12.0+cu130`, `lightly 1.5.25`, `scikit-learn 1.9.0`,
`numpy 2.4.6`, `scipy 1.17.1`, `pandas 3.0.3`, `scikit-image 0.26.0`,
`tifffile 2026.6.1`, `matplotlib 3.11.0`, `seaborn 0.13.2`, `PyYAML 6.0.3`,
`tqdm 4.68.2`, `dask 2026.6.0`, `edt 3.1.1`, `fast-regionprops 0.2.0`, `kornia 0.8.3`.

`uv sync` (per `pyproject.toml`) recreates the environment; some producers also
need `lightly` and `edt`.

### 2.3 Cluster paths (for H100 runs)

All cluster paths (`REPO_ROOT`, `TRK`, `BENCH`, `DATA_DIR`) are read
from `~/.bench.env` by `slurm/load_cluster_env.sh`.  Each slurm script
sources `~/.bench.env` first, then the loader.

#### One-time setup (login node)

```bash
git clone <repo-url> bench-transformer-cell-tracking
cd bench-transformer-cell-tracking
bash slurm/setup_env.sh          # creates ~/.bench.env, runs uv sync
```

`setup_env.sh` will prompt you to edit `~/.bench.env` — replace the
`/path/to/...` placeholders with your actual cluster paths.  Then re-run
the script to complete the setup (creates `.venv/` in each subproject
and installs all deps).  Slurm jobs use `.venv/bin/python` directly —
**no `uv sync` runs inside the job**, so the full time budget is
available for the actual experiment.

After `git pull`, the slurm files never need to be edited again — they
read `REPO_ROOT` from `~/.bench.env`.

| Constant | Value |
|---|---|
| `TRK` | `/path/to/trackastra` |
| `DATA_DIR` | `/path/to/celltracking/data` |
| Partition / account | `gpu-h100` / `p_scads_celltracking` |

---

## 3. Experiment inventory

| Experiment | Entrypoint | Output artifact | Status |
|---|---|---|---|
| Identity-BCE mini SSL | `python -m src.training.ssl_pretrainer` | `runs/ssl_v1/best_model.pt`, `training_log.csv` | DONE (A500) |
| Attention K-sweep SSL | `python -m src.training.ssl_trainer_multi` | `runs/ssl_dense/`, `runs/ssl_K=4/8/16/32/` | DONE (A500) |
| SSL vs random fine-tuning | `python -m src.analysis.downstream_comparison` | `runs/downstream_compare/comparison_legacy.csv` | DONE (A500) |
| Feature-signal analysis (5 sets) | `python -m src.analysis.signal_analyzer` | `runs/feature_signal*/feature_signal_results.csv` + `.html` | DONE (A500) |
| Convergence predictor + micro-SSL | `python -m src.analysis.convergence_predictor` | `runs/convergence_prediction/*.html` + CSVs | DONE (A500) |
| Downstream convergence | `python -m src.analysis.downstream_convergence` | `runs/downstream_convergence/convergence.csv` | DONE (A500) |
| DINO backbone comparison | `python -m src.analysis.dino_backbone_comparison` | `runs/dino_comparison/comparison.csv`, `.png`, `verdict.txt` | DONE (A500) |
| Coordinate shortcut diagnostic | `python -m src.analysis.coord_shortcut_diagnostic` | `runs/diagnose_coord_shortcut*/mode_{A,B,C}.csv`, PNG | DONE (A500) |
| End-to-end SSL → downstream | `python -m src.analysis.end_to_end_diagnostic` | `runs/diagnose_end_to_end/downstream_{R,A,B,C}.csv`, PNG | DONE (A500) |
| Unified edge probe (10-fold CV + per-pair significance) | `python -m src.edge_probing.unified_edge_probe` + `python -m src.analysis.probe_paired_significance` | `results/probes/unified_probe_results_cv10.json` + `probe_paired_significance.json` | DONE (A500, refreshed 2026-09-24; supersedes 5-fold) |
| DINOv2 full SSL pretraining | — | `$TRK/runs/ssl_dino_pretrain/` | **NOT-NEEDED** (never completed) |
| Multi-seed K-sweep (42/43/44) | `run_single_config.slurm` (cluster) | `results/knn_sweep/` variance estimates | PENDING (cluster) |
| DeepCell cross-dataset eval | `slurm/cross_dataset_eval.slurm` (cluster) | `$TRK/results/cross_dataset/deepcell/...` | PENDING (cluster) |

> **Note on `comparison_legacy.csv`:** The on-disk artifact was produced by
> the *earlier* fine-tuning variant (`epoch,model,train_loss,train_acc,val_loss,val_acc`).
> The current `src/analysis/downstream_comparison.py` is a rewrite that
> performs embedding + Hungarian tracking and writes a different schema
> (`model,val_accuracy,val_correct,val_total,test_accuracy,test_correct,test_total,test_mismatches,test_misses`).
> The legacy file is renamed `comparison_legacy.csv` to avoid confusion.

---

## 4. Measured results

### 4.1 NT-Xent contrastive SSL collapse (H100)
NT-Xent train loss 3.46 → 2.97; global inter-cell cosine similarity (collapse): **0.89**.
SSL provides zero improvement over random embeddings (knn_SSL 65.1% vs random 64.8%).

### 4.2 Identity-BCE SSL pretraining
val_loss final: baseline **0.416** vs SSL-finetune **0.404** — nearly identical.
SSL provides zero benefit over random initialization.

> **Variant note:** `runs/ssl_v1/` artifacts (`training_log.csv`,
> `best_model.pt`, TensorBoard event) were produced by the older
> **identity-BCE** pretraining variant, not by the current NT-Xent rewrite
> in `src/training/ssl_pretrainer.py`. The archived
> `runs/ssl_v1/config.yaml` (`d_model=128, nhead=4`) is the authoritative
> config for this run, superseding `docs/results_analysis.md` which
> incorrectly states `d_model=64, nhead=2`.

### 4.3 DINOv2 feature micro-tests
DINO gap **0.292** vs regionprops 0.047 (6.2×); effective dim 108 vs 3.
Micro-SSL generalization: val gap 0.481 → 0.577 (+0.095, genuine learning).
Hard pipeline: DINO collapses to gap 0.040; after 200 micro-SSL steps gap 0.359.

### 4.4 Feature-signal analysis (`python -m src.analysis.signal_analyzer`)
All feature sets give separation gap ≈ 0.045–0.047 and effective dim 3 (FAIL).
Richer features (shape/Hu/patch) do not improve the gap.

### 4.5 Convergence predictor (`python -m src.analysis.convergence_predictor`)
Baseline DINO gap 0.292; micro-SSL val gap 0.561 → 0.698 in 200 steps.
Hard pipeline gap 0.040 → 0.359 (+0.319, no collapse).

`--test` / `--outdir` mapping (each invocation writes one sub-run):

| `--test` | `--outdir` | Artifacts |
|---|---|---|
| `all` | `runs/convergence_prediction/` | All tests below |
| `gap_strength` | `runs/convergence_prediction/` | `gap_vs_strength.csv` |
| `micro_ssl` | `runs/convergence_prediction/` | `micro_ssl.csv` |
| `generalization` | `runs/generalization_test/` | `micro_ssl_generalization.csv` |
| `hard` | `runs/hard_test/` | `micro_ssl_hard.csv` |
| `ranking` | `runs/convergence_prediction/` | `distortion_ranking.csv` |
| `pca` | `runs/convergence_prediction/` | `pca_analysis.csv` |

### 4.6 Downstream convergence (`python -m src.analysis.downstream_convergence`)
SSL-pretrained vs Random-init over 20 epochs: final val_acc 0.504 vs 0.498 — no advantage.

### 4.7 Coordinate shortcut (`python -m src.analysis.coord_shortcut_diagnostic`)
**SMOKING GUN** — Mode C (PE only, zero visual input) solves NT-Xent in step 2.
Removing coordinates improves final val gap by +0.016 (full) and +0.228 (jitter4).

### 4.8 End-to-end transfer (`python -m src.analysis.end_to_end_diagnostic`)
PE+coords+DINO reaches val acc 0.969 in 10 steps; PE-only never exceeds 0.750.
SSL does not help downstream convergence at this scale.

### 4.9 DINO backbone comparison (`python -m src.analysis.dino_backbone_comparison`)
| backbone | gap | recall@1 | eff. rank | cells/s | VRAM (MiB) |
|---|---|---|---|---|---|
| DINOv2-vits14 | 0.2435 | 0.9715 | 100 | 115.9 | 263 |
| DINOv2-vitb14 | 0.2649 | 0.9714 | 195 | 25.3 | 657 |

Verdict: "Feature quality bottleneck is domain mismatch, not model capacity."

### 4.10 Unified edge probe (10-fold CV + per-pair significance) — `results/probes/unified_probe_results_cv10.json`

Supersedes the 2026-08 5-fold run (`results/probes/unified_probe_results_cv.json`,
kept for provenance): the probe script evolved since (hoct19 renamed hoct2d,
Fourier-PE feature variants added), so the old numbers are not reproducible with
current code and the table below replaces the one in the report.

Command (from `benchmark_ssl/`, local A500 or any CUDA GPU):

    .venv/bin/python -m src.edge_probing.unified_edge_probe \
      --data-root ../data/vanvliet --features all --probe both \
      --cv-folds 10 --max-pairs 30 --seed 42 --dump-predictions \
      --checkpoint results/checkpoints/cnn/cnn_ntxent_large.pt \
      --output results/probes/unified_probe_results_cv10.json
    .venv/bin/python -m src.analysis.probe_paired_significance \
      --input results/probes/probe_paired_significance.json \
      --output results/probes/probe_paired_significance.json --check

Per-feature balanced accuracy (MLP probe, 10-fold, seed 42):

| Feature (MLP probe) | balanced acc (mean ± std) | F1 |
|---|---|---|
| Regionprops 7D + Fourier PE | 0.9626 ± 0.0235 | 0.735 ± 0.097 |
| HOCT 2D + Fourier PE | 0.9287 ± 0.0952 | 0.680 |
| DINOv2 (frozen) | 0.9226 ± 0.0511 | 0.719 |
| HOCT 2D (13D) | 0.8644 ± 0.1528 | 0.531 |
| Regionprops 7D | 0.8206 ± 0.0908 | 0.378 |
| CNN NT-Xent (frozen) | 0.7753 ± 0.0779 | 0.354 |
| CNN end-to-end | 0.5000 ± 0.0000 | 0.000 ± 0.000 |

Significance vs the 7D-regionprops reference (primary unit = matched frame pair,
n = 29; paired t-test, bootstrap 95% CI, Monte-Carlo sign-flip permutation,
Holm across feature sets; computed by `src/analysis/probe_paired_significance.py`,
verified values in `docs/agent-state/specs/0006-probe-ci/SPEC.md`):

| Feature | Δ bal_acc [95% CI] | p (Holm) |
|---|---|---|
| Regionprops 7D + Fourier PE | +0.171 [+0.121, +0.220] | 1.9e-06 |
| DINOv2 (frozen) | +0.160 [+0.109, +0.212] | 9.1e-06 |
| HOCT 2D + Fourier PE | +0.123 [+0.061, +0.178] | 1.1e-03 |
| CNN NT-Xent (frozen) | +0.065 [−0.008, +0.136] | 0.18 (n.s.) |
| HOCT 2D | +0.039 [−0.028, +0.102] | 0.26 (n.s.) |
| CNN end-to-end | −0.235 [−0.285, −0.182] | 1.2e-08 |

Note: fold-level statistics (per-batch-mean balanced accuracy) are reported as a
secondary unit in `probe_paired_significance.json`; the frame pair is the
statistical unit because every frame pair is validated exactly once.

---

## 5. Early-stopping guidance

Never run to 500 epochs if val_loss has plateaued — patience 83 already stops
near-optimal. `--epochs 500` is only an upper bound.

---

## 6. Known discrepancies & reproducibility notes

1. **No cluster K-sweep / CNN / DeepCell artifacts are reproducible locally**
   without pulling clean run dirs off Capella.
2. **DINOv2 full SSL was never completed** — no checkpoint exists.