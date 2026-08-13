# SPEC.md — `benchmark_ssl` Refactor

*Status: APPROVED — 2026-08-12. User decisions: (1) pretrain.py — add `--loss {identity_bce,ntxent}` CLI flag; (2) downstream_compare.py — add `--mode {finetune,embedding}` CLI flag; (3) exploratory/test_local.py — DELETE.*

---

## 1. Goal

Restructure `benchmark_ssl/` so it follows the standard project layout mandated by `research-proj/AGENTS.md` (`src/`, `tests/`, `scripts/`, `slurm/`, `configs/`, `results/`, `docs/`), eliminate deprecated and duplicated code, make every module/file conform to the naming and documentation rules, and update all SLURM scripts, configs, and reproduction guides so they reference the new layout.

The deliverable is a repository that a new contributor can navigate without a guided tour.

## 2. Problem statement (current state)

Measured on `2026-08-12`:

| Symptom | Evidence |
|---|---|
| Shared modules live at the repo root | `ssl_pipeline.py`, `distortions.py`, `track_encoder.py`, `model_parts.py`, `rich_features.py` — five root-level files that every subdirectory imports |
| Code duplication | `_border_dist_fast` defined identically in `ssl_pipeline.py` and `rich_features.py`; feature extraction logic reimplemented across `ssl/`, `probe/`, `cnn_encoder/` |
| Deprecated code mixed with active code | `cnn_encoder/cnn_probe.py` (531 lines, 0.500 result, superseded); `probe/edge_probe.py` (1000 lines, older single-split); `ssl/pretrain.py` (rewrite that no longer produces `training_log.csv`); `ssl/pretrain_multi.py` (references undefined `ROOT`); `ssl/downstream_compare.py` (rewrite, different CSV schema) |
| Inconsistent naming | `pretrain.py` vs `pretrain_multi.py` vs `train_ssl.py`; `cnn_ssl.py` vs `cnn_probe.py`; `ssl_pipeline.py` vs `ssl_convergence_test.py`; `model_parts.py` is a grab-bag of 7 attention classes |
| No `tests/` directory | Zero unit tests for 17,208 lines of Python |
| Results mixed with source | `runs/`, `results/`, `probe/feature_cache/`, `cnn_encoder/probe/checkpoints` sit next to source files |
| Nested package | `benchmark_ssl/benchmark_ssl/results/...` — confusing double-nested package directory |
| Scattered documentation | Three `REPRODUCTION.md` files (root, `cnn_encoder/`, `probe/`); `results_analysis.md` at root; `benchmark_ssl.html`, `expected_metrics.html` in root |
| Build artifacts present | `.venv/`, `__pycache__/` directories on disk (gitignored but should not be tracked) |
| Random HTML reports | `benchmark_ssl.html`, `expected_metrics.html` in root |

## 3. Target structure

Standard `src/` layout per `AGENTS.md`:

```
benchmark_ssl/
├── src/          # Python package (importable as `benchmark_ssl`)
│   ├── __init__.py
│   ├── data/                   # Data loading + feature extraction
│   │   ├── __init__.py
│   │   ├── frame_loader.py     # load_experiment_frames (from ssl_pipeline.py)
│   │   ├── feature_extraction.py # features_from_frame, rich_features, _border_dist_fast (consolidated)
│   │   └── ssl_dataset.py      # SSLDataset, collate_ssl (from ssl_pipeline.py)
│   ├── distortions/            # Augmentation pipeline
│   │   ├── __init__.py
│   │   └── distortion_pipeline.py # DistortionPipeline + all 6 distortion families (from distortions.py)
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   ├── cell_embedder.py    # CellEmbedder (from track_encoder.py)
│   │   ├── attention.py        # All attention classes (from model_parts.py)
│   │   └── scaled_cnn.py       # ScaledCNN (from cnn_encoder/cnn_ssl.py)
│   ├── training/               # SSL training
│   │   ├── __init__.py
│   │   ├── ssl_pretrainer.py   # NT-Xent pretraining (from ssl/pretrain.py)
│   │   └── ssl_trainer_multi.py # Multi-config SSL training (from ssl/pretrain_multi.py, ROOT fixed)
│   ├── analysis/               # Signal/convergence/downstream analyses
│   │   ├── __init__.py
│   │   ├── signal_analyzer.py  # Data-level signal analysis (from ssl/analyze_signal.py)
│   │   ├── convergence_predictor.py # Micro-SSL predictor (from ssl/ssl_convergence_test.py)
│   │   ├── downstream_convergence.py # Downstream convergence test (from ssl/downstream_convergence.py)
│   │   ├── coord_shortcut_diagnostic.py # Coordinate shortcut diagnostic (from ssl/diagnose_coord_shortcut.py)
│   │   ├── end_to_end_diagnostic.py # End-to-end SSL diagnostic (from ssl/diagnose_end_to_end.py)
│   │   ├── dino_backbone_comparison.py # DINO backbone comparison (from ssl/compare_dino_backbones.py)
│   │   ├── downstream_comparison.py # SSL vs random fine-tune (from ssl/downstream_compare.py)
│   │   ├── feature_probe.py    # Feature probing (from ssl/probe_features.py)
│   │   ├── dino_test.py        # DINO separation test (from ssl/test_dino.py)
│   │   ├── plot_ssl.py         # SSL plotting (from analysis/plot_ssl.py)
│   │   └── plot_expected_metrics.py # Expected metrics plotting (from analysis/plot_expected_metrics.py)
│   ├── probing/                # Edge probing framework
│   │   ├── __init__.py
│   │   ├── unified_edge_probe.py # Unified 5-fold CV edge probe (from probe/unified_edge_probe.py)
│   │   ├── train_cnn_probe.py  # TinyCNN e2e probe (from probe/train_cnn_probe.py)
│   │   ├── verify_normalization.py # Normalization verification (from probe/verify_normalization.py)
│   │   └── feature_distribution_histograms.py # Feature distribution viz (from probe/feature_distribution_histograms.py)
│   ├── cnn_encoder/            # CNN-specific experiments
│   │   ├── __init__.py
│   │   ├── cnn_ssl.py          # ScaledCNN NT-Xent pretraining (from cnn_encoder/cnn_ssl.py)
│   │   ├── cnn_convergence.py  # CNN convergence test (from cnn_encoder/cnn_convergence.py)
│   │   ├── h100_convergence.py # H100 convergence race (from cnn_encoder/h100_convergence.py)
│   │   ├── cross_dataset_eval.py # DeepCell cross-dataset eval (from cnn_encoder/cross_dataset_eval.py)
│   │   ├── scale_sweep.py      # Scale sweep (from cnn_encoder/scale_sweep.py)
│   │   ├── visualize_embeddings.py # Embedding visualization (from cnn_encoder/visualize_embeddings.py)
│   │   └── smoke_test.py       # Smoke test (from cnn_encoder/smoke_test_cnn_trackastra.py)
│   └── mini_trackastra/        # Mini trackastra experiments
│       ├── __init__.py
│       ├── compare_backbones.py
│       ├── end_to_end.py
│       ├── end_to_end_bce.py
│       ├── temporal_ssl.py
│       ├── test_barlow.py
│       ├── test_cell_dino.py
│       └── test_freeze.py
├── scripts/                     # Thin entry-point wrappers (one per runnable)
│   ├── run_ssl_pretrain.py
│   ├── run_ssl_trainer_multi.py
│   ├── run_signal_analyzer.py
│   ├── run_convergence_predictor.py
│   ├── run_downstream_convergence.py
│   ├── run_coord_shortcut_diagnostic.py
│   ├── run_end_to_end_diagnostic.py
│   ├── run_dino_backbone_comparison.py
│   ├── run_downstream_comparison.py
│   ├── run_feature_probe.py
│   ├── run_dino_test.py
│   ├── run_unified_edge_probe.py
│   ├── run_train_cnn_probe.py
│   ├── run_verify_normalization.py
│   ├── run_feature_distribution_histograms.py
│   ├── run_cnn_ssl.py
│   ├── run_cnn_convergence.py
│   ├── run_h100_convergence.py
│   ├── run_cross_dataset_eval.py
│   ├── run_scale_sweep.py
│   ├── run_visualize_embeddings.py
│   ├── run_smoke_test.py
│   ├── run_plot_ssl.py
│   └── run_plot_expected_metrics.py
├── tests/                       # Unit and integration tests (NEW)
│   ├── __init__.py
│   ├── test_distortions.py     # Smoke tests for each distortion family
│   ├── test_feature_extraction.py # Regionprops + rich features
│   ├── test_ssl_dataset.py     # Dataset + collator
│   ├── test_cell_embedder.py   # Forward pass, shapes
│   ├── test_attention.py       # Attention modules shapes
│   └── test_scaled_cnn.py      # ScaledCNN shapes
├── slurm/                       # HPC submission scripts (all here)
│   ├── run_ssl_dino_pretrain.slurm      (from slurm/)
│   ├── run_full_cnn_race.slurm          (from cnn_encoder/)
│   ├── run_full_cnn_extended.slurm      (from cnn_encoder/)
│   ├── run_lambda_decay.slurm           (from cnn_encoder/)
│   ├── run_h100_conv_race.slurm         (from cnn_encoder/)
│   ├── run_cross_dataset_eval.slurm     (from cnn_encoder/)
│   ├── run_viz.slurm                    (from cnn_encoder/)
│   └── run_unified_probe.slurm          (from probe/)
├── configs/                     # All YAML configs
│   └── config.yaml             # (from root)
├── results/                     # All experiment outputs (moved here)
│   ├── runs/                   # (from runs/)
│   ├── probes/                 # (from results/probes/ + from probe/feature_cache/ — UNIFY)
│   ├── checkpoints/            # (from cnn_encoder/probe/)
│   └── feature_cache/          # (from probe/feature_cache/ — UNIFY)
├── docs/                        # Documentation
│   ├── SPEC.md                 # (this document)
│   └── REPRODUCTION.md         # (consolidated from 3 REPRODUCTION.md files)
├── pyproject.toml               # Updated for src/ layout
├── uv.lock
├── .gitignore
└── README.md                    # Updated with new structure
```

### 3.1 What stays where

| Original location | New location | Reason |
|---|---|---|
| `ssl_pipeline.py` (root) | `src/data/{ssl_dataset,frame_loader,feature_extraction}.py` | Split by responsibility per AGENTS.md ("one clear responsibility per file") |
| `distortions.py` (root) | `src/distortions/distortion_pipeline.py` | All 6 distortion families + pipeline are one cohesive module |
| `track_encoder.py` (root) | `src/models/cell_embedder.py` | One model per file |
| `model_parts.py` (root) | `src/models/attention.py` | All attention-related classes together |
| `rich_features.py` (root) | `src/data/feature_extraction.py` | Merged with features_from_frame; deduplicate `_border_dist_fast` |
| `ssl/*` | `src/{training,analysis}/*` | Split by purpose: training scripts → `training/`, analysis/diagnostic scripts → `analysis/` |
| `cnn_encoder/*` | `src/cnn_encoder/*` | Same name, new path under src/ |
| `probe/*` | `src/probing/*` | Note: renamed `probe` → `probing` (verb form) to avoid namespace collision with package functions like `cnn_probe.py` |
| `analysis/*` | `src/analysis/*` | Same name, new path |
| `exploratory/*` | `src/exploratory/*` (or `scripts/exploratory/`) | Keep as exploratory scripts |
| `mini_trackastra/*` | `src/mini_trackastra/*` | Same name, new path |
| `runs/` | `results/runs/` | Per AGENTS.md standard layout |
| `results/` | `results/` (keep, merge contents) | |
| `probe/feature_cache/` | `results/feature_cache/` | Unify with results |
| `cnn_encoder/probe/` | `results/checkpoints/` | Unify with results |
| `slurm/` | `slurm/` | Same name, merge all `.slurm` files from other directories |
| `config.yaml` | `configs/config.yaml` | Per AGENTS.md standard layout |
| `REPRODUCTION.md` (root) | `docs/REPRODUCTION.md` | Consolidated |
| `REPRODUCTION.md` (cnn_encoder) | `docs/REPRODUCTION.md` (merged section) | |
| `REPRODUCTION.md` (probe) | `docs/REPRODUCTION.md` (merged section) | |
| `results_analysis.md` | `docs/results_analysis.md` | |
| `benchmark_ssl.html`, `expected_metrics.html` | **DELETE** | Generated artifacts, not source |
| `benchmark_ssl/benchmark_ssl/` (nested) | **DELETE** | Confusing double-nested package |
| `.venv/`, `__pycache__/` | **DELETE** (gitignored anyway) | |

### 3.2 Deprecated code to remove

| File | Reason | Evidence |
|---|---|---|
| `cnn_encoder/cnn_probe.py` | Superseded by `unified_edge_probe.py`; 0.500 result explicitly marked "must NOT be cited" | `REPRODUCTION.md` §4.2 |
| `probe/edge_probe.py` | Older single-split probe; 80/20 split replaced by 5-fold CV | `probe/REPRODUCTION.md` §3 |
| `ssl/pretrain_multi.py` | References undefined `ROOT`; raises `NameError` on run | `REPRODUCTION.md` §10.1 |

### 3.3 Files with known discrepancies to resolve

These files have rewrites that no longer match the on-disk results. The user must decide:

| File | Issue | Options |
|---|---|---|
| `ssl/pretrain.py` | Current is NT-Xent rewrite; `runs/ssl_v1/` was produced by identity-BCE variant | (a) keep rewrite, mark old variant as deprecated in REPRODUCTION; (b) revert to identity-BCE; (c) add a CLI flag to switch |
| `ssl/downstream_compare.py` | Current is embedding + Hungarian tracking rewrite; `runs/downstream_compare/` was produced by older fine-tune variant | Same as above |

*Decision deferred to user — see §10.*

### 3.4 Import convention

All internal imports use the package name:

```python
# CORRECT
from benchmark_ssl.data import SSLDataset, collate_ssl
from benchmark_ssl.distortions import DistortionPipeline
from benchmark_ssl.models import CellEmbedder

# WRONG (old style)
from ssl_pipeline import SSLDataset, collate_ssl
from distortions import DistortionPipeline
from track_encoder import CellEmbedder
```

Scripts under `scripts/` and modules under `src/` must never use relative imports across subpackages. Within a subpackage, relative imports are allowed.

## 4. Acceptance criteria

The refactor is **DONE** when **all** of the following are true:

### 4.1 Structure
- [ ] `src/` is an importable Python package (`python -c "import benchmark_ssl"` succeeds)
- [ ] No Python files remain at the repo root (root-level `*.py` files all moved to `src/`)
- [ ] No `__pycache__/` directories or `.venv/` in the tracked tree
- [ ] The nested `benchmark_ssl/benchmark_ssl/` directory is removed
- [ ] All root-level `.html` files (`benchmark_ssl.html`, `expected_metrics.html`) are deleted

### 4.2 Code quality
- [ ] Every module, class, function, and script has a docstring (per AGENTS.md)
- [ ] No single-letter names except loop counters `i, j, k` and math notation
- [ ] No `from ssl_pipeline import …` style imports — all internal imports go through `benchmark_ssl.…`
- [ ] `_border_dist_fast` appears in exactly **one** place (`src/data/feature_extraction.py`)
- [ ] Deprecated files removed: `cnn_encoder/cnn_probe.py`, `probe/edge_probe.py`, `ssl/pretrain_multi.py`
- [ ] Every file in `src/` has a unique, descriptive name (no `model_parts.py` style grab-bags)

### 4.3 Results consolidation
- [ ] All experiment outputs under `results/`: `runs/`, `probes/`, `checkpoints/`, `feature_cache/`
- [ ] No result directories (`runs/`, `probe/feature_cache/`, `cnn_encoder/probe/`) remain at the old locations

### 4.4 Configuration & deployment
- [ ] All `.slurm` scripts under `slurm/`
- [ ] `config.yaml` under `configs/`
- [ ] Every SLURM script's path references updated to the new layout
- [ ] `pyproject.toml` updated for src/ layout (add `[tool.setuptools.packages.find]` with `where = ["src"]`)

### 4.5 Documentation
- [ ] `docs/SPEC.md` exists (this document)
- [ ] `docs/REPRODUCTION.md` is a single consolidated reproduction guide (merged from the 3 original REPRODUCTION.md files)
- [ ] `README.md` reflects the new structure and how to run each script

### 4.6 Verification
- [ ] Every script in `scripts/` can be invoked with `--help` without ImportError
- [ ] `python -c "from benchmark_ssl.data import SSLDataset, collate_ssl"` succeeds
- [ ] `python -c "from benchmark_ssl.distortions import DistortionPipeline"` succeeds
- [ ] `python -c "from benchmark_ssl.models import CellEmbedder"` succeeds
- [ ] `python -c "from benchmark_ssl.models.attention import RotaryPositionalEncoding, GatherSparseAttention, RelativePositionalAttention"` succeeds
- [ ] `python -c "from benchmark_ssl.probing.unified_edge_probe import …"` succeeds
- [ ] All `tests/` pass (new tests written for the refactored core modules)

## 5. Migration map (file by file)

### 5.1 Root-level shared modules

| Source | Destination | Notes |
|---|---|---|
| `ssl_pipeline.py::features_from_frame` + `rich_features.py` | `src/data/feature_extraction.py` | Merge, deduplicate `_border_dist_fast` |
| `ssl_pipeline.py::load_experiment_frames` | `src/data/frame_loader.py` | |
| `ssl_pipeline.py::SSLDataset` + `collate_ssl` | `src/data/ssl_dataset.py` | |
| `distortions.py` (entire file) | `src/distortions/distortion_pipeline.py` | |
| `track_encoder.py::CellEmbedder` | `src/models/cell_embedder.py` | |
| `model_parts.py` (entire file) | `src/models/attention.py` | |
| `cnn_encoder/cnn_ssl.py::ScaledCNN` | `src/models/scaled_cnn.py` | Extract ScaledCNN class from cnn_ssl.py |

### 5.2 ssl/ scripts

| Source | Destination | Notes |
|---|---|---|
| `ssl/pretrain.py` | `src/training/ssl_pretrainer.py` | See §3.3 for rewrite decision |
| `ssl/pretrain_multi.py` | `src/training/ssl_trainer_multi.py` | Fix `ROOT` global (replace with `Path(__file__).resolve().parents[3]`) |
| `ssl/train_ssl.py` | `src/training/ssl_trainer.py` | |
| `ssl/analyze_signal.py` | `src/analysis/signal_analyzer.py` | |
| `ssl/ssl_convergence_test.py` | `src/analysis/convergence_predictor.py` | |
| `ssl/downstream_convergence.py` | `src/analysis/downstream_convergence.py` | |
| `ssl/diagnose_coord_shortcut.py` | `src/analysis/coord_shortcut_diagnostic.py` | |
| `ssl/diagnose_end_to_end.py` | `src/analysis/end_to_end_diagnostic.py` | |
| `ssl/compare_dino_backbones.py` | `src/analysis/dino_backbone_comparison.py` | |
| `ssl/downstream_compare.py` | `src/analysis/downstream_comparison.py` | See §3.3 |
| `ssl/probe_features.py` | `src/analysis/feature_probe.py` | |
| `ssl/test_dino.py` | `src/analysis/dino_test.py` | |

### 5.3 cnn_encoder/ scripts

| Source | Destination | Notes |
|---|---|---|
| `cnn_encoder/cnn_ssl.py` (minus ScaledCNN class) | `src/cnn_encoder/cnn_ssl.py` | ScaledCNN moves to `models/scaled_cnn.py` |
| `cnn_encoder/cnn_probe.py` | **DELETE** | Deprecated |
| `cnn_encoder/cnn_convergence.py` | `src/cnn_encoder/cnn_convergence.py` | |
| `cnn_encoder/h100_convergence.py` | `src/cnn_encoder/h100_convergence.py` | |
| `cnn_encoder/cross_dataset_eval.py` | `src/cnn_encoder/cross_dataset_eval.py` | |
| `cnn_encoder/scale_sweep.py` | `src/cnn_encoder/scale_sweep.py` | |
| `cnn_encoder/visualize_embeddings.py` | `src/cnn_encoder/visualize_embeddings.py` | |
| `cnn_encoder/smoke_test_cnn_trackastra.py` | `src/cnn_encoder/smoke_test.py` | Rename to drop redundant `_cnn_trackastra` |

### 5.4 probe/ scripts

| Source | Destination | Notes |
|---|---|---|
| `probe/unified_edge_probe.py` | `src/probing/unified_edge_probe.py` | |
| `probe/edge_probe.py` | **DELETE** | Deprecated (older single-split) |
| `probe/train_cnn_probe.py` | `src/probing/train_cnn_probe.py` | |
| `probe/verify_normalization.py` | `src/probing/verify_normalization.py` | |
| `probe/feature_distribution_histograms.py` | `src/probing/feature_distribution_histograms.py` | |

### 5.5 analysis/, exploratory/, mini_trackastra/

| Source | Destination |
|---|---|
| `analysis/plot_ssl.py` | `src/analysis/plot_ssl.py` |
| `analysis/plot_expected_metrics.py` | `src/analysis/plot_expected_metrics.py` |
| `exploratory/test_local.py` | `scripts/exploratory_test_local.py` (or delete if unused) |
| `mini_trackastra/*.py` | `src/mini_trackastra/*.py` |

### 5.6 SLURM scripts

| Source | Destination |
|---|---|
| `slurm/run_ssl_dino_pretrain.slurm` | `slurm/run_ssl_dino_pretrain.slurm` |
| `cnn_encoder/run_full_cnn_race.slurm` | `slurm/run_full_cnn_race.slurm` |
| `cnn_encoder/run_full_cnn_extended.slurm` | `slurm/run_full_cnn_extended.slurm` |
| `cnn_encoder/run_lambda_decay.slurm` | `slurm/run_lambda_decay.slurm` |
| `cnn_encoder/run_h100_conv_race.slurm` | `slurm/run_h100_conv_race.slurm` |
| `cnn_encoder/run_cross_dataset_eval.slurm` | `slurm/run_cross_dataset_eval.slurm` |
| `cnn_encoder/run_viz.slurm` | `slurm/run_viz.slurm` |
| `probe/run_unified_probe.slurm` | `slurm/run_unified_probe.slurm` |

### 5.7 Configs

| Source | Destination |
|---|---|
| `config.yaml` | `configs/config.yaml` |

### 5.8 Results

| Source | Destination |
|---|---|
| `runs/` | `results/runs/` |
| `results/probes/` | `results/probes/` (keep, merge with anything in probe/feature_cache) |
| `probe/feature_cache/` | `results/feature_cache/` |
| `cnn_encoder/probe/cnn_*.pt` | `results/checkpoints/cnn/` |
| `cnn_encoder/probe/cnn_*_compare/` | `results/checkpoints/cnn/compare/` |

### 5.9 Documentation

| Source | Destination | Notes |
|---|---|---|
| `REPRODUCTION.md` (root) | `docs/REPRODUCTION.md` | Merge all three |
| `cnn_encoder/REPRODUCTION.md` | `docs/REPRODUCTION.md` (merged) | |
| `probe/REPRODUCTION.md` | `docs/REPRODUCTION.md` (merged) | |
| `results_analysis.md` | `docs/results_analysis.md` | |
| `benchmark_ssl.html` | **DELETE** | Generated artifact |
| `expected_metrics.html` | **DELETE** | Generated artifact |

## 6. Task breakdown for multi-agent execution

The refactor is split into **10 tasks**. Tasks 1–2 must complete before tasks 3–6 (shared modules are imported by everything). Tasks 7–9 can run in parallel after their prerequisites. Task 10 (cleanup) runs last.

```
Phase A — Foundation (sequential)
  T1  Create src/ package skeleton with __init__.py files
  T2  Migrate shared modules (ssl_pipeline, distortions, track_encoder, model_parts, rich_features)

Phase B — Script migration (parallel after T2)
  T3  Migrate ssl/ scripts → src/{training,analysis}/
  T4  Migrate cnn_encoder/ scripts → src/cnn_encoder/
  T5  Migrate probe/ scripts → src/probing/

Phase C — Consolidation (parallel after Phase B)
  T6  Migrate analysis/, exploratory/, mini_trackastra/ scripts
  T7  Consolidate all results under results/
  T8  Move + update all SLURM scripts, configs, pyproject.toml

Phase D — Cleanup and verification (sequential)
  T9  Clean up deprecated code, __pycache__, nested directories, root-level HTML
  T10 Verification: run all acceptance checks from §4
```

Each task is assigned to a separate agent. Agents in Phase B and C can run in parallel because they touch disjoint files. Agents in Phase B must not modify files that other Phase B agents are modifying.

### Task specifications

#### T1: Package skeleton

**Agent scope:** Create directories and `__init__.py` files only. No code movement yet.

**Files to create:**
```
src/__init__.py
src/data/__init__.py
src/distortions/__init__.py
src/models/__init__.py
src/training/__init__.py
src/analysis/__init__.py
src/probing/__init__.py
src/cnn_encoder/__init__.py
src/mini_trackastra/__init__.py
src/exploratory/__init__.py   # if exploratory scripts are kept
scripts/__init__.py
tests/__init__.py
slurm/                                          # already exists at root
configs/                                        # create
results/                                        # create (consolidate later)
results/runs/                                   # create
results/probes/                                 # create
results/checkpoints/                            # create
results/feature_cache/                          # create
docs/                                           # already created
```

**Acceptance:** All directories exist with `__init__.py`. `python -c "import benchmark_ssl"` fails with ModuleNotFoundError (expected — package is empty).

#### T2: Migrate shared modules

**Agent scope:** Move 5 root-level shared modules into the new package, deduplicate `_border_dist_fast`.

**Files to touch:**
- Read: `ssl_pipeline.py`, `distortions.py`, `track_encoder.py`, `model_parts.py`, `rich_features.py`
- Write: `src/data/feature_extraction.py`, `src/data/frame_loader.py`, `src/data/ssl_dataset.py`, `src/distortions/distortion_pipeline.py`, `src/models/cell_embedder.py`, `src/models/attention.py`

**Splitting `ssl_pipeline.py`:**
- `features_from_frame` + `_border_dist_fast` + rich-features constants → `data/feature_extraction.py`
- `load_experiment_frames` → `data/frame_loader.py`
- `SSLDataset`, `collate_ssl` → `data/ssl_dataset.py`

**Deduplication:**
- `_border_dist_fast` exists in both `ssl_pipeline.py` (line 40) and `rich_features.py` (line 43). Keep one copy in `data/feature_extraction.py`. The `ssl_pipeline.py` version is the canonical one (rich_features.py has a comment saying "reused from ssl_pipeline").

**Import updates within the package:**
```python
# In src/data/ssl_dataset.py
from benchmark_ssl.data.feature_extraction import features_from_frame

# In src/data/feature_extraction.py
# (no internal cross-imports needed)
```

**Acceptance:** `python -c "from benchmark_ssl.data import SSLDataset, collate_ssl, features_from_frame, load_experiment_frames"` succeeds.

#### T3: Migrate ssl/ scripts (Agent: ssl-migrator)

**Agent scope:** Move all 12 files from `ssl/` into `src/{training,analysis}/`, update all imports, fix `pretrain_multi.py` ROOT global.

**Files to touch:**
- Read: `ssl/pretrain.py`, `ssl/pretrain_multi.py`, `ssl/train_ssl.py`, `ssl/analyze_signal.py`, `ssl/ssl_convergence_test.py`, `ssl/downstream_convergence.py`, `ssl/diagnose_coord_shortcut.py`, `ssl/diagnose_end_to_end.py`, `ssl/compare_dino_backbones.py`, `ssl/downstream_compare.py`, `ssl/probe_features.py`, `ssl/test_dino.py`
- Write: corresponding files in `src/{training,analysis}/`

**Fix for `ssl/pretrain_multi.py`:** Replace `ROOT` global with computed path:
```python
# OLD (broken)
ROOT = ...  # undefined

# NEW
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]
```

**Acceptance:** Every migrated script imports successfully via `python -c "import benchmark_ssl.training.ssl_pretrainer"` etc.

#### T4: Migrate cnn_encoder/ scripts (Agent: cnn-encoder-migrator)

**Agent scope:** Move CNN encoder scripts, remove deprecated `cnn_probe.py`, extract `ScaledCNN` to `models/scaled_cnn.py`.

**Files to touch:**
- Read: `cnn_encoder/cnn_ssl.py` (extract ScaledCNN class), `cnn_encoder/cnn_convergence.py`, `cnn_encoder/h100_convergence.py`, `cnn_encoder/cross_dataset_eval.py`, `cnn_encoder/scale_sweep.py`, `cnn_encoder/visualize_embeddings.py`, `cnn_encoder/smoke_test_cnn_trackastra.py`
- Write: `src/cnn_encoder/cnn_ssl.py`, `src/cnn_encoder/cnn_convergence.py`, `src/cnn_encoder/h100_convergence.py`, `src/cnn_encoder/cross_dataset_eval.py`, `src/cnn_encoder/scale_sweep.py`, `src/cnn_encoder/visualize_embeddings.py`, `src/cnn_encoder/smoke_test.py`
- Write: `src/models/scaled_cnn.py` (extracted from `cnn_ssl.py`)
- Delete: `cnn_encoder/cnn_probe.py`

**Acceptance:** `python -c "from benchmark_ssl.cnn_encoder.cnn_ssl import ScaledCNN"` succeeds.

#### T5: Migrate probe/ scripts (Agent: probe-migrator)

**Agent scope:** Move probe scripts, remove deprecated `edge_probe.py`.

**Files to touch:**
- Read: `probe/unified_edge_probe.py`, `probe/train_cnn_probe.py`, `probe/verify_normalization.py`, `probe/feature_distribution_histograms.py`
- Write: `src/probing/unified_edge_probe.py`, `src/probing/train_cnn_probe.py`, `src/probing/verify_normalization.py`, `src/probing/feature_distribution_histograms.py`
- Delete: `probe/edge_probe.py`

**Acceptance:** `python -c "import benchmark_ssl.probing.unified_edge_probe"` succeeds.

#### T6: Migrate analysis/, exploratory/, mini_trackastra/ (Agent: ancillary-migrator)

**Agent scope:** Move remaining scripts.

**Files to touch:**
- Read: `analysis/plot_ssl.py`, `analysis/plot_expected_metrics.py`, `exploratory/test_local.py`, `mini_trackastra/*.py`
- Write: `src/analysis/plot_ssl.py`, `src/analysis/plot_expected_metrics.py`, `src/mini_trackastra/*.py`
- Decision: Keep `exploratory/test_local.py` as `scripts/run_exploratory_test_local.py` or delete (user decision)

**Acceptance:** All migrated files import successfully.

#### T7: Consolidate results (Agent: results-consolidator)

**Agent scope:** Move all experiment outputs to `results/`.

**File operations:**
```bash
mv runs/* results/runs/
mv probe/feature_cache/* results/feature_cache/
mv cnn_encoder/probe/cnn_*.pt results/checkpoints/cnn/
mv cnn_encoder/probe/cnn_*_compare results/checkpoints/cnn/compare/
# results/probes/ already exists with rp_*.json — keep
rmdir runs probe/feature_cache cnn_encoder/probe
```

**Acceptance:** `results/` contains `runs/`, `probes/`, `checkpoints/`, `feature_cache/`. No result directories at old locations.

#### T8: SLURM, configs, pyproject.toml (Agent: deploy-configurator)

**Agent scope:** Move all `.slurm` files, update path references, update `pyproject.toml`.

**Files to touch:**
- Move: `cnn_encoder/run_*.slurm` → `slurm/`
- Move: `probe/run_unified_probe.slurm` → `slurm/`
- Move: `config.yaml` → `configs/config.yaml`
- Update `pyproject.toml`:
  ```toml
  [tool.setuptools.packages.find]
  where = ["src"]

  [project.scripts]
  # Optional: add entry points for common scripts
  ```
- Update all path references in SLURM scripts:
  - `benchmark_ssl/cnn_encoder/...` → `benchmark_ssl/cnn_encoder/...` (same package path, but relative to repo root changes)
  - `probe/feature_cache/...` → `results/feature_cache/...`
  - `cnn_encoder/probe/cnn_ntxent_large.pt` → `results/checkpoints/cnn/cnn_ntxent_large.pt`
  - `configs/...` references in SLURM scripts

**Acceptance:** Every SLURM script's `python ...` or `srun python ...` line references files that exist at the new locations.

#### T9: Cleanup (Agent: cleanup)

**Agent scope:** Remove deprecated files, build artifacts, nested directories.

**File operations:**
```bash
# Remove deprecated code
rm ssl/pretrain_multi.py
rm cnn_encoder/cnn_probe.py
rm probe/edge_probe.py

# Remove nested package directory
rm -rf benchmark_ssl/benchmark_ssl/

# Remove root-level HTML artifacts
rm benchmark_ssl.html expected_metrics.html

# Remove all __pycache__ (will be regenerated)
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
```

**Acceptance:** None of the above exist after T9.

#### T10: Verification (Agent: verifier)

**Agent scope:** Run all acceptance checks from §4.

**Commands:**
```bash
# §4.1 Structure
python -c "import benchmark_ssl"
test ! -f benchmark_ssl/ssl_pipeline.py  # root-level Python files gone
test ! -d benchmark_ssl/__pycache__
test ! -d benchmark_ssl/.venv
test ! -d benchmark_ssl/benchmark_ssl/benchmark_ssl  # nested gone
test ! -f benchmark_ssl/benchmark_ssl.html

# §4.2 Code quality
grep -r "from ssl_pipeline" benchmark_ssl/src benchmark_ssl/scripts && echo "FAIL" || echo "PASS"
grep -r "from distortions" benchmark_ssl/src benchmark_ssl/scripts && echo "FAIL" || echo "PASS"
grep -r "from track_encoder" benchmark_ssl/src benchmark_ssl/scripts && echo "FAIL" || echo "PASS"
grep -rn "_border_dist_fast" benchmark_ssl/src | wc -l  # should be 1

# §4.3 Results
ls benchmark_ssl/results/{runs,probes,checkpoints,feature_cache}

# §4.4 Config
ls benchmark_ssl/slurm/*.slurm | wc -l  # should be 8
test -f benchmark_ssl/configs/config.yaml

# §4.6 Verification
for script in benchmark_ssl/scripts/run_*.py; do
  python "$script" --help > /dev/null 2>&1 && echo "OK: $script" || echo "FAIL: $script"
done
```

**Acceptance:** All checks pass.

## 7. Verification plan

After T10, the verifier agent must confirm every item in §4. If any check fails, the verifier reports the failure with file path and line number, and the corresponding task agent fixes it.

### 7.1 What we do NOT verify (out of scope)

- Re-running experiments (cluster-only, expensive)
- Numerical reproducibility of results (results/ is moved as-is, not regenerated)
- SLURM job submission (cluster access not available in the local sandbox)
- The `cnn_encoder/cnn_probe.py` 0.500 result is not re-derived; we trust the existing REPRODUCTION.md

### 7.2 What we DO verify

- Every Python file under `src/` can be imported without ImportError
- Every script under `scripts/` can be invoked with `--help`
- All old import paths (`from ssl_pipeline`, `from distortions`, etc.) are gone
- `_border_dist_fast` is defined exactly once
- All deprecated files are deleted
- All SLURM scripts reference existing paths in the new layout
- `pyproject.toml` is valid and installable (`pip install -e .` succeeds)

## 8. Risks & open questions

1. **`ssl/pretrain.py` and `ssl/downstream_compare.py` rewrites** — **RESOLVED**: Add CLI flags.
   - `pretrain.py`: `--loss {identity_bce,ntxent}` (default: `ntxent`)
   - `downstream_compare.py`: `--mode {finetune,embedding}` (default: `embedding`)

2. **`exploratory/test_local.py`** — **RESOLVED**: DELETE.

3. **CNN checkpoint paths in configs** — the YAML configs reference `cnn_checkpoint: <HPC path>`. After refactor, the path changes to `results/checkpoints/cnn/cnn_ntxent_large.pt` (relative) or stays as HPC-absolute. The HPC-absolute paths in `cross_dataset_eval.py` and SLURM scripts are cluster-specific and should not be changed.

4. **`mini_trackastra/` dependency on broken `trackastra` install** — `REPRODUCTION.md` notes the trackastra install is broken. This is pre-existing and out of scope for the refactor.

5. **Large file sizes** — `unified_edge_probe.py` (1882 lines) and `model_parts.py` (978 lines) are large. Splitting `unified_edge_probe.py` further is out of scope (would require understanding the CV protocol in depth). Splitting `model_parts.py` into per-attention-class files is tempting but adds import overhead; keeping it as `attention.py` is the chosen compromise.

## 9. Execution order

```
Week 1:
  Day 1: T1 (package skeleton) → T2 (shared modules)
  Day 2: T3 (ssl/) ∥ T4 (cnn_encoder/) ∥ T5 (probe/)  [parallel]
  Day 3: T6 (ancillary) ∥ T7 (results) ∥ T8 (SLURM/configs)  [parallel]
  Day 4: T9 (cleanup) → T10 (verification)
  Day 5: Fix any T10 failures
```

## 10. User decisions (all resolved 2026-08-12)

| Question | Decision |
|---|---|
| `ssl/pretrain.py` rewrite | Add `--loss {identity_bce,ntxent}` CLI flag (default: `ntxent`). Both variants preserved. |
| `ssl/downstream_compare.py` rewrite | Add `--mode {finetune,embedding}` CLI flag (default: `embedding`). Both variants preserved. |
| `exploratory/test_local.py` | DELETE. |

*No further user input required — T1 may begin.*

---

*Document version: 0.1 — DRAFT*
*Date: 2026-08-12*
