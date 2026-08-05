# REPRODUCTION.md — benchmark_ssl

Complete reproduction guide for every experiment in `benchmark_ssl/`, including all
resources (data, code, environments, hardware) needed to re-run them and the measured
results already obtained.

Authoritative source for statuses and verdicts:
`labbook/2026-08-03_experiment_readiness_inventory.md` (incl. the "Curation decision
(2026-08-04)"). Numbers quoted here are copied verbatim from the labbook entries and
from the result files on disk in `benchmark_ssl/runs/`.

---

## 1. Local (A500) vs Cluster (H100) split

| Scope | Hardware | Where the code lives | Environment |
|---|---|---|---|
| **Local SSL experiments** (`benchmark_ssl/runs/*`) | NVIDIA RTX A500 Laptop, 4096 MiB, compute cap 8.6 (Ampere) | this directory (`benchmark_ssl/`) | `benchmark_ssl/.venv` (Python 3.14.4, torch 2.12.0+cu130, lightly 1.5.25, scikit-learn 1.9.0) |
| **Full DINOv2 SSL pretraining** (`slurm/run_ssl_dino_pretrain.slurm`) | H100 (80 GB, TU Dresden Capella, SLURM partition `gpu-h100`, account `p_scads_celltracking`, 1 GPU per node) | trackastra checkout at `$TRK` | `$TRK/.venv` (built inside the slurm script) |

Local data: `data/vanvliet/` (3.1 GB; conditions `rpsM, recA, pheA, metA, cib, trpL`,
33 experiment dirs total — `rpsM` 8, `recA` 7, `pheA` 5, `metA` 5, `cib` 2, `trpL` 6).
Every local script resolves data relative to this repo root (default `../data/vanvliet`
from `benchmark_ssl/`, or `ROOT / "data/vanvliet"`).

All `runs/*` experiments were already executed on the A500 and are **DONE**. The full
DINOv2 SSL pretraining has **never completed** and is **NOT-NEEDED** per the curation
decision (see section 5).

---

## 2. Producing scripts → run directory map

| Producing script | Produced run directories |
|---|---|
| `ssl/pretrain.py` (identity-BCE variant of 2026-05-19) | `runs/ssl_v1` |
| `ssl/pretrain_multi.py` | `runs/ssl_dense`, `runs/ssl_K=4`, `runs/ssl_K=8`, `runs/ssl_K=16`, `runs/ssl_K=32` |
| `ssl/downstream_compare.py` (earlier fine-tuning variant) | `runs/downstream_compare` |
| `ssl/analyze_signal.py` | `runs/feature_signal`, `runs/feature_signal_full`, `runs/feature_signal_shape`, `runs/feature_signal_hu`, `runs/feature_signal_patch` |
| `ssl/ssl_convergence_test.py` | `runs/convergence_prediction`, `runs/generalization_test`, `runs/hard_test` |
| `ssl/downstream_convergence.py` | `runs/downstream_convergence` |
| `ssl/compare_dino_backbones.py` | `runs/dino_comparison` |
| `ssl/diagnose_coord_shortcut.py` | `runs/diagnose_coord_shortcut`, `runs/diagnose_coord_shortcut_jitter4` |
| `ssl/diagnose_end_to_end.py` | `runs/diagnose_end_to_end` |

Run all local scripts with `benchmark_ssl/.venv/bin/python` (from the `benchmark_ssl/`
directory unless stated otherwise).

---

## 3. Prerequisites

### 3.1 Data
- `data/vanvliet/` at repo root (complete, 3.1 GB). Structure per condition:
  `data/vanvliet/<condition>/<experiment>/img/tNNNNNN.tif` + `TRA/man_trackNNNNNN.tif` + `man_track.txt`.
- All 6 conditions used: `rpsM, recA, pheA, metA, cib, trpL`.

### 3.2 Local environment (`benchmark_ssl/.venv`)
Key packages (verified via `pip list`):
```
Python     3.14.4
torch      2.12.0  (+cu130, CUDA 13.0, torchvision 0.27.0)
lightly    1.5.25
scikit-learn 1.9.0
numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, scikit-image 0.26.0, tifffile 2026.6.1
matplotlib 3.11.0, seaborn 0.13.2, PyYAML 6.0.3, tqdm 4.68.2, dask 2026.6.0, edt 3.1.1
fast-regionprops 0.2.0, kornia 0.8.3
```
Note: the editable trackastra install in this venv is stale/broken
(`from trackastra.model import Trackastra` fails). None of the `runs/*` experiments
import trackastra — they use only the self-contained modules in `benchmark_ssl/`
(`ssl_pipeline.py`, `distortions.py`, `track_encoder.py`, `model_parts.py`,
`rich_features.py`).

`uv sync` (per `pyproject.toml`, deps: torch>=2.0, numpy, scipy, scikit-image, tifffile,
matplotlib, seaborn, pandas, pyyaml, scikit-learn, tqdm) recreates the environment;
the runs also need `lightly` and `edt` for some producers.

---

## 4. Local (A500) experiments — all DONE

### 4.1 `runs/ssl_v1` — identity-BCE mini SSL

- **Producing script:** `ssl/pretrain.py` (identity-BCE variant that produced the run on
  2026-05-19; the currently checked-in `ssl/pretrain.py` is the NT-Xent rewrite and does
  NOT write the on-disk `training_log.csv`). The labbook inventory classifies this run
  as "ssl_v1 (identity-BCE mini)".
- **Exact CLI:**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  $V ssl/pretrain.py runs/ssl_v1/config.yaml
  ```
  The run writes a copy of the active config to `runs/ssl_v1/config.yaml` — that file
  is the authoritative record of the exact hyperparameters.
- **Key hyperparameters** (verbatim from `runs/ssl_v1/config.yaml`): `name: ssl_v1`,
  `data_root: ../data/vanvliet`, all 6 conditions, distortions
  `[affine, elastic, jitter, dropout, photometric]` (no feature_noise),
  `encoder: d_model 128, nhead 4, num_layers 4, dim_feedforward 256, dropout 0.1`,
  `training: batch_size 4, lr 3e-4, weight_decay 0.01, epochs 5, val_split 0.1,
  checkpoint_every 10, max_tokens 2048`.
- **Data inputs:** `data/vanvliet` (all 6 conditions), 7D regionprops2 features.
- **Output artifacts:** `runs/ssl_v1/best_model.pt` (5.07 MB),
  `runs/ssl_v1/training_log.csv`, `runs/ssl_v1/config.yaml`,
  `runs/ssl_v1/events.out.tfevents.1779185392.MI-DD-CN24021L.213240.0`
  (host `MI-DD-CN24021L` = the A500 laptop; the TensorBoard event confirms the
  `ssl/pretrain.py`-family trainer).
- **Status:** DONE (A500, 2026-05-19). Analyzed in `results_analysis.md` sections 1–5.
- **Key result:** train loss 0.4967 → 0.1549, val loss 0.3706 → 0.1678, train acc
  0.8944 → 0.9699, val acc 0.9111 → 0.9678 over 5 epochs; no overfitting.

### 4.2 `runs/ssl_dense` + `runs/ssl_K=4/8/16/32` — attention K-sweep SSL

- **Producing script:** `ssl/pretrain_multi.py` (BCE association head, pos_weight 10.0,
  dense vs `GatherSparseAttention` with KNN K).
- **Exact CLI:**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd <repo root>   # required: pretrain_multi.py resolves ROOT/data/vanvliet and ROOT/benchmark_ssl/runs
  $V benchmark_ssl/ssl/pretrain_multi.py
  ```
  The `__main__` block loops `for k in [None, 4, 8, 16, 32]`, i.e. it produces
  `ssl_dense` (K=None) and `ssl_K=4/8/16/32` in a single invocation.
- **Key hyperparameters** (in code): conditions `[rpsM, recA, pheA]`, 90/10
  train/val split (seed 42), distortion pipeline from `benchmark_ssl/config.yaml`
  (all 6 families), `SSLDataset` ndim 2, batch_size 4, `AdamW(lr=3e-4,
  weight_decay=0.01)`, `pos_weight=10.0` BCE, 5 epochs, `SSLModel` d_model 128,
  nhead 4, num_layers 4, dim_feedforward 256.
- **Data inputs:** `data/vanvliet` (rpsM, recA, pheA).
- **Output artifacts:** `runs/ssl_dense/best_model.pt` (3.46 MB),
  `runs/ssl_K=4/best_model.pt`, `runs/ssl_K=8/best_model.pt`,
  `runs/ssl_K=16/best_model.pt`, `runs/ssl_K=32/best_model.pt` (each ~3.47 MB).
  Only `best_model.pt` is persisted (the per-epoch numbers went to stdout/stderr logs,
  which are not archived).
- **Status:** DONE (A500, 2026-05-19).
- **Reproducibility caveat:** the checked-in `ssl/pretrain_multi.py` references an
  undefined `ROOT` global (`ROOT / "data/vanvliet"`, `ROOT / "benchmark_ssl" / "runs"`)
  — it will raise `NameError` unless `ROOT` is injected (repo root). The run was made
  with a working version of this file.

### 4.3 `runs/downstream_compare` — SSL-pretrained vs random-init fine-tuning

- **Producing script:** `ssl/downstream_compare.py` (earlier variant that fine-tuned
  SSL-pretrained vs random-init `CellEmbedder` on real adjacent-frame pairs). The
  currently checked-in `ssl/downstream_compare.py` is a rewrite (embedding + Hungarian
  tracking, writes different CSV columns) — see section 10.
- **Exact CLI (at the time):**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  $V ssl/downstream_compare.py runs/ssl_v1/best_model.pt
  ```
- **Data inputs:** `data/vanvliet` (all 6 conditions from `config.yaml`), consecutive
  frame pairs within each experiment, 90/10 pair split (seed 42), 15 epochs.
- **Output artifact:** `runs/downstream_compare/comparison.csv`
  (columns `epoch,model,train_loss,train_acc,val_loss,val_acc`).
- **Status:** DONE (A500, 2026-05-19). Analyzed in `results_analysis.md` section 7.
- **Key result:** final train loss SSL 0.0277 vs random 0.0325; final val loss SSL
  0.0244 vs random 0.0329 (−26%); val accuracy 99.8% for both; SSL reaches
  val_loss < 0.03 by epoch 3 vs epoch 9+ for random (~3× faster to low-loss regime).

### 4.4 `runs/feature_signal*` — data-level signal analysis

- **Producing script:** `ssl/analyze_signal.py` (no model; measures whether distorted
  features carry enough same-cell/different-cell separation for contrastive SSL).
- **Exact CLI (one per run; only `--feature-set` and `--outdir` differ):**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  $V ssl/analyze_signal.py --conditions rpsM,recA,pheA,metA,cib,trpL --max-frames 500 \
      --feature-set basic --outdir runs/feature_signal
  $V ssl/analyze_signal.py --feature-set basic --outdir runs/feature_signal_full
  $V ssl/analyze_signal.py --feature-set shape --outdir runs/feature_signal_shape
  $V ssl/analyze_signal.py --feature-set hu    --outdir runs/feature_signal_hu
  $V ssl/analyze_signal.py --feature-set patch --outdir runs/feature_signal_patch
  ```
  (README reference command: `uv run python3 ssl/analyze_signal.py --conditions
  rpsM,recA,pheA,metA,cib,trpL --max-frames 500`.) The exact flag values used on
  2026-06-15 are not persisted; the feature set per run is confirmed by the CSV
  columns (`basic` 7D, `shape` +shape descriptors, `hu` +Hu moments, `patch` +36 PCA
  patch features). `--interpret` re-prints the verdict from a saved CSV.
- **Data inputs:** `data/vanvliet`, `config.yaml` distortion pipeline, per-distortion
  breakdown over the 6 families.
- **Output artifacts:** `feature_signal_results.csv` + `feature_signal_report.html`
  in each `runs/feature_signal*` dir.
- **Status:** DONE (A500, 2026-06-15).
- **Key result:** all feature sets give separation gap ≈ 0.045–0.047 and effective
  dim 3 (FAIL verdict, gap < 0.1); recall@1 0.213–0.215. Adding shape/Hu/patch
  features did NOT improve the gap (verbatim labbook: "All gave gap ≈ 0.047,
  effective dim = 3").

### 4.5 `runs/convergence_prediction`, `runs/generalization_test`, `runs/hard_test` — convergence predictor

- **Producing script:** `ssl/ssl_convergence_test.py` (micro-SSL + generalization +
  hard-pipeline tests to predict full-scale SSL convergence before HPC submission).
- **Exact CLI:**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  # full battery (all 6 tests)
  $V ssl/ssl_convergence_test.py --max-frames 200 --outdir runs/convergence_prediction
  # generalization-only run (wrote into its own dir)
  $V ssl/ssl_convergence_test.py --test generalization --outdir runs/generalization_test
  # hard-pipeline-only run
  $V ssl/ssl_convergence_test.py --test hard --outdir runs/hard_test
  ```
  Defaults: `--data-root ../data/vanvliet`, `--conditions
  rpsM,recA,pheA,metA,cib,trpL`, `--max-frames 200`, `--test all`, `--outdir
  runs/convergence_prediction`.
- **Data inputs:** `data/vanvliet` (all 6 conditions); DINOv2 `vits14` via
  `torch.hub` (cached at `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`).
- **Output artifacts:**
  - `runs/convergence_prediction/`: `convergence_prediction.html`,
    `gap_vs_strength.csv`, `micro_ssl.csv`, `distortion_ranking.csv`,
    `pca_analysis.csv`
  - `runs/generalization_test/`: `micro_ssl_generalization.csv`,
    `convergence_prediction.html`
  - `runs/hard_test/`: `micro_ssl_hard.csv`, `convergence_prediction.html`
- **Status:** DONE (A500, 2026-06-15).
- **Key results:** baseline DINO gap 0.292 (distortion none); micro-SSL val gap
  0.561 → 0.698 in 200 steps; generalization val gap 0.481 → 0.577 (+0.095, genuine
  learning); hard pipeline gap 0.040 → 0.359 (+0.319) with final inter_sim 0.044
  (no collapse). See ledger in section 9.

### 4.6 `runs/downstream_convergence` — downstream convergence test

- **Producing script:** `ssl/downstream_convergence.py` (micro-SSL pretrain a projection
  head, then fine-tune SSL-pretrained vs random-init on real adjacent-frame pairs).
- **Exact CLI:**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  $V ssl/downstream_convergence.py --max-pairs 60 --epochs 20
  ```
  Defaults: `--data-root ../data/vanvliet`, `--conditions
  rpsM,recA,pheA,metA,cib,trpL`, `--max-pairs 100`, `--epochs 30`, `--outdir
  runs/downstream_convergence`. The on-disk CSV covers epochs 0–19, i.e. the run
  used `--epochs 20`.
- **Data inputs:** `data/vanvliet`, adjacent frame pairs, 5 masks per condition for
  micro-SSL pretraining (200 steps, 30 frames), 80/20 pair split.
- **Output artifacts:** `runs/downstream_convergence/convergence.csv`,
  `convergence.html`.
- **Status:** DONE (A500, 2026-06-15).
- **Key result:** SSL-pretrained vs Random-init downstream val_acc converges to the
  same ~0.49–0.50 level over 20 epochs; val_loss tracks each other (0.494 vs 0.498 at
  epoch 19). See section 9.

### 4.7 `runs/dino_comparison` — DINO backbone comparison

- **Producing script:** `ssl/compare_dino_backbones.py` (DINOv2 vits14 vs vitb14 feature
  separation on vanvliet bacteria).
- **Exact CLI:**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  $V ssl/compare_dino_backbones.py --data-root ../data/vanvliet --outdir runs/dino_comparison
  ```
  Defaults: `--conditions rpsM recA pheA metA cib trpL`, `--max-frames 40`,
  `--jitter-std 4.0`, `--seed 42`.
- **Data inputs:** `data/vanvliet` (40 frames, 904 cells per verdict.txt), DINOv2
  checkpoints from `torch.hub` (vits14 cached, vitb14 downloaded during the run).
- **Output artifacts:** `runs/dino_comparison/comparison.csv`, `comparison.png`,
  `verdict.txt`.
- **Status:** DONE (A500, 2026-06-23).
- **Key result:** v2-S gap 0.2435 vs v2-B gap 0.2649 (1.09×), recall@1 0.9715 vs
  0.9714, effective rank 100 vs 195, 115.9 vs 25.3 cells/s, 263 vs 657 MiB peak VRAM.
  Verdict: "Feature quality bottleneck is domain mismatch, not model capacity."
  (Cell-DINO and DINOv3 backbones inaccessible: 403 / `pretrained_url=None`.)

### 4.8 `runs/diagnose_coord_shortcut` + `runs/diagnose_coord_shortcut_jitter4` — coordinate shortcut diagnostic

- **Producing script:** `ssl/diagnose_coord_shortcut.py` (three modes: A PE+coords+DINO,
  B PE+noise+DINO, C PE+coords ONLY — is NT-Xent solvable from position alone?).
- **Exact CLI:**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  # full distortion pipeline (defaults)
  $V ssl/diagnose_coord_shortcut.py
  # jitter4-only distortion, separate outdir
  $V ssl/diagnose_coord_shortcut.py --distortion jitter4 --outdir runs/diagnose_coord_shortcut_jitter4
  ```
  Defaults: `--data-root ../data/vanvliet`, `--conditions rpsM`, `--max-frames 30`,
  `--steps 300`, `--lr 1e-3`, `--d-model 256`, `--pos-per-dim 32`, `--distortion
  full`, `--eval-every 25`, `--outdir runs/diagnose_coord_shortcut`. Both on-disk
  runs used the defaults (frames=30, d_model=256, pe_dim=128).
- **Data inputs:** `data/vanvliet` (rpsM), DINOv2 vits14.
- **Output artifacts:** `mode_A.csv`, `mode_B.csv`, `mode_C.csv`, `summary.csv`,
  `summary.txt`, `coordinate_shortcut.png` in each run dir.
- **Status:** DONE (A500, 2026-06-23).
- **Key result:** SMOKING GUN — Mode C (PE only, zero visual input) solves NT-Xent
  in step 2 (full) / step 3 (jitter4). "COORDINATE SHORTCUT IS REAL AND DOMINANT."
  Removing coordinates (Mode B vs A) improves final val gap by +0.016 (full) and
  +0.228 (jitter4).

### 4.9 `runs/diagnose_end_to_end` — end-to-end SSL → downstream transfer

- **Producing script:** `ssl/diagnose_end_to_end.py` (micro-SSL then downstream
  fine-tuning on real pairs, modes R/A/B/C).
- **Exact CLI:**
  ```bash
  V=benchmark_ssl/.venv/bin/python
  cd benchmark_ssl
  $V ssl/diagnose_end_to_end.py
  ```
  Defaults: `--data-root ../data/vanvliet`, `--conditions rpsM`, `--max-frames 25`,
  `--max-pairs 20`, `--ssl-steps 200`, `--ssl-lr 1e-3`, `--downstream-steps 100`,
  `--downstream-lr 1e-3`, `--d-model 256`, `--pos-per-dim 32`, `--distortion
  jitter4`, `--outdir runs/diagnose_end_to_end`.
- **Data inputs:** `data/vanvliet` (rpsM), DINOv2 vits14.
- **Output artifacts:** `downstream_A.csv`, `downstream_B.csv`, `downstream_C.csv`,
  `downstream_R.csv`, `end_to_end.png`, `verdict.txt`.
- **Status:** DONE (A500, 2026-06-23).
- **Key result:** Random init and PE-only never reach val acc > 0.8; PE+coords+DINO
  and PE+noise+DINO reach 0.969 (10 steps). Verdict: "SSL does not help downstream
  convergence at this scale. The coordinate shortcut exists but fixing it alone is
  insufficient. The decoder gap or NT-Xent/BCE mismatch may dominate."

---

## 5. Cluster (H100) experiment: `slurm/run_ssl_dino_pretrain.slurm`

Full DINOv2-feature SSL pretraining (24 h budget). **Code-ready, never completed,
status NOT-NEEDED** (curation decision 2026-08-04: micro-tests gap 0.292→0.040 under
distortions + the edge-probe ceiling already support the report's architectural
conclusion; hedged in the report).

### 5.1 SBATCH resources (verbatim from the script)

```
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=ssl_dino_pt
#SBATCH --output=logs/slurm-ssl_dino-%j.out
#SBATCH --error=logs/slurm-ssl_dino-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
```
SLURM partition `gpu-h100`, 1 GPU per node (cluster default; no `--partition` line
in the script).

### 5.2 Environment setup performed inside the script (Normal mode)
```bash
export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/.venv"
CFG_DIR="$SLURM_SUBMIT_DIR/configs"
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib scikit-learn
cd "$TRK"
git pull origin cached-dist-attn
python -m pip install -e .
mkdir -p logs
```

### 5.3 Training command (verbatim)
```bash
python -m trackastra.model.ssl_dino_trainer \
    --config "$CFG_DIR/ssl_dino_pretrain.yaml" \
    --ssl_epochs 20 \
    --ssl_batch_size 4 \
    --ssl_lr 3e-4 \
    --ssl_temperature 0.05 \
    --outdir runs/ssl_dino_pretrain
```

### 5.4 Config file — `configs/ssl_dino_pretrain.yaml`
- `name: ssl_dino_pretrain`
- `data_root: /data/cat/ws/mawe985g-data/data/celltracking/vanvliet` (HPC path;
  must be rewritten for local runs — see section 6)
- `conditions: [rpsM, recA, pheA, metA, cib, trpL]`, `ndim: 2`
- Model: `d_model 320, nhead 4, num_encoder_layers 6, num_decoder_layers 6,
  dropout 0.01, window 10, pos_embed_per_dim 32, feat_embed_per_dim 8,
  knn_neighbors 16, use_dino true`
- Distortions: `[affine, elastic, jitter, dropout, photometric, feature_noise]`
  (with the same parameter blocks as the local `config.yaml`)
- SSL: `epochs 20, lr 3e-4, temperature 0.05, batch_size 4, val_split 0.1`
- `seed: 42`

### 5.5 Data paths
- Input: `$DATA_DIR/vanvliet` (= `/data/cat/ws/mawe985g-data/data/celltracking/vanvliet`).
- Trainer code: `$TRK/trackastra/model/ssl_dino_trainer.py` + `dino_encoder.py`
  (both present in the repo). DINOv2 vits14 weights cached by torch.hub.

### 5.6 Outputs
- Checkpoint → `runs/ssl_dino_pretrain` (relative to `$TRK`), i.e.
  `$TRK/runs/ssl_dino_pretrain/` (no checkpoint has ever been produced).
- SLURM logs → `logs/slurm-ssl_dino-%j.{out,err}` relative to the submission dir.
- **No full DINO-SSL checkpoint exists anywhere** (labbook inventory §2.5).

### 5.7 Test mode
Setting `TEST=1` (env var) runs only the environment smoke test (pip installs,
`git pull origin cached-dist-attn`, `pip install -e .`, imports trackastra + CUDA +
`DINOBackbone.load()` + torch.hub DINOv2) and exits 0.

### 5.8 Submission
```bash
sbatch slurm/run_ssl_dino_pretrain.slurm
```

### 5.9 Status
**NOT-NEEDED** (curation decision 2026-08-04). Do not resubmit for the current
report. The A500 micro-tests (`ssl/ssl_convergence_test.py`, `ssl/test_dino.py`) and the
edge-probe ceiling (DINOv2 0.896 vs regionprops 0.860 balanced accuracy) already
support the report's conclusion.

---

## 6. Cluster requirements & path constants

Shared, verified against the scripts:

| Constant | Value |
|---|---|
| `TRK` | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra` (trackastra checkout + venv + runs) |
| `BENCH` | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking` (bench repo on cluster) |
| `ENV_DIR` | `$TRK/.venv` |
| `DATA_DIR` | `/data/cat/ws/mawe985g-data/data/celltracking` (vanvliet subdir: `.../celltracking/vanvliet`) |
| Partition / account | `gpu-h100` / `p_scads_celltracking`, 1 GPU per node |
| SSH host | `capella` (VPN required); local machine cannot reach `/data/cat/ws/...` directly |

Local equivalents:
- Local venv for the SSL scripts: `benchmark_ssl/.venv`
  (`/home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj/benchmark_ssl/.venv`).
- Local data: `data/vanvliet` (repo root).
- DINOv2 vits14 weights cached locally at
  `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` (88 MB).

---

## 7. Experiment status summary

| Experiment | Status | Resources | Outputs |
|---|---|---|---|
| `runs/ssl_v1` (identity-BCE mini SSL) | DONE (A500) | 1× RTX A500, `benchmark_ssl/.venv`, `data/vanvliet` | `best_model.pt`, `training_log.csv`, `config.yaml`, tfevents |
| `runs/ssl_dense`, `runs/ssl_K=4/8/16/32` (attention K-sweep SSL) | DONE (A500) | 1× RTX A500, `benchmark_ssl/.venv`, `data/vanvliet` (rpsM/recA/pheA) | `best_model.pt` per K |
| `runs/downstream_compare` (SSL vs random fine-tune) | DONE (A500) | 1× RTX A500, `benchmark_ssl/.venv`, `data/vanvliet` | `comparison.csv` |
| `runs/feature_signal*` (5 feature sets) | DONE (A500) | CPU+GPU negligible, `benchmark_ssl/.venv`, `data/vanvliet` | `feature_signal_results.csv` + `.html` per set |
| `runs/convergence_prediction` (+ generalization/hard sub-tests) | DONE (A500) | 1× RTX A500, DINOv2 vits14 cached, `data/vanvliet` | `.html` + 4 CSVs (+ `micro_ssl_generalization.csv`, `micro_ssl_hard.csv`) |
| `runs/downstream_convergence` | DONE (A500) | 1× RTX A500, DINOv2 vits14, `data/vanvliet` | `convergence.csv`, `convergence.html` |
| `runs/dino_comparison` | DONE (A500) | 1× RTX A500, DINOv2 vits14 + vitb14, `data/vanvliet` | `comparison.csv`, `comparison.png`, `verdict.txt` |
| `runs/diagnose_coord_shortcut` (+ `_jitter4`) | DONE (A500) | 1× RTX A500, DINOv2 vits14, `data/vanvliet` (rpsM) | `mode_{A,B,C}.csv`, `summary.csv`, `summary.txt`, PNG |
| `runs/diagnose_end_to_end` | DONE (A500) | 1× RTX A500, DINOv2 vits14, `data/vanvliet` (rpsM) | `downstream_{R,A,B,C}.csv`, `end_to_end.png`, `verdict.txt` |
| Unified edge probe (5-fold CV) | DONE (A500, 2026-08-04) | 1× RTX A500, DINOv2 + CNN checkpoints, `data/vanvliet` | `results/unified_probe_results_cv.json` |
| DINOv2 full SSL pretraining (`slurm/run_ssl_dino_pretrain.slurm`, 24 h) | **NOT-NEEDED** | 1× H100, 90 GB, 8 CPU, 24 h, `$TRK/.venv` | `$TRK/runs/ssl_dino_pretrain/` (never produced) |
| Low-label regime sweep (12 h, H100) | **NOT-NEEDED** | H100 | — (SSL already showed zero improvement at 10% labels) |
| `ssl_d2d` (diagnostic) | **NOT-NEEDED** | — | — (earlier exploratory; not required) |
| Multi-seed K-sweep (42/43/44 × 5 K, 15 jobs) | PENDING (cluster, NECESSARY) | 15× H100, 48 h each (`benchmark_combined`) | `results/knn_sweep/` variance estimates, K=4/K=64 checkpoints |
| DeepCell cross-dataset eval incl. KNN checkpoints | PENDING (cluster, NECESSARY) | 1 GPU, data + CHOTAMetric on cluster | `$TRK/results/cross_dataset/deepcell/...` |

Not needed (curation decision): `phase1_*`, `dist_ablation`, `diag2`, `quick_bench`,
`ssl_only`, `ssl_d2d` — diagnostic/earlier exploratory, not required to support the
report's claims.

---

## 8. Early-stopping guidance (all training runs)

From the curation decision (2026-08-04):

> **Early-stopping guidance for all long runs:** never run to 500 epochs if val_loss
> has plateaued — patience 83 already stops near-optimal. `--epochs 500` is only an
> upper bound.

Applies to any cluster training (multi-seed K-sweep via `run_single_config.slurm`,
and the DINOv2 SSL trainer had it been run). The A500 micro-SSL runs already
demonstrated convergence within 200 steps, so no A500 training run needs full epochs.

---

## 9. Measured results ledger (DONE experiments, verbatim)

### 9.1 NT-Xent contrastive SSL collapse (H100 phase1) — `labbook/2026-05-24_ssl_collapse_findings.md`
- NT-Xent train loss: 3.46 → 2.97 (plateau after epoch 10); val loss 3.29 → 2.82.
- Cosine consistency: epoch 1 = 0.974, epoch 50 = 0.919.
- Global inter-cell cosine similarity (collapse): **0.89**.
- Downstream tracking: knn_SSL frozen **65.1%** vs knn random **64.8% avg** vs dense
  random 64.3% avg — "SSL provides zero improvement over random embeddings".
- Fixes tested locally (2026-05-24): stronger augmentations → collapse WORSE
  (inter-sim 0.90 → 0.96); remove coordinate encoding → crash; smaller model
  (d=32, L=2) → same collapse ~0.90; 32×32 patches + shallow CNN → loss flat at 3.9,
  inter-sim 0.99.
- Synthetic verification (16-dim unique signatures): loss 1.41 → 0.03 in 20 epochs,
  inter-cell sim 0.68 → 0.17, generalization pos_sim 0.80 vs neg_sim 0.13
  (gap = 0.67) — architecture is correct, failure is feature-quality.

### 9.2 Identity BCE SSL pretraining — `labbook/2026-06-01_identity_bce_negative_result.md`
- Setup: identity BCE on distorted single frames (Trackastra full model, 20 epochs),
  fine-tune on 10% labels (100 epochs), baseline from scratch on 10% labels.
- val_loss_epoch (final): baseline **0.416** vs SSL-finetune **0.404**.
- val_loss_step (final): 0.115 vs 0.111.
- "Nearly identical convergence curves. SSL provides zero benefit over random
  initialization."

### 9.3 DINOv2 feature micro-tests — `labbook/2026-06-15_dino_contrastive_ssl.md`
- Test A (separation, no distortion): DINO gap **0.292** vs regionprops 0.047
  (6.2× improvement); effective dim 108 vs 3 (36×).
- Test B (micro-SSL generalization, 20 train / 10 held-out frames): val gap
  0.481 → 0.577 (+0.095) — "genuine learning, not memorization".
- Test C (hard pipeline affine+jitter+dropout): DINO collapses to gap 0.040; after
  200 micro-SSL steps gap 0.359 (+0.319); inter-cell similarity 0.044 (no collapse).
- Test D (gap vs distortion): none 0.292; jitter 4 px 0.217; jitter 8 px 0.214;
  jitter 16 px 0.072; rotation 10° 0.137; rotation 25° 0.031.

### 9.4 Feature-signal analysis (`ssl/analyze_signal.py`, on-disk CSVs, 2026-06-15)
- `feature_signal` (basic 7D): all-condition separation gap 0.0452, recall@1 0.2131,
  effective dim 3 (FAIL, gap < 0.1).
- `feature_signal_full`: gap 0.0467, recall@1 0.2151, effective dim 3.
- `feature_signal_shape`: gap 0.0469; `feature_signal_hu`: gap 0.0469;
  `feature_signal_patch`: gap 0.0455 — richer features do not raise the gap
  (labbook: "All gave gap ≈ 0.047, effective dim = 3").

### 9.5 Convergence predictor (`ssl/ssl_convergence_test.py`, on-disk CSVs, 2026-06-15)
- `gap_vs_strength.csv`: none 0.2916; jitter 4 0.2417; jitter 8 0.2144; jitter 16
  0.1440; rotation 10 0.0344; rotation 25 0.0022.
- `micro_ssl.csv`: val_gap 0.561 → 0.698 in 200 steps (train_loss 0.130 → 3.4e-6).
- `micro_ssl_generalization.csv`: val_gap 0.481 → 0.577 (+0.096).
- `micro_ssl_hard.csv`: val_gap 0.040 → 0.359 (+0.319), final inter_sim 0.044.
- `distortion_ranking.csv`: none gap 0.3009 / recall1 1.0; jitter_8px 0.1836 / 0.83;
  rotation_30 0.0049 / 0.24; affine_strong 0.0007 / 0.18.
- `pca_analysis.csv`: PC1 22.8%, PC1–20 cumulative 75.1%.

### 9.6 Downstream convergence (`ssl/downstream_convergence.py`, on-disk CSV, 2026-06-15)
- SSL-pretrained vs Random-init over 20 epochs (0–19) on real adjacent-frame pairs:
  final (epoch 19) val_acc 0.504 vs 0.498, val_loss 1.460 vs 1.471 — no downstream
  advantage at this scale (consistent with `diagnose_end_to_end` verdict).

### 9.7 Coordinate shortcut (`ssl/diagnose_coord_shortcut.py`, on-disk summaries, 2026-06-23)
- `full`: Mode C (PE only) converges at step 2 (final train gap 0.849); Mode A
  (PE+coords+DINO) conv step 3; Mode B (PE+noise+DINO) conv step 42. "SMOKING GUN:
  COORDINATE SHORTCUT EXISTS ... The model solves NT-Xent using POSITION ALONE."
- `jitter4`: Mode C conv step 3 (final gap 0.836); Mode B achieves +0.228 better
  final val gap than Mode A.

### 9.8 End-to-end transfer (`ssl/diagnose_end_to_end.py`, on-disk verdict, 2026-06-23)
- Random init val acc 0.750 → 0.750 (never reaches 0.8; val loss 0.5225).
- PE+coords+DINO: 0.750 → 0.969, steps to acc>0.8 = 10, val loss 0.1084.
- PE+noise+DINO: 0.719 → 0.969, steps = 10, val loss 0.0752.
- PE only: 0.531 → 0.750, never, val loss 0.8740.
- Verdict: "SSL does not help downstream convergence at this scale. The coordinate
  shortcut exists but fixing it alone is insufficient."

### 9.9 DINO backbone comparison (`ssl/compare_dino_backbones.py`, on-disk CSV, 2026-06-23)
| backbone | dim | params (M) | intra | inter | gap | recall@1 | eff. rank | cells/s | VRAM (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2-vits14 | 384 | 22.1 | 0.9465 | 0.703 | 0.2435 | 0.9715 | 100 | 115.9 | 263 |
| DINOv2-vitb14 | 768 | 86.6 | 0.9341 | 0.6692 | 0.2649 | 0.9714 | 195 | 25.3 | 657 |
- Gap ratio (B/S) 1.09×, recall ratio 1.00×, VRAM ratio 2.50×, speed ratio 0.22×.
- Verdict: "Feature quality bottleneck is domain mismatch, not model capacity."

### 9.10 Unified edge probe (5-fold CV) — `results/unified_probe_results_cv.json` + labbook §5.1 (2026-08-04)
| Feature (MLP probe) | balanced acc (mean ± std) | F1 |
|---|---|---|
| DINOv2 (frozen) | 0.8958 ± 0.0362 | 0.566 ± 0.090 |
| HOCT 19D (2D → 12D) | 0.8799 ± 0.0201 | 0.423 ± 0.076 |
| Regionprops 7D | 0.8595 ± 0.0137 | 0.396 ± 0.056 |
| CNN NT-Xent (frozen) | 0.7031 ± 0.0554 | 0.289 ± 0.126 |
| CNN end-to-end | 0.5000 ± 0.0000 | 0.000 ± 0.000 |
- Linear probes: 0.50–0.57 (report cites only MLP). Shuffle baseline 0.50–0.55 (no
  label leakage). Runtime 50 min 53 s, GPU peak ~3.35 GiB (labbook §5.1).
- Reproducing command (labbook §2.1):
  ```bash
  V=benchmark_ssl/.venv/bin/python
  $V benchmark_ssl/probe/unified_edge_probe.py --features all --probe both --epochs 200 \
      --cv-folds 5 --shuffle-baseline --data-root data/vanvliet \
      --output results/unified_probe_results_cv.json
  ```

---

## 10. Discrepancies & reproducibility notes

1. **`ssl/pretrain_multi.py` references an undefined `ROOT`** (lines 155, 162, 178). The
   checked-in file raises `NameError`; the 2026-05-19 run used a working version.
   Inject `ROOT = <repo root>` (the directory containing `benchmark_ssl/` and `data/`)
   to re-run.
2. **`ssl/downstream_compare.py` is a rewrite.** The on-disk `runs/downstream_compare/
   comparison.csv` (columns `epoch,model,train_loss,train_acc,val_loss,val_acc`, a
   15-epoch fine-tuning comparison) was produced by the earlier version. The current
   file performs embedding + Hungarian tracking and writes different columns.
3. **`ssl/pretrain.py` is a rewrite.** `runs/ssl_v1/training_log.csv` (columns including
   `train_acc, train_f1, val_prec, val_rec`) and the TensorBoard event were produced
   by the 2026-05-19 identity-BCE variant; the current `ssl/pretrain.py` is NT-Xent and
   writes neither `training_log.csv` nor acc/F1 metrics. `results_analysis.md` states
   `d_model=64, nhead=2` but the archived `runs/ssl_v1/config.yaml` says `d_model=128,
   nhead=4` — treat the archived config as authoritative.
4. **No cluster K-sweep / CNN / DeepCell artifacts are reproducible locally** without
   pulling the clean run dirs off Capella (`$TRK/runs/2026-06-01_23-04-*_clean/`).
   Those experiments live outside `benchmark_ssl/` and are out of scope here.
5. **DINOv2 full SSL was never completed** — no checkpoint exists. Documented for
   completeness; status NOT-NEEDED.
6. **Feature-set flags for the `runs/feature_signal*` runs are not persisted** in the
   output files; the CSV columns confirm which `--feature-set` produced each run.
7. The `runs/generalization_test` and `runs/hard_test` outputs were written by
   separate `ssl/ssl_convergence_test.py` invocations (`--test generalization`,
   `--test hard`), not by the `--test all` run that populated
   `runs/convergence_prediction`.
