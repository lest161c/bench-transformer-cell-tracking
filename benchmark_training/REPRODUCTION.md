# REPRODUCTION.md — benchmark_training (cluster training experiments)

Reproduction guide for every SLURM experiment script physically present in this
directory (`benchmark_training/slurm/`). Each entry gives the exact SBATCH resources,
the verbatim command(s) from the script body, the required configs and input
data, the outputs produced, and how to launch the job.

## Scope and status conventions

- This document covers **only** the 15 `.slurm` scripts that exist in this
  directory. The clean multi-seed K-sweep + DeepCell evaluation harness lives in
  the **mirror repo** on the cluster
  (`bench-transformer-cell-tracking/benchmark_combined/` — `run_single_config.slurm`,
  `vanvliet_{baseline,sparse_k4,sparse_k16,sparse_k32,sparse_k64}_clean.yaml`)
  and is documented in a separate REPRODUCTION.md there. Do not look for those
  files here.
- Status legend: **DONE** = the experiment was run historically and its evidence
  is recorded (see "Results ledger"); **PENDING** = still to be run;
  **NOT-NEEDED** = de-prioritized by the curation decision of 2026-08-04
  (diagnostic or already covered), do not re-run.
- The curation decision (authoritative: `labbook/2026-08-03_experiment_readiness_inventory.md`,
  section "Curation decision (2026-08-04)") marks exactly two things NECESSARY:
  the multi-seed K-sweep and the DeepCell cross-dataset eval including KNN
  checkpoints. Both live in the mirror repo, not here.

---

## Cluster requirements

- **Cluster:** Capella, TU Dresden (SSH: `ssh capella`, VPN required).
- **Partition:** `gpu-h100`. NOTE: none of the scripts set `#SBATCH --partition`;
  they rely on the cluster default for the account (gpu-h100).
- **Account:** `p_scads_celltracking` (set in every script).
- **GPU:** NVIDIA H100 80 GB, 1 GPU per node (`#SBATCH --gpus-per-node=1`).
- **Python environment:** `$TRK/.venv` (used by all scripts). The historical
  `setup_env.sh` in this directory instead builds a conda env at
  `$TRK/trackastra_env` — that is a legacy alternative, not what the scripts use.
- **Dependencies** (installed inside the job by each script unless noted):
  `torch torchvision` (cu124 wheels), `lightning pandas scikit-image tifffile
  edt tqdm configargparse wandb tensorboard dask joblib`; plus `pyyaml`
  (phase1_* / dist_ablation), `uv` + `wandb scikit-learn` (ssl_dino_pretrain).
- **wandb:** most scripts export `WANDB_DISABLED=true` + `WANDB_MODE=disabled`.
  The exceptions are `run_baseline.slurm` and `run_ssl_dino_pretrain.slurm`
  (they do not disable wandb; `run_sparse_k16.slurm` explicitly uses
  `--logger wandb --wandb_project trackastra`).

### Path constants (verified against the script bodies)

| Constant | Value |
|---|---|
| `TRK` | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra` |
| `WS` | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601` |
| `BENCH` (`BM_DIR`) | `$WS/bench-transformer-cell-tracking` |
| `ENV_DIR` | `$TRK/.venv` |
| `DATA_DIR` | `/data/cat/ws/mawe985g-data/data/celltracking` |
| vanvliet data | `$DATA_DIR/vanvliet` (6 conditions: `rpsM recA pheA metA cib trpL`) |
| deepcell data | `$DATA_DIR/deepcell` (used by the eval harness in the mirror repo) |
| `CFG_DIR` | `$SLURM_SUBMIT_DIR/benchmark_training/configs` (i.e. the scripts expect to be submitted from the bench-repo root, where `benchmark_training/configs/` is the config subdirectory) |

### Git branches pinned by the scripts

- `feature/sparse-attention-gather`: run_baseline, run_sparse_k4, run_sparse_k16,
  run_sparse_k16_ssl, run_sparse_k32, run_ssl_only, run_quick_bench_baseline,
  run_dist_ablation, run_phase1_scaling, run_phase1_ssl, run_phase1_sweep
  (`git pull origin <branch>` from `$TRK`).
- `cached-dist-attn`: run_cached_dist, run_quick_bench_cached,
  run_ssl_dino_pretrain.
- `master`: run_diag2 pulls only the bench repo (`$BM_DIR`, `git pull origin master`).

Reproducibility caveat: scripts pin **branches, not commits** — results depend
on the branch state at submission time. The bench repo is cloned on demand if
missing (`git clone git@github.com:lest161c/bench-transformer-cell-tracking.git "$BM_DIR"`
— dist_ablation / phase1_*).

### Common job preamble (identical across scripts)

Every training script sets these exports before doing anything else:
`PIP_REQUIRE_VIRTUALENV=false`, `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`,
`MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `NCCL_DEBUG=WARN`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONNOUSERSITE=1`,
`unset PYTHONPATH`, `unset PYTHONHOME`. (`run_sparse_k16_ssl.slurm` additionally
sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `run_ssl_dino_pretrain.slurm`
sets `UV_LINK_MODE=copy` and does not set `PIP_REQUIRE_VIRTUALENV`.)

Standard "normal mode" setup (verbatim, from run_baseline.slurm):
```bash
. "$ENV_DIR/bin/activate"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs
```

---

## Script index

| Script | Status | Purpose | SBATCH time |
|---|---|---|---|
| `run_baseline.slurm` | DONE (historical) | Dense (K=-1) Trackastra training on vanvliet | 48:00:00 |
| `run_cached_dist.slurm` | DONE (historical) | CachedDistAttention training variant (branch `cached-dist-attn`) | 48:00:00 |
| `run_sparse_k4.slurm` | DONE (historical) | Gather-sparse KNN training, K=4 (CLI config, window 10) | 48:00:00 |
| `run_sparse_k16.slurm` | DONE (historical) | Gather-sparse KNN training, K=16, SSL-init encoder + wandb | 48:00:00 |
| `run_sparse_k32.slurm` | DONE (historical) | Gather-sparse KNN training, K=32 (CLI config, window 10) | 48:00:00 |
| `run_sparse_k16_ssl.slurm` | DONE (historical) | K=16 training with SSL pretraining stage (YAML config, window 4) | 48:00:00 |
| `run_ssl_only.slurm` | NOT-NEEDED | SSL pretraining stage only, no supervised fine-tuning | 2:00:00 |
| `run_ssl_dino_pretrain.slurm` | NOT-NEEDED | DINOv2 + contrastive SSL pretraining (never completed; de-prioritized) | 24:00:00 |
| `run_phase1_scaling.slurm` | NOT-NEEDED | Synthetic attention scaling benchmark (N x K x L) on H100 | 2:00:00 |
| `run_phase1_ssl.slurm` | NOT-NEEDED | Phase-1 contrastive SSL pretraining (50 epochs) | 6:00:00 |
| `run_phase1_sweep.slurm` | NOT-NEEDED | Label-fraction sweep, frozen-embedding downstream eval | 4:00:00 |
| `run_dist_ablation.slurm` | NOT-NEEDED | Distortion-family ablation (7 variants, SSL pretrain) | 2:00:00 |
| `run_diag2.slurm` | NOT-NEEDED | Embedding-collapse diagnostic (random vs SSL init) | 0:10:00 |
| `run_quick_bench_baseline.slurm` | NOT-NEEDED (used 2026-05-29) | 10-epoch dense throughput/parity bench | 1:00:00 |
| `run_quick_bench_cached.slurm` | NOT-NEEDED (used 2026-05-29) | 10-epoch cached-dist throughput/parity bench | 1:00:00 |

---

## Per-script details

### 1. `run_baseline.slurm`

| Field | Value |
|---|---|
| Status | DONE (historical) |
| Purpose | Dense (K=-1) Trackastra training on vanvliet; the "baseline" training run for the K-sweep narrative. Clean single-seed baseline results recorded in the README/labbook. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `48:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set — cluster default `gpu-h100`) |

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --config "$CFG_DIR/vanvliet_baseline.yaml"
```
Note: no `WANDB_DISABLED` export; wandb is enabled in this script.

**Config / inputs**
- Config: `benchmark_training/configs/vanvliet_baseline.yaml` (this directory). Key
  settings: `epochs: 500`, `warmup_epochs: 5`, `window: 4`, `attn_dist_mode: v1`,
  `delta_cutoff: 1`, `d_model: 320`, 6+6 layers, `dropout: 0.05`, `lr: 0.0001`,
  `batch_size: 48`, `max_tokens: 2048`, `crop_size: [320,320]`,
  `features: wrfeat`, `causal_norm: quiet_softmax`,
  `attn_positional_bias: rope`, `mixedp: true`, `num_workers: 8`, `seed: 42`.
- Data: 22 train + 2 val vanvliet experiments under
  `/data/cat/ws/mawe985g-data/data/celltracking/vanvliet/<cond>/<exp>`,
  `detection_folders: [TRA]` (absolute paths baked into the YAML).

**Outputs / artifacts**
- Run directory `$TRK/runs/<timestamp>_vanvliet_baseline/` with checkpoints
  (`model.pt`, Lightning `.ckpt`), `train_config.yaml`, tensorboard logs.
- wandb project `trackastra` (not disabled).
- Slurm logs `logs/slurm-baseline-<jobid>.out` / `.err` in the submission dir.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking            # so $SLURM_SUBMIT_DIR/benchmark_training/configs resolves
sbatch benchmark_training/slurm/run_baseline.slurm                         # normal
sbatch --export=TEST=1 benchmark_training/slurm/run_baseline.slurm         # env-only smoke test
```
TEST mode: activates the venv, installs setuptools/torch/deps, pulls the branch,
`pip install -e .`, then prints `trackastra OK:` + `CUDA:` and exits — it does
not train.

---

### 2. `run_cached_dist.slurm`

| Field | Value |
|---|---|
| Status | DONE (historical) |
| Purpose | Full-scale training with `CachedDistAttention` (amortized per-sample cdist) on branch `cached-dist-attn`, same config as baseline. Historical 10-epoch parity evidence is recorded (labbook 2026-05-29); the report's CachedDistAttention claim has been qualified to the small-N regime. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `48:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin cached-dist-attn
python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --config "$CFG_DIR/vanvliet_baseline.yaml"
```

**Config / inputs**
- Same config as `run_baseline.slurm`: `benchmark_training/configs/vanvliet_baseline.yaml`
  (identical hyperparameters — the ONLY intended difference is the attention
  implementation from the `cached-dist-attn` branch).
- Same 22 train + 2 val vanvliet experiments.

**Outputs / artifacts**
- Run dir `$TRK/runs/<timestamp>_vanvliet_baseline/` (name comes from the
  config), checkpoints, TB logs. wandb disabled.
- Slurm logs `logs/slurm-cached-dist-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_cached_dist.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_cached_dist.slurm   # smoke test (also imports CachedDistAttention)
```
TEST mode additionally verifies `from trackastra.model.model_parts import CachedDistAttention`.

Note: for the report it is sufficient to cite the 10-epoch parity runs
(`run_quick_bench_baseline` / `run_quick_bench_cached`, see §14/§15) and the
isolated A500 benchmark (`benchmark_attn/benchmark_cached_dist.py` +
`cached_dist_results.csv`). A full 48 h cached-dist training run is not required:
attention is not the training bottleneck at vanvliet scale (labbook 2026-05-29).

---

### 3. `run_sparse_k4.slurm`

| Field | Value |
|---|---|
| Status | DONE (historical) |
| Purpose | Gather-sparse KNN training with `knn_neighbors=4`, full CLI config (`window 10`, `epochs 100`). Historical run; the resulting checkpoint is an earlier/contaminated variant (window=10, attn_dist v0) — see "Results ledger" caveat. The clean K=4 run lives in the mirror repo. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `48:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true; export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"; git pull origin feature/sparse-attention-gather; python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --name sparse_k4 --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
    --detection_folders TRA \
    --input_train /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/metA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/cib /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/trpL \
    --input_val /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 4096 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.01 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 --knn_neighbors 4 \
    --mixedp True --num_workers 8 --seed 42 --augment 3
```

**Config / inputs**
- No YAML config; the entire configuration is passed on the CLI (above).
- Train data: all 6 vanvliet condition directories; val data: `rpsM recA pheA`.
- `--detection_folders TRA`, `--features wrfeat`, `--seed 42`.

**Outputs / artifacts**
- Run dir `$TRK/runs/<timestamp>_sparse_k4/` (name from `--name sparse_k4`),
  checkpoints, TB logs. wandb disabled.
- Slurm logs `logs/slurm-sparse_k4-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_sparse_k4.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_sparse_k4.slurm   # smoke test
```
Note: `--window 10` and `--epochs 100` here differ from the clean K-sweep
(window 4, epochs 500, `vanvliet_sparse_k4_clean.yaml` in the mirror repo). The
clean K=4 numbers in the results ledger come from the mirror, not this script.

---

### 4. `run_sparse_k16.slurm`

| Field | Value |
|---|---|
| Status | DONE (historical) |
| Purpose | Gather-sparse KNN training with `knn_neighbors=16`, seeded from the DINO-SSL encoder (`--init_encoder runs/ssl_dino_pretrain`), logging to wandb. Historical run; checkpoint contaminated (window=10, attn_dist v0) — do not use for the report (clean K=16 lives in the mirror). |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `48:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --name sparse_k16 --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
    --detection_folders TRA \
    --input_train /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/metA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/cib /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/trpL \
    --input_val /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 4096 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.01 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 --knn_neighbors 16 \
    --init_encoder runs/ssl_dino_pretrain \
    --logger wandb --wandb_project trackastra \
    --mixedp True --num_workers 8 --seed 42 --augment 3
```

**Config / inputs**
- No YAML config; all settings on the CLI (same shape as `run_sparse_k4.slurm`
  but `--knn_neighbors 16`).
- **Dependency:** `--init_encoder runs/ssl_dino_pretrain` expects the encoder
  checkpoint dir produced by `run_ssl_dino_pretrain.slurm` (`--outdir runs/ssl_dino_pretrain`).
  Because that run was never completed, this script in its current form has no
  valid init encoder to load.
- wandb is used (`--logger wandb --wandb_project trackastra`); no
  `WANDB_DISABLED` export in this script.

**Outputs / artifacts**
- Run dir `$TRK/runs/<timestamp>_sparse_k16/`, checkpoints, TB logs; wandb run in
  project `trackastra`.
- Slurm logs `logs/slurm-sparse_k16-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_sparse_k16.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_sparse_k16.slurm   # smoke test
```
Note: the TEST-mode comment inside the file says `run_baseline.slurm` — a
cosmetic copy-paste artifact; the behavior is the standard env smoke test.

---

### 5. `run_sparse_k32.slurm`

| Field | Value |
|---|---|
| Status | DONE (historical) |
| Purpose | Gather-sparse KNN training with `knn_neighbors=32`, full CLI config (window 10, epochs 100). Historical run; checkpoint contaminated — clean K=32 lives in the mirror repo. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `48:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true; export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"; git pull origin feature/sparse-attention-gather; python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --name sparse_k32 --ndim 2 --window 10 --epochs 100 --warmup_epochs 10 \
    --detection_folders TRA \
    --input_train /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/metA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/cib /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/trpL \
    --input_val /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/rpsM /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/recA /data/cat/ws/mawe985g-data/data/celltracking/vanvliet/pheA \
    --outdir runs --distributed False \
    --batch_size 48 --max_tokens 4096 \
    --d_model 320 --num_encoder_layers 6 --num_decoder_layers 6 \
    --pos_embed_per_dim 32 --feat_embed_per_dim 8 \
    --dropout 0.01 --lr 1e-4 \
    --features wrfeat --causal_norm quiet_softmax \
    --attn_positional_bias rope --spatial_pos_cutoff 256 \
    --delta_cutoff 2 --knn_neighbors 32 \
    --mixedp True --num_workers 8 --seed 42 --augment 3
```

**Config / inputs**
- No YAML config; all settings on the CLI (same as K=4 but `--knn_neighbors 32`).
- Train: 6 vanvliet conditions; val: `rpsM recA pheA`; seed 42.

**Outputs / artifacts**
- Run dir `$TRK/runs/<timestamp>_sparse_k32/`, checkpoints, TB logs. wandb disabled.
- Slurm logs `logs/slurm-sparse_k32-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_sparse_k32.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_sparse_k32.slurm   # smoke test
```

---

### 6. `run_sparse_k16_ssl.slurm`

| Field | Value |
|---|---|
| Status | DONE (historical) |
| Purpose | K=16 training with an SSL pretraining stage (`--ssl_pretrain True`), driven by the YAML config `vanvliet_sparse_k16_ssl.yaml` (window 4, epochs 500, `attn_dist_mode: v0`, `knn_neighbors: 16`, `ssl_epochs: 10`). Used historically. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `48:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

Extra env var: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(plus the standard `WANDB_DISABLED=true` / `WANDB_MODE=disabled`).

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --config "$CFG_DIR/vanvliet_sparse_k16_ssl.yaml" \
    --ssl_pretrain True
```

**Config / inputs**
- Config: `benchmark_training/configs/vanvliet_sparse_k16_ssl.yaml`. Key settings:
  `name: sparse_k16_ssl`, `epochs: 500`, `ssl_epochs: 10`,
  `ssl_conditions: [rpsM, recA, pheA, metA, cib, trpL]`, `window: 4`,
  `attn_dist_mode: v0`, `delta_cutoff: 1`, `d_model: 320`, 6+6 layers,
  `dropout: 0.05`, `lr: 0.0001`, `batch_size: 48`, `max_tokens: 2048`,
  `crop_size: [320,320]`, `knn_neighbors: 16`, `features: wrfeat`,
  `causal_norm: quiet_softmax`, `attn_positional_bias: rope`, `mixedp: true`,
  `num_workers: 8`, `seed: 42`.
- Data: the same 22 train + 2 val vanvliet experiments (absolute cluster paths
  in the YAML), plus `ssl_conditions` for the SSL stage.

**Outputs / artifacts**
- SSL stage weights + run dir `$TRK/runs/<timestamp>_sparse_k16_ssl/`,
  checkpoints, TB logs. wandb disabled.
- Slurm logs `logs/slurm-sparse_k16_ssl-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_sparse_k16_ssl.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_sparse_k16_ssl.slurm   # smoke test
```

---

### 7. `run_ssl_only.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (diagnostic; curation decision 2026-08-04 lists `ssl_only` as not required) |
| Purpose | Run ONLY the SSL pretraining stage of `vanvliet_sparse_k16_ssl.yaml` and skip supervised fine-tuning (`--ssl_only True --epochs 0`). |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `2:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

python scripts/train.py \
    --config "$CFG_DIR/vanvliet_sparse_k16_ssl.yaml" \
    --ssl_pretrain True --ssl_only True --epochs 0
```

**Config / inputs**
- Config: `benchmark_training/configs/vanvliet_sparse_k16_ssl.yaml` (same as §6).
- Data: vanvliet, `ssl_conditions` all 6 conditions.

**Outputs / artifacts**
- SSL-pretrained encoder weights (SSL stage only; no supervised model), TB logs
  in `$TRK/runs/<timestamp>_...`. wandb disabled.
- Slurm logs `logs/slurm-ssl_only-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_ssl_only.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_ssl_only.slurm   # smoke test
```
Not needed for the report: SSL showed zero improvement at 10% labels and the
SSL narrative is already covered by the completed identity-BCE / contrastive
experiments and the edge-probe ceiling.

---

### 8. `run_ssl_dino_pretrain.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (never completed; de-prioritized by curation decision — DINOv2 full SSL is in the NOT-NEEDED set) |
| Purpose | DINOv2 (frozen vits14) + NT-Xent contrastive SSL pretraining of the Trackastra encoder via `ssl_dino_trainer.py`. Code-ready, but **no full DINO-SSL checkpoint exists anywhere** (only A500 micro-tests, labbook 2026-06-15). |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `24:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

This is the only script that uses `uv pip` (`export UV_LINK_MODE=copy`; no
`PIP_REQUIRE_VIRTUALENV`); it does NOT disable wandb, and `CFG_DIR` here is
`$SLURM_SUBMIT_DIR/configs` (not `benchmark_training/configs`).

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install wandb scikit-learn
cd "$TRK"
git pull origin cached-dist-attn
uv pip install -e .
mkdir -p logs

python -m trackastra.model.ssl_dino_trainer \
    --config "$CFG_DIR/ssl_dino_pretrain.yaml" \
    --ssl_epochs 20 \
    --ssl_batch_size 4 \
    --ssl_lr 3e-4 \
    --ssl_temperature 0.05 \
    --outdir runs/ssl_dino_pretrain
```

**Config / inputs**
- Config: `$SLURM_SUBMIT_DIR/configs/ssl_dino_pretrain.yaml` — i.e. the script
  expects a `configs/ssl_dino_pretrain.yaml` next to wherever `sbatch` is called.
  In practice submit from the trackastra checkout so that
  `$SLURM_SUBMIT_DIR/configs` resolves to `$TRK/configs`. The config's
  `data_root` points at the HPC vanvliet path.
- DINOv2 weights: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`
  (cached locally at `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`).

**Outputs / artifacts**
- Encoder checkpoint dir `$TRK/runs/ssl_dino_pretrain/` (this is the directory
  that `run_sparse_k16.slurm` references via `--init_encoder runs/ssl_dino_pretrain`).
- Slurm logs `logs/slurm-ssl_dino-<jobid>.out` / `.err`.

**How to run**
```bash
cd $TRK     # so $SLURM_SUBMIT_DIR/configs/ssl_dino_pretrain.yaml resolves
sbatch $WS/bench-transformer-cell-tracking/benchmark_combined/run_ssl_dino_pretrain.slurm
sbatch --export=TEST=1 .../run_ssl_dino_pretrain.slurm   # smoke test (loads DINOv2 via torch.hub)
```
TEST mode additionally verifies that DINOv2 can be loaded from `torch.hub`.
Not needed for the report: micro-tests (gap 0.292 -> 0.040 under distortions) +
edge-probe ceiling already support the architectural conclusion (hedged in the
report).

---

### 9. `run_phase1_scaling.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (diagnostic; curation decision lists `phase1_*` as not required) |
| Purpose | Synthetic attention scaling benchmark on H100: forward-only time + peak memory for dense vs KNN-gather, sweeping N in [128..8192], K in [4,16,32,64], L in [1,6,12]. This is an earlier diagnostic; the authoritative A500/H100 attention benchmarks live in `benchmark_attn/`. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `2:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

No TEST mode. Standard exports plus `WANDB_DISABLED=true` / `WANDB_MODE=disabled`.

**Command (verbatim shell-level; the body is a `python - << 'PYEOF' ... PYEOF` heredoc)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib pyyaml
cd "$TRK"; git pull origin feature/sparse-attention-gather; python -m pip install -e .

# pull/clone the bench repo, then:
mkdir -p logs
cd "$BM_DIR/benchmark_ssl"
python - << 'PYEOF'
...embedded python (full source is in the script body)...
PYEOF
```
The embedded Python (faithful summary — the full 115-line source is in the
script body and is the executable spec):
- `d_model=256, nhead=4, B=2, warmup=5, repeat=10`, fp16.
- `Ns=[128,256,512,1024,2048,4096,8192]`, `Ls=[1,6,12]`, `Ks=[4,16,32,64]`.
- Dense baseline via masked matmul (mask = `cdist > 100 -> -65504.0`), skipped
  with `ERR` at `N >= 4096`; sparse via `torch.gather` on the K nearest
  coordinates + `scaled_dot_product_attention`.
- Per-row records `{N, L, K, variant, time_ms_per_step, mem_mb}`.

**Config / inputs**
- No data files needed (synthetic `torch.randn` tensors); requires the bench
  repo (`$BM_DIR/benchmark_ssl`) for the working directory only.

**Outputs / artifacts**
- `$BM_DIR/benchmark_ssl/runs/phase1/scaling_h100.csv` (columns
  `N,L,K,variant,time_ms_per_step,mem_mb`).
- Slurm logs `logs/slurm-scale-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_phase1_scaling.slurm
```
Not needed for the report — superseded by the A500 real-attention benchmarks
(`benchmark_attn/`).

---

### 10. `run_phase1_ssl.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (diagnostic; `phase1_*` not required) |
| Purpose | Phase-1 contrastive SSL pretraining on the 7D regionprops encoder (`pretrain.py config_cluster.yaml`, 50 epochs, all 6 vanvliet conditions). Produced `runs/ssl_phase1/best_model.pt` consumed by §11/§12. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `6:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

No TEST mode. `WANDB_DISABLED=true` / `WANDB_MODE=disabled`.

**Command (verbatim shell-level; writes a YAML config then runs pretrain.py)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib pyyaml
cd "$TRK"; git pull origin feature/sparse-attention-gather; python -m pip install -e .
# pull/clone bench repo, then:
cd "$BM_DIR/benchmark_ssl"
mkdir -p runs
cat > config_cluster.yaml << 'YEOF'
name: ssl_phase1
data_root: DATA_DIR_PLACEHOLDER/vanvliet
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
training: {batch_size: 16, lr: 0.0003, weight_decay: 0.01, epochs: 50, val_split: 0.1, checkpoint_every: 10}
seed: 42
YEOF
sed -i "s|DATA_DIR_PLACEHOLDER|$DATA_DIR|g" config_cluster.yaml

python pretrain.py config_cluster.yaml
echo "Job B (SSL pretrain) done."
```

**Config / inputs**
- Generated config `config_cluster.yaml` (template shown above) written into
  `$BM_DIR/benchmark_ssl/`.
- Data: `$DATA_DIR/vanvliet`, all 6 conditions, `features: regionprops2`.
- Requires bench-repo modules `pretrain.py` (and its imports) in
  `$BM_DIR/benchmark_ssl`.

**Outputs / artifacts**
- `$BM_DIR/benchmark_ssl/runs/ssl_phase1/best_model.pt` (SSL checkpoint consumed
  by `run_phase1_sweep.slurm` and `run_diag2.slurm`).
- Slurm logs `logs/slurm-ssl-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_phase1_ssl.slurm
```
Not needed for the report (diagnostic phase-1 pipeline).

---

### 11. `run_phase1_sweep.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (curation decision: low-label sweep is de-prioritized; `phase1_*` diagnostic) |
| Purpose | Label-fraction downstream sweep using the frozen SSL-pretrained encoder: cosine-similarity + Hungarian matching at {1,5,10,25,50,100}% of training frame pairs; compares `dense_noSSL`, `knn_noSSL`, `knn_SSL`. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `4:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

No TEST mode. `WANDB_DISABLED=true` / `WANDB_MODE=disabled`.

**Command (verbatim shell-level; body is a `python - << PYEOF ... PYEOF` heredoc)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib pyyaml
cd "$TRK"; git pull origin feature/sparse-attention-gather; python -m pip install -e .
# pull/clone bench repo, then:
cd "$BM_DIR/benchmark_ssl"
mkdir -p runs/phase1
python - << PYEOF
...embedded python (full source is in the script body)...
PYEOF
echo "Job C (label sweep) done."
```
The embedded Python (faithful summary — full source is in the script body):
- Loads vanvliet frames via `ssl_pipeline.load_experiment_frames` (all 6
  conditions), builds consecutive frame pairs, 10% val / 90% test split
  (seed 42).
- `build_model(load_ssl=...)` creates a `CellEmbedder(feat_dim=7, coord_dim=2,
  d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1)` and
  optionally loads `runs/ssl_phase1/best_model.pt`.
- Evaluation: `model.encode` -> cosine cost -> `linear_sum_assignment`,
  matching threshold 50 px; reports `test_accuracy`.
- `fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]`, methods
  `dense_noSSL`, `knn_noSSL`, `knn_SSL`.

**Config / inputs**
- Requires `$BM_DIR/benchmark_ssl` modules (`ssl_pipeline.py`, `track_encoder.py`).
- Data: `$DATA_DIR/vanvliet` (all 6 conditions).
- Checkpoint: `runs/ssl_phase1/best_model.pt` from `run_phase1_ssl.slurm`
  (logs a warning if absent).

**Outputs / artifacts**
- `$BM_DIR/benchmark_ssl/runs/phase1/label_sweep.csv` (columns
  `label_fraction,method,test_accuracy,test_correct,test_total,n_train_pairs`).
- Slurm logs `logs/slurm-sweep-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_phase1_sweep.slurm
```
Not needed for the report — the low-label regime already showed no SSL benefit
at 10% labels (curation decision).

---

### 12. `run_dist_ablation.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (curation decision lists `dist_ablation` as not required) |
| Purpose | Distortion-family ablation for the phase-1 SSL pipeline: 7 variants ("full" + each of jitter/dropout/feature_noise/photometric/elastic/affine removed), each `pretrain.py` run of 20 epochs. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `2:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

No TEST mode. `WANDB_DISABLED=true` / `WANDB_MODE=disabled`. Also installs
`pyyaml`.

**Command (verbatim shell-level; generates one config per variant)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib pyyaml
cd "$TRK"; git pull origin feature/sparse-attention-gather; python -m pip install -e .
# pull/clone bench repo, then:
cd "$BM_DIR/benchmark_ssl"; mkdir -p runs/dist_abl logs
DISTORTIONS=(
  "full:affine,elastic,jitter,dropout,photometric,feature_noise"
  "no_jitter:affine,elastic,dropout,photometric,feature_noise"
  "no_dropout:affine,elastic,jitter,photometric,feature_noise"
  "no_feature_noise:affine,elastic,jitter,dropout,photometric"
  "no_photometric:affine,elastic,jitter,dropout,feature_noise"
  "no_elastic:affine,jitter,dropout,photometric,feature_noise"
  "no_affine:elastic,jitter,dropout,photometric,feature_noise"
)
for variant in "${DISTORTIONS[@]}"; do
  NAME="${variant%%:*}"; DISTS="${variant#*:}"
  cat > config_abl.yaml << YEOF
name: dist_abl_$NAME
data_root: $DATA_DIR/vanvliet
conditions: [rpsM, recA, pheA, metA, cib, trpL]
ndim: 2
features: regionprops2
distortions: [DISTS_PLACEHOLDER]
affine: {degrees: 15, scale: [0.85, 1.15], shear: [0.1, 0.1]}
elastic: {alpha: [10, 50], sigma: [5, 15]}
jitter: {std: [2, 8], p_cell_jitter: 0.8}
dropout: {p_drop: [0.05, 0.2]}
photometric: {scale: [0.5, 2.0], shift: [-0.1, 0.1]}
feature_noise: {std: [0.02, 0.15]}
encoder: {d_model: 128, nhead: 4, num_layers: 4, dim_feedforward: 256, dropout: 0.1}
ssl: {temperature: 0.05}
training: {batch_size: 16, lr: 0.0003, weight_decay: 0.01, epochs: 20, val_split: 0.1, checkpoint_every: 10}
seed: 42
YEOF
  sed -i "s|DISTS_PLACEHOLDER|$DISTS|g" config_abl.yaml
  python pretrain.py config_abl.yaml
  echo "Done: $NAME"
done
echo "=== All distortion ablations complete ==="
```

**Config / inputs**
- Generated `config_abl.yaml` per variant (template above; `data_root=$DATA_DIR/vanvliet`,
  all 6 conditions, `features: regionprops2`, 20 epochs, seed 42).
- Requires `$BM_DIR/benchmark_ssl` `pretrain.py`.

**Outputs / artifacts**
- One run per variant under `$BM_DIR/benchmark_ssl/runs/dist_abl/dist_abl_<name>/`
  (checkpoints, logs); `logs/slurm-dist-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_dist_ablation.slurm
```
Not needed for the report (diagnostic distortion attribution).

---

### 13. `run_diag2.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (diagnostic; curation decision lists `diag2` as not required) |
| Purpose | Embedding-collapse diagnostic: compares inter-cell / positive-pair cosine similarities and embedding variance for a random-init vs the SSL-pretrained `CellEmbedder` on rpsM vanvliet frames. Prints to stdout only. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `0:10:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `4` (the only script with 4 CPUs) |
| `--mem` | `20G` (the only script with 20G) |
| `--partition` | (not set) |

No TEST mode. This script does **not** install anything and does **not** pull the
trackastra repo; it activates the venv and pulls only the bench repo
(`git pull origin master`).

**Command (verbatim shell-level; body is a `python -c '...'` one-liner heredoc)**
```bash
TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
. "$TRK/.venv/bin/activate"
WS=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601
BM_DIR="$WS/bench-transformer-cell-tracking"
cd "$BM_DIR" && git pull origin master 2>/dev/null
cd "$BM_DIR/benchmark_ssl"

python -c '
import torch, numpy as np, sys
sys.path.insert(0, ".")
from track_encoder import CellEmbedder
from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl
from distortions import DistortionPipeline
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
frames = load_experiment_frames("/data/cat/ws/mawe985g-data/data/celltracking/vanvliet", conditions=["rpsM"])
...
ckpt = torch.load("runs/ssl_phase1/best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
...
' 2>&1
echo "Job done"
```
(Full Python source is in the script body; key parameters: `d_model=128,
nhead=4, num_layers=4, dim_feedforward=256`, distortions
`jitter/dropout/feature_noise`, batch of 4, loads
`runs/ssl_phase1/best_model.pt`.)

**Config / inputs**
- No config file; requires `$BM_DIR/benchmark_ssl` modules
  (`track_encoder.py`, `ssl_pipeline.py`, `distortions.py`).
- Data: vanvliet `rpsM` only.
- Checkpoint: `runs/ssl_phase1/best_model.pt` (from `run_phase1_ssl.slurm`).

**Outputs / artifacts**
- None on disk — diagnostic prints only (`RANDOM INIT` / `SSL CHECKPOINT` /
  `VARIANCE` sections). Slurm logs `logs/slurm-diag2-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_diag2.slurm
```
Not needed for the report.

---

### 14. `run_quick_bench_baseline.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (curation decision lists `quick_bench` as not required). Used historically on 2026-05-29 for the 10-epoch attention-is-not-the-bottleneck parity runs. |
| Purpose | 10-epoch dense training run (`--epochs 10 --name quick_bench_baseline`) to measure per-epoch wall time / convergence parity on branch `feature/sparse-attention-gather`. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `1:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

`WANDB_DISABLED=true` / `WANDB_MODE=disabled`.

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .
mkdir -p logs

echo "=== BASELINE BENCH (feature/sparse-attention-gather) ==="
date
python scripts/train.py \
    --config "$CFG_DIR/vanvliet_baseline.yaml" \
    --epochs 10 \
    --name quick_bench_baseline
echo "=== DONE ==="
date
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State,MaxRSS --noheader
```

**Config / inputs**
- Config: `benchmark_training/configs/vanvliet_baseline.yaml` (see §1), overridden with
  `--epochs 10 --name quick_bench_baseline`.

**Outputs / artifacts**
- Run dir `$TRK/runs/<timestamp>_quick_bench_baseline/` (10-epoch run), plus the
  `sacct` resource summary in the slurm log. Slurm logs
  `logs/slurm-quickbench-base-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_quick_bench_baseline.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_quick_bench_baseline.slurm   # smoke test
```

---

### 15. `run_quick_bench_cached.slurm`

| Field | Value |
|---|---|
| Status | NOT-NEEDED (curation decision lists `quick_bench` as not required). Used historically on 2026-05-29 for the 10-epoch CachedDistAttention parity runs. |
| Purpose | 10-epoch training run on branch `cached-dist-attn` (`--epochs 10 --name quick_bench_cached`) to measure per-epoch wall time / convergence parity vs baseline. |

**SBATCH resources**

| Directive | Value |
|---|---|
| `--account` | `p_scads_celltracking` |
| `--time` | `1:00:00` |
| `--nodes` / `--ntasks` | `1` / `1` |
| `--gpus-per-node` | `1` (H100 80GB) |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--partition` | (not set) |

`WANDB_DISABLED=true` / `WANDB_MODE=disabled`.

**Command (verbatim, normal mode)**
```bash
. "$ENV_DIR/bin/activate"
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export WANDB_DISABLED=true
export WANDB_MODE=disabled
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib
cd "$TRK"
git pull origin cached-dist-attn
python -m pip install -e .
mkdir -p logs

echo "=== CACHED-DIST BENCH (cached-dist-attn) ==="
date
python scripts/train.py \
    --config "$CFG_DIR/vanvliet_baseline.yaml" \
    --epochs 10 \
    --name quick_bench_cached
echo "=== DONE ==="
date
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,State,MaxRSS --noheader
```

**Config / inputs**
- Config: `benchmark_training/configs/vanvliet_baseline.yaml`, overridden with
  `--epochs 10 --name quick_bench_cached`; branch `cached-dist-attn`.

**Outputs / artifacts**
- Run dir `$TRK/runs/<timestamp>_quick_bench_cached/` (10-epoch run) + `sacct`
  summary in the slurm log. Slurm logs `logs/slurm-quickbench-cd-<jobid>.out` / `.err`.

**How to run**
```bash
cd $WS/bench-transformer-cell-tracking
sbatch benchmark_training/slurm/run_quick_bench_cached.slurm
sbatch --export=TEST=1 benchmark_training/slurm/run_quick_bench_cached.slurm   # smoke test
```

---

## Experiment status summary (curation decision 2026-08-04 applied)

NECESSARY per the curation decision: multi-seed K-sweep and DeepCell
cross-dataset eval including KNN checkpoints — both live in the **mirror repo**
(`bench-transformer-cell-tracking/benchmark_combined/` and
`benchmark_ssl/cnn_encoder/cross_dataset_eval.py`), not in this directory.

| Experiment | Script(s) | Status | Rationale |
|---|---|---|---|
| Dense baseline training (K=-1) | `run_baseline.slurm` | DONE (historical) | Single-seed done; multi-seed baseline lives in mirror (NECESSARY set) |
| CachedDistAttention training | `run_cached_dist.slurm` | DONE (historical; 10-epoch parity) | Claim qualified to small-N; no full 48h re-run required |
| K-sweep K=4 / K=16 / K=32 training | `run_sparse_k4/16/32.slurm` | DONE (historical; checkpoints contaminated, DO NOT USE) | Clean single-seed runs done in mirror; multi-seed sweep is NECESSARY (mirror) |
| K=16 + SSL stage | `run_sparse_k16_ssl.slurm` | DONE (historical) | Used historically; not re-run |
| SSL-only pretraining | `run_ssl_only.slurm` | NOT-NEEDED | Diagnostic; SSL no benefit at 10% labels |
| DINOv2 full SSL pretrain | `run_ssl_dino_pretrain.slurm` | NOT-NEEDED | Never completed; de-prioritized by curation decision |
| Phase-1 attention scaling | `run_phase1_scaling.slurm` | NOT-NEEDED | Diagnostic; superseded by `benchmark_attn/` |
| Phase-1 SSL pretrain | `run_phase1_ssl.slurm` | NOT-NEEDED | Diagnostic phase-1 pipeline |
| Phase-1 label-fraction sweep | `run_phase1_sweep.slurm` | NOT-NEEDED | Low-label sweep de-prioritized |
| Distortion ablation | `run_dist_ablation.slurm` | NOT-NEEDED | Diagnostic |
| Embedding-collapse diagnostic | `run_diag2.slurm` | NOT-NEEDED | Diagnostic |
| Quick-bench parity (dense) | `run_quick_bench_baseline.slurm` | NOT-NEEDED (used 2026-05-29) | Diagnostic |
| Quick-bench parity (cached-dist) | `run_quick_bench_cached.slurm` | NOT-NEEDED (used 2026-05-29) | Diagnostic |
| **Multi-seed K-sweep (5K x seeds 42/43/44)** | mirror: `run_single_config.slurm` + `vanvliet_*_clean.yaml` | **PENDING** | **NECESSARY** (mirror repo) |
| **DeepCell eval incl. KNN checkpoints** | mirror: `cross_dataset_eval.py` (edit `CHECKPOINTS`) | **PENDING** | **NECESSARY** (mirror repo) |

---

## Early-stopping guidance (all long runs)

- Never run to the `--epochs 500` upper bound if `val_loss` has plateaued.
  Patience **83** (`epochs // 6`) already stops near the converged optimum.
- `--epochs 500` is only an upper bound; expect early stopping at ~237–406
  epochs.
- Measured cost on H100 (clean single-seed K-sweep, seed 42): **6.4–11 h per
  run** at **~1.63 min/epoch** (e.g. baseline 372 epochs / 606 min; K=16 406
  epochs / 661 min).
- Budget each job `--time 48:00:00` (as the scripts do) so early-stopped runs
  are never killed by the scheduler.

---

## Results ledger (DONE experiments)

### Clean single-seed K-sweep (seed 42) — TRA/AOGM on 32 vanvliet experiments

Source: `benchmark_training/README.md` (matches `labbook/2026-06-02_experiment_completion_status.md`,
rows verified: epochs x ~1.63 min/epoch = reported times).

| Model | Mean TRA | Mean AOGM | Edge F1 | Div F1 | Epochs | Time (min / h) |
|---|---|---|---|---|---|---|
| Baseline (K=-1) | 0.9963 | 51.0 | 0.991 | 0.970 | 372 | 606 min / 10.1 h |
| K=4 | 0.9957 | 62.0 | 0.991 | 0.971 | 237 | 381 min / 6.4 h |
| K=16 | **0.9972** | **37.7** | 0.993 | 0.976 | 406 | 661 min / 11.0 h |
| K=32 | 0.9963 | 52.6 | 0.991 | 0.971 | 362 | 591 min / 9.9 h |
| K=64 | 0.9969 | 39.8 | 0.993 | 0.977 | 366 | 599 min / 10.0 h |

Cluster run dirs (mirror repo runs, `$TRK/runs/`):
`2026-06-01_23-04-{13_baseline,12_sparse_k4,18_sparse_k16,29_sparse_k32,09_sparse_k64}_clean/`.
NOTE: the k16/k32 `model.pt` in this directory's history are CONTAMINATED
(window=10, `attn_dist` v0 — do not use); the clean checkpoints above exist only
on the cluster. K=4 and K=64 `model.pt` are missing locally.

### Attention-is-not-the-bottleneck parity runs (2026-05-29, H100)

Three 10-epoch runs via the quick-bench scripts, identical config
`vanvliet_baseline.yaml`, comparing `cached-dist-attn` vs
`feature/sparse-attention-gather`:

| Variant | batch_size | Elapsed (10 epochs) | val_loss @ ep 5 |
|---|---|---|---|
| baseline | 48 | 14:17 | 0.118 |
| cached-dist | 48 | 14:14 | 0.107 |
| baseline | 128 | 14:21 | — |
| cached-dist | 128 | 14:02 | — |

Max wall-time difference 2.2% — attention is not the training bottleneck at
vanvliet scale (N ~140–350 per sample). The isolated CachedDistAttention
speedup (~1.5x at L=12, small N) is qualified in labbook 2026-08-03 §1.4
(~2x only at N <= 256; ~1.05x at N=512–2048; OOM at 8192 on A500).

### Local A500 artifacts in this directory

- `results/training_speed_memory_by_N_K.csv` — full-model speed/memory table (fp32, fwd+bwd+AdamW)
  produced by `benchmark_speed_mem.py` on 2026-08-04 (A500). K=64 @ N=512:
  194.1 ms / 1604.1 MB (fits A500); full K{0,4,8,16,32,64} x N{128,256,512}
  table present.

### Supporting local (A500) results from the readiness inventory (labbook 2026-08-03/04)

- 5-fold edge probe (2026-08-04): DINOv2 0.8958±0.0362, HOCT19 0.8799±0.0201,
  7D regionprops 0.8595±0.0137, CNN-NT-Xent 0.7031±0.0554, CNN-e2e 0.5000±0.0000
  (`results/unified_probe_results_cv.json`).
- CachedDistAttention real benchmark (`benchmark_attn/cached_dist_results.csv`),
  GatherSparseAttentionV3 (`gather_v3_results.csv`), sparse forward sweep,
  backward N=8192, NSA small-N — see labbook inventory §1.
- Vanvliet inference eval of baseline checkpoint (2026-08-04,
  `eval_traccuracy/results_local_baseline.csv`): mean TRA 0.9952 vs published
  0.9963 (matches within ~1%).

---

## Discrepancies / notes found between the labbook inventory and the actual files

1. **DINO slurm path:** labbook §2.5 and §6 reference the script as
   `benchmark_ssl/run_ssl_dino_pretain.slurm`; the file physically lives at
   `benchmark_training/slurm/run_ssl_dino_pretrain.slurm`.
2. **`run_ssl_only.slurm` wall time:** `README.md` lists it as 6 h; the actual
   `#SBATCH --time=2:00:00` is 2 h.
3. **K-sweep provenance:** the clean single-seed K-sweep results in the README
   were produced by the **mirror repo** (`vanvliet_*_clean.yaml`,
   `run_single_config.slurm`), not by the CLI-config scripts in this directory
   (`run_sparse_k4/16/32.slurm` use `--window 10 --epochs 100`, which produced
   the contaminated checkpoints). `run_sparse_k64.slurm` does not exist here.
4. **README config table** lists `vanvliet_sparse_k16.yaml` at the directory
   root; it only exists under `cluster_configs/` (root has
   `vanvliet_baseline.yaml` and `vanvliet_sparse_k16_ssl.yaml`).
5. **Setup env mismatch:** `setup_env.sh` creates a conda env at
   `$TRK/trackastra_env`, but every `.slurm` script uses `$TRK/.venv`.
6. **wandb inconsistency:** `run_baseline.slurm` and `run_ssl_dino_pretrain.slurm`
   do not set `WANDB_DISABLED`; `run_sparse_k16.slurm` explicitly logs to wandb
   (`--logger wandb --wandb_project trackastra`) while its sibling K scripts
   disable wandb.
7. **Git branches pinned, not commits:** scripts `git pull origin
   feature/sparse-attention-gather|cached-dist-attn`; results are therefore not
   bit-reproducible across branch history.
8. **run_sparse_k16 dependency on a never-run artifact:** `--init_encoder
   runs/ssl_dino_pretrain` requires the output of `run_ssl_dino_pretrain.slurm`,
   which was never completed — the script as written has no valid init encoder.
9. **`run_sparse_k16.slurm` TEST-mode comment** references `run_baseline.slurm`
   (cosmetic copy-paste artifact only).
10. **No `--partition` anywhere:** all 15 scripts rely on the cluster default
    (gpu-h100) rather than setting `#SBATCH --partition`.
