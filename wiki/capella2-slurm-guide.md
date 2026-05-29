# Capella HPC Cluster — Slurm Operations Guide

**Cluster hostname:** `capella` (ssh via TU Dresden VPN — `eduvpn` required first)
**SSH key:** `~/.ssh/trackastra`
**Connect:**
```bash
eval $(ssh-agent) && ssh-add ~/.ssh/trackastra
ssh capella
```

---

## Project Paths (on cluster)

| Resource | Path |
|----------|------|
| Workspace | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/` |
| Trackastra repo | `.../trackastra/` (branch `feature/sparse-attention-gather`) |
| Benchmark repo | `.../bench-transformer-cell-tracking/` (branch `feature/imgfeat-ssl`) |
| SSL pipeline | `.../bench-transformer-cell-tracking/benchmark_ssl/` |
| Python venv | `.../trackastra/.venv/` |
| Old conda env | `~/miniconda3/envs/trackastra/` |
| Data root | `/data/cat/ws/mawe985g-data/data/celltracking/vanvliet/` |

---

## Submit Jobs

All `.slurm` scripts in `benchmark_combined/`. Submit from project root:

```bash
# Trackastra training (dense, KNN)
sbatch benchmark_combined/run_baseline.slurm
sbatch benchmark_combined/run_sparse_k4.slurm
sbatch benchmark_combined/run_sparse_k16.slurm
sbatch benchmark_combined/run_sparse_k32.slurm

# Trackastra with SSL pretraining
sbatch benchmark_combined/run_sparse_k16_ssl.slurm
sbatch benchmark_combined/run_ssl_only.slurm

# Phase 1 benchmarks (SSL pipeline in benchmark_ssl/)
sbatch benchmark_combined/run_phase1_scaling.slurm  # synthetic attention perf (2h)
sbatch benchmark_combined/run_phase1_ssl.slurm       # SSL pretrain (6h)
sbatch benchmark_combined/run_phase1_sweep.slurm     # label-fraction eval (4h)
sbatch benchmark_combined/run_diag2.slurm             # embedding collapse check (10min)
sbatch benchmark_combined/run_dist_ablation.slurm     # distortion ablation (2h)
```

### Quick Test Mode

Any Trackastra training script supports `TEST=1` — env setup only, no training:

```bash
sbatch --export=TEST=1 run_baseline.slurm
```

### Dependency Chains

Phase 1 jobs can chain via `--dependency`:

```bash
# Submit scaling (A), then SSL (B), then sweep (C)
sbatch run_phase1_scaling.slurm
JOB_B=$(sbatch --dependency=afterok:<JOB_A> run_phase1_ssl.slurm)
sbatch --dependency=afterok:$JOB_B run_phase1_sweep.slurm
```

---

## Monitor Jobs

```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j <JOB_ID>

# Detailed info
scontrol show job <JOB_ID>

# Find which node
squeue -j <JOB_ID> -o "%i %N %j %T"
```

---

## View Logs

Logs written to `benchmark_combined/logs/` on cluster. Pattern: `slurm-<jobname>-<jobid>.out` / `.err`.

```bash
# List logs
ls -la ~/trackastra/logs/       # via project-login on capella
ls -la benchmark_combined/logs/ # relative to project repo

# Tail running job log
tail -f logs/slurm-baseline-<JOB_ID>.out

# Check stderr for errors
cat logs/slurm-baseline-<JOB_ID>.err

# All recent errors
tail -n 100 logs/*.err
```

**Local copies:** Some logs synced to `logs/` in local project repo. Check `logs/slurm-*.err`.

---

## Check Job Output (sacct)

```bash
# Summary (exit code, elapsed, cpu/mem)
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,NodeList

# All recent jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed,Timelimit --starttime 2026-05-01

# Failed jobs only
sacct -u $USER --state=FAILED --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## Cancel Jobs

```bash
scancel <JOB_ID>           # single
scancel -u $USER           # all your jobs
scancel --name tr_baseline # by job name
scancel --state=PENDING    # queued only (keep running)
```

---

## Common sbatch Flags Used in This Project

| Flag | Value | Meaning |
|------|-------|---------|
| `--account` | `p_scads_celltracking` | Project allocation |
| `--partition` | `gpu-h100` | GPU queue (H100) |
| `--time` | `48:00:00` | 48h max (baseline), 2-6h (phase1) |
| `--gpus-per-node` | `1` | 1 GPU per job |
| `--cpus-per-task` | `8` | CPU threads |
| `--mem` | `90G` | RAM |

---

## Env Setup (inside slurm scripts)

Every `.slurm` script does this boilerplate:

```bash
export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export WANDB_DISABLED=true
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/.venv"
. "$ENV_DIR/bin/activate"

# Install fresh each run
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib pyyaml

cd "$TRK" && git pull origin feature/sparse-attention-gather && python -m pip install -e .
```

---

## Pull Checkpoints & Evaluate Locally

From local machine (VPN connected):

```bash
bash scripts/pull_and_eval.sh
```

This scp-s model.pt + config from cluster runs to `scripts/checkpoints/<model>/`, then runs `scripts/eval_model.py` on each.

---

## Known Slurm Script Bugs (Fixed)

| Bug | Fix |
|-----|-----|
| K4/K16/K32 all share `--name sparse_k16` → same run dir overwrites | Use distinct `--name sparse_k4` / `--name sparse_k16` / `--name sparse_k32` |
| Phase1 jobs used hardcoded `DATA_DIR_PLACEHOLDER` in inline config | `sed` replace after heredoc |
| Dense benchmark fp16 overflow (`-1e9` too negative for fp16) | Use `-65504.0` (fp16 min) |

---

## GPU Node Naming

Nodes named `c69`, `c90`, `c116`, etc. (under `gpu-h100` partition). Single job per node.

---

## Common Commands Quick Ref

```bash
ssh capella                          # login (VPN first)
sbatch script.slurm                  # submit
squeue -u $USER                      # list my jobs
squeue -j 3576409 -o "%i %N %j %T"  # job + node
scancel 3576409                      # kill
sacct -j 3576409 --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -f logs/slurm-baseline-3576409.out  # live log
sacct -u $USER --starttime 2026-05-01    # history
```
