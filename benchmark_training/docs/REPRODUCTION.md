# REPRODUCTION.md — benchmark_training

Concise reproduction guide for every experiment in this repository.

## Scope

This repository benchmarks **training performance** of Trackastra —
both resource usage (time, memory) and evaluation results (loss,
accuracy). Isolated attention benchmarks live in `benchmark_attn/`;
isolated SSL proof-of-concept benchmarks live in `benchmark_ssl/`.

## Prerequisites

### Environment variables

```bash
export TRACKASTRA_DIR=/path/to/trackastra
export DATA_DIR=/path/to/vanvliet
export BENCH_DIR=/path/to/bench-transformer-cell-tracking
```

### Python environment

```
torch torchvision (cu124 wheels)
lightning pandas scikit-image tifffile edt tqdm
configargparse wandb tensorboard dask joblib
pyyaml
```

### Hardware requirements

| Experiment | GPU | VRAM | Runtime |
|---|---|---|---|
| `benchmark_speed_mem.py` (N=8192) | A500 | 4 GB | ~10 min |
| `benchmark_combined.py` | A500 | 4 GB | ~30 min |
| `benchmark_ablation.py` | A500 | 4 GB | ~45 min |
| `run_training.slurm` | H100 | 80 GB | 6–11 h |
| `run_diagnostic.slurm` | H100 | 80 GB | <5 min |

---

## 1. N=8192 training resource benchmark

**Script:** `scripts/benchmarks/benchmark_speed_mem.py`
**Output:** `results/combined_n8192.csv`
**GPU:** A500 (local, B=1)

```bash
python scripts/benchmarks/benchmark_speed_mem.py
```

Measures training step time + peak memory at N=8192 for five
attention+SSL variants. Tests the memory ceiling of the full
Trackastra association model (with pair_norm).

---

## 2. Combined 2×2 factorial convergence benchmark

**Script:** `scripts/benchmarks/benchmark_combined.py`
**Output:** `results/combined_attention_ssl_2x2.csv`
**GPU:** A500 (local)

```bash
python scripts/benchmarks/benchmark_combined.py --epochs 15
```

Tests 6 variants: dense/sparseK4/sparseK16 × random/SSL init.
Measures wall-clock convergence (val loss over epochs).

---

## 3. Convergence ablation benchmark

**Script:** `scripts/benchmarks/benchmark_ablation.py`
**Output:** `results/full_attention_ablation_results.csv`
**GPU:** A500 (local)

```bash
python scripts/benchmarks/benchmark_ablation.py --epochs 15
```

Measures train/val convergence for dense, sparseK4, sparseK16 at
L=1,4, N=512 over 15 epochs.

---

## 4. Analysis and plotting

| Script | Input | Output |
|---|---|---|
| `scripts/analysis/plot_ablation.py` | `full_attention_ablation_results.csv` | `full_attention_ablation_report.html` |
| `scripts/analysis/plot_combined.py` | `combined_attention_ssl_2x2.csv` | `combined_attention_ssl_report.html` |

```bash
python scripts/analysis/plot_ablation.py
python scripts/analysis/plot_combined.py
```

---

## 5. SLURM training runs (H100 cluster)

> **Before submitting any slurm script, you MUST replace the `REPO_ROOT`
> placeholder** at the top of the script (see the `TODO: REPLACE` comment
> block).  Slurm copies the script to `/var/spool/slurmd/jobXXX/` before
> running, so neither `${BASH_SOURCE[0]}` nor `$SLURM_SUBMIT_DIR` reliably
> resolves to the file's real location — the script must know its repo
> root explicitly.  This is the only per-cluster value in the file;
> everything else is resolved relative to it.

### Unified training script

**Script:** `scripts/slurm/run_training.slurm`
**GPU:** H100 80 GB (Capella cluster, TU Dresden)
**Partition:** `gpu-h100`
**Account:** `p_scads_celltracking`

Variants:

| `VARIANT` | Description | Config |
|---|---|---|
| `baseline` | Dense (K=-1) Trackastra training | `configs/vanvliet_baseline.yaml` |
| `cached_dist` | CachedDistAttention training | same config, branch `cached-dist-attn` |
| `sparse_k4` | Gather-sparse KNN, K=4 | CLI config, window 10 |
| `sparse_k16` | Gather-sparse KNN, K=16 | CLI config, window 10 |
| `sparse_k32` | Gather-sparse KNN, K=32 | CLI config, window 10 |
| `sparse_k16_ssl` | K=16 + SSL pretraining stage | `configs/vanvliet_sparse_k16_ssl.yaml` |

```bash
sbatch --export=VARIANT=baseline scripts/slurm/run_training.slurm
sbatch --export=VARIANT=baseline,TEST=1 scripts/slurm/run_training.slurm
```

### Diagnostic script

**Script:** `scripts/slurm/run_diagnostic.slurm`

```bash
sbatch --export=MODE=smoke scripts/slurm/run_diagnostic.slurm
sbatch --export=MODE=quick_bench scripts/slurm/run_diagnostic.slurm
```

---

## Config files

| File | Purpose |
|---|---|
| `configs/vanvliet_baseline.yaml` | Dense training (d_model=320, 6L+6L, window=4) |
| `configs/vanvliet_sparse_k16_ssl.yaml` | K=16 training with SSL pretraining stage |

---

## Results directory

The `results/` directory contains CSV data and HTML reports produced
by the benchmark and analysis scripts. All files are reproducible by
re-running the corresponding script.

| File | Producer |
|---|---|
| `combined_n8192.csv` | `benchmark_speed_mem.py` |
| `combined_attention_ssl_2x2.csv` | `benchmark_combined.py` |
| `full_attention_ablation_results.csv` | `benchmark_ablation.py` |
| `full_attention_ablation_report.html` | `plot_ablation.py` |
| `combined_attention_ssl_report.html` | `plot_combined.py` |

---

## Migrated benchmarks

The following benchmarks were migrated to their appropriate repositories:

| Benchmark | Migrated to | Reason |
|---|---|---|
| Speed sweep mode (`--mode sweep`) | `benchmark_attn/` | Pure attention layer timing; redundant with `benchmark_sweep.py` |
| Ablation Phase 1 (speed scaling) | `benchmark_attn/` | 1-epoch attention scaling; redundant with `benchmark_sweep.py` |
| Downstream SSL transfer evaluation | `benchmark_ssl/` | Isolated SSL proof-of-concept; tests SSL transfer quality |
