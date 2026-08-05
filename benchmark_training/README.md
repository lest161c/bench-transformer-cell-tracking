# benchmark_training — Full-scale Training Benchmarks

SLURM submission scripts, configs, and analysis for Trackastra training runs on the HPC cluster (Capella, H100 GPUs).

> Note (2026-08-05): directory renamed from `benchmark_combined/` to `benchmark_training/`
> and restructured into `benchmarks/`, `analysis/`, `slurm/`, `configs/`, `results/`.

## Slurm Scripts (`slurm/`)

| Script | Description | Est. time |
|--------|-------------|-----------|
| `slurm/run_baseline.slurm` | Dense Trackastra training (baseline) | 48h |
| `slurm/run_cached_dist.slurm` | CachedDistAttention variant (amortized cdist) | 48h |
| `slurm/run_sparse_k4.slurm` | Gather-sparse KNN, K=4 | 48h |
| `slurm/run_sparse_k16.slurm` | Gather-sparse KNN, K=16 (best TRA: 0.9972) | 48h |
| `slurm/run_sparse_k32.slurm` | Gather-sparse KNN, K=32 | 48h |
| `slurm/run_sparse_k16_ssl.slurm` | K16 + SSL-pretrained init | 48h |
| **`slurm/run_ssl_dino_pretrain.slurm`** | **DINOv2 + contrastive SSL pretraining (new)** | **24h** |
| `slurm/run_dist_ablation.slurm` | Distortion family ablation | 2h |

## Configs (`configs/`)

| File | For |
|------|-----|
| `configs/vanvliet_baseline.yaml` | Dense training (d_model=320, 6L+6L, window=4) |
| `configs/vanvliet_sparse_k16.yaml` | KNN=16 training |
| `configs/vanvliet_sparse_k16_ssl.yaml` | K16 + SSL pretraining |

> Note: the `cluster_configs/` variants and the phase-1 / quick-bench / diag slurm scripts
> were removed as obsolete (2026-08-05); the current K-sweep configs and the multi-seed
> harness live in the `bench-transformer-cell-tracking/benchmark_combined/` repo.

## Results summary (vanvliet)

| Model | TRA | AOGM | Edge F1 | Epochs | Time |
|-------|-----|------|---------|--------|------|
| Baseline (dense) | 0.9963 | 51.0 | 0.9913 | 372 | 606m |
| K=4 | 0.9957 | 62.0 | 0.9914 | 237 | 381m |
| **K=16** | **0.9972** | **37.7** | **0.9929** | **406** | **661m** |
| K=32 | 0.9963 | 52.6 | 0.9914 | 362 | 591m |
| K=64 | 0.9969 | 39.8 | 0.9926 | 366 | 599m |

## Analysis scripts (`benchmarks/` and `analysis/`)

| Script | Purpose |
|--------|---------|
| `benchmarks/benchmark_combined.py` | Speed + memory benchmark (synthetic Q/K/V) |
| `benchmarks/benchmark_speed_mem.py` | Detailed throughput/memory measurement |
| `benchmarks/downstream_matched.py` | SSL-pretrained vs random init on tracking pairs |
| `benchmarks/ssl_realdata.py` | SSL downstream evaluation on real data |
| `benchmarks/ssl_reinvestigation.py` | SSL post-mortem analysis |
| `benchmarks/ablation_full.py` | KNN ablation study (K=-1,4,16,32,64) |
| `analysis/plot_ablation.py` | Ablation result plots |
| `analysis/plot_combined.py` | Combined benchmark plots |
| `analysis/plot_comprehensive.py` | Comprehensive comparison plots |

## HPC workflow

```bash
# 1. SSH to cluster
ssh capella

# 2. Pull latest code (two repos)
cd ~/bench-transformer-cell-tracking && git pull origin cached-dist-attn
cd ~/trackastra && git pull origin cached-dist-attn

# 3. Authenticate with wandb
wandb login

# 4. Submit job
sbatch benchmark_training/slurm/run_ssl_dino_pretrain.slurm

# 5. Monitor
squeue -u $USER
tail -f logs/slurm-ssl_dino-<JOB_ID>.out
```

## Pre-submission validation

Run this locally before any `sbatch` to catch errors early:

```bash
cd ~/bench-transformer-cell-tracking
bash scripts/validate_submission.sh
```

Validates: syntax, model init, config types, import paths, end-to-end config+model instantiation.
