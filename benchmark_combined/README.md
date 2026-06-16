# benchmark_combined — Full-scale Training Benchmarks

SLURM submission scripts, configs, and analysis for Trackastra training runs on the HPC cluster (Capella, H100 GPUs).

## Slurm Scripts

| Script | Description | Est. time |
|--------|-------------|-----------|
| `run_baseline.slurm` | Dense Trackastra training (baseline) | 48h |
| `run_cached_dist.slurm` | CachedDistAttention variant (amortized cdist) | 48h |
| `run_sparse_k4.slurm` | Gather-sparse KNN, K=4 | 48h |
| `run_sparse_k16.slurm` | Gather-sparse KNN, K=16 (best TRA: 0.9972) | 48h |
| `run_sparse_k32.slurm` | Gather-sparse KNN, K=32 | 48h |
| `run_sparse_k16_ssl.slurm` | K16 + SSL-pretrained init | 48h |
| `run_ssl_only.slurm` | SSL pretraining only (no fine-tuning) | 6h |
| **`run_ssl_dino_pretrain.slurm`** | **DINOv2 + contrastive SSL pretraining (new)** | **24h** |
| `run_phase1_scaling.slurm` | Attention speed/memory benchmark (synthetic) | 2h |
| `run_phase1_ssl.slurm` | SSL pretraining (phase 1) | 6h |
| `run_phase1_sweep.slurm` | Label-fraction downstream evaluation | 4h |
| `run_diag2.slurm` | Embedding collapse check | 10min |
| `run_dist_ablation.slurm` | Distortion family ablation | 2h |

## Configs

| File | For |
|------|-----|
| `vanvliet_baseline.yaml` | Dense training (d_model=320, 6L+6L, window=4) |
| `vanvliet_sparse_k16.yaml` | KNN=16 training |
| `vanvliet_sparse_k16_ssl.yaml` | K16 + SSL pretraining |
| `cluster_configs/vanvliet_baseline.yaml` | Same, for cluster submission |
| `cluster_configs/vanvliet_sparse_k16.yaml` | Same, K=16 |
| `cluster_configs/vanvliet_sparse_k16_ssl.yaml` | Same, K16+SSL |

## Results summary (vanvliet)

| Model | TRA | AOGM | Edge F1 | Epochs | Time |
|-------|-----|------|---------|--------|------|
| Baseline (dense) | 0.9963 | 51.0 | 0.9913 | 372 | 606m |
| K=4 | 0.9957 | 62.0 | 0.9914 | 237 | 381m |
| **K=16** | **0.9972** | **37.7** | **0.9929** | **406** | **661m** |
| K=32 | 0.9963 | 52.6 | 0.9914 | 362 | 591m |
| K=64 | 0.9969 | 39.8 | 0.9926 | 366 | 599m |

## Analysis scripts

| Script | Purpose |
|--------|---------|
| `benchmark_combined.py` | Speed + memory benchmark (synthetic Q/K/V) |
| `benchmark_speed_mem.py` | Detailed throughput/memory measurement |
| `downstream_matched.py` | SSL-pretrained vs random init on tracking pairs |
| `ssl_realdata.py` | SSL downstream evaluation on real data |
| `ssl_reinvestigation.py` | SSL post-mortem analysis |
| `ablation_full.py` | KNN ablation study (K=-1,4,16,32,64) |
| `plot_ablation.py` | Ablation result plots |
| `plot_combined.py` | Combined benchmark plots |
| `plot_comprehensive.py` | Comprehensive comparison plots |

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
sbatch benchmark_combined/run_ssl_dino_pretrain.slurm

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
