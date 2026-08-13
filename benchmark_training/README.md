# benchmark_training — Training Benchmarks for Trackastra

SLURM submission scripts, configs, and benchmark/analysis code for
Trackastra training runs on the HPC cluster (Capella, H100 GPUs) and
local benchmark experiments on an A500 laptop GPU.

## Scope

This repository benchmarks **training performance** (both resource
usage and evaluation results) of Trackastra with the new sparse
attention implementations. It does NOT contain isolated attention
benchmarks (those live in `benchmark_attn/`) or isolated SSL
proof-of-concept benchmarks (those live in `benchmark_ssl/`).

## Directory layout

```
benchmark_training/
├── src/                                # shared library
│   ├── models.py                       # positional encoding, dense / sparse encoders, KNN
│   ├── data.py                         # synthetic batch + real-vanvliet pair loaders
│   ├── ssl_transfer.py                 # dense + sparse SSL weight-transfer utilities
│   └── bench_utils.py                  # timing / memory measurement, CSV I/O
├── scripts/
│   ├── benchmarks/                     # training performance benchmarks
│   │   ├── benchmark_speed_mem.py      # N=8192 training resource benchmark
│   │   ├── benchmark_combined.py       # 2x2 factorial: sparse + SSL vs dense + rand
│   │   └── benchmark_ablation.py       # convergence ablation: attention × K × L
│   ├── analysis/                       # plot / report generators
│   │   ├── plot_ablation.py
│   │   ├── plot_combined.py
│   │   └── plot_comprehensive.py
│   └── slurm/                          # HPC submission scripts
│       ├── run_training.slurm          # unified training, --export=VARIANT=<name>
│       └── run_diagnostic.slurm        # smoke tests + quick-bench
├── configs/                            # training YAML configs
├── results/                            # generated CSVs and HTML reports (reproducible)
└── README.md                           # this file
```

## Hardware requirements

### A500 laptop GPU (local, 4 GB VRAM)

| Script | Runtime | What it measures |
|---|---|---|
| `benchmark_speed_mem.py` | ~10 min | Training step time + memory at N=8192 (B=1) |
| `benchmark_combined.py` | ~30 min | 2×2 factorial convergence: dense/sparse × random/SSL |
| `benchmark_ablation.py` | ~45 min | Convergence ablation: dense/sparseK4/sparseK16 × L=1,4 |

### H100 80 GB (HPC cluster, Capella)

Full-scale Trackastra training runs require an H100 GPU:

| Script | Runtime | Notes |
|---|---|---|
| `run_training.slurm` | 6–11 h | One K-sweep training run per variant |
| `run_diagnostic.slurm` | <5 min | Smoke test or quick-bench |

**Partition:** `gpu-h100`
**Account:** `p_scads_celltracking`

## Quick start

### Local benchmarks (A500)

```bash
# N=8192 training resource benchmark
python scripts/benchmarks/benchmark_speed_mem.py

# Combined 2x2 factorial convergence
python scripts/benchmarks/benchmark_combined.py --epochs 15

# Convergence ablation
python scripts/benchmarks/benchmark_ablation.py --epochs 15

# Generate HTML reports
python scripts/analysis/plot_ablation.py
python scripts/analysis/plot_combined.py
python scripts/analysis/plot_comprehensive.py
```

### HPC training runs (H100)

```bash
# Submit a training run (choose variant)
sbatch --export=VARIANT=baseline scripts/slurm/run_training.slurm
sbatch --export=VARIANT=sparse_k16 scripts/slurm/run_training.slurm

# Smoke test (environment setup only)
sbatch --export=VARIANT=baseline,TEST=1 scripts/slurm/run_training.slurm
```

## Configs

| File | Purpose |
|---|---|
| `configs/vanvliet_baseline.yaml` | Dense training (d_model=320, 6L+6L, window=4) |
| `configs/vanvliet_sparse_k16_ssl.yaml` | K=16 training with SSL pretraining stage |

## Results

The `results/` directory contains CSV data and HTML reports produced
by the benchmark and analysis scripts. All files are reproducible by
re-running the corresponding script.

## Training variants

The unified SLURM training script supports the following variants:

| `VARIANT` | Description | Branch |
|---|---|---|
| `baseline` | Dense (K=-1) Trackastra training | `feature/sparse-attention-gather` |
| `cached_dist` | CachedDistAttention training | `cached-dist-attn` |
| `sparse_k4` | Gather-sparse KNN, K=4 | `feature/sparse-attention-gather` |
| `sparse_k16` | Gather-sparse KNN, K=16 | `feature/sparse-attention-gather` |
| `sparse_k32` | Gather-sparse KNN, K=32 | `feature/sparse-attention-gather` |
| `sparse_k16_ssl` | K=16 + SSL pretraining stage | `feature/sparse-attention-gather` |

## Related repositories

| Repository | Purpose |
|---|---|
| `benchmark_attn/` | Isolated attention layer benchmarks (scaling, memory, speed) |
| `benchmark_ssl/` | Isolated SSL proof-of-concept benchmarks (downstream transfer quality) |
| `benchmark_training/` | This repo: actual Trackastra training performance benchmarks |
