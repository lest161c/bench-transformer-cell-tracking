#!/bin/bash
#SBATCH --job-name=phase1_scaling_20260524_003147
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/runs/phase1/jobA_%j.out
#SBATCH --error=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/runs/phase1/jobA_%j.err

source ~/miniconda3/bin/activate trackastra 2>/dev/null || true
cd "/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/benchmark_ssl"
python "/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/runs/phase1/jobA_scaling.py"
echo "Job A done."
