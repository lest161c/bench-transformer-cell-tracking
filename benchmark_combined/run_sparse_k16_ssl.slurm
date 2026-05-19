#!/bin/bash
#SBATCH --job-name=tr_sparse_k16_ssl
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=90GB
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8

source $(conda info --base)/etc/profile.d/conda.sh
conda activate trackastra_env

cd /data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra

git pull origin feature/sparse-attention-gather

cat > /tmp/vanvliet_sparse_k16_ssl.yaml << 'CONFIG'
name: vanvliet_sparse_k16_ssl
ndim: 2
window: 10
epochs: 100
warmup_epochs: 10
detection_folders: [TRA]
input_train: [data/vanvliet/rpsM, data/vanvliet/recA, data/vanvliet/pheA, data/vanvliet/metA, data/vanvliet/cib, data/vanvliet/trpL]
input_val: [data/vanvliet/rpsM, data/vanvliet/recA, data/vanvliet/pheA]
outdir: runs
distributed: False
batch_size: 8
max_tokens: 8192
d_model: 256
num_encoder_layers: 4
num_decoder_layers: 4
nhead: 4
pos_embed_per_dim: 32
feat_embed_per_dim: 8
dropout: 0.01
lr: 1e-4
features: wrfeat
causal_norm: quiet_softmax
attn_positional_bias: rope
spatial_pos_cutoff: 256
delta_cutoff: 2
knn_neighbors: 16
mixedp: True
num_workers: 8
seed: 42
augment: 3
ssl_pretrain: True
ssl_epochs: 5
ssl_conditions: [rpsM, recA, pheA, metA, cib, trpL]
CONFIG

python scripts/train.py -c /tmp/vanvliet_sparse_k16_ssl.yaml
