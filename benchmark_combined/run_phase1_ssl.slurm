#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=phs1_ssl
#SBATCH --output=logs/slurm-ssl-%j.out
#SBATCH --error=logs/slurm-ssl-%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G

export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export WANDB_DISABLED=true
export WANDB_MODE=disabled
unset PYTHONHOME

eval "$(conda shell.bash hook)"
TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV_DIR="$TRK/trackastra_env"
WS=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601
DATA_DIR=/data/cat/ws/mawe985g-data/data/celltracking

conda activate "$ENV_DIR" 2>/dev/null || {
    echo "Creating env..."
    conda create --prefix "$ENV_DIR" python=3.10 -y
    conda activate "$ENV_DIR"
}
python -m pip install --upgrade --force-reinstall setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib pyyaml

cd "$TRK"
git pull origin feature/sparse-attention-gather
python -m pip install -e .

# Pull benchmark repo for the SSL pretraining pipeline
BM_DIR="$WS/bench-transformer-cell-tracking"
if [ -d "$BM_DIR" ]; then
    cd "$BM_DIR"
    git pull origin master
else
    git clone git@github.com:lest161c/bench-transformer-cell-tracking.git "$BM_DIR"
fi

mkdir -p logs
cd "$BM_DIR/benchmark_ssl"
mkdir -p runs

cat > config_cluster.yaml << 'YEOF'
name: ssl_phase1
data_root: DATA_DIR_PLACEHOLDER
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
