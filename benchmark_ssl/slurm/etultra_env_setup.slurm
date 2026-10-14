#!/usr/bin/env bash
#SBATCH --job-name=etultra_env
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x_%j.log
set -euo pipefail
export UV_LINK_MODE=copy
BENCH=/data/cat/ws/lest161c-ssl_cell_tracking/bench-transformer-cell-tracking
UV=/home/lest161c/.local/bin/uv
FORK_DIR="$BENCH/cyril_trackastra"
[ -d "$FORK_DIR/.git" ] || git clone https://github.com/C-Achard/trackastra.git "$FORK_DIR"
cd "$FORK_DIR" && git fetch origin && git checkout cy/pretrained_feats_train
"$UV" venv --clear --python 3.12 .venv-etultra
PY=.venv-etultra/bin/python
"$UV" pip install --python "$PY" --no-cache -e ".[pretrained_feats]"
"$UV" pip install --python "$PY" "traccuracy==0.3.0" motmetrics gitpython psutil seaborn "zarr<3"
"$UV" pip install --python "$PY" git+https://github.com/bentaculum/Trackastra-et-Ultra.git
"$PY" -c "import torch, zarr; print('torch', torch.__version__, 'zarr', zarr.__version__)"
"$PY" -c "from trackastra_pretrained_feats import FeatureExtractor; from trackastra_pretrained_feats.pretrained_features import AVAILABLE_PRETRAINED_BACKBONES as B; print('BACKBONES:', sorted(B))"
echo ETULTRA_ENV_OK
