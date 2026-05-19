#!/bin/bash
# One-shot: nuke old env, create fresh in workspace, install everything
set -euo pipefail

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
ENV="$TRK/trackastra_env"

echo "=== Removing old env ==="
rm -rf "$ENV"

echo "=== Creating fresh env ==="
conda create --prefix "$ENV" python=3.10 -y

echo "=== Activating ==="
conda activate "$ENV"

echo "=== Python path check ==="
which python
python --version

echo "=== Installing deps ==="
python -m pip install setuptools
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install lightning pandas scikit-image tifffile edt tqdm configargparse wandb tensorboard dask joblib

echo "=== Installing trackastra ==="
cd "$TRK"
python -m pip install -e .

echo "=== Verification ==="
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import trackastra
from trackastra.model import TrackingTransformer
print(f'trackastra: OK')
print(f'Python: ', end='')
import sys; print(f'{sys.version[:5]}')
"
echo "=== ENV READY ==="
