#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=phs1_scale
#SBATCH --output=logs/slurm-scale-%j.out
#SBATCH --error=logs/slurm-scale-%j.err
#SBATCH --time=2:00:00
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

# Pull benchmark repo for the scaling script
BM_DIR="$WS/bench-transformer-cell-tracking"
if [ -d "$BM_DIR" ]; then
    cd "$BM_DIR"
    git pull origin master
else
    git clone git@github.com:lest161c/bench-transformer-cell-tracking.git "$BM_DIR"
fi

mkdir -p logs
cd "$BM_DIR/benchmark_ssl"

python - << 'PYEOF'
"""Synthetic attention scaling benchmark on H100.

Sweeps N ∈ [128, 256, 512, 1024, 2048, 4096, 8192], K ∈ [4, 16, 32, 64],
L ∈ [1, 6, 12]. Dense baseline at each N (OOMs at N≥4096).
FP16, measures time + peak memory.

Output: runs/phase1/scaling_h100.csv
"""
import torch, time, csv
from pathlib import Path

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

d_model, nhead, B, warmup, repeat = 256, 4, 2, 5, 10
head_dim = d_model // nhead
Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
Ls = [1, 6, 12]
Ks = [4, 16, 32, 64]
results = []

def sparse_attn(q, k, v, knn_idx):
    """KNN gather: expand q to (B,N,1,d), k_sel to (B,N,K,d) for FA."""
    B, N, K = knn_idx.shape
    d = q.shape[-1]
    idx_flat = knn_idx.view(B, -1)
    k_sel = torch.gather(k, 1, idx_flat.unsqueeze(-1).expand(-1,-1,d)).view(B,N,K,d)
    v_sel = torch.gather(v, 1, idx_flat.unsqueeze(-1).expand(-1,-1,d)).view(B,N,K,d)
    out = torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(2), k_sel, v_sel)
    return out.squeeze(2)

for L in Ls:
    for N in Ns:
        # Dense baseline
        if N <= 4096:
            try:
                torch.cuda.empty_cache()
                coords = torch.randn(N, 2, device=device, dtype=torch.float16)
                dist = torch.cdist(coords, coords)
                attn_mask = torch.zeros(N, N, device=device, dtype=torch.float16)
                attn_mask[dist > 100] = -65504.0  # fp16 max negative, avoids overflow

                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for _ in range(warmup):
                        x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                        for _ in range(L):
                            q = k = v = x
                            scores = torch.matmul(q, k.transpose(-2,-1)) / (head_dim**0.5) + attn_mask
                            x = torch.matmul(torch.softmax(scores, dim=-1), v)
                torch.cuda.synchronize(); mem = torch.cuda.max_memory_allocated() / (1024**2)
                torch.cuda.empty_cache()

                t0 = time.perf_counter()
                for _ in range(repeat):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    for _ in range(L):
                        q = k = v = x
                        scores = torch.matmul(q, k.transpose(-2,-1)) / (head_dim**0.5) + attn_mask
                        x = torch.matmul(torch.softmax(scores, dim=-1), v)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / repeat * 1000
                results.append({"N":N,"L":L,"K":"dense","variant":"masked",
                               "time_ms_per_step":f"{dt:.3f}","mem_mb":f"{mem:.1f}"})
                print(f"  DENSE  N={N:5d} L={L:2d}: {dt:8.3f}ms  {mem:8.1f}MB")
            except RuntimeError as e:
                results.append({"N":N,"L":L,"K":"dense","variant":"masked",
                               "time_ms_per_step":"ERR","mem_mb":"ERR"})
                print(f"  DENSE  N={N:5d} L={L:2d}: ERR: {str(e)[:100]}")

        # Sparse KNN
        for K in Ks:
            if K >= N: continue
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for _ in range(warmup):
                        x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                        coords = torch.randn(B, N, 2, device=device, dtype=torch.float16)
                        d2 = torch.cdist(coords, coords)
                        _, kidx = torch.topk(d2, k=K, dim=-1, largest=False)
                        for _ in range(L):
                            x = sparse_attn(x, x, x, kidx)
                torch.cuda.synchronize(); mem = torch.cuda.max_memory_allocated() / (1024**2)
                torch.cuda.empty_cache()

                t0 = time.perf_counter()
                for _ in range(repeat):
                    x = torch.randn(B, N, d_model, device=device, dtype=torch.float16)
                    coords = torch.randn(B, N, 2, device=device, dtype=torch.float16)
                    d2 = torch.cdist(coords, coords)
                    _, kidx = torch.topk(d2, k=K, dim=-1, largest=False)
                    for _ in range(L):
                        x = sparse_attn(x, x, x, kidx)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / repeat * 1000
                results.append({"N":N,"L":L,"K":f"K={K}","variant":"knn_gather",
                               "time_ms_per_step":f"{dt:.3f}","mem_mb":f"{mem:.1f}"})
                print(f"  KNN K={K:2d} N={N:5d} L={L:2d}: {dt:8.3f}ms  {mem:8.1f}MB")
            except RuntimeError as e:
                results.append({"N":N,"L":L,"K":f"K={K}","variant":"knn_gather",
                               "time_ms_per_step":"ERR","mem_mb":"ERR"})
                print(f"  KNN K={K:2d} N={N:5d} L={L:2d}: ERR: {str(e)[:100]}")
        print()

Path("runs/phase1").mkdir(parents=True, exist_ok=True)
with open("runs/phase1/scaling_h100.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["N","L","K","variant","time_ms_per_step","mem_mb"])
    w.writeheader(); w.writerows(results)
print("\nResults: runs/phase1/scaling_h100.csv")
print("DONE")
PYEOF

echo "Job A (scaling) done."
