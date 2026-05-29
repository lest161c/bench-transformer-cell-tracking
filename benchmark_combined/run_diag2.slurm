#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=diag2
#SBATCH --output=logs/slurm-diag2-%j.out
#SBATCH --error=logs/slurm-diag2-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
. "$TRK/.venv/bin/activate"
WS=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601
BM_DIR="$WS/bench-transformer-cell-tracking"
cd "$BM_DIR" && git pull origin master 2>/dev/null
cd "$BM_DIR/benchmark_ssl"

python -c '
import torch, numpy as np, sys
sys.path.insert(0, ".")
from track_encoder import CellEmbedder
from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl
from distortions import DistortionPipeline

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

frames = load_experiment_frames("/data/cat/ws/mawe985g-data/data/celltracking/vanvliet", conditions=["rpsM"])
print(f"Frames: {len(frames)}")

cfg = {
    "distortions": ["jitter","dropout","feature_noise"],
    "jitter": {"std":[2,8],"p_cell_jitter":0.8},
    "dropout": {"p_drop":[0.05,0.2]},
    "feature_noise": {"std":[0.02,0.1]},
    "seed": 42,
}
dist = DistortionPipeline.from_config(cfg)
ds = SSLDataset(frames[:50], distortion_pipeline=dist, ndim=2)
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_ssl)

batch = next(iter(loader))
c1, c2 = batch["coords1"].to(device), batch["coords2"].to(device)
f1, f2 = batch["features1"].to(device), batch["features2"].to(device)
pm1, pm2 = batch["padding_mask1"].to(device), batch["padding_mask2"].to(device)
vp = batch["valid_pair"]
print(f"Batch: N_max={c1.shape[1]}, total_cells={vp.sum().item()}")

# Random init
model = CellEmbedder(feat_dim=7, coord_dim=2, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1).to(device)
with torch.no_grad():
    z1 = model(c1, f1, pm1)
    z2 = model(c2, f2, pm2)
    z1_n = torch.nn.functional.normalize(z1, dim=-1)
    z2_n = torch.nn.functional.normalize(z2, dim=-1)

print("\n=== RANDOM INIT ===")
all_z = []
for b in range(c1.shape[0]):
    v = ~pm1[b]; n = v.sum().item()
    if n < 3: continue
    e = z1_n[b][v]
    sim = e @ e.T
    inter = sim[~torch.eye(n,dtype=torch.bool,device=device)].mean().item()
    vp2 = ~pm1[b] & ~pm2[b]
    pos = (z1_n[b][vp2]*z2_n[b][vp2]).sum(dim=-1).mean().item() if vp2.sum()>0 else 0
    print(f"  Frame {b}: n={n:3d} inter_cell_sim={inter:.4f} pos_pair_sim={pos:.4f}")
    all_z.append(z1_n[b][v])

# Collapse test: all embeddings similar?
all_z_cat = torch.cat(all_z, dim=0)
M = all_z_cat.shape[0]
all_sim = all_z_cat @ all_z_cat.T
avg_inter = all_sim[~torch.eye(M,dtype=torch.bool,device=device)].mean().item()
print(f"Global inter-sim (random): {avg_inter:.4f} (0=orthogonal, 1=collapsed)")

# SSL checkpoint
ckpt = torch.load("runs/ssl_phase1/best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
print(f"\n=== SSL CHECKPOINT (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}) ===")

with torch.no_grad():
    z1 = model(c1, f1, pm1)
    z2 = model(c2, f2, pm2)
    z1_n = torch.nn.functional.normalize(z1, dim=-1)
    z2_n = torch.nn.functional.normalize(z2, dim=-1)

all_z = []
for b in range(c1.shape[0]):
    v = ~pm1[b]; n = v.sum().item()
    if n < 3: continue
    e = z1_n[b][v]
    sim = e @ e.T
    inter = sim[~torch.eye(n,dtype=torch.bool,device=device)].mean().item()
    vp2 = ~pm1[b] & ~pm2[b]
    pos = (z1_n[b][vp2]*z2_n[b][vp2]).sum(dim=-1).mean().item() if vp2.sum()>0 else 0
    print(f"  Frame {b}: n={n:3d} inter_cell_sim={inter:.4f} pos_pair_sim={pos:.4f}")
    all_z.append(z1_n[b][v])

all_z_cat = torch.cat(all_z, dim=0)
M = all_z_cat.shape[0]
all_sim = all_z_cat @ all_z_cat.T
avg_inter = all_sim[~torch.eye(M,dtype=torch.bool,device=device)].mean().item()
print(f"Global inter-sim (SSL): {avg_inter:.4f} (0=orthogonal, 1=collapsed)")

# Also check: SSL vs random embedding variance
print("\n=== VARIANCE ===")
model_rand = CellEmbedder(feat_dim=7, coord_dim=2, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1).to(device)
with torch.no_grad():
    zr = model_rand(c1, f1, pm1)
    zr_n = torch.nn.functional.normalize(zr, dim=-1)
    zs = model(c1, f1, pm1)
    zs_n = torch.nn.functional.normalize(zs, dim=-1)
real = ~pm1
print(f"Random emb std per dim: {zr_n[real].std(dim=0).mean():.6f}")
print(f"SSL emb std per dim:    {zs_n[real].std(dim=0).mean():.6f}")
print("DONE")
' 2>&1
echo "Job done"
