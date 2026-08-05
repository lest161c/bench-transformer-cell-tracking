"""SSL pretraining with configurable attention (dense or sparse K)."""

import logging, yaml, time, math, gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark_ssl root (sibling modules)

from model_parts import GatherSparseAttention

from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl
from distortions import DistortionPipeline


# AgentCentricNormalization moved here (removed from track_encoder.py)
class AgentCentricNormalization(nn.Module):
    def __init__(self, ndim=2):
        super().__init__()
        self.ndim = ndim
    def forward(self, coords_src, coords_tgt):
        B, N_src, D = coords_src.shape
        _, N_tgt, _ = coords_tgt.shape
        disp = coords_tgt[:, None, :, :] - coords_src[:, :, None, :]
        dist = torch.norm(disp, dim=-1, keepdim=True)
        dir_vec = F.normalize(disp + 1e-8, dim=-1)
        return torch.cat([disp, dir_vec, dist], dim=-1)


# ============================================================
# Sparse AssociationEncoder (same as dense but with sparse transformer)
# ============================================================

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SparseTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1, knn_k=16):
        super().__init__()
        self.knn_k = knn_k
        self.pos_enc = SinusoidalPE(d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = GatherSparseAttention(d_model, nhead, knn_k, dropout=dropout, mode="none")
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation="gelu", batch_first=True, norm_first=True)
            self.layers.append(nn.ModuleDict({
                "attn": attn, "linear1": layer.linear1, "linear2": layer.linear2,
                "norm1": layer.norm1, "norm2": layer.norm2,
                "dropout": layer.dropout, "dropout1": layer.dropout1, "dropout2": layer.dropout2,
            }))
        self.norm = nn.LayerNorm(d_model)

    def compute_knn(self, coords):
        B, N, _ = coords.shape
        yx = coords[..., -2:]
        dist = torch.cdist(yx, yx)
        k = min(self.knn_k, N)
        _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
        if idx.shape[-1] < self.knn_k:
            pad = idx[:, :, -1:].expand(-1, -1, self.knn_k - idx.shape[-1])
            idx = torch.cat([idx, pad], dim=-1)
        return idx

    def forward(self, x, mask=None, coords=None):
        x = self.pos_enc(x)
        if coords is None:
            return self.norm(x)
        knn_idx = self.compute_knn(coords)
        for layer in self.layers:
            attn_out = layer["attn"](x, x, x, knn_idx, coords)
            x = layer["norm1"](x + layer["dropout1"](attn_out))
            ff = layer["linear2"](layer["dropout"](F.gelu(layer["linear1"](x))))
            x = layer["norm2"](x + layer["dropout2"](ff))
        return self.norm(x)


class DenseTransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.pos_enc = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, mask=None, coords=None):
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.norm(x)


class SSLModel(nn.Module):
    """SSL model with configurable attention. Matches architecture for downstream transfer."""
    def __init__(self, feat_dim=7, coord_dim=2, d_model=128, nhead=4,
                 num_layers=4, dim_feedforward=256, dropout=0.1, sparse_k=None):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.pair_norm = AgentCentricNormalization(ndim=coord_dim)
        self.pair_proj = nn.Linear(coord_dim * 2 + 1, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        if sparse_k is not None and sparse_k > 0:
            self.encoder = SparseTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout, knn_k=sparse_k)
        else:
            self.encoder = DenseTransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.head = nn.Linear(d_model * 2, 1)
        self.sparse_k = sparse_k

    def forward(self, cs, fs, ct, ft, ps=None, pt=None):
        f_src, f_tgt = self.feat_proj(fs), self.feat_proj(ft)
        c_src, c_tgt = self.coord_proj(cs), self.coord_proj(ct)
        pair = self.pair_norm(cs, ct)
        pair_enc = self.pair_proj(pair)
        ctx_src, ctx_tgt = pair_enc.mean(dim=2), pair_enc.mean(dim=1)
        src_feat = self.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))
        both = torch.cat([src_feat, tgt_feat], dim=1)
        pad_both = torch.cat([ps, pt], dim=1) if ps is not None else None
        coords_both = torch.cat([cs, ct], dim=1)
        encoded = self.encoder(both, mask=pad_both,
                               coords=coords_both if self.sparse_k else None)
        N_s = cs.shape[1]
        src_enc, tgt_enc = encoded[:, :N_s], encoded[:, N_s:]
        B, N1, D = src_enc.shape; N2 = tgt_enc.shape[1]
        src_e = src_enc[:, :, None, :].expand(-1, -1, N2, -1)
        tgt_e = tgt_enc[:, None, :, :].expand(-1, N1, -1, -1)
        return self.head(torch.cat([src_e, tgt_e], dim=-1)).squeeze(-1)


def pretrain(sparse_k, n_epochs=5, device="cuda"):
    """Pretrain SSL model with given attention type."""
    tag = "dense" if sparse_k is None else f"K={sparse_k}"
    logger.info(f"\n{'='*60}\nPretraining {tag}\n{'='*60}")

    # Data
    frames = load_experiment_frames(str(ROOT / "data/vanvliet"), conditions=["rpsM", "recA", "pheA"])
    np.random.seed(42); np.random.shuffle(frames)
    n_val = max(1, int(len(frames) * 0.1))
    val_frames, train_frames = frames[:n_val], frames[n_val:]
    logger.info(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    # Distortion pipeline (from config)
    with open(ROOT / "benchmark_ssl" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    dist = DistortionPipeline.from_config(cfg)

    train_ds = SSLDataset(train_frames, distortion_pipeline=dist, ndim=2)
    val_ds = SSLDataset(val_frames, distortion_pipeline=dist, ndim=2)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_ssl)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_ssl)

    # Model
    model = SSLModel(sparse_k=sparse_k).to(device)
    logger.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    pw = torch.tensor(10.0, device=device)

    outdir = ROOT / "benchmark_ssl" / "runs" / f"ssl_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    best_vl = float("inf")
    for epoch in range(1, n_epochs + 1):
        model.train()
        tls = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            cs, ct = batch["coords_src"].to(device), batch["coords_tgt"].to(device)
            fs, ft = batch["features_src"].to(device), batch["features_tgt"].to(device)
            a = batch["assoc_matrix"].to(device)
            ps, pt = batch["padding_mask_src"].to(device), batch["padding_mask_tgt"].to(device)
            opt.zero_grad()
            logits = model(cs, fs, ct, ft, ps, pt)
            loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pw)
            valid = ~(ps[:, :, None] | pt[:, None, :])
            loss = (loss * valid.float()).sum() / valid.float().sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tls.append(loss.item())

        # Val
        model.eval()
        vls, vas = [], []
        with torch.no_grad():
            for batch in val_loader:
                cs, ct = batch["coords_src"].to(device), batch["coords_tgt"].to(device)
                fs, ft = batch["features_src"].to(device), batch["features_tgt"].to(device)
                a = batch["assoc_matrix"].to(device)
                ps, pt = batch["padding_mask_src"].to(device), batch["padding_mask_tgt"].to(device)
                logits = model(cs, fs, ct, ft, ps, pt)
                loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pw)
                valid = ~(ps[:, :, None] | pt[:, None, :])
                loss = (loss * valid.float()).sum() / valid.float().sum()
                vls.append(loss.item())
                pred = (logits > 0).float()
                acc = ((pred == a).float() * valid.float()).sum() / valid.float().sum()
                vas.append(acc.item())

        tl, vl = float(np.mean(tls)), float(np.mean(vls))
        va = float(np.mean(vas))
        logger.info(f"  Ep {epoch:2d}: tl={tl:.4f} vl={vl:.4f} acc={va:.4f}")

        if vl < best_vl:
            best_vl = vl
            ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": vl,
                    "config": {"sparse_k": sparse_k, "d_model": 128, "nhead": 4, "num_layers": 4}}
            torch.save(ckpt, outdir / "best_model.pt")
            logger.info(f"  Saved best (epoch {epoch}, vl={vl:.4f})")

    return str(outdir / "best_model.pt")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    for k in [None, 4, 8, 16, 32]:
        pretrain(k, n_epochs=5, device=device)
        gc.collect()
        torch.cuda.empty_cache()
