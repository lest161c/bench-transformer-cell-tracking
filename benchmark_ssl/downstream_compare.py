"""Downstream comparison: SSL-pretrained vs random init.

Shows SSL pretraining leads to faster convergence on real tracking data.
Uses the SSL encoder directly + trains a simple association head on top.
"""

import csv, logging, sys, yaml, time, os
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from distortions import DistortionPipeline
from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl, features_from_frame
from track_encoder import AssociationEncoder, FeedForward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RealTrackingDataset(Dataset):
    """Loads real tracking windows from vanvliet data with ground truth assoc matrices."""

    def __init__(self, frames, window_size=6, ndim=2):
        self.frames = frames
        self.window_size = window_size
        self.ndim = ndim

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        _, _, frame_idx, mask_path, img_path = self.frames[idx]

        from tifffile import imread
        mask = imread(mask_path)
        img = imread(img_path).astype(np.float32)
        p1, p998 = np.percentile(img, (1, 99.8))
        img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

        result = features_from_frame(mask, img)
        if result is None:
            return {"coords": torch.zeros(0, self.ndim), "features": torch.zeros(0, 7),
                    "labels": torch.zeros(0, dtype=torch.long), "n": 0}

        coords, labels, feats_dict = result
        feats = np.concatenate(list(feats_dict.values()), axis=-1).astype(np.float32)

        return {
            "coords": torch.from_numpy(coords).float(),
            "features": torch.from_numpy(feats).float(),
            "labels": torch.from_numpy(labels).long(),
            "n": len(labels),
        }


class SimpleAssocHead(nn.Module):
    """Simple association head on top of encoder features."""

    def __init__(self, d_model=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_feats, tgt_feats):
        B, N_src, D = src_feats.shape
        N_tgt = tgt_feats.shape[1]

        src_exp = src_feats[:, :, None, :].expand(-1, -1, N_tgt, -1)
        tgt_exp = tgt_feats[:, None, :, :].expand(-1, N_src, -1, -1)
        pair = torch.cat([src_exp, tgt_exp], dim=-1)

        logits = self.net(pair).squeeze(-1)
        return logits


class DownstreamModel(nn.Module):
    """Encoder + association head for tracking."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, coords_src, feats_src, coords_tgt, feats_tgt, pad_src=None, pad_tgt=None):
        # Use encoder's inner representations
        f_src = self.encoder.feat_proj(feats_src)
        f_tgt = self.encoder.feat_proj(feats_tgt)
        c_src = self.encoder.coord_proj(coords_src)
        c_tgt = self.encoder.coord_proj(coords_tgt)

        pair = self.encoder.pair_norm(coords_src, coords_tgt)
        pair_enc = self.encoder.pair_proj(pair)

        ctx_src = pair_enc.mean(dim=2)
        ctx_tgt = pair_enc.mean(dim=1)

        src_feat = self.encoder.fusion(torch.cat([f_src, c_src, ctx_src], dim=-1))
        tgt_feat = self.encoder.fusion(torch.cat([f_tgt, c_tgt, ctx_tgt], dim=-1))

        # Encode
        both = torch.cat([src_feat, tgt_feat], dim=1)
        pad_both = torch.cat([pad_src, pad_tgt], dim=1) if pad_src is not None else None
        encoded = self.encoder.encoder(both, padding_mask=pad_both)

        N_src = coords_src.shape[1]
        src_enc = encoded[:, :N_src]
        tgt_enc = encoded[:, N_src:]

        return self.head(src_enc, tgt_enc)


def create_real_batches(frames, batch_size=4, max_tokens=2048):
    """Create batches from real tracking data using adjacent frames."""
    batches = []
    for i in range(0, len(frames) - 1, 2):
        try:
            src = frames[i]
            tgt = frames[i + 1]

            _, _, _, mask_path_src, img_path_src = src
            _, _, _, mask_path_tgt, img_path_tgt = tgt

            from tifffile import imread
            mask_src, mask_tgt = imread(mask_path_src), imread(mask_path_tgt)
            img_src = np.clip((imread(img_path_src).astype(np.float32) - np.percentile(imread(img_path_src).astype(np.float32), 1)) /
                              (np.percentile(imread(img_path_src).astype(np.float32), 99.8) - np.percentile(imread(img_path_src).astype(np.float32), 1) + 1e-8), 0, 1)
            img_tgt = np.clip((imread(img_path_tgt).astype(np.float32) - np.percentile(imread(img_path_tgt).astype(np.float32), 1)) /
                              (np.percentile(imread(img_path_tgt).astype(np.float32), 99.8) - np.percentile(imread(img_path_tgt).astype(np.float32), 1) + 1e-8), 0, 1)

            r_src = features_from_frame(mask_src, img_src)
            r_tgt = features_from_frame(mask_tgt, img_tgt)
            if r_src is None or r_tgt is None:
                continue

            cs, ls, fs_d = r_src
            ct, lt, ft_d = r_tgt

            # Build assoc: match labels across frames (simplified: label persistence)
            fs = np.concatenate(list(fs_d.values()), axis=-1).astype(np.float32)
            ft = np.concatenate(list(ft_d.values()), axis=-1).astype(np.float32)
            n_src, n_tgt = len(ls), len(lt)
            assoc = np.zeros((n_src, n_tgt), dtype=np.float32)
            label_to_tgt = {int(l): j for j, l in enumerate(lt)}
            for i_s, l_s in enumerate(ls):
                j = label_to_tgt.get(int(l_s))
                if j is not None:
                    assoc[i_s, j] = 1.0

            if n_src == 0 or n_tgt == 0:
                continue
            batches.append({
                "coords_src": torch.from_numpy(cs).float(),
                "coords_tgt": torch.from_numpy(ct).float(),
                "features_src": torch.from_numpy(fs).float(),
                "features_tgt": torch.from_numpy(ft).float(),
                "assoc_matrix": torch.from_numpy(assoc).float(),
                "padding_mask_src": torch.zeros(n_src, dtype=torch.bool),
                "padding_mask_tgt": torch.zeros(n_tgt, dtype=torch.bool),
                "n_src": n_src, "n_tgt": n_tgt,
            })

            if len(batches) >= batch_size * 10:
                break
        except Exception:
            continue

    logger.info(f"Created {len(batches)} real tracking pairs")
    return batches


def collate_downstream(batch):
    # Filter out empty entries
    batch = [b for b in batch if b["n_src"] > 0 and b["n_tgt"] > 0]
    if len(batch) == 0:
        return None
    max_src = max(b["n_src"] for b in batch)
    max_tgt = max(b["n_tgt"] for b in batch)
    B = len(batch)

    cs = torch.zeros(B, max_src, 2)
    ct = torch.zeros(B, max_tgt, 2)
    fs = torch.zeros(B, max_src, 7)
    ft = torch.zeros(B, max_tgt, 7)
    a = torch.zeros(B, max_src, max_tgt)
    ps = torch.ones(B, max_src, dtype=torch.bool)
    pt = torch.ones(B, max_tgt, dtype=torch.bool)

    for i, b in enumerate(batch):
        ns, nt = b["n_src"], b["n_tgt"]
        if ns > 0:
            cs[i, :ns] = b["coords_src"]; fs[i, :ns] = b["features_src"]; ps[i, :ns] = False
        if nt > 0:
            ct[i, :nt] = b["coords_tgt"]; ft[i, :nt] = b["features_tgt"]; pt[i, :nt] = False
        if ns > 0 and nt > 0:
            a[i, :ns, :nt] = b["assoc_matrix"]

    return {"coords_src": cs, "coords_tgt": ct, "features_src": fs, "features_tgt": ft,
            "assoc_matrix": a, "padding_mask_src": ps, "padding_mask_tgt": pt,
            "n_src": torch.tensor([b["n_src"] for b in batch]),
            "n_tgt": torch.tensor([b["n_tgt"] for b in batch])}


def train_compare(ssl_ckpt_path, config_path="config.yaml", n_epochs=15):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    frames = load_experiment_frames(cfg["data_root"], cfg.get("conditions"))
    np.random.seed(42)
    np.random.shuffle(frames)
    n_val = max(1, int(len(frames) * 0.1))
    train_frames, val_frames = frames[n_val:], frames[:n_val]

    train_batches = create_real_batches(train_frames, batch_size=cfg["training"]["batch_size"])
    val_batches = create_real_batches(val_frames, batch_size=cfg["training"]["batch_size"])

    # Models
    enc_cfg = cfg.get("encoder", {})
    ssl_encoder = AssociationEncoder(
        feat_dim=7, coord_dim=2,
        d_model=enc_cfg.get("d_model", 128),
        nhead=enc_cfg.get("nhead", 4),
        num_layers=enc_cfg.get("num_layers", 4),
        dim_feedforward=enc_cfg.get("dim_feedforward", 256),
        dropout=enc_cfg.get("dropout", 0.1),
    )

    # Load SSL weights
    if ssl_ckpt_path and os.path.exists(ssl_ckpt_path):
        ckpt = torch.load(ssl_ckpt_path, map_location="cpu", weights_only=False)
        ssl_encoder.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded SSL checkpoint from {ssl_ckpt_path}")

    head = SimpleAssocHead(d_model=enc_cfg.get("d_model", 128))

    # Two models
    model_ssl = DownstreamModel(deepcopy(ssl_encoder), deepcopy(head)).to(device)
    model_rand = DownstreamModel(AssociationEncoder(
        feat_dim=7, coord_dim=2,
        d_model=enc_cfg.get("d_model", 128),
        nhead=enc_cfg.get("nhead", 4),
        num_layers=enc_cfg.get("num_layers", 4),
        dim_feedforward=enc_cfg.get("dim_feedforward", 256),
        dropout=enc_cfg.get("dropout", 0.1),
    ), deepcopy(head)).to(device)

    n_ssl = sum(p.numel() for p in model_ssl.parameters())
    logger.info(f"SSL model params: {n_ssl:,}")

    outdir = Path("runs") / "downstream_compare"
    outdir.mkdir(parents=True, exist_ok=True)

    # Training
    results = {"ssl": [], "rand": []}
    for label, model in [("ssl", model_ssl), ("rand", model_rand)]:
        logger.info(f"\n{'='*60}\nTraining {label.upper()} model\n{'='*60}")
        optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        pos_weight = torch.tensor(10.0, device=device)

        for epoch in range(1, n_epochs + 1):
            model.train()
            train_losses, train_accs = [], []
            # Collate into mini-batches of 4
            for i in range(0, len(train_batches), 4):
                batch_items = train_batches[i:i+4]
                batch = collate_downstream(batch_items)
                if batch is None:
                    continue
                cs, ct = batch["coords_src"].to(device), batch["coords_tgt"].to(device)
                fs, ft = batch["features_src"].to(device), batch["features_tgt"].to(device)
                a = batch["assoc_matrix"].to(device)
                ps, pt = batch["padding_mask_src"].to(device), batch["padding_mask_tgt"].to(device)

                optimizer.zero_grad()
                logits = model(cs, fs, ct, ft, ps, pt)
                loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pos_weight)
                valid = ~(ps[:, :, None] | pt[:, None, :])
                loss = (loss * valid.float()).sum() / valid.float().sum()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())
                pred = (logits.detach() > 0).float()
                acc = ((pred == a).float() * valid.float()).sum() / valid.float().sum()
                train_accs.append(acc.item())

            # Val
            model.eval()
            val_losses, val_accs = [], []
            with torch.no_grad():
                for i in range(0, len(val_batches), 4):
                    batch_items = val_batches[i:i+4]
                    batch = collate_downstream(batch_items)
                    if batch is None:
                        continue
                    cs, ct = batch["coords_src"].to(device), batch["coords_tgt"].to(device)
                    fs, ft = batch["features_src"].to(device), batch["features_tgt"].to(device)
                    a = batch["assoc_matrix"].to(device)
                    ps, pt = batch["padding_mask_src"].to(device), batch["padding_mask_tgt"].to(device)
                    logits = model(cs, fs, ct, ft, ps, pt)
                    loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pos_weight)
                    valid = ~(ps[:, :, None] | pt[:, None, :])
                    loss = (loss * valid.float()).sum() / valid.float().sum()
                    pred = (logits > 0).float()
                    acc = ((pred == a).float() * valid.float()).sum() / valid.float().sum()
                    val_losses.append(loss.item())
                    val_accs.append(acc.item())

            tloss, tacc = np.mean(train_losses), np.mean(train_accs)
            vloss, vacc = np.mean(val_losses), np.mean(val_accs)
            results[label].append({"epoch": epoch, "train_loss": tloss, "train_acc": tacc,
                                   "val_loss": vloss, "val_acc": vacc})
            logger.info(f"{label.upper()} Epoch {epoch}: train_loss={tloss:.4f} train_acc={tacc:.4f} "
                         f"val_loss={vloss:.4f} val_acc={vacc:.4f}")

    # Save results CSV
    csv_path = outdir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "model", "train_loss", "train_acc", "val_loss", "val_acc"])
        for label in ["ssl", "rand"]:
            for r in results[label]:
                w.writerow([r["epoch"], label, f"{r['train_loss']:.6f}", f"{r['train_acc']:.4f}",
                           f"{r['val_loss']:.6f}", f"{r['val_acc']:.4f}"])

    logger.info(f"Results saved to {csv_path}")
    return results


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ssl_v1/best_model.pt"
    train_compare(ckpt)
