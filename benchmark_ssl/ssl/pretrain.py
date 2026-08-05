"""SSL contrastive pretraining with TensorBoard logging.

Trains CellEmbedder using SimCLR-style contrastive learning (NT-Xent loss).
Two augmented views per frame → embeddings z1, z2.
Positive pairs: same cell across views. Negatives: all other cells.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark_ssl root (sibling modules)

from distortions import DistortionPipeline
from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl
from track_encoder import CellEmbedder

logger = logging.getLogger(__name__)


def nt_xent_loss(z1, z2, padding_mask1, padding_mask2, temperature=0.05):
    B, N_max, D = z1.shape
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    total_loss = 0.0
    n_frames = 0
    for b in range(B):
        valid = ~padding_mask1[b] & ~padding_mask2[b]
        n = valid.sum().item()
        if n < 2:
            continue
        e1 = z1[b][valid]
        e2 = z2[b][valid]
        features = torch.cat([e1, e2], dim=0)
        sim = torch.matmul(features, features.T) / temperature
        sim = sim - torch.eye(2 * n, device=sim.device) * 1e9
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(features.device)
        loss = F.cross_entropy(sim, labels, reduction="mean")
        total_loss += loss
        n_frames += 1
    if n_frames == 0:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)
    return total_loss / n_frames


def embedding_consistency(z1, z2, padding_mask1, padding_mask2):
    valid = ~padding_mask1 & ~padding_mask2
    if valid.sum() == 0:
        return 0.0
    z1_n = F.normalize(z1, dim=-1)
    z2_n = F.normalize(z2, dim=-1)
    return (z1_n * z2_n).sum(dim=-1)[valid].mean().item()


def train_epoch(model, loader, optimizer, device, epoch, temperature, writer=None):
    model.train()
    total_loss = 0.0
    total_cons = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        c1 = batch["coords1"].to(device)
        c2 = batch["coords2"].to(device)
        f1 = batch["features1"].to(device)
        f2 = batch["features2"].to(device)
        pm1 = batch["padding_mask1"].to(device)
        pm2 = batch["padding_mask2"].to(device)

        optimizer.zero_grad()
        z1 = model(c1, f1, pm1)
        z2 = model(c2, f2, pm2)
        loss = nt_xent_loss(z1, z2, pm1, pm2, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cons += embedding_consistency(z1.detach(), z2.detach(), pm1, pm2)
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(n_batches, 1)
    avg_cons = total_cons / max(n_batches, 1)

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/consistency", avg_cons, epoch)

    return avg_loss, avg_cons


@torch.no_grad()
def validate(model, loader, device, epoch, temperature, writer=None):
    model.eval()
    total_loss = 0.0
    total_cons = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=f"Val Epoch {epoch}", leave=False):
        if batch is None:
            continue
        c1 = batch["coords1"].to(device)
        c2 = batch["coords2"].to(device)
        f1 = batch["features1"].to(device)
        f2 = batch["features2"].to(device)
        pm1 = batch["padding_mask1"].to(device)
        pm2 = batch["padding_mask2"].to(device)

        z1 = model(c1, f1, pm1)
        z2 = model(c2, f2, pm2)
        loss = nt_xent_loss(z1, z2, pm1, pm2, temperature)

        total_loss += loss.item()
        total_cons += embedding_consistency(z1, z2, pm1, pm2)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_cons = total_cons / max(n_batches, 1)

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/consistency", avg_cons, epoch)

    logger.info(f"Val Epoch {epoch}: loss={avg_loss:.4f}, consistency={avg_cons:.4f}")
    return avg_loss, avg_cons


def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    frames = load_experiment_frames(cfg.get("data_root", "../data/vanvliet"), conditions=cfg.get("conditions"))
    logger.info(f"Total frames: {len(frames)}")
    np.random.seed(42)
    np.random.shuffle(frames)
    n_val = max(1, int(len(frames) * cfg["training"]["val_split"]))
    val_frames, train_frames = frames[:n_val], frames[n_val:]
    logger.info(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    dist = DistortionPipeline.from_config(cfg)
    train_ds = SSLDataset(train_frames, distortion_pipeline=dist, ndim=cfg.get("ndim", 2))
    val_ds = SSLDataset(val_frames, distortion_pipeline=dist, ndim=cfg.get("ndim", 2))

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_ssl, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, collate_fn=collate_ssl, num_workers=0)

    enc_cfg = cfg.get("encoder", {})
    ssl_cfg = cfg.get("ssl", {})
    temperature = ssl_cfg.get("temperature", 0.05)

    model = CellEmbedder(
        feat_dim=7, coord_dim=cfg.get("ndim", 2),
        d_model=enc_cfg.get("d_model", 128),
        nhead=enc_cfg.get("nhead", 4),
        num_layers=enc_cfg.get("num_layers", 4),
        dim_feedforward=enc_cfg.get("dim_feedforward", 256),
        dropout=enc_cfg.get("dropout", 0.1),
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    train_cfg = cfg["training"]
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    log_dir = Path("runs") / cfg.get("name", "ssl_contrastive")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    best_val_loss = float("inf")
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_cons = train_epoch(model, train_loader, optimizer, device, epoch, temperature, writer)
        val_loss, val_cons = validate(model, val_loader, device, epoch, temperature, writer)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f} train_cons={train_cons:.4f} "
                     f"val_loss={val_loss:.4f} val_cons={val_cons:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss, "val_consistency": val_cons, "config": cfg}
            torch.save(ckpt, log_dir / "best_model.pt")
            logger.info(f"Saved best model (epoch {epoch}, val_loss={val_loss:.4f})")

        if epoch % train_cfg.get("checkpoint_every", 10) == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                       log_dir / f"checkpoint_epoch_{epoch}.pt")

    writer.close()
    logger.info(f"Training complete. Best model: {log_dir / 'best_model.pt'}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
