"""SSL pretraining script.

Trains the association encoder on synthetic frame pairs.
Loss: binary cross-entropy on association matrix (identity matching).

Monitoring for overfitting:
- If association accuracy on held-out validation set diverges from training
  and starts memorizing distortion-specific patterns → overfitting.
- If encoder collapses to predicting all pairs as matching → degenerate.
- Check: validation accuracy on novel distortion types.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from distortions import DistortionPipeline
from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl
from track_encoder import AssociationEncoder

logger = logging.getLogger(__name__)


def assoc_accuracy(logits, targets, threshold=0.0):
    """Compute accuracy, precision, recall for association prediction.

    logits: (B, N_src, N_tgt)
    targets: (B, N_src, N_tgt) — binary
    """
    preds = (logits > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return acc, prec, rec, f1


class SSLLoss(nn.Module):
    """Binary cross-entropy with positive weighting for association matrix.

    Association matrices are sparse (~1% positive). Use focal-style weighting
    or pos_weight to handle class imbalance.
    """

    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets, padding_mask_src=None, padding_mask_tgt=None):
        # BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor(self.pos_weight, device=logits.device)
        )

        # Zero out padded positions (already handled by logits being masked)
        if padding_mask_src is not None and padding_mask_tgt is not None:
            valid = ~(padding_mask_src[:, :, None] | padding_mask_tgt[:, None, :])
            loss = (loss * valid.float()).sum() / valid.float().sum()

        return loss


def train_epoch(model, loader, optimizer, loss_fn, device, epoch, writer=None):
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    for batch in pbar:
        coords_src = batch["coords_src"].to(device)
        coords_tgt = batch["coords_tgt"].to(device)
        feats_src = batch["features_src"].to(device)
        feats_tgt = batch["features_tgt"].to(device)
        assoc = batch["assoc_matrix"].to(device)
        pad_src = batch["padding_mask_src"].to(device)
        pad_tgt = batch["padding_mask_tgt"].to(device)

        optimizer.zero_grad()

        logits = model(
            coords_src, feats_src,
            coords_tgt, feats_tgt,
            padding_mask_src=pad_src,
            padding_mask_tgt=pad_tgt,
        )

        loss = loss_fn(logits, assoc, pad_src, pad_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        acc, prec, rec, f1 = assoc_accuracy(logits.detach(), assoc)
        total_acc += acc.item()
        n_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.3f}"})

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    if writer:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/acc", avg_acc, epoch)

    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, loss_fn, device, epoch, writer=None):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_f1 = 0
    n_batches = 0

    for batch in tqdm(loader, desc=f"Val Epoch {epoch}", leave=False):
        coords_src = batch["coords_src"].to(device)
        coords_tgt = batch["coords_tgt"].to(device)
        feats_src = batch["features_src"].to(device)
        feats_tgt = batch["features_tgt"].to(device)
        assoc = batch["assoc_matrix"].to(device)
        pad_src = batch["padding_mask_src"].to(device)
        pad_tgt = batch["padding_mask_tgt"].to(device)

        logits = model(
            coords_src, feats_src,
            coords_tgt, feats_tgt,
            padding_mask_src=pad_src,
            padding_mask_tgt=pad_tgt,
        )

        loss = loss_fn(logits, assoc, pad_src, pad_tgt)
        acc, prec, rec, f1 = assoc_accuracy(logits, assoc)

        total_loss += loss.item()
        total_acc += acc.item()
        total_f1 += f1.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_f1 = total_f1 / n_batches

    if writer:
        writer.add_scalar("val/loss", avg_loss, epoch)
        writer.add_scalar("val/acc", avg_acc, epoch)
        writer.add_scalar("val/f1", avg_f1, epoch)

    logger.info(f"Val Epoch {epoch}: loss={avg_loss:.4f}, acc={avg_acc:.4f}, f1={avg_f1:.4f}")
    return avg_loss, avg_acc, avg_f1


def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    frames = load_experiment_frames(
        cfg.get("data_root", "../data/vanvliet"),
        conditions=cfg.get("conditions"),
    )
    logger.info(f"Total frames available: {len(frames)}")
    np.random.seed(42)
    np.random.shuffle(frames)

    n_val = max(1, int(len(frames) * cfg["training"]["val_split"]))
    val_frames = frames[:n_val]
    train_frames = frames[n_val:]
    logger.info(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    dist = DistortionPipeline.from_config(cfg)

    train_ds = SSLDataset(train_frames, distortion_pipeline=dist, ndim=cfg.get("ndim", 2))
    val_ds = SSLDataset(val_frames, distortion_pipeline=dist, ndim=cfg.get("ndim", 2))

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, collate_fn=collate_ssl, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_ssl, num_workers=0,
    )

    # Model
    feat_dim = train_ds[0]["features_src"].shape[-1] if len(train_ds) > 0 and train_ds[0]["features_src"].numel() > 0 else 7
    logger.info(f"Feature dimension: {feat_dim}")

    enc_cfg = cfg.get("encoder", {})
    model = AssociationEncoder(
        feat_dim=feat_dim,
        coord_dim=cfg.get("ndim", 2),
        d_model=enc_cfg.get("d_model", 128),
        nhead=enc_cfg.get("nhead", 4),
        num_layers=enc_cfg.get("num_layers", 4),
        dim_feedforward=enc_cfg.get("dim_feedforward", 256),
        dropout=enc_cfg.get("dropout", 0.1),
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    train_cfg = cfg["training"]
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    # Positive weight: roughly inverse of positive ratio (~1% -> weight ~100)
    # But start lower to avoid training instability
    loss_fn = SSLLoss(pos_weight=10.0)

    # Logging
    log_dir = Path("runs") / cfg.get("name", "ssl_v1")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    # Save config
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    best_val_loss = float("inf")
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, loss_fn, device, epoch, writer
        )

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": cfg,
            }
            torch.save(ckpt, log_dir / "best_model.pt")
            logger.info(f"Saved best model (epoch {epoch}, val_loss={val_loss:.4f})")

        # Periodic checkpoint
        if epoch % train_cfg.get("checkpoint_every", 10) == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                log_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    writer.close()
    logger.info(f"Training complete. Best model: {log_dir / 'best_model.pt'}")
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
