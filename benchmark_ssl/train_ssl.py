"""Standalone SSL training script. Logs metrics to CSV for plotting."""

import csv, logging, sys, yaml, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from distortions import DistortionPipeline
from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl
from track_encoder import AssociationEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def assoc_accuracy(logits, targets):
    preds = (logits > 0).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return acc.item(), prec.item(), rec.item(), f1.item()


def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    outdir = Path("runs") / cfg.get("name", "ssl_bench")
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Data
    frames = load_experiment_frames(cfg["data_root"], cfg.get("conditions"))
    logger.info(f"Total frames: {len(frames)}")
    np.random.seed(42)
    np.random.shuffle(frames)
    n_val = max(1, int(len(frames) * cfg["training"]["val_split"]))
    val_frames, train_frames = frames[:n_val], frames[n_val:]
    logger.info(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    dist = DistortionPipeline.from_config(cfg)
    train_ds = SSLDataset(train_frames, distortion_pipeline=dist, ndim=cfg.get("ndim", 2))
    val_ds = SSLDataset(val_frames, distortion_pipeline=dist, ndim=cfg.get("ndim", 2))

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_ssl)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_ssl)

    feat_dim = 7
    enc_cfg = cfg.get("encoder", {})
    model = AssociationEncoder(
        feat_dim=feat_dim, coord_dim=cfg.get("ndim", 2),
        d_model=enc_cfg.get("d_model", 128),
        nhead=enc_cfg.get("nhead", 4),
        num_layers=enc_cfg.get("num_layers", 4),
        dim_feedforward=enc_cfg.get("dim_feedforward", 256),
        dropout=enc_cfg.get("dropout", 0.1),
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    tcfg = cfg["training"]
    optimizer = AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    pos_weight = torch.tensor(tcfg.get("pos_weight", 10.0), device=device)

    # CSV log
    csv_path = outdir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "val_prec", "val_rec", "time_s"])

    best_val_loss = float("inf")
    train_times = []
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_losses, train_accs, train_f1s = [], [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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
            acc, _, _, f1 = assoc_accuracy(logits.detach(), a)
            train_accs.append(acc)
            train_f1s.append(f1)

        # Val
        model.eval()
        val_losses, val_accs, val_f1s, val_precs, val_recs = [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                cs, ct = batch["coords_src"].to(device), batch["coords_tgt"].to(device)
                fs, ft = batch["features_src"].to(device), batch["features_tgt"].to(device)
                a = batch["assoc_matrix"].to(device)
                ps, pt = batch["padding_mask_src"].to(device), batch["padding_mask_tgt"].to(device)

                logits = model(cs, fs, ct, ft, ps, pt)
                loss = F.binary_cross_entropy_with_logits(logits, a, pos_weight=pos_weight)
                valid = ~(ps[:, :, None] | pt[:, None, :])
                loss = (loss * valid.float()).sum() / valid.float().sum()
                acc, prec, rec, f1 = assoc_accuracy(logits, a)
                val_losses.append(loss.item())
                val_accs.append(acc)
                val_f1s.append(f1)
                val_precs.append(prec)
                val_recs.append(rec)

        epoch_time = time.perf_counter() - t0
        train_times.append(epoch_time)

        tloss, tacc, tf1 = np.mean(train_losses), np.mean(train_accs), np.mean(train_f1s)
        vloss, vacc, vf1 = np.mean(val_losses), np.mean(val_accs), np.mean(val_f1s)
        vprec, vrec = np.mean(val_precs), np.mean(val_recs)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tloss:.6f}", f"{tacc:.4f}", f"{tf1:.4f}",
                                     f"{vloss:.6f}", f"{vacc:.4f}", f"{vf1:.4f}",
                                     f"{vprec:.4f}", f"{vrec:.4f}", f"{epoch_time:.1f}"])

        logger.info(f"Epoch {epoch}: train_loss={tloss:.4f} train_acc={tacc:.4f} "
                     f"val_loss={vloss:.4f} val_acc={vacc:.4f} val_f1={vf1:.4f} [{epoch_time:.0f}s]")

        if vloss < best_val_loss:
            best_val_loss = vloss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": vloss, "val_acc": vacc},
                       outdir / "best_model.pt")
            logger.info(f"  Best model saved (epoch {epoch})")

    logger.info(f"Done. CSV at {csv_path}, best model at {outdir / 'best_model.pt'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
