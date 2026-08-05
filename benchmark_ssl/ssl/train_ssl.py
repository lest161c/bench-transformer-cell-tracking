"""Standalone contrastive SSL training with InfoNCE (NT-Xent) loss.

Trains CellEmbedder using SimCLR-style contrastive learning:
- Two augmented views per frame → embeddings z1, z2
- Positive pairs: same cell across views
- Negative pairs: all other cells in the frame (in-batch negatives)
- Loss: Normalized Temperature-scaled Cross Entropy (NT-Xent)

Replaces old approach: BCE loss on association matrix + wrong ASCENT architecture.
"""

import csv, logging, sys, yaml, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark_ssl root (sibling modules)

from distortions import DistortionPipeline
from ssl_pipeline import load_experiment_frames, SSLDataset, collate_ssl
from track_encoder import CellEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def nt_xent_loss(z1, z2, padding_mask1, padding_mask2, temperature=0.05):
    """NT-Xent (InfoNCE) loss for contrastive representation learning.

    Computes per-frame: for each cell present in both views, the positive
    pair is (embed_view1, embed_view2). All other cells in the frame serve
    as negatives.

    Args:
        z1:             (B, N, D) — embeddings from view 1
        z2:             (B, N, D) — embeddings from view 2
        padding_mask1:  (B, N)   — True for padded positions in view 1
        padding_mask2:  (B, N)   — True for padded positions in view 2
        temperature:    float    — softmax temperature (0.05 standard in SimCLR)

    Cells at position i in both views are assumed to be the same cell
    (guaranteed by label-sorted collation).

    Returns:
        scalar loss averaged over frames (and valid cells within frames).
    """
    B, N_max, D = z1.shape

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    total_loss = 0.0
    n_frames = 0

    for b in range(B):
        pm1 = padding_mask1[b]
        pm2 = padding_mask2[b]
        valid = ~pm1 & ~pm2
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
    """Measure how consistent embeddings are across views (cosine similarity of matching pairs)."""
    valid = ~padding_mask1 & ~padding_mask2
    if valid.sum() == 0:
        return 0.0

    z1_n = F.normalize(z1, dim=-1)
    z2_n = F.normalize(z2, dim=-1)

    sim = (z1_n * z2_n).sum(dim=-1)
    consistency = sim[valid].mean().item()
    return consistency


def main(config_path="config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    outdir = Path("runs") / cfg.get("name", "ssl_contrastive")
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

    # Model
    feat_dim = 7
    enc_cfg = cfg.get("encoder", {})
    ssl_cfg = cfg.get("ssl", {})
    temperature = ssl_cfg.get("temperature", 0.05)

    model = CellEmbedder(
        feat_dim=feat_dim,
        coord_dim=cfg.get("ndim", 2),
        d_model=enc_cfg.get("d_model", 128),
        nhead=enc_cfg.get("nhead", 4),
        num_layers=enc_cfg.get("num_layers", 4),
        dim_feedforward=enc_cfg.get("dim_feedforward", 256),
        dropout=enc_cfg.get("dropout", 0.1),
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    tcfg = cfg["training"]
    optimizer = AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    # CSV log
    csv_path = outdir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_consistency", "val_loss", "val_consistency", "time_s"])

    best_val_loss = float("inf")

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        train_losses, train_cons = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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

            train_losses.append(loss.item())
            train_cons.append(embedding_consistency(z1.detach(), z2.detach(), pm1, pm2))

        # Val
        model.eval()
        val_losses, val_cons = [], []
        with torch.no_grad():
            for batch in val_loader:
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
                val_losses.append(loss.item())
                val_cons.append(embedding_consistency(z1, z2, pm1, pm2))

        epoch_time = time.perf_counter() - t0

        tloss = np.mean(train_losses) if train_losses else 0
        tcons = np.mean(train_cons) if train_cons else 0
        vloss = np.mean(val_losses) if val_losses else 0
        vcons = np.mean(val_cons) if val_cons else 0

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tloss:.6f}", f"{tcons:.4f}",
                                     f"{vloss:.6f}", f"{vcons:.4f}", f"{epoch_time:.1f}"])

        logger.info(f"Epoch {epoch}: train_loss={tloss:.4f} train_cons={tcons:.4f} "
                     f"val_loss={vloss:.4f} val_cons={vcons:.4f} [{epoch_time:.0f}s]")

        if vloss < best_val_loss:
            best_val_loss = vloss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": vloss, "val_consistency": vcons},
                       outdir / "best_model.pt")
            logger.info(f"  Best model saved (epoch {epoch})")

    logger.info(f"Done. CSV at {csv_path}, best model at {outdir / 'best_model.pt'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
