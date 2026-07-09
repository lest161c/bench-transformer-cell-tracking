#!/usr/bin/env python3
"""H100 convergence race: R (7D baseline) vs C (7D + frozen CNN NT-Xent).

Mini Trackastra at full scale: d_model=320, nhead=8, 6 enc + 6 dec layers.
Data: ALL vanvliet frames (6 conditions), consecutive pairs.
Training: BCE(pos_weight=100.0), AdamW, cosine schedule, early stopping.

Usage:
    python h100_convergence.py                              # full run
    python h100_convergence.py --epochs 10 --max-pairs 5    # quick test
"""

import argparse
import copy
import csv
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import regionprops as sk_regionprops
from tifffile import imread

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("h100_conv_race")

# ─── Global settings ──────────────────────────────────────────────────────────
SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 64
CNN_FEAT_DIM = 128
PE_DIM_PER_COORD = 16
COORD_DIM = 2
PE_DIM = PE_DIM_PER_COORD * COORD_DIM * 2  # 64

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
#  TinyCNN — ScaledCNN architecture (must match cnn_ssl.py checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledCNN(nn.Module):
    """ConvNet for 64×64 grayscale patches. Output: 128-dim embedding."""
    def __init__(self, scale="medium", out_dim=CNN_FEAT_DIM):
        super().__init__()
        if scale == "small":
            ch = [8, 16, 32]
        elif scale == "medium":
            ch = [16, 32, 64]
        else:
            ch = [32, 64, 128]
        layers = []
        in_ch = 1
        for c in ch:
            layers += [nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = c
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, patches):
        return self.fc(self.conv(patches))


# ═══════════════════════════════════════════════════════════════════════════════
#  CNNPatchExtractor — runs on CPU, frozen, produces 128-dim numpy embeddings
# ═══════════════════════════════════════════════════════════════════════════════

class CNNPatchExtractor:
    """Lightweight CNN feature extractor running on CPU patches.

    Args:
        checkpoint_path: path to NT-Xent pretrained ScaledCNN checkpoint.
    """
    def __init__(self, checkpoint_path, scale="medium"):
        self.cpu = torch.device("cpu")
        self.model = ScaledCNN(scale=scale, out_dim=CNN_FEAT_DIM)
        ckpt = torch.load(checkpoint_path, map_location=self.cpu)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        logger.info(f"  CNNPatchExtractor: loaded {checkpoint_path} "
                     f"({sum(p.numel() for p in self.model.parameters()):,} params, CPU)")

    @torch.no_grad()
    def extract(self, patches_np):
        """patches_np: (N, 1, 64, 64) float32 → (N, 128) float32."""
        t = torch.from_numpy(patches_np).to(self.cpu)
        emb = self.model(t)
        return emb.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#  Fourier Positional Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class FourierPE(nn.Module):
    """Fourier feature encoding for 2D coordinates."""
    def __init__(self, pos_per_dim=PE_DIM_PER_COORD, coord_dim=COORD_DIM):
        super().__init__()
        self.d = coord_dim * pos_per_dim * 2
        self.coord_dim = coord_dim
        self.pos_per_dim = pos_per_dim
        self.register_buffer("freqs", 2.0 ** torch.linspace(0.0, 10.0, pos_per_dim))

    def forward(self, c):
        """c: (B, N, 2) or (N, 2) — coordinates."""
        squeeze = c.dim() == 2
        if squeeze:
            c = c.unsqueeze(0)
        parts = []
        for i in range(self.coord_dim):
            parts.append(torch.sin(c[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        for i in range(self.coord_dim):
            parts.append(torch.cos(c[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        out = torch.cat(parts, dim=-1)
        if squeeze:
            out = out.squeeze(0)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Mini Trackastra — Encoder / Decoder / Full Model
# ═══════════════════════════════════════════════════════════════════════════════

class MiniEncoder(nn.Module):
    """Encoder: proj each input, concat, fuse → LayerNorm → TransformerEncoder.

    Mode R (baseline):      concat(pe_proj, feat_proj) → fuse → encoder
    Mode C (CNN residual):  concat(pe_proj, feat_proj, cnn_proj) → fuse → encoder
    """
    def __init__(self, d_model=320, nhead=8, num_layers=6, pe_dim=PE_DIM,
                 feat_dim=7, cnn_feat_dim=None):
        super().__init__()
        self.d_model = d_model
        self.cnn_feat_dim = cnn_feat_dim

        self.pe_proj = nn.Linear(pe_dim, d_model)
        self.feat_proj = nn.Linear(feat_dim, d_model)

        if cnn_feat_dim is not None:
            self.cnn_proj = nn.Linear(cnn_feat_dim, d_model)
            self.concat_fuse = nn.Linear(d_model * 3, d_model)
        else:
            self.concat_fuse = nn.Linear(d_model * 2, d_model)

        self.norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats, pe, cnn_feats=None):
        pe_p = self.pe_proj(pe)          # (N, d_model)
        f_p = self.feat_proj(feats)      # (N, d_model)
        if self.cnn_feat_dim is not None and cnn_feats is not None:
            c_p = self.cnn_proj(cnn_feats)  # (N, d_model)
            x = torch.cat([pe_p, f_p, c_p], dim=-1)
        else:
            x = torch.cat([pe_p, f_p], dim=-1)
        x = self.norm(self.concat_fuse(x))
        return self.encoder(x)


class MiniDecoder(nn.Module):
    """Decoder: TransformerDecoder for cross-attention on pair."""
    def __init__(self, d_model=320, nhead=8, num_layers=6):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)


class MiniTrackingTransformer(nn.Module):
    """Full miniature Trackastra: encoder + decoder + association heads."""
    def __init__(self, encoder: MiniEncoder, decoder: MiniDecoder, d_head=32):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = encoder.d_model
        self.head_x = nn.Linear(self.d_model, d_head)
        self.head_y = nn.Linear(self.d_model, d_head)
        for p in self.head_x.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.head_y.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_t, pe_t, feat_n, pe_n, cnn_t=None, cnn_n=None):
        enc_t = self.encoder(feat_t, pe_t, cnn_t)   # (N1, d_model)
        enc_n = self.encoder(feat_n, pe_n, cnn_n)   # (N2, d_model)
        N1 = enc_t.shape[0]
        enc_all = torch.cat([enc_t, enc_n], dim=0)   # (N1+N2, d_model)
        dec_all = self.decoder(enc_all, enc_all)      # (N1+N2, d_model)
        dec_n = dec_all[N1:]                           # (N2, d_model)
        hx = F.normalize(self.head_x(enc_t), dim=-1)  # (N1, d_head)
        hy = F.normalize(self.head_y(dec_n), dim=-1)  # (N2, d_head)
        logits = (hx @ hy.T) * (self.head_x.out_features ** 0.5)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Returns (coords, labels, img) or None."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    from skimage.measure import regionprops_table
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack(
        [props["centroid-0"], props["centroid-1"]], axis=-1
    ).astype(np.float32)
    return coords, props["label"].astype(np.int32), img


def scan_consecutive_pairs(data_root, conditions, max_pairs=None):
    """Find ALL consecutive frame pairs across conditions."""
    dr = Path(data_root)
    pairs = []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir():
                continue
            tra = exp / "TRA"
            img_dir = exp / "img"
            if not tra.exists() or not img_dir.exists():
                continue
            masks = sorted(tra.glob("man_track*.tif"))
            for i in range(len(masks) - 1):
                st1 = masks[i].stem.replace("man_track", "")
                st2 = masks[i + 1].stem.replace("man_track", "")
                try:
                    f1, f2 = int(st1), int(st2)
                except ValueError:
                    continue
                if f2 == f1 + 1:
                    ip1 = img_dir / f"t{f1:06d}.tif"
                    ip2 = img_dir / f"t{f2:06d}.tif"
                    if ip1.exists() and ip2.exists():
                        pairs.append(
                            (str(masks[i]), str(masks[i + 1]), str(ip1), str(ip2))
                        )
                        if max_pairs is not None and len(pairs) >= max_pairs:
                            return pairs
    return pairs


def extract_patches(img, centroids, patch_size=PATCH_SIZE):
    """Extract square patches centered on centroids. Handles edge padding."""
    h, w = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i, cx_i = np.clip(cy_i, 0, h - 1), np.clip(cx_i, 0, w - 1)
        y1, x1 = cy_i - half, cx_i - half
        y2, x2 = cy_i + half, cx_i + half
        pt, pb = max(0, -y1), max(0, y2 - h)
        pl, pr = max(0, -x1), max(0, x2 - w)
        y1c, x1c = max(0, y1), max(0, x1)
        y2c, x2c = min(h, y2), min(w, x2)
        crop = (
            img[y1c:y2c, x1c:x2c]
            if (y2c > y1c and x2c > x1c)
            else np.zeros((1, 1), dtype=np.float32)
        )
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(
                crop,
                tuple((0, max(0, t)) for t in [patch_size - s for s in crop.shape]),
                mode="reflect",
            )[:patch_size, :patch_size]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, patch_size, patch_size), dtype=np.float32)


def extract_regionprops_7d_by_label(mask, img, labels_of_interest):
    """Extract 7D regionprops for specific labels only.

    Returns (N, 7) float32 array ordered by labels_of_interest.
    """
    props_list = sk_regionprops(mask, intensity_image=img)
    label_to_row = {}
    for p in props_list:
        label_to_row[int(p.label)] = [
            p.area,
            p.eccentricity,
            p.perimeter,
            p.solidity if p.solidity is not None else 1.0,
            p.extent if p.extent is not None else 1.0,
            p.orientation if p.orientation is not None else 0.0,
            p.intensity_mean if p.intensity_mean is not None else 0.0,
        ]
    rows = []
    for lbl in labels_of_interest:
        rows.append(label_to_row.get(int(lbl), [0.0] * 7))
    return np.array(rows, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss & metrics
# ═══════════════════════════════════════════════════════════════════════════════

def build_target_matrix(labels_t, labels_n, dev=None):
    """Build (N1, N2) binary target matrix: 1 where same label."""
    N1, N2 = len(labels_t), len(labels_n)
    target = torch.zeros(N1, N2, dtype=torch.float32)
    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_n):
            if lt == ln:
                target[i, j] = 1.0
    if dev is not None:
        target = target.to(dev)
    return target


def bce_assoc_loss(logits, labels_t, labels_n, pos_weight=100.0):
    """Weighted BCE loss. pos_weight gives extra weight to positive pairs."""
    target = build_target_matrix(labels_t, labels_n, logits.device)
    weight = torch.where(target == 1.0, pos_weight, 1.0)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def balanced_accuracy(logits, labels_t, labels_n):
    """Balanced accuracy: mean of positive and negative accuracy."""
    pred = (torch.sigmoid(logits) > 0.5).cpu()
    target = build_target_matrix(labels_t, labels_n, "cpu").bool()
    pos_mask = target
    neg_mask = ~target
    pos_acc = (
        (pred[pos_mask] == target[pos_mask]).float().mean().item()
        if pos_mask.any()
        else 0.5
    )
    neg_acc = (
        (pred[neg_mask] == target[neg_mask]).float().mean().item()
        if neg_mask.any()
        else 0.5
    )
    return (pos_acc + neg_acc) / 2.0


def accuracy(logits, labels_t, labels_n):
    """Simple accuracy."""
    pred = (torch.sigmoid(logits) > 0.5).cpu()
    target = build_target_matrix(labels_t, labels_n, "cpu").bool()
    return (pred == target).float().mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, pair_data, mode_name, pe_mod, opt=None, scheduler=None):
    """Run one epoch over train or val data.

    Args:
        opt: if provided, run training (backprop); else eval only.
    Returns:
        (avg_loss, avg_bal_acc, avg_acc)
    """
    is_train = opt is not None
    model.train() if is_train else model.eval()

    losses = []
    bal_accs = []
    accs = []
    feat_key = "feat_7d"
    cnn_key = "cnn"

    items = list(pair_data)
    if is_train:
        np.random.shuffle(items)

    with torch.set_grad_enabled(is_train):
        for pd_item in items:
            feat_t = pd_item[f"{feat_key}_t"].to(device)
            feat_n = pd_item[f"{feat_key}_n"].to(device)
            ct = pd_item["coords_t"].unsqueeze(0).to(device)
            cn = pd_item["coords_n"].unsqueeze(0).to(device)
            lt = pd_item["labels_t"]
            ln = pd_item["labels_n"]

            pe_t = pe_mod(ct).squeeze(0)
            pe_n = pe_mod(cn).squeeze(0)

            # Mode C: use CNN features; Mode R: pass None
            cnn_t = pd_item[f"{cnn_key}_t"].to(device) if mode_name == "C" else None
            cnn_n = pd_item[f"{cnn_key}_n"].to(device) if mode_name == "C" else None

            scores = model(feat_t, pe_t, feat_n, pe_n, cnn_t=cnn_t, cnn_n=cnn_n)
            loss = bce_assoc_loss(scores, lt, ln, pos_weight=100.0)

            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            losses.append(loss.item())
            with torch.no_grad():
                bal_accs.append(balanced_accuracy(scores, lt, ln))
                accs.append(accuracy(scores, lt, ln))

    avg_loss = float(np.mean(losses))
    avg_bal_acc = float(np.mean(bal_accs))
    avg_acc = float(np.mean(accs))

    if is_train and scheduler is not None:
        scheduler.step()

    return avg_loss, avg_bal_acc, avg_acc


# ═══════════════════════════════════════════════════════════════════════════════
#  Precompute data
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_data(pairs, cnn_extractor, min_cells=4):
    """Precompute features for all frame pairs.

    Returns dict with 'train', 'val', 'test' splits (80/10/10).
    """
    # Split pairs deterministically
    np.random.seed(SEED)
    idx = np.random.permutation(len(pairs))
    n_total = len(idx)
    n_test = max(1, int(n_total * 0.1))
    n_val = max(1, int(n_total * 0.1))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    logger.info(f"  Split: {len(train_idx)} train + {len(val_idx)} val "
                f"+ {len(test_idx)} test (total {n_total} pairs)")

    pair_data = {"train": [], "val": [], "test": []}
    skipped = {"no_share": 0, "min_cells": 0, "feat": 0}

    for split_name, indices in split_indices.items():
        for pi in indices:
            mt, mn, it, i_n = pairs[pi]
            rt = load_frame(mt, it)
            rn = load_frame(mn, i_n)
            if rt is None or rn is None:
                skipped["no_share"] += 1
                continue
            ct, lt, imgt = rt
            cn, ln, imgn = rn

            mask_t = imread(mt)
            mask_n = imread(mn)

            shared = set(lt) & set(ln)
            if len(shared) < min_cells:
                skipped["no_share"] += 1
                continue
            if len(lt) < min_cells or len(ln) < min_cells:
                skipped["min_cells"] += 1
                continue

            idx_t = [i for i, l in enumerate(lt) if l in shared]
            idx_n = [i for i, l in enumerate(ln) if l in shared]
            ct_s, lt_s = ct[idx_t], lt[idx_t]
            cn_s, ln_s = cn[idx_n], ln[idx_n]

            # 7D regionprops
            feat_7d_t = extract_regionprops_7d_by_label(mask_t, imgt, lt_s)
            feat_7d_n = extract_regionprops_7d_by_label(mask_n, imgn, ln_s)

            if len(feat_7d_t) < 2 or len(feat_7d_n) < 2:
                skipped["feat"] += 1
                continue

            # CNN features via frozen CPU extractor
            pt_s = extract_patches(imgt, ct_s)  # (N, 64, 64)
            pn_s = extract_patches(imgn, cn_s)
            pt_cnn = pt_s[:, None, :, :]  # (N, 1, 64, 64)
            pn_cnn = pn_s[:, None, :, :]

            cnn_feat_t = cnn_extractor.extract(pt_cnn)  # (N, 128)
            cnn_feat_n = cnn_extractor.extract(pn_cnn)

            pair_data[split_name].append({
                "feat_7d_t": torch.from_numpy(feat_7d_t).float(),
                "feat_7d_n": torch.from_numpy(feat_7d_n).float(),
                "cnn_t": torch.from_numpy(cnn_feat_t).float(),
                "cnn_n": torch.from_numpy(cnn_feat_n).float(),
                "coords_t": torch.from_numpy(ct_s).float(),
                "coords_n": torch.from_numpy(cn_s).float(),
                "labels_t": torch.from_numpy(lt_s).long(),
                "labels_n": torch.from_numpy(ln_s).long(),
            })

    for split_name in ["train", "val", "test"]:
        logger.info(f"    {split_name}: {len(pair_data[split_name])} items")

    total_skipped = sum(skipped.values())
    if total_skipped > 0:
        logger.info(f"    skipped: {skipped}")

    return pair_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint & CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(state, mode, is_best=False):
    ckpt_dir = Path("checkpoints") / "h100_conv_race"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    suffix = "best" if is_best else "latest"
    path = ckpt_dir / f"model_{mode}_{suffix}.pt"
    torch.save(state, path)
    return path


def append_csv(mode, epoch, metrics):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"h100_conv_race_{mode}.csv"
    fieldnames = list(metrics.keys())
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(metrics)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main convergence race
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="H100 convergence race: R (7D) vs C (7D + CNN NT-Xent)"
    )
    parser.add_argument("--data-root", default="../../data/vanvliet")
    parser.add_argument(
        "--conditions",
        default="rpsM,recA,pheA,metA,cib,trpL",
        help="Comma-separated condition names",
    )
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit pairs (None = all)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--min-cells", type=int, default=4)
    parser.add_argument("--pos-weight", type=float, default=100.0)
    parser.add_argument("--checkpoint", type=str,
                        default="probe/cnn_ntxent.pt",
                        help="Path to NT-Xent pretrained CNN")
    parser.add_argument("--scale", choices=["small", "medium", "large"],
                        default="medium",
                        help="CNN architecture scale matching checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    conditions = [c.strip() for c in args.conditions.split(",")]

    logger.info("=" * 70)
    logger.info("H100 CONVERGENCE RACE — Mini Trackastra @ full scale")
    logger.info("=" * 70)
    logger.info(f"  Architecture: d_model=320, nhead=8, enc=6, dec=6")
    logger.info(f"  PE dim: {PE_DIM}")
    logger.info(f"  Data: vanvliet ALL conditions ({conditions})")
    logger.info(f"  Max epochs: {args.epochs}, patience: {args.patience}")
    logger.info(f"  LR: {args.lr} (AdamW, cosine), pos_weight: {args.pos_weight}")
    logger.info(f"  Split: 80/10/10, seed={SEED}")
    logger.info(f"  Device: {device}")
    logger.info(f"  CNN checkpoint: {args.checkpoint}")
    logger.info(f"  WANDB: {'disabled' if args.no_wandb else 'enabled'}")
    logger.info("")

    # ─── 1. Load frozen CNN extractor ──────────────────────────────────────
    logger.info("[1/5] Loading frozen CNN extractor (CPU)...")
    cnn_extractor = CNNPatchExtractor(args.checkpoint, scale=args.scale)

    # ─── 2. Scan pairs ─────────────────────────────────────────────────────
    logger.info("[2/5] Scanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(
        args.data_root, conditions, max_pairs=args.max_pairs
    )
    logger.info(f"  Found {len(pairs)} consecutive frame pairs")
    if len(pairs) < 5:
        logger.error("Need at least 5 pairs. Cannot continue.")
        sys.exit(1)

    # ─── 3. Precompute features ────────────────────────────────────────────
    logger.info("[3/5] Precomputing features (7D regionprops + CNN)...")
    t0 = time.time()
    pair_data = precompute_data(pairs, cnn_extractor, min_cells=args.min_cells)
    logger.info(f"  Precomputation took {time.time() - t0:.1f}s")

    if len(pair_data["train"]) < 2 or len(pair_data["val"]) < 1:
        logger.error("Not enough training/val data after filtering.")
        sys.exit(1)

    # ─── 4. Run both modes ─────────────────────────────────────────────────
    logger.info("[4/5] Running convergence race...")

    all_results = {}

    for mode in ["R", "C"]:
        logger.info("")
        logger.info(f"{'─' * 60}")
        logger.info(f"  MODE {mode}: {'7D baseline (concat)' if mode == 'R' else '7D + CNN residual (concat)'}")
        logger.info(f"{'─' * 60}")

        # Build model
        enc = MiniEncoder(
            d_model=320, nhead=8, num_layers=6, pe_dim=PE_DIM,
            feat_dim=7,
            cnn_feat_dim=CNN_FEAT_DIM if mode == "C" else None,
        ).to(device)

        dec = MiniDecoder(d_model=320, nhead=8, num_layers=6).to(device)
        model = MiniTrackingTransformer(encoder=enc, decoder=dec, d_head=32).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model params: {n_params:,}")

        pe_mod = FourierPE(pos_per_dim=PE_DIM_PER_COORD).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs
        )

        # W&B init
        wandb_run = None
        if not args.no_wandb:
            try:
                import wandb
                wandb_run = wandb.init(
                    project="trackastra-cnn-convergence",
                    name=f"h100_conv_race_{mode}",
                    reinit=True,
                    config={
                        "d_model": 320,
                        "nhead": 8,
                        "enc_layers": 6,
                        "dec_layers": 6,
                        "mode": mode,
                        "modes": ["R_baseline", "C_cnn_residual"],
                        "data": "vanvliet_all",
                        "loss": "weighted_BCE_pos100",
                        "lr": args.lr,
                        "epochs": args.epochs,
                        "patience": args.patience,
                        "seed": SEED,
                    },
                )
            except Exception as e:
                logger.warning(f"  W&B init failed: {e}")
                wandb_run = None

        # Training loop
        best_val_loss = float("inf")
        best_epoch = -1
        best_state = None
        patience_counter = 0
        history = []

        for epoch in range(args.epochs):
            t_ep = time.time()

            train_loss, train_bal, train_acc = run_epoch(
                model, pair_data["train"], mode, pe_mod, opt=opt, scheduler=scheduler
            )
            val_loss, val_bal, val_acc = run_epoch(
                model, pair_data["val"], mode, pe_mod, opt=None
            )

            elapsed = time.time() - t_ep

            metrics = {
                "epoch": epoch,
                "mode": mode,
                "train_loss": train_loss,
                "train_bal_acc": train_bal,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_bal_acc": val_bal,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
                "epoch_time": elapsed,
            }
            history.append(metrics)

            # Print progress
            if epoch % 10 == 0 or epoch == args.epochs - 1 or val_loss < best_val_loss:
                logger.info(
                    f"  [{mode}] ep {epoch:3d}/{args.epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_bal={val_bal:.4f} | val_acc={val_acc:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
                )

            # W&B log
            if wandb_run is not None:
                try:
                    import wandb
                    wandb.log(metrics)
                except Exception:
                    pass

            # CSV log
            append_csv(mode, epoch, metrics)

            # Early stopping: track best val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                # Save best checkpoint
                ckpt_path = save_checkpoint(
                    {
                        "epoch": epoch,
                        "mode": mode,
                        "model_state_dict": model.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "best_val_loss": best_val_loss,
                        "args": vars(args),
                    },
                    mode,
                    is_best=True,
                )
                logger.info(f"  ✓ New best val_loss={best_val_loss:.4f} at ep {epoch} "
                            f"→ saved {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(
                        f"  ✗ Early stopping at epoch {epoch} "
                        f"(patience={args.patience}, best was ep {best_epoch})"
                    )
                    break

        # Save latest checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "mode": mode,
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "args": vars(args),
            },
            mode,
            is_best=False,
        )

        # Restore best model for final eval
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on test set
        if len(pair_data["test"]) > 0:
            test_loss, test_bal, test_acc = run_epoch(
                model, pair_data["test"], mode, pe_mod, opt=None
            )
            logger.info(
                f"  [TEST  {mode}] loss={test_loss:.4f} bal_acc={test_bal:.4f} "
                f"acc={test_acc:.4f}"
            )
        else:
            test_loss, test_bal, test_acc = None, None, None

        all_results[mode] = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
            "test_loss": test_loss,
            "test_bal_acc": test_bal,
            "test_acc": test_acc,
        }

        # W&B finish
        if wandb_run is not None:
            try:
                import wandb
                wandb.log(
                    {
                        f"{mode}/best_val_loss": best_val_loss,
                        f"{mode}/best_epoch": best_epoch,
                        f"{mode}/test_loss": test_loss,
                        f"{mode}/test_bal_acc": test_bal,
                    }
                )
                wandb.finish()
            except Exception:
                pass

        # Clean up GPU memory
        del model, enc, dec, opt, scheduler, pe_mod
        torch.cuda.empty_cache()

    # ─── 5. Summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("[5/5] Final summary")
    print()
    print("=" * 70)
    print("H100 CONVERGENCE RACE — RESULTS")
    print("=" * 70)

    header = (
        f"{'Mode':<8} | {'Best Val Loss':<14} | {'Best Ep':<8} | "
        f"{'Test Loss':<10} | {'Test BalAcc':<12} | {'Test Acc':<10}"
    )
    print(header)
    print("-" * 70)

    for mode in ["R", "C"]:
        r = all_results.get(mode, {})
        bvl = r.get("best_val_loss", float("nan"))
        bep = r.get("best_epoch", -1)
        tl = r.get("test_loss", float("nan"))
        tb = r.get("test_bal_acc", float("nan"))
        ta = r.get("test_acc", float("nan"))
        tl_str = f"{tl:.4f}" if tl is not None else "N/A"
        tb_str = f"{tb:.4f}" if tb is not None else "N/A"
        ta_str = f"{ta:.4f}" if ta is not None else "N/A"
        print(
            f"{mode:<8} | {bvl:<14.4f} | {bep:<8} | "
            f"{tl_str:<10} | {tb_str:<12} | {ta_str:<10}"
        )
        # Convergence speed
        history = r.get("history", [])
        steps_1 = next((h["epoch"] for h in history if h["val_loss"] < 1.0), None)
        steps_05 = next((h["epoch"] for h in history if h["val_loss"] < 0.5), None)
        print(f"          Val loss < 1.0 at epoch: {steps_1}")
        print(f"          Val loss < 0.5 at epoch: {steps_05}")

    print("-" * 70)

    # Delta
    if "R" in all_results and "C" in all_results:
        delta = (all_results["C"].get("test_bal_acc") or 0) - (
            all_results["R"].get("test_bal_acc") or 0
        )
        if abs(delta) > 0.001:
            verdict = (
                f"IMPROVES by Δ={delta:.4f}"
                if delta > 0
                else f"UNDERPERFORMS by Δ={delta:.4f}"
            )
            print(f"VERDICT: CNN residual {verdict} vs baseline (test bal_acc)")
        else:
            print("VERDICT: CNN residual similar to baseline (Δ < 0.001)")

    print("=" * 70)
    logger.info("H100 convergence race complete.")


if __name__ == "__main__":
    main()
