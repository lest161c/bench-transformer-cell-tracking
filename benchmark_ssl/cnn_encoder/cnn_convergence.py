#!/usr/bin/env python3
"""Convergence race: 7D baseline vs CNN NT-Xent residual at maxed-out A500 config.

Two modes, all trained from scratch:
  - Mode R: pos_proj(PE) + feat_proj(7D) -> encoder -> downstream
  - Mode C: pos_proj(PE) + feat_proj(7D) + cnn_proj(CNN_NTXent) -> encoder -> downstream

Usage:
    python cnn_convergence.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import regionprops_table
from skimage.measure import regionprops as sk_regionprops
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cnn_convergence")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PATCH_SIZE = 64
CNN_FEAT_DIM = 128  # ScaledCNN output dim


# ═══════════════════════════════════════════════════════════════════════════════
#  ScaledCNN (same arch as cnn_ssl.py, for loading checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledCNN(nn.Module):
    """ConvNet for 64x64 grayscale cell patches. Output: 128-dim embedding."""
    def __init__(self, scale='medium', out_dim=CNN_FEAT_DIM):
        super().__init__()
        PATCH_SIZE = 64
        if scale == 'small':
            ch = [8, 16, 32]
        elif scale == 'medium':
            ch = [16, 32, 64]
        else:  # large
            ch = [32, 64, 128, 256]
        layers = []
        in_ch = 1
        for channel_dim in ch:
            layers += [nn.Conv2d(in_ch, channel_dim, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = channel_dim
        self.conv = nn.Sequential(*layers)
        spatial = PATCH_SIZE // (2 ** len(ch))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, patches):
        """patches: (N, 1, PATCH_SIZE, PATCH_SIZE) → embeddings: (N, out_dim)."""
        return self.fc(self.conv(patches))


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_frame(mask_path, img_path):
    """Returns (coords, labels, img) or None."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return coords, props["label"].astype(np.int32), img


def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame pairs."""
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
                        pairs.append((str(masks[i]), str(masks[i + 1]), str(ip1), str(ip2)))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def extract_patches(img, centroids, patch_size=PATCH_SIZE):
    """Extract square patches centered on centroids."""
    height, width = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i, cx_i = np.clip(cy_i, 0, height - 1), np.clip(cx_i, 0, width - 1)
        y1, x1 = cy_i - half, cx_i - half
        y2, x2 = cy_i + half, cx_i + half
        pt, pb = max(0, -y1), max(0, y2 - height)
        pl, pr = max(0, -x1), max(0, x2 - width)
        y1c, x1c = max(0, y1), max(0, x1)
        y2c, x2c = min(height, y2), min(width, x2)
        crop = img[y1c:y2c, x1c:x2c] if (y2c > y1c and x2c > x1c) else np.zeros((1, 1), dtype=np.float32)
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(crop, tuple((0, max(0, pad_amount)) for pad_amount in [patch_size - dim_size for dim_size in crop.shape]), mode="reflect")[:patch_size, :patch_size]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, patch_size, patch_size), dtype=np.float32)


def extract_regionprops_7d(mask, img):
    """Extract 7D regionprops: area, ecc, perimeter, solidity, extent, orient, mean_int.
    Returns (coords, labels, feats) or (None, None, None).
    """
    props_list = sk_regionprops(mask, intensity_image=img)
    if len(props_list) == 0:
        return None, None, None
    n_regions = len(props_list)
    feats = np.zeros((n_regions, 7), dtype=np.float32)
    coords = np.zeros((n_regions, 2), dtype=np.float32)
    labels = np.zeros(n_regions, dtype=np.int32)
    for i, region in enumerate(props_list):
        feats[i, 0] = region.area
        feats[i, 1] = region.eccentricity
        feats[i, 2] = region.perimeter
        feats[i, 3] = region.solidity if region.solidity is not None else 1.0
        feats[i, 4] = region.extent if region.extent is not None else 1.0
        feats[i, 5] = region.orientation if region.orientation is not None else 0.0
        feats[i, 6] = region.intensity_mean if region.intensity_mean is not None else 0.0
        coords[i] = region.centroid
        labels[i] = region.label
    return coords, labels, feats


# ═══════════════════════════════════════════════════════════════════════════════
#  Positional Encodings
# ═══════════════════════════════════════════════════════════════════════════════

class FourierPE(nn.Module):
    """Fourier feature encoding for coordinates."""
    def __init__(self, pos_per_dim=16, coord_dim=2):
        """Register per-coordinate Fourier frequencies; output dim is
        coord_dim * pos_per_dim * 2."""
        super().__init__()
        self.pe_dim = coord_dim * pos_per_dim * 2
        self.coord_dim = coord_dim
        self.pos_per_dim = pos_per_dim
        self.register_buffer("freqs", 2.0 ** torch.linspace(0.0, 10.0, pos_per_dim))

    def forward(self, coords):
        """Encode coords (N, coord_dim) into Fourier features (N, pe_dim)."""
        parts = []
        for i in range(self.coord_dim):
            parts.append(torch.sin(coords[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        for i in range(self.coord_dim):
            parts.append(torch.cos(coords[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        return torch.cat(parts, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MiniEncoder / MiniDecoder / MiniTrackingTransformer
# ═══════════════════════════════════════════════════════════════════════════════

class MiniEncoder(nn.Module):
    """Encoder: feat_proj + PE_proj + optional CNN residual + LayerNorm + TransformerEncoder."""
    def __init__(self, d_model=128, nhead=4, num_layers=2, pe_dim=64,
                 feat_dim=7, cnn_feat_dim=None):
        super().__init__()
        self.d_model = d_model
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.pe_proj = nn.Linear(pe_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.cnn_feat_dim = cnn_feat_dim

        if cnn_feat_dim is not None:
            self.cnn_proj = nn.Linear(cnn_feat_dim, 64)          # project to smaller dim
            self.fuse_proj = nn.Linear(d_model + 64, d_model)   # concat → d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            activation="relu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feats, pe, cnn_feats=None):
        x = self.norm(self.pe_proj(pe) + self.feat_proj(feats))  # (N, d_model)
        if self.cnn_feat_dim is not None and cnn_feats is not None:
            c = self.cnn_proj(cnn_feats)                          # (N, 64)
            x = self.fuse_proj(torch.cat([x, c], dim=-1))         # (N, d_model) — learnable gating
        return self.encoder(x)


class MiniDecoder(nn.Module):
    """Decoder: TransformerDecoder for cross-attention on pair."""
    def __init__(self, d_model=128, nhead=4, num_layers=2):
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
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        for p in self.head_y.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, feat_t, pe_t, feat_n, pe_n, cnn_t=None, cnn_n=None):
        enc_t = self.encoder(feat_t, pe_t, cnn_t)
        enc_n = self.encoder(feat_n, pe_n, cnn_n)
        N1 = enc_t.shape[0]
        enc_all = torch.cat([enc_t, enc_n], dim=0)
        dec_all = self.decoder(enc_all, enc_all)
        dec_n = dec_all[N1:]
        hx = F.normalize(self.head_x(enc_t), dim=-1)
        hy = F.normalize(self.head_y(dec_n), dim=-1)
        logits = (hx @ hy.T) * (self.head_x.out_features ** 0.5)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss & metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _build_target_matrix(labels_t, labels_n, dev):
    N1, N2 = len(labels_t), len(labels_n)
    target = torch.zeros(N1, N2, device=dev)
    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_n):
            if lt == ln:
                target[i, j] = 1.0
    return target


def bce_assoc_loss(logits, labels_t, labels_n, pos_weight=10.0):
    """Weighted BCE loss. positive weight compensates for class imbalance."""
    target = _build_target_matrix(labels_t, labels_n, logits.device)
    # pos_weight gives NEG× more weight to positive pairs (same cell)
    weight = torch.where(target == 1, pos_weight, 1.0)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def check_collapse(val_bal_acc, stuck_counter, step, min_steps=30):
    """Return (collapsed, updated_counter). Collapsed if bal_acc stays at 0.5
    for `min_steps` consecutive evaluations."""
    if abs(val_bal_acc - 0.5) < 0.001 or val_bal_acc < 0.5:
        stuck_counter += 1
    else:
        stuck_counter = 0
    if stuck_counter >= min_steps // 10:  # eval every 10 steps, so check after 30 steps
        return True, stuck_counter
    return False, stuck_counter


def balanced_accuracy(logits, labels_t, labels_n):
    pred = (torch.sigmoid(logits) > 0.5).cpu()
    target = _build_target_matrix(labels_t, labels_n, "cpu").bool()
    pos_mask = target
    neg_mask = ~target
    pos_acc = (pred[pos_mask] == target[pos_mask]).float().mean().item() if pos_mask.any() else 0.5
    neg_acc = (pred[neg_mask] == target[neg_mask]).float().mean().item() if neg_mask.any() else 0.5
    return (pos_acc + neg_acc) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Convergence Race
# ═══════════════════════════════════════════════════════════════════════════════

def run_convergence_race(pair_data, args, pe_dim):
    """Run 2-mode convergence race: R (7D) and C (7D+CNN+NTXent)."""

    # Build encoder for mode R (7D baseline, no CNN)
    enc_r = MiniEncoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_encoder_layers, pe_dim=pe_dim,
        feat_dim=7,
    ).to(device)
    pe_r = FourierPE(pos_per_dim=args.pos_per_dim).to(device)

    # Build encoder for mode C (7D + CNN residual)
    enc_c = MiniEncoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_encoder_layers, pe_dim=pe_dim,
        feat_dim=7, cnn_feat_dim=CNN_FEAT_DIM,
    ).to(device)
    pe_c = FourierPE(pos_per_dim=args.pos_per_dim).to(device)

    configs = [
        ("R", "7D regionprops only", enc_r, pe_r, "feat_7d", None),
        ("C", "7D + PE + CNN",      enc_c, pe_c, "feat_7d", "cnn"),
    ]

    results = {}
    for mode, mode_name, enc, pe_mod, feat_key, cnn_key in configs:
        dec = MiniDecoder(d_model=args.d_model, nhead=args.nhead,
                          num_layers=args.num_decoder_layers).to(device)
        model = MiniTrackingTransformer(encoder=enc, decoder=dec, d_head=args.d_head).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info(f"\n  Convergence Race [{mode}] {mode_name}")

        rows = []
        stuck_counter = 0
        for step in range(args.steps):
            model.train()
            train_losses, train_bal_accs = [], []

            # Shuffle downstream data each epoch
            train_items = list(pair_data["train"])
            np.random.shuffle(train_items)

            for pd_item in train_items:
                feat_t = pd_item[f"{feat_key}_t"].to(device)
                feat_n = pd_item[f"{feat_key}_n"].to(device)
                ct = pd_item["coords_t"].unsqueeze(0).to(device)
                cn = pd_item["coords_n"].unsqueeze(0).to(device)
                lt = pd_item["labels_t"]
                ln = pd_item["labels_n"]

                pe_t = pe_mod(ct).squeeze(0)
                pe_n = pe_mod(cn).squeeze(0)

                cnn_t = pd_item[f"{cnn_key}_t"].to(device) if cnn_key else None
                cnn_n = pd_item[f"{cnn_key}_n"].to(device) if cnn_key else None

                scores = model(feat_t, pe_t, feat_n, pe_n, cnn_t=cnn_t, cnn_n=cnn_n)
                loss = bce_assoc_loss(scores, lt, ln, pos_weight=args.pos_weight)
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.item())
                sd = scores.detach()
                train_bal_accs.append(balanced_accuracy(sd, lt, ln))

            if step % 10 == 0 or step == args.steps - 1:
                model.eval()
                val_losses, val_bal_accs = [], []
                with torch.no_grad():
                    for pd_item in pair_data["val"]:
                        feat_t = pd_item[f"{feat_key}_t"].to(device)
                        feat_n = pd_item[f"{feat_key}_n"].to(device)
                        ct = pd_item["coords_t"].unsqueeze(0).to(device)
                        cn = pd_item["coords_n"].unsqueeze(0).to(device)
                        lt = pd_item["labels_t"]
                        ln = pd_item["labels_n"]

                        pe_t = pe_mod(ct).squeeze(0)
                        pe_n = pe_mod(cn).squeeze(0)

                        cnn_t = pd_item[f"{cnn_key}_t"].to(device) if cnn_key else None
                        cnn_n = pd_item[f"{cnn_key}_n"].to(device) if cnn_key else None

                        scores = model(feat_t, pe_t, feat_n, pe_n, cnn_t=cnn_t, cnn_n=cnn_n)
                        val_losses.append(bce_assoc_loss(scores, lt, ln, pos_weight=args.pos_weight).item())
                        val_bal_accs.append(balanced_accuracy(scores, lt, ln))

                t_loss = float(np.mean(train_losses))
                t_bal = float(np.mean(train_bal_accs))
                v_loss = float(np.mean(val_losses))
                v_bal = float(np.mean(val_bal_accs))
                rows.append({
                    "step": step,
                    "train_loss": t_loss,
                    "train_bal_acc": t_bal,
                    "val_loss": v_loss,
                    "val_bal_acc": v_bal,
                })
                logger.info(f"    [{mode}] step {step:4d}/{args.steps}: "
                            f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
                            f"val_bal={v_bal:.4f}")

                collapsed, stuck_counter = check_collapse(v_bal, stuck_counter, step)
                if collapsed:
                    logger.warning(f"  Mode {mode}: BAL_ACC STUCK AT 0.5 — model collapsed to all-negatives. Aborting.")
                    break

        results[mode] = rows
        if rows:
            final = rows[-1]
            logger.info(f"    [{mode}] done: val_loss={final['val_loss']:.4f}  "
                        f"val_bal_acc={final['val_bal_acc']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convergence race: 7D vs CNN NT-Xent residual (maxed-out A500 config)"
    )
    parser.add_argument("--data-root", default="../../data/vanvliet")
    parser.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL")
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-cells", type=int, default=6)

    # Architecture (maxed-out A500 config)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--d-head", type=int, default=32)
    parser.add_argument("--pos-per-dim", type=int, default=16)

    # CNN checkpoint
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--checkpoint", default="probe/cnn_ntxent.pt",
                        help="Path to NT-Xent pretrained ScaledCNN checkpoint")
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="Weight for positive pairs in BCE loss (1.0 = unweighted)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global SEED
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    conditions = [c.strip() for c in args.conditions.split(",")]
    pe_dim = args.pos_per_dim * 2 * 2  # coord_dim=2

    logger.info(f"{'='*60}")
    logger.info(f"CONVERGENCE RACE: R (7D) vs C (7D + CNN NT-Xent)")
    logger.info(f"{'='*60}")
    logger.info(f"  d_model={args.d_model}, nhead={args.nhead}, "
                f"enc={args.num_encoder_layers}, dec={args.num_decoder_layers}")
    logger.info(f"  PE dim: {pe_dim}")
    logger.info(f"  Steps: {args.steps}, lr={args.lr}")
    logger.info(f"  Pairs: {args.max_pairs}, Conditions: {conditions}")
    logger.info(f"  CNN checkpoint: {args.checkpoint}")
    logger.info(f"  Device: {device}")

    # ─── Load frozen NT-Xent CNN ──────────────────────────────────────────
    logger.info("Loading frozen NT-Xent CNN...")
    cnn_model = ScaledCNN(scale=args.scale, out_dim=CNN_FEAT_DIM).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cnn_model.load_state_dict(ckpt["model_state_dict"])
    cnn_model.eval()
    for p in cnn_model.parameters():
        p.requires_grad = False
    logger.info(f"Loaded NT-Xent CNN ({args.scale}, "
                f"{sum(p.numel() for p in cnn_model.parameters()):,} params)")

    # ─── Scan pairs ───────────────────────────────────────────────────────
    logger.info("Scanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Found {len(pairs)} consecutive frame pairs")
    if len(pairs) < 3:
        logger.error("Need at least 3 pairs. Increase --max-pairs.")
        sys.exit(1)

    # Split train/val
    np.random.seed(SEED + 1)
    idx = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * 0.2))
    train_pairs = [pairs[i] for i in idx[:-n_val]]
    val_pairs = [pairs[i] for i in idx[-n_val:]]
    logger.info(f"Pairs: {len(train_pairs)} train + {len(val_pairs)} val")

    # ─── Precompute features for both modes ───────────────────────────────
    pair_data = {"train": [], "val": []}
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for mt, mn, it, i_n in split_pairs:
            rt = load_frame(mt, it)
            rn = load_frame(mn, i_n)
            if rt is None or rn is None:
                continue
            ct, lt, imgt = rt
            cn, ln, imgn = rn

            # Also load masks for regionprops
            mask_t = imread(mt)
            mask_n = imread(mn)

            shared = set(lt) & set(ln)
            if len(shared) < args.min_cells:
                continue
            if len(lt) < args.min_cells or len(ln) < args.min_cells:
                continue
            idx_t = [i for i, l in enumerate(lt) if l in shared]
            idx_n = [i for i, l in enumerate(ln) if l in shared]
            ct_s, lt_s = ct[idx_t], lt[idx_t]
            cn_s, ln_s = cn[idx_n], ln[idx_n]

            # Extract patches (for CNN)
            pt_s = extract_patches(imgt, ct_s)
            pn_s = extract_patches(imgn, cn_s)

            # 7D regionprops features (Modes R, C) — aligned by shared labels
            feats_7d_t_list, feats_7d_n_list = [], []
            for l in lt_s:
                props = sk_regionprops(mask_t, intensity_image=imgt)
                for p in props:
                    if p.label == int(l):
                        feats_7d_t_list.append([
                            p.area, p.eccentricity, p.perimeter,
                            p.solidity if p.solidity is not None else 1.0,
                            p.extent if p.extent is not None else 1.0,
                            p.orientation if p.orientation is not None else 0.0,
                            p.intensity_mean if p.intensity_mean is not None else 0.0,
                        ])
                        break
            for l in ln_s:
                props = sk_regionprops(mask_n, intensity_image=imgn)
                for p in props:
                    if p.label == int(l):
                        feats_7d_n_list.append([
                            p.area, p.eccentricity, p.perimeter,
                            p.solidity if p.solidity is not None else 1.0,
                            p.extent if p.extent is not None else 1.0,
                            p.orientation if p.orientation is not None else 0.0,
                            p.intensity_mean if p.intensity_mean is not None else 0.0,
                        ])
                        break

            if len(feats_7d_t_list) < 2 or len(feats_7d_n_list) < 2:
                continue

            feat_7d_t = torch.tensor(feats_7d_t_list, dtype=torch.float32)
            feat_7d_n = torch.tensor(feats_7d_n_list, dtype=torch.float32)

            # CNN features (Mode C) via frozen NT-Xent pretrained CNN
            pt_t = torch.from_numpy(pt_s).float().unsqueeze(1).to(device)
            pn_t = torch.from_numpy(pn_s).float().unsqueeze(1).to(device)
            with torch.no_grad():
                cnn_t = cnn_model(pt_t).cpu()
                cnn_n = cnn_model(pn_t).cpu()

            pair_data[split_name].append({
                "feat_7d_t": feat_7d_t,
                "feat_7d_n": feat_7d_n,
                "cnn_t": cnn_t,
                "cnn_n": cnn_n,
                "coords_t": torch.from_numpy(ct_s).float(),
                "coords_n": torch.from_numpy(cn_s).float(),
                "labels_t": torch.from_numpy(lt_s).long(),
                "labels_n": torch.from_numpy(ln_s).long(),
            })

    logger.info(f"Race data: {len(pair_data['train'])} train, "
                f"{len(pair_data['val'])} val")
    if len(pair_data["train"]) < 1 or len(pair_data["val"]) < 1:
        logger.error("Not enough pairs after filtering.")
        sys.exit(1)

    # ─── Run convergence race ─────────────────────────────────────────────
    race_results = run_convergence_race(pair_data, args, pe_dim)

    # ─── Print comparison table ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("CONVERGENCE RACE — COMPARISON TABLE (Maxed-out A500 Config)")
    print("=" * 70)
    print(f"{'Mode':<12} | {'Val Loss':<12} | {'BalAcc':<12}")
    print("-" * 38)

    table_rows = []
    for mode in ["R", "C"]:
        if mode not in race_results:
            continue
        rows = race_results[mode]
        if not rows:
            continue
        final = rows[-1]
        v_loss = final["val_loss"]
        v_bal = final["val_bal_acc"]
        table_rows.append((mode, v_loss, v_bal))
        print(f"{mode:<12} | {v_loss:<12.4f} | {v_bal:<12.4f}")

    print("-" * 38)
    print()

    # Verdict
    print("VERDICT:")
    results_dict = {m: (vl, ba) for m, vl, ba in table_rows}
    if "C" in results_dict and "R" in results_dict:
        delta_bal = results_dict["C"][1] - results_dict["R"][1]
        if delta_bal > 0.02:
            print(f"  CNN residual IMPROVES over 7D baseline at scale "
                  f"(Δ={delta_bal:.4f} bal_acc)")
        elif delta_bal < -0.02:
            print(f"  CNN residual UNDERPERFORMS vs 7D baseline at scale "
                  f"(Δ={delta_bal:.4f} bal_acc)")
        else:
            print(f"  CNN residual similar to 7D baseline at scale "
                  f"(Δ={delta_bal:.4f} bal_acc)")

    print("=" * 70)
    logger.info("Convergence race done.")


if __name__ == "__main__":
    main()
