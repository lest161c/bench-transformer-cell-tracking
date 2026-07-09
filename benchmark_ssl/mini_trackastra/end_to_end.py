#!/usr/bin/env python3
"""End-to-end SSL -> downstream benchmark with MiniTrackingTransformer.

Tests whether fixing the coordinate shortcut in SSL pretraining speeds up
downstream convergence of the full tracking model.

Architecture:
  DINO patches -> DINOv2 frozen -> Linear(384, d_model) -> (+ PE) -> LayerNorm
    -> Encoder x2 (MultiheadSelfAttention + FFN)
    -> Decoder x2 (MultiheadCrossAttention: tgt=enc_all, memory=enc_all + FFN)
    -> head_x(enc_t), head_y(dec_n) -> assoc = head_x @ head_y^T -> BCE

Modes:
  A: FourierPE(coords)+DINO -> SSL encoder -> Full model (FourierPE)
  B: NoPE+DINO -> SSL encoder -> Full model (NoPE)
  S: FourierPE(shuffled)+DINO -> SSL encoder -> Full model (FourierPE) [coords adversarial]
  R: Random init -> Full model (FourierPE) [baseline]

Usage:
  cd benchmark_ssl/mini_trackastra
  .venv/bin/python end_to_end.py
  .venv/bin/python end_to_end.py --ssl-steps 50 --downstream-steps 20 \\
      --max-ssl-frames 8 --max-downstream-pairs 5 --d-model 64 --nhead 2
"""

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import regionprops_table
from skimage.measure import regionprops as sk_regionprops
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mini_trackastra")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DINO_DIM = 384
PATCH_SIZE = 64
CNN_FEAT_DIM = 64  # output dim of TinyCNN

# ─── TinyCNN (frozen residual injection) ──────────────────────────────────────

class TinyCNN(nn.Module):
    """Tiny CNN feature extractor. ~3K params."""
    def __init__(self, out_dim=CNN_FEAT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, patches):
        x = self.conv(patches)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ─── DINO backbone (lazy-loaded) ─────────────────────────────────────────────
_dino = None


def get_dino():
    global _dino
    if _dino is None:
        logger.info("Loading DINOv2 backbone...")
        _dino = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            .to(device)
            .eval()
        )
    elif next(_dino.parameters()).device != device:
        # Move back to GPU if it was freed
        _dino = _dino.to(device)
    return _dino


@torch.no_grad()
def compute_dino_embs(patches_np):
    if len(patches_np) == 0:
        return np.zeros((0, DINO_DIM), dtype=np.float32)
    pmin = patches_np.min(axis=(1, 2), keepdims=True)
    pmax = patches_np.max(axis=(1, 2), keepdims=True)
    pn = (patches_np - pmin) / (pmax - pmin + 1e-8)
    t = torch.from_numpy(pn).float().unsqueeze(1).to(device)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = t.expand(-1, 3, -1, -1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return get_dino()(t).cpu().numpy()


def extract_patches(img, centroids):
    h, w = img.shape[-2:]
    half = PATCH_SIZE // 2
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
            if y2c > y1c and x2c > x1c
            else np.zeros((1, 1), dtype=np.float32)
        )
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(
                crop,
                tuple(
                    (0, max(0, t)) for t in [PATCH_SIZE - s for s in crop.shape]
                ),
                mode="reflect",
            )[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)


# ─── Data loading ────────────────────────────────────────────────────────────

def load_frame(mask_path, img_path):
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2:
        return None
    coords = np.stack(
        [props["centroid-0"], props["centroid-1"]], axis=-1
    ).astype(np.float32)
    return coords, props["label"].astype(np.int32), img


def scan_frames(data_root, conditions, max_frames):
    dr = Path(data_root)
    frames = []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir():
                continue
            tra = exp / "TRA"
            img_dir = exp / "img"
            if not tra.exists() or not img_dir.exists():
                continue
            for m in sorted(tra.glob("man_track*.tif")):
                stem = m.stem.replace("man_track", "")
                try:
                    fi = int(stem)
                except ValueError:
                    continue
                ip = img_dir / f"t{fi:06d}.tif"
                if ip.exists():
                    frames.append((str(m), str(ip)))
                if len(frames) >= max_frames:
                    return frames
    return frames


def scan_consecutive_pairs(data_root, conditions, max_pairs):
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
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


# ─── Distortions (SSL view generation) ───────────────────────────────────────

def apply_jitter(c, std):
    return c + np.random.randn(*c.shape).astype(np.float32) * std


def apply_affine(c, deg, sr):
    t = np.random.uniform(-deg, deg) / 180 * np.pi
    sx, sy = np.random.uniform(*sr), np.random.uniform(*sr)
    return c @ np.array(
        [[sx * np.cos(t), -sx * np.sin(t)], [sy * np.sin(t), sy * np.cos(t)]]
    )


def apply_dropout(c, l, p):
    k = np.random.rand(len(l)) > p
    return c[k], l[k]


def distort(coords, labels, mode):
    if mode == "full":
        c = apply_affine(coords.copy(), 10, (0.9, 1.1))
        c = apply_jitter(c, 4)
        c, l = apply_dropout(c, labels.copy(), 0.1)
        return c, l
    # Default: jitter only (std=4)
    return apply_jitter(coords.copy(), 4), labels.copy()


# ─── Positional Encodings ────────────────────────────────────────────────────

class FourierPE(nn.Module):
    """Fourier feature encoding for coordinates."""

    def __init__(self, pos_per_dim=16, coord_dim=2):
        super().__init__()
        self.d = coord_dim * pos_per_dim * 2
        self.coord_dim = coord_dim
        self.pos_per_dim = pos_per_dim
        self.register_buffer("freqs", 2.0 ** torch.linspace(0.0, 10.0, pos_per_dim))

    def forward(self, c):
        """c: (B, N, coord_dim) -> (B, N, d)"""
        parts = []
        for i in range(self.coord_dim):
            parts.append(torch.sin(c[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        for i in range(self.coord_dim):
            parts.append(torch.cos(c[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        return torch.cat(parts, dim=-1)


class NoPE(nn.Module):
    """Learned constant positional encoding (no coordinate info)."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, c):
        return self.token.expand(c.shape[0], c.shape[1], -1)


# ─── MiniTrackingTransformer ─────────────────────────────────────────────────

class MiniEncoder(nn.Module):
    """Encoder: feat_proj + PE_proj + optional CNN residual + LayerNorm + TransformerEncoder."""

    def __init__(self, d_model=128, nhead=4, num_layers=2, pe_dim=64,
                 feat_dim=DINO_DIM, cnn_feat_dim=None):
        super().__init__()
        self.d_model = d_model
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.pe_proj = nn.Linear(pe_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.cnn_feat_dim = cnn_feat_dim

        if cnn_feat_dim is not None:
            self.cnn_proj = nn.Linear(cnn_feat_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True,
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
        """feats: (N, D), pe: (N, pe_dim), cnn_feats: (N, CNN_FEAT_DIM) -> (N, d_model)"""
        x = self.norm(self.pe_proj(pe) + self.feat_proj(feats))
        if self.cnn_feat_dim is not None and cnn_feats is not None:
            x = x + self.cnn_proj(cnn_feats)
        return self.encoder(x)


class MiniDecoder(nn.Module):
    """Decoder-only part: TransformerDecoder for cross-attention on pair."""

    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory):
        """tgt: (N, d_model), memory: (N, d_model) -> (N, d_model)"""
        return self.decoder(tgt, memory)


class MiniTrackingTransformer(nn.Module):
    """Full miniature Trackastra: encoder + decoder + association heads."""

    def __init__(
        self,
        encoder: MiniEncoder,
        decoder: MiniDecoder,
        d_head=32,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = encoder.d_model

        self.head_x = nn.Linear(self.d_model, d_head)
        self.head_y = nn.Linear(self.d_model, d_head)

        # Init heads
        for p in self.head_x.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.head_y.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_t, pe_t, feat_n, pe_n, cnn_t=None, cnn_n=None):
        """Full forward for a frame pair.

        Args:
            feat_t: (N1, D) frame t features (DINO or 7D regionprops)
            pe_t: (N1, pe_dim) frame t positional encoding
            feat_n: (N2, D) frame n features
            pe_n: (N2, pe_dim) frame n positional encoding
            cnn_t: (N1, CNN_FEAT_DIM) optional CNN residual features
            cnn_n: (N2, CNN_FEAT_DIM) optional CNN residual features

        Returns:
            logits: (N1, N2) association logits (NOT sigmoided)
        """
        # Encode both frames independently
        enc_t = self.encoder(feat_t, pe_t, cnn_t)  # (N1, d_model)
        enc_n = self.encoder(feat_n, pe_n, cnn_n)  # (N2, d_model)

        N1 = enc_t.shape[0]

        # Concat for joint decoder processing
        enc_all = torch.cat([enc_t, enc_n], dim=0)  # (N1+N2, d_model)

        # Decoder: cross-attention within the pair
        dec_all = self.decoder(enc_all, enc_all)  # (N1+N2, d_model)

        # Split: encoder output for frame t, decoder output for frame n
        dec_n = dec_all[N1:]  # (N2, d_model)

        # Association logits: head_x(enc_t) @ head_y(dec_n)^T
        # Normalize heads to unit vectors (like Trackastra normalize_output)
        # then scale by 1/temperature. Prevents logit explosion when encoder is pretrained.
        hx = F.normalize(self.head_x(enc_t), dim=-1)  # (N1, d_head)
        hy = F.normalize(self.head_y(dec_n), dim=-1)  # (N2, d_head)
        logits = (hx @ hy.T) * (self.head_x.out_features ** 0.5)  # (N1, N2)
        return logits


# ─── Loss functions ──────────────────────────────────────────────────────────

def nt_xent_loss(z, temp=0.05):
    """NT-Xent contrastive loss for paired views.

    Args:
        z: (2*N, D) where first N = view1, last N = view2, corresponding order.
    Returns:
        Scalar loss.
    """
    B = z.shape[0] // 2
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temp
    # Remove self-similarity diagonal
    sim = sim - torch.eye(2 * B, device=sim.device) * 1e9
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


@torch.no_grad()
def _build_target_matrix(labels_t, labels_n, device):
    """Build binary target matrix for association.

    target[i,j] = 1 if labels_t[i] == labels_n[j] else 0
    """
    N1, N2 = len(labels_t), len(labels_n)
    target = torch.zeros(N1, N2, device=device)
    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_n):
            if lt == ln:
                target[i, j] = 1.0
    return target


def bce_assoc_loss(logits, labels_t, labels_n):
    """BCE loss on association logits (uses BCEWithLogitsLoss internally).

    Positive pairs: cells with the same label.
    Negative pairs: cells with different labels.

    Args:
        logits: (N1, N2) raw logits (NOT sigmoided)
        labels_t: (N1,) frame t cell labels
        labels_n: (N2,) frame n cell labels
    Returns:
        Scalar loss.
    """
    target = _build_target_matrix(labels_t, labels_n, logits.device)
    return F.binary_cross_entropy_with_logits(logits, target)


def assoc_accuracy(logits, labels_t, labels_n):
    """Accuracy at threshold 0.5 (applies sigmoid to logits)."""
    pred = (torch.sigmoid(logits) > 0.5).cpu()
    target = _build_target_matrix(labels_t, labels_n, "cpu").bool()
    return (pred == target).float().mean().item()


def balanced_accuracy(logits, labels_t, labels_n):
    """Balanced accuracy: mean of per-class accuracy.
    
    Handles class imbalance: with N positive and N²-N negative pairs,
    standard accuracy = (N_correct_neg + N_correct_pos) / N² is dominated by negatives.
    Balanced accuracy = (recall_pos + recall_neg) / 2.
    """
    pred = (torch.sigmoid(logits) > 0.5).cpu()
    target = _build_target_matrix(labels_t, labels_n, "cpu").bool()
    
    pos_mask = target
    neg_mask = ~target
    
    pos_acc = (pred[pos_mask] == target[pos_mask]).float().mean().item() if pos_mask.any() else 0.5
    neg_acc = (pred[neg_mask] == target[neg_mask]).float().mean().item() if neg_mask.any() else 0.5
    
    return (pos_acc + neg_acc) / 2.0


# ─── Free DINO from GPU ──────────────────────────────────────────────────────

@torch.no_grad()
def free_dino():
    """Free DINO backbone from GPU memory after precomputation."""
    global _dino
    if _dino is not None:
        _dino = _dino.cpu()
        torch.cuda.empty_cache()
        logger.info("DINO freed from GPU memory")


# ─── 7D Regionprops features ──────────────────────────────────────────────────

@torch.no_grad()
def extract_regionprops_7d(mask, img):
    """Extract 7D regionprops: area, ecc, perimeter, solidity, extent, orient, mean_int.

    Returns (coords, labels, feats) or (None, None, None).
    """
    props_list = sk_regionprops(mask, intensity_image=img)
    if len(props_list) == 0:
        return None, None, None
    N = len(props_list)
    feats = np.zeros((N, 7), dtype=np.float32)
    coords = np.zeros((N, 2), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int32)
    for i, r in enumerate(props_list):
        feats[i, 0] = r.area
        feats[i, 1] = r.eccentricity
        feats[i, 2] = r.perimeter
        feats[i, 3] = r.solidity if r.solidity is not None else 1.0
        feats[i, 4] = r.extent if r.extent is not None else 1.0
        feats[i, 5] = r.orientation if r.orientation is not None else 0.0
        feats[i, 6] = r.intensity_mean if r.intensity_mean is not None else 0.0
        coords[i] = r.centroid
        labels[i] = r.label
    return coords, labels, feats


# ─── CNN model loading (frozen) ───────────────────────────────────────────────

_cnn_model = None


def load_cnn_model(cnn_path):
    """Load frozen TinyCNN from saved state_dict."""
    global _cnn_model
    if _cnn_model is None:
        logger.info(f"Loading TinyCNN from {cnn_path}")
        _cnn_model = TinyCNN(out_dim=CNN_FEAT_DIM).to(device)
        _cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
        _cnn_model.eval()
        for p in _cnn_model.parameters():
            p.requires_grad = False
        logger.info(f"TinyCNN loaded (frozen, {sum(p.numel() for p in _cnn_model.parameters())} params)")
    return _cnn_model


@torch.no_grad()
def compute_cnn_feats(patches_np):
    """Compute TinyCNN features for (N, 64, 64) numpy patches."""
    if len(patches_np) == 0:
        return np.zeros((0, CNN_FEAT_DIM), dtype=np.float32)
    t = torch.from_numpy(patches_np).float().unsqueeze(1).to(device)  # (N, 1, 64, 64)
    return _cnn_model(t).cpu().numpy()


# ─── Convergence Race ─────────────────────────────────────────────────────────

def run_convergence_race(pair_data, args, pe_dim, cnn_path=None):
    """Run 3-mode convergence race (no SSL): R, D, C.

    Args:
        pair_data: dict with "train"/"val" lists of items containing
            the appropriate feature keys.
        args: parsed arguments
        pe_dim: positional encoding dimension
        cnn_path: path to saved TinyCNN weights (for Mode C)

    Returns:
        dict: {mode: pd.DataFrame} with columns [step, train_loss, val_loss, val_bal_acc]
    """
    # The pair_data items for convergence race have different keys:
    # For Mode D (DINO):     "dino_t", "dino_n", "coords_t", "coords_n", "labels_t", "labels_n"
    # For Modes R/C (7D):    "feat_7d_t", "feat_7d_n", "coords_t", "coords_n", "labels_t", "labels_n"
    # For Mode C (CNN):      also "cnn_t", "cnn_n"

    # Build 3 encoders
    # Mode R: 7D regionprops only, no CNN, no DINO
    # Mode D: DINO + PE (standard)
    # Mode C: 7D + PE + CNN residual

    enc_r = MiniEncoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_encoder_layers, pe_dim=pe_dim,
        feat_dim=7,  # 7D regionprops
    ).to(device)
    pe_r = FourierPE(pos_per_dim=args.pos_per_dim).to(device)

    enc_d = MiniEncoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_encoder_layers, pe_dim=pe_dim,
        feat_dim=DINO_DIM,
    ).to(device)
    pe_d = FourierPE(pos_per_dim=args.pos_per_dim).to(device)

    enc_c = MiniEncoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_encoder_layers, pe_dim=pe_dim,
        feat_dim=7, cnn_feat_dim=CNN_FEAT_DIM,
    ).to(device)
    pe_c = FourierPE(pos_per_dim=args.pos_per_dim).to(device)

    configs = [
        ("R", "7D regionprops only", enc_r, pe_r, "feat_7d", None),
        ("D", "DINO + PE",          enc_d, pe_d, "dino",    None),
        ("C", "7D + PE + CNN",      enc_c, pe_c, "feat_7d", "cnn"),
    ]

    results = {}
    for mode, mode_name, enc, pe_mod, feat_key, cnn_key in configs:
        dec = MiniDecoder(
            d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_decoder_layers,
        ).to(device)
        model = MiniTrackingTransformer(
            encoder=enc, decoder=dec, d_head=args.d_head
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.downstream_lr)
        logger.info(f"\n  Convergence Race [{mode}] {mode_name}")

        rows = []
        for step in range(args.downstream_steps):
            model.train()
            train_losses, train_bal_accs = [], []
            for pd_item in pair_data["train"]:
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
                loss = bce_assoc_loss(scores, lt, ln)
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.item())
                sd = scores.detach()
                train_bal_accs.append(balanced_accuracy(sd, lt, ln))

            if step % 10 == 0 or step == args.downstream_steps - 1:
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
                        val_losses.append(bce_assoc_loss(scores, lt, ln).item())
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

        results[mode] = pd.DataFrame(rows)
        if rows:
            final = rows[-1]
            logger.info(
                f"    [{mode}] done: val_loss={final['val_loss']:.4f}  "
                f"val_bal_acc={final['val_bal_acc']:.4f}"
            )

    return results


def print_convergence_race_summary(results, downstream_steps):
    """Print the comparison table with verdict."""
    print()
    print("=" * 70)
    print("CONVERGENCE RACE — SUMMARY")
    print("=" * 70)

    # Find steps to bal_acc > 0.65 for each mode
    info = {}
    for mode in ["R", "D", "C"]:
        if mode not in results:
            continue
        df = results[mode]
        final = df.iloc[-1]
        v_loss = final["val_loss"]
        v_bal = final["val_bal_acc"]
        hit = df[df["val_bal_acc"] > 0.65]
        steps_to_065 = hit["step"].iloc[0] if len(hit) else downstream_steps
        info[mode] = (v_loss, v_bal, steps_to_065)

    mode_labels = {
        "R": "Random (7D only)",
        "D": "DINO residual",
        "C": "CNN residual",
    }

    print(f"\n{'Mode':<20} | {'Val Loss @ 100':<15} | {'BalAcc @ 100':<14} | {'Steps to bal_acc > 0.65':<25}")
    print("-" * 80)
    for mode in ["R", "D", "C"]:
        if mode not in info:
            continue
        vl, ba, st = info[mode]
        print(f"{mode_labels[mode]:<20} | {vl:<15.4f} | {ba:<14.4f} | {st:<25}")
    print("-" * 80)

    # Verdict
    print()
    print("VERDICT:")
    if "R" in info and "C" in info:
        delta_r = info["R"][2]
        delta_c = info["C"][2]
        if delta_c < delta_r:
            print(f"  ✓ CNN injection speeds up convergence over 7D baseline "
                  f"({delta_r - delta_c} steps faster to bal_acc>0.65)")
        elif delta_c == delta_r:
            print(f"  ∼ CNN injection same convergence speed as 7D baseline "
                  f"(both {delta_r} steps)")
        else:
            print(f"  ✗ CNN injection does NOT speed up convergence over 7D baseline "
                  f"({delta_c - delta_r} steps slower)")

    if "D" in info and "R" in info:
        delta_d = info["D"][2]
        delta_r = info["R"][2]
        if delta_d < delta_r:
            print(f"  ✓ DINO residual converges {delta_r - delta_d} steps faster than 7D baseline")
        elif delta_d > delta_r:
            print(f"  ✗ DINO residual converges {delta_d - delta_r} steps slower than 7D baseline")
        else:
            print(f"  ∼ DINO residual same speed as 7D baseline")

    print("=" * 70)


# ─── SSL Phase ───────────────────────────────────────────────────────────────

def run_ssl_phase(ssl_data, args, pe_dim):
    """Train SSL encoder for Modes A and B using NT-Xent contrastive loss.

    Args:
        ssl_data: list of (e1, e2, c1, c2, lbl) with precomputed DINO features
        args: parsed arguments
        pe_dim: positional encoding dimension

    Returns:
        dict: {mode: {"encoder": MiniEncoder, "pos_enc": PE_module}}
    """
    logger.info(f"SSL data: {len(ssl_data)} frames loaded")

    # Build PE modules
    fourier_pe = FourierPE(pos_per_dim=args.pos_per_dim).to(device)
    noise_pe = NoPE(pe_dim).to(device)

    configs = [
        ("A", fourier_pe, "FourierPE + DINO", False),
        ("B", noise_pe, "NoPE + DINO", False),
        ("S", fourier_pe, "FourierPE + shuffled coords + DINO", True),
    ]

    # Filter by requested modes
    requested = [m.strip() for m in args.ssl_modes.split(",")]
    configs = [c for c in configs if c[0] in requested]

    encoders = {}
    for mode, pe, desc, use_shuffled in configs:
        enc = MiniEncoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_encoder_layers,
            pe_dim=pe_dim,
        ).to(device)

        opt = torch.optim.Adam(enc.parameters(), lr=args.ssl_lr)
        logger.info(f"  SSL Mode {mode} ({desc}): {args.ssl_steps} steps")

        # Move data to GPU once
        gpu_data = [
            (
                torch.from_numpy(e1).float().to(device),
                torch.from_numpy(e2).float().to(device),
                torch.from_numpy(c1).float().unsqueeze(0).to(device),
                torch.from_numpy(c2_shuffled if use_shuffled else c2).float().unsqueeze(0).to(device),
                lbl,
            )
            for e1, e2, c1, c2, c2_shuffled, lbl in ssl_data
        ]

        for step in range(args.ssl_steps):
            enc.train()
            loss_total = 0.0
            n_batches = 0
            # Shuffle data each step to prevent memorization
            perm = torch.randperm(len(gpu_data))
            for idx in perm:
                de1, de2, c1, c2, lbl = gpu_data[idx]
                n = min(len(de1), len(de2))
                if n < 2:
                    continue
                pe1 = pe(c1).squeeze(0)[:n]
                pe2 = pe(c2).squeeze(0)[:n]
                z1 = enc(de1[:n], pe1)
                z2 = enc(de2[:n], pe2)
                z = torch.cat([z1, z2])
                loss = nt_xent_loss(z)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
                loss_total += loss.item()
                n_batches += 1
            if (step + 1) % 50 == 0 or step == 0:
                avg_loss = loss_total / max(1, n_batches)
                logger.info(f"    [{mode}] step {step+1:4d}: loss={avg_loss:.4f}")

        encoders[mode] = {"encoder": enc, "pos_enc": pe}
        final_loss = loss_total / max(1, n_batches)
        logger.info(f"    [{mode}] done: final loss={final_loss:.6f}")

    return encoders


# ─── Downstream Phase ────────────────────────────────────────────────────────

def run_downstream(encoders, pe_dim, pair_data, args):
    """Train full MiniTrackingTransformer with BCE on frame pairs.

    Tests four modes:
      A: SSL with FourierPE, downstream with FourierPE
      B: SSL with NoPE, downstream with NoPE
      S: SSL with FourierPE + shuffled coords, downstream with FourierPE
      R: Random init (no SSL), downstream with FourierPE

    Args:
        encoders: dict from run_ssl_phase
        pe_dim: positional encoding dimension
        pair_data: dict with "train" and "val" lists of precomputed items
        args: parsed arguments

    Returns:
        dict: {mode: pd.DataFrame} with columns [step, train_loss, train_acc,
              val_loss, val_acc]
    """
    if len(pair_data.get("train", [])) < 2:
        logger.error("Not enough training pairs. Need more data.")
        return None

    # Define test modes
    test_modes = [
        ("A", "FourierPE (SSL w/ coords)", encoders.get("A")),
        ("B", "NoPE (SSL w/o coords)", encoders.get("B")),
        ("S", "FourierPE (SSL w/ shuffled coords)", encoders.get("S")),
        ("R", "Random init baseline", None),
    ]

    results = {}
    for mode, mode_name, ssl_info in test_modes:
        if mode in ("A", "B", "S") and ssl_info is None:
            logger.warning(f"  Mode {mode}: SSL encoder missing, skipping")
            continue

        # Build encoder
        if mode == "R":
            enc = MiniEncoder(
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_encoder_layers,
                pe_dim=pe_dim,
            ).to(device)
            pe_mod = FourierPE(pos_per_dim=args.pos_per_dim).to(device)
        else:
            # Deep copy SSL encoder weights
            enc = MiniEncoder(
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_encoder_layers,
                pe_dim=pe_dim,
            ).to(device)
            ssl_enc = ssl_info["encoder"]
            enc.load_state_dict(ssl_enc.state_dict())
            pe_mod = ssl_info["pos_enc"]

        # Build decoder + full model
        dec = MiniDecoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_decoder_layers,
        ).to(device)

        model = MiniTrackingTransformer(
            encoder=enc, decoder=dec, d_head=args.d_head
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.downstream_lr)

        logger.info(f"\n  Downstream [{mode}] {mode_name}")

        rows = []
        for step in range(args.downstream_steps):
            model.train()
            train_losses, train_accs, train_bal_accs = [], [], []
            for pd_item in pair_data["train"]:
                de_t = pd_item["dino_t"].to(device)
                de_n = pd_item["dino_n"].to(device)
                ct = pd_item["coords_t"].unsqueeze(0).to(device)
                cn = pd_item["coords_n"].unsqueeze(0).to(device)
                lt = pd_item["labels_t"]
                ln = pd_item["labels_n"]

                pe_t = pe_mod(ct).squeeze(0)
                pe_n = pe_mod(cn).squeeze(0)

                scores = model(de_t, pe_t, de_n, pe_n)
                loss = bce_assoc_loss(scores, lt, ln)
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.item())
                sd = scores.detach()
                train_accs.append(assoc_accuracy(sd, lt, ln))
                train_bal_accs.append(balanced_accuracy(sd, lt, ln))

            # Evaluate every 10 steps
            if step % 10 == 0 or step == args.downstream_steps - 1:
                model.eval()
                val_losses, val_accs, val_bal_accs = [], [], []
                with torch.no_grad():
                    for pd_item in pair_data["val"]:
                        de_t = pd_item["dino_t"].to(device)
                        de_n = pd_item["dino_n"].to(device)
                        ct = pd_item["coords_t"].unsqueeze(0).to(device)
                        cn = pd_item["coords_n"].unsqueeze(0).to(device)
                        lt = pd_item["labels_t"]
                        ln = pd_item["labels_n"]

                        pe_t = pe_mod(ct).squeeze(0)
                        pe_n = pe_mod(cn).squeeze(0)

                        scores = model(de_t, pe_t, de_n, pe_n)
                        val_losses.append(
                            bce_assoc_loss(scores, lt, ln).item()
                        )
                        val_accs.append(assoc_accuracy(scores, lt, ln))
                        val_bal_accs.append(balanced_accuracy(scores, lt, ln))

                t_loss = float(np.mean(train_losses))
                t_acc = float(np.mean(train_accs))
                v_loss = float(np.mean(val_losses))
                v_acc = float(np.mean(val_accs))
                v_bal = float(np.mean(val_bal_accs))
                rows.append(
                    {
                        "step": step,
                        "train_loss": t_loss,
                        "train_acc": t_acc,
                        "val_loss": v_loss,
                        "val_acc": v_acc,
                        "val_bal_acc": v_bal,
                    }
                )

        results[mode] = pd.DataFrame(rows)
        if rows:
            final = rows[-1]
            logger.info(
                f"    [{mode}] done: val_loss={final['val_loss']:.4f}  "
                f"val_acc={final['val_acc']:.4f}  bal_acc={final['val_bal_acc']:.4f}"
            )

    return results


# ─── Plotting and Reporting ──────────────────────────────────────────────────

def make_report(downstream_results, outdir, args):
    """Generate convergence plot, CSVs, and verdict text."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"A": "#e74c3c", "B": "#2ecc71", "S": "#3498db", "R": "#95a5a6"}
    labels_map = {
        "A": "Mode A: FourierPE + DINO (SSL w/ coords)",
        "B": "Mode B: NoPE + DINO (SSL w/o coords)",
        "S": "Mode S: FourierPE + DINO (SSL w/ shuffled coords)",
        "R": "Mode R: Random init (baseline)",
    }

    plot_order = sorted(
        set(downstream_results.keys()),
        key=lambda m: {"R": "z", "A": "a", "B": "b", "S": "c"}.get(m, m),
    )

    # Panel 1: Val loss
    ax = axes[0]
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        ax.plot(
            df["step"],
            df["val_loss"],
            color=colors.get(mode, "#333"),
            label=labels_map.get(mode, mode),
            linewidth=2,
        )
    ax.set_xlabel("Downstream training step")
    ax.set_ylabel("Val BCE Loss")
    ax.set_title("Downstream Convergence: Val Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Balanced accuracy (fair metric for imbalanced +/−)
    ax = axes[1]
    for mode in plot_order:
        if mode not in downstream_results: continue
        df = downstream_results[mode]
        ax.plot(df["step"], df.get("val_bal_acc", df["val_acc"]),
                color=colors.get(mode, "#333"), linewidth=2)
    ax.set_xlabel("Downstream training step")
    ax.set_ylabel("Val Balanced Accuracy")
    ax.set_title("Downstream Convergence: Balanced Acc (baseline=50%)")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: Convergence speed (steps to bal_acc > 0.7)
    ax = axes[2]
    mode_names = []; conv_steps = []; bar_colors = []
    bar_order = sorted(plot_order, key=lambda m: {"R": 99, "A": 1, "B": 2, "S": 3}.get(m, 50))
    for mode in bar_order:
        if mode not in downstream_results: continue
        df = downstream_results[mode]
        acc_col = "val_bal_acc" if "val_bal_acc" in df.columns else "val_acc"
        hit = df[df[acc_col] > 0.7]
        s = hit["step"].iloc[0] if len(hit) else args.downstream_steps
        mode_names.append(labels_map.get(mode, mode).split("(")[0].strip())
        conv_steps.append(s)
        bar_colors.append(colors.get(mode, "#333"))
    bars = ax.bar(mode_names, conv_steps, color=bar_colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Steps to bal_acc > 0.7"); ax.set_title("Convergence Speed (lower = faster)")
    for bar, v in zip(bars, conv_steps):
        label = str(v) if v < args.downstream_steps else "never"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, label, ha="center", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"MiniTrackingTransformer: SSL -> Downstream Transfer\n"
        f"{args.max_ssl_frames} SSL frames, {args.ssl_steps} SSL steps, "
        f"{args.downstream_steps} downstream steps",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_path = outdir / "convergence_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {fig_path}")

    # ─── Save CSVs ───────────────────────────────────────────────────────
    summary_rows = []
    for mode in ["A", "B", "S", "R"]:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        csv_path = outdir / f"downstream_{mode}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved: {csv_path}")

        # Summary stats
        final = df.iloc[-1]
        initial = df.iloc[0]
        acc_final = final["val_acc"]
        loss_final = final["val_loss"]
        acc0 = initial["val_acc"]
        hit = df[df["val_acc"] > 0.8]
        conv_steps = hit["step"].iloc[0] if len(hit) else args.downstream_steps
        summary_rows.append(
            {
                "mode": mode,
                "val_acc_start": acc0,
                "val_acc_final": acc_final,
                "val_loss_final": loss_final,
                "steps_to_acc_08": conv_steps,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved: {summary_path}")

    # ─── Verdict text ────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("MINI TRACKASTRA: SSL -> DOWNSTREAM TRANSFER -- VERDICT")
    lines.append("=" * 65)
    lines.append("")

    for mode in ["R", "A", "B"]:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        final = df.iloc[-1]
        initial = df.iloc[0]
        acc = final["val_acc"]
        loss = final["val_loss"]
        acc0 = initial["val_acc"]
        hit = df[df["val_acc"] > 0.8]
        conv = hit["step"].iloc[0] if len(hit) else args.downstream_steps
        lines.append(f"  {labels_map[mode]}:")
        lines.append(f"    Val acc:  {acc0:.3f} -> {acc:.3f}  "
                     f"(Delta={acc-acc0:+.3f})")
        lines.append(f"    Val loss: {loss:.4f}")
        lines.append(f"    Steps to acc>0.8: {conv}")
        lines.append("")

    # Compare B vs A
    if "A" in downstream_results and "B" in downstream_results:
        df_a = downstream_results["A"]
        df_b = downstream_results["B"]
        hit_a = df_a[df_a["val_acc"] > 0.8]
        hit_b = df_b[df_b["val_acc"] > 0.8]
        conv_a = hit_a["step"].iloc[0] if len(hit_a) else args.downstream_steps
        conv_b = hit_b["step"].iloc[0] if len(hit_b) else args.downstream_steps
        acc_a = df_a.iloc[-1]["val_acc"]
        acc_b = df_b.iloc[-1]["val_acc"]

        if conv_b < conv_a - 2:
            lines.append(
                f"  *** Mode B converges {conv_a - conv_b} steps FASTER "
                f"than Mode A ***"
            )
            lines.append(
                "    -> Removing coordinates during SSL IMPROVES "
                "downstream convergence."
            )
        elif conv_b > conv_a + 2:
            lines.append(
                f"  *** Mode A converges {conv_b - conv_a} steps FASTER "
                f"than Mode B ***"
            )
        else:
            lines.append(
                f"  ~ Similar convergence speed (Delta={conv_a - conv_b} steps)"
            )

        if acc_b > acc_a + 0.02:
            lines.append(
                f"  Mode B achieves higher final accuracy "
                f"(+{acc_b - acc_a:.3f})"
            )
        elif acc_a > acc_b + 0.02:
            lines.append(
                f"  Mode A achieves higher final accuracy "
                f"(+{acc_a - acc_b:.3f})"
            )
        else:
            lines.append(
                f"  ~ Similar final accuracy (Delta={acc_b - acc_a:.3f})"
            )

    # Mode R comparison
    if "R" in downstream_results:
        df_r = downstream_results["R"]
        hit_r = df_r[df_r["val_acc"] > 0.8]
        conv_r = hit_r["step"].iloc[0] if len(hit_r) else args.downstream_steps
        acc_r = df_r.iloc[-1]["val_acc"]

        for mode in ["A", "B"]:
            if mode not in downstream_results:
                continue
            df_m = downstream_results[mode]
            hit_m = df_m[df_m["val_acc"] > 0.8]
            conv_m = hit_m["step"].iloc[0] if len(hit_m) else args.downstream_steps
            acc_m = df_m.iloc[-1]["val_acc"]
            delta_s = conv_r - conv_m
            delta_a = acc_m - acc_r
            lines.append(f"\n  Mode {mode} vs Random:")
            if delta_s > 2:
                lines.append(
                    f"    Converges {delta_s} steps FASTER than random"
                )
            elif delta_s < -2:
                lines.append(
                    f"    Converges {-delta_s} steps SLOWER than random"
                )
            else:
                lines.append("    Similar convergence speed to random")
            if delta_a > 0.02:
                lines.append(f"    Final accuracy +{delta_a:.3f} better")
            elif delta_a < -0.02:
                lines.append(f"    Final accuracy {delta_a:.3f} worse")
            else:
                lines.append("    Similar final accuracy")

    lines.append("")
    lines.append("  Interpretation:")
    if "B" in downstream_results and "A" in downstream_results:
        conv_b_val = (
            conv_b if isinstance(conv_b, int) else args.downstream_steps
        )
        conv_a_val = (
            conv_a if isinstance(conv_a, int) else args.downstream_steps
        )
        if conv_b_val < conv_a_val:
            lines.append(
                "    SSL WITHOUT coordinates -> FASTER downstream convergence"
            )
            lines.append(
                "    The coordinate shortcut is REAL and REMOVING it helps."
            )
            lines.append(
                "    Fix: NoPE during SSL pretraining."
            )
        else:
            lines.append(
                "    SSL with coordinates converges at least as fast."
            )
            lines.append(
                "    The coordinate shortcut may not hurt downstream "
                "at this scale,"
            )
            lines.append(
                "    or the decoder gap / NT-Xent/BCE mismatch dominates."
            )

    lines.append("")
    lines.append(f"Full results: {outdir}")
    lines.append(f"  {outdir}/convergence_comparison.png")
    for mode in ["A", "B", "R"]:
        if mode in downstream_results:
            lines.append(f"  {outdir}/downstream_{mode}.csv")
    lines.append(f"  {outdir}/summary.csv")
    lines.append("=" * 65)

    verdict = "\n".join(lines)
    print("\n" + verdict)
    with open(outdir / "verdict.txt", "w") as f:
        f.write(verdict)
    logger.info(f"Verdict saved: {outdir / 'verdict.txt'}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MiniTrackingTransformer SSL -> downstream benchmark"
    )
    p.add_argument("--data-root", default="../../data/vanvliet")
    p.add_argument("--conditions", default="rpsM")
    p.add_argument(
        "--max-ssl-frames", type=int, default=25,
        help="Max single frames for SSL phase"
    )
    p.add_argument(
        "--max-downstream-pairs", type=int, default=15,
        help="Max consecutive frame pairs for downstream"
    )
    p.add_argument(
        "--min-cells", type=int, default=8,
        help="Minimum cells per frame (and shared) for downstream pairs. "
             "Filters out trivial frames with < 2 cells."
    )
    p.add_argument("--ssl-steps", type=int, default=200)
    p.add_argument("--ssl-lr", type=float, default=1e-3)
    p.add_argument("--downstream-steps", type=int, default=100)
    p.add_argument("--downstream-lr", type=float, default=1e-3)

    # Architecture
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--num-decoder-layers", type=int, default=2)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--pos-per-dim", type=int, default=16)
    p.add_argument("--pe-dim-coord", type=int, default=2)

    # Distortion
    p.add_argument(
        "--distortion", default="full",
        choices=["jitter4", "full"]
    )
    p.add_argument(
        "--shuffle-coords", action="store_true",
        help="Shuffle coordinates across cells in view 2 during SSL. "
             "Forces model to learn position-invariant identity features."
    )

    # Output
    p.add_argument("--outdir", default="runs/mini_trackastra")
    p.add_argument("--label-fraction", type=float, default=1.0,
                   help="Fraction of labeled training pairs, e.g. 0.1 for 10pct")
    p.add_argument("--ssl-modes", default="A,B,S",
                   help="Comma-separated modes to run: A,B,S")

    # Convergence race args
    p.add_argument("--cnn-path", default=None,
                   help="Path to saved TinyCNN state_dict for Mode C")
    p.add_argument("--race", action="store_true",
                   help="Run convergence race (Modes R,D,C) instead of SSL benchmark")

    return p.parse_args(argv)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]

    pe_dim = args.pos_per_dim * args.pe_dim_coord * 2

    if args.race:
        # ═══════════════════════════════════════════════════════════════════
        #  CONVERGENCE RACE PATH (no SSL)
        # ═══════════════════════════════════════════════════════════════════
        logger.info(f"{'='*60}")
        logger.info(f"CONVERGENCE RACE: R (7D) vs D (DINO) vs C (7D+CNN)")
        logger.info(f"{'='*60}")
        logger.info(f"  d_model={args.d_model}, nhead={args.nhead}, "
                     f"enc={args.num_encoder_layers}, dec={args.num_decoder_layers}")
        logger.info(f"  PE dim: {pe_dim}")
        logger.info(f"  Downstream: {args.downstream_steps} steps, lr={args.downstream_lr}")
        logger.info(f"  Pairs: {args.max_downstream_pairs}")
        logger.info(f"  CNN path: {args.cnn_path}")
        logger.info(f"  Device: {device}")

        # Scan consecutive frame pairs
        logger.info("Scanning consecutive frame pairs...")
        pairs = scan_consecutive_pairs(
            args.data_root, conditions, args.max_downstream_pairs
        )
        logger.info(f"Found {len(pairs)} consecutive frame pairs")
        if len(pairs) < 3:
            logger.error("Need at least 3 pairs. Increase --max-downstream-pairs.")
            sys.exit(1)

        # Load frozen CNN if --cnn-path provided
        if args.cnn_path:
            load_cnn_model(args.cnn_path)

        # Split train/val
        np.random.seed(SEED + 1)
        idx = np.random.permutation(len(pairs))
        n_val = max(1, int(len(pairs) * 0.2))
        train_pairs = [pairs[i] for i in idx[:-n_val]]
        val_pairs = [pairs[i] for i in idx[-n_val:]]
        logger.info(f"Pairs: {len(train_pairs)} train + {len(val_pairs)} val")

        # Precompute features for all modes
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
                min_shared = getattr(args, 'min_cells', 8)
                if len(shared) < min_shared:
                    continue
                if len(lt) < min_shared or len(ln) < min_shared:
                    continue
                idx_t = [i for i, l in enumerate(lt) if l in shared]
                idx_n = [i for i, l in enumerate(ln) if l in shared]
                ct_s, lt_s = ct[idx_t], lt[idx_t]
                cn_s, ln_s = cn[idx_n], ln[idx_n]

                # Extract patches (for DINO and CNN)
                pt_s = extract_patches(imgt, ct_s)
                pn_s = extract_patches(imgn, cn_s)

                # DINO features (Mode D)
                dino_t = torch.from_numpy(compute_dino_embs(pt_s)).float()
                dino_n = torch.from_numpy(compute_dino_embs(pn_s)).float()

                # 7D regionprops features (Modes R, C)
                _, _, feats_7d_t = extract_regionprops_7d(mask_t, imgt)
                _, _, feats_7d_n = extract_regionprops_7d(mask_n, imgn)
                # Align with shared labels
                if feats_7d_t is not None and feats_7d_n is not None:
                    # Use regionprops ordering (by label)
                    # Build label->index maps
                    rp_labels_t = np.array(sorted([int(l) for l in lt]))
                    rp_labels_n = np.array(sorted([int(l) for l in ln]))
                    # Actually, we need to map the shared labels to regionprops indices
                    # extract_regionprops_7d returns props sorted by label
                    # Let's use a simpler approach: just use the label ordering directly
                    pass

                # Re-extract 7D features ensuring label alignment
                # Use masks directly for the shared cells
                feats_7d_t_list, feats_7d_n_list = [], []
                for l in lt_s:  # shared labels in frame t order
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

                # CNN features (Mode C) via frozen TinyCNN
                if args.cnn_path:
                    cnn_t = torch.from_numpy(compute_cnn_feats(pt_s)).float()
                    cnn_n = torch.from_numpy(compute_cnn_feats(pn_s)).float()
                else:
                    cnn_t = cnn_n = None

                pair_data[split_name].append({
                    "dino_t": dino_t,
                    "dino_n": dino_n,
                    "feat_7d_t": feat_7d_t,
                    "feat_7d_n": feat_7d_n,
                    "cnn_t": cnn_t,
                    "cnn_n": cnn_n,
                    "coords_t": torch.from_numpy(ct_s).float(),
                    "coords_n": torch.from_numpy(cn_s).float(),
                    "labels_t": torch.from_numpy(lt_s).long(),
                    "labels_n": torch.from_numpy(ln_s).long(),
                })

        logger.info(
            f"Race data: {len(pair_data['train'])} train, "
            f"{len(pair_data['val'])} val"
        )
        if len(pair_data["train"]) < 1 or len(pair_data["val"]) < 1:
            logger.error("Not enough pairs after filtering.")
            sys.exit(1)

        # Free DINO from GPU
        free_dino()

        # Run convergence race
        race_results = run_convergence_race(pair_data, args, pe_dim, cnn_path=args.cnn_path)

        # Print summary
        print_convergence_race_summary(race_results, args.downstream_steps)

        logger.info("Convergence race done.")
        return

    # ═════════════════════════════════════════════════════════════════════
    #  NORMAL SSL BENCHMARK PATH
    # ═════════════════════════════════════════════════════════════════════
    logger.info(
        f"MiniTrackingTransformer benchmark\n"
        f"  Architecture: d_model={args.d_model}, nhead={args.nhead}, "
        f"enc_layers={args.num_encoder_layers}, "
        f"dec_layers={args.num_decoder_layers}\n"
        f"  PE dim: {pe_dim} "
        f"(pos_per_dim={args.pos_per_dim}, coord_dim={args.pe_dim_coord})\n"
        f"  SSL: {args.ssl_steps} steps, lr={args.ssl_lr}, "
        f"frames={args.max_ssl_frames}\n"
        f"  Downstream: {args.downstream_steps} steps, lr={args.downstream_lr}, "
        f"pairs={args.max_downstream_pairs}\n"
        f"  Distortion: {args.distortion}\n"
        f"  Device: {device}"
    )

    # ─── Phase 1: Collect data ──────────────────────────────────────────
    logger.info("Scanning single frames for SSL...")
    frames = scan_frames(args.data_root, conditions, args.max_ssl_frames)
    logger.info(f"Found {len(frames)} single frames for SSL")

    if len(frames) < 2:
        logger.error("Need at least 2 frames for SSL. Check --data-root.")
        sys.exit(1)

    logger.info("Scanning consecutive frame pairs for downstream...")
    pairs = scan_consecutive_pairs(
        args.data_root, conditions, args.max_downstream_pairs
    )
    logger.info(f"Found {len(pairs)} consecutive frame pairs")

    if len(pairs) < 3:
        logger.error(
            "Need at least 3 consecutive frame pairs. "
            "Increase --max-downstream-pairs or check data."
        )
        sys.exit(1)

    # ─── Phase 2: Precompute ALL DINO features upfront ─────────────────
    ssl_data = []
    n_loaded = 0
    for mp, ip in frames:
        if n_loaded >= args.max_ssl_frames:
            break
        r = load_frame(mp, ip)
        if r is None:
            continue
        c1, lbl, img = r
        c2, _ = distort(c1.copy(), lbl.copy(), args.distortion)
        c2_shuffled = c2[np.random.permutation(len(c2))].copy()
        p1 = extract_patches(img, c1)
        p2 = extract_patches(img, c2)
        e1 = compute_dino_embs(p1)
        e2 = compute_dino_embs(p2)
        ssl_data.append((e1, e2, c1, c2, c2_shuffled, lbl))
        n_loaded += 1
    logger.info(f"SSL precomputed: {len(ssl_data)} frames")

    # Downstream data: split pairs into train/val, precompute features
    np.random.seed(SEED + 1)
    idx = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * 0.2))
    train_pairs = [pairs[i] for i in idx[:-n_val]]
    val_pairs = [pairs[i] for i in idx[-n_val:]]
    if args.label_fraction < 1.0 and len(train_pairs) > 2:
        n_labeled = max(2, int(len(train_pairs) * args.label_fraction))
        train_pairs = train_pairs[:n_labeled]
    logger.info(
        f"Downstream pairs: {len(train_pairs)} train + {len(val_pairs)} val"
    )

    pair_data = {"train": [], "val": []}
    for split_name, split_pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
    ]:
        for mt, mn, it, i_n in split_pairs:
            rt = load_frame(mt, it)
            rn = load_frame(mn, i_n)
            if rt is None or rn is None:
                continue
            ct, lt, imgt = rt
            cn, ln, imgn = rn
            shared = set(lt) & set(ln)
            min_shared = getattr(args, 'min_cells', 8)
            if len(shared) < min_shared:
                continue
            if len(lt) < min_shared or len(ln) < min_shared:
                continue
            idx_t = [i for i, l in enumerate(lt) if l in shared]
            idx_n = [i for i, l in enumerate(ln) if l in shared]
            ct_s, lt_s = ct[idx_t], lt[idx_t]
            cn_s, ln_s = cn[idx_n], ln[idx_n]
            pt_s = extract_patches(imgt, ct_s)
            pn_s = extract_patches(imgn, cn_s)
            pair_data[split_name].append({
                "dino_t": torch.from_numpy(compute_dino_embs(pt_s)).float(),
                "dino_n": torch.from_numpy(compute_dino_embs(pn_s)).float(),
                "coords_t": torch.from_numpy(ct_s).float(),
                "coords_n": torch.from_numpy(cn_s).float(),
                "labels_t": torch.from_numpy(lt_s).long(),
                "labels_n": torch.from_numpy(ln_s).long(),
            })
    logger.info(
        f"Downstream precomputed: {len(pair_data['train'])} train, "
        f"{len(pair_data['val'])} val"
    )

    # Free DINO from GPU — all features are precomputed
    free_dino()

    # ─── Phase 3: SSL Pretraining ───────────────────────────────────────
    encoders = run_ssl_phase(ssl_data, args, pe_dim)

    # ─── Phase 4: Downstream Training ───────────────────────────────────
    downstream_results = run_downstream(encoders, pe_dim, pair_data, args)

    if downstream_results is None:
        logger.error("Downstream phase failed.")
        sys.exit(1)

    # ─── Phase 5: Report ────────────────────────────────────────────────
    make_report(downstream_results, outdir, args)
    logger.info(f"Done. All outputs in {outdir}/")


if __name__ == "__main__":
    main()
