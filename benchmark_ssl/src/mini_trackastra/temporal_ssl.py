#!/usr/bin/env python3
"""Temporal SSL — DynaCLR-inspired: pretrain on consecutive frame pairs.

Key difference from end_to_end.py:
  - SSL uses TWO DIFFERENT frames (t, t+1) instead of two distorted
    copies of the SAME frame.
  - Positive pairs = same cells that persisted across the time gap.
  - Cell has genuinely MOVED → coordinate-based matching is IMPOSSIBLE.
  - No positional encoding during SSL (NoPE only).
  - No distortion pipeline needed.

Modes:
  T: Temporal NoPE SSL → encoder → Full model (NoPE) [DynaCLR-style]
  B: Single-frame NoPE SSL → encoder → Full model (NoPE) [best from end_to_end]
  R: Random init → Full model (NoPE) [baseline]

Usage:
  cd src/mini_trackastra
  .venv/bin/python temporal_ssl.py
  .venv/bin/python temporal_ssl.py --ssl-steps 200 --downstream-steps 100 \
      --max-ssl-frames 30 --d-model 128 --nhead 4
"""

import argparse
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
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("temporal_ssl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DINO_DIM = 384
PATCH_SIZE = 64
_dino_model = None


# ─── DINO Backbone ────────────────────────────────────────────────────────────

def _get_dino():
    global _dino_model
    if _dino_model is None:
        logger.info("Loading DINOv2 backbone...")
        _dino_model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            .to(device)
            .eval()
        )
    return _dino_model


def compute_dino_embs(patches_np):
    """patches_np: (N, 64, 64) float32 [0,1] → (N, 384) float32 on CPU."""
    if len(patches_np) == 0:
        return np.zeros((0, DINO_DIM), dtype=np.float32)
    t = torch.from_numpy(patches_np).float().unsqueeze(1)  # (N, 1, 64, 64)
    t_224 = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t_224 = t_224.expand(-1, 3, -1, -1).to(device)
    mean_norm = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_norm = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    dino = _get_dino()
    with torch.no_grad():
        emb = dino((t_224 - mean_norm) / std_norm)
    return emb.cpu().numpy()


def free_dino():
    global _dino_model
    if _dino_model is not None:
        _dino_model.cpu()
        _dino_model = None
    torch.cuda.empty_cache()


# ─── Data Loading ─────────────────────────────────────────────────────────────

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


def extract_patches(img, coords):
    if len(coords) == 0:
        return np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    half = PATCH_SIZE // 2
    patches = []
    for cy, cx in coords:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, img.shape[0] - 1)
        cx_i = np.clip(cx_i, 0, img.shape[1] - 1)
        y1, x1 = cy_i - half, cx_i - half
        y2, x2 = y1 + PATCH_SIZE, x1 + PATCH_SIZE
        pt, pb = max(0, -y1), max(0, y2 - img.shape[0])
        pl, pr = max(0, -x1), max(0, x2 - img.shape[1])
        crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            pad_h = max(0, PATCH_SIZE - crop.shape[0])
            pad_w = max(0, PATCH_SIZE - crop.shape[1])
            crop = np.pad(crop, ((0, pad_h), (0, pad_w)), mode="reflect")[
                :PATCH_SIZE, :PATCH_SIZE
            ]
        patches.append(crop)
    return np.stack(patches).astype(np.float32)


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


# ─── Distortions (for Mode B single-frame SSL) ───────────────────────────────

def apply_jitter(c, std):
    return c + np.random.randn(*c.shape).astype(np.float32) * std


def apply_affine(c, deg, sr):
    angle = np.deg2rad(np.random.uniform(-deg, deg))
    scale = np.random.uniform(sr[0], sr[1])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = c.mean(axis=0)
    return ((c - center) @ R.T * scale + center).astype(np.float32)


def apply_dropout(c, labels, p):
    if p <= 0:
        return c, labels
    keep = np.random.random(len(c)) > p
    if sum(keep) < 2:
        keep[:2] = True
    return c[keep], labels[keep]


def distort(coords, labels, mode):
    if mode == "full":
        c = apply_affine(coords.copy(), 10, (0.9, 1.1))
        c = apply_jitter(c, 4)
        c, l = apply_dropout(c, labels.copy(), 0.1)
        return c, l
    return apply_jitter(coords.copy(), 4), labels.copy()


# ─── Positional Encodings ────────────────────────────────────────────────────

class NoPE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, c):
        return self.token.expand(c.shape[0], c.shape[1], -1)


class FourierPE(nn.Module):
    def __init__(self, pos_per_dim=16, coord_dim=2):
        super().__init__()
        self.d = coord_dim * pos_per_dim * 2
        self.coord_dim = coord_dim
        self.pos_per_dim = pos_per_dim
        self.register_buffer("freqs", 2.0 ** torch.linspace(0.0, 10.0, pos_per_dim))

    def forward(self, c):
        parts = []
        for i in range(self.coord_dim):
            parts.append(torch.sin(c[:, :, i:i + 1] * self.freqs.view(1, 1, -1)))
        for i in range(self.coord_dim):
            parts.append(torch.cos(c[:, :, i:i + 1] * self.freqs.view(1, 1, -1)))
        return torch.cat(parts, dim=-1)


# ─── MiniTrackingTransformer ─────────────────────────────────────────────────

class MiniEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, pe_dim=64):
        super().__init__()
        self.dino_proj = nn.Linear(DINO_DIM, d_model)
        self.pos_proj = nn.Linear(pe_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, dino_emb, pe):
        x = self.dino_proj(dino_emb) + self.pos_proj(pe)
        x = self.norm(x)
        return self.transformer(x.unsqueeze(0)).squeeze(0)


class MiniDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

    def forward(self, tgt, memory):
        return self.transformer(tgt.unsqueeze(0), memory.unsqueeze(0)).squeeze(0)


class MiniTrackingTransformer(nn.Module):
    def __init__(self, encoder, decoder, d_head=32):
        super().__init__()
        d_model = encoder.dino_proj.out_features
        self.encoder = encoder
        self.decoder = decoder
        self.head_x = nn.Linear(d_model, d_head)
        self.head_y = nn.Linear(d_model, d_head)
        self.scale = d_head ** 0.5

    def forward(self, de_t, pe_t, de_n, pe_n):
        enc_t = self.encoder(de_t, pe_t)
        enc_n = self.encoder(de_n, pe_n)
        dec_out = self.decoder(enc_n, enc_t)
        x = F.normalize(self.head_x(enc_t), dim=-1)
        y = F.normalize(self.head_y(dec_out), dim=-1)
        return (x @ y.T) * self.scale


# ─── Losses ───────────────────────────────────────────────────────────────────

def nt_xent_loss(z, temp=0.05):
    N = z.shape[0] // 2
    zn = F.normalize(z, dim=-1)
    sim = (zn[:N] @ zn[N:].T) / temp
    labels = torch.arange(N, device=z.device)
    return F.cross_entropy(sim, labels)


def bce_assoc_loss(logits, labels_t, labels_n):
    N1, N2 = logits.shape
    target = torch.zeros(N1, N2, device=logits.device)
    for i in range(min(N1, N2)):
        if labels_t[i] == labels_n[i]:
            target[i, i] = 1.0
    return F.binary_cross_entropy_with_logits(logits, target)


def assoc_accuracy(logits, labels_t, labels_n):
    N1, N2 = logits.shape
    correct = 0
    for i in range(min(N1, N2)):
        pred = torch.argmax(logits[i]).item()
        if labels_t[i] == labels_n[pred]:
            correct += 1
    total = max(1, min(N1, N2))
    return correct / total


def balanced_accuracy(logits, labels_t, labels_n):
    N1, N2 = logits.shape
    min_n = min(N1, N2)
    target = torch.zeros(N1, N2, device=logits.device)
    for i in range(min_n):
        if labels_t[i] == labels_n[i]:
            target[i, i] = 1.0
    pred = (logits > 0).float()
    tn = ((pred == 0) & (target == 0)).float().sum()
    tp = ((pred == 1) & (target == 1)).float().sum()
    fp = ((pred == 1) & (target == 0)).float().sum()
    fn = ((pred == 0) & (target == 1)).float().sum()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    return ((tpr + tnr) / 2).item()


# ─── SSL Phase: Temporal ─────────────────────────────────────────────────────

def run_temporal_ssl(temporal_data, single_frame_data, args, pe_dim):
    """Train encoders on temporal and single-frame SSL.

    temporal_data: list of (e_t, e_t1, lbl_t, lbl_t1) — consecutive frames
    single_frame_data: list of (e1, e2, lbl) — distorted single frames (Mode B)
    """
    logger.info(
        f"Temporal SSL data: {len(temporal_data)} pairs, "
        f"Single-frame: {len(single_frame_data)} frames"
    )

    noise_pe = NoPE(pe_dim).to(device)

    configs = [
        ("T", noise_pe, "Temporal NoPE SSL", True),   # use temporal_data
        ("B", noise_pe, "Single-frame NoPE SSL", False),  # use single_frame_data
    ]

    encoders = {}
    for mode, pe, desc, use_temporal in configs:
        enc = MiniEncoder(
            d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_encoder_layers, pe_dim=pe_dim,
        ).to(device)

        opt = torch.optim.Adam(enc.parameters(), lr=args.ssl_lr)
        logger.info(f"  SSL Mode {mode} ({desc}): {args.ssl_steps} steps")

        data_src = temporal_data if use_temporal else single_frame_data

        if use_temporal:
            # Temporal: (e_t, e_t1, lbl_t, lbl_t1) → aligned pairs
            gpu_data = []
            for e_t, e_t1, lbl_t, lbl_t1 in data_src:
                shared = [i for i, l in enumerate(lbl_t) if l in set(lbl_t1)]
                if len(shared) < 2:
                    continue
                idx_t = np.array([np.where(lbl_t == lbl_t[i])[0][0]
                                  for i in shared])
                idx_t1 = np.array([np.where(lbl_t1 == lbl_t[shared[j]])[0][0]
                                   for j in range(len(shared))])
                n = len(shared)
                gpu_data.append((
                    torch.from_numpy(e_t[idx_t]).float().to(device),
                    torch.from_numpy(e_t1[idx_t1]).float().to(device),
                    n,
                ))
        else:
            # Single-frame: (e1, e2, c1, c2, c2_shuffled, lbl) from end_to_end format
            gpu_data = []
            for item in data_src:
                e1, e2 = item[0], item[1]
                gpu_data.append((
                    torch.from_numpy(e1).float().to(device),
                    torch.from_numpy(e2).float().to(device),
                    min(len(e1), len(e2)),
                ))

        if len(gpu_data) == 0:
            logger.warning(f"  Mode {mode}: no valid data, skipping")
            continue

        for step in range(args.ssl_steps):
            enc.train()
            loss_total, n_batches = 0.0, 0
            perm = torch.randperm(len(gpu_data))
            for idx in perm:
                de1, de2, n = gpu_data[idx]
                n = min(n, min(len(de1), len(de2)))
                if n < 2:
                    continue
                # Use a dummy coordinate tensor for NoPE
                dummy_c = torch.zeros(1, n, 2, device=device)
                pe_val = pe(dummy_c).squeeze(0)
                z1 = enc(de1[:n], pe_val)
                z2 = enc(de2[:n], pe_val)
                z = torch.cat([z1, z2])
                loss = nt_xent_loss(z)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()
                loss_total += loss.item()
                n_batches += 1
            if (step + 1) % 50 == 0 or step == 0:
                avg = loss_total / max(1, n_batches)
                logger.info(f"    [{mode}] step {step+1:4d}: loss={avg:.4f}")

        encoders[mode] = {"encoder": enc, "pos_enc": pe}
        final_loss = loss_total / max(1, n_batches)
        logger.info(f"    [{mode}] done: final loss={final_loss:.6f}")

    return encoders


# ─── Downstream Phase ────────────────────────────────────────────────────────

def run_downstream(encoders, pe_dim, pair_data, args):
    if len(pair_data.get("train", [])) < 2:
        logger.error("Not enough training pairs.")
        return None

    test_modes = [
        ("T", "Temporal NoPE SSL", encoders.get("T")),
        ("B", "Single-frame NoPE SSL", encoders.get("B")),
        ("R", "Random init baseline", None),
    ]

    results = {}
    for mode, mode_name, ssl_info in test_modes:
        if mode in ("T", "B") and ssl_info is None:
            logger.warning(f"  Mode {mode}: SSL encoder missing, skipping")
            continue

        if mode == "R":
            enc = MiniEncoder(
                d_model=args.d_model, nhead=args.nhead,
                num_layers=args.num_encoder_layers, pe_dim=pe_dim,
            ).to(device)
            pe_mod = NoPE(pe_dim).to(device)
        else:
            enc = MiniEncoder(
                d_model=args.d_model, nhead=args.nhead,
                num_layers=args.num_encoder_layers, pe_dim=pe_dim,
            ).to(device)
            enc.load_state_dict(ssl_info["encoder"].state_dict())
            pe_mod = ssl_info["pos_enc"]

        dec = MiniDecoder(
            d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_decoder_layers,
        ).to(device)

        model = MiniTrackingTransformer(encoder=enc, decoder=dec, d_head=args.d_head).to(device)
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
                        val_losses.append(bce_assoc_loss(scores, lt, ln).item())
                        val_accs.append(assoc_accuracy(scores, lt, ln))
                        val_bal_accs.append(balanced_accuracy(scores, lt, ln))

                train_loss = np.mean(train_losses)
                train_acc = np.mean(train_accs)
                train_bal = np.mean(train_bal_accs)
                val_loss = np.mean(val_losses)
                val_acc = np.mean(val_accs)
                val_bal = np.mean(val_bal_accs)
                rows.append({
                    "step": step, "train_loss": train_loss,
                    "train_acc": train_acc, "train_bal_acc": train_bal,
                    "val_loss": val_loss, "val_acc": val_acc,
                    "val_bal_acc": val_bal,
                })
                if step % 50 == 0 or step == args.downstream_steps - 1:
                    logger.info(
                        f"    [{mode}] step {step:3d}: "
                        f"val_loss={val_loss:.4f}  bal_acc={val_bal:.4f}"
                    )

        results[mode] = pd.DataFrame(rows)
        final = rows[-1]
        logger.info(
            f"    [{mode}] done: val_loss={final['val_loss']:.4f}  "
            f"val_acc={final['val_acc']:.4f}  bal_acc={final['val_bal_acc']:.4f}"
        )

    return results


# ─── Report ──────────────────────────────────────────────────────────────────

def make_report(downstream_results, outdir, args):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"T": "#e67e22", "B": "#2ecc71", "R": "#95a5a6"}
    labels_map = {
        "T": "Mode T: Temporal SSL (DynaCLR-style)",
        "B": "Mode B: NoPE single-frame SSL",
        "R": "Mode R: Random init (baseline)",
    }

    plot_order = sorted(
        downstream_results.keys(),
        key=lambda m: {"R": "z", "B": "b", "T": "a"}.get(m, m),
    )

    # Panel 1: Val loss
    ax = axes[0]
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        ax.plot(df["step"], df["val_loss"], color=colors.get(mode, "#333"),
                label=labels_map.get(mode, mode), linewidth=2)
    ax.set_xlabel("Downstream training step")
    ax.set_ylabel("Val BCE Loss")
    ax.set_title("Downstream Convergence: Val Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Balanced accuracy
    ax = axes[1]
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        ax.plot(df["step"], df.get("val_bal_acc", df["val_acc"]),
                color=colors.get(mode, "#333"), linewidth=2)
    ax.set_xlabel("Downstream training step")
    ax.set_ylabel("Val Balanced Accuracy")
    ax.set_title("Downstream Convergence: Balanced Acc (baseline=50%)")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Convergence speed (steps to bal_acc > 0.6)
    ax = axes[2]
    mode_names, conv_steps, bar_colors = [], [], []
    bar_order = sorted(plot_order, key=lambda m: {"R": 99, "T": 1, "B": 2}.get(m, 50))
    for mode in bar_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        acc_col = "val_bal_acc" if "val_bal_acc" in df.columns else "val_acc"
        hit = df[df[acc_col] > 0.6]
        s = hit["step"].iloc[0] if len(hit) else args.downstream_steps
        mode_names.append(labels_map.get(mode, mode).split("(")[0].strip())
        conv_steps.append(s)
        bar_colors.append(colors.get(mode, "#333"))
    bars = ax.bar(mode_names, conv_steps, color=bar_colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Steps to bal_acc > 0.6")
    ax.set_title("Convergence Speed (lower = faster)")
    for bar, v in zip(bars, conv_steps):
        label = str(v) if v < args.downstream_steps else "never"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha="center", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"Temporal SSL (DynaCLR-inspired): SSL -> Downstream Transfer\n"
        f"{args.max_ssl_frames} SSL pairs, {args.ssl_steps} SSL steps, "
        f"{args.downstream_steps} downstream steps",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = outdir / "convergence_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {fig_path}")

    # Save CSVs
    summary_rows = []
    for mode in ["T", "B", "R"]:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        csv_path = outdir / f"downstream_{mode}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved: {csv_path}")

        final = df.iloc[-1]
        initial = df.iloc[0]
        acc_col = "val_bal_acc" if "val_bal_acc" in df.columns else "val_acc"
        summary_rows.append({
            "mode": mode,
            "val_loss_final": final["val_loss"],
            "val_acc_final": final["val_acc"],
            "val_bal_acc_start": initial[acc_col],
            "val_bal_acc_final": final[acc_col],
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_path = outdir / "summary.csv"
    sum_df.to_csv(sum_path, index=False)
    logger.info(f"Summary saved: {sum_path}")

    # Verdict
    verdict_lines = [
        "=" * 65,
        "TEMPORAL SSL: SSL -> DOWNSTREAM TRANSFER -- VERDICT",
        "=" * 65,
    ]
    for _, row in sum_df.iterrows():
        name = labels_map.get(row["mode"], row["mode"])
        verdict_lines.append(
            f"\n  {name}:"
            f"\n    Final val_loss: {row['val_loss_final']:.4f}"
            f"\n    Val bal_acc: {row['val_bal_acc_start']:.3f} -> {row['val_bal_acc_final']:.3f}"
        )

    # Compare
    if "T" in downstream_results and "R" in downstream_results:
        t_final = downstream_results["T"]["val_bal_acc"].iloc[-1]
        r_final = downstream_results["R"]["val_bal_acc"].iloc[-1]
        delta = t_final - r_final
        verdict_lines.append(f"\n  Temporal SSL vs Random: Δ bal_acc = {delta:+.4f}")
        if delta > 0.05:
            verdict_lines.append("  ✓ Temporal SSL provides meaningful improvement")
        elif delta > 0.01:
            verdict_lines.append("  ~ Marginal improvement")
        else:
            verdict_lines.append("  ✗ No benefit from temporal SSL")

    if "T" in downstream_results and "B" in downstream_results:
        t_final = downstream_results["T"]["val_bal_acc"].iloc[-1]
        b_final = downstream_results["B"]["val_bal_acc"].iloc[-1]
        delta = t_final - b_final
        verdict_lines.append(f"\n  Temporal (T) vs Single-frame (B): Δ bal_acc = {delta:+.4f}")
        if delta > 0.03:
            verdict_lines.append("  ✓ Temporal beats single-frame — movement helps")
        elif delta > -0.03:
            verdict_lines.append("  ~ Roughly equivalent")
        else:
            verdict_lines.append("  ✗ Single-frame beats temporal")

    verdict_lines.append(f"\nFull results: {outdir}/")
    verdict_text = "\n".join(verdict_lines)
    vp = outdir / "verdict.txt"
    vp.write_text(verdict_text)
    logger.info(f"Verdict saved: {vp}")
    print("\n" + verdict_text)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Temporal SSL (DynaCLR-inspired)")
    p.add_argument("--data-root", default="../../data/vanvliet")
    p.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL")
    p.add_argument("--distortion", default="full", choices=["jitter4", "full"])

    # SSL
    p.add_argument("--ssl-steps", type=int, default=200)
    p.add_argument("--ssl-lr", type=float, default=0.001)
    p.add_argument("--max-ssl-frames", type=int, default=30)

    # Downstream
    p.add_argument("--downstream-steps", type=int, default=100)
    p.add_argument("--downstream-lr", type=float, default=0.001)
    p.add_argument("--max-downstream-pairs", type=int, default=30)

    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--num-decoder-layers", type=int, default=2)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--pos-per-dim", type=int, default=16)
    p.add_argument("--pe-dim-coord", type=int, default=2)

    p.add_argument("--outdir", default="runs/temporal_ssl")

    return p.parse_args(argv)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]
    pe_dim = args.pos_per_dim * args.pe_dim_coord * 2

    logger.info(
        f"Temporal SSL benchmark\n"
        f"  Architecture: d_model={args.d_model}, nhead={args.nhead}, "
        f"enc_layers={args.num_encoder_layers}, dec_layers={args.num_decoder_layers}\n"
        f"  PE dim: {pe_dim} (pos_per_dim={args.pos_per_dim}, coord_dim={args.pe_dim_coord})\n"
        f"  SSL: {args.ssl_steps} steps, lr={args.ssl_lr}, frames={args.max_ssl_frames}\n"
        f"  Downstream: {args.downstream_steps} steps, lr={args.downstream_lr}, "
        f"pairs={args.max_downstream_pairs}\n"
        f"  Distortion (Mode B): {args.distortion}\n"
        f"  Device: {device}"
    )

    # ─── Phase 1: Collect data ────────────────────────────────────────────
    logger.info("Scanning consecutive frame pairs for temporal SSL...")
    t_pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_ssl_frames)
    logger.info(f"Found {len(t_pairs)} temporal pairs for SSL")

    if len(t_pairs) < 2:
        logger.error("Need at least 2 temporal pairs.")
        sys.exit(1)

    logger.info("Scanning single frames for Mode B SSL...")
    all_frames = []
    dr = Path(args.data_root)
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
                    all_frames.append((str(m), str(ip)))
                if len(all_frames) >= args.max_ssl_frames:
                    break
            if len(all_frames) >= args.max_ssl_frames:
                break
        if len(all_frames) >= args.max_ssl_frames:
            break
    logger.info(f"Found {len(all_frames)} single frames for Mode B SSL")

    # Scan downstream pairs
    logger.info("Scanning consecutive frame pairs for downstream...")
    downstream_pairs = scan_consecutive_pairs(
        args.data_root, conditions, args.max_downstream_pairs
    )
    logger.info(f"Found {len(downstream_pairs)} pairs for downstream")

    if len(downstream_pairs) < 3:
        logger.error("Need at least 3 pairs for downstream.")
        sys.exit(1)

    # ─── Phase 2: Precompute DINO features ─────────────────────────────────

    # Temporal SSL data: (e_t, e_t1, lbl_t, lbl_t1)
    temporal_data = []
    for mp_t, mp_t1, ip_t, ip_t1 in t_pairs:
        rt = load_frame(mp_t, ip_t)
        rn = load_frame(mp_t1, ip_t1)
        if rt is None or rn is None:
            continue
        ct, lt, imgt = rt
        cn, ln, imgn = rn
        shared = set(lt) & set(ln)
        if len(shared) < 4:
            continue
        pt = extract_patches(imgt, ct)
        pn = extract_patches(imgn, cn)
        et = compute_dino_embs(pt)
        et1 = compute_dino_embs(pn)
        temporal_data.append((et, et1, lt, ln))

    logger.info(f"Temporal SSL precomputed: {len(temporal_data)} valid pairs")

    # Single-frame SSL data: (e1, e2, lbl) — distort then DINO
    single_frame_data = []
    for mp, ip in all_frames:
        r = load_frame(mp, ip)
        if r is None:
            continue
        c1, lbl, img = r
        c2, _ = distort(c1.copy(), lbl.copy(), args.distortion)
        p1 = extract_patches(img, c1)
        p2 = extract_patches(img, c2)
        e1 = compute_dino_embs(p1)
        e2 = compute_dino_embs(p2)
        single_frame_data.append((e1, e2, lbl))

    logger.info(f"Single-frame SSL precomputed: {len(single_frame_data)} frames")

    # Downstream data
    np.random.seed(SEED + 1)
    idx = np.random.permutation(len(downstream_pairs))
    n_val = max(1, int(len(downstream_pairs) * 0.2))
    train_pairs = [downstream_pairs[i] for i in idx[:-n_val]]
    val_pairs = [downstream_pairs[i] for i in idx[-n_val:]]
    logger.info(f"Downstream pairs: {len(train_pairs)} train + {len(val_pairs)} val")

    pair_data = {"train": [], "val": []}
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for mt, mn, it, i_n in split_pairs:
            rt = load_frame(mt, it)
            rn = load_frame(mn, i_n)
            if rt is None or rn is None:
                continue
            ct, lt, imgt = rt
            cn, ln, imgn = rn
            shared = set(lt) & set(ln)
            if len(shared) < 8:
                continue
            if len(lt) < 8 or len(ln) < 8:
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

    free_dino()

    # ─── Phase 3: SSL ─────────────────────────────────────────────────────
    encoders = run_temporal_ssl(temporal_data, single_frame_data, args, pe_dim)

    # ─── Phase 4: Downstream ──────────────────────────────────────────────
    downstream_results = run_downstream(encoders, pe_dim, pair_data, args)

    if downstream_results is None:
        logger.error("Downstream phase failed.")
        sys.exit(1)

    # ─── Phase 5: Report ──────────────────────────────────────────────────
    make_report(downstream_results, outdir, args)
    logger.info(f"Done. All outputs in {outdir}/")


if __name__ == "__main__":
    main()
