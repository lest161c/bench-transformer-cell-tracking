#!/usr/bin/env python3
"""Convergence race: Mode R (7D baseline) vs Mode C (large CNN residual)
at the maxed-out A500 scale (d_model=256, nhead=8, 3L/3L).

Runs for 200 downstream steps with pos_weight=100.0.
Aborts if bal_acc stuck at 0.5 for 30+ consecutive eval steps.
CNN features are CONCATENATED (with learned fusion projection), not added.

Usage:
    python scale_sweep.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import regionprops as sk_regionprops
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scale_sweep")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PATCH_SIZE = 64
CNN_FEAT_DIM = 128

# ─── ScaledCNN (same arch as cnn_ssl.py / cnn_convergence.py) ────────────

class ScaledCNN(nn.Module):
    """ConvNet for 64x64 grayscale cell patches. Output: 128-dim embedding."""
    def __init__(self, scale="large", out_dim=CNN_FEAT_DIM):
        super().__init__()
        if scale == "small":
            ch = [8, 16, 32]           # 3 conv layers
        elif scale == "medium":
            ch = [16, 32, 64]          # 3 conv layers
        else:  # large
            ch = [32, 64, 128, 256]    # 4 conv layers
        layers = []
        in_ch = 1
        for c in ch:
            layers += [nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = c
        self.conv = nn.Sequential(*layers)
        # After pooling: small/medium: 64->32->16->8,  large: 64->32->16->8->4
        spatial = PATCH_SIZE // (2 ** len(ch))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, patches):
        return self.fc(self.conv(patches))


# ─── Data loading (from cnn_convergence.py) ───────────────────────────────

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
        crop = img[y1c:y2c, x1c:x2c] if (y2c > y1c and x2c > x1c) else np.zeros((1, 1), dtype=np.float32)
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(crop, tuple((0, max(0, t)) for t in [patch_size - s for s in crop.shape]), mode="reflect")[:patch_size, :patch_size]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, patch_size, patch_size), dtype=np.float32)


# ─── Positional Encoding ──────────────────────────────────────────────────

class FourierPE(nn.Module):
    """Fourier feature encoding for coordinates."""
    def __init__(self, pos_per_dim=16, coord_dim=2):
        super().__init__()
        self.d = coord_dim * pos_per_dim * 2
        self.coord_dim = coord_dim
        self.pos_per_dim = pos_per_dim
        self.register_buffer("freqs", 2.0 ** torch.linspace(0.0, 10.0, pos_per_dim))

    def forward(self, c):
        parts = []
        for i in range(self.coord_dim):
            parts.append(torch.sin(c[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        for i in range(self.coord_dim):
            parts.append(torch.cos(c[:, :, i:i+1] * self.freqs.view(1, 1, -1)))
        return torch.cat(parts, dim=-1)


# ─── MiniEncoder / MiniDecoder / MiniTrackingTransformer ──────────────────

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


# ─── Loss & metrics ───────────────────────────────────────────────────────

def _build_target_matrix(labels_t, labels_n, dev):
    N1, N2 = len(labels_t), len(labels_n)
    target = torch.zeros(N1, N2, device=dev)
    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_n):
            if lt == ln:
                target[i, j] = 1.0
    return target


def bce_assoc_loss(logits, labels_t, labels_n, pos_weight=100.0):
    """Weighted BCE loss. positive weight compensates for class imbalance."""
    target = _build_target_matrix(labels_t, labels_n, logits.device)
    # pos_weight gives NEG× more weight to positive pairs (same cell)
    weight = torch.where(target == 1, pos_weight, 1.0)
    return F.binary_cross_entropy_with_logits(logits, target, weight=weight)


def check_collapse(val_bal_acc, stuck_counter, step, stuck_limit=30):
    """Return (collapsed, updated_counter). Collapsed if bal_acc stays at 0.5
    for `stuck_limit` consecutive evaluations."""
    if abs(val_bal_acc - 0.5) < 0.001 or val_bal_acc < 0.5:
        stuck_counter += 1
    else:
        stuck_counter = 0
    if stuck_counter >= stuck_limit:
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


# ─── Per-scale race runner ────────────────────────────────────────────────

def run_scale(label, d_model, nhead, enc_layers, dec_layers,
              frames, pair_data, pe_dim, cnn_model, downstream_steps=200):
    """Run Mode R and Mode C at a given scale, return results dict."""
    logger.info("")
    logger.info("═" * 60)
    logger.info(f"  SCALE: {label}  (d_model={d_model}, nhead={nhead}, "
                f"enc={enc_layers}, dec={dec_layers}, frames={frames})")
    logger.info("═" * 60)

    # Shared FourierPE
    pe_mod = FourierPE(pos_per_dim=16).to(device)

    configs = []
    # Mode R (7D only)
    enc_r = MiniEncoder(
        d_model=d_model, nhead=nhead,
        num_layers=enc_layers, pe_dim=pe_dim,
        feat_dim=7,
    ).to(device)
    configs.append(("R", "7D only", enc_r, "feat_7d", None))

    # Mode C (7D + CNN residual)
    enc_c = MiniEncoder(
        d_model=d_model, nhead=nhead,
        num_layers=enc_layers, pe_dim=pe_dim,
        feat_dim=7, cnn_feat_dim=CNN_FEAT_DIM,
    ).to(device)
    configs.append(("C", "7D + CNN", enc_c, "feat_7d", "cnn"))

    results = {}
    for mode, mode_name, enc, feat_key, cnn_key in configs:
        dec = MiniDecoder(d_model=d_model, nhead=nhead,
                          num_layers=dec_layers).to(device)
        model = MiniTrackingTransformer(encoder=enc, decoder=dec, d_head=32).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        logger.info(f"\n    [{mode}] {mode_name} — training {downstream_steps} steps")

        rows = []
        stuck_counter = 0
        for step in range(downstream_steps):
            model.train()
            train_losses, train_bal_accs = [], []

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

                scores = model(feat_t, pe_t, feat_n, pe_n,
                               cnn_t=cnn_t, cnn_n=cnn_n)
                loss = bce_assoc_loss(scores, lt, ln)
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.item())
                sd = scores.detach()
                train_bal_accs.append(balanced_accuracy(sd, lt, ln))

            # Evaluate every 5 steps to track convergence
            if step % 5 == 0 or step == downstream_steps - 1:
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

                        scores = model(feat_t, pe_t, feat_n, pe_n,
                                       cnn_t=cnn_t, cnn_n=cnn_n)
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

                logger.info(f"      [{mode}] step {step:3d}/{downstream_steps}: "
                            f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
                            f"val_bal={v_bal:.4f}")

                collapsed, stuck_counter = check_collapse(v_bal, stuck_counter, step)
                if collapsed:
                    logger.warning(f"  Mode {mode}: BAL_ACC STUCK AT 0.5 — model collapsed to all-negatives. Aborting.")
                    break

        results[mode] = rows
        logger.info(f"    [{mode}] done.")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    data_root = "../../data/vanvliet"
    conditions = ["rpsM", "recA", "pheA", "metA", "cib", "trpL"]
    max_pairs = 60  # maxed-out config
    min_cells = 6
    checkpoint_path = "probe/cnn_ntxent_large.pt"

    logger.info("=" * 60)
    logger.info("SCALE SWEEP: CNN Residual Injection Convergence Analysis")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Min cells: {min_cells}")
    logger.info(f"CNN checkpoint: {checkpoint_path}")

    # ─── Load frozen CNN (large scale checkpoint) ─────────────────────────
    logger.info("\nLoading frozen NT-Xent CNN checkpoint...")
    cnn_model = ScaledCNN(scale="large", out_dim=CNN_FEAT_DIM).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cnn_model.load_state_dict(ckpt["model_state_dict"])
    cnn_model.eval()
    for p in cnn_model.parameters():
        p.requires_grad = False
    logger.info(f"Loaded CNN ({sum(p.numel() for p in cnn_model.parameters()):,} params)")

    # ─── Scan pairs ───────────────────────────────────────────────────────
    logger.info("\nScanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(data_root, conditions, max_pairs)
    logger.info(f"Found {len(pairs)} consecutive frame pairs")
    if len(pairs) < 3:
        logger.error("Need at least 3 pairs.")
        sys.exit(1)

    np.random.seed(SEED + 1)
    idx = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * 0.2))
    train_pairs = [pairs[i] for i in idx[:-n_val]]
    val_pairs = [pairs[i] for i in idx[-n_val:]]
    logger.info(f"Split: {len(train_pairs)} train + {len(val_pairs)} val")

    # ─── Precompute features (shared across all scales) ───────────────────
    logger.info("\nPrecomputing features (7D regionprops + CNN embeddings)...")
    pair_data = {"train": [], "val": []}
    skipped_no_share = 0
    skipped_min_cells = 0
    skipped_feat = 0
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for mt, mn, it, i_n in split_pairs:
            rt = load_frame(mt, it)
            rn = load_frame(mn, i_n)
            if rt is None or rn is None:
                continue
            ct, lt, imgt = rt
            cn, ln, imgn = rn

            # Load masks for regionprops
            mask_t = imread(mt)
            mask_n = imread(mn)

            shared = set(lt) & set(ln)
            if len(shared) < min_cells:
                skipped_no_share += 1
                continue
            if len(lt) < min_cells or len(ln) < min_cells:
                skipped_min_cells += 1
                continue

            idx_t = [i for i, l in enumerate(lt) if l in shared]
            idx_n = [i for i, l in enumerate(ln) if l in shared]
            ct_s, lt_s = ct[idx_t], lt[idx_t]
            cn_s, ln_s = cn[idx_n], ln[idx_n]

            # Extract patches (for CNN)
            pt_s = extract_patches(imgt, ct_s)
            pn_s = extract_patches(imgn, cn_s)

            # 7D regionprops
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
                skipped_feat += 1
                continue

            feat_7d_t = torch.tensor(feats_7d_t_list, dtype=torch.float32)
            feat_7d_n = torch.tensor(feats_7d_n_list, dtype=torch.float32)

            # CNN features via frozen model
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

    logger.info(f"Data prepared: {len(pair_data['train'])} train, "
                f"{len(pair_data['val'])} val items "
                f"(skipped: {skipped_no_share} no-share, "
                f"{skipped_min_cells} min-cells, {skipped_feat} feats)")
    if len(pair_data["train"]) < 1 or len(pair_data["val"]) < 1:
        logger.error("Not enough data.")
        sys.exit(1)

    #     ─── Define config: only maxed-out Large with 3L/3L ───────────────────
    label = "Large"
    d_model = 256
    nhead = 8
    enc_layers = 3
    dec_layers = 3

    pe_dim = 16 * 2 * 2  # pos_per_dim=16, coord_dim=2

    # ─── Run ──────────────────────────────────────────────────────────────
    t0 = time.time()
    res = run_scale(label, d_model, nhead, enc_layers, dec_layers, None,
                    pair_data, pe_dim, cnn_model, downstream_steps=200)
    elapsed = time.time() - t0
    logger.info(f"  [{label}] completed in {elapsed:.1f}s")

    # ─── Extract results ──────────────────────────────────────────────────
    r_rows = res.get("R", [])
    c_rows = res.get("C", [])
    r_final = r_rows[-1] if r_rows else None
    c_final = c_rows[-1] if c_rows else None

    def steps_to_loss_under_05(rows):
        for r in rows:
            if r["val_loss"] < 0.5:
                return r["step"]
        return None

    def was_collapsed(rows):
        if not rows:
            return True
        return rows[-1]["step"] < 199  # didn't reach final step (199 = 200-1)

    r_steps_under = steps_to_loss_under_05(r_rows)
    c_steps_under = steps_to_loss_under_05(c_rows)
    r_collapsed = was_collapsed(r_rows)
    c_collapsed = was_collapsed(c_rows)

    r_loss = r_final["val_loss"] if r_final else float("nan")
    r_bal = r_final["val_bal_acc"] if r_final else float("nan")
    c_loss = c_final["val_loss"] if c_final else float("nan")
    c_bal = c_final["val_bal_acc"] if c_final else float("nan")

    # ─── Print comparison table ──────────────────────────────────────────
    print()
    print("=" * 75)
    print("CONVERGENCE RACE — Large CNN vs 7D baseline (A500 scale)")
    print("=" * 75)
    header = f"{'Mode':<8} | {'Val Loss':<9} | {'BalAcc':<7} | {'Collapsed?':<10} | {'Steps to loss<0.5':<17}"
    sep = "-" * 75
    print(header)
    print(sep)
    print(f"{'R (7D)':<8} | {r_loss:<9.4f} | {r_bal:<7.4f} | {'Yes' if r_collapsed else 'No':<10} | {str(r_steps_under) if r_steps_under is not None else 'N/A':<17}")
    print(f"{'C (CNN)':<8} | {c_loss:<9.4f} | {c_bal:<7.4f} | {'Yes' if c_collapsed else 'No':<10} | {str(c_steps_under) if c_steps_under is not None else 'N/A':<17}")
    print(sep)

    # ─── Verdict ─────────────────────────────────────────────────────────
    improved = c_loss < r_loss and c_bal > r_bal
    verdict = "improves" if improved else "does not improve"
    print(f"\nVerdict: Large CNN {verdict} over 7D baseline.")

    # Compare vs old medium CNN result (hardcoded reference)
    print(f"vs old medium CNN: [check against previous run]")

    print()
    print("=" * 75)
    logger.info("Scale sweep complete.")


if __name__ == "__main__":
    main()
