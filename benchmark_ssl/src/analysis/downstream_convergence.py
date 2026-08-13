"""Downstream convergence test: does SSL-pretrained init converge faster?

Simulates Trackastra fine-tuning at micro scale:
  - Real adjacent-frame pairs from vanvliet (with association labels)
  - Frozen DINO features + learned projection → cross-attention → BCE
  - Compare: SSL-pretrained proj vs random init
  - Metric: tracking accuracy @ epoch N → convergence speedup

Usage:
    uv run python -m src.analysis.downstream_convergence
"""

import argparse
import base64
import io
import logging
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import regionprops_table
from tifffile import imread

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("downstream")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# DINO model
logger.info("Loading DINOv2...")
# Pin the "main" branch so torch.hub does not probe the network for the
# default branch (fails offline / on dropped connections).
dino = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vits14").to(device).eval()
DINO_DIM = 384
PATCH_SIZE = 64

# ─── data ──────────────────────────────────────────────────────────────────────

def extract_patches(img, centroids):
    """Extract square patches (PATCH_SIZE x PATCH_SIZE) centered on each centroid.

    Out-of-bounds regions are padded by reflection; returns a stacked float32
    array of shape (N, PATCH_SIZE, PATCH_SIZE).
    """
    h, w = img.shape[-2:]
    half = PATCH_SIZE // 2
    patches = []
    for cy, cx in centroids:
        cy_i = max(0, min(h-1, int(round(float(cy)))))
        cx_i = max(0, min(w-1, int(round(float(cx)))))
        y1, y2 = cy_i - half, cy_i + half
        x1, x2 = cx_i - half, cx_i + half
        pt = max(0, -y1); pb = max(0, y2 - h)
        pl = max(0, -x1); pr = max(0, x2 - w)
        y1c, y2c = max(0, y1), min(h, y2)
        x1c, x2c = max(0, x1), min(w, x2)
        if y2c <= y1c or x2c <= x1c:
            patches.append(np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32))
            continue
        crop = img[y1c:y2c, x1c:x2c]
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(crop,
                ((0, max(0, PATCH_SIZE - crop.shape[0])), (0, max(0, PATCH_SIZE - crop.shape[1]))),
                mode="reflect")[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    return np.stack(patches) if patches else np.zeros((0, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

@torch.no_grad()
def compute_dino(patches_np):
    """Compute 384D DINOv2 embeddings for a batch of normalized patches."""
    if len(patches_np) == 0: return np.zeros((0, DINO_DIM), dtype=np.float32)
    pn = (patches_np - patches_np.min(axis=(1,2), keepdims=True)) / (patches_np.max(axis=(1,2), keepdims=True) - patches_np.min(axis=(1,2), keepdims=True) + 1e-8)
    patch_tensor = F.interpolate(torch.from_numpy(pn).float().unsqueeze(1).to(device), size=(224,224), mode="bilinear", align_corners=False)
    patch_tensor = patch_tensor.expand(-1, 3, -1, -1)
    mean, std = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1), torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    return dino((patch_tensor - mean) / std).cpu().numpy()

def load_frame(mask_path, img_path):
    """Load a mask+image frame and return coords, labels, and DINO embeddings for its cells."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) == 0: return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    labels = props["label"].astype(np.int32)
    patches = extract_patches(img, coords)
    dino_emb = compute_dino(patches)
    return coords, labels, dino_emb

def scan_pairs(data_root, conditions, max_pairs=200):
    """Scan for adjacent frame pairs with tracking ground truth."""
    pairs = []
    dr = Path(data_root)
    for cond in conditions:
        for exp in sorted((dr / cond).iterdir()):
            if not exp.is_dir(): continue
            tra_dir, img_dir = exp / "TRA", exp / "img"
            if not tra_dir.exists() or not img_dir.exists(): continue
            masks = sorted(tra_dir.glob("man_track*.tif"))
            for i in range(len(masks) - 1):
                stem_a = masks[i].stem.replace("man_track", "")
                stem_b = masks[i+1].stem.replace("man_track", "")
                try:
                    fa, fb = int(stem_a), int(stem_b)
                except ValueError: continue
                if fb - fa != 1: continue  # only adjacent
                if masks[i+1].exists() and (img_dir / f"t{fb:06d}.tif").exists():
                    pairs.append((str(masks[i]), str(masks[i+1]),
                                  str(img_dir / f"t{fa:06d}.tif"), str(img_dir / f"t{fb:06d}.tif")))
                    if len(pairs) >= max_pairs: return pairs
    return pairs

def build_association(label_a, label_b):
    """Build association matrix: assoc[i,j] = 1 if label_a[i] == label_b[j]."""
    n_src, n_tgt = len(label_a), len(label_b)
    assoc = np.zeros((n_src, n_tgt), dtype=np.float32)
    lbl_to_b = {int(l): j for j, l in enumerate(label_b)}
    for i, lbl in enumerate(label_a):
        if int(lbl) in lbl_to_b:
            assoc[i, lbl_to_b[int(lbl)]] = 1.0
    return assoc


# ─── model ─────────────────────────────────────────────────────────────────────

class SSLPretrainedProj(nn.Module):
    """MLP projection head from DINO dim → d_model. No BatchNorm (avoids state dict issues)."""
    def __init__(self, in_dim=DINO_DIM, hidden=128, out_dim=64):
        """Build a 2-layer MLP mapping in_dim to out_dim."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, features):
        """Normalize the MLP-projected features to the unit sphere."""
        return F.normalize(self.net(features), dim=-1)


class SimpleTracker(nn.Module):
    """Minimal cross-attention tracker for DINO features.

    Simulates Trackastra's downstream task: (src_emb, tgt_emb) → association logits.
    """
    def __init__(self, in_dim=64, d_model=64, nhead=4):
        """Build projection, cross-attention, LayerNorm, and linear head modules."""
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        """Map (B, N_src, in_dim) x (B, N_tgt, in_dim) to association scores (B, N_src, N_tgt)."""
        # src, tgt: (B, N, in_dim)
        src_proj = self.proj(src)
        tgt_proj = self.proj(tgt)
        out, _ = self.cross_attn(src_proj, tgt_proj, tgt_proj)
        out = self.norm(src_proj + out)
        logits = self.head(out).squeeze(-1)  # (B, N)
        # For each src cell, compute match scores: (B, N_src, N_tgt) via outer product
        score_matrix = torch.einsum("bnd,bmd->bnm", src_proj, tgt_proj)
        return score_matrix


# ─── downstream training ──────────────────────────────────────────────────────

def prepare_data(pairs, max_pairs=80, train_frac=0.6):
    """Load pairs, extract DINO features, return train/val sets."""
    np.random.seed(42)
    pairs = list(pairs)
    np.random.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    def load(pair_list):
        src_embs, tgt_embs, assoc_mats = [], [], []
        for mp_a, mp_b, ip_a, ip_b in pair_list[:max_pairs]:
            ra = load_frame(mp_a, ip_a)
            rb = load_frame(mp_b, ip_b)
            if ra is None or rb is None: continue
            c_a, lb_a, eb_a = ra
            c_b, lb_b, eb_b = rb
            if len(lb_a) < 2 or len(lb_b) < 2: continue
            assoc = build_association(lb_a, lb_b)
            if assoc.sum() < 1: continue  # skip pairs with no positive
            src_embs.append(eb_a)
            tgt_embs.append(eb_b)
            assoc_mats.append(assoc)
        return src_embs, tgt_embs, assoc_mats

    train = load(train_pairs)
    val = load(val_pairs)
    return train, val


def collate_downstream(batch_src, batch_tgt, batch_assoc):
    """Pad a batch of variable-size pairs."""
    B = len(batch_src)
    max_ns = max(e.shape[0] for e in batch_src)
    max_nt = max(e.shape[0] for e in batch_tgt)
    D = batch_src[0].shape[-1]
    src = torch.zeros(B, max_ns, D)
    tgt = torch.zeros(B, max_nt, D)
    assoc = torch.zeros(B, max_ns, max_nt)
    pm_s = torch.ones(B, max_ns, dtype=torch.bool)
    pm_t = torch.ones(B, max_nt, dtype=torch.bool)
    for i in range(B):
        ns, nt = len(batch_src[i]), len(batch_tgt[i])
        src[i, :ns] = torch.from_numpy(batch_src[i]).float()
        tgt[i, :nt] = torch.from_numpy(batch_tgt[i]).float()
        assoc[i, :ns, :nt] = torch.from_numpy(batch_assoc[i]).float()
        pm_s[i, :ns] = False
        pm_t[i, :nt] = False
    return src, tgt, assoc, pm_s, pm_t


def downstream_loss(logits, assoc, pm_s, pm_t, pos_weight=5.0):
    """BCE loss with pos_weight. Mask padded positions."""
    mask = ~pm_s.unsqueeze(-1) & ~pm_t.unsqueeze(-2)
    logits = logits[mask]
    targets = assoc[mask]
    if logits.numel() == 0: return torch.tensor(0.0, device=logits.device, requires_grad=True)
    n_pos = targets.sum()
    n_neg = targets.numel() - n_pos
    pw = (n_neg / max(n_pos, 1)) * pos_weight if n_pos > 0 else pos_weight
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(pw, device=logits.device))
    with torch.no_grad():
        acc = ((logits.sigmoid() > 0.5) == targets.bool()).float().mean().item()
    return loss, acc


def run_downstream(train_data, val_data, ssl_proj_state=None, n_epochs=30, lr=1e-3, label="SSL-init"):
    """Train SimpleTracker with DINO features ± SSL-pretrained projection."""
    emb_dim = 384 if ssl_proj_state is None else 64

    # Build model
    proj = SSLPretrainedProj(in_dim=384, hidden=128, out_dim=64)
    if ssl_proj_state is not None:
        proj.load_state_dict(ssl_proj_state)
        logger.info(f"  Loaded SSL-pretrained projection weights")
    else:
        logger.info(f"  Random projection init")
    tracker = SimpleTracker(in_dim=64, d_model=64, nhead=4)
    proj = proj.to(device)
    tracker = tracker.to(device)

    params = list(tracker.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    train_src, train_tgt, train_assoc = train_data
    val_src, val_tgt, val_assoc = val_data
    batch_size = 8
    rows = []

    for epoch in range(n_epochs):
        proj.train(); tracker.train()
        # Mini-batch training
        idx = np.random.permutation(len(train_src))
        tr_loss, tr_acc, n_b = 0, 0, 0
        for start in range(0, len(train_src), batch_size):
            batch_idx = idx[start:start+batch_size]
            src_b = [train_src[i] for i in batch_idx]
            tgt_b = [train_tgt[i] for i in batch_idx]
            assoc_b = [train_assoc[i] for i in batch_idx]
            src, tgt, assoc, pm_s, pm_t = collate_downstream(src_b, tgt_b, assoc_b)
            s = src.to(device); t = tgt.to(device)
            a = assoc.to(device); ps = pm_s.to(device); pt = pm_t.to(device)
            es = proj(s); et = proj(t)
            logits = tracker(es, et)
            loss, acc = downstream_loss(logits, a, ps, pt, pos_weight=5.0)
            if loss.item() == 0: continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_loss += loss.item(); tr_acc += acc; n_b += 1

        # Validation
        proj.eval(); tracker.eval()
        with torch.no_grad():
            val_loss, val_acc, n_v = 0, 0, 0
            for start in range(0, len(val_src), batch_size):
                src_b = val_src[start:start+batch_size]
                tgt_b = val_tgt[start:start+batch_size]
                assoc_b = val_assoc[start:start+batch_size]
                src, tgt, assoc, pm_s, pm_t = collate_downstream(src_b, tgt_b, assoc_b)
                s = src.to(device); t = tgt.to(device)
                a = assoc.to(device); ps = pm_s.to(device); pt = pm_t.to(device)
                es = proj(s); et = proj(t)
                logits = tracker(es, et)
                loss, acc = downstream_loss(logits, a, ps, pt, pos_weight=5.0)
                if loss.item() == 0: continue
                val_loss += loss.item(); val_acc += acc; n_v += 1

        rows.append({
            "run": label, "epoch": epoch,
            "train_loss": tr_loss / max(n_b, 1), "train_acc": tr_acc / max(n_b, 1),
            "val_loss": val_loss / max(n_v, 1), "val_acc": val_acc / max(n_v, 1),
        })

    return pd.DataFrame(rows)


# ─── Micro-SSL pretraining (same as before) ────────────────────────────────────

def nt_xent_loss(z, temperature=0.05):
    """NT-Xent loss over concatenated two-view embeddings (first half = view 1)."""
    B = z.shape[0] // 2
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, targets)

def micro_ssl_pretrain(frames, n_steps=200, lr=1e-3, n_frames=30):
    """Returns trained SSLPretrainedProj state_dict."""
    proj = SSLPretrainedProj().to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    feats_list, lbls_list = [], []
    for mp, ip in frames[:n_frames]:
        r = load_frame(mp, ip)
        if r is None: continue
        c, lbl, emb = r
        if len(lbl) < 2: continue
        c2 = c + np.random.randn(*c.shape).astype(np.float32) * 4
        p2 = extract_patches(imread(ip).astype(np.float32), c2)  # reload for fair DINO
        e2 = compute_dino(p2)
        feats_list.append((torch.from_numpy(emb).float(), torch.from_numpy(e2).float()))
        lbls_list.append(lbl)
    if not feats_list: return None
    for step in range(n_steps):
        losses = []
        proj.train()
        for (e1, e2), lbl in zip(feats_list, lbls_list):
            n = min(len(e1), len(e2))
            if n < 2: continue
            z = torch.cat([proj(e1[:n].to(device)), proj(e2[:n].to(device))], dim=0)
            loss = nt_xent_loss(z)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
    return proj.state_dict()


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    """Run the SSL-init vs random-init downstream convergence comparison."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="../data/vanvliet")
    p.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL")
    p.add_argument("--max-pairs", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--outdir", default="runs/downstream_convergence")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]
    pairs = scan_pairs(args.data_root, conditions, max_pairs=args.max_pairs)
    logger.info(f"Adjacent frame pairs: {len(pairs)}")
    train_data, val_data = prepare_data(pairs, max_pairs=80)
    logger.info(f"  Train: {len(train_data[0])} pairs, Val: {len(val_data[0])} pairs")

    # Get frames for SSL pretraining (first N frames of each condition)
    frames = []
    for cond in conditions:
        dr = Path(args.data_root)
        for exp in sorted((dr / cond).iterdir()):
            if not exp.is_dir(): continue
            tra_dir = exp / "TRA"; img_dir = exp / "img"
            if not tra_dir.exists(): continue
            for m in sorted(tra_dir.glob("man_track*.tif"))[:5]:
                stem = m.stem.replace("man_track", "")
                try:
                    fi = int(stem)
                except ValueError: continue
                ip = img_dir / f"t{fi:06d}.tif"
                if ip.exists():
                    frames.append((str(m), str(ip)))

    logger.info(f"SSL pretraining frames: {len(frames)}")
    logger.info("Running Micro-SSL pretraining...")
    ssl_state = micro_ssl_pretrain(frames, n_steps=200, n_frames=30)
    logger.info(f"  Micro-SSL done: {'success' if ssl_state else 'failed'}")

    # Downstream: SSL-init
    logger.info("Training: SSL-pretrained init...")
    df_ssl = run_downstream(train_data, val_data, ssl_proj_state=ssl_state,
                            n_epochs=args.epochs, label="SSL-pretrained")

    # Downstream: random init
    logger.info("Training: Random init...")
    df_rand = run_downstream(train_data, val_data, ssl_proj_state=None,
                             n_epochs=args.epochs, label="Random-init")

    df = pd.concat([df_ssl, df_rand], ignore_index=True)
    df.to_csv(outdir / "convergence.csv", index=False)

    # Summary
    ssl_best = df_ssl[df_ssl["val_acc"] == df_ssl["val_acc"].max()]
    rand_best = df_rand[df_rand["val_acc"] == df_rand["val_acc"].max()]
    ssl_epoch_to_90 = df_ssl[df_ssl["val_acc"] >= 0.90]["epoch"].min() if (df_ssl["val_acc"] >= 0.90).any() else None
    rand_epoch_to_90 = df_rand[df_rand["val_acc"] >= 0.90]["epoch"].min() if (df_rand["val_acc"] >= 0.90).any() else None

    print("\n" + "="*65)
    print("DOWNSTREAM CONVERGENCE COMPARISON")
    print("="*65)
    print(f"  {'':<25} {'SSL-init':>12} {'Random':>12} {'Δ':>12}")
    print(f"  {'-'*49}")
    print(f"  {'Best val acc':<25} {ssl_best['val_acc'].values[0]:>10.4f}  {rand_best['val_acc'].values[0]:>10.4f}  "
          f"{ssl_best['val_acc'].values[0] - rand_best['val_acc'].values[0]:>+10.4f}")
    print(f"  {'Best val loss':<25} {ssl_best['val_loss'].values[0]:>10.4f}  {rand_best['val_loss'].values[0]:>10.4f}  "
          f"{ssl_best['val_loss'].values[0] - rand_best['val_loss'].values[0]:>+10.4f}")
    if ssl_epoch_to_90 and rand_epoch_to_90:
        speedup = rand_epoch_to_90 / max(ssl_epoch_to_90, 1)
        print(f"  {'Epochs to 90% val acc':<25} {ssl_epoch_to_90:>10}  {rand_epoch_to_90:>10}  "
              f"{'~' + str(speedup) + 'x faster':>12}")
    elif ssl_epoch_to_90 and not rand_epoch_to_90:
        print(f"  {'SSL reached 90% in':<25} {ssl_epoch_to_90:>10}  {'NEVER':>12}  {'SSL only':>12}")
    print("="*65)

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, color in [("SSL-pretrained", "#2ecc71"), ("Random-init", "#e74c3c")]:
        sub = df[df["run"] == label]
        axes[0].plot(sub["epoch"], sub["val_loss"], color=color, label=label, linewidth=2)
        axes[1].plot(sub["epoch"], sub["val_acc"], color=color, label=label, linewidth=2)
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    for ax in axes: ax.legend(); ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Downstream Convergence: SSL vs Random</title>",
        "<style>body{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px}",
        "img{max-width:100%;margin:15px 0}</style></head><body>",
        "<h1>Downstream Convergence Test</h1>",
        f"<p>SSL-pretrained DINO projection vs Random init on {len(train_data[0])} train / {len(val_data[0])} val adjacent-frame pairs.</p>",
        f'<img src="data:image/png;base64,{b64}" /></body></html>'
    ]
    with open(outdir / "convergence.html", "w") as f:
        f.write("\n".join(html))
    logger.info(f"Report: {outdir / 'convergence.html'}")

if __name__ == "__main__":
    main()
