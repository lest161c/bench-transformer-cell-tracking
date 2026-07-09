"""End-to-end SSL → downstream transfer test. Proves (or disproves) sufficiency.

Phase 1: SSL pretraining comparison (coordinate shortcut diagnostic)
Phase 2: Downstream transfer — frozen encoder + association head trained on real tracking data

Answers: "Does fixing the coordinate shortcut actually speed up downstream convergence?"

Usage:
    cd benchmark_ssl
    .venv/bin/python diagnose_end_to_end.py
    .venv/bin/python diagnose_end_to_end.py --ssl-steps 200 --downstream-steps 100 --max-frames 25
"""

import argparse, logging, time, sys, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tifffile import imread
from skimage.measure import regionprops_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("e2e")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ─── Shared constants ───────────────────────────────────────────────────────────
DINO_DIM, PATCH_SIZE = 384, 64

# ─── DINO backbone ─────────────────────────────────────────────────────────────
logger.info("Loading DINOv2...")
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

@torch.no_grad()
def compute_dino_embs(patches_np):
    if len(patches_np) == 0: return np.zeros((0, DINO_DIM), dtype=np.float32)
    pmin = patches_np.min(axis=(1,2), keepdims=True)
    pmax = patches_np.max(axis=(1,2), keepdims=True)
    pn = (patches_np - pmin) / (pmax - pmin + 1e-8)
    t = torch.from_numpy(pn).float().unsqueeze(1).to(device)
    t = F.interpolate(t, size=(224,224), mode="bilinear", align_corners=False)
    t = t.expand(-1, 3, -1, -1)
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    t = (t - mean) / std
    return dino(t).cpu().numpy()

def extract_patches(img, centroids):
    h, w = img.shape[-2:]; half = PATCH_SIZE // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        cy_i, cx_i = np.clip(cy_i, 0, h-1), np.clip(cx_i, 0, w-1)
        y1, x1 = cy_i-half, cx_i-half
        y2, x2 = cy_i+half, cx_i+half
        pt, pb = max(0,-y1), max(0, y2-h)
        pl, pr = max(0,-x1), max(0, x2-w)
        y1c, x1c = max(0, y1), max(0, x1)
        y2c, x2c = min(h, y2), min(w, x2)
        crop = img[y1c:y2c, x1c:x2c] if y2c > y1c and x2c > x1c else np.zeros((1,1), dtype=np.float32)
        if pt or pb or pl or pr: crop = np.pad(crop, ((pt,pb),(pl,pr)), mode="reflect")
        if crop.shape != (PATCH_SIZE, PATCH_SIZE):
            crop = np.pad(crop, tuple((0, max(0, t)) for t in [PATCH_SIZE - s for s in crop.shape]),
                          mode="reflect")[:PATCH_SIZE, :PATCH_SIZE]
        patches.append(crop)
    return np.stack(patches).astype(np.float32) if patches else np.zeros((0,PATCH_SIZE,PATCH_SIZE), dtype=np.float32)

# ─── Positional encodings ──────────────────────────────────────────────────────
class FourierPE(nn.Module):
    def __init__(self, cd=2, pd=32):
        super().__init__()
        self.d = cd * pd * 2
        self.register_buffer("freqs", 2.0 ** torch.linspace(0.0, 10.0, pd))
    def forward(self, c):
        return torch.cat([torch.sin(c[:,:,i:i+1] * self.freqs.view(1,1,-1)) for i in range(c.shape[-1])] +
                         [torch.cos(c[:,:,i:i+1] * self.freqs.view(1,1,-1)) for i in range(c.shape[-1])], dim=-1)

class NoPE(nn.Module):
    def __init__(self, d): super().__init__(); self.d = d; self.token = nn.Parameter(torch.randn(1,1,d)*0.02)
    def forward(self, c): return self.token.expand(c.shape[0], c.shape[1], -1)

# ─── SSL Encoder ────────────────────────────────────────────────────────────────
class SSLEncoder(nn.Module):
    def __init__(self, pe_dim, d_model=256, out_dim=64, use_dino=True):
        super().__init__()
        self.use_dino = use_dino
        self.pe_proj = nn.Linear(pe_dim, d_model)
        if use_dino: self.dino_proj = nn.Linear(DINO_DIM, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model,128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, dino_feats, pe):
        p = F.normalize(self.pe_proj(pe), dim=-1)
        x = self.norm(p + F.normalize(self.dino_proj(dino_feats), dim=-1)) if self.use_dino else self.norm(p)
        return F.normalize(self.mlp(x), dim=-1)

def nt_xent_loss(z, temp=0.05):
    B = z.shape[0] // 2
    sim = z @ z.T / temp; sim.fill_diagonal_(-1e9)
    return F.cross_entropy(sim, torch.cat([torch.arange(B,2*B), torch.arange(B)]).to(z.device))

# ─── Association head (mini Trackastra decoder) ─────────────────────────────────
class MiniAssocHead(nn.Module):
    """Simplified Trackastra association: head_x(emb_t) · head_y(emb_{t+1})^T → BCE."""
    def __init__(self, in_dim=64, head_dim=32):
        super().__init__()
        self.proj_x = nn.Linear(in_dim, head_dim)
        self.proj_y = nn.Linear(in_dim, head_dim)
    def forward(self, z_t, z_next):
        """z_t: (N1,D), z_next: (N2,D) → scores: (N1,N2)"""
        return torch.sigmoid(self.proj_x(z_t) @ self.proj_y(z_next).T)

def bce_assoc_loss(scores, labels_t, labels_next):
    """BCE on association matrix. Positive: same label. Negative: different label."""
    N1, N2 = len(labels_t), len(labels_next)
    target = torch.zeros(N1, N2, device=scores.device)
    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_next):
            if lt == ln: target[i, j] = 1.0
    return F.binary_cross_entropy(scores, target)

def assoc_accuracy(scores, labels_t, labels_next):
    """Accuracy at threshold 0.5."""
    N1, N2 = len(labels_t), len(labels_next)
    pred = (scores > 0.5).cpu()
    target = torch.zeros(N1, N2, dtype=torch.bool)
    for i, lt in enumerate(labels_t):
        for j, ln in enumerate(labels_next):
            if lt == ln: target[i, j] = True
    return (pred == target).float().mean().item()

# ─── Data loading ──────────────────────────────────────────────────────────────
def load_frame(mask_path, img_path):
    mask = imread(mask_path); img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8)); img = np.clip((img-p1)/(p998-p1+1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label","centroid"))
    if not props or len(props["label"]) < 2: return None
    coords = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return coords, props["label"].astype(np.int32), img

def scan_frames(data_root, conditions, max_frames):
    dr = Path(data_root); frames = []
    for cond in conditions:
        for exp in sorted((dr/cond).iterdir()):
            if not exp.is_dir(): continue
            tra, img_dir = exp / "TRA", exp / "img"
            if not tra.exists() or not img_dir.exists(): continue
            for m in sorted(tra.glob("man_track*.tif")):
                stem = m.stem.replace("man_track","")
                try: fi = int(stem)
                except ValueError: continue
                ip = img_dir / f"t{fi:06d}.tif"
                if ip.exists(): frames.append((str(m),str(ip)))
                if len(frames) >= max_frames: return frames
    return frames

def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame PAIRS from same experiment."""
    dr = Path(data_root); pairs = []
    for cond in conditions:
        for exp in sorted((dr/cond).iterdir()):
            if not exp.is_dir(): continue
            tra, img_dir = exp / "TRA", exp / "img"
            if not tra.exists() or not img_dir.exists(): continue
            masks = sorted(tra.glob("man_track*.tif"))
            for i in range(len(masks)-1):
                st1 = masks[i].stem.replace("man_track","")
                st2 = masks[i+1].stem.replace("man_track","")
                try: f1, f2 = int(st1), int(st2)
                except ValueError: continue
                if f2 == f1 + 1:
                    ip1 = img_dir / f"t{f1:06d}.tif"
                    ip2 = img_dir / f"t{f2:06d}.tif"
                    if ip1.exists() and ip2.exists():
                        pairs.append((str(masks[i]), str(masks[i+1]), str(ip1), str(ip2)))
                if len(pairs) >= max_pairs: return pairs
    return pairs

# ─── Distortions (SSL only) ────────────────────────────────────────────────────
def apply_jitter(c, std): return c + np.random.randn(*c.shape).astype(np.float32)*std
def apply_affine(c, deg, sr):
    t = np.random.uniform(-deg,deg)/180*np.pi; sx, sy = np.random.uniform(*sr), np.random.uniform(*sr)
    return c @ np.array([[sx*np.cos(t), -sx*np.sin(t)], [sy*np.sin(t), sy*np.cos(t)]])
def apply_dropout(c, l, p): k = np.random.rand(len(l))>p; return c[k], l[k]
def distort(coords, labels, mode):
    if mode=="full":
        c = apply_affine(coords.copy(), 10, (0.9,1.1)); c = apply_jitter(c, 4)
        c, l = apply_dropout(c, labels.copy(), 0.1); return c, l
    return apply_jitter(coords.copy(), 4), labels.copy()

# ─── Phase 1: SSL pretraining ─────────────────────────────────────────────────
def phase1_ssl(frames, args):
    """Train SSL encoders for three modes. Return {mode: encoder, pos_enc}."""
    logger.info(f"\n{'='*60}\nPHASE 1: SSL PRETRAINING\n{'='*60}")

    # Precompute SSL data (single frames, two distorted views)
    ssl_data = []
    for mp, ip in frames[:args.max_frames]:
        r = load_frame(mp, ip)
        if r is None: continue
        c1, lbl, img = r
        c2, _ = distort(c1.copy(), lbl.copy(), args.distortion)
        p1, p2 = extract_patches(img, c1), extract_patches(img, c2)
        ssl_data.append((compute_dino_embs(p1), compute_dino_embs(p2), c1, c2, lbl))
    logger.info(f"SSL frames: {len(ssl_data)}")

    pe_dim = 2 * args.pos_per_dim * 2
    fourier_pe = FourierPE(pd=args.pos_per_dim).to(device)
    noise_pe   = NoPE(pe_dim).to(device)

    configs = [
        ("A", fourier_pe, True,  "PE+coords + DINO"),
        ("B", noise_pe,   True,  "PE+noise  + DINO"),
        ("C", fourier_pe, False, "PE+coords ONLY"),
    ]

    encoders = {}
    for mode, pe, use_dino, desc in configs:
        enc = SSLEncoder(pe_dim, args.d_model, use_dino=use_dino).to(device)
        opt = torch.optim.Adam(enc.parameters(), lr=args.ssl_lr)
        logger.info(f"  SSL Mode {mode} ({desc}): {args.ssl_steps} steps")

        gpu_data = [(torch.from_numpy(e1).float().to(device),
                     torch.from_numpy(e2).float().to(device),
                     torch.from_numpy(c1).float().to(device),
                     torch.from_numpy(c2).float().to(device), lbl)
                    for e1, e2, c1, c2, lbl in ssl_data]

        for step in range(args.ssl_steps):
            enc.train(); loss_total = 0.0; n_batches = 0
            for de1, de2, c1, c2, lbl in gpu_data:
                n = min(len(de1), len(de2))
                if n < 2: continue
                c1n, c2n = c1[:n].unsqueeze(0), c2[:n].unsqueeze(0)
                pe1, pe2 = pe(c1n).squeeze(0), pe(c2n).squeeze(0)
                z = torch.cat([enc(de1[:n], pe1), enc(de2[:n], pe2)])
                loss = nt_xent_loss(z)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_total += loss.item(); n_batches += 1
            if (step+1) % 50 == 0:
                logger.info(f"    [{mode}] step {step+1:4d}: loss={loss_total/max(1,n_batches):.4f}")

        encoders[mode] = {"encoder": enc, "pos_enc": pe}
        logger.info(f"    [{mode}] done: final loss={loss_total/max(1,n_batches):.6f}")

    return encoders, pe_dim

# ─── Phase 2: Downstream transfer ──────────────────────────────────────────────
def phase2_downstream(encoders, pe_dim, pairs, args):
    """Train association heads on consecutive frame pairs. Compare convergence."""
    logger.info(f"\n{'='*60}\nPHASE 2: DOWNSTREAM TRANSFER\n{'='*60}")

    # Split pairs into train/val
    np.random.seed(SEED)
    idx = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * 0.2))
    train_pairs = [pairs[i] for i in idx[:-n_val]]
    val_pairs   = [pairs[i] for i in idx[-n_val:]]
    logger.info(f"Consecutive frame pairs: {len(train_pairs)} train + {len(val_pairs)} val")

    # Precompute DINO features for all frames
    pair_data = {"train": [], "val": []}
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for mt, mn, it, in_ in split_pairs:
            rt = load_frame(mt, it); rn = load_frame(mn, in_)
            if rt is None or rn is None: continue
            ct, lt, imgt = rt; cn, ln, imgn = rn
            # Only keep cells present in BOTH frames (labels_in_both)
            shared = set(lt) & set(ln)
            if len(shared) < 2: continue
            idx_t = [i for i, l in enumerate(lt) if l in shared]
            idx_n = [i for i, l in enumerate(ln) if l in shared]
            ct_s, lt_s = ct[idx_t], lt[idx_t]
            cn_s, ln_s = cn[idx_n], ln[idx_n]
            pt_s = extract_patches(imgt, ct_s); pn_s = extract_patches(imgn, cn_s)
            pair_data[split_name].append({
                "dino_t": torch.from_numpy(compute_dino_embs(pt_s)).float(),
                "dino_n": torch.from_numpy(compute_dino_embs(pn_s)).float(),
                "coords_t": torch.from_numpy(ct_s).float(),
                "coords_n": torch.from_numpy(cn_s).float(),
                "labels_t": torch.from_numpy(lt_s).long(),
                "labels_n": torch.from_numpy(ln_s).long(),
            })
    logger.info(f"Processed: {len(pair_data['train'])} train pairs, {len(pair_data['val'])} val pairs")

    if len(pair_data["train"]) < 2:
        logger.error("Not enough training pairs. Need more frames.")
        return None

    # For each pretrained encoder + random baseline
    results = {}
    all_modes = dict(encoders)
    all_modes["R"] = {"encoder": None, "pos_enc": None}  # random init marker

    # Only test modes that exist
    available = sorted(set(all_modes.keys()) - {"R"})  # e.g. ["A", "B"] or ["A", "B", "C"]
    test_order = available + ["R"]  # R always last

    for mode in test_order:
        info = all_modes[mode]
        mode_labels = {"A": "PE+coords+DINO", "B": "PE+noise+DINO", "C": "PE ONLY", "R": "RANDOM init"}
        mode_name = mode_labels.get(mode, mode)

        if mode == "R":
            # Random encoder — use FourierPE for fair comparison to A
            enc = SSLEncoder(pe_dim, args.d_model, use_dino=(mode != "C")).to(device)
            pe_mod = FourierPE(pd=args.pos_per_dim).to(device)
        else:
            enc = copy.deepcopy(info["encoder"])
            pe_mod = info["pos_enc"]  # use SAME PE as SSL training for consistency
            # Important: Mode B uses NoPE for BOTH SSL and downstream — fair!

        # Freeze encoder
        for p in enc.parameters(): p.requires_grad = False
        enc.eval()

        # Association head (trained from scratch)
        head = MiniAssocHead(in_dim=64, head_dim=32).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=args.downstream_lr)

        logger.info(f"\n  Downstream [{mode}] {mode_name}")

        rows = []
        for step in range(args.downstream_steps):
            head.train()
            train_losses, train_accs = [], []
            for pd_item in pair_data["train"]:
                de_t = pd_item["dino_t"].to(device); de_n = pd_item["dino_n"].to(device)
                ct = pd_item["coords_t"].unsqueeze(0).to(device)
                cn = pd_item["coords_n"].unsqueeze(0).to(device)
                lt = pd_item["labels_t"]; ln = pd_item["labels_n"]

                pe_t = pe_mod(ct).squeeze(0); pe_n = pe_mod(cn).squeeze(0)
                with torch.no_grad():
                    zt = enc(de_t, pe_t); zn = enc(de_n, pe_n)
                scores = head(zt, zn)
                loss = bce_assoc_loss(scores, lt, ln)
                opt.zero_grad(); loss.backward(); opt.step()
                train_losses.append(loss.item())
                train_accs.append(assoc_accuracy(scores.detach(), lt, ln))

            # Eval every 10 steps
            if step % 10 == 0 or step == args.downstream_steps - 1:
                head.eval()
                val_losses, val_accs = [], []
                with torch.no_grad():
                    for pd_item in pair_data["val"]:
                        de_t = pd_item["dino_t"].to(device); de_n = pd_item["dino_n"].to(device)
                        ct = pd_item["coords_t"].unsqueeze(0).to(device)
                        cn = pd_item["coords_n"].unsqueeze(0).to(device)
                        lt = pd_item["labels_t"]; ln = pd_item["labels_n"]
                        pe_t = pe_mod(ct).squeeze(0); pe_n = pe_mod(cn).squeeze(0)
                        zt = enc(de_t, pe_t); zn = enc(de_n, pe_n)
                        scores = head(zt, zn)
                        val_losses.append(bce_assoc_loss(scores, lt, ln).item())
                        val_accs.append(assoc_accuracy(scores, lt, ln))

                t_loss = float(np.mean(train_losses)); t_acc = float(np.mean(train_accs))
                v_loss = float(np.mean(val_losses)); v_acc = float(np.mean(val_accs))
                rows.append({"step": step, "train_loss": t_loss, "train_acc": t_acc,
                            "val_loss": v_loss, "val_acc": v_acc})

        results[mode] = pd.DataFrame(rows)
        final = rows[-1]
        logger.info(f"    [{mode}] step {args.downstream_steps}: val_loss={final['val_loss']:.4f}  val_acc={final['val_acc']:.4f}")

    return results


# ─── Report ─────────────────────────────────────────────────────────────────────
def make_report(ssl_encoders, downstream_results, outdir, args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"A": "#e74c3c", "B": "#2ecc71", "C": "#3498db", "R": "#95a5a6"}
    labels = {"A": "PE+coords+DINO (current)", "B": "PE+noise+DINO (fix)", "C": "PE only (smoking gun)", "R": "Random init (baseline)"}

    plot_order = sorted(set(downstream_results.keys()), key=lambda m: {"R": "z", "A": "a", "B": "b", "C": "c"}.get(m, m))

    # Panel 1: Downstream val loss
    ax = axes[0]
    for mode in plot_order:
        if mode not in downstream_results: continue
        df = downstream_results[mode]
        ax.plot(df["step"], df["val_loss"], color=colors.get(mode, "#333"), label=labels.get(mode, mode), linewidth=2)
    ax.set_xlabel("Downstream training step"); ax.set_ylabel("Val BCE Loss")
    ax.set_title("Downstream Convergence: Val Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Downstream val accuracy
    ax = axes[1]
    for mode in plot_order:
        if mode not in downstream_results: continue
        df = downstream_results[mode]
        ax.plot(df["step"], df["val_acc"], color=colors.get(mode, "#333"), linewidth=2)
    ax.set_xlabel("Downstream training step"); ax.set_ylabel("Val Accuracy")
    ax.set_title("Downstream Convergence: Val Accuracy")
    ax.grid(True, alpha=0.3)

    # Panel 3: Steps to reach 0.8 accuracy (speed comparison)
    ax = axes[2]
    mode_names = []; conv_steps = []; bar_colors = []
    bar_order = sorted(plot_order, key=lambda m: {"R": 99, "A": 1, "B": 2, "C": 3}.get(m, 50))
    for mode in bar_order:
        if mode not in downstream_results: continue
        df = downstream_results[mode]
        # Find first step where val_acc > 0.8
        hit = df[df["val_acc"] > 0.8]
        s = hit["step"].iloc[0] if len(hit) else args.downstream_steps
        mode_names.append(labels[mode].split("(")[0].strip())
        conv_steps.append(s)
        bar_colors.append(colors[mode])
    bars = ax.bar(mode_names, conv_steps, color=bar_colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Steps to val_acc > 0.8"); ax.set_title("Convergence Speed (lower = faster)")
    for bar, v in zip(bars, conv_steps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(v), ha="center", fontweight="bold")
    ax.axhline(args.downstream_steps, color="gray", ls=":", alpha=0.5, label="max steps")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"End-to-End SSL → Downstream Transfer Test\n"
                 f"{args.max_frames} SSL frames, {args.ssl_steps} SSL steps, {args.downstream_steps} downstream steps",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = outdir / "end_to_end.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig); logger.info(f"Figure: {path}")

    # ─── Summary text ───────────────────────────────────────────────────────
    lines = []
    lines.append("="*65)
    lines.append("END-TO-END SSL → DOWNSTREAM TRANSFER — VERDICT")
    lines.append("="*65)
    lines.append("")

    for mode in ["R", "A", "B", "C"]:
        if mode not in downstream_results: continue
        df = downstream_results[mode]
        final = df.iloc[-1]
        initial = df.iloc[0]
        acc = final["val_acc"]; loss = final["val_loss"]
        acc0 = initial["val_acc"]
        hit = df[df["val_acc"] > 0.8]
        conv = hit["step"].iloc[0] if len(hit) else "never"
        lines.append(f"  {labels[mode]}:")
        lines.append(f"    Val acc:  {acc0:.3f} → {acc:.3f}  (Δ={acc-acc0:+.3f})")
        lines.append(f"    Val loss: {loss:.4f}")
        lines.append(f"    Steps to acc>0.8: {conv}")

    lines.append("")
    # Compare B vs A
    if "A" in downstream_results and "B" in downstream_results:
        df_a = downstream_results["A"]; df_b = downstream_results["B"]
        hit_a = df_a[df_a["val_acc"] > 0.8]; hit_b = df_b[df_b["val_acc"] > 0.8]
        conv_a = hit_a["step"].iloc[0] if len(hit_a) else args.downstream_steps
        conv_b = hit_b["step"].iloc[0] if len(hit_b) else args.downstream_steps
        acc_a = df_a.iloc[-1]["val_acc"]; acc_b = df_b.iloc[-1]["val_acc"]

        if conv_b < conv_a - 2:
            lines.append(f"  ✓ Mode B converges {conv_a - conv_b} steps FASTER than Mode A")
            lines.append(f"    → Removing coordinates during SSL IMPROVES downstream convergence.")
        elif conv_b > conv_a + 2:
            lines.append(f"  ⚠ Mode A converges {conv_b - conv_a} steps FASTER than Mode B")
        else:
            lines.append(f"  ~ Similar convergence speed (Δ={conv_a-conv_b} steps)")

        if acc_b > acc_a + 0.02:
            lines.append(f"  ✓ Mode B achieves higher final accuracy (+{acc_b-acc_a:.3f})")
        elif acc_a > acc_b + 0.02:
            lines.append(f"  ⚠ Mode A achieves higher final accuracy (+{acc_a-acc_b:.3f})")
        else:
            lines.append(f"  ~ Similar final accuracy (Δ={acc_b-acc_a:.3f})")

    lines.append("")
    lines.append("  Interpretation:")
    if "B" in downstream_results and "A" in downstream_results:
        conv_b_val = conv_b if isinstance(conv_b, int) else args.downstream_steps
        conv_a_val = conv_a if isinstance(conv_a, int) else args.downstream_steps
        if conv_b_val < conv_a_val:
            lines.append("    SSL WITHOUT coordinates → FASTER downstream convergence")
            lines.append("    The coordinate shortcut is REAL and REMOVING it helps.")
            lines.append("    Fix: NoPositionalEncoding during SSL pretraining.")
        else:
            lines.append("    SSL does not help downstream convergence at this scale.")
            lines.append("    The coordinate shortcut exists but fixing it alone is insufficient.")
            lines.append("    The decoder gap or NT-Xent/BCE mismatch may dominate.")

    lines.append("")
    lines.append(f"Full results: {outdir}")
    lines.append(f"  {outdir}/end_to_end.png")
    for mode in downstream_results:
        if mode in downstream_results:
            lines.append(f"  {outdir}/downstream_{mode}.csv")
    lines.append("="*65)

    verdict = "\n".join(lines)
    print(verdict)
    with open(outdir / "verdict.txt", "w") as f: f.write(verdict)

    # Save CSVs
    for mode in ["R", "A", "B", "C"]:
        if mode in downstream_results:
            downstream_results[mode].to_csv(outdir / f"downstream_{mode}.csv", index=False)

    return fig


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="End-to-end SSL → downstream transfer test")
    p.add_argument("--data-root", default="../data/vanvliet")
    p.add_argument("--conditions", default="rpsM")
    p.add_argument("--max-frames", type=int, default=25, help="SSL frames (single)")
    p.add_argument("--max-pairs", type=int, default=20, help="Downstream frame pairs")
    p.add_argument("--ssl-steps", type=int, default=200)
    p.add_argument("--ssl-lr", type=float, default=1e-3)
    p.add_argument("--downstream-steps", type=int, default=100)
    p.add_argument("--downstream-lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--pos-per-dim", type=int, default=32)
    p.add_argument("--distortion", default="jitter4", choices=["jitter4","full"])
    p.add_argument("--outdir", default="runs/diagnose_end_to_end")
    p.add_argument("--skip-c", action="store_true", help="Skip Mode C (PE only)")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",")]

    # ─── Phase 1: SSL pretraining ──────────────────────────────────────────
    frames = scan_frames(args.data_root, conditions, args.max_frames)
    logger.info(f"Phase 1: {len(frames)} single frames for SSL")

    if args.skip_c:
        # Only run A and B
        logger.info("Skipping Mode C (--skip-c)")
        ssl_encoders, pe_dim = phase1_ssl_ab(frames, args)
    else:
        ssl_encoders, pe_dim = phase1_ssl(frames, args)

    # ─── Phase 2: Downstream transfer ──────────────────────────────────────
    pairs = scan_consecutive_pairs(args.data_root, conditions, args.max_pairs)
    logger.info(f"Phase 2: {len(pairs)} consecutive frame pairs for downstream")

    if len(pairs) < 3:
        logger.error("Need at least 3 consecutive frame pairs. Increase --max-frames or check data.")
        sys.exit(1)

    downstream_results = phase2_downstream(ssl_encoders, pe_dim, pairs, args)

    if downstream_results is None:
        logger.error("Downstream phase failed. Check data and error messages.")
        sys.exit(1)

    # ─── Report ────────────────────────────────────────────────────────────
    make_report(ssl_encoders, downstream_results, outdir, args)
    logger.info(f"Done. Results in {outdir}/")


def phase1_ssl_ab(frames, args):
    """SSL pretraining for modes A and B only (faster)."""
    pe_dim = 2 * args.pos_per_dim * 2
    fourier_pe = FourierPE(pd=args.pos_per_dim).to(device)
    noise_pe = NoPE(pe_dim).to(device)

    ssl_data = []
    for mp, ip in frames[:args.max_frames]:
        r = load_frame(mp, ip)
        if r is None: continue
        c1, lbl, img = r
        c2, _ = distort(c1.copy(), lbl.copy(), args.distortion)
        p1, p2 = extract_patches(img, c1), extract_patches(img, c2)
        ssl_data.append((compute_dino_embs(p1), compute_dino_embs(p2), c1, c2, lbl))

    encoders = {}
    for mode, pe, desc in [("A", fourier_pe, "PE+coords+DINO"), ("B", noise_pe, "PE+noise+DINO")]:
        enc = SSLEncoder(pe_dim, args.d_model, use_dino=True).to(device)
        opt = torch.optim.Adam(enc.parameters(), lr=args.ssl_lr)
        gpu_data = [(torch.from_numpy(e1).float().to(device),
                     torch.from_numpy(e2).float().to(device),
                     torch.from_numpy(c1).float().to(device),
                     torch.from_numpy(c2).float().to(device), lbl)
                    for e1, e2, c1, c2, lbl in ssl_data]
        for step in range(args.ssl_steps):
            enc.train(); loss_total, n = 0.0, 0
            for de1, de2, c1, c2, lbl in gpu_data:
                m = min(len(de1), len(de2))
                if m < 2: continue
                c1n, c2n = c1[:m].unsqueeze(0), c2[:m].unsqueeze(0)
                pe1, pe2 = pe(c1n).squeeze(0), pe(c2n).squeeze(0)
                z = torch.cat([enc(de1[:m], pe1), enc(de2[:m], pe2)])
                loss = nt_xent_loss(z)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_total += loss.item(); n += 1
            if (step+1) % 50 == 0:
                logger.info(f"    SSL [{mode}] step {step+1}: loss={loss_total/max(1,n):.4f}")
        encoders[mode] = {"encoder": enc, "pos_enc": pe}
    return encoders, pe_dim


if __name__ == "__main__":
    main()
