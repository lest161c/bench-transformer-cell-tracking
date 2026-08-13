#!/usr/bin/env python3
"""Encoder-freeze downstream test: does SSL encoder produce genuinely useful features?

Runs one config (Mode B NoPE SSL) through downstream twice:
  1. Full training (encoder + decoder + heads) — existing approach
  2. Frozen encoder (decoder + heads only) — isolates encoder quality
  3. Random init frozen encoder — baseline control

If frozen SSL encoder >> frozen random encoder: features are genuinely useful.
If frozen SSL encoder ≈ frozen random encoder: SSL gain is just warm-start.
If full training >> frozen SSL: the encoder must adapt during downstream (SSL/BCE mismatch).

Usage:
  cd src/mini_trackastra
  .venv/bin/python test_freeze.py
"""

import argparse, logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from skimage.measure import regionprops_table
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("freeze_test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, DINO_DIM, PATCH_SIZE = 42, 384, 64
torch.manual_seed(SEED); np.random.seed(SEED)
_dino_model = None


# ─── DINO ────────────────────────────────────────────────────────────────────

def _get_dino():
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    return _dino_model

def compute_dino_embs(patches_np):
    if len(patches_np) == 0: return np.zeros((0, DINO_DIM), dtype=np.float32)
    t = torch.from_numpy(patches_np).float().unsqueeze(1)
    t_224 = F.interpolate(t, size=(224,224), mode="bilinear", align_corners=False)
    t_224 = t_224.expand(-1,3,-1,-1).to(device)
    mn = torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
    sd = torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
    with torch.no_grad(): emb = _get_dino()((t_224-mn)/sd)
    return emb.cpu().numpy()

def free_dino():
    global _dino_model
    if _dino_model: _dino_model.cpu(); _dino_model = None
    torch.cuda.empty_cache()


# ─── Data ────────────────────────────────────────────────────────────────────

def load_frame(mp, ip):
    mask, img = imread(mp), imread(ip).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img-p1)/(p998-p1+1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label","centroid"))
    if not props or len(props["label"])<2: return None
    c = np.stack([props["centroid-0"],props["centroid-1"]], axis=-1).astype(np.float32)
    return c, props["label"].astype(np.int32), img

def extract_patches(img, coords):
    if len(coords)==0: return np.zeros((0,PATCH_SIZE,PATCH_SIZE), dtype=np.float32)
    half = PATCH_SIZE//2; patches=[]
    for cy,cx in coords:
        ci, cj = int(round(float(cy))), int(round(float(cx)))
        ci, cj = np.clip(ci,0,img.shape[0]-1), np.clip(cj,0,img.shape[1]-1)
        y1,x1,y2,x2 = ci-half, cj-half, ci+half, cj+half
        pt,pb = max(0,-y1), max(0,y2-img.shape[0])
        pl,pr = max(0,-x1), max(0,x2-img.shape[1])
        c = img[max(0,y1):min(img.shape[0],y2), max(0,x1):min(img.shape[1],x2)]
        if pt or pb or pl or pr: c = np.pad(c, ((pt,pb),(pl,pr)), mode="reflect")
        if c.shape != (PATCH_SIZE,PATCH_SIZE):
            pad_h = max(0,PATCH_SIZE-c.shape[0]); pad_w = max(0,PATCH_SIZE-c.shape[1])
            c = np.pad(c, ((0,pad_h),(0,pad_w)), mode="reflect")[:PATCH_SIZE,:PATCH_SIZE]
        patches.append(c)
    return np.stack(patches).astype(np.float32)

def scan_pairs(data_root, conditions, max_pairs):
    dr, pairs = Path(data_root), []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir(): continue
            tra, img_dir = exp/"TRA", exp/"img"
            if not tra.exists() or not img_dir.exists(): continue
            masks = sorted(tra.glob("man_track*.tif"))
            for i in range(len(masks)-1):
                try: f1,f2 = int(masks[i].stem.replace("man_track","")), int(masks[i+1].stem.replace("man_track",""))
                except: continue
                if f2!=f1+1: continue
                ip1, ip2 = img_dir/f"t{f1:06d}.tif", img_dir/f"t{f2:06d}.tif"
                if ip1.exists() and ip2.exists():
                    pairs.append((str(masks[i]),str(masks[i+1]),str(ip1),str(ip2)))
                if len(pairs)>=max_pairs: return pairs
    return pairs

def scan_frames(data_root, conditions, max_frames):
    dr, frames = Path(data_root), []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir(): continue
            tra, img_dir = exp/"TRA", exp/"img"
            if not tra.exists() or not img_dir.exists(): continue
            for m in sorted(tra.glob("man_track*.tif")):
                try: fi = int(m.stem.replace("man_track",""))
                except: continue
                ip = img_dir/f"t{fi:06d}.tif"
                if ip.exists(): frames.append((str(m),str(ip)))
                if len(frames)>=max_frames: return frames
    return frames


# ─── Distortions ──────────────────────────────────────────────────────────────

def apply_jitter(c, std): return c + np.random.randn(*c.shape).astype(np.float32)*std
def apply_affine(c, deg, sr):
    a = np.deg2rad(np.random.uniform(-deg,deg))
    s = np.random.uniform(sr[0],sr[1])
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    cn = c.mean(axis=0)
    return ((c-cn)@R.T*s+cn).astype(np.float32)
def apply_dropout(c, labels, p):
    if p<=0: return c, labels
    keep = np.random.random(len(c))>p
    if sum(keep)<2: keep[:2]=True
    return c[keep], labels[keep]
def distort(coords, labels, mode="full"):
    c = apply_affine(coords.copy(),10,(0.9,1.1))
    c = apply_jitter(c,4)
    return apply_dropout(c,labels.copy(),0.1)


# ─── Model ───────────────────────────────────────────────────────────────────

class NoPE(nn.Module):
    def __init__(self,d): super().__init__(); self.token=nn.Parameter(torch.randn(1,1,d)*0.02)
    def forward(self,c): return self.token.expand(c.shape[0],c.shape[1],-1)

class MiniEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, pe_dim=64):
        super().__init__()
        self.dino_proj = nn.Linear(DINO_DIM, d_model)
        self.pos_proj = nn.Linear(pe_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        enc = nn.TransformerEncoderLayer(d_model,nhead,d_model*4,batch_first=True,dropout=0.)
        self.transformer = nn.TransformerEncoder(enc, num_layers)
    def forward(self, de, pe):
        x = self.dino_proj(de)+self.pos_proj(pe); x=self.norm(x)
        return self.transformer(x.unsqueeze(0)).squeeze(0)

class MiniDecoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        dec = nn.TransformerDecoderLayer(d_model,nhead,d_model*4,batch_first=True,dropout=0.)
        self.transformer = nn.TransformerDecoder(dec, num_layers)
    def forward(self, tgt, mem):
        return self.transformer(tgt.unsqueeze(0), mem.unsqueeze(0)).squeeze(0)

class MiniTrackingTransformer(nn.Module):
    def __init__(self, encoder, decoder, d_head=32):
        super().__init__()
        d = encoder.dino_proj.out_features
        self.encoder, self.decoder = encoder, decoder
        self.head_x, self.head_y = nn.Linear(d,d_head), nn.Linear(d,d_head)
        self.scale = d_head**0.5
    def forward(self, de_t, pe_t, de_n, pe_n):
        et = self.encoder(de_t, pe_t); en = self.encoder(de_n, pe_n)
        dec = self.decoder(en, et)
        x = F.normalize(self.head_x(et), dim=-1)
        y = F.normalize(self.head_y(dec), dim=-1)
        return (x@y.T)*self.scale


# ─── Loss/Metrics ─────────────────────────────────────────────────────────────

def nt_xent_loss(z, temp=0.05):
    N=z.shape[0]//2; zn=F.normalize(z,dim=-1); sim=(zn[:N]@zn[N:].T)/temp
    return F.cross_entropy(sim, torch.arange(N,device=z.device))

def bce_assoc_loss(logits, lt, ln):
    N1,N2=logits.shape; target=torch.zeros(N1,N2,device=logits.device)
    for i in range(min(N1,N2)):
        if lt[i]==ln[i]: target[i,i]=1.
    return F.binary_cross_entropy_with_logits(logits, target)

def assoc_acc(logits, lt, ln):
    N=min(len(lt),len(ln)); correct=sum(1 for i in range(N) if lt[i]==ln[logits[i].argmax().item()])
    return correct/max(1,N)

def balanced_acc(logits, lt, ln):
    N1,N2=logits.shape; m=min(N1,N2)
    target=torch.zeros(N1,N2,device=logits.device)
    for i in range(m):
        if lt[i]==ln[i]: target[i,i]=1.
    pred=(logits>0).float()
    tn=((pred==0)&(target==0)).float().sum()
    tp=((pred==1)&(target==1)).float().sum()
    fp=((pred==1)&(target==0)).float().sum()
    fn=((pred==0)&(target==1)).float().sum()
    return ((tp/(tp+fn+1e-8)+tn/(tn+fp+1e-8))/2).item()


# ─── Run one downstream config ────────────────────────────────────────────────

def run_one_downstream(enc, pe_mod, pair_data, steps, lr, d_head, freeze_enc):
    """Train and return DataFrame."""
    dec = MiniDecoder(d_model=enc.dino_proj.out_features, nhead=enc.transformer.layers[0].self_attn.num_heads).to(device)
    model = MiniTrackingTransformer(enc, dec, d_head).to(device)

    params = []
    if freeze_enc:
        for p in enc.parameters(): p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    rows = []
    for step in range(steps):
        model.train()
        tl, ta, tb = [], [], []
        for pd_ in pair_data["train"]:
            de_t, de_n = pd_["dino_t"].to(device), pd_["dino_n"].to(device)
            ct, cn = pd_["coords_t"].unsqueeze(0).to(device), pd_["coords_n"].unsqueeze(0).to(device)
            lt, ln = pd_["labels_t"], pd_["labels_n"]
            pet, pen = pe_mod(ct).squeeze(0), pe_mod(cn).squeeze(0)
            scores = model(de_t, pet, de_n, pen)
            loss = bce_assoc_loss(scores, lt, ln)
            opt.zero_grad(); loss.backward(); opt.step()
            tl.append(loss.item()); sd=scores.detach()
            ta.append(assoc_acc(sd,lt,ln)); tb.append(balanced_acc(sd,lt,ln))

        if step%10==0 or step==steps-1:
            model.eval(); vl,va,vb=[],[],[]
            with torch.no_grad():
                for pd_ in pair_data["val"]:
                    de_t,de_n=pd_["dino_t"].to(device),pd_["dino_n"].to(device)
                    ct,cn=pd_["coords_t"].unsqueeze(0).to(device),pd_["coords_n"].unsqueeze(0).to(device)
                    lt,ln=pd_["labels_t"],pd_["labels_n"]
                    pet,pen=pe_mod(ct).squeeze(0),pe_mod(cn).squeeze(0)
                    scores=model(de_t,pet,de_n,pen)
                    vl.append(bce_assoc_loss(scores,lt,ln).item())
                    va.append(assoc_acc(scores,lt,ln))
                    vb.append(balanced_acc(scores,lt,ln))
            rows.append({"step":step, "train_loss":np.mean(tl), "train_acc":np.mean(ta),
                         "train_bal_acc":np.mean(tb), "val_loss":np.mean(vl),
                         "val_acc":np.mean(va), "val_bal_acc":np.mean(vb)})

    if freeze_enc:
        for p in enc.parameters(): p.requires_grad = True  # restore
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args(a=None):
    p = argparse.ArgumentParser(description="Encoder-freeze downstream test")
    p.add_argument("--data-root", default="../../data/vanvliet")
    p.add_argument("--conditions", default="rpsM,recA,pheA,metA,cib,trpL")
    p.add_argument("--max-ssl-frames", type=int, default=30)
    p.add_argument("--max-pairs", type=int, default=30)
    p.add_argument("--ssl-steps", type=int, default=200)
    p.add_argument("--ssl-lr", type=float, default=0.001)
    p.add_argument("--downstream-steps", type=int, default=100)
    p.add_argument("--downstream-lr", type=float, default=0.001)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--pos-per-dim", type=int, default=16)
    p.add_argument("--outdir", default="runs/freeze_test")
    return p.parse_args(a)


def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(",")]
    pe_dim = args.pos_per_dim*2*2

    logger.info(f"Encoder-freeze test: d_model={args.d_model}, {args.num_layers}L")
    logger.info("Phase 1: Data collection...")

    # Single frames for SSL
    frames = scan_frames(args.data_root, conds, args.max_ssl_frames)
    logger.info(f"  SSL frames: {len(frames)}")

    # Downstream pairs
    pairs = scan_pairs(args.data_root, conds, args.max_pairs)
    np.random.seed(SEED+1); idx=np.random.permutation(len(pairs))
    nv=max(1,int(len(pairs)*0.2))
    tr_pairs, vl_pairs = [pairs[i] for i in idx[:-nv]], [pairs[i] for i in idx[-nv:]]
    logger.info(f"  Downstream: {len(tr_pairs)} train + {len(vl_pairs)} val")

    # Precompute DINO
    ssl_data=[]
    for mp, ip in frames:
        r=load_frame(mp,ip)
        if r is None: continue
        c1, lbl, img = r
        c2, _ = distort(c1.copy(), lbl.copy())
        e1 = compute_dino_embs(extract_patches(img, c1))
        e2 = compute_dino_embs(extract_patches(img, c2))
        ssl_data.append((e1, e2, lbl))

    pair_data = {"train":[], "val":[]}
    for sn, sp in [("train",tr_pairs),("val",vl_pairs)]:
        for mt,mn,it,i_n in sp:
            rt,rn = load_frame(mt,it), load_frame(mn,i_n)
            if rt is None or rn is None: continue
            ct,lt,imgt = rt; cn,ln,imgn = rn
            shared = set(lt)&set(ln)
            if len(shared)<8: continue
            ixt = [i for i,l in enumerate(lt) if l in shared]
            ixn = [i for i,l in enumerate(ln) if l in shared]
            pair_data[sn].append({
                "dino_t": torch.from_numpy(compute_dino_embs(extract_patches(imgt,ct[ixt]))).float(),
                "dino_n": torch.from_numpy(compute_dino_embs(extract_patches(imgn,cn[ixn]))).float(),
                "coords_t": torch.from_numpy(ct[ixt]).float(),
                "coords_n": torch.from_numpy(cn[ixn]).float(),
                "labels_t": torch.from_numpy(lt[ixt]).long(),
                "labels_n": torch.from_numpy(ln[ixn]).long(),
            })
    logger.info(f"  Downstream precomputed: {len(pair_data['train'])} train, {len(pair_data['val'])} val")
    free_dino()

    # Phase 2: SSL pretraining (Mode B = NoPE, best from earlier tests)
    logger.info("Phase 2: SSL pretraining (NoPE + distortions)...")
    noise_pe = NoPE(pe_dim).to(device)
    ssl_enc = MiniEncoder(args.d_model, args.nhead, args.num_layers, pe_dim).to(device)
    opt = torch.optim.Adam(ssl_enc.parameters(), lr=args.ssl_lr)

    gpu_data = [(torch.from_numpy(e1).float().to(device),
                  torch.from_numpy(e2).float().to(device), min(len(e1),len(e2)))
                 for e1,e2,_ in ssl_data]

    for s in range(args.ssl_steps):
        ssl_enc.train(); tl,nb=0.,0
        for idx in torch.randperm(len(gpu_data)):
            de1,de2,n = gpu_data[idx]; n=min(n,len(de1),len(de2))
            if n<2: continue
            dummy=torch.zeros(1,n,2,device=device)
            pe=noise_pe(dummy).squeeze(0)
            z = torch.cat([ssl_enc(de1[:n],pe), ssl_enc(de2[:n],pe)])
            loss = nt_xent_loss(z)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_enc.parameters(),1.0); opt.step()
            tl+=loss.item(); nb+=1
        if (s+1)%50==0 or s==0:
            logger.info(f"  SSL step {s+1:3d}: loss={tl/max(1,nb):.4f}")
    logger.info(f"  SSL done: final loss={tl/max(1,nb):.6f}")

    # Phase 3: Downstream — 4 conditions
    logger.info("\nPhase 3: Downstream comparison")
    logger.info("="*50)

    results = {}
    for label, enc_src, freeze in [
        ("SSL+full", ssl_enc, False),
        ("SSL+frozen", ssl_enc, True),
        ("Rand+full", MiniEncoder(args.d_model,args.nhead,args.num_layers,pe_dim).to(device), False),
        ("Rand+frozen", MiniEncoder(args.d_model,args.nhead,args.num_layers,pe_dim).to(device), True),
    ]:
        pe = noise_pe.to(device) if enc_src is ssl_enc else NoPE(pe_dim).to(device)

        if enc_src is not ssl_enc:
            # Random init encoder (fresh)
            enc_copy = enc_src
        else:
            # Deep copy SSL encoder
            enc_copy = MiniEncoder(args.d_model,args.nhead,args.num_layers,pe_dim).to(device)
            enc_copy.load_state_dict(ssl_enc.state_dict())

        logger.info(f"\n  [{label}] freeze={freeze}")
        df = run_one_downstream(enc_copy, pe, pair_data, args.downstream_steps,
                                args.downstream_lr, args.d_head, freeze)
        results[label] = df
        final = df.iloc[-1]
        logger.info(f"  [{label}] final: val_loss={final['val_loss']:.4f}, "
                     f"val_acc={final['val_acc']:.4f}, bal_acc={final['val_bal_acc']:.4f}")

    # Phase 4: Report
    import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"SSL+full": "#2ecc71", "SSL+frozen": "#27ae60",
              "Rand+full": "#e74c3c", "Rand+frozen": "#c0392b"}
    for ax, metric in [(axes[0],"val_loss"), (axes[1],"val_bal_acc")]:
        for label, df in results.items():
            ax.plot(df["step"], df[metric], color=colors.get(label,"#333"),
                    label=label, linewidth=2)
        ax.set_xlabel("Step"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_title("Val BCE Loss (lower=better)")
    axes[1].set_title("Val Balanced Accuracy (higher=better)")
    axes[1].axhline(0.5, color="gray", ls=":", alpha=0.5)
    plt.suptitle("Encoder-Freeze Ablation: Does SSL Encoder Help?\n"
                 f"({'frozen'}=encoder weights locked, {'full'}=encoder adapts during downstream)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outdir/"freeze_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV
    for label, df in results.items():
        df.to_csv(outdir/f"downstream_{label}.csv", index=False)

    # Verdict
    lines = ["="*60, "ENCODER-FREEZE TEST: VERDICT", "="*60]
    sff = results["SSL+frozen"]
    rff = results["Rand+frozen"]
    sfu = results["SSL+full"]
    rfu = results["Rand+full"]

    delta_frozen = sff["val_bal_acc"].iloc[-1] - rff["val_bal_acc"].iloc[-1]
    delta_full = sfu["val_bal_acc"].iloc[-1] - rfu["val_bal_acc"].iloc[-1]
    ssl_adapt = sfu["val_bal_acc"].iloc[-1] - sff["val_bal_acc"].iloc[-1]

    lines.append(f"\n  Frozen encoder:  SSL={sff['val_bal_acc'].iloc[-1]:.4f}  "
                 f"Rand={rff['val_bal_acc'].iloc[-1]:.4f}  Δ={delta_frozen:+.4f}")
    lines.append(f"  Full training:   SSL={sfu['val_bal_acc'].iloc[-1]:.4f}  "
                 f"Rand={rfu['val_bal_acc'].iloc[-1]:.4f}  Δ={delta_full:+.4f}")
    lines.append(f"  SSL adaptation gain: {ssl_adapt:+.4f} (full - frozen)")

    if delta_frozen > 0.03:
        lines.append("\n  ✓ SSL encoder features are GENUINELY useful (frozen SSL > frozen Rand).")
    else:
        lines.append("\n  ✗ SSL encoder features add LITTLE VALUE (frozen SSL ≈ frozen Rand).")
    if ssl_adapt > 0.03:
        lines.append("  ✓ Adaptation during downstream is SIGNIFICANT (full >> frozen).")
        lines.append("    → SSL + NT-Xent / BCE mismatch means encoder must unlearn/relearn.")
    else:
        lines.append("  ~ Adaptation gain is MINIMAL (frozen ≈ full).")

    verdict = "\n".join(lines)
    (outdir/"verdict.txt").write_text(verdict)
    print("\n"+verdict)
    logger.info(f"Done. Outputs in {outdir}/")


if __name__ == "__main__":
    main()
