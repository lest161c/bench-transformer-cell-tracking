#!/usr/bin/env python3
"""Cell-DINO HPA vitl14 vs DINOv2 vits14: bacteria feature gap comparison."""
import sys, time, logging, gc
import numpy as np
import torch, torch.nn.functional as F
from pathlib import Path
from skimage.measure import regionprops_table
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cell_dino_test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = "/home/leonard.starke@mediainterface.de/.cache/torch/hub/facebookresearch_dinov2_main"
CKPT = "/home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj/cell_dino_vitl14_pretrain_hpa_fov_highres-f57e7934.pth"
log.info(f"Device: {device}")

def load_frame(mp, ip):
    mask, img = imread(mp), imread(ip).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if not props or len(props["label"]) < 2: return None
    c = np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)
    return c, props["label"], img

def scan_frames(root, n=20):
    frames, dr = [], Path(root)
    for d in sorted(dr.glob("rpsM/*")):
        if not d.is_dir(): continue
        tra, imgd = d/"TRA", d/"img"
        if not tra.exists() or not imgd.exists(): continue
        for m in sorted(tra.glob("man_track*.tif")):
            try: fi = int(m.stem.replace("man_track",""))
            except: continue
            ip = imgd/f"t{fi:06d}.tif"
            if ip.exists(): frames.append((str(m), str(ip)))
            if len(frames) >= n: return frames
    return frames

def extract_patch(img, cy, cx, sz):
    h, w = img.shape[-2:]; half = sz//2
    cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
    cy_i, cx_i = np.clip(cy_i,0,h-1), np.clip(cx_i,0,w-1)
    y1,x1,y2,x2 = cy_i-half, cx_i-half, cy_i+half, cx_i+half
    pt,pb = max(0,-y1), max(0,y2-h); pl,pr = max(0,-x1), max(0,x2-w)
    crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if pt or pb or pl or pr: crop = np.pad(crop, ((pt,pb),(pl,pr)), mode="reflect")
    for d in range(2):
        if crop.shape[d] < sz: crop = np.pad(crop, ((0,sz-crop.shape[0]) if d==0 else (0,0), (0,sz-crop.shape[1]) if d==1 else (0,0)), mode="reflect")
    return crop[:sz,:sz]

# ─── Extract all frames first (disk I/O, then GPU processing) ────────────────
frames = scan_frames("../../data/vanvliet", n=20)
log.info(f"Frames: {len(frames)}")

frame_data = []
for i, (mp, ip) in enumerate(frames):
    r = load_frame(mp, ip)
    if r is None: continue
    c, labels, img = r
    patches = np.stack([extract_patch(img, cy, cx, 64) for cy,cx in c])
    pn = (patches - patches.min((1,2), keepdims=True)) / (patches.max((1,2), keepdims=True)-patches.min((1,2), keepdims=True)+1e-8)
    t = torch.from_numpy(pn).float().unsqueeze(1)  # (N, 1, 64, 64)
    t_224 = F.interpolate(t, size=(224,224), mode="bilinear", align_corners=False)
    frame_data.append({"tensor_224": t_224, "labels": labels})
    log.info(f"  Frame {i+1}: {len(c)} cells prepped")

# ─── DINOv2 extraction ───────────────────────────────────────────────────────
log.info("Extracting DINOv2 features...")
dino_ = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
dino_feats = []
for fd in frame_data:
    t = fd["tensor_224"].expand(-1,3,-1,-1).to(device)  # 3 ch
    mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
    with torch.no_grad():
        out = dino_((t - mean) / std)
    dino_feats.append(out.cpu())
dino_.cpu(); del dino_; gc.collect(); torch.cuda.empty_cache()
log.info(f"DINOv2 done. {sum(len(x) for x in dino_feats)} cells total")

# ─── Cell-DINO extraction ────────────────────────────────────────────────────
log.info(f"Loading Cell-DINO vitl14 HPA from {CKPT}...")
cdino_ = torch.hub.load(REPO, 'cell_dino_hpa_vitl14', source='local', pretrained_url=f'file://{CKPT}').to(device).eval()
nparams = sum(p.numel() for p in cdino_.parameters())
log.info(f"Loaded: {nparams/1e6:.0f}M params")

cdino_feats = []
BATCH = 6  # small batches to avoid OOM on 4GB
for fi, fd in enumerate(frame_data):
    t = fd["tensor_224"].expand(-1,4,-1,-1)  # 4 ch HPA
    outs = []
    for i in range(0, len(t), BATCH):
        batch = t[i:i+BATCH].to(device)
        with torch.no_grad():
            out = cdino_(batch)
        outs.append(out.cpu())
    cdino_feats.append(torch.cat(outs))
    if (fi+1) % 10 == 0: log.info(f"  Cell-DINO: {fi+1}/{len(frame_data)} frames done")
cdino_.cpu(); del cdino_; gc.collect(); torch.cuda.empty_cache()
log.info("Cell-DINO done.")

# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_gap(f):
    fn = f / (np.linalg.norm(f, axis=-1, keepdims=True)+1e-8)
    sim = fn @ fn.T; N = f.shape[0]
    return sim[~np.eye(N,dtype=bool)].mean()

def recall_at_1(f, lbl):
    fn = f / (np.linalg.norm(f, axis=-1, keepdims=True)+1e-8)
    sim = fn @ fn.T; np.fill_diagonal(sim, -np.inf)
    return float((lbl[np.argmax(sim,axis=1)] == lbl).mean())

def eff_rank(f, t=0.95):
    c = f - f.mean(0,keepdims=1)
    try:
        _, S, _ = np.linalg.svd(c, full_matrices=False)
        return int(np.searchsorted(np.cumsum(S**2)/np.sum(S**2), t)+1)
    except: return f.shape[-1]

d_all = np.concatenate([x.numpy() for x in dino_feats])
c_all = np.concatenate([x.numpy() for x in cdino_feats])
l_all = np.concatenate([fd["labels"] for fd in frame_data])

print("\n" + "="*60)
print("CELL-DINO HPA vitl14 vs DINOv2 vits14: BACTERIA FEATURES")
print("="*60)
for name, feats in [("DINOv2 vits14 (ImageNet)", d_all), ("Cell-DINO vitl14 (HPA)", c_all)]:
    gap = compute_gap(feats)
    r1 = recall_at_1(feats, l_all)
    er = eff_rank(feats)
    print(f"\n  {name}:")
    print(f"    Inter-cell cos sim: {gap:.4f}  (lower = better)")
    print(f"    Recall@1:           {r1:.4f}")
    print(f"    Effective rank:     {er}")
    print(f"    Shape:              {feats.shape}")

delta = compute_gap(d_all) - compute_gap(c_all)
print(f"\n  Gap delta (DINOv2 - CellDINO): {delta:+.4f}")
if delta > 0.03: print("  ✓ Cell-DINO is MORE discriminative")
elif delta < -0.03: print("  ✗ DINOv2 is MORE discriminative")
else: print("  ~ Roughly equivalent")
print("="*60)
