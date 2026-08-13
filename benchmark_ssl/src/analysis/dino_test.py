"""Test: Are DINOv2 cell-patch embeddings separable enough for contrastive SSL?

Runs three separation tests on vanvliet cell patches:
  Test 1 (identity): same patch vs itself — upper bound on separation.
  Test 2 (jitter): original vs jittered-center patches.
  Test 3 (affine): original vs affine-warped patches.

Usage:
    uv run python -m src.analysis.dino_test
"""

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from skimage.measure import regionprops_table
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dino_test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

# Load DINOv2 once
logger.info("Loading DINOv2...")
# Pin the "main" branch so torch.hub does not probe the network for the
# default branch (fails offline / on dropped connections).
dino = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14').to(device).eval()

PATCH_SIZE = 64  # extraction size
DINO_SIZE = 224  # DINO input size


def extract_patches(img, centroids, patch_size=PATCH_SIZE):
    """Extract square patches around centroids from image. Always returns (N, 64, 64)."""
    h, w = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i, cx_i = int(round(cy)), int(round(cx))
        y1 = cy_i - half
        y2 = cy_i + half
        x1 = cx_i - half
        x2 = cx_i + half
        # Compute padding
        pad_top = max(0, -y1)
        pad_bot = max(0, y2 - h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)
        # Clamp
        y1_c, y2_c = max(0, y1), min(h, y2)
        x1_c, x2_c = max(0, x1), min(w, x2)
        crop = img[y1_c:y2_c, x1_c:x2_c]
        if crop.size == 0 and (pad_top > 0 or pad_bot > 0):
            patches.append(np.zeros((patch_size, patch_size), dtype=np.float32))
            continue
        if pad_top > 0 or pad_bot > 0 or pad_left > 0 or pad_right > 0:
            crop = np.pad(crop, ((pad_top, pad_bot), (pad_left, pad_right)), mode='reflect')
        patches.append(crop)
    return np.stack(patches, axis=0) if patches else np.zeros((0, patch_size, patch_size), dtype=np.float32)


def patches_to_dino(patches_np):
    """(N,64,64) -> normalize -> resize -> (N,384)."""
    if len(patches_np) == 0:
        return np.zeros((0, 384), dtype=np.float32)
    # Normalize per-patch to [0,1]
    p_min = patches_np.min(axis=(1,2), keepdims=True)
    p_max = patches_np.max(axis=(1,2), keepdims=True)
    patches_norm = (patches_np - p_min) / (p_max - p_min + 1e-8)
    # Resize to 224
    t = torch.from_numpy(patches_norm).float().unsqueeze(1).to(device)  # (N,1,64,64)
    t = torch.nn.functional.interpolate(t, size=(DINO_SIZE, DINO_SIZE), mode='bilinear', align_corners=False)
    t = t.expand(-1, 3, -1, -1)  # (N,3,224,224)
    # ImageNet normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    t = (t - mean) / std
    with torch.no_grad():
        emb = dino(t)  # (N,384)
    return emb.cpu().numpy()


def load_frame(mask_path, img_path):
    """Load a mask+image frame and return coords, labels, patches, and normalized image."""
    mask = imread(mask_path)
    img = imread(img_path).astype(np.float32)
    p1, p998 = np.percentile(img, (1, 99.8))
    img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)
    props = regionprops_table(mask, properties=('label', 'centroid'))
    if not props or len(props['label']) == 0:
        return None
    coords = np.stack([props['centroid-0'], props['centroid-1']], axis=-1).astype(np.float32)
    labels = props['label'].astype(np.int32)
    patches = extract_patches(img, coords)
    return coords, labels, patches, img


def scan_frames(data_root, conditions, max_frames=200):
    """Collect (mask_path, image_path) pairs across conditions, up to max_frames."""
    frames = []
    data_root = Path(data_root)
    for cond in conditions:
        for exp_path in sorted((data_root / cond).iterdir()):
            if not exp_path.is_dir():
                continue
            tra_dir = exp_path / "TRA"
            img_dir = exp_path / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            for mpath in sorted(tra_dir.glob("man_track*.tif")):
                stem = mpath.stem.replace("man_track", "")
                try: fidx = int(stem)
                except ValueError: continue
                ipath = img_dir / f"t{fidx:06d}.tif"
                if ipath.exists():
                    frames.append((str(mpath), str(ipath)))
                    if len(frames) >= max_frames:
                        return frames
    return frames


def compute_separation(embs_a, embs_b, labels_a, labels_b):
    """Compute intra/inter cosine similarity gap."""
    if len(embs_a) < 2 or len(embs_b) < 2:
        return None
    fa = embs_a / (np.linalg.norm(embs_a, axis=1, keepdims=True) + 1e-12)
    fb = embs_b / (np.linalg.norm(embs_b, axis=1, keepdims=True) + 1e-12)
    sim = fa @ fb.T
    intra, inter = [], []
    for i, lbl in enumerate(labels_a):
        matches = np.where(labels_b == lbl)[0]
        for j in range(len(labels_b)):
            if j in matches:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])
    return {
        'intra': float(np.mean(intra)) if intra else np.nan,
        'inter': float(np.mean(inter)) if inter else np.nan,
        'gap': float(np.mean(intra) - np.mean(inter)) if intra and inter else np.nan,
    }


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_no_distortion(frames, max_frames=100):
    """Compare DINO features of each cell vs itself (same patch = identity test)."""
    gaps = []
    n = 0
    for mpath, ipath in frames[:max_frames]:
        result = load_frame(mpath, ipath)
        if result is None:
            continue
        coords, labels, patches, img = result
        if len(labels) < 2:
            continue
        emb = patches_to_dino(patches)
        s = compute_separation(emb, emb, labels, labels)
        if s is not None:
            gaps.append(s['gap'])
            n += 1
    return {'gap_mean': float(np.mean(gaps)), 'gap_std': float(np.std(gaps)), 'n': n, 'samples': gaps}


def test_jitter_distortion(frames, max_frames=100, jitter_std=4.0):
    """Same cell, original vs jittered coord -> different patch -> DINO features still match?"""
    gaps = []
    n = 0
    for mpath, ipath in frames[:max_frames]:
        result = load_frame(mpath, ipath)
        if result is None:
            continue
        coords, labels, patches, img = result
        if len(labels) < 2:
            continue
        # Original DINO emb
        emb_orig = patches_to_dino(patches)
        # Jittered patches: extract at noisy coords
        jitter = np.random.randn(*coords.shape).astype(np.float32) * jitter_std
        coords_j = coords + jitter
        patches_j = extract_patches(img, coords_j, PATCH_SIZE)
        emb_jit = patches_to_dino(patches_j)
        s = compute_separation(emb_orig, emb_jit, labels, labels)
        if s is not None:
            gaps.append(s['gap'])
            n += 1
    return {'gap_mean': float(np.mean(gaps)), 'gap_std': float(np.std(gaps)), 'n': n}


def test_affine_distortion(frames, max_frames=100):
    """Apply affine warp to coords + image, extract patches at warped coords."""
    gaps = []
    n = 0
    for mpath, ipath in frames[:max_frames]:
        result = load_frame(mpath, ipath)
        if result is None:
            continue
        coords, labels, patches_orig, img = result
        if len(labels) < 2:
            continue
        emb_orig = patches_to_dino(patches_orig)
        # Random affine
        theta = np.random.uniform(-15, 15) / 180 * np.pi
        s = np.random.uniform(0.85, 1.15, 2)
        cos, sin = np.cos(theta), np.sin(theta)
        M = np.array([[s[0]*cos, -s[0]*sin],
                      [s[1]*sin,  s[1]*cos]])
        # Warp coords
        coords_w = coords @ M.T
        # Extract patches at warped coords from ORIGINAL image
        patches_w = extract_patches(img, coords_w, PATCH_SIZE)
        emb_warp = patches_to_dino(patches_w)
        s = compute_separation(emb_orig, emb_warp, labels, labels)
        if s is not None:
            gaps.append(s['gap'])
            n += 1
    return {'gap_mean': float(np.mean(gaps)), 'gap_std': float(np.std(gaps)), 'n': n}


def print_header(label):
    """Print a section header line."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


def print_result(name, result, threshold=0.3):
    """Print one test result with a pass/border/fail verdict."""
    gap = result['gap_mean']
    status = "✓ PASS" if gap > threshold else ("~ BORDER" if gap > 0.15 else "✗ FAIL")
    print(f"  {name:<30} gap={gap:.4f}±{result['gap_std']:.4f}  n={result['n']}  {status}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(data_root="../data/vanvliet", conditions=("rpsM", "recA", "pheA", "metA", "cib", "trpL"), max_frames=300):
    """Run the three DINO separation tests and print the summary verdict.

    Args:
        data_root: path to the vanvliet data root.
        conditions: condition names to scan for frames.
        max_frames: maximum number of frames to scan.

    Returns:
        dict with the three per-test result dicts under keys r1, r2, r3.
    """
    frames = scan_frames(data_root, list(conditions), max_frames=max_frames)
    logger.info(f"Scanned {len(frames)} frames")

    print_header("DINO FEATURE SEPARATION ANALYSIS (vanvliet)")
    print(f"\n  DINO model: dinov2_vits14 (384D embeddings)")
    print(f"  Patch size: {PATCH_SIZE}×{PATCH_SIZE} → resize to {DINO_SIZE}×{DINO_SIZE}")
    print(f"  Frames sampled: {len(frames)}")

    # --- Test 1: No distortion (identity test) ---
    logger.info("Test 1: No distortion — same patches, same coords")
    r1 = test_no_distortion(frames, max_frames=100)
    print_header("TEST 1: Identity (no distortion)")
    print_result("Same patch vs same patch", r1)
    print("  (intra always 1.0; gap = 1.0 - inter. Tells cell similarity)")

    # --- Test 2: Jitter distortion ---
    logger.info("Test 2: Jitter — patches at jittered coords")
    r2 = test_jitter_distortion(frames, max_frames=100, jitter_std=4.0)
    print_header("TEST 2: Jitter (std=4px)")
    print_result("Original vs jittered patches", r2)

    # --- Test 3: Affine distortion ---
    logger.info("Test 3: Affine — patches at warped coords")
    r3 = test_affine_distortion(frames, max_frames=100)
    print_header("TEST 3: Affine (rot=±15°, scale=0.85-1.15)")
    print_result("Original vs affine-warped patches", r3)

    # --- Summary ---
    print_header("SUMMARY")
    print(f"  Regionprops gap (from prior analysis):      0.047 (FAIL)")
    print(f"  DINO no-distortion gap:                     {r1['gap_mean']:.4f}")
    print(f"  DINO jitter gap:                            {r2['gap_mean']:.4f}")
    print(f"  DINO affine gap:                            {r3['gap_mean']:.4f}")
    print()

    if r1['gap_mean'] > 0.3:
        print("  → DINO embeddings separate cells even without distortion.")
        print("  → Contrastive SSL should work with DINO patch features.")
    elif r1['gap_mean'] > 0.15:
        print("  → DINO embeddings show moderate separation.")
        print("  → Contrastive may work with careful tuning.")
    else:
        print("  → DINO does not help. Cells look too similar even at pixel level.")

    return {"r1": r1, "r2": r2, "r3": r3}


if __name__ == "__main__":
    main()
