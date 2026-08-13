"""Frame loading, consecutive-pair scanning, and patch extraction for edge probing.

Shared by all feature types:

- ``load_frame`` — read a mask+image pair into the
  ``(coords, labels, img, mask)`` representation used by every extractor.
- ``load_tracklets`` — parse ``man_track.txt`` into a
  ``label -> {t1, t2, parent}`` lineage annotation.
- ``scan_consecutive_pairs`` — locate all (t, t+1) frame pairs across the
  condition/experiment directory tree.
- ``extract_patches`` — crop square patches centered on cell centroids,
  used by the CNN/DINOv2 feature extractors.

This module is pure data plumbing: it has no knowledge of probes, features,
or training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tifffile import imread

PATCH_SIZE = 64


def load_frame(mask_path, img_path):
    """Load mask + image, return (coords, labels, img, mask) or None."""
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
    return coords, props["label"].astype(np.int32), img, mask


def load_tracklets(man_track_path):
    """Parse man_track.txt into dict: label -> {t1, t2, parent}."""
    df = pd.read_csv(
        man_track_path, delimiter=' ', header=None,
        names=['label', 't1', 't2', 'parent']
    )
    tracklets = {}
    for _, row in df.iterrows():
        tracklets[int(row['label'])] = {
            't1': int(row['t1']),
            't2': int(row['t2']),
            'parent': int(row['parent']),
        }
    return tracklets


def scan_consecutive_pairs(data_root, conditions, max_pairs):
    """Find consecutive frame pairs. Returns list of (mask_t, mask_n, img_t,
    img_n, man_track_path, condition, experiment)."""
    dr = Path(data_root)
    pairs = []
    for cond in conditions:
        for exp in sorted(dr.glob(f"{cond}/*")):
            if not exp.is_dir():
                continue
            tra_dir = exp / "TRA"
            img_dir = exp / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            man_track_txt = tra_dir / "man_track.txt"
            if not man_track_txt.exists():
                continue
            masks = sorted(tra_dir.glob("man_track*.tif"))
            for i in range(len(masks) - 1):
                stem1 = masks[i].stem.replace("man_track", "")
                stem2 = masks[i + 1].stem.replace("man_track", "")
                try:
                    f1, f2 = int(stem1), int(stem2)
                except ValueError:
                    continue
                if f2 == f1 + 1:
                    ip1 = img_dir / f"t{f1:06d}.tif"
                    ip2 = img_dir / f"t{f2:06d}.tif"
                    if ip1.exists() and ip2.exists():
                        pairs.append((
                            str(masks[i]), str(masks[i + 1]),
                            str(ip1), str(ip2),
                            str(man_track_txt),
                            cond, exp.name,
                        ))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def extract_patches(img, centroids, patch_size=PATCH_SIZE):
    """Extract square patches centered on centroids. Handles edge padding."""
    height, width = img.shape[-2:]
    half = patch_size // 2
    patches = []
    for cy, cx in centroids:
        cy_i = int(round(float(cy)))
        cx_i = int(round(float(cx)))
        cy_i = np.clip(cy_i, 0, height - 1)
        cx_i = np.clip(cx_i, 0, width - 1)
        y1 = cy_i - half
        x1 = cx_i - half
        y2 = cy_i + half
        x2 = cx_i + half
        pt = max(0, -y1)
        pb = max(0, y2 - height)
        pl = max(0, -x1)
        pr = max(0, x2 - width)
        y1c = max(0, y1)
        x1c = max(0, x1)
        y2c = min(height, y2)
        x2c = min(width, x2)
        crop = (
            img[y1c:y2c, x1c:x2c]
            if y2c > y1c and x2c > x1c
            else np.zeros((1, 1), dtype=np.float32)
        )
        if pt or pb or pl or pr:
            crop = np.pad(crop, ((pt, pb), (pl, pr)), mode="reflect")
        if crop.shape != (patch_size, patch_size):
            crop = np.pad(
                crop,
                tuple((0, max(0, pad_amount)) for pad_amount in [patch_size - dim_size for dim_size in crop.shape]),
                mode="reflect",
            )[:patch_size, :patch_size]
        patches.append(crop)
    if patches:
        return np.stack(patches).astype(np.float32)
    return np.zeros((0, patch_size, patch_size), dtype=np.float32)
