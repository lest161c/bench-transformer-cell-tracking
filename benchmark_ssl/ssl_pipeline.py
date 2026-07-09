"""SSL contrastive-learning pipeline.

Loads cell detection data (masks + images), extracts WRFeatures per frame,
applies independent geometric/feature distortions to create two augmented
views. The encoder must learn to produce consistent embeddings for the same
cell across both views (positive pairs) while pushing apart different cells
(negative pairs).

Contrastive learning eliminates the need for tracking annotations.
Only segmentations (or detections) are required.
"""

import logging
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from edt import edt
from skimage.measure import regionprops_table, regionprops as sk_regionprops
from tifffile import imread
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

_PROPERTIES = {
    "regionprops": (
        "area", "intensity_mean", "intensity_max", "intensity_min", "inertia_tensor",
    ),
    "regionprops2": (
        "equivalent_diameter_area", "intensity_mean", "inertia_tensor", "border_dist",
    ),
}


def _border_dist_fast(mask, cutoff=5):
    cutoff = int(cutoff)
    border = np.ones(mask.shape, dtype=np.float32)
    ndim = mask.ndim
    for axis, size in enumerate(mask.shape):
        if axis < ndim - 2:
            continue
        band_vals = np.arange(cutoff, dtype=np.float32) / cutoff
        band_vals = band_vals[:size]
        low_slices = [slice(None)] * ndim
        low_slices[axis] = slice(0, cutoff)
        border_low = border[tuple(low_slices)]
        border_low_vals = np.minimum(border_low, band_vals[(...,) + (None,) * (ndim - axis - 1)])
        border[tuple(low_slices)] = border_low_vals
        high_slices = [slice(None)] * ndim
        high_slices[axis] = slice(max(0, size - cutoff), size)
        band_vals_rev = band_vals[::-1]
        border_high = border[tuple(high_slices)]
        border_high_vals = np.minimum(border_high, band_vals_rev[(...,) + (None,) * (ndim - axis - 1)])
        border[tuple(high_slices)] = border_high_vals
    dist = 1 - border
    return tuple(r.intensity_max for r in sk_regionprops(mask, intensity_image=dist))


def features_from_frame(mask, img, properties="regionprops2"):
    """Extract regionprops features from single frame mask+img."""
    ndim = mask.ndim
    props = _PROPERTIES[properties]
    use_border = "border_dist" in props
    if use_border:
        props = tuple(p for p in props if p != "border_dist")

    df_props = ("label", "centroid", *props)
    df = pd.DataFrame(regionprops_table(mask, intensity_image=img, properties=df_props))

    if use_border:
        df["border_dist"] = _border_dist_fast(mask)

    if len(df) == 0:
        return None

    coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    full_props = _PROPERTIES[properties]
    features = OrderedDict()
    for p in full_props:
        cols = [c for c in df.columns if c.startswith(p)]
        if cols:
            features[p] = np.stack([df[c].values.astype(np.float32) for c in cols], axis=-1)

    return coords, labels, features


def load_experiment_frames(exp_dir, conditions=None):
    """Scan directory for CTC-format experiments, return list of (cond, exp_name, frame_idx, mask_path, img_path)."""
    frames = []
    data_root = Path(exp_dir)
    if conditions is None:
        conditions = sorted(d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    for cond in conditions:
        cond_path = data_root / cond
        if not cond_path.is_dir():
            continue
        for exp_path in sorted(cond_path.iterdir()):
            if not exp_path.is_dir():
                continue
            tra_dir = exp_path / "TRA"
            img_dir = exp_path / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            masks = sorted(tra_dir.glob("man_track*.tif"))
            for m_path in masks:
                stem = m_path.stem.replace("man_track", "")
                try:
                    frame_idx = int(stem)
                except ValueError:
                    continue
                img_path = img_dir / f"t{frame_idx:06d}.tif"
                if img_path.exists():
                    frames.append((cond, exp_path.name, frame_idx, str(m_path), str(img_path)))
    return frames


class SSLDataset(Dataset):
    """Dataset for contrastive SSL pretraining.

    Produces two independently augmented views from each frame.
    No tracking labels needed — only segmentations.

    Cells are sorted by label in both views so that matching cells
    occupy the same index positions. This enables straightforward
    positive-pair identification in the NT-Xent loss.
    """

    def __init__(self, frames, distortion_pipeline=None, ndim=2, features="regionprops2"):
        self.frames = frames
        self.distortion_pipeline = distortion_pipeline
        self.ndim = ndim
        self.features = features
        logger.info(f"SSLDataset: {len(frames)} frames")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        _, _, _, mask_path, img_path = self.frames[idx]

        mask = imread(mask_path)
        img = imread(img_path).astype(np.float32)
        p1, p998 = np.percentile(img, (1, 99.8))
        img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

        result = features_from_frame(mask, img, self.features)
        if result is None:
            return self._empty_item()

        coords_src, labels_src, feats_dict_src = result

        if self.distortion_pipeline is not None:
            c1, f1, l1, c2, f2, l2 = self.distortion_pipeline(
                coords_src, feats_dict_src, labels_src
            )
        else:
            c1, f1, l1 = coords_src.copy(), {k: v.copy() for k, v in feats_dict_src.items()}, labels_src.copy()
            c2, f2, l2 = coords_src.copy(), {k: v.copy() for k, v in feats_dict_src.items()}, labels_src.copy()

        feats1 = np.concatenate(list(f1.values()), axis=-1).astype(np.float32)
        feats2 = np.concatenate(list(f2.values()), axis=-1).astype(np.float32)

        # Sort both views by label so matching cells occupy same indices
        idx1 = np.argsort(l1)
        idx2 = np.argsort(l2)

        return {
            "coords1": torch.from_numpy(c1[idx1]).float(),
            "coords2": torch.from_numpy(c2[idx2]).float(),
            "features1": torch.from_numpy(feats1[idx1]).float(),
            "features2": torch.from_numpy(feats2[idx2]).float(),
            "labels1": torch.from_numpy(l1[idx1]).long(),
            "labels2": torch.from_numpy(l2[idx2]).long(),
            "n1": len(l1),
            "n2": len(l2),
        }

    def _empty_item(self):
        return {
            "coords1": torch.zeros(0, self.ndim),
            "coords2": torch.zeros(0, self.ndim),
            "features1": torch.zeros(0, 7),
            "features2": torch.zeros(0, 7),
            "labels1": torch.zeros(0, dtype=torch.long),
            "labels2": torch.zeros(0, dtype=torch.long),
            "n1": 0,
            "n2": 0,
        }


def collate_ssl(batch):
    """Collate batch with padding for two-view contrastive learning.

    Both views are padded to max(N1, N2) across the batch.
    Cells are sorted by label, so position i in view1 corresponds
    to the same cell (if present in both views) at position i in view2.

    Returns dict with keys: coords1, coords2, features1, features2,
    padding_mask1, padding_mask2, valid_pair (which positions have
    matching cells in both views), n1, n2.
    """
    batch = [b for b in batch if b["n1"] > 0 or b["n2"] > 0]
    if len(batch) == 0:
        return None

    max_n = max(max(b["n1"], b["n2"]) for b in batch)
    B = len(batch)

    ndim = 2
    fdim = 7
    for b_i in batch:
        if b_i["n1"] > 0:
            ndim = b_i["coords1"].shape[-1]
            fdim = b_i["features1"].shape[-1]
            break

    c1 = torch.zeros(B, max_n, ndim)
    c2 = torch.zeros(B, max_n, ndim)
    f1 = torch.zeros(B, max_n, fdim)
    f2 = torch.zeros(B, max_n, fdim)
    l1 = torch.full((B, max_n,), -1, dtype=torch.long)
    l2 = torch.full((B, max_n,), -1, dtype=torch.long)
    pm1 = torch.ones(B, max_n, dtype=torch.bool)
    pm2 = torch.ones(B, max_n, dtype=torch.bool)

    for i, b in enumerate(batch):
        n1, n2 = b["n1"], b["n2"]
        if n1 > 0:
            c1[i, :n1] = b["coords1"]
            f1[i, :n1] = b["features1"]
            l1[i, :n1] = b["labels1"]
            pm1[i, :n1] = False
        if n2 > 0:
            c2[i, :n2] = b["coords2"]
            f2[i, :n2] = b["features2"]
            l2[i, :n2] = b["labels2"]
            pm2[i, :n2] = False

    valid_pair = (l1 == l2) & ~pm1 & ~pm2

    return {
        "coords1": c1, "coords2": c2,
        "features1": f1, "features2": f2,
        "padding_mask1": pm1, "padding_mask2": pm2,
        "valid_pair": valid_pair,
        "n1": torch.tensor([b["n1"] for b in batch]),
        "n2": torch.tensor([b["n2"] for b in batch]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frames = load_experiment_frames("../data/vanvliet", conditions=["rpsM"])
    print(f"Found {len(frames)} frames in rpsM")

    from distortions import DistortionPipeline
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    dist = DistortionPipeline.from_config(cfg)

    ds = SSLDataset(frames[:20], distortion_pipeline=dist)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_ssl)
    for batch in loader:
        print(f"coords1: {batch['coords1'].shape}, coords2: {batch['coords2'].shape}")
        print(f"  valid pairs: {batch['valid_pair'].sum().item()}")
        break
