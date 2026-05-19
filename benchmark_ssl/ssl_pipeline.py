"""SSL pretext task pipeline.

Loads cell tracking data (masks + images), extracts WRFeatures per frame,
applies geometric distortions to create synthetic frame pairs with identity
association labels.

Output format compatible with Trackastra:
  coords:       (N, ndim) — spatial only (time added by Trackastra)
  features:     (N, feat_dim)
  assoc_matrix: (N_src, N_tgt) — binary, identity i→i

Design avoids shortcut learning via:
- Agent-centric coordinate normalization (like ASCENT)
- Distortion diversity prevents memorization
- Per-cell jitter hardest/best pretext
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
    """Dataset for SSL pretraining.

    Produces synthetic frame pairs with identity association labels.
    No real tracking labels needed — only segmentations.

    Args:
        frames: List of (cond, exp, frame_idx, mask_path, img_path)
        distortion_pipeline: DistortionPipeline instance
        ndim: 2 or 3
        features: "regionprops" or "regionprops2"
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
            return {
                "coords_src": torch.zeros(0, self.ndim, dtype=torch.float32),
                "coords_tgt": torch.zeros(0, self.ndim, dtype=torch.float32),
                "features_src": torch.zeros(0, 7, dtype=torch.float32),
                "features_tgt": torch.zeros(0, 7, dtype=torch.float32),
                "assoc_matrix": torch.zeros(0, 0, dtype=torch.float32),
                "n_src": 0, "n_tgt": 0,
            }

        coords_src, labels_src, feats_dict_src = result

        if self.distortion_pipeline is not None:
            _, _, coords_tgt, feats_dict_tgt, labels_tgt = self.distortion_pipeline(
                coords_src, feats_dict_src, labels_src
            )
        else:
            coords_tgt = coords_src.copy()
            feats_dict_tgt = {k: v.copy() for k, v in feats_dict_src.items()}
            labels_tgt = labels_src.copy()

        # Build identity association matrix
        n_src, n_tgt = len(labels_src), len(labels_tgt)
        assoc = np.zeros((n_src, n_tgt), dtype=np.float32)
        label_to_tgt = {int(lbl): j for j, lbl in enumerate(labels_tgt)}
        for i, lbl in enumerate(labels_src):
            j = label_to_tgt.get(int(lbl))
            if j is not None:
                assoc[i, j] = 1.0

        # Stack features
        feats_src = np.concatenate(list(feats_dict_src.values()), axis=-1).astype(np.float32)
        feats_tgt = np.concatenate(list(feats_dict_tgt.values()), axis=-1).astype(np.float32)

        return {
            "coords_src": torch.from_numpy(coords_src).float(),
            "coords_tgt": torch.from_numpy(coords_tgt).float(),
            "features_src": torch.from_numpy(feats_src).float(),
            "features_tgt": torch.from_numpy(feats_tgt).float(),
            "assoc_matrix": torch.from_numpy(assoc).float(),
            "n_src": n_src, "n_tgt": n_tgt,
        }


def collate_ssl(batch):
    """Collate batch with padding (matches trackastra convention)."""
    max_src = max(b["n_src"] for b in batch)
    max_tgt = max(b["n_tgt"] for b in batch)
    B = len(batch)

    ndim = 2
    fdim = 7
    for b in batch:
        if b["n_src"] > 0:
            ndim = b["coords_src"].shape[-1]
            fdim = b["features_src"].shape[-1]
            break

    coords_src = torch.zeros(B, max_src, ndim)
    coords_tgt = torch.zeros(B, max_tgt, ndim)
    feats_src = torch.zeros(B, max_src, fdim)
    feats_tgt = torch.zeros(B, max_tgt, fdim)
    assoc = torch.zeros(B, max_src, max_tgt)
    pad_src = torch.ones(B, max_src, dtype=torch.bool)
    pad_tgt = torch.ones(B, max_tgt, dtype=torch.bool)

    for i, b in enumerate(batch):
        ns, nt = b["n_src"], b["n_tgt"]
        if ns > 0:
            coords_src[i, :ns] = b["coords_src"]
            feats_src[i, :ns] = b["features_src"]
            pad_src[i, :ns] = False
        if nt > 0:
            coords_tgt[i, :nt] = b["coords_tgt"]
            feats_tgt[i, :nt] = b["features_tgt"]
            pad_tgt[i, :nt] = False
        if ns > 0 and nt > 0:
            assoc[i, :ns, :nt] = b["assoc_matrix"]

    return {
        "coords_src": coords_src, "coords_tgt": coords_tgt,
        "features_src": feats_src, "features_tgt": feats_tgt,
        "assoc_matrix": assoc,
        "padding_mask_src": pad_src, "padding_mask_tgt": pad_tgt,
        "n_src": torch.tensor([b["n_src"] for b in batch]),
        "n_tgt": torch.tensor([b["n_tgt"] for b in batch]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frames = load_experiment_frames("../data/vanvliet", conditions=["rpsM"])
    print(f"Found {len(frames)} frames in rpsM")

    from distortions import DistortionPipeline
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    dist = DistortionPipeline.from_config(cfg)

    ds = SSLDataset(frames[:20], distortion_pipeline=dist)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_ssl)
    for batch in loader:
        print(f"coords_src: {batch['coords_src'].shape}, assoc: {batch['assoc_matrix'].shape}")
        print(f"  positives: {batch['assoc_matrix'].sum().item()}")
        break
