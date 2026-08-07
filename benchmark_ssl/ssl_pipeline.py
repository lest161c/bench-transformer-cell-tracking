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
    """Compute a fast border-distance estimate per labeled region.

    Builds a distance-from-border image as 1 minus a normalized band of ones
    that fades toward the image edges (applied only to the last two axes),
    then returns, for every region in mask, the maximum of that image inside
    the region.

    Args:
        mask: integer label image (ndim-dimensional).
        cutoff: width in pixels of the edge band used to estimate distance.

    Returns:
        Tuple of per-region border-distance values, ordered by region as
        returned by skimage.measure.regionprops.
    """
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
    return tuple(region.intensity_max for region in sk_regionprops(mask, intensity_image=dist))


def features_from_frame(mask, img, properties="regionprops2"):
    """Extract regionprops features from a single frame mask+img.

    Args:
        mask: integer label image.
        img: intensity image matching mask's spatial shape.
        properties: regionprops feature profile name ("regionprops" or
            "regionprops2"); regionprops2 additionally includes a computed
            border-distance feature.

    Returns:
        (coords, labels, features) where coords is (N, ndim), labels is (N,),
        and features is an OrderedDict of per-cell feature arrays, or None if
        the frame contains no cells.
    """
    ndim = mask.ndim
    property_names = _PROPERTIES[properties]
    use_border = "border_dist" in property_names
    if use_border:
        property_names = tuple(prop for prop in property_names if prop != "border_dist")

    df_props = ("label", "centroid", *property_names)
    df = pd.DataFrame(regionprops_table(mask, intensity_image=img, properties=df_props))

    if use_border:
        df["border_dist"] = _border_dist_fast(mask)

    if len(df) == 0:
        return None

    coords = df[[f"centroid-{i}" for i in range(ndim)]].values.astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    full_property_names = _PROPERTIES[properties]
    features = OrderedDict()
    for prop in full_property_names:
        cols = [col for col in df.columns if col.startswith(prop)]
        if cols:
            features[prop] = np.stack([df[col].values.astype(np.float32) for col in cols], axis=-1)

    return coords, labels, features


def load_experiment_frames(exp_dir, conditions=None):
    """Scan directory for CTC-format experiments, return list of (cond, exp_name, frame_idx, mask_path, img_path)."""
    frames = []
    data_root = Path(exp_dir)
    if conditions is None:
        conditions = sorted(
            dir_entry.name for dir_entry in data_root.iterdir()
            if dir_entry.is_dir() and not dir_entry.name.startswith(".")
        )
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
        """Build the SSL dataset from a list of frame metadata tuples.

        Args:
            frames: list of (cond, exp_name, frame_idx, mask_path, img_path)
                tuples as returned by load_experiment_frames.
            distortion_pipeline: optional DistortionPipeline that produces the
                two augmented views; if None, both views are copies of the
                original frame.
            ndim: spatial dimensionality of the frames.
            features: regionprops feature profile name.
        """
        self.frames = frames
        self.distortion_pipeline = distortion_pipeline
        self.ndim = ndim
        self.features = features
        logger.info(f"SSLDataset: {len(frames)} frames")

    def __len__(self):
        """Return the number of frames in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx):
        """Load one frame and produce two independently augmented views.

        Args:
            idx: frame index into self.frames.

        Returns:
            Dict with keys coords1/coords2, features1/features2,
            labels1/labels2 (both views sorted by cell label) and n1/n2 cell
            counts. Frames with no cells yield an empty item.
        """
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
            coords1, feats1, labels1, coords2, feats2, labels2 = self.distortion_pipeline(
                coords_src, feats_dict_src, labels_src
            )
        else:
            coords1, feats1, labels1 = coords_src.copy(), {key: value.copy() for key, value in feats_dict_src.items()}, labels_src.copy()
            coords2, feats2, labels2 = coords_src.copy(), {key: value.copy() for key, value in feats_dict_src.items()}, labels_src.copy()

        feats1 = np.concatenate(list(feats1.values()), axis=-1).astype(np.float32)
        feats2 = np.concatenate(list(feats2.values()), axis=-1).astype(np.float32)

        # Sort both views by label so matching cells occupy same indices
        idx1 = np.argsort(labels1)
        idx2 = np.argsort(labels2)

        return {
            "coords1": torch.from_numpy(coords1[idx1]).float(),
            "coords2": torch.from_numpy(coords2[idx2]).float(),
            "features1": torch.from_numpy(feats1[idx1]).float(),
            "features2": torch.from_numpy(feats2[idx2]).float(),
            "labels1": torch.from_numpy(labels1[idx1]).long(),
            "labels2": torch.from_numpy(labels2[idx2]).long(),
            "n1": len(labels1),
            "n2": len(labels2),
        }

    def _empty_item(self):
        """Return a dict of empty tensors for frames that contain no cells."""
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
    batch = [sample for sample in batch if sample["n1"] > 0 or sample["n2"] > 0]
    if len(batch) == 0:
        return None

    max_n = max(max(sample["n1"], sample["n2"]) for sample in batch)
    batch_size = len(batch)

    ndim = 2
    feature_dim = 7
    for sample in batch:
        if sample["n1"] > 0:
            ndim = sample["coords1"].shape[-1]
            feature_dim = sample["features1"].shape[-1]
            break

    coords1 = torch.zeros(batch_size, max_n, ndim)
    coords2 = torch.zeros(batch_size, max_n, ndim)
    features1 = torch.zeros(batch_size, max_n, feature_dim)
    features2 = torch.zeros(batch_size, max_n, feature_dim)
    labels1 = torch.full((batch_size, max_n,), -1, dtype=torch.long)
    labels2 = torch.full((batch_size, max_n,), -1, dtype=torch.long)
    padding_mask1 = torch.ones(batch_size, max_n, dtype=torch.bool)
    padding_mask2 = torch.ones(batch_size, max_n, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n1, n2 = sample["n1"], sample["n2"]
        if n1 > 0:
            coords1[i, :n1] = sample["coords1"]
            features1[i, :n1] = sample["features1"]
            labels1[i, :n1] = sample["labels1"]
            padding_mask1[i, :n1] = False
        if n2 > 0:
            coords2[i, :n2] = sample["coords2"]
            features2[i, :n2] = sample["features2"]
            labels2[i, :n2] = sample["labels2"]
            padding_mask2[i, :n2] = False

    valid_pair = (labels1 == labels2) & ~padding_mask1 & ~padding_mask2

    return {
        "coords1": coords1, "coords2": coords2,
        "features1": features1, "features2": features2,
        "padding_mask1": padding_mask1, "padding_mask2": padding_mask2,
        "valid_pair": valid_pair,
        "n1": torch.tensor([sample["n1"] for sample in batch]),
        "n2": torch.tensor([sample["n2"] for sample in batch]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    frames = load_experiment_frames("../data/vanvliet", conditions=["rpsM"])
    print(f"Found {len(frames)} frames in rpsM")

    from distortions import DistortionPipeline
    with open("config.yaml") as file_handle:
        cfg = yaml.safe_load(file_handle)
    dist = DistortionPipeline.from_config(cfg)

    ds = SSLDataset(frames[:20], distortion_pipeline=dist)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_ssl)
    for batch in loader:
        print(f"coords1: {batch['coords1'].shape}, coords2: {batch['coords2'].shape}")
        print(f"  valid pairs: {batch['valid_pair'].sum().item()}")
        break
