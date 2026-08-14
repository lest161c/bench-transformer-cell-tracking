"""SSL contrastive learning dataset and collation."""

import logging
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from tifffile import imread
from scipy.ndimage import map_coordinates

from src.data.feature_extraction import features_from_frame

logger = logging.getLogger(__name__)


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
