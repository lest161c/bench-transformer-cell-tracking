"""Edge pair datasets, standardization, and balanced sampling.

Provides the ``Dataset`` subclasses and helpers that convert per-frame
cell features into pair-level training samples for the edge probe,
plus the leakage-safe z-score standardization and the balanced
sampler that equalizes positive/negative pairs per batch.

All per-frame data is labeled by its role in the edge-probing
paradigm (see ``evaluation.py`` for the full convention): the
**teacher** frame (frame t) provides the anchor cells, and the
**student** frame (frame t+1) provides the cells to be matched.
Consequently, the dict keys, class attributes, and function
parameters use ``_teacher`` / ``_student`` suffixes.
"""

import copy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class EdgePairDataset(Dataset):
    """Dataset of individual cell-cell pairs.

    Each sample is (feat_teacher, feat_student, label) where
      feat_teacher = feature of cell i in frame t
      feat_student = feature of cell j in frame t+1
      label  = 1 if same-tracklet, 0 otherwise
    """

    def __init__(self, feat_teacher, feat_student, target):
        """Flatten an (n_cells_teacher, n_cells_student) target matrix into one sample per pair.

        Args:
            feat_teacher: (n_cells_teacher, D) features of cells in frame t.
            feat_student: (n_cells_student, D) features of cells in frame t+1.
            target: (n_cells_teacher, n_cells_student) binary edge matrix; each entry yields
                one training sample.
        """
        n_cells_teacher, n_cells_student = target.shape
        pairs_teacher, pairs_student, labels = [], [], []
        for i in range(n_cells_teacher):
            for j in range(n_cells_student):
                pairs_teacher.append(feat_teacher[i])
                pairs_student.append(feat_student[j])
                labels.append(int(target[i, j].item()))
        self.pairs_teacher = torch.stack(pairs_teacher) if pairs_teacher else torch.empty(0, feat_teacher.shape[-1])
        self.pairs_student = torch.stack(pairs_student) if pairs_student else torch.empty(0, feat_student.shape[-1])
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Return the number of cell-cell pair samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return the (feat_teacher, feat_student, label) sample at index idx."""
        return self.pairs_teacher[idx], self.pairs_student[idx], self.labels[idx]


class EdgePairDatasetPatches(Dataset):
    """Dataset of individual cell-cell pairs where features are patches.

    Each sample is (patch_teacher, patch_student, label) where
      patch_teacher = image patch of cell i in frame t
      patch_student = image patch of cell j in frame t+1
      label  = 1 if same-tracklet, 0 otherwise

    Unlike EdgePairDataset, the per-cell features are raw image patches
    (used for CNN end-to-end training).

    Args:
        patches_teacher: (n_cells_teacher, 1, H, W) patches of cells in frame t.
        patches_student: (n_cells_student, 1, H, W) patches of cells in frame t+1.
        target: (n_cells_teacher, n_cells_student) binary edge matrix.
    """

    def __init__(self, patches_teacher, patches_student, target):
        n_cells_teacher, n_cells_student = target.shape
        pairs_teacher, pairs_student, labels = [], [], []
        for i in range(n_cells_teacher):
            for j in range(n_cells_student):
                pairs_teacher.append(patches_teacher[i])
                pairs_student.append(patches_student[j])
                labels.append(int(target[i, j].item()))
        self.pairs_teacher = torch.stack(pairs_teacher) if pairs_teacher else torch.empty(0, *patches_teacher.shape[1:])
        self.pairs_student = torch.stack(pairs_student) if pairs_student else torch.empty(0, *patches_student.shape[1:])
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """Return the number of cell-cell pair samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return the (patch_teacher, patch_student, label) sample at index idx."""
        return self.pairs_teacher[idx], self.pairs_student[idx], self.labels[idx]


def _gather_labels(dataset):
    """Gather all labels from an EdgePairDataset or ConcatDataset of them."""
    if hasattr(dataset, 'labels'):
        return dataset.labels.numpy()
    elif isinstance(dataset, torch.utils.data.ConcatDataset):
        labels_list = []
        for ds in dataset.datasets:
            if hasattr(ds, 'labels'):
                labels_list.append(ds.labels.numpy())
        if labels_list:
            return np.concatenate(labels_list)
    raise TypeError(f"Cannot gather labels from {type(dataset)}")


def make_balanced_dataloader(dataset, batch_size, shuffle=True):
    """Create a DataLoader with WeightedRandomSampler for balanced batches."""
    labels = _gather_labels(dataset)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        # Fall back to regular loader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Weight: minority class gets higher weight
    weight_pos = 1.0 / n_pos
    weight_neg = 1.0 / n_neg
    sample_weights = np.where(labels == 1, weight_pos, weight_neg)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def fit_feature_standardizer(datasets):
    """Fit per-feature z-score statistics (mean, std) on training datasets.

    The HOCT paper standardizes all node features per dataset; since raw
    handcrafted features mix wildly different scales (centroids ~10² px vs
    intensities ~[0,1] vs border distances ~10⁰–10¹ px), unregularized
    probes would otherwise weight features by their numeric magnitude.

    Statistics are pooled over all node features from BOTH sides of every
    training pair (pairs_teacher and pairs_student) — the exact distribution
    the probe is trained on. Note the stats are computed on pair-expanded
    rows, so cells are implicitly weighted by their pair multiplicity; the
    probe sees the same weighting, so this is consistent.

    IMPORTANT: fit only on the training split of each fold, never on the
    full dataset — otherwise validation statistics leak into training.

    Args:
        datasets: list of EdgePairDataset (training split only).

    Returns:
        (mean, std): (D,) float32 tensors. std is clamped to >= 1e-6 so
        constant features (e.g. a degenerate inertia entry) stay finite and
        map to 0 after transformation.
    """
    feats = torch.cat(
        [ds.pairs_teacher for ds in datasets] + [ds.pairs_student for ds in datasets],
        dim=0,
    )
    mean = feats.mean(dim=0)
    std = feats.std(dim=0).clamp(min=1e-6)
    return mean, std


def apply_feature_standardizer(datasets, mean, std):
    """Apply a fitted standardizer to datasets, returning transformed copies.

    Applies z = (x - mean) / std to both sides of every pair. The fitted
    statistics must come from fit_feature_standardizer() on the training
    split; validation data is only transformed, never refit.

    Datasets without pair features (EdgePairDatasetPatches for cnn_e2e —
    learned features are exempt from standardization) pass through unchanged.

    Args:
        datasets: list of EdgePairDataset / EdgePairDatasetPatches.
        mean, std: (D,) tensors from fit_feature_standardizer().

    Returns:
        New list of shallow copies with standardized pair features; the
        input datasets are not modified.
    """
    out = []
    for dataset in datasets:
        if not hasattr(dataset, "pairs_teacher"):
            out.append(dataset)  # e.g. EdgePairDatasetPatches (cnn_e2e): skip
            continue
        dataset_copy = copy.copy(dataset)
        dataset_copy.pairs_teacher = (dataset.pairs_teacher - mean) / std
        dataset_copy.pairs_student = (dataset.pairs_student - mean) / std
        out.append(dataset_copy)
    return out


def flatten_to_pairs(edge_data):
    """Convert list of per-frame-pair dicts into a flat list of datasets, one per frame pair."""
    datasets = []
    for item in edge_data:
        if "feat_teacher" in item:
            dataset = EdgePairDataset(item["feat_teacher"], item["feat_student"], item["target"])
        else:
            # For cnn_e2e, patches are stored
            dataset = EdgePairDatasetPatches(item["patches_teacher"], item["patches_student"], item["target"])
        if len(dataset) > 0:
            datasets.append(dataset)
    return datasets


def concatenate_datasets(datasets):
    """Concatenate multiple EdgePairDatasets (or EdgePairDatasetPatches) into one."""
    if not datasets:
        return None
    # Use torch.utils.data.ConcatDataset
    return torch.utils.data.ConcatDataset(datasets)


def shuffle_edge_data_features(edge_data_list, seed=42):
    """
    Create a copy of edge_data with feature vectors shuffled independently
    per frame pair to destroy all structure.
    Returns a new list with same structure but shuffled features.
    """
    rng = np.random.RandomState(seed)
    shuffled = []
    for item in edge_data_list:
        item_copy = dict(item)
        if "feat_teacher" in item:
            n1 = len(item["feat_teacher"])
            n2 = len(item["feat_student"])
            idx_teacher = torch.from_numpy(rng.permutation(n1))
            idx_student = torch.from_numpy(rng.permutation(n2))
            item_copy["feat_teacher"] = item["feat_teacher"][idx_teacher].clone()
            item_copy["feat_student"] = item["feat_student"][idx_student].clone()
        elif "patches_teacher" in item:
            n1 = len(item["patches_teacher"])
            n2 = len(item["patches_student"])
            idx_teacher = torch.from_numpy(rng.permutation(n1))
            idx_student = torch.from_numpy(rng.permutation(n2))
            item_copy["patches_teacher"] = item["patches_teacher"][idx_teacher].clone()
            item_copy["patches_student"] = item["patches_student"][idx_student].clone()
        shuffled.append(item_copy)
    return shuffled
