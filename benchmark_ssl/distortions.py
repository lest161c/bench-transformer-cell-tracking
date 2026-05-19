"""Distortion families for SSL pretext task.

Each distortion takes a frame's WRFeatures (coords + features + labels)
and returns a synthetic "next frame" with identity associations preserved.

Key principle: distortions must be diverse enough to prevent shortcut learning.
If the model can predict identity just from warp geometry, SSL fails.
"""

import numpy as np
from collections import OrderedDict
from scipy.ndimage import map_coordinates, gaussian_filter


def _transform_affine_feature(k, v, M):
    """Transform WRFeatures under affine matrix M (ndim×ndim)."""
    ndim = M.shape[-1]
    if k == "area":
        return np.linalg.det(M) * v
    elif k == "equivalent_diameter_area":
        return np.linalg.det(M) ** (1 / ndim) * v
    elif k == "inertia_tensor":
        v = v.reshape(-1, ndim, ndim)
        v = np.einsum("ijk,mk->ijm", v, M)
        v = np.einsum("ij,kjm->kim", M, v)
        return v.reshape(-1, ndim * ndim)
    elif k in ("intensity_mean", "intensity_max", "intensity_min", "border_dist"):
        return v
    else:
        return v  # pass through unknown feats


class AffineDistortion:
    """Global affine warp: rotation, scale, shear."""

    def __init__(self, degrees=15, scale=(0.85, 1.15), shear=(0.1, 0.1), rng=None):
        self.degrees = degrees
        self.scale = scale
        self.shear = shear
        self.rng = rng if rng is not None else np.random.RandomState()

    def _make_matrix(self, ndim):
        theta = self.rng.uniform(-self.degrees, self.degrees) / 180 * np.pi
        s = self.rng.uniform(*self.scale, 3)
        shy = self.rng.uniform(-self.shear[0], self.shear[0])
        shx = self.rng.uniform(-self.shear[1], self.shear[1])

        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ])
        S = np.diag(s)
        Sh = np.array([[1, 0, 0], [0, 1 + shx * shy, shy], [0, shx, 1]])
        M = R @ S @ Sh
        return M[-ndim:, -ndim:]

    def __call__(self, coords, features, labels):
        ndim = coords.shape[-1]
        M = self._make_matrix(ndim)
        coords_t = coords @ M.T
        feats_t = OrderedDict(
            (k, _transform_affine_feature(k, v, M))
            for k, v in features.items() if k != "pretrained_feats"
        )
        return coords_t, feats_t


class ElasticDistortion:
    """Elastic deformation via interpolated random displacement field."""

    def __init__(self, alpha=(10, 50), sigma=(5, 15), rng=None):
        self.alpha_range = alpha
        self.sigma_range = sigma
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        ndim = coords.shape[-1]
        alpha = self.rng.uniform(*self.alpha_range)
        sigma = self.rng.uniform(*self.sigma_range)

        # Build coarse displacement grid
        grid_size = 16
        grid_axes = [np.linspace(0, 1, grid_size) for _ in range(ndim)]
        grid_pts = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, ndim)
        rand_disp = self.rng.randn(grid_size ** ndim, ndim).astype(np.float32)

        # Smooth per-coordinate with gaussian_filter
        for d_i in range(ndim):
            field = rand_disp[:, d_i].reshape(*([grid_size] * ndim))
            field = gaussian_filter(field, sigma, mode="nearest")
            rand_disp[:, d_i] = field.ravel()
        rand_disp *= alpha

        # Normalize coords to [0, 1] grid
        coords_norm = coords.copy().astype(np.float32)
        cmin = coords_norm.min(axis=0)
        cmax = coords_norm.max(axis=0)
        cmax = np.where(cmax == cmin, cmin + 1, cmax)
        coords_norm = (coords_norm - cmin) / (cmax - cmin)

        # Interpolate: nearest neighbor for speed
        from scipy.spatial import cKDTree
        tree = cKDTree(grid_pts)
        _, idx = tree.query(coords_norm)
        cell_disp = rand_disp[idx]

        coords_t = coords + cell_disp.astype(np.float32)

        # Feature changes: area approximately preserved with small noise
        feats_t = OrderedDict()
        for k, v in features.items():
            if k == "pretrained_feats":
                continue
            elif k in ("area", "equivalent_diameter_area"):
                noise = self.rng.uniform(0.95, 1.05, size=v.shape).astype(np.float32)
                feats_t[k] = v * noise
            else:
                feats_t[k] = v.copy()

        return coords_t, feats_t


class JitterDistortion:
    """Per-cell independent jitter — simulates independent cell motion.

    This is the most realistic distortion for tracking: each cell moves
    independently, like real biological motion. The model MUST learn
    cell identity features (area, shape, intensity) to solve this.
    """

    def __init__(self, std=(2, 8), p_cell_jitter=0.8, rng=None):
        self.std_range = std
        self.p_cell_jitter = p_cell_jitter
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        ndim = coords.shape[-1]
        n = len(labels)
        std = self.rng.uniform(*self.std_range)

        # Each cell gets independent displacement
        jitter = self.rng.randn(n, ndim).astype(np.float32) * std

        # Some cells don't move (to prevent pure-distortion shortcut)
        mask = self.rng.rand(n) < self.p_cell_jitter
        jitter[~mask] = 0

        coords_t = coords + jitter

        # Preserve all features exactly (identity of cell unchanged)
        feats_t = OrderedDict(
            (k, v.copy()) for k, v in features.items() if k != "pretrained_feats"
        )

        return coords_t, feats_t


class DropoutDistortion:
    """Simulate segmentation failures — drop random subset of cells.

    Dropped cells have NO correspondence in the synthetic next frame.
    Teaches the model to handle false negatives / cell death.
    """

    def __init__(self, p_drop=(0.05, 0.2), rng=None):
        self.p_drop_range = p_drop
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        n = len(labels)
        p_drop = self.rng.uniform(*self.p_drop_range)
        keep = self.rng.rand(n) > p_drop

        # Return only surviving cells
        coords_t = coords[keep]
        feats_t = OrderedDict(
            (k, v[keep]) for k, v in features.items() if k != "pretrained_feats"
        )
        return coords_t, feats_t


class PhotometricDistortion:
    """Intensity/value shifts (only affects intensity features, not geometry).

    Useful only if image-based features are used. For shallow features,
    this adjusts intensity_mean/max/min values.
    """

    def __init__(self, scale=(0.5, 2.0), shift=(-0.1, 0.1), rng=None):
        self.scale_range = scale
        self.shift_range = shift
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        scale = self.rng.uniform(*self.scale_range)
        shift = self.rng.uniform(*self.shift_range)

        feats_t = OrderedDict()
        for k, v in features.items():
            if k == "pretrained_feats":
                continue
            if "intensity" in k:
                feats_t[k] = v * scale + shift
            else:
                feats_t[k] = v.copy()

        return coords.copy(), feats_t


class DistortionPipeline:
    """Apply a sequence of distortions to produce synthetic frame pair."""

    def __init__(self, distortions, seed=None):
        self.rng = np.random.RandomState(seed)
        self.distortions = distortions

    @classmethod
    def from_config(cls, config):
        rng = np.random.RandomState(config.get("seed", 42))
        distortion_names = config.get("distortions", ["jitter"])
        affine_cfg = config.get("affine", {})
        elastic_cfg = config.get("elastic", {})
        jitter_cfg = config.get("jitter", {})
        dropout_cfg = config.get("dropout", {})
        photometric_cfg = config.get("photometric", {})

        name_to_cls = {
            "affine": lambda: AffineDistortion(**affine_cfg, rng=rng),
            "elastic": lambda: ElasticDistortion(**elastic_cfg, rng=rng),
            "jitter": lambda: JitterDistortion(**jitter_cfg, rng=rng),
            "dropout": lambda: DropoutDistortion(**dropout_cfg, rng=rng),
            "photometric": lambda: PhotometricDistortion(**photometric_cfg, rng=rng),
        }

        distortions = []
        for name in distortion_names:
            if name not in name_to_cls:
                raise ValueError(f"Unknown distortion: {name}")
            distortions.append(name_to_cls[name]())

        return cls(distortions, seed=config.get("seed"))

    def __call__(self, coords, features, labels):
        """Apply distortions sequentially. Returns (src_coords, src_feats, tgt_coords, tgt_feats, tgt_labels).

        src = original frame (reference)
        tgt = distorted frame (synthetic "next frame")

        Identity association: i → i for all cells that survive.
        Key for successful SSL: model must predict identity correspondence
        using cell identity features, not warp shortcuts.
        """
        coords_src = coords.copy()
        feats_src = {k: v.copy() for k, v in features.items()}

        # Apply each distortion in sequence
        coords_t = coords.copy()
        feats_t = {k: v.copy() for k, v in features.items()}
        labels_t = labels.copy()

        for distortion in self.distortions:
            coords_t, feats_t = distortion(coords_t, feats_t, labels_t)
            # Handle dropout: if cells removed, trim labels too
            if isinstance(distortion, DropoutDistortion):
                labels_t = labels_t[:len(coords_t)]

        return coords_src, feats_src, coords_t, feats_t, labels_t
