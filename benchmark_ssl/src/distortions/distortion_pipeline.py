"""Distortion families and pipeline for SSL contrastive learning."""

import numpy as np
from collections import OrderedDict
from scipy.ndimage import map_coordinates, gaussian_filter


def _transform_affine_feature(key, value, affine_matrix):
    """Transform WRFeatures under affine matrix affine_matrix (ndim×ndim).

    Args:
        key: feature name (e.g. "area", "inertia_tensor").
        value: feature value array of shape (N,) or (N, ndim*ndim) depending
            on the feature kind.
        affine_matrix: ndim×ndim affine transformation matrix.

    Returns:
        The feature values transformed to match the affine warp. Features
        whose values are invariant or unmodelled by the affine transform are
        returned unchanged.
    """
    ndim = affine_matrix.shape[-1]
    if key == "area":
        return np.linalg.det(affine_matrix) * value
    elif key == "equivalent_diameter_area":
        return np.linalg.det(affine_matrix) ** (1 / ndim) * value
    elif key == "inertia_tensor":
        value = value.reshape(-1, ndim, ndim)
        value = np.einsum("ijk,mk->ijm", value, affine_matrix)
        value = np.einsum("ij,kjm->kim", affine_matrix, value)
        return value.reshape(-1, ndim * ndim)
    elif key in ("intensity_mean", "intensity_max", "intensity_min", "border_dist"):
        return value
    else:
        return value


class AffineDistortion:
    """Global affine warp: rotation, scale, shear."""

    def __init__(self, degrees=15, scale=(0.85, 1.15), shear=(0.1, 0.1), rng=None):
        """Store affine distortion hyperparameters and the RNG.

        Args:
            degrees: maximum rotation magnitude in degrees (sampled uniformly).
            scale: (min, max) range for the three uniform scale factors.
            shear: (shear_y, shear_x) maximum shear magnitudes along each axis.
            rng: optional numpy RandomState; a new one is created if None.
        """
        self.degrees = degrees
        self.scale = scale
        self.shear = shear
        self.rng = rng if rng is not None else np.random.RandomState()

    def _make_matrix(self, ndim):
        """Sample and compose the affine matrix for the given dimensionality.

        Args:
            ndim: number of spatial dimensions (2 or 3).

        Returns:
            ndim×ndim affine matrix combining random rotation, scale, and shear.
        """
        theta = self.rng.uniform(-self.degrees, self.degrees) / 180 * np.pi
        scale_factors = self.rng.uniform(*self.scale, 3)
        shear_y = self.rng.uniform(-self.shear[0], self.shear[0])
        shear_x = self.rng.uniform(-self.shear[1], self.shear[1])
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ])
        scale_matrix = np.diag(scale_factors)
        shear_matrix = np.array([[1, 0, 0], [0, 1 + shear_x * shear_y, shear_y], [0, shear_x, 1]])
        affine_matrix = rotation_matrix @ scale_matrix @ shear_matrix
        return affine_matrix[-ndim:, -ndim:]

    def __call__(self, coords, features, labels):
        """Apply a randomly sampled affine warp to coordinates and features.

        Args:
            coords: (N, ndim) — cell coordinates.
            features: OrderedDict of (N, *) — regionprops features.
            labels: (N,) — cell identity labels (passed through unchanged).

        Returns:
            (coords_t, feats_t): warped coordinates and feature values.
        """
        ndim = coords.shape[-1]
        affine_matrix = self._make_matrix(ndim)
        coords_t = coords @ affine_matrix.T
        feats_t = OrderedDict(
            (key, _transform_affine_feature(key, value, affine_matrix))
            for key, value in features.items() if key != "pretrained_feats"
        )
        return coords_t, feats_t


class ElasticDistortion:
    """Elastic deformation via interpolated random displacement field."""

    def __init__(self, alpha=(10, 50), sigma=(5, 15), rng=None):
        """Store elastic deformation hyperparameters and the RNG.

        Args:
            alpha: (min, max) displacement magnitude range.
            sigma: (min, max) smoothing kernel width range.
            rng: optional numpy RandomState; a new one is created if None.
        """
        self.alpha_range = alpha
        self.sigma_range = sigma
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        """Apply a random elastic deformation to coordinates.

        Displacements come from a smoothed random field evaluated at the
        nearest grid point of each cell. Scale-like features receive small
        multiplicative noise; all other features are copied unchanged.

        Args:
            coords: (N, ndim) — cell coordinates.
            features: OrderedDict of (N, *) — regionprops features.
            labels: (N,) — cell identity labels (unused here).

        Returns:
            (coords_t, feats_t): deformed coordinates and feature values.
        """
        ndim = coords.shape[-1]
        alpha = self.rng.uniform(*self.alpha_range)
        sigma = self.rng.uniform(*self.sigma_range)
        grid_size = 16
        grid_axes = [np.linspace(0, 1, grid_size) for _ in range(ndim)]
        grid_pts = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, ndim)
        rand_disp = self.rng.randn(grid_size ** ndim, ndim).astype(np.float32)
        for d_i in range(ndim):
            field = rand_disp[:, d_i].reshape(*([grid_size] * ndim))
            field = gaussian_filter(field, sigma, mode="nearest")
            rand_disp[:, d_i] = field.ravel()
        rand_disp *= alpha
        coords_norm = coords.copy().astype(np.float32)
        cmin = coords_norm.min(axis=0)
        cmax = coords_norm.max(axis=0)
        cmax = np.where(cmax == cmin, cmin + 1, cmax)
        coords_norm = (coords_norm - cmin) / (cmax - cmin)
        from scipy.spatial import cKDTree
        tree = cKDTree(grid_pts)
        _, idx = tree.query(coords_norm)
        cell_disp = rand_disp[idx]
        coords_t = coords + cell_disp.astype(np.float32)
        feats_t = OrderedDict()
        for key, value in features.items():
            if key == "pretrained_feats":
                continue
            elif key in ("area", "equivalent_diameter_area"):
                noise = self.rng.uniform(0.95, 1.05, size=value.shape).astype(np.float32)
                feats_t[key] = value * noise
            else:
                feats_t[key] = value.copy()
        return coords_t, feats_t


class JitterDistortion:
    """Per-cell independent jitter — simulates independent cell motion.

    This is the single most effective augmentation (per ASCENT ablation).
    Each cell moves independently, forcing the model to learn identity
    features beyond coordinate matching.
    """

    def __init__(self, std=(2, 8), p_cell_jitter=0.8, rng=None):
        """Store jitter hyperparameters and the RNG.

        Args:
            std: (min, max) jitter standard deviation range in pixels.
            p_cell_jitter: probability that a given cell is jittered.
            rng: optional numpy RandomState; a new one is created if None.
        """
        self.std_range = std
        self.p_cell_jitter = p_cell_jitter
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        """Add independent Gaussian jitter to each cell's coordinates.

        Args:
            coords: (N, ndim) — cell coordinates.
            features: OrderedDict of (N, *) — regionprops features.
            labels: (N,) — cell identity labels (unused here).

        Returns:
            (coords_t, feats_t): jittered coordinates, features copied.
        """
        ndim = coords.shape[-1]
        n_cells = len(labels)
        std = self.rng.uniform(*self.std_range)
        jitter = self.rng.randn(n_cells, ndim).astype(np.float32) * std
        mask = self.rng.rand(n_cells) < self.p_cell_jitter
        jitter[~mask] = 0
        coords_t = coords + jitter
        feats_t = OrderedDict(
            (key, value.copy()) for key, value in features.items() if key != "pretrained_feats"
        )
        return coords_t, feats_t


class DropoutDistortion:
    """Simulate segmentation failures — drop random subset of cells.

    Dropped cells have NO counterpart in the other view.
    Teaches the model to handle false negatives / detection failures.
    """

    def __init__(self, p_drop=(0.05, 0.2), rng=None):
        """Store dropout hyperparameters and the RNG.

        Args:
            p_drop: (min, max) drop probability range.
            rng: optional numpy RandomState; a new one is created if None.
        """
        self.p_drop_range = p_drop
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        """Drop a random subset of cells from the view entirely.

        Args:
            coords: (N, ndim) — cell coordinates.
            features: OrderedDict of (N, *) — regionprops features.
            labels: (N,) — cell identity labels.

        Returns:
            (coords_t, feats_t): the surviving cells only, in original order.
        """
        n_cells = len(labels)
        p_drop = self.rng.uniform(*self.p_drop_range)
        keep = self.rng.rand(n_cells) > p_drop
        coords_t = coords[keep]
        feats_t = OrderedDict(
            (key, value[keep]) for key, value in features.items() if key != "pretrained_feats"
        )
        return coords_t, feats_t


class PhotometricDistortion:
    """Intensity/value shifts for intensity-based features."""

    def __init__(self, scale=(0.5, 2.0), shift=(-0.1, 0.1), rng=None):
        """Store photometric hyperparameters and the RNG.

        Args:
            scale: (min, max) multiplicative scale range.
            shift: (min, max) additive shift range.
            rng: optional numpy RandomState; a new one is created if None.
        """
        self.scale_range = scale
        self.shift_range = shift
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        """Apply a random scale+shift to all intensity-based features.

        Args:
            coords: (N, ndim) — cell coordinates (copied unchanged).
            features: OrderedDict of (N, *) — regionprops features.
            labels: (N,) — cell identity labels (unused here).

        Returns:
            (coords_t, feats_t): unchanged coordinates, intensity features
            transformed, all other features copied.
        """
        scale = self.rng.uniform(*self.scale_range)
        shift = self.rng.uniform(*self.shift_range)
        feats_t = OrderedDict()
        for key, value in features.items():
            if key == "pretrained_feats":
                continue
            if "intensity" in key:
                feats_t[key] = value * scale + shift
            else:
                feats_t[key] = value.copy()
        return coords.copy(), feats_t


class FeatureNoise:
    """Add Gaussian noise to all features to prevent shortcut memorization.

    Without feature noise, the encoder can memorize exact feature values
    (area, inertia_tensor, etc.) to match cells across views instead of
    learning robust identity representations.
    """

    def __init__(self, std=(0.02, 0.15), rng=None):
        """Store feature-noise hyperparameters and the RNG.

        Args:
            std: (min, max) base noise standard deviation range.
            rng: optional numpy RandomState; a new one is created if None.
        """
        self.std_range = std
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, coords, features, labels):
        """Add feature-scaled Gaussian noise to every feature.

        The noise is additionally scaled to ~10% of each feature's own
        standard deviation so that low-variance features are not destroyed.

        Args:
            coords: (N, ndim) — cell coordinates (copied unchanged).
            features: OrderedDict of (N, *) — regionprops features.
            labels: (N,) — cell identity labels (unused here).

        Returns:
            (coords_t, feats_t): unchanged coordinates, noisy feature values.
        """
        std = self.rng.uniform(*self.std_range)
        feats_t = OrderedDict()
        for key, value in features.items():
            if key == "pretrained_feats":
                continue
            noise = self.rng.randn(*value.shape).astype(np.float32) * std
            feat_std = np.std(value)
            if feat_std > 1e-6:
                noise = noise * (feat_std * 0.1)  # Scale noise to ~10% of feature std
            feats_t[key] = value + noise
        return coords.copy(), feats_t


class DistortionPipeline:
    """Apply sequence of distortions to generate two independent augmented views.

    Each call produces two views with independently sampled distortions.
    This is the core of the contrastive SSL pretext task: the encoder must
    learn to produce consistent embeddings for the same cell across two
    differently-distorted views of the same frame.
    """

    def __init__(self, distortions, seed=None):
        """Wrap an ordered list of distortions with a shared RNG.

        Args:
            distortions: list of distortion callables applied in order.
            seed: optional RNG seed for reproducible view generation.
        """
        self.rng = np.random.RandomState(seed)
        self.distortions = distortions

    @classmethod
    def from_config(cls, config):
        """Build a DistortionPipeline from a config dict.

        Args:
            config: dict with optional keys "seed", "distortions" (list of
                distortion names), and per-distortion config dicts ("affine",
                "elastic", "jitter", "dropout", "photometric",
                "feature_noise").

        Returns:
            A configured DistortionPipeline instance.

        Raises:
            ValueError: if a requested distortion name is unknown.
        """
        rng = np.random.RandomState(config.get("seed", 42))
        distortion_names = config.get("distortions", ["jitter"])
        affine_cfg = config.get("affine", {})
        elastic_cfg = config.get("elastic", {})
        jitter_cfg = config.get("jitter", {})
        dropout_cfg = config.get("dropout", {})
        photometric_cfg = config.get("photometric", {})
        feature_noise_cfg = config.get("feature_noise", {})

        name_to_cls = {
            "affine": lambda: AffineDistortion(**affine_cfg, rng=rng),
            "elastic": lambda: ElasticDistortion(**elastic_cfg, rng=rng),
            "jitter": lambda: JitterDistortion(**jitter_cfg, rng=rng),
            "dropout": lambda: DropoutDistortion(**dropout_cfg, rng=rng),
            "photometric": lambda: PhotometricDistortion(**photometric_cfg, rng=rng),
            "feature_noise": lambda: FeatureNoise(**feature_noise_cfg, rng=rng),
        }

        distortions = []
        for name in distortion_names:
            if name not in name_to_cls:
                raise ValueError(f"Unknown distortion: {name}")
            distortions.append(name_to_cls[name]())

        return cls(distortions, seed=config.get("seed"))

    def __call__(self, coords, features, labels):
        """Generate two independent augmented views.

        Args:
            coords:   (N, ndim) — cell coordinates
            features: OrderedDict of (N, *) — regionprops features
            labels:   (N,) — cell identity labels

        Returns:
            coords1, feats1, labels1: view 1 (independently distorted)
            coords2, feats2, labels2: view 2 (independently distorted)
        """

        def _apply_view(coords_src, features_src, labels_src):
            """Apply the full distortion sequence to one view.

            Args:
                coords_src: (N, ndim) — source cell coordinates.
                features_src: OrderedDict of (N, *) — source features.
                labels_src: (N,) — source cell identity labels.

            Returns:
                (view_coords, view_features, view_labels): the distorted view.
                DropoutDistortion truncates labels to the surviving cells.
            """
            view_coords = coords_src.copy()
            view_features = {key: value.copy() for key, value in features_src.items()}
            view_labels = labels_src.copy()
            for dist in self.distortions:
                view_coords, view_features = dist(view_coords, view_features, view_labels)
                if isinstance(dist, DropoutDistortion):
                    view_labels = view_labels[:len(view_coords)]
            return view_coords, view_features, view_labels

        coords1, feats1, labels1 = _apply_view(coords, features, labels)
        coords2, feats2, labels2 = _apply_view(coords, features, labels)

        return coords1, feats1, labels1, coords2, feats2, labels2
