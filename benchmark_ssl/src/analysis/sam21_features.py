"""Frozen SAM2.1-Hiera-B+ feature adapter for the KNN edge probe.

SPEC 0046 adds the feature row ``sam21`` without making the SAM 2 package a
core dependency of ``benchmark_ssl``.  This module is the thin adapter between
the benchmark's per-frame ``(image, centroid)`` representation and the
Trackastra fork's ``SAM2Features`` extractor.  The model is frozen: this module
only runs inference and never updates weights.

Why a separate module instead of extending
``src/edge_probing/harness/feature_extractors.py``: the fork imports heavy
optional packages, and ``trackastra_pretrained_feats/__init__.py`` leaves
``SAM2Features`` commented out.  Keeping the import lazy means importing this
module (and therefore the rest of the analysis package) does not require
``sam2``, Hugging Face weights, or a GPU.

Model: ``facebook/sam2.1-hiera-base-plus``.  The fork's ``nearest_patch`` mode
resizes each frame to 1024x1024, extracts a 64x64 feature grid, maps each
pixel centroid to the nearest 16x16 grid cell, and returns a 256-D feature
vector.  The fork L2-normalizes the selected vectors by default.

Numerics: inference runs under the fork's ``torch.no_grad`` and autocast path
in float32 output.  The model is placed in ``eval`` mode and no randomness is
used, so repeated calls on the same frame and coordinates are deterministic
subject only to the usual CUDA kernel non-determinism.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Hugging Face model identifier used by the Trackastra fork.
SAM21_MODEL_NAME = "facebook/sam2.1-hiera-base-plus"

#: Dimensionality of the frozen SAM2.1-Hiera-B+ feature vector.
SAM21_FEATURE_DIM = 256

#: Exact package installation command required by the optional dependency.
SAM21_INSTALL_HINT = "pip install git+https://github.com/facebookresearch/sam2.git"

#: Repository root, used to locate the vendored Trackastra fork.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

#: Directory containing ``trackastra_pretrained_feats``.
_FORK_ROOT = _REPOSITORY_ROOT / "Trackastra-et-Ultra"

#: Module path of the fork's feature extractors.
_FORK_MODULE_PATH = (
    _FORK_ROOT / "trackastra_pretrained_feats" / "pretrained_features.py"
)

#: Import name of the fork module.  The package ``__init__`` deliberately does
#: not export ``SAM2Features``, so the submodule is imported directly.
_FORK_MODULE_NAME = "trackastra_pretrained_feats.pretrained_features"


def _import_fork_module():
    """Import the Trackastra fork's ``pretrained_features`` module.

    The fork is vendored as a plain directory rather than installed into the
    benchmark virtual environment.  This helper temporarily adds the fork root
    to ``sys.path`` and imports the concrete submodule, avoiding the package's
    ``__init__`` export list where ``SAM2Features`` is commented out.

    Returns:
        The imported ``trackastra_pretrained_feats.pretrained_features``
        module.

    Raises:
        ImportError: If the vendored fork path is missing, or if importing it
            fails because one of its dependencies is unavailable.
    """
    if not _FORK_MODULE_PATH.is_file():
        raise ImportError(
            "SAM2.1 feature adapter could not find the Trackastra fork at "
            f"{_FORK_MODULE_PATH}. Expected "
            "'Trackastra-et-Ultra/trackastra_pretrained_feats/"
            "pretrained_features.py'."
        )

    fork_root_string = str(_FORK_ROOT)
    path_was_added = fork_root_string not in sys.path
    if path_was_added:
        sys.path.insert(0, fork_root_string)
    try:
        return importlib.import_module(_FORK_MODULE_NAME)
    except ImportError as import_error:
        raise ImportError(
            "SAM2.1 feature adapter could not import the Trackastra fork from "
            f"{_FORK_MODULE_PATH}: {import_error}. Install the fork's "
            "dependencies (transformers, zarr, numcodecs, scikit-image, "
            "joblib, tqdm) and a compatible 'trackastra' package."
        ) from import_error
    finally:
        if path_was_added:
            sys.path.remove(fork_root_string)


class Sam21FeatureExtractor:
    """Extract frozen SAM2.1-Hiera-B+ features for per-cell centroids.

    Use one instance per run/protocol, not one per frame: construction loads
    the Hugging Face checkpoint and consumes GPU memory.  Call
    :meth:`extract_frame` for each frame and :meth:`close` when finished.

    The wrapper supports only the fork's ``nearest_patch`` mode.  Other modes
    require masks and labels that the KNN probe does not provide, so they are
    rejected instead of being silently approximated.

    Determinism: inference is run in evaluation mode without gradient
    computation and with no sampling; identical image and coordinate inputs
    produce identical features (up to CUDA kernel nondeterminism).  The
    extractor recomputes embeddings for every frame, so no frame-to-frame
    cache can go stale.

    VRAM: expect roughly 2-4 GB for one 1024x1024 frame on a CUDA GPU.  The
    weights are about 0.3 GB in float32; encoder activations dominate.  The
    spec's 4 GB A500 is a tight smoke-test target, not a full-run GPU.  Use
    ``device="cpu"`` only for debugging, since it is much slower.
    """

    def __init__(
        self,
        device: str = "cuda",
        mode: str = "nearest_patch",
        image_size: tuple[int, int] = (1024, 1024),
    ) -> None:
        """Load the frozen SAM2.1 checkpoint through the Trackastra fork.

        Args:
            device: Torch device string, for example ``"cuda"`` or ``"cpu"``.
            mode: Extraction mode.  Only ``"nearest_patch"`` is supported by
                this adapter.
            image_size: Initial ``(height, width)`` of the source frames used
                to construct the fork extractor.  :meth:`extract_frame`
                re-syncs this to each frame's actual shape before mapping
                pixel coordinates.

        Raises:
            ValueError: If ``mode`` is not ``"nearest_patch"`` or
                ``image_size`` is not a two-element tuple.
            ImportError: If the vendored fork is missing, or if the ``sam2``
                package is missing.  The ``sam2`` error contains
                :data:`SAM21_INSTALL_HINT` verbatim.
        """
        if mode != "nearest_patch":
            raise ValueError(
                "Sam21FeatureExtractor supports only mode='nearest_patch'; "
                f"received {mode!r}. Other fork modes need masks and labels."
            )
        if len(tuple(image_size)) != 2:
            raise ValueError(
                "image_size must be a (height, width) tuple; "
                f"received {image_size!r}."
            )

        if importlib.util.find_spec("sam2") is None:
            raise ImportError(
                "SAM2.1 feature extraction requires the optional 'sam2' "
                f"package. Install it with: {SAM21_INSTALL_HINT}"
            )

        fork_module = _import_fork_module()

        self.device = device
        self.mode = mode
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="sam21_features_"
        )
        try:
            self._extractor = fork_module.SAM2Features(
                image_size=self.image_size,
                save_path=self._temporary_directory.name,
                device=device,
                mode=mode,
                batch_size=1,
                normalize_embeddings=True,
            )
        except Exception:
            self._temporary_directory.cleanup()
            raise

        # The adapter owns frame lifetime; never reuse or write fork caches.
        self._extractor.do_save = False
        self._extractor.force_recompute = True
        self._extractor.model.eval()

    def extract_frame(self, image: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Return SAM2.1 features for every ``(y, x)`` centroid in one frame.

        The fork computes a 1024x1024 image embedding once for this frame and
        then selects the nearest 16x16 feature-map cell for each centroid.
        Feature vectors are L2-normalized by the fork.

        Args:
            image: One grayscale frame with shape ``(H, W)`` and float values
                in ``[0, 1]``, as produced by
                ``src.edge_probing.harness.frame_data.load_frame``.
            coords: ``(N, 2)`` pixel coordinates in ``(y, x)`` order.  The
                coordinates are interpreted in the passed image's pixel
                frame, not in the model's 1024x1024 input frame.

        Returns:
            ``(N, 256)`` float32 feature matrix.  An empty coordinate array
            returns ``(0, 256)`` without running the model.

        Raises:
            RuntimeError: If the extractor has been closed, or if the fork
                returns an unexpected feature shape.
            ValueError: If ``image`` is not 2-D or ``coords`` is not
                ``(N, 2)``.
        """
        if self._extractor is None:
            raise RuntimeError(
                "Sam21FeatureExtractor is closed; construct a new instance."
            )

        image_array = np.asarray(image)
        if image_array.ndim != 2:
            raise ValueError(
                f"image must be a 2-D grayscale array; got shape {image_array.shape}."
            )
        if image_array.dtype != np.float32:
            image_array = image_array.astype(np.float32, copy=False)

        coords_array = np.asarray(coords)
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError(
                "coords must have shape (N, 2) in (y, x) order; "
                f"got shape {coords_array.shape}."
            )
        if coords_array.shape[0] == 0:
            return np.zeros((0, SAM21_FEATURE_DIM), dtype=np.float32)

        # Pixel centroids refer to this frame, so map them with its real shape.
        self._extractor.orig_image_size = (
            int(image_array.shape[0]),
            int(image_array.shape[1]),
        )
        coords_txy = np.column_stack(
            [
                np.zeros(coords_array.shape[0], dtype=np.float32),
                coords_array.astype(np.float32, copy=False),
            ]
        )

        self._extractor.precompute_image_embeddings([image_array])
        feature_tensor = self._extractor.compute_region_features(coords_txy)
        features = (
            feature_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        if features.shape != (coords_txy.shape[0], SAM21_FEATURE_DIM):
            raise RuntimeError(
                "SAM2.1 feature extractor returned an unexpected shape: "
                f"{features.shape}, expected "
                f"{(coords_txy.shape[0], SAM21_FEATURE_DIM)}."
            )
        return np.ascontiguousarray(features, dtype=np.float32)

    def close(self) -> None:
        """Release the model, GPU cache, and the temporary cache directory.

        Safe to call more than once.  After closing, :meth:`extract_frame`
        raises ``RuntimeError`` and a new extractor must be constructed.
        """
        extractor = getattr(self, "_extractor", None)
        if extractor is not None:
            clear_model = getattr(extractor, "clear_model", None)
            if callable(clear_model):
                clear_model()
            self._extractor = None

        temporary_directory = getattr(self, "_temporary_directory", None)
        if temporary_directory is not None:
            temporary_directory.cleanup()
            self._temporary_directory = None
