"""Standalone frozen SAM2.1-Hiera-B+ features for the SPEC 0046 KNN probe.

This module is deliberately independent of the Trackastra fork and of
``Trackastra-et-Ultra``'s ``SAM2Features``.  It loads the official ``sam2``
package (``facebookresearch/sam2``) and uses the package's Hugging Face entry
point for ``facebook/sam2.1-hiera-base-plus``.  The fork remains the
feature-injection adapter in :mod:`src.analysis.sam21_features`; it is not
imported here.

Extraction semantics
--------------------
- **Preprocessing:** a grayscale frame is replicated to three channels, resized
  to ``1024 x 1024`` with bilinear interpolation (antialiased), and normalized
  with the ImageNet mean and standard deviation used by SAM 2.  Float frames
  are assumed to be in ``[0, 1]``, matching
  :func:`src.edge_probing.harness.frame_data.load_frame`; integer frames are
  divided by ``255``.
- **Feature map:** the image encoder's final FPN level
  (``backbone_out["vision_features"]``), which is a ``256 x 64 x 64`` map for
  a ``1024 x 1024`` input.  Each grid cell therefore covers a ``16 x 16`` input
  patch.
- **Centroid mapping:** centroids are interpreted as original-frame ``(y, x)``
  pixels.  Each coordinate is scaled into the ``1024 x 1024`` model frame and
  assigned to the ``16 x 16`` patch that contains it, clamped to the valid
  ``0 .. 63`` grid range.
- **Pooling:** nearest patch, i.e. the 256-D vector at the mapped grid cell is
  selected without averaging over neighbouring cells.
- **Dimensionality:** ``D = 256``.
- **Normalization:** the selected 256-D vectors are L2-normalized, matching the
  semantics previously used for the ``sam21`` row.
- **Determinism:** the model is frozen and evaluated under ``torch.no_grad()``
  with no sampling.  Repeated calls on the same device and inputs are
  deterministic up to CUDA kernel non-determinism; CPU results are
  deterministic.

The ``sam2`` import is lazy.  Importing this module and inspecting
:meth:`Sam21ProbeFeatureExtractor.metadata` do not require ``sam2``, a
checkpoint, or a GPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

#: Hugging Face model identifier used for the standalone SAM2.1 features.
SAM21_MODEL_NAME = "facebook/sam2.1-hiera-base-plus"

#: Dimensionality of the frozen SAM2.1-Hiera-B+ feature vector.
SAM21_FEATURE_DIM = 256

#: Exact package installation command required by the optional dependency.
SAM21_INSTALL_HINT = "pip install git+https://github.com/facebookresearch/sam2.git"

#: Square model input resolution used by SAM 2.
SAM21_IMAGE_SIZE = 1024

#: Spatial size of the image-encoder feature map for a ``1024 x 1024`` input.
SAM21_FEATURE_GRID_SIZE = 64

#: Size in input pixels of one feature-map cell.
SAM21_PATCH_SIZE = SAM21_IMAGE_SIZE // SAM21_FEATURE_GRID_SIZE

#: ImageNet channel mean used by the official SAM 2 preprocessing transform.
SAM21_IMAGENET_MEAN = (0.485, 0.456, 0.406)

#: ImageNet channel standard deviation used by the official SAM 2 transform.
SAM21_IMAGENET_STD = (0.229, 0.224, 0.225)


def map_centroids_to_feature_grid(
    centroids: np.ndarray,
    image_shape: tuple[int, int],
    feature_grid_shape: tuple[int, int],
    input_size: int = SAM21_IMAGE_SIZE,
) -> np.ndarray:
    """Map original-image centroids to feature-map row/column indices.

    Each centroid is scaled from the original frame into the square model
    input, then assigned to the feature-map cell whose half-open input-patch
    interval contains the scaled coordinate.  Away from exact patch boundaries
    this is the nearest patch; a point exactly on a boundary belongs to the
    higher-index patch.  Out-of-frame coordinates are clamped to the valid grid
    range so a malformed centroid cannot index outside the feature map.

    Args:
        centroids: ``(N, 2)`` array of ``(y, x)`` coordinates in the original
            frame's pixel space.
        image_shape: Original frame ``(height, width)`` in pixels.
        feature_grid_shape: Feature-map ``(height, width)`` in cells.
        input_size: Square model input resolution in pixels.

    Returns:
        ``(N, 2)`` ``int64`` array of ``(row, column)`` feature-map indices.

    Raises:
        ValueError: If the centroid, image-shape, or grid-shape arguments do
            not have the expected form.
    """
    centroid_array = np.asarray(centroids)
    if centroid_array.ndim != 2 or centroid_array.shape[1] != 2:
        raise ValueError(
            "centroids must have shape (N, 2) in (y, x) order; "
            f"got shape {centroid_array.shape}."
        )

    image_height, image_width = (int(image_shape[0]), int(image_shape[1]))
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            f"image_shape must be positive, got {(image_height, image_width)}."
        )

    grid_height, grid_width = (
        int(feature_grid_shape[0]),
        int(feature_grid_shape[1]),
    )
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(
            "feature_grid_shape must be positive, got "
            f"{(grid_height, grid_width)}."
        )

    if centroid_array.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)

    patch_height = float(input_size) / float(grid_height)
    patch_width = float(input_size) / float(grid_width)
    scaled_y = centroid_array[:, 0].astype(np.float64) * (
        float(input_size) / float(image_height)
    )
    scaled_x = centroid_array[:, 1].astype(np.float64) * (
        float(input_size) / float(image_width)
    )

    row_indices = np.floor(scaled_y / patch_height).astype(np.int64)
    column_indices = np.floor(scaled_x / patch_width).astype(np.int64)
    row_indices = np.clip(row_indices, 0, grid_height - 1)
    column_indices = np.clip(column_indices, 0, grid_width - 1)
    return np.column_stack([row_indices, column_indices])


def pool_feature_map_at_centroids(
    feature_map: np.ndarray,
    centroids: np.ndarray,
    image_shape: tuple[int, int],
    input_size: int = SAM21_IMAGE_SIZE,
) -> np.ndarray:
    """Select one feature vector per centroid from a dense feature map.

    This is the testable pooling half of the extractor.  It accepts a
    ``(channels, height, width)`` map or a leading singleton batch axis and
    returns the feature vectors at the mapped grid cells.  No model or optional
    dependency is needed.

    Args:
        feature_map: Dense map with shape ``(C, H, W)`` or ``(1, C, H, W)``.
        centroids: ``(N, 2)`` original-frame ``(y, x)`` coordinates.
        image_shape: Original frame ``(height, width)`` in pixels.
        input_size: Square model input resolution in pixels.

    Returns:
        ``(N, C)`` ``float32`` feature matrix in centroid row order.  An empty
        centroid array yields ``(0, C)``.

    Raises:
        ValueError: If the feature map does not have three or four dimensions,
            or if the centroid or shape arguments are invalid.
    """
    feature_array = np.asarray(feature_map)
    if feature_array.ndim == 4 and feature_array.shape[0] == 1:
        feature_array = feature_array[0]
    if feature_array.ndim != 3:
        raise ValueError(
            "feature_map must have shape (C, H, W) or (1, C, H, W); "
            f"got shape {feature_array.shape}."
        )

    feature_grid_shape = (int(feature_array.shape[1]), int(feature_array.shape[2]))
    grid_indices = map_centroids_to_feature_grid(
        centroids,
        image_shape,
        feature_grid_shape,
        input_size=input_size,
    )
    if grid_indices.shape[0] == 0:
        return np.zeros((0, int(feature_array.shape[0])), dtype=np.float32)

    row_indices = grid_indices[:, 0]
    column_indices = grid_indices[:, 1]
    selected_features = feature_array[:, row_indices, column_indices].T
    return np.ascontiguousarray(selected_features, dtype=np.float32)


def l2_normalize_rows(feature_matrix: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized features with a stable zero-row guard.

    Args:
        feature_matrix: ``(N, D)`` feature matrix.

    Returns:
        A new ``float32`` matrix whose non-zero rows have unit L2 norm.  A
        zero row stays zero instead of becoming ``NaN``.

    Raises:
        ValueError: If the input is not two-dimensional.
    """
    feature_array = np.asarray(feature_matrix, dtype=np.float32)
    if feature_array.ndim != 2:
        raise ValueError(
            f"feature_matrix must have shape (N, D); got {feature_array.shape}."
        )
    row_norms = np.linalg.norm(feature_array, axis=1, keepdims=True)
    safe_norms = np.maximum(row_norms, np.float32(1e-12))
    return np.ascontiguousarray(feature_array / safe_norms, dtype=np.float32)


class Sam21ProbeFeatureExtractor:
    """Extract frozen SAM2.1-Hiera-B+ features through the official package.

    Construct one instance per run or protocol: construction loads the Hugging
    Face checkpoint and consumes GPU memory.  Reuse it for every frame, then
    call :meth:`close`.

    Args:
        device: Torch device string, for example ``"cuda"`` or ``"cpu"``.

    Raises:
        ImportError: If the optional ``sam2`` package is missing or cannot be
            imported.  The message contains :data:`SAM21_INSTALL_HINT`
            verbatim.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Load the official SAM2.1 checkpoint and its preprocessing transform.

        Args:
            device: Torch device string, for example ``"cuda"`` or ``"cpu"``.

        Raises:
            ImportError: If the optional ``sam2`` package or one of its
                dependencies is unavailable.  The message contains
                :data:`SAM21_INSTALL_HINT`.
        """
        import importlib.util

        if importlib.util.find_spec("sam2") is None:
            raise ImportError(
                "SAM2.1 probe features require the optional 'sam2' package. "
                f"Install it with: {SAM21_INSTALL_HINT}"
            )

        try:
            from sam2.build_sam import build_sam2_hf
        except ImportError as import_error:
            raise ImportError(
                "SAM2.1 probe features could not import the official 'sam2' "
                f"package ({import_error}). Install it with: "
                f"{SAM21_INSTALL_HINT}"
            ) from import_error

        self.device = torch.device(device)
        self._model = build_sam2_hf(
            SAM21_MODEL_NAME,
            device=str(self.device),
            mode="eval",
        )
        self._model.eval()

    @classmethod
    def metadata(cls) -> dict[str, Any]:
        """Return the extraction configuration recorded in per-run JSON files.

        Returns:
            A JSON-serialisable mapping that describes the model, input
            preprocessing, feature-map layer and grid, centroid mapping,
            pooling, normalization, and determinism.
        """
        return {
            "feature": "sam21",
            "implementation": (
                "standalone official sam2 package; no Trackastra fork import"
            ),
            "model": SAM21_MODEL_NAME,
            "feature_dim": int(SAM21_FEATURE_DIM),
            "extraction_mode": "nearest_patch",
            "image_preprocessing": {
                "color": "grayscale replicated to three RGB channels",
                "value_range": "float input in [0, 1]; integer input divided by 255",
                "resize": [int(SAM21_IMAGE_SIZE), int(SAM21_IMAGE_SIZE)],
                "resize_interpolation": "bilinear, antialias=True",
                "normalization": "ImageNet mean/std",
                "mean": list(SAM21_IMAGENET_MEAN),
                "std": list(SAM21_IMAGENET_STD),
            },
            "feature_map_layer": (
                "image encoder final FPN level "
                "(backbone_out['vision_features'], stride 16)"
            ),
            "feature_map_shape": [
                int(SAM21_FEATURE_DIM),
                int(SAM21_FEATURE_GRID_SIZE),
                int(SAM21_FEATURE_GRID_SIZE),
            ],
            "feature_grid": [
                int(SAM21_FEATURE_GRID_SIZE),
                int(SAM21_FEATURE_GRID_SIZE),
            ],
            "patch_size": [int(SAM21_PATCH_SIZE), int(SAM21_PATCH_SIZE)],
            "centroid_mapping": (
                "scale original-frame (y, x) into the 1024x1024 model input, "
                "then floor into 16x16 patches and clamp to 0..63"
            ),
            "pooling": (
                "nearest patch: select the 256-D vector at the mapped grid "
                "cell without spatial averaging"
            ),
            "embedding_normalization": "l2",
            "l2_normalize": True,
            "determinism": (
                "frozen weights, eval mode, torch.no_grad(), no sampling; "
                "deterministic up to CUDA kernel non-determinism"
            ),
            "install_hint": SAM21_INSTALL_HINT,
        }

    def _prepare_model_input(self, image: np.ndarray) -> torch.Tensor:
        """Convert one original frame into the SAM 2 input tensor.

        Args:
            image: Grayscale ``(H, W)`` or channel-last ``(H, W, 1)`` /
                ``(H, W, 3)`` frame.  Float values are assumed to be in
                ``[0, 1]``; integer values are divided by ``255``.

        Returns:
            A ``(1, 3, 1024, 1024)`` float32 tensor on :attr:`device`.

        Raises:
            ValueError: If the frame does not have a supported shape.
            TypeError: If the frame has a non-numeric dtype.
        """
        image_array = np.asarray(image)
        if image_array.dtype == np.bool_ or not np.issubdtype(
            image_array.dtype, np.number
        ):
            raise TypeError(
                "image must have a numeric dtype; "
                f"got {image_array.dtype}."
            )

        if image_array.ndim == 2:
            grayscale_image = image_array
        elif image_array.ndim == 3 and image_array.shape[2] in (1, 3):
            if image_array.shape[2] == 1:
                grayscale_image = image_array[:, :, 0]
            else:
                grayscale_image = image_array
        else:
            raise ValueError(
                "image must have shape (H, W), (H, W, 1) or (H, W, 3); "
                f"got shape {image_array.shape}."
            )

        if np.issubdtype(grayscale_image.dtype, np.integer):
            image_float = grayscale_image.astype(np.float32) / 255.0
        else:
            image_float = grayscale_image.astype(np.float32, copy=False)
        image_float = np.ascontiguousarray(image_float)

        image_tensor = torch.from_numpy(image_float)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        else:
            image_tensor = image_tensor.permute(2, 0, 1)
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        image_tensor = image_tensor.unsqueeze(0).to(
            device=self.device, dtype=torch.float32
        )
        image_tensor = torch_functional.interpolate(
            image_tensor,
            size=(SAM21_IMAGE_SIZE, SAM21_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean_tensor = torch.tensor(
            SAM21_IMAGENET_MEAN, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        std_tensor = torch.tensor(
            SAM21_IMAGENET_STD, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        return (image_tensor - mean_tensor) / std_tensor

    def extract_frame(self, image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return one frozen SAM2.1 feature vector per centroid in a frame.

        Args:
            image: One grayscale frame with shape ``(H, W)`` and float values
                in ``[0, 1]`` (or an integer frame, scaled internally), as
                produced by
                :func:`src.edge_probing.harness.frame_data.load_frame`.
            centroids: ``(N, 2)`` original-frame ``(y, x)`` centroids.  The
                coordinates refer to the passed frame, not to the resized
                model input.

        Returns:
            ``(N, 256)`` L2-normalized ``float32`` features in centroid row
            order.  An empty centroid array returns ``(0, 256)`` without
            running the model.

        Raises:
            RuntimeError: If the extractor has been closed or the model returns
                an unexpected feature-map shape.
            ValueError: If the image or centroid arguments are invalid.
        """
        if getattr(self, "_model", None) is None:
            raise RuntimeError(
                "Sam21ProbeFeatureExtractor is closed; construct a new instance."
            )

        image_array = np.asarray(image)
        if image_array.ndim not in (2, 3):
            raise ValueError(
                "image must have shape (H, W), (H, W, 1) or (H, W, 3); "
                f"got shape {image_array.shape}."
            )

        centroid_array = np.asarray(centroids)
        if centroid_array.ndim != 2 or centroid_array.shape[1] != 2:
            raise ValueError(
                "centroids must have shape (N, 2) in (y, x) order; "
                f"got shape {centroid_array.shape}."
            )
        if centroid_array.shape[0] == 0:
            return np.zeros((0, SAM21_FEATURE_DIM), dtype=np.float32)

        model_input = self._prepare_model_input(image_array)
        with torch.no_grad():
            backbone_output = self._model.forward_image(model_input)
        feature_map = backbone_output["vision_features"]
        if feature_map.ndim != 4 or feature_map.shape[0] != 1:
            raise RuntimeError(
                "SAM2.1 image encoder returned an unexpected feature map: "
                f"{tuple(feature_map.shape)}."
            )
        if tuple(feature_map.shape[1:]) != (
            SAM21_FEATURE_DIM,
            SAM21_FEATURE_GRID_SIZE,
            SAM21_FEATURE_GRID_SIZE,
        ):
            raise RuntimeError(
                "SAM2.1 image encoder returned an unexpected feature-map "
                f"shape {tuple(feature_map.shape)}; expected "
                f"(1, {SAM21_FEATURE_DIM}, {SAM21_FEATURE_GRID_SIZE}, "
                f"{SAM21_FEATURE_GRID_SIZE})."
            )

        feature_map_array = (
            feature_map[0].detach().to(dtype=torch.float32).cpu().numpy()
        )
        selected_features = pool_feature_map_at_centroids(
            feature_map_array,
            centroid_array,
            image_shape=(int(image_array.shape[0]), int(image_array.shape[1])),
        )
        normalized_features = l2_normalize_rows(selected_features)
        if normalized_features.shape != (
            centroid_array.shape[0],
            SAM21_FEATURE_DIM,
        ):
            raise RuntimeError(
                "SAM2.1 pooling returned an unexpected shape: "
                f"{normalized_features.shape}, expected "
                f"{(centroid_array.shape[0], SAM21_FEATURE_DIM)}."
            )
        return normalized_features

    def close(self) -> None:
        """Release the model and cached GPU memory.

        Safe to call more than once.  After closing, :meth:`extract_frame`
        raises ``RuntimeError`` and a new extractor must be constructed.
        """
        self._model = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
