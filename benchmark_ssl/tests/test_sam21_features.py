"""Tests for the frozen SAM2.1 feature adapter.

Only the missing-package and constant tests run without the optional ``sam2``
dependency.  The model-loading test is skipped when ``sam2`` or the Hugging
Face checkpoint is unavailable; no test installs or downloads anything beyond
what the optional dependency already provides.
"""

import importlib.util

import numpy as np
import pytest

from src.analysis.sam21_features import (
    SAM21_FEATURE_DIM,
    SAM21_INSTALL_HINT,
    SAM21_MODEL_NAME,
    Sam21FeatureExtractor,
)


def _sam2_is_installed() -> bool:
    """Return whether the optional ``sam2`` package can be located."""
    try:
        return importlib.util.find_spec("sam2") is not None
    except (ImportError, ValueError):
        return False


def test_constants_match_spec() -> None:
    """The adapter exposes the exact model, dimension, and install hint."""
    assert SAM21_MODEL_NAME == "facebook/sam2.1-hiera-base-plus"
    assert SAM21_FEATURE_DIM == 256
    assert (
        SAM21_INSTALL_HINT
        == "pip install git+https://github.com/facebookresearch/sam2.git"
    )


@pytest.mark.skipif(
    _sam2_is_installed(),
    reason="sam2 is installed, so the missing-package path is not exercised",
)
def test_missing_sam2_raises_exact_install_hint() -> None:
    """Construction without ``sam2`` raises ImportError with the install hint."""
    with pytest.raises(ImportError) as error_info:
        Sam21FeatureExtractor(device="cpu")

    assert SAM21_INSTALL_HINT in str(error_info.value)


def test_empty_coords_returns_empty_feature_matrix() -> None:
    """Empty coordinates return ``(0, 256)`` float32 without running a frame."""
    pytest.importorskip(
        "sam2", reason="optional SAM2.1 dependency is not installed"
    )

    try:
        extractor = Sam21FeatureExtractor(device="cpu", image_size=(64, 64))
    except (ImportError, OSError, RuntimeError, ValueError) as load_error:
        pytest.skip(f"SAM2.1 model or checkpoint is unavailable: {load_error}")

    try:
        image = np.zeros((64, 64), dtype=np.float32)
        features = extractor.extract_frame(image, np.zeros((0, 2), dtype=np.float32))
    finally:
        extractor.close()

    assert features.shape == (0, SAM21_FEATURE_DIM)
    assert features.dtype == np.float32
