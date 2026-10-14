"""Bridge that exposes the DynaCLR ScaledCNN+BN encoder to Achard's fork trainer.

Why this exists
---------------
The fork's ``scripts/train.py`` resolves feature backbones from
``trackastra.data.pretrained_features.AVAILABLE_PRETRAINED_BACKBONES``; that
module is a self-contained copy of the feature-extraction API. Our adapter
``ScaledCNNDynaCLRFeatures`` registers into the *different* registry shipped by
the ``trackastra_pretrained_feats`` package, so the fork cannot see
``dynaclr-scaledcnn``. Registering alone is also insufficient: the fork
instantiates the backbone as ``backbone(..., model_folder=<config.model_path>)``
while our adapter reads weights from ``checkpoint_path``, and the fork's
``extract_embedding`` passes coordinates as ``(t, y, x)`` while our adapter
expects ``(y, x)`` plus a separate ``timepoints`` array. This subclass maps all
three.

How it is loaded
----------------
Import this module before ``scripts/train.py`` builds its argparse parser; the
import itself registers the backbone. See ``benchmark_ssl/docs/etultra_bridge.md``.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from .dynaclr_features import ScaledCNNDynaCLRFeatures

try:  # pragma: no cover - depends on the runtime environment
    from trackastra.data.pretrained_features import (
        AVAILABLE_PRETRAINED_BACKBONES as _FORK_BACKBONES,
        register_backbone as _fork_register_backbone,
    )
except ImportError:  # the fork is not importable (e.g. a plain unit-test run)
    _FORK_BACKBONES = None
    _fork_register_backbone = None

BACKBONE_NAME = "dynaclr-scaledcnn"
BACKBONE_FEAT_DIM = 128


class ForkScaledCNNDynaCLRFeatures(ScaledCNNDynaCLRFeatures):
    """Adapter matching the fork's construction and coordinate conventions."""

    def __init__(self, image_size=None, save_path=None, model_folder=None,
                 checkpoint_path=None, **kwargs):
        """Map the fork's ``model_folder`` to the checkpoint; force a writable cache.

        The base extractor creates ``save_path``. During evaluation the fork passes
        a path inside the read-only dataset directory, so fall back to a temp dir
        (this adapter keeps normalised frames in memory and does not use the cache).
        """
        if save_path is None:
            save_path = Path(tempfile.gettempdir()) / "etultra_dynaclr_features"
        else:
            try:
                Path(save_path).mkdir(parents=True, exist_ok=True)
            except OSError:
                save_path = Path(tempfile.gettempdir()) / "etultra_dynaclr_features"
        super().__init__(
            image_size=image_size,
            save_path=save_path,
            checkpoint_path=checkpoint_path or model_folder,
            **kwargs,
        )

    def precompute_image_embeddings(self, images, **kwargs):
        """Delegate to the parent and expose an ``embeddings`` placeholder.

        The fork's ``FeatureExtractorAugWrapper`` moves ``extractor.embeddings``
        to CPU during cleanup; this per-cell adapter stores normalised frames in
        ``_frames`` and never builds a patch-grid embedding tensor, so an empty
        placeholder satisfies that call.
        """
        result = super().precompute_image_embeddings(images, **kwargs)
        self.embeddings = torch.empty(0, device=self.device)
        return result

    def compute_region_features(self, coords=None, masks=None, timepoints=None, labels=None):
        """Drop the base class's leading time column from ``coords``.

        The fork's ``extract_embedding`` prepends the timepoint, giving rows of
        ``(t, y, x)``; the parent adapter expects ``(y, x)`` and a separate
        ``timepoints`` array.
        """
        if coords is not None:
            coords_array = np.asarray(coords, dtype=float)
            if coords_array.ndim == 2 and coords_array.shape[1] == 3:
                coords = coords_array[:, 1:]
        return super().compute_region_features(
            coords=coords, masks=masks, timepoints=timepoints, labels=labels
        )


def register_dynaclr_backbone():
    """Register the DynaCLR backbone in the fork registry; idempotent."""
    if _fork_register_backbone is None or _FORK_BACKBONES is None:
        raise RuntimeError(
            "trackastra.data.pretrained_features is not importable; set PYTHONPATH "
            "to include the fork before importing this bridge."
        )
    if BACKBONE_NAME not in _FORK_BACKBONES:
        _fork_register_backbone(BACKBONE_NAME, BACKBONE_FEAT_DIM)(
            ForkScaledCNNDynaCLRFeatures
        )
    return BACKBONE_NAME


register_dynaclr_backbone()
