"""DynaCLR-trained ScaledCNN (with BatchNorm) as an Et-Ultra FeatureExtractor.

Adapts the ScaledCNN encoder trained with the time-shifted NT-Xent/HCL objective
(SPEC 0026 R7, benchmark_ssl) to the Trackastra-et-Ultra interface: `_run_model`
crops 64x64 at each cell centroid from the per-frame-normalised frame and embeds
each crop with the frozen encoder (the base class patch-grid machinery is
bypassed; `compute_region_features` is overridden).

The training distribution was 64x64 cell-centred crops normalised per frame
(median -> 0, 99th percentile -> 1), so the sliding windows are taken at full
resolution on per-frame-normalised images. Checkpoint:
`cnn_bn100k_step100000.pt` (100K iterations, H100).
"""

import numpy as np
import torch
import torch.nn as nn

from trackastra_pretrained_feats.pretrained_features import FeatureExtractor, register_backbone

PATCH_SIZE = 64


class ScaledCNNBN(nn.Module):
    """ScaledCNN with BatchNorm (vendored from benchmark_ssl ScaledCNN(batch_norm=True))."""

    def __init__(self, scale="large", out_dim=128):
        super().__init__()
        if scale == "small":
            channels = [8, 16, 32]
        elif scale == "medium":
            channels = [16, 32, 64]
        else:
            channels = [32, 64, 128, 256]
        layers = []
        in_channels = 1
        for channel_dim in channels:
            layers += [
                nn.Conv2d(in_channels, channel_dim, 3, padding=1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = channel_dim
        self.conv = nn.Sequential(*layers)
        spatial = PATCH_SIZE // (2 ** len(channels))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, patches):
        """patches: (N, 1, 64, 64) -> embeddings: (N, out_dim)."""
        return self.fc(self.conv(patches))

INPUT_SIZE = (1024, 1024)
GRID_SIZE = (16, 16)


@register_backbone("dynaclr-scaledcnn", 128)
class ScaledCNNDynaCLRFeatures(FeatureExtractor):
    """Feature extractor backed by the DynaCLR-trained ScaledCNN with BatchNorm.

    Produces a 16x16 grid of 128-D window embeddings per frame by sliding a
    64x64 window (the training crop) over the 1024x1024 canvas without
    overlap; per-cell embeddings are then aggregated by the base class
    (mode `nearest_patch` maps each centroid to its containing window).
    """

    model_name = "dynaclr-scaledcnn"

    _dynaclr_checkpoint = None

    @classmethod
    def from_model_name(cls, model_name, image_shape, save_path, checkpoint_path=None, **kwargs):
        """Extend the base factory to forward the DynaCLR checkpoint path."""
        cls._dynaclr_checkpoint = checkpoint_path
        return super().from_model_name(model_name, image_shape, save_path, **kwargs)

    def __init__(self, image_size, save_path, batch_size=16,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 mode="nearest_patch", checkpoint_path=None, **kwargs):
        super().__init__(image_size=image_size, save_path=save_path,
                         batch_size=batch_size, device=device, mode=mode, **kwargs)
        self._input_size = INPUT_SIZE
        self._final_grid_size = GRID_SIZE
        self.model_patch_size = (PATCH_SIZE, PATCH_SIZE)
        self.n_channels = 1
        self.hidden_state_size = 128
        checkpoint_path = checkpoint_path or type(self)._dynaclr_checkpoint
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required for the dynaclr backbone")
        self.encoder = ScaledCNNBN(scale="large", out_dim=128)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["model_state_dict"])
        self.encoder.eval().to(self.device)
        self.model = self.encoder


    def precompute_image_embeddings(self, images, **kwargs):
        """Store per-frame normalised images (median -> 0, 99th pct -> 1).

        The ScaledCNN Encoder crops 64x64 windows from these frames in
        `compute_region_features`; no patch grid is needed because the encoder
        produces one global embedding per crop (it was trained on 64x64
        cell-centred crops, not on spatial feature maps).
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(np.ascontiguousarray(images))
        if isinstance(images, list):
            # Each element is one frame (2D grayscale); convert then stack so
            # the downstream per-frame normalisation sees a (T, H, W) tensor.
            images = [torch.as_tensor(np.asarray(image)) for image in images]
            images = torch.stack(images)
        images = images.float().squeeze()
        if images.dim() == 2:
            images = images[None]
        normed = []
        for frame in images:
            median = torch.median(frame)
            p99 = torch.quantile(frame, 0.99)
            normed.append(((frame - median) / (p99 - median + 1e-8)).clamp(0.0, 1.0))
        self._frames = torch.stack(normed).to(self.device)
        print(f"  precomputed {self._frames.shape[0]} normalised frames")

    def compute_region_features(
        self,
        coords=None,
        masks=None,
        timepoints=None,
        labels=None,
    ):
        """Per-cell 128-D embeddings: 64x64 crop at each centroid -> frozen encoder.

        Overrides the base class: the ScaledCNN encoder produces one global
        embedding per 64x64 crop (no spatial feature map), so each cell is
        embedded directly from a crop centred on its centroid instead of
        aggregating a patch grid.
        """
        feats = []
        for coord, timepoint in zip(coords, timepoints):
            frame = self._frames[int(timepoint)]
            height, width = frame.shape[-2:]
            half = PATCH_SIZE // 2
            cy = max(0, min(height - 1, int(round(float(coord[0])))))
            cx = max(0, min(width - 1, int(round(float(coord[1])))))
            top, left = cy - half, cx - half
            bottom, right = top + PATCH_SIZE, left + PATCH_SIZE
            pad_top, pad_bottom = max(0, -top), max(0, bottom - height)
            pad_left, pad_right = max(0, -left), max(0, right - width)
            crop = frame[max(0, top):min(height, bottom), max(0, left):min(width, right)]
            if crop.numel() == 0:
                crop = torch.zeros(PATCH_SIZE, PATCH_SIZE)
            if pad_top or pad_bottom or pad_left or pad_right:
                crop = torch.nn.functional.pad(
                    crop, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect"
                )
            crop = crop[:PATCH_SIZE, :PATCH_SIZE][None, None]
            with torch.no_grad():
                embedding = self.encoder(crop).squeeze(0)
            feats.append(embedding)
        return torch.stack(feats)


    def _run_model(self, images, **kwargs):
        """Not used by this backbone (documented stub for the ABC).

        `compute_region_features` crops 64x64 at each centroid and embeds per
        cell directly, bypassing the base class's patch-grid path. Present to
        satisfy the FeatureExtractor abstract interface.
        """
        raise NotImplementedError(
            "ScaledCNNDynaCLRFeatures embeds per cell in compute_region_features; "
            "the _run_model patch-grid path is not used."
        )
