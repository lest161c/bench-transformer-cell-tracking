"""Smoke tests for SSLDataset and collate_ssl."""

import numpy as np
import pytest
import torch
from skimage.draw import disk

from src.data import SSLDataset, collate_ssl


def _make_frame_metadata(tmp_path, n_frames=3, n_cells=5, img_size=128):
    """Create a temporary CTC-format directory with synthetic frames."""
    from pathlib import Path
    cond_dir = tmp_path / "test_cond" / "exp01"
    img_dir = cond_dir / "img"
    tra_dir = cond_dir / "TRA"
    img_dir.mkdir(parents=True)
    tra_dir.mkdir(parents=True)

    rng = np.random.RandomState(42)
    frames = []
    for frame_idx in range(n_frames):
        mask = np.zeros((img_size, img_size), dtype=np.int32)
        for i in range(1, n_cells + 1):
            cy, cx = rng.randint(20, img_size - 20, size=2)
            rr, cc = disk((cy, cx), radius=8, shape=mask.shape)
            mask[rr, cc] = i
        img = rng.rand(img_size, img_size).astype(np.float32) * 200 + 50
        from tifffile import imwrite
        mask_path = tra_dir / f"man_track{frame_idx:06d}.tif"
        img_path = img_dir / f"t{frame_idx:06d}.tif"
        imwrite(str(mask_path), mask)
        imwrite(str(img_path), img)
        frames.append(("test_cond", "exp01", frame_idx, str(mask_path), str(img_path)))
    return frames


def test_ssl_dataset_returns_dict(tmp_path):
    from src.distortions import DistortionPipeline
    frames = _make_frame_metadata(tmp_path)
    pipeline = DistortionPipeline.from_config({"distortions": ["jitter"], "jitter": {"std": [2, 4]}, "seed": 42})
    ds = SSLDataset(frames, distortion_pipeline=pipeline, ndim=2)
    assert len(ds) == len(frames)
    item = ds[0]
    assert "coords1" in item
    assert "coords2" in item
    assert "features1" in item
    assert "features2" in item
    assert "labels1" in item
    assert "labels2" in item


def test_collate_ssl_shapes(tmp_path):
    from src.distortions import DistortionPipeline
    frames = _make_frame_metadata(tmp_path, n_frames=3)
    pipeline = DistortionPipeline.from_config({"distortions": ["jitter"], "jitter": {"std": [2, 4]}, "seed": 42})
    ds = SSLDataset(frames, distortion_pipeline=pipeline, ndim=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_ssl)
    for batch in loader:
        assert "coords1" in batch
        assert "padding_mask1" in batch
        assert "valid_pair" in batch
        assert batch["coords1"].dim() == 3
        assert batch["valid_pair"].dtype == torch.bool
        break
