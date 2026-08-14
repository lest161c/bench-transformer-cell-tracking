"""Frame loading utilities for CTC-format cell tracking data."""

from pathlib import Path


def load_experiment_frames(exp_dir, conditions=None):
    """Scan directory for CTC-format experiments, return list of (cond, exp_name, frame_idx, mask_path, img_path)."""
    frames = []
    data_root = Path(exp_dir)
    if conditions is None:
        conditions = sorted(
            dir_entry.name for dir_entry in data_root.iterdir()
            if dir_entry.is_dir() and not dir_entry.name.startswith(".")
        )
    for cond in conditions:
        cond_path = data_root / cond
        if not cond_path.is_dir():
            continue
        for exp_path in sorted(cond_path.iterdir()):
            if not exp_path.is_dir():
                continue
            tra_dir = exp_path / "TRA"
            img_dir = exp_path / "img"
            if not tra_dir.exists() or not img_dir.exists():
                continue
            masks = sorted(tra_dir.glob("man_track*.tif"))
            for m_path in masks:
                stem = m_path.stem.replace("man_track", "")
                try:
                    frame_idx = int(stem)
                except ValueError:
                    continue
                img_path = img_dir / f"t{frame_idx:06d}.tif"
                if img_path.exists():
                    frames.append((cond, exp_path.name, frame_idx, str(m_path), str(img_path)))
    return frames
