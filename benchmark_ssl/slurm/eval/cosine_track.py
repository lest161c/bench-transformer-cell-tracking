"""Cosine-similarity greedy tracking with the DynaCLR encoder (SPEC 0033 S5).

Why: Achard's Tables 3.2/3.3 score pre-trained features with a *learned-free*
cosine tracker. Running our frozen encoder through the same tracker isolates
feature quality from the trained association head.

Protocol (thesis 3.1): per-object feature = frozen encoder embedding; edges added
by decreasing cosine similarity if score > 0.5, spatial distance <= cutoff, at
most one parent and two children per object, then metrics via his cy/metrics code.

Usage:
    python cosine_track.py <config_yaml> <eval_outdir> [cutoff]

Writes per-test-video predictions (mask*.tif + res_track.txt) and a
summary_mean_std.csv over videos.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import yaml
from skimage.measure import regionprops

import etultra_dynaclr.fork_registry_bridge  # noqa: F401  registers dynaclr-scaledcnn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import metrics  # noqa: E402
from metrics_utils import aggregate_metrics  # noqa: E402

CKPT = os.environ["DYNACLR_CHECKPOINT"]


def _load_video(video: Path):
    """Return (image stack, per-frame label masks, frame names)."""
    frames = sorted((video / "img").glob("*.tif"))
    masks = sorted((video / "TRA").glob("man_track*.tif"))
    images = np.stack([tifffile.imread(f).astype(np.float32) for f in frames])
    labels = [tifffile.imread(f) for f in masks]
    return images, labels, frames


def _per_object_features(extractor, images, labels):
    """Embed each object in each frame; return list of (coords_yx, feats)."""
    extractor.precompute_image_embeddings(images)
    per_frame = []
    for t, mask in enumerate(labels):
        props = regionprops(mask)
        coords_yx = np.array([p.centroid for p in props], dtype=np.float32)
        local = np.array([p.label for p in props])
        if len(coords_yx) == 0:
            per_frame.append((local, coords_yx, np.zeros((0, 128), dtype=np.float32)))
            continue
        timepoints = np.full(len(coords_yx), t)
        feats = extractor.compute_region_features(coords=coords_yx, timepoints=timepoints)
        per_frame.append((local, coords_yx, feats.detach().cpu().numpy()))
    return per_frame


def _greedy_link(per_frame, cutoff):
    """Link detections across frames by cosine similarity; return track table."""
    tracks = {}          # track_id -> [start, end, parent]
    children = {}        # track_id -> count
    frame_track = []     # per frame: {local_label: track_id}
    next_id = 1
    prev_feats = None
    prev_coords = None
    prev_track = None
    for t, (local, coords, feats) in enumerate(per_frame):
        mapping = {}
        assigned_parent = {}
        if prev_feats is not None and len(feats):
            norm_prev = prev_feats / (np.linalg.norm(prev_feats, axis=1, keepdims=True) + 1e-8)
            norm_cur = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            sim = norm_cur @ norm_prev.T
            order = np.dstack(np.unravel_index(np.argsort(-sim, axis=None), sim.shape))[0]
            for child_idx, parent_idx in order:
                score = sim[child_idx, parent_idx]
                if score <= 0.5:
                    break
                if child_idx in assigned_parent:
                    continue
                parent_track = prev_track[parent_idx]
                if children.get(parent_track, 0) >= 2:
                    continue
                distance = np.linalg.norm(coords[child_idx] - prev_coords[parent_idx])
                if distance > cutoff:
                    continue
                assigned_parent[child_idx] = parent_track
                children[parent_track] = children.get(parent_track, 0) + 1
        for j, label in enumerate(local):
            if j in assigned_parent:
                track_id = assigned_parent[j]
                tracks[track_id][1] = t
            else:
                track_id = next_id
                next_id += 1
                parent = assigned_parent.get(j, 0)
                tracks[track_id] = [t, t, parent]
            mapping[int(label)] = track_id
        frame_track.append(mapping)
        prev_feats, prev_coords, prev_track = feats, coords, [mapping[int(x)] for x in local]
    return tracks, frame_track


def _write_prediction(outdir: Path, labels, frame_track, tracks):
    """Write CTC prediction (mask*.tif + res_track.txt) for one video."""
    outdir.mkdir(parents=True, exist_ok=True)
    for t, (mask, mapping) in enumerate(zip(labels, frame_track)):
        relabelled = np.zeros_like(mask)
        for local_label, track_id in mapping.items():
            relabelled[mask == local_label] = track_id
        tifffile.imwrite(outdir / f"mask{t:03d}.tif", relabelled.astype(np.uint16))
    with (outdir / "res_track.txt").open("w") as handle:
        for track_id, (start, end, parent) in sorted(tracks.items()):
            handle.write(f"{track_id} {start} {end} {parent}\n")


def main() -> None:
    """Run the cosine-only comparison over the config's test videos."""
    config = yaml.safe_load(open(sys.argv[1]))
    eval_root = Path(sys.argv[2])
    cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    from trackastra.data.pretrained_features import FeatureExtractor

    rows = []
    for video in config["input_test"]:
        video_path = Path(video)
        images, labels, _ = _load_video(video_path)
        extractor = FeatureExtractor.from_model_name(
            "dynaclr-scaledcnn", image_shape=images.shape[-2:], save_path="/tmp/cosine",
            device="cuda" if torch.cuda.is_available() else "cpu",
            mode="nearest_patch", model_folder=CKPT,
        )
        per_frame = _per_object_features(extractor, images, labels)
        tracks, frame_track = _greedy_link(per_frame, cutoff)
        pred_dir = eval_root / f"cosine_{video_path.name}"
        _write_prediction(pred_dir, labels, frame_track, tracks)
        gt_dir = video_path / "TRA"
        metrics_df, *_ = metrics(gt=gt_dir, pred=pred_dir, outdir=pred_dir / "metrics", n_workers=8)
        agg = aggregate_metrics(metrics_df)
        agg.insert(0, "video", video_path.name)
        rows.append(agg)
        print(f"{video_path.name}: AOGM={float(agg['AOGM'].iloc[0]):.2f} DivF1={float(agg['Division F1'].iloc[0]):.4f}", flush=True)

    per_video = pd.concat(rows, ignore_index=True)
    per_video.to_csv(eval_root / "summary_per_video.csv", index=False)
    numeric = per_video.select_dtypes(include="number")
    print(numeric.mean().to_string())
    numeric.mean().to_frame("mean").to_csv(eval_root / "summary_mean.csv")
    print("COSINE_DONE_OK")


if __name__ == "__main__":
    main()
