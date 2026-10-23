#!/usr/bin/env python3
"""Coordinate-solvability of the proposal §4.2 warp-identity pretext.

Question (SPEC 0035, revised by SPEC 0047): the §4.2 pretext labels a pair of
cells as associated iff they are the same cell under a geometric warp. How much
of that correspondence is recoverable from *coordinates alone*?

What this experiment measures, and what it does not:

- It measures whether the identity labels are largely a *function of the
  coordinate inputs* the pretext model already receives (a positional shortcut).
  The matchers below use only cell centroids; the strongest of them
  (``family_grid_global``) knows the pretext's own warp family
  (rotation x independent x/y scale), which is legitimate for measuring
  solvability *of this pretext* because the warp is drawn from that family by
  construction.
- It does NOT measure what a *trained* model learns from the pretext, and it
  does NOT establish that the pretext is a poor pretraining objective.
  Coordinate-solvability does not entail absence of feature-learning signal:
  correspondence pretraining from geometrically transformed views is an
  established, successful recipe in the literature, and whether gradient
  descent on the actual model exploits the positional shortcut instead of
  visual features is a separate, untested question (SPEC 0035 Tier 2).

Method (closed-form, no training, no GPU): draw cell centroids from vanvliet
frames, apply the pretext's distortion, then match source to warped cells using
coordinate-only matchers and score against the true identity correspondence,
per distortion family. Matchers:

- ``nearest_neighbour``, ``mutual_nn``, ``hungarian``: raw distance baselines.
- ``registration``: single-start ICP (Hungarian init + least-squares affine).
  Initialization-limited: under large rotations it converges to a local
  optimum in a sizeable fraction of runs (see the per-config
  ``converged``/``stuck`` diagnostics). Its accuracy is a *lower* bound on
  coordinate-solvability; it is kept for continuity with the 2026-10-14
  artifact.
- ``whitened_global``: covariance whitening + rotation-angle scan; exact for
  the pure-affine family, but degrades under per-cell jitter (whitening
  amplifies jitter along the cloud's compressed axis).
- ``family_grid_global``: centroid-aligned grid search over the pretext's own
  warp family (3 parameters) + trimmed multi-candidate refinement;
  initialization-free and robust under jitter and dropout.
- ``oracle_affine``: Hungarian matching after applying the TRUE warp; the
  jitter-limited ceiling of coordinate-only matching.
- ``random``: chance level.

The distortion functions are numerically identical to
``src/mini_trackastra/end_to_end.py`` (``apply_affine``: uniform rotation in
+/-deg about the image origin, independent x/y scale; ``apply_jitter``:
Gaussian per-cell jitter; ``apply_dropout``: per-cell drop) and consume random
numbers in the same order. The pretext's actual configuration is mode "full":
affine(deg=10, scale 0.9-1.1) + jitter(4) + dropout(0.1).

The global matchers (``whitened_global``, ``family_grid_global``) run only on
configs that contain a global warp component (all except ``jitter4`` and
``jitter8``, where no global transform exists beyond the identity and the raw
distance matchers already bound the answer); their entries are ``null`` there.

Reproduce (from the repository root)::

    benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_coordinate_solvability.py

Outputs ``benchmark_ssl/results/pretext_coordinate_solvability.json`` and prints
a table. Deterministic (numpy seed 42).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from tifffile import imread

SEED = 42
N_REPS = 15
MIN_CELLS = 20
# Only frames with at least this many cells are used: sparse frames make the
# assignment problem trivial and the accuracy noisy.

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "vanvliet"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Experiments across conditions; each contributes up to FRAMES_PER_EXP frames
# that pass the MIN_CELLS density floor.
EXPERIMENTS = [
    ("cib", "140409-03"),
    ("cib", "140415-08"),
    ("rpsM", "151101_E4-12"),
    ("rpsM", "151029_E1-1"),
    ("pheA", "160112-06"),
    ("recA", "151027-05"),
    ("metA", "150317-07"),
    ("metA", "150331-12"),
]
FRAMES_PER_EXP = 12

# Matchers that require a global warp component to be meaningful.
GLOBAL_MATCHERS = ("whitened_global", "family_grid_global")
JITTER_ONLY_CONFIGS = ("jitter4", "jitter8")

# Family-grid search parameters (the pretext warp family: rotation about the
# image origin composed with independent x/y scale).
FAMILY_GRID_ANGLE_STEP_DEG = 3.0
FAMILY_GRID_SCALE_RANGE = (0.7, 1.3)
FAMILY_GRID_SCALE_STEP = 0.1
FAMILY_GRID_TOP_REFINE = 3

# Whitened rotation-search parameters.
WHITENED_COARSE_STEP_DEG = 0.25
WHITENED_FINE_STEP_DEG = 0.05
WHITENED_INLIER_TOLERANCE = 0.05
WHITENED_FINE_WINDOW_DEG = 2.5

KD_WORKERS = -1


# ─── Distortions (numerically identical to end_to_end.py helpers) ────────────

def apply_affine(coords: np.ndarray, deg: float, scale_range):
    """Uniform rotation in +/-deg about the image origin, independent x/y scale.

    Returns ``(warped_coords, transform)``; the random draws happen in the same
    order as in ``end_to_end.py`` (angle, then scale_x, then scale_y), so the
    distorted coordinates are identical to the pretext's.
    """
    theta = np.random.uniform(-deg, deg) / 180 * np.pi
    scale_x, scale_y = np.random.uniform(*scale_range), np.random.uniform(*scale_range)
    transform = np.array(
        [[scale_x * np.cos(theta), -scale_x * np.sin(theta)],
         [scale_y * np.sin(theta), scale_y * np.cos(theta)]]
    )
    return coords @ transform, transform


def apply_jitter(coords: np.ndarray, std: float) -> np.ndarray:
    """Add independent Gaussian position noise to each cell."""
    return coords + np.random.randn(*coords.shape).astype(np.float32) * std


def apply_translation(coords: np.ndarray, magnitude: float):
    """Shift every cell by the same random 2D offset; returns (warped, offset)."""
    offset = np.random.uniform(-magnitude, magnitude, size=2)
    return coords + offset, offset


def apply_dropout(coords: np.ndarray, labels: np.ndarray, p: float):
    """Drop each cell independently with probability p (detection-failure proxy)."""
    keep = np.random.rand(len(labels)) > p
    return coords[keep], labels[keep]


def distort(coords: np.ndarray, labels: np.ndarray, config: str):
    """Apply a named distortion configuration.

    Returns ``(warped_coords, labels, transform, offset)`` where
    ``transform``/``offset`` describe the global warp that was applied
    (``warped = source @ transform + offset``); per-cell jitter is noise, not
    part of the transform, so for jitter-only configs the transform is the
    identity. The oracle matcher uses the true transform.

    ``full_actual`` is the pretext's own configuration (end_to_end.py, mode
    "full"); ``full_strong`` uses the wider ranges stated in the (removed)
    report text, for a sensitivity comparison.
    """
    identity = np.eye(2)
    if config == "identity":
        return coords.copy(), labels.copy(), identity, np.zeros(2)
    if config == "translation":
        warped, offset = apply_translation(coords.copy(), 20.0)
        return warped, labels.copy(), identity, offset
    if config == "affine_actual":
        warped, transform = apply_affine(coords.copy(), 10, (0.9, 1.1))
        return warped, labels.copy(), transform, np.zeros(2)
    if config == "affine_strong":
        warped, transform = apply_affine(coords.copy(), 25, (0.7, 1.3))
        return warped, labels.copy(), transform, np.zeros(2)
    if config == "jitter4":
        return apply_jitter(coords.copy(), 4), labels.copy(), identity, np.zeros(2)
    if config == "jitter8":
        return apply_jitter(coords.copy(), 8), labels.copy(), identity, np.zeros(2)
    if config == "full_actual":
        warped, transform = apply_affine(coords.copy(), 10, (0.9, 1.1))
        warped = apply_jitter(warped, 4)
        warped_kept, labels_kept = apply_dropout(warped, labels.copy(), 0.1)
        return warped_kept, labels_kept, transform, np.zeros(2)
    if config == "full_strong":
        warped, transform = apply_affine(coords.copy(), 25, (0.7, 1.3))
        warped = apply_jitter(warped, 8)
        warped_kept, labels_kept = apply_dropout(warped, labels.copy(), 0.1)
        return warped_kept, labels_kept, transform, np.zeros(2)
    raise ValueError(f"unknown config: {config}")


CONFIGS = [
    "identity", "translation", "affine_actual", "affine_strong",
    "jitter4", "jitter8", "full_actual", "full_strong",
]


# ─── Data loading ────────────────────────────────────────────────────────────

def load_frame_centroids(mask_path: Path) -> np.ndarray | None:
    """Return (N, 2) centroid array (y, x) for a frame mask, or None if sparse."""
    mask = imread(mask_path)
    props = regionprops_table(mask, properties=("label", "centroid"))
    if len(props["label"]) < MIN_CELLS:
        return None
    return np.stack([props["centroid-0"], props["centroid-1"]], axis=-1).astype(np.float32)


def collect_frames() -> list[np.ndarray]:
    """Collect centroid arrays for a handful of frames per experiment."""
    frames = []
    for condition, experiment in EXPERIMENTS:
        tra = DATA_ROOT / condition / experiment / "TRA"
        if not tra.exists():
            continue
        taken = 0
        for mask_path in sorted(tra.glob("man_track*.tif")):
            coords = load_frame_centroids(mask_path)
            if coords is None:
                continue
            frames.append(coords)
            taken += 1
            if taken >= FRAMES_PER_EXP:
                break
    return frames


# ─── Coordinate-only matchers ────────────────────────────────────────────────
#
# All matchers share the signature (source, warped, transform, offset) so they
# are interchangeable; matchers that do not estimate the warp ignore the
# transform arguments. "source" and "warped" are (N,2)/(M,2) centroid arrays,
# "transform"/"offset" describe the true global warp (warped = source @ transform
# + offset) and are used only by the oracle.

def _distance_matrix(source: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix, source x warped."""
    return np.linalg.norm(source[:, None, :] - warped[None, :, :], axis=-1)


def match_nearest_neighbour(source: np.ndarray, warped: np.ndarray,
                            transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """For each source cell, the index of the nearest warped cell."""
    return np.argmin(_distance_matrix(source, warped), axis=1)


def match_mutual_nn(source: np.ndarray, warped: np.ndarray,
                    transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Nearest warped cell, kept only if the relation is mutual, else -1."""
    dist = _distance_matrix(source, warped)
    forward = np.argmin(dist, axis=1)
    backward = np.argmin(dist, axis=0)
    return np.array(
        [j if backward[j] == i else -1 for i, j in enumerate(forward)]
    )


def match_hungarian(source: np.ndarray, warped: np.ndarray,
                    transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Minimum-total-distance one-to-one assignment; -1 for unmatched sources."""
    dist = _distance_matrix(source, warped)
    rows, cols = linear_sum_assignment(dist)
    assignment = np.full(len(source), -1, dtype=int)
    assignment[rows] = cols
    return assignment


def match_random(source: np.ndarray, warped: np.ndarray,
                 transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Uniformly random warped index per source cell: the chance-level baseline."""
    return np.random.randint(0, len(warped), size=len(source))


def match_registration(source: np.ndarray, warped: np.ndarray,
                       transform: np.ndarray, offset: np.ndarray,
                       iters: int = 5) -> np.ndarray:
    """Single-start ICP coordinate matcher (Hungarian init + LSQ affine).

    The pretext warp is a *global* affine, so a coordinate-only solver that
    estimates the warp is stronger than raw nearest-distance matching. This
    implementation bootstraps from the raw Hungarian assignment, which fails
    when the warp displacement exceeds the inter-cell spacing: the fit then
    converges to a local optimum. Its accuracy is therefore an
    *initialization-limited lower bound* on coordinate-solvability, not an
    upper bound; the converged/stuck run diagnostics in the JSON quantify the
    failure mode. Kept for continuity with the 2026-10-14 artifact.
    """
    assignment = match_hungarian(source, warped, transform, offset)
    for _ in range(iters):
        matched = assignment >= 0
        if matched.sum() < 3:
            break
        design = np.hstack([source[matched], np.ones((int(matched.sum()), 1))])
        target = warped[assignment[matched]]
        params, *_ = np.linalg.lstsq(design, target, rcond=None)
        transformed = np.hstack(
            [source, np.ones((len(source), 1))]
        ) @ params
        assignment = match_hungarian(transformed, warped, transform, offset)
    return assignment


def refine_from_projection(source: np.ndarray, warped: np.ndarray,
                           transformed: np.ndarray) -> np.ndarray:
    """Trimmed ICP refinement starting from an initial projection of source.

    Alternates Hungarian assignment and least-squares affine fitting, but trims
    gross outlier pairs (dropout-forced mismatches, jitter extremes) from the
    fit; keeping them drags the affine away from the true transform.
    """
    assignment = match_hungarian(transformed, warped, np.eye(2), np.zeros(2))
    for _ in range(10):
        matched = assignment >= 0
        if matched.sum() < 3:
            break
        residuals = np.linalg.norm(
            transformed[matched] - warped[assignment[matched]], axis=-1)
        inlier_threshold = max(3.0 * float(np.median(residuals)), 1.0)
        trimmed = matched.copy()
        trimmed[np.where(matched)[0][residuals > inlier_threshold]] = False
        if trimmed.sum() < 3:
            break
        design = np.hstack([source[trimmed], np.ones((int(trimmed.sum()), 1))])
        target = warped[assignment[trimmed]]
        params, *_ = np.linalg.lstsq(design, target, rcond=None)
        transformed = np.hstack([source, np.ones((len(source), 1))]) @ params
        assignment = match_hungarian(transformed, warped, np.eye(2), np.zeros(2))
    return assignment


def match_whitened_global(source: np.ndarray, warped: np.ndarray,
                          transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Initialization-free global affine matcher, exact for pure affines.

    Any invertible affine mapping between two point sets acts as an orthogonal
    transform after whitening both sets (centroid removal + covariance
    whitening). In 2D that orthogonal transform is a single rotation angle, so
    the global optimum is found by scanning the angle, matching in whitened
    space, and refining the recovered affine in pixel space. Under per-cell
    jitter the whitened-space criterion degrades (whitening amplifies jitter
    along the cloud's compressed axis), which is why the family grid search is
    preferred for jittered configs.
    """
    source_center = source.mean(axis=0)
    warped_center = warped.mean(axis=0)
    source_centered = source - source_center
    warped_centered = warped - warped_center
    source_covariance = source_centered.T @ source_centered / len(source)
    warped_covariance = warped_centered.T @ warped_centered / len(warped)
    source_whitening_inverse = np.linalg.inv(np.linalg.cholesky(source_covariance))
    warped_whitening_inverse = np.linalg.inv(np.linalg.cholesky(warped_covariance))
    source_whitened = source_centered @ source_whitening_inverse.T
    warped_whitened = warped_centered @ warped_whitening_inverse.T

    # Coarse scan: full circle, count inliers in whitened space.
    best_inliers, best_angle = -1, 0.0
    for angle_deg in np.arange(0.0, 360.0, WHITENED_COARSE_STEP_DEG):
        angle = angle_deg / 180 * np.pi
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        rotated = source_whitened @ rotation
        distance = np.linalg.norm(
            rotated[:, None, :] - warped_whitened[None, :, :], axis=-1)
        inliers = int((distance.min(axis=1) < WHITENED_INLIER_TOLERANCE).sum())
        if inliers > best_inliers:
            best_inliers, best_angle = inliers, angle

    # Fine scan around the coarse optimum: the coarse grid can leave a ~1 deg
    # error, which at whitened radius 2.5 displaces points by ~0.04 units --
    # enough to misassign dense cells during refinement.
    best_residual = np.inf
    for angle_deg in (best_angle * 180 / np.pi
                      + np.arange(-WHITENED_FINE_WINDOW_DEG,
                                  WHITENED_FINE_WINDOW_DEG,
                                  WHITENED_FINE_STEP_DEG)):
        angle = angle_deg / 180 * np.pi
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        rotated = source_whitened @ rotation
        distance = np.linalg.norm(
            rotated[:, None, :] - warped_whitened[None, :, :], axis=-1)
        mean_residual = float(distance.min(axis=1).mean())
        if mean_residual < best_residual:
            best_residual, best_angle = mean_residual, angle

    # Map the best whitened-space rotation back to a pixel-space affine
    # (row-vector convention: with Q the whitened-space rotation, the pixel
    # affine is L_s^{-T} R L_y^{T} for R = Q^T, plus the centroid shift).
    rotation = np.array([[np.cos(best_angle), -np.sin(best_angle)],
                         [np.sin(best_angle), np.cos(best_angle)]])
    pixel_transform = (source_whitening_inverse.T @ rotation
                       @ np.linalg.inv(warped_whitening_inverse).T)
    transformed = (source - source_center) @ pixel_transform + warped_center
    return refine_from_projection(source, warped, transformed)


def match_family_grid_global(source: np.ndarray, warped: np.ndarray,
                             transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Initialization-free matcher restricted to the pretext's warp family.

    The pretext warp is a rotation (about the image origin) composed with an
    independent x/y scale -- a 3-parameter family. Translation is eliminated by
    aligning centroids, so a coarse grid over (angle, scale_x, scale_y) covers
    the family; candidates are ranked by a trimmed mean nearest-warped distance
    (robust to the 10% dropout), and the top candidates are refined by trimmed
    ICP and re-ranked by their final median assignment residual. Initialization-
    free and robust under jitter and dropout, unlike the whitening scan.
    """
    source_center = source.mean(axis=0)
    warped_center = warped.mean(axis=0)
    centered = source - source_center
    warped_tree = cKDTree(warped)

    angles = np.arange(0.0, 360.0, FAMILY_GRID_ANGLE_STEP_DEG) / 180 * np.pi
    scales = np.arange(FAMILY_GRID_SCALE_RANGE[0],
                       FAMILY_GRID_SCALE_RANGE[1] + FAMILY_GRID_SCALE_STEP / 2,
                       FAMILY_GRID_SCALE_STEP)
    candidate_transforms = np.array([
        [[scale_x * np.cos(angle), -scale_x * np.sin(angle)],
         [scale_y * np.sin(angle), scale_y * np.cos(angle)]]
        for angle in angles
        for scale_x in scales
        for scale_y in scales
    ])
    projected = np.einsum("nd,kdc->knc", centered, candidate_transforms) \
        + warped_center
    distances, _ = warped_tree.query(
        projected.reshape(-1, 2), workers=KD_WORKERS)
    distances = distances.reshape(len(candidate_transforms), len(source))
    kept = max(3, int(0.9 * len(source)))
    scores = np.mean(np.sort(distances, axis=1)[:, :kept], axis=1)
    ranked = np.argsort(scores)

    best_assignment, best_residual = None, np.inf
    for candidate_index in ranked[:FAMILY_GRID_TOP_REFINE]:
        transformed = projected[candidate_index]
        assignment = refine_from_projection(source, warped, transformed)
        matched = assignment >= 0
        if matched.sum() < 3:
            continue
        residual = float(np.median(np.linalg.norm(
            transformed[matched] - warped[assignment[matched]], axis=-1)))
        if residual < best_residual:
            best_residual, best_assignment = residual, assignment
    if best_assignment is None:
        return match_hungarian(source, warped, transform, offset)
    return best_assignment


def match_oracle_affine(source: np.ndarray, warped: np.ndarray,
                        transform: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Hungarian matching after applying the TRUE warp; the solvability ceiling.

    Holds the transform fixed at the value that generated the pair, isolating
    the residual per-cell ambiguity (jitter vs. inter-cell spacing) from the
    transform-recovery problem.
    """
    return match_hungarian(source @ transform + offset, warped,
                           transform, offset)


MATCHERS = {
    "nearest_neighbour": match_nearest_neighbour,
    "mutual_nn": match_mutual_nn,
    "hungarian": match_hungarian,
    "registration": match_registration,
    "whitened_global": match_whitened_global,
    "family_grid_global": match_family_grid_global,
    "oracle_affine": match_oracle_affine,
    "random": match_random,
}


# ─── Evaluation ──────────────────────────────────────────────────────────────

def score_assignment(predicted: np.ndarray, source_labels: np.ndarray,
                     warped_labels: np.ndarray) -> float:
    """Fraction of surviving source cells correctly matched to their identity.

    A source cell survives if its label still appears in the warped view (i.e.
    was not dropped). The predicted match is correct if the warped cell it points
    to carries the same label.
    """
    label_to_warped = {int(l): k for k, l in enumerate(warped_labels)}
    correct = total = 0
    for i, label in enumerate(source_labels):
        if int(label) not in label_to_warped:
            continue
        total += 1
        j = predicted[i]
        if j >= 0 and int(warped_labels[j]) == int(label):
            correct += 1
    return correct / total if total else float("nan")


def registration_diagnostics(scores: list[float]) -> dict[str, float]:
    """Bimodality diagnostics for the initialization-limited ICP matcher.

    ``converged`` counts runs at >= 0.95 accuracy (ICP found the correct
    basin); ``stuck`` counts runs at < 0.50 (ICP locked into a wrong local
    optimum from the raw Hungarian initialization).
    """
    scores_array = np.asarray(scores)
    return {
        "converged_fraction_0.95": float((scores_array >= 0.95).mean()),
        "stuck_fraction_0.50": float((scores_array < 0.50).mean()),
    }


def evaluate() -> dict:
    """Run all matchers across all distortion configs on the collected frames."""
    np.random.seed(SEED)
    frames = collect_frames()
    if not frames:
        raise RuntimeError("no frames found; check DATA_ROOT and EXPERIMENTS")

    results: dict[str, dict[str, dict | None]] = {c: {} for c in CONFIGS}
    for config in CONFIGS:
        for matcher_name, matcher in MATCHERS.items():
            if config in JITTER_ONLY_CONFIGS and matcher_name in GLOBAL_MATCHERS:
                results[config][matcher_name] = None
                continue
            scores = []
            for coords in frames:
                labels = np.arange(1, len(coords) + 1)
                for _ in range(N_REPS):
                    warped, warped_labels, transform, offset = distort(
                        coords, labels, config)
                    if len(warped) == 0:
                        continue
                    predicted = matcher(coords, warped, transform, offset)
                    scores.append(score_assignment(predicted, labels, warped_labels))
            entry: dict = {
                "mean_accuracy": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
            if matcher_name == "registration":
                entry.update(registration_diagnostics(scores))
            results[config][matcher_name] = entry

    return {
        "seed": SEED,
        "n_reps": N_REPS,
        "n_frames": len(frames),
        "cells_per_frame": {
            "min": int(min(len(f) for f in frames)),
            "max": int(max(len(f) for f in frames)),
            "mean": float(np.mean([len(f) for f in frames])),
        },
        "min_cells": MIN_CELLS,
        "experiments": [f"{c}/{e}" for c, e in EXPERIMENTS],
        "matcher_coverage_note": (
            "whitened_global and family_grid_global are null for jitter4 and "
            "jitter8: those configs apply no global transform beyond the "
            "identity, so global-warp solvers reduce to the hungarian matcher "
            "there."
        ),
        "matcher_notes": {
            "registration": (
                "single-start ICP; initialization-limited lower bound; see "
                "converged/stuck diagnostics"
            ),
            "family_grid_global": (
                "initialization-free; exploits the pretext's own warp family "
                "(rotation x independent x/y scale)"
            ),
            "oracle_affine": (
                "hungarian matching with the true warp applied; the "
                "jitter-limited ceiling of coordinate-only matching"
            ),
        },
        "configs": results,
    }


def main() -> int:
    """Run the analysis, write JSON, print the table."""
    result = evaluate()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "pretext_coordinate_solvability.json"
    out.write_text(json.dumps(result, indent=2))

    header = f"{'config':<16}" + "".join(f"{m:>20}" for m in MATCHERS)
    print(f"frames={result['n_frames']}  reps={result['n_reps']}  seed={result['seed']}")
    print(header)
    for config in CONFIGS:
        row = f"{config:<16}"
        for matcher_name in MATCHERS:
            entry = result["configs"][config][matcher_name]
            value = "skipped" if entry is None else f"{entry['mean_accuracy']:>20.3f}"
            row += f"{value:>20}"
        print(row)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
