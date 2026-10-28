#!/usr/bin/env python3
"""Bug-exclusion probe for the trpL pretext collapse (2026-10-28).

Question (owner, 2026-10-28): before the report's pretext section is rewritten
on top of the sample-efficiency verdict (``pretext_sample_efficiency_test.py``,
artifact ``results/runs/pretext_sample_efficiency_test/result.json``), EXCLUDE
that a bug in our training path causes the pretext training to fail (trpL pool:
SSL loss = all-negative entropy floor, all three arms 0.500).

Red/green design (thresholds fixed before running, deterministic, seed 42):

- RED = any input-corruption signal fires:
    (a) geometry: BCE targets are built by label equality
        (``bce_assoc_loss`` -> ``_build_target_matrix``), so target corruption
        would require a label/coordinate misalignment. Check: for every
        positive pair (i, j), ``coords_warped[j]`` must equal
        ``coords_clean[i] @ T`` up to jitter (sigma = 4 per axis; residual is
        Rayleigh with mean ~5.0 px, max over ~2k pairs ~17 px; a misalignment
        bug would give ~100+ px residuals).
    (b) appearance: for easy positives (displacement <= 24 px), the Pearson
        correlation between the clean patch and the warped-image patch must
        exceed 0.5, with a gap > 0.1 over the unmatched-pair baseline.
    (c) alignment: clean/warped array lengths must be internally consistent
        (asserted per frame).
- GREEN = all checks pass AND the differential loop holds: the identical code
  (same model, seed, hyperparameters) learns on the rpsM pool while collapsing
  on the trpL pool. Pool identity is cross-checked against the decisive
  artifact (5 train / 8 val pairs, 60 trpL pool frames).

Provenance: the probe reuses the decisive run's own pool-building code
(``pretext_sample_efficiency_test.build_downstream_pairs`` /
``build_ssl_pool``) with the DINO extractor wrapped to also record the raw
patches, so the audited inputs are exactly the decisive run's training data.
Output mappings (true-identity argmax accuracy, partner-vs-best-off-target
logit gap) follow SPEC 0048.

Verdict: BUG EXCLUDED. The collapse is a data-conditioned optimization
degeneracy, not a harness bug.

Reproduce (from the repository root)::

    benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_collapse_bug_exclusion.py

Outputs ``benchmark_ssl/results/pretext_collapse_bug_exclusion.json`` and
prints the verdict. Runs on the local GPU in well under 30 minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from imageio.v3 import imread

BENCHMARK_SSL_ROOT = Path(__file__).resolve().parents[2]
if str(BENCHMARK_SSL_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SSL_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import src.mini_trackastra.end_to_end as e2e  # noqa: E402
import pretext_sample_efficiency_test as P  # noqa: E402

RESULTS_DIR = BENCHMARK_SSL_ROOT / "results"


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def entropy_floor(pos_frac: float) -> float:
    p = min(max(pos_frac, 1e-9), 1 - 1e-9)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


# Wrappers (installed in main): the DINO wrapper keeps the raw patches for the
# appearance checks; the distortion wrapper keeps the recorded transform for
# the geometry checks. Neither changes any random draw, so pool identity is
# preserved exactly.
_store: list[np.ndarray] = []
_dwt_records: list[dict] = []


def _install_instrumentation() -> None:
    real_dino = e2e.compute_dino_embs

    def rec_dino(patches):
        out = real_dino(patches)
        _store.append(np.asarray(patches, dtype=np.float32).copy())
        return out

    P.compute_dino_embs = rec_dino

    orig_dwt = P.distort_with_transform

    def rec_dwt(coords, labels, mode):
        warped, kept, T = orig_dwt(coords, labels, mode)
        _dwt_records.append({
            "T": np.asarray(T, dtype=np.float64).copy(),
            "n_src": int(len(coords)),
            "n_surv": int(len(warped)),
        })
        return warped, kept, T

    P.distort_with_transform = rec_dwt


def _img_path_for(mask_path: str) -> Path:
    m = Path(mask_path)
    fi = int(m.stem.replace("man_track", ""))
    return m.parent.parent / "img" / f"t{fi:06d}.tif"


def pool_metrics(pool: list[dict], records: list[dict]) -> tuple[list[dict], dict]:
    """Per-frame input-corruption metrics; RED if any check fails."""
    frames = []
    worst = {"max_pred_err": 0.0, "min_easy_corr": 1.0, "corr_gap": 1e9}
    for k, (f, rec) in enumerate(zip(pool, records)):
        lc = np.asarray(f["labels_clean"])
        lw = np.asarray(f["labels_warped"])
        cc = np.asarray(f["coords_clean"], dtype=np.float64)
        cw = np.asarray(f["coords_warped"], dtype=np.float64)
        n1 = len(lc)
        # Pool entries are truncated to n = min(n_src, n_surv); the store keeps
        # the untruncated call outputs, so slice them to the pool's length.
        pc = _store[2 * k][:n1]
        pw = _store[2 * k + 1][:n1]
        T = rec["T"]
        assert len(cc) == n1 == len(pc), f"frame {k}: clean view misaligned"
        assert len(lw) == len(cw) == len(pw), f"frame {k}: warped view misaligned"
        assert rec["n_surv"] == len(lw), f"frame {k}: dropout bookkeeping mismatch"

        pos = lc[:, None] == lw[None, :]
        pos_frac = float(pos.mean())
        ij = np.argwhere(pos)

        # (a) geometry: surviving warped coord must be T * its clean coord
        # (up to jitter).
        pred = cc[ij[:, 0]] @ T
        err = (np.linalg.norm(cw[ij[:, 1]] - pred, axis=1)
               if len(ij) else np.array([0.0]))
        # (b) appearance: matched-patch correlation vs unmatched baseline,
        # stratified by displacement.
        if len(ij):
            disp = np.linalg.norm(cw[ij[:, 1]] - cc[ij[:, 0]], axis=1)
            corr = np.array([pearson(pc[i], pw[j]) for i, j in ij])
            easy = disp <= 24.0
            easy_corr = float(corr[easy].mean()) if easy.any() else None
            rng = np.random.default_rng(0)
            jj = rng.permutation(len(ij))
            base = np.array([pearson(pc[i], pw[ij[jj[m], 1]])
                             for m, i in enumerate(ij[:, 0])])
            base_corr = float(base.mean())
        else:
            disp = corr = np.array([])
            easy_corr, base_corr = None, None

        # (c) out-of-frame exposure of view 2 (border/black-patch channel).
        img = imread(_img_path_for(f["mask_path"]))
        h, w = img.shape[-2:]
        oof = float(((cw[:, 0] < 0) | (cw[:, 0] >= h)
                     | (cw[:, 1] < 0) | (cw[:, 1] >= w)).mean())

        frames.append({
            "mask": Path(f["mask_path"]).name,
            "n_clean": int(n1), "n_warped": int(len(lw)),
            "pos_frac": pos_frac,
            "entropy_floor": entropy_floor(pos_frac),
            "n_pos": int(pos.sum()),
            "coord_pred_err_max": float(err.max()),
            "coord_pred_err_mean": float(err.mean()),
            "disp_mean": float(disp.mean()) if len(disp) else None,
            "corr_easy_mean": easy_corr,
            "corr_all_mean": float(corr.mean()) if len(corr) else None,
            "corr_unmatched_base": base_corr,
            "warped_out_of_frame_frac": oof,
        })
        worst["max_pred_err"] = max(worst["max_pred_err"], float(err.max()))
        if easy_corr is not None:
            worst["min_easy_corr"] = min(worst["min_easy_corr"], easy_corr)
            worst["corr_gap"] = min(worst["corr_gap"], easy_corr - base_corr)
    agg = {
        "frames": len(frames),
        "cells_per_frame_median": float(np.median([fr["n_clean"] for fr in frames])),
        "positive_fraction_mean": float(np.mean([fr["pos_frac"] for fr in frames])),
        "entropy_floor_mean": float(np.mean([fr["entropy_floor"] for fr in frames])),
        "coord_pred_err_max_global": worst["max_pred_err"],
        "coord_pred_err_mean_global": float(np.mean(
            [fr["coord_pred_err_mean"] for fr in frames])),
        "corr_easy_min": worst["min_easy_corr"],
        "corr_gap_min": worst["corr_gap"],
        "warped_out_of_frame_frac_mean": float(np.mean(
            [fr["warped_out_of_frame_frac"] for fr in frames])),
    }
    return frames, agg


def output_mappings(ssl: dict, pool: list[dict], k: int = 12) -> list[dict]:
    """Per-frame true-identity argmax accuracy + partner-vs-best-off-target gap."""
    model, pe = ssl["model"], ssl["pe"]
    model.eval()
    rows = []
    with torch.no_grad():
        for f in pool[:k]:
            f1 = torch.from_numpy(np.asarray(f["features_clean"])).float().to(e2e.device)
            f2 = torch.from_numpy(np.asarray(f["features_warped"])).float().to(e2e.device)
            c1 = torch.from_numpy(np.asarray(f["coords_clean"])).float() \
                .unsqueeze(0).to(e2e.device)
            c2 = torch.from_numpy(np.asarray(f["coords_warped"])).float() \
                .unsqueeze(0).to(e2e.device)
            logits = model(f1, pe(c1).squeeze(0), f2, pe(c2).squeeze(0))
            pos = (np.asarray(f["labels_clean"])[:, None]
                   == np.asarray(f["labels_warped"])[None, :])
            li = logits.cpu().numpy()
            has_pos = pos.any(axis=1)
            if has_pos.sum() == 0:
                continue
            am = li.argmax(axis=1)
            hit = pos[np.arange(len(am)), am]
            acc = float(hit[has_pos].mean())
            gaps = []
            for i in np.where(has_pos)[0]:
                jstar = np.where(pos[i])[0]
                off = [k2 for k2 in range(li.shape[1]) if not pos[i, k2]]
                if off:
                    gaps.append(float(li[i, jstar].max() - li[i, off].max()))
            rows.append({
                "mask": Path(f["mask_path"]).name,
                "n": int(pos.shape[0]),
                "true_identity_argmax_acc": acc,
                "partner_minus_best_offtarget_mean": float(np.mean(gaps)) if gaps else None,
            })
    return rows


def main() -> int:
    torch.manual_seed(e2e.SEED)
    np.random.seed(e2e.SEED)
    _install_instrumentation()
    print(f"device={e2e.device}")

    # 1) downstream pairs — must reproduce the decisive run's 5 train / 8 val
    train_pairs, val_pairs, excluded_masks = P.build_downstream_pairs()
    print(f"downstream: train={len(train_pairs)} val={len(val_pairs)} "
          f"(decisive artifact: 5 / 8)")
    assert len(train_pairs) == 5, "train-pair selection diverged from decisive run"
    assert len(val_pairs) == 8, "val-pair selection diverged from decisive run"

    # 2) trpL pool — must reproduce n_pool=60
    _store.clear()
    _dwt_records.clear()
    pool_trpL = P.build_ssl_pool(excluded_masks)
    rec_trpL = list(_dwt_records)[-len(pool_trpL):]
    print(f"trpL pool: {len(pool_trpL)} frames (decisive artifact: 60)")
    assert len(pool_trpL) == 60, "pool selection diverged from decisive run"

    frames_trpL, agg_trpL = pool_metrics(pool_trpL, rec_trpL)
    print("trpL input metrics:", json.dumps(agg_trpL, indent=2))

    # 3) differential SSL: same code, trpL vs rpsM pool
    print("SSL on trpL pool (300 steps)...")
    ssl_trpL = P.train_ssl(pool_trpL, 300)
    print(f"  trpL SSL final loss {ssl_trpL['losses'][-1]:.4f} "
          f"(floor ~{agg_trpL['entropy_floor_mean']:.4f})")
    om_trpL = output_mappings(ssl_trpL, pool_trpL)
    acc_trpL = float(np.mean([r["true_identity_argmax_acc"] for r in om_trpL]))
    print(f"  trpL true-identity argmax acc: {acc_trpL:.3f}")

    _store.clear()
    _dwt_records.clear()
    P.ALL_CONDITIONS = ["rpsM"]
    pool_rpsM = P.build_ssl_pool(set())
    rec_rpsM = list(_dwt_records)[-len(pool_rpsM):]
    print(f"rpsM pool: {len(pool_rpsM)} frames")
    frames_rpsM, agg_rpsM = pool_metrics(pool_rpsM, rec_rpsM)
    print("rpsM input metrics:", json.dumps(agg_rpsM, indent=2))

    print("SSL on rpsM pool (300 steps)...")
    ssl_rpsM = P.train_ssl(pool_rpsM, 300)
    print(f"  rpsM SSL final loss {ssl_rpsM['losses'][-1]:.4f} "
          f"(floor ~{agg_rpsM['entropy_floor_mean']:.4f})")
    om_rpsM = output_mappings(ssl_rpsM, pool_rpsM)
    acc_rpsM = float(np.mean([r["true_identity_argmax_acc"] for r in om_rpsM]))
    print(f"  rpsM true-identity argmax acc: {acc_rpsM:.3f}")

    # 4) pre-registered verdict
    # Geometry: jitter sigma=4 per axis -> positive-pair residual is Rayleigh
    # (mean ~5.0, max over ~2k pairs ~17); corruption would give ~100+ px.
    geo_ok = (agg_trpL["coord_pred_err_mean_global"] <= 8.0
              and agg_trpL["coord_pred_err_max_global"] <= 20.0)
    app_ok = (agg_trpL["corr_easy_min"] is not None
              and agg_trpL["corr_easy_min"] > 0.5
              and agg_trpL["corr_gap_min"] > 0.1)
    learned_rpsM = acc_rpsM > 0.5
    collapsed_trpL = acc_trpL < 0.2

    verdict = {
        "geometry_targets_clean": bool(geo_ok),
        "appearance_clean": bool(app_ok),
        "rpsM_learns_same_code": bool(learned_rpsM),
        "trpL_collapses_same_code": bool(collapsed_trpL),
        "BUG_EXCLUDED": bool(geo_ok and app_ok and learned_rpsM and collapsed_trpL),
    }
    out = {
        "verdict": verdict,
        "provenance": {
            "downstream_train_val": [len(train_pairs), len(val_pairs)],
            "n_pool_trpL": len(pool_trpL),
            "decisive_artifact": "results/runs/pretext_sample_efficiency_test/result.json",
            "decisive_ssl_final_loss": 0.1605545369287332,
            "ssl_loss_trpL_this_run": ssl_trpL["losses"][-1],
        },
        "agg_trpL": agg_trpL, "agg_rpsM": agg_rpsM,
        "frames_trpL": frames_trpL, "frames_rpsM": frames_rpsM,
        "ssl_loss_trpL": ssl_trpL["losses"][-1],
        "ssl_loss_rpsM": ssl_rpsM["losses"][-1],
        "argmax_acc_trpL": acc_trpL, "argmax_acc_rpsM": acc_rpsM,
        "output_mappings_trpL": om_trpL, "output_mappings_rpsM": om_rpsM,
    }
    out_path = RESULTS_DIR / "pretext_collapse_bug_exclusion.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")
    print("\n=== PROBE VERDICT ===")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
