#!/usr/bin/env python3
"""Error-level autopsy for one fold of the CV spatial-cutoff ablation.

Question (owner, 2026-10-28): the cib fold's DenseFlash AOGM is 35.25 vs the
masked baseline's 2.0 (seed 42) — a ~17x relative jump on a tiny baseline.
WHICH links did it get wrong?

Re-runs the exact eval path of ``eval_cv.py`` (GT detections via
``track_from_disk``, greedy, candidate ``max_distance=50``, CTCMatcher +
CTCMetrics, traccuracy) and then decomposes the matched graphs into the
concrete CTC error flags — counts AND locations:

AOGM weights (CTCMetrics): node NS=5, FN=10, FP=1; edge FP=1, FN=1.5, WS=1.
With GT detections as input, node errors are ~0, so AOGM is dominated by edge
errors: missing GT edges (weight 1.5), added edges (1), wrong-semantic (1).

For every flagged edge the script reports the frame, the GT/segmentation node
ids, both centroids (px), and the displacement — enough to classify a flipped
link (long-distance? competitor cell? division frame?) by eye or by a histogram.

Usage (needs the run dirs with model.pt; cluster where the cv500 runs live,
or a local mirror)::

    python benchmark_training/scripts/analysis/cv_fold_autopsy.py \
        --run-dir <TRK>/runs/2026-10-23_*_cv500_masked_fold4_cib_s42 \
        --run-dir <TRK>/runs/2026-10-23_*_cv500_dense_flash_fold4_cib_s42 \
        --data-root /data/cat/ws/mawe985g-data/data/celltracking/vanvliet \
        --condition cib --output cv_fold4_cib_autopsy.json

Multiple ``--run-dir`` are allowed (e.g. both seeds) and compared per sequence
against the first masked run given.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def centroids_for_frame(mask: np.ndarray) -> dict[int, tuple[float, float]]:
    """label -> (y, x) centroid for one labelled frame."""
    labels = np.unique(mask)
    labels = labels[labels != 0]
    out: dict[int, tuple[float, float]] = {}
    for lab in labels:
        ys, xs = np.nonzero(mask == lab)
        out[int(lab)] = (float(ys.mean()), float(xs.mean()))
    return out


def node_attrs(node) -> tuple[int, int]:
    """traccuracy CTC node ids are (t, label) tuples."""
    t, lab = node
    return int(t), int(lab)


def edge_report(edges, centroids_by_frame, kind: str) -> list[dict]:
    rows = []
    for u, v in edges:
        (t1, l1), (t2, l2) = node_attrs(u), node_attrs(v)
        c1 = centroids_by_frame.get(t1, {}).get(l1)
        c2 = centroids_by_frame.get(t2, {}).get(l2)
        disp = (float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                if c1 and c2 else None)
        rows.append({
            "kind": kind, "t": t1, "src_label": l1, "dst_label": l2,
            "src_centroid": c1, "dst_centroid": c2, "displacement_px": disp,
        })
    return rows


def autopsy_run(run_dir: Path, data_root: Path, condition: str,
                device: str = "cuda") -> dict:
    from trackastra.model import Trackastra
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics
    from traccuracy.flags import EdgeFlag, NodeFlag
    from trackastra.tracking.utils import graph_to_ctc
    import tempfile
    from imageio.v3 import imread

    model = Trackastra.from_folder(run_dir, device=device)
    exp_dirs = sorted(d for d in (data_root / condition).iterdir()
                      if d.is_dir() and (d / "TRA").exists())
    out: dict = {}
    for exp_dir in exp_dirs:
        img_dir, gt_dir = exp_dir / "img", exp_dir / "TRA"
        track_graph, masks = model.track_from_disk(
            imgs_path=str(img_dir), masks_path=str(gt_dir),
            mode="greedy", max_distance=50,
        )
        with tempfile.TemporaryDirectory() as tmp:
            graph_to_ctc(track_graph, masks, outdir=Path(tmp))
            gt_data = load_ctc_data(str(gt_dir))
            pred_data = load_ctc_data(str(tmp))
        ctc_result, matched = run_metrics(
            gt_data, pred_data, CTCMatcher(), [CTCMetrics()]
        )
        res = ctc_result[0]["results"]

        # centroids per frame from the GT masks
        cent: dict[int, dict[int, tuple[float, float]]] = {}
        for f, m in enumerate(sorted(gt_dir.glob("man_track*.tif"))):
            cent[f] = centroids_for_frame(imread(m))

        pred_g, gt_g = matched.pred_graph, matched.gt_graph
        errors = []
        errors += edge_report(gt_g.get_edges_with_flag(EdgeFlag.CTC_FALSE_NEG),
                              cent, "missing_gt_edge")
        errors += edge_report(pred_g.get_edges_with_flag(EdgeFlag.CTC_FALSE_POS),
                              cent, "added_edge")
        errors += edge_report(pred_g.get_edges_with_flag(EdgeFlag.WRONG_SEMANTIC),
                              cent, "wrong_semantic_edge")
        ns_nodes = [node_attrs(n) for n in
                    pred_g.get_nodes_with_flag(NodeFlag.NON_SPLIT)]
        out[exp_dir.name] = {
            "TRA": res.get("TRA"), "AOGM": res.get("AOGM"),
            "counts": {
                "missing_gt_edges": sum(e["kind"] == "missing_gt_edge" for e in errors),
                "added_edges": sum(e["kind"] == "added_edge" for e in errors),
                "wrong_semantic_edges": sum(e["kind"] == "wrong_semantic_edge" for e in errors),
                "non_split_nodes": len(ns_nodes),
            },
            "errors": errors,
            "non_split_nodes": ns_nodes,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--condition", default="cib")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    results = {}
    for rd in args.run_dir:
        p = Path(rd)
        print(f"=== {p.name} ===")
        results[p.name] = autopsy_run(p, Path(args.data_root),
                                      args.condition, args.device)

    # compact cross-run comparison per sequence
    print("\n=== per-sequence error counts ===")
    seqs = sorted({s for r in results.values() for s in r})
    header = ["sequence"] + [f"{k.split('_')[-1] if k else k}:{k}" for k in results]
    print(" | ".join(["seq"] + list(results)))
    kinds = ["missing_gt_edges", "added_edges", "wrong_semantic_edges",
             "non_split_nodes"]
    for s in seqs:
        row = [s]
        for name, r in results.items():
            c = r.get(s, {}).get("counts", {})
            row.append(" ".join(f"{k.split('_')[0][:3]}{k.split('_')[-1][0]}={c.get(k)}"
                                for k in kinds))
        print(" | ".join(row))

    # displacement histogram of the flagged edges (the 'why' signature)
    print("\n=== flagged-edge displacement (px) by run ===")
    for name, r in results.items():
        disps = [e["displacement_px"] for s in r.values() for e in s["errors"]
                 if e["displacement_px"] is not None]
        if disps:
            q = np.percentile(disps, [50, 90, 100])
            print(f"{name}: n={len(disps)} median={q[0]:.1f} "
                  f"p90={q[1]:.1f} max={q[2]:.1f}")
        kinds_seen = Counter(e["kind"] for s in r.values() for e in s["errors"])
        print(f"  kinds: {dict(kinds_seen)}")

    Path(args.output).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
