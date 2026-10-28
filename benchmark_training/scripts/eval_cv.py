#!/usr/bin/env python3
"""Evaluate a CV-trained Trackastra checkpoint on its held-out condition.

Used by run_cv.slurm and run_eval_multiseed.slurm. Finds all experiment
subdirectories under the held-out condition and runs TRA/AOGM metrics on each.

Soundness features (SPEC 0039; the K-sweep eval produced byte-identical
results across independently trained checkpoints, so the eval path must make
wrong-checkpoint and leaked-era evaluations impossible or at least loudly
recorded):

* **Era guard**: for ``--scope heldout`` (the default), the run's
  ``train_config.yaml`` must NOT contain the evaluated condition in its
  ``input_train``/``input_val`` paths — otherwise the checkpoint has seen the
  data it is being evaluated on. Override only with ``--allow-leaked-era``
  (for deliberately re-measuring the historical leaked campaign). For
  ``--scope insample`` (the K-sweep in-training conditions) the guard is
  skipped and the membership recorded in the provenance sidecar.
* **Pre-flight**: the run dir must contain ``model.pt``.
* **Provenance sidecar**: every invocation writes
  ``<output>.provenance.json`` recording which checkpoint (path + md5), which
  parameters and which library versions produced the numbers. The results
  JSON format (``{experiment: {TRA, AOGM}}``) is unchanged.
* **CTC decomposition** (2026-10-28): every result also records the raw CTC
  error counts (``decomposition``), the derived ``AOGM_A`` (association) and
  ``AOGM_D`` (detection) terms, and the GT edge count — the convention used
  by the injection table (``eval_etultra.py``: AOGM = AOGM_A + AOGM_D).

Usage:
    python eval_cv.py --run-dir <run_dir> --heldout <cond> --data-root <root> \
        --output <results.json>
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Owner decision D-11 (OWNER_DECISIONS_2026-10-08.md): the deliberately
# bad-flagged vanvliet experiment is excluded from every evaluation scope, in
# both the K-sweep and CV protocols, so all reported columns stay like-for-like.
# Mirrors the EXCLUDED_SEQUENCES constant in
# report/scripts/aggregate_multiseed_eval.py; the CV trpL fold contains this
# sequence, so the exclusion must also happen here for summary.json to match.
EXCLUDED_SEQUENCES = ("150303-08_BAD",)


def md5_of_file(path: Path) -> str:
    """Return the md5 hex digest of a file, read in 1 MiB chunks.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_trackastra_git_commit(run_dir: Path) -> str | None:
    """Read the ``git_commit`` field from the run's train_config.yaml.

    Parameters
    ----------
    run_dir : Path
        Trackastra run directory containing train_config.yaml.

    Returns
    -------
    str | None
        The commit hash, or None when the file or field is absent.
    """
    config_path = run_dir / "train_config.yaml"
    if not config_path.is_file():
        return None
    for line in config_path.read_text(errors="replace").splitlines():
        if line.startswith("git_commit:"):
            return line.split(":", 1)[1].strip()
    return None


def find_heldout_condition_in_config(run_dir: Path, heldout: str) -> list[str]:
    """Return the config lines that mention the held-out condition.

    Only meaningful lines are considered (input path lists and their
    continuation lines); the distinctive condition tokens make false positives
    from unrelated fields unlikely.

    Parameters
    ----------
    run_dir : Path
        Trackastra run directory containing train_config.yaml.
    heldout : str
        Held-out condition token (e.g. ``"trpL"``).

    Returns
    -------
    list[str]
        The offending lines (empty when the condition does not appear).
    """
    config_path = run_dir / "train_config.yaml"
    if not config_path.is_file():
        return []
    offending = []
    for line in config_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if heldout in stripped and stripped.startswith(("-", "input_train", "input_val")):
            offending.append(stripped)
    return offending


def library_versions() -> dict[str, str | None]:
    """Return the library versions relevant to reproducibility.

    Returns
    -------
    dict[str, str | None]
        python/torch/numpy versions; torch and traccuracy fall back to None
        when not importable at this point of the invocation.
    """
    versions: dict[str, str | None] = {"python": platform.python_version()}
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = None
    try:
        from importlib.metadata import version as dist_version

        versions["traccuracy"] = dist_version("traccuracy")
    except Exception:
        versions["traccuracy"] = None
    versions["numpy"] = np.__version__
    return versions


def build_provenance(
    run_dir: Path,
    heldout: str,
    scope: str,
    condition_in_training: bool,
    data_root: str,
    max_distance: int,
    device: str,
    experiment_count: int,
    excluded_present: list[str],
    failed_count: int,
    started_at: datetime,
    finished_at: datetime,
    output_path: Path,
    summary_path: Path,
    script_path: Path,
) -> dict:
    """Build the provenance record for one eval invocation.

    Parameters
    ----------
    run_dir : Path
        The evaluated Trackastra run directory.
    heldout : str
        Evaluated condition name.
    scope : str
        ``"heldout"`` or ``"insample"`` — whether the condition is claimed to
        be unseen by the checkpoint.
    condition_in_training : bool
        Whether the condition appears in the run's train_config.yaml paths.
    data_root : str
        Data root passed on the command line.
    max_distance : int
        Maximum link distance passed to the tracker.
    device : str
        Compute device used.
    experiment_count : int
        Number of experiments found under the condition.
    excluded_present : list[str]
        Owner-excluded sequences (EXCLUDED_SEQUENCES) that were present under
        the condition and dropped from evaluation.
    failed_count : int
        Number of experiments that failed (tracking or metrics).
    started_at, finished_at : datetime
        UTC timestamps of the invocation.
    output_path, summary_path : Path
        The files this invocation writes.
    script_path : Path
        Path of this script (hashed into the record).

    Returns
    -------
    dict
        JSON-serializable provenance record.
    """
    checkpoint_records = []
    for path in sorted(run_dir.glob("*.pt")) + sorted((run_dir / "checkpoints").glob("*")):
        if path.is_file():
            checkpoint_records.append({"path": str(path), "md5": md5_of_file(path)})
    return {
        "run_dir": str(run_dir),
        "checkpoint_files": checkpoint_records,
        "trackastra_git_commit": read_trackastra_git_commit(run_dir),
        "heldout_condition": heldout,
        "scope": scope,
        "condition_in_training": condition_in_training,
        "data_root": data_root,
        "max_distance": max_distance,
        "mode": "greedy",
        "device": device,
        "n_experiments": experiment_count,
        "excluded_sequences": list(EXCLUDED_SEQUENCES),
        "excluded_present": excluded_present,
        "n_failed": failed_count,
        "started_utc": started_at.isoformat(),
        "finished_utc": finished_at.isoformat(),
        "outputs": [str(output_path), str(summary_path)],
        "eval_script": {"path": str(script_path), "md5": md5_of_file(script_path)},
        "library_versions": library_versions(),
        "era_guard_overridden": era_guard_overridden,
    }


# Module-level flag set by main() so the provenance record can report whether
# the era guard was bypassed for this invocation.
era_guard_overridden = False


def main():
    """Run the guarded evaluation and write results plus provenance.

    Returns
    -------
    None
        Exits with status 1 on pre-flight failures (missing run dir or model,
        era-guard violation, missing held-out directory).
    """
    global era_guard_overridden

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--heldout", type=str, required=True,
                        help="Name of the held-out condition (e.g. rpsM)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory containing all conditions")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-distance", type=int, default=50,
                        help="Max distance for candidate graph edges")
    parser.add_argument("--allow-leaked-era", action="store_true",
                        help="Evaluate even when train_config.yaml contains the "
                             "held-out condition (deliberate re-measurement of "
                             "the historical leaked campaign; recorded in the "
                             "provenance sidecar)")
    parser.add_argument("--scope", choices=["heldout", "insample"], default="heldout",
                        help="heldout: the condition must NOT appear in the run's "
                             "training config (era guard active — the CV/K-sweep "
                             "held-out columns). insample: the condition is "
                             "expected to be in training (K-sweep in-sample "
                             "columns); the era guard is skipped and the "
                             "membership is recorded in the provenance sidecar")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    script_path = Path(__file__).resolve()
    started_at = datetime.now(timezone.utc)

    # ── Pre-flight ────────────────────────────────────────────────────
    if not run_dir.is_dir():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)
    if not (run_dir / "model.pt").is_file():
        print(f"ERROR: {run_dir} contains no model.pt — refusing to evaluate "
              f"an incomplete or wrongly-bound run dir")
        sys.exit(1)

    # ── Era guard (held-out scope only — see --scope) ─────────────────
    offending_lines = find_heldout_condition_in_config(run_dir, args.heldout)
    condition_in_training = bool(offending_lines)
    if args.scope == "heldout" and offending_lines and not args.allow_leaked_era:
        print(f"ERROR: era guard — train_config.yaml of {run_dir} contains the "
              f"held-out condition {args.heldout!r} in its training paths; this "
              f"checkpoint has SEEN the data it would be evaluated on:")
        for line in offending_lines[:5]:
            print(f"    {line}")
        print("Re-run with --allow-leaked-era to override (recorded in provenance).")
        sys.exit(1)
    era_guard_overridden = bool(offending_lines)

    heldout_dir = data_root / args.heldout
    if not heldout_dir.exists():
        print(f"ERROR: Held-out directory not found: {heldout_dir}")
        sys.exit(1)

    # Find all experiment subdirectories, dropping owner-excluded sequences
    # (D-11) so the CV summary uses the same sequence population as every
    # other table. `excluded_present` records what was dropped for provenance.
    all_exp_dirs = sorted(d for d in heldout_dir.iterdir()
                          if d.is_dir() and (d / "TRA").exists())
    exp_dirs = [d for d in all_exp_dirs if d.name not in EXCLUDED_SEQUENCES]
    excluded_present = [d.name for d in all_exp_dirs
                        if d.name in EXCLUDED_SEQUENCES]

    print(f"Held-out: {args.heldout}")
    print(f"Experiments: {len(exp_dirs)}")
    if excluded_present:
        print(f"Excluded (D-11): {', '.join(excluded_present)}")
    for d in exp_dirs:
        print(f"  {d.name}")
    print(f"Model: {run_dir}")

    # Lazy imports
    from trackastra.model import Trackastra
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics

    print(f"\nLoading model … ", end="", flush=True)
    t0 = time.time()
    model = Trackastra.from_folder(run_dir, device=args.device)
    print(f"done ({time.time() - t0:.1f}s)")

    results = {}
    for exp_dir in exp_dirs:
        img_dir = exp_dir / "img"
        gt_dir = exp_dir / "TRA"

        print(f"\n  [{exp_dir.name}] tracking … ", end="", flush=True)
        t0 = time.time()
        try:
            track_graph, masks = model.track_from_disk(
                imgs_path=img_dir,
                masks_path=gt_dir,
                mode="greedy",
                max_distance=args.max_distance,
            )
            print(f"done ({time.time() - t0:.1f}s) "
                  f"graph: {track_graph.number_of_nodes()} nodes, "
                  f"{track_graph.number_of_edges()} edges")
        except Exception as exc:
            print(f"FAILED: {exc}")
            results[exp_dir.name] = {"error": str(exc)}
            continue

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from trackastra.tracking.utils import graph_to_ctc
            graph_to_ctc(track_graph, masks, outdir=Path(tmpdir))

            print(f"  [{exp_dir.name}] metrics … ", end="", flush=True)
            t0 = time.time()
            try:
                gt_data = load_ctc_data(str(gt_dir))
                pred_data = load_ctc_data(str(tmpdir))

                ctc_result, _ = run_metrics(
                    gt_data, pred_data, CTCMatcher(), [CTCMetrics()]
                )

                # CTC error decomposition (CTCMetrics, TRA weights):
                #   AOGM   = AOGM_A + AOGM_D
                #   AOGM_A = fp_edges + 1.5*fn_edges + ws_edges   (association/linking)
                #   AOGM_D = fp_nodes + 10*fn_nodes + 5*ns_nodes  (detection)
                # Same convention as eval_etultra.py:126-129 (injection table).
                metrics = ctc_result[0]["results"]
                decomp = {k: metrics.get(k) for k in (
                    "ns_nodes", "fp_nodes", "fn_nodes",
                    "fp_edges", "fn_edges", "ws_edges")}
                aogm_a = ((decomp["fp_edges"] or 0)
                          + 1.5 * (decomp["fn_edges"] or 0)
                          + (decomp["ws_edges"] or 0))
                aogm_d = ((decomp["fp_nodes"] or 0)
                          + 10 * (decomp["fn_nodes"] or 0)
                          + 5 * (decomp["ns_nodes"] or 0))
                gt_graph = getattr(gt_data, "graph", None)
                gt_edges = (int(gt_graph.number_of_edges())
                            if gt_graph is not None
                            and hasattr(gt_graph, "number_of_edges") else None)

                tra = metrics.get("TRA", None)
                aogm = metrics.get("AOGM", None)

                print(f"done ({time.time() - t0:.1f}s) "
                      f"TRA={tra:.4f}  AOGM={aogm:.1f}  "
                      f"AOGM_A={aogm_a:.1f}  AOGM_D={aogm_d:.1f}")

                results[exp_dir.name] = {
                    "TRA": float(tra) if tra is not None else None,
                    "AOGM": float(aogm) if aogm is not None else None,
                    "AOGM_A": float(aogm_a) if aogm is not None else None,
                    "AOGM_D": float(aogm_d) if aogm is not None else None,
                    "gt_edge_count": gt_edges,
                    "decomposition": decomp,
                }
            except Exception as exc:
                print(f"FAILED: {exc}")
                results[exp_dir.name] = {"error": str(exc)}

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    valid = [r for r in results.values() if "error" not in r]
    if valid:
        tra_vals = [r["TRA"] for r in valid if r.get("TRA") is not None]
        aogm_vals = [r["AOGM"] for r in valid if r.get("AOGM") is not None]

        summary = {
            "n_sequences": len(valid),
            "n_failed": len(results) - len(valid),
            "mean_TRA": float(np.mean(tra_vals)) if tra_vals else None,
            "std_TRA": float(np.std(tra_vals)) if tra_vals else None,
            "mean_AOGM": float(np.mean(aogm_vals)) if aogm_vals else None,
            "std_AOGM": float(np.std(aogm_vals)) if aogm_vals else None,
            "mean_AOGM_A": float(np.mean(a_vals)) if (a_vals := [
                r["AOGM_A"] for r in valid if r.get("AOGM_A") is not None]) else None,
            "std_AOGM_A": float(np.std(a_vals)) if a_vals else None,
            "mean_AOGM_D": float(np.mean(d_vals)) if (d_vals := [
                r["AOGM_D"] for r in valid if r.get("AOGM_D") is not None]) else None,
            "std_AOGM_D": float(np.std(d_vals)) if d_vals else None,
        }

        print(f"\n{'=' * 60}")
        print(f"  SUMMARY ({len(valid)}/{len(results)} experiments)")
        print(f"{'=' * 60}")
        print(f"  mean TRA:  {summary['mean_TRA']:.4f} ± {summary['std_TRA']:.4f}")
        print(f"  mean AOGM: {summary['mean_AOGM']:.1f} ± {summary['std_AOGM']:.1f}")
        print(f"  mean AOGM_A (association): "
              f"{summary['mean_AOGM_A']:.1f} ± {summary['std_AOGM_A']:.1f}")
        print(f"  mean AOGM_D (detection):   "
              f"{summary['mean_AOGM_D']:.1f} ± {summary['std_AOGM_D']:.1f}")

        # Save summary alongside results
        summary_path = output_path.with_name("summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {output_path}  +  {summary_path}")
    else:
        summary_path = output_path.with_name("summary.json")

    # Provenance sidecar (written for every invocation, success or partial)
    finished_at = datetime.now(timezone.utc)
    failed_count = sum(1 for r in results.values() if "error" in r)
    provenance = build_provenance(
        run_dir=run_dir,
        heldout=args.heldout,
        scope=args.scope,
        condition_in_training=condition_in_training,
        data_root=str(data_root),
        max_distance=args.max_distance,
        device=args.device,
        experiment_count=len(exp_dirs),
        excluded_present=excluded_present,
        failed_count=failed_count,
        started_at=started_at,
        finished_at=finished_at,
        output_path=output_path,
        summary_path=summary_path,
        script_path=script_path,
    )
    provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
