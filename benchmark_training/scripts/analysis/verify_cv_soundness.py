#!/usr/bin/env python3
r"""Soundness verifier for the 6-fold LOCO-CV cutoff ablation (Table 6).

Why this script exists: the report's central claim (the spatial cutoff is
dispensable, ``tab:cv_results``) rests on the 12 CV run artifacts on the
cluster. The K-sweep eval pass was found to contain byte-identical, duplicated
per-run eval files, so the CV eval pass must be checked mechanically against
its raw artifacts, not by argument. This script performs the read-only checks
C1–C6 of ``docs/agent-state/specs/0038-cv-soundness-verifier/SPEC.md``:

  C1  run discovery       6 runs per arm, folds 0–5, name ↔ condition consistent
  C2  leakage             held-out condition absent from ``train_config.yaml``
  C3  checkpoint identity the 12 runs trained distinct weights
  C4  eval duplication    no ``cv_eval/`` file is shared between two runs
  C5  report provenance   stored per-fold means reproduce ``tab:cv_results``
  C6  timing sanity       eval artifacts are newer than the checkpoints

Usage::

    # cluster (default runs root is the one used by plot_cv_results.py)
    python benchmark_training/scripts/analysis/verify_cv_soundness.py

    # explicit runs root or local copy
    python benchmark_training/scripts/analysis/verify_cv_soundness.py \
        --runs-dir /path/to/runs

    # checkpoint file pattern can be adapted if trackastra's layout differs
    python benchmark_training/scripts/analysis/verify_cv_soundness.py \
        --checkpoint-glob "checkpoints/*.ckpt"

Exit status: 0 when no check FAILED, 1 otherwise (WARN does not fail).
The verifier is strictly read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The six vanvliet conditions, in fold order (must match run_cv.slurm).
CONDITIONS: tuple[str, ...] = ("rpsM", "recA", "pheA", "metA", "cib", "trpL")

# Arm name tokens in the run directory names written by run_cv.slurm
# (RUN_NAME="cv_${CONFIG}_fold${FOLD}_${HELDOUT_COND}", date-prefixed by
# train.py's outdir naming).
ARM_PATTERNS: dict[str, str] = {
    "masked": "*cv_masked_fold*",
    "dense_flash": "*cv_dense_flash_fold*",
}

# Per-fold values published in report/appendix.tex tab:cv_results. Column
# mapping verified during the table-layout investigation: the tuple is
# (condition, masked TRA, flash TRA, masked AOGM, flash AOGM).
REPORT_TABLE: dict[int, tuple[str, float, float, float, float]] = {
    0: ("rpsM", 0.9928, 0.9933, 174.4, 164.4),
    1: ("recA", 0.9203, 0.9184, 861.5, 882.4),
    2: ("pheA", 0.9987, 0.9986, 37.4, 38.9),
    3: ("metA", 0.9905, 0.9903, 213.8, 221.6),
    4: ("cib", 0.9998, 0.9998, 6.0, 7.5),
    5: ("trpL", 0.9947, 0.9948, 55.9, 58.0),
}

# Rounding tolerances for C5 (the table rounds TRA to 4 decimals and AOGM to 1).
TRA_TOLERANCE = 5e-5
AOGM_TOLERANCE = 0.05

# Default runs root: the trackastra runs directory on the cluster, as used by
# benchmark_training/scripts/analysis/plot_cv_results.py.
DEFAULT_RUNS_DIR = (
    "/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra/runs"
)


@dataclass
class CvRun:
    """One CV run directory (one arm, one fold).

    Attributes
    ----------
    arm : str
        Arm token, ``"masked"`` or ``"dense_flash"``.
    fold : int
        Fold index 0–5 (index into CONDITIONS).
    heldout_condition : str
        Held-out condition parsed from the run name.
    path : Path
        Run directory.
    checkpoint_hashes : list[str]
        md5 hashes of the run's checkpoint files (filled by C3).
    checkpoint_paths : list[Path]
        The checkpoint files that were hashed (filled by C3; used by C6 for
        mtime comparison).
    """

    arm: str
    fold: int
    heldout_condition: str
    path: Path
    checkpoint_hashes: list[str] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)


@dataclass
class CheckResult:
    """Outcome of one soundness check.

    Attributes
    ----------
    name : str
        Check identifier (C1…C6).
    failures : list[str]
        Human-readable failure messages (empty = pass).
    warnings : list[str]
        Non-fatal caveats (missing files that make a check undecidable).
    """

    name: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def md5_of_file(path: Path) -> str:
    """Return the md5 hex digest of a file, read in chunks.

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


def parse_run_name(name: str) -> tuple[str, int, str] | None:
    """Parse arm, fold and held-out condition from a CV run directory name.

    Parameters
    ----------
    name : str
        Directory name such as ``2026-08-18_12-00-01_cv_masked_fold0_rpsM``.

    Returns
    -------
    tuple[str, int, str] | None
        (arm, fold, held-out condition), or None when the name is not a CV
        run of a known arm.
    """
    for arm, pattern in ARM_PATTERNS.items():
        token = pattern.replace("*", "")
        if token not in name:
            continue
        try:
            fold = int(name.split("fold")[1].split("_")[0])
            heldout = name.rsplit("_", 1)[1]
        except (IndexError, ValueError):
            return None
        return arm, fold, heldout
    return None


def discover_runs(runs_dir: Path) -> list[CvRun]:
    """Find and parse all CV run directories under the runs root.

    Parameters
    ----------
    runs_dir : Path
        Runs root directory.

    Returns
    -------
    list[CvRun]
        All parseable CV runs (any arm, any fold).
    """
    runs: list[CvRun] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_run_name(entry.name)
        if parsed is None:
            continue
        arm, fold, heldout = parsed
        runs.append(CvRun(arm=arm, fold=fold, heldout_condition=heldout, path=entry))
    return runs


def check_run_discovery(runs: list[CvRun]) -> CheckResult:
    """C1: verify 6 runs per arm with consistent fold/condition naming.

    Parameters
    ----------
    runs : list[CvRun]
        Discovered CV runs.

    Returns
    -------
    CheckResult
        Failures for missing/duplicate/extra runs and name↔condition
        inconsistencies.
    """
    result = CheckResult(name="C1 run discovery")
    for arm in ARM_PATTERNS:
        arm_runs = [run for run in runs if run.arm == arm]
        folds = sorted(run.fold for run in arm_runs)
        if folds != list(range(6)):
            result.failures.append(
                f"arm {arm}: expected folds 0-5, found {folds} "
                f"({len(arm_runs)} runs)"
            )
        for run in arm_runs:
            expected_condition = CONDITIONS[run.fold]
            if run.heldout_condition != expected_condition:
                result.failures.append(
                    f"arm {arm} fold {run.fold}: run name holds out "
                    f"{run.heldout_condition!r} but fold {run.fold} is "
                    f"{expected_condition!r}"
                )
    if len(runs) > 12:
        result.failures.append(f"found {len(runs)} CV runs, expected 12 (duplicate runs?)")
    return result


def check_leakage(runs: list[CvRun]) -> CheckResult:
    """C2: verify the held-out condition never appears in train_config.yaml.

    The condition tokens are distinctive enough that any occurrence in the
    config (e.g. inside an input path) indicates the held-out data reached
    training or validation.

    Parameters
    ----------
    runs : list[CvRun]
        Discovered CV runs.

    Returns
    -------
    CheckResult
        FAIL per leak found, WARN per missing config file.
    """
    result = CheckResult(name="C2 leakage (artifact level)")
    for run in runs:
        config_path = run.path / "train_config.yaml"
        if not config_path.is_file():
            result.warnings.append(f"{run.path.name}: train_config.yaml missing, cannot check")
            continue
        lines = config_path.read_text(errors="replace").splitlines()
        offending = [
            line.strip()
            for line in lines
            if run.heldout_condition in line and "git_commit" not in line
        ]
        if offending:
            result.failures.append(
                f"{run.path.name}: held-out condition {run.heldout_condition!r} "
                f"appears in train_config.yaml: {offending[:3]}"
            )
    return result


def check_checkpoint_identity(
    runs: list[CvRun], checkpoint_glob: str
) -> CheckResult:
    """C3: verify every run has checkpoints and no two runs share weights.

    Parameters
    ----------
    runs : list[CvRun]
        Discovered CV runs.
    checkpoint_glob : str
        Glob pattern (relative to the run directory) for checkpoint files.

    Returns
    -------
    CheckResult
        FAIL when two runs share a checkpoint hash; WARN when a run has no
        checkpoint files matching the glob.
    """
    result = CheckResult(name="C3 checkpoint identity")
    hash_to_runs: dict[str, list[str]] = {}
    for run in runs:
        files = sorted(p for p in run.path.glob(checkpoint_glob) if p.is_file())
        if not files:
            result.warnings.append(
                f"{run.path.name}: no checkpoint files match {checkpoint_glob!r}"
            )
            continue
        for path in files:
            digest = md5_of_file(path)
            run.checkpoint_hashes.append(digest)
            run.checkpoint_paths.append(path)
            hash_to_runs.setdefault(digest, []).append(run.path.name)
    for digest, names in sorted(hash_to_runs.items()):
        if len(names) > 1:
            result.failures.append(
                f"checkpoint hash {digest[:12]} shared by {len(names)} runs: {names}"
            )
    return result


def collect_eval_files(run: CvRun) -> list[Path]:
    """Return all eval artifact files of a run (the ``cv_eval/`` directory).

    Parameters
    ----------
    run : CvRun
        CV run.

    Returns
    -------
    list[Path]
        Files under ``<run>/cv_eval/`` (empty when the directory is absent).
    """
    eval_dir = run.path / "cv_eval"
    if not eval_dir.is_dir():
        return []
    return sorted(p for p in eval_dir.iterdir() if p.is_file())


def check_eval_duplication(runs: list[CvRun]) -> CheckResult:
    """C4: verify no eval artifact file content is shared between two runs.

    Byte-identical eval files across independently trained runs are the
    signature of the K-sweep duplication bug; honest per-run evals cannot
    produce them.

    Parameters
    ----------
    runs : list[CvRun]
        Discovered CV runs.

    Returns
    -------
    CheckResult
        FAIL per shared file hash; WARN per run without eval artifacts.
    """
    result = CheckResult(name="C4 eval duplication")
    hash_to_files: dict[str, list[str]] = {}
    for run in runs:
        eval_files = collect_eval_files(run)
        if not eval_files:
            result.warnings.append(f"{run.path.name}: no cv_eval/ artifacts")
            continue
        for path in eval_files:
            digest = md5_of_file(path)
            hash_to_files.setdefault(digest, []).append(f"{run.path.name}/{path.name}")
    for digest, locations in sorted(hash_to_files.items()):
        if len(locations) > 1:
            result.failures.append(
                f"eval file hash {digest[:12]} shared by {len(locations)} files: "
                f"{locations}"
            )
    return result


def check_report_provenance(runs: list[CvRun]) -> CheckResult:
    """C5: verify the stored per-fold means reproduce Table 6.

    Each run's ``cv_eval/summary.json`` (written by eval_cv.py) must match the
    published per-fold value of its arm within rounding tolerance.

    Parameters
    ----------
    runs : list[CvRun]
        Discovered CV runs.

    Returns
    -------
    CheckResult
        FAIL per deviation beyond tolerance or missing/invalid summary.
    """
    result = CheckResult(name="C5 report provenance (tab:cv_results)")
    for run in runs:
        summary_path = run.path / "cv_eval" / "summary.json"
        if not summary_path.is_file():
            result.failures.append(f"{run.path.name}: cv_eval/summary.json missing")
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            result.failures.append(f"{run.path.name}: summary.json unparseable ({exc})")
            continue
        _, masked_tra, flash_tra, masked_aogm, flash_aogm = REPORT_TABLE[run.fold]
        if run.arm == "masked":
            expected_values = (("mean_TRA", masked_tra), ("mean_AOGM", masked_aogm))
        else:
            expected_values = (("mean_TRA", flash_tra), ("mean_AOGM", flash_aogm))
        for metric, expected_value in expected_values:
            stored_value = summary.get(metric)
            if stored_value is None:
                result.failures.append(f"{run.path.name}: summary.json has no {metric}")
                continue
            tolerance = TRA_TOLERANCE if metric == "mean_TRA" else AOGM_TOLERANCE
            if abs(float(stored_value) - expected_value) > tolerance:
                result.failures.append(
                    f"{run.path.name}: {metric} stored {stored_value:.6f} vs Table 6 "
                    f"{expected_value} (tolerance {tolerance})"
                )
    return result


def check_timing(runs: list[CvRun]) -> CheckResult:
    """C6: verify eval artifacts were written after the checkpoints.

    An eval file older than the run's newest checkpoint means the metrics were
    computed before (or from something other than) the finished training run.

    Parameters
    ----------
    runs : list[CvRun]
        Discovered CV runs.

    Returns
    -------
    CheckResult
        FAIL per run whose eval artifacts predate its checkpoints; WARN when
        checkpoints are unknown so the check is undecidable.
    """
    result = CheckResult(name="C6 timing sanity")
    for run in runs:
        eval_files = collect_eval_files(run)
        if not eval_files:
            result.warnings.append(f"{run.path.name}: no cv_eval/ artifacts, timing unknown")
            continue
        if not run.checkpoint_paths:
            result.warnings.append(
                f"{run.path.name}: no checkpoint files hashed, timing check skipped"
            )
            continue
        newest_checkpoint = max(p.stat().st_mtime for p in run.checkpoint_paths)
        oldest_eval = min(p.stat().st_mtime for p in eval_files)
        if oldest_eval < newest_checkpoint:
            result.failures.append(
                f"{run.path.name}: eval artifacts older than newest checkpoint "
                f"(eval {oldest_eval:.0f} < checkpoint {newest_checkpoint:.0f})"
            )
    return result


def print_result(result: CheckResult) -> None:
    """Print one check result with PASS/FAIL/WARN lines.

    Parameters
    ----------
    result : CheckResult
        The check outcome.
    """
    status = "FAIL" if result.failures else "PASS"
    print(f"[{status}] {result.name}")
    for failure in result.failures:
        print(f"    FAIL: {failure}")
    for warning in result.warnings:
        print(f"    WARN: {warning}")


def main(argv: list[str] | None = None) -> int:
    """Run all soundness checks and report the verdict.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (defaults to sys.argv).

    Returns
    -------
    int
        0 when no check failed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(DEFAULT_RUNS_DIR),
        help="Runs root directory containing the 12 CV run directories",
    )
    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default="checkpoints/*",
        help="Glob (relative to each run dir) matching checkpoint files",
    )
    arguments = parser.parse_args(argv)

    if not arguments.runs_dir.is_dir():
        print(f"ERROR: runs dir not found: {arguments.runs_dir}", file=sys.stderr)
        return 1

    runs = discover_runs(arguments.runs_dir)
    if not runs:
        print(f"ERROR: no CV runs found under {arguments.runs_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(runs)} CV runs under {arguments.runs_dir}")

    results = [
        check_run_discovery(runs),
        check_leakage(runs),
    ]
    identity_result = check_checkpoint_identity(runs, arguments.checkpoint_glob)
    results.append(identity_result)
    results.append(check_eval_duplication(runs))
    results.append(check_report_provenance(runs))
    results.append(check_timing(runs))

    for result in results:
        print_result(result)

    failed = [result for result in results if result.failures]
    warned = [result for result in results if result.warnings and not result.failures]
    print(
        f"\nVerdict: {len(results) - len(failed)}/{len(results)} checks passed, "
        f"{len(failed)} failed, {len(warned)} passed with warnings."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
