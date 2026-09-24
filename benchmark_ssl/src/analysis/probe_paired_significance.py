r"""Paired significance tests for the edge-probe feature-engineering ceiling.

Why this script exists
----------------------
The report claims that the feature-engineering ceiling between 7D regionprops
and 384D DINOv2 embeddings is small (+0.036 balanced accuracy in the original
5-fold run).  Every mean difference, t value, confidence interval and p value
printed in the report must come from this script.  It mirrors the methodology
of ``report/scripts/cv_paired_significance.py`` (paired two-sided t-test,
95% CI of the mean difference, Cohen's d_z, sign-flip permutation test) so
the significance claims in the report share one consistent procedure, and
applies Holm correction across the feature sets compared against the
7D-regionprops reference.

Two statistical units
---------------------
* **Frame pairs (primary)**: with ``--dump-predictions`` the probe records
  per-frame-pair metrics for each held-out frame pair, and every frame pair
  is validated exactly once across all folds.  The frame pair is therefore
  the natural statistical unit; the test is a paired t-test plus a bootstrap
  CI and a Monte-Carlo sign-flip permutation over the matched pairs.
* **Folds (secondary)**: the legacy per-fold metric averages per-batch
  balanced accuracy, which smears one frame pair across batch boundaries; it
  is reported for reference only.  With n folds the exact sign-flip
  enumeration has ``2**n`` assignments.

Data provenance
---------------
Per-fold and per-frame-pair metrics of the unified edge probe, produced by
``python -m src.edge_probing.unified_edge_probe --features all --probe both
--cv-folds <k> --max-pairs 30 --seed 42 --dump-predictions`` and stored in
``results/probes/unified_probe_results_cv10.json`` of this subproject.  The
frozen encoders (DINOv2, CNN NT-Xent checkpoint) are identical across folds
and feature sets, so the only paired quantity that varies between feature
sets is the probe training itself.

Usage
-----
    python -m src.analysis.probe_paired_significance [--check] [--unit both]

Without ``--check`` the script prints a human-readable summary and the LaTeX
rows used in the report.  With ``--check`` it validates the computed
statistics against the values recorded in
``docs/agent-state/specs/0006-probe-ci/SPEC.md`` and exits non-zero on any
deviation, so the reported numbers can be verified mechanically.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


# Human-readable label per short feature key used in the probe JSON.
FEATURE_LABELS: dict[str, str] = {
    "rp": "Regionprops 7D",
    "rp_fourier": "Regionprops 7D + Fourier PE",
    "hoct2d": "HOCT 2D",
    "hoct2d_fourier": "HOCT 2D + Fourier PE",
    "dino": "DINOv2 (frozen)",
    "cnn_frozen": "CNN NT-Xent (frozen)",
    "cnn_e2e": "CNN end-to-end",
}

# Per-fold metric field inside each entry of ``fold_results``.
METRIC_FIELDS: dict[str, str] = {
    "bal_acc": "final_bal_acc",
    "f1": "final_f1",
}

# Default feature compared against the 7D-regionprops reference.
REFERENCE_FEATURE = "rp"

# Number of bootstrap resamples for the per-pair confidence interval.
BOOTSTRAP_RESAMPLES = 10000

# Number of random sign assignments for the Monte-Carlo permutation test
# (the exact enumeration over 29 frame pairs would need 2**29 evaluations).
SIGN_FLIP_RESAMPLES = 100000

# RNG seeds for the resampling procedures, fixed for reproducibility.
BOOTSTRAP_SEED = 42
SIGN_FLIP_SEED = 42

# SPEC 0006 verified values, keyed by (feature, metric):
# (mean_difference, t_statistic, p_value, ci_low, ci_high, cohens_dz,
#  permutation_p, holm_adjusted_p).  Filled from the verified run; --check
# compares against these with a relative tolerance.
EXPECTED: dict[tuple[str, str], tuple[float, ...]] = {
    ("cnn_e2e", "bal_acc"): (-0.234811, -8.680985, 0.00000000, -0.284895, -0.181904, -1.612019, 0.00001000, 0.00000001),
    ("cnn_e2e", "f1"): (-0.407776, -14.770711, 0.00000000, -0.461866, -0.356535, -2.742852, 0.00001000, 0.00000000),
    ("cnn_frozen", "bal_acc"): (0.065273, 1.751869, 0.09074380, -0.007602, 0.135656, 0.325314, 0.09143909, 0.18148761),
    ("cnn_frozen", "f1"): (0.088861, 2.335738, 0.02689418, 0.014816, 0.161418, 0.433736, 0.02766972, 0.02689418),
    ("dino", "bal_acc"): (0.159624, 5.920115, 0.00000227, 0.109072, 0.212062, 1.099338, 0.00002000, 0.00000908),
    ("dino", "f1"): (0.327610, 9.923605, 0.00000000, 0.264115, 0.391177, 1.842767, 0.00001000, 0.00000000),
    ("hoct2d_fourier", "bal_acc"): (0.123325, 4.058622, 0.00035902, 0.060954, 0.177720, 0.753667, 0.00030000, 0.00107705),
    ("hoct2d_fourier", "f1"): (0.240811, 5.545097, 0.00000627, 0.152638, 0.320157, 1.029699, 0.00001000, 0.00001880),
    ("hoct2d", "bal_acc"): (0.038670, 1.141252, 0.26343218, -0.028444, 0.102118, 0.211925, 0.26102739, 0.26343218),
    ("hoct2d", "f1"): (0.116953, 2.940830, 0.00649943, 0.039079, 0.190883, 0.546098, 0.00742993, 0.01299887),
    ("rp_fourier", "bal_acc"): (0.170632, 6.596475, 0.00000037, 0.121427, 0.220213, 1.224935, 0.00001000, 0.00000187),
    ("rp_fourier", "f1"): (0.304429, 8.031930, 0.00000001, 0.229657, 0.375192, 1.491492, 0.00001000, 0.00000004),
}
CHECK_RTOL = 5e-3


@dataclass
class PairedResult:
    """Statistics of one paired two-sided comparison (feature - reference).

    Attributes
    ----------
    mean_difference : float
        Mean of the paired differences (feature minus reference).
    sd_difference : float
        Sample standard deviation of the differences (ddof = 1).
    t_statistic : float
        Paired t statistic on ``unit_count - 1`` degrees of freedom.
    p_value : float
        Two-sided p value of the paired t-test.
    ci_low, ci_high : float
        95% confidence interval of the mean paired difference.
    cohens_dz : float
        Paired effect size, mean(differences) / sd(differences).
    permutation_p : float
        Sign-flip permutation p value (exact for folds, Monte-Carlo for
        frame pairs).
    holm_p : float
        Holm-adjusted p value across the comparisons of one metric.
    unit_count : int or None
        Number of paired units used (folds or matched frame pairs).
    dropped_units : int or None
        Number of units present for one side but not the other (frame-pair
        mode only; folds are always fully paired).
    """

    mean_difference: float
    sd_difference: float
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    cohens_dz: float
    permutation_p: float
    holm_p: float = float("nan")
    unit_count: int | None = None
    dropped_units: int | None = None


def load_fold_values(probe_json: dict, feature: str, probe: str, metric: str) -> list[float]:
    """Return the per-fold metric values of one feature set and probe type.

    Args:
        probe_json: Parsed probe results file.
        feature: Short feature key (see :data:`FEATURE_LABELS`).
        probe: Probe type, ``"mlp"`` or ``"linear"``.
        metric: Metric key, ``"bal_acc"`` or ``"f1"`` (see :data:`METRIC_FIELDS`).

    Returns:
        One metric value per cross-validation fold, in fold order.
    """
    field = METRIC_FIELDS[metric]
    folds = probe_json["results"][feature][probe]["fold_results"]
    return [float(fold[field]) for fold in folds]


def load_pair_values(probe_json: dict, feature: str, probe: str, metric: str) -> dict[str, float] | None:
    """Return per-frame-pair metric values of one feature set, keyed by pair id.

    The per-frame-pair dump is produced by ``--dump-predictions`` and lives in
    ``pair_results`` of the probe results file.  Each held-out frame pair is
    scored exactly once across all folds, so the mapping has one entry per
    frame pair.

    Args:
        probe_json: Parsed probe results file.
        feature: Short feature key.
        probe: Probe type, ``"mlp"`` or ``"linear"``.
        metric: Metric key, ``"bal_acc"`` or ``"f1"``.

    Returns:
        Mapping ``pair_id -> metric value``, or ``None`` when the feature set
        carries no per-pair dump.
    """
    entry = probe_json.get("results", {}).get(feature)
    if not isinstance(entry, dict) or not isinstance(entry.get(probe), dict):
        return None
    pair_results = entry[probe].get("pair_results") or []
    values: dict[str, float] = {}
    for fold in pair_results:
        for pair in fold.get("pairs", []):
            pair_id = pair.get("pair_id")
            if pair_id is not None:
                # Pair-dump rows are keyed directly by metric ("bal_acc"/"f1"),
            # unlike fold_results entries (which use "final_bal_acc"/"final_f1").
                values[pair_id] = float(pair[metric])
    return values or None


def paired_statistics(reference_values: list[float], feature_values: list[float]) -> PairedResult:
    """Return the paired two-sided t-test statistics (feature - reference).

    Args:
        reference_values: Per-unit metric of the 7D-regionprops reference.
        feature_values: Per-unit metric of the compared feature set, aligned
            by unit index.

    Returns:
        PairedResult with mean, sd, t statistic, p value, 95% CI and Cohen's
        d_z.  The caller fills ``permutation_p`` and ``holm_p`` separately.
    """
    differences = [feature - reference for reference, feature in zip(reference_values, feature_values)]
    unit_count = len(differences)
    mean_difference = sum(differences) / unit_count
    sd_difference = math.sqrt(
        sum((diff - mean_difference) ** 2 for diff in differences) / (unit_count - 1)
    )
    t_statistic, p_value = stats.ttest_rel(feature_values, reference_values)
    standard_error = sd_difference / math.sqrt(unit_count)
    t_critical = stats.t.ppf(0.975, unit_count - 1)
    return PairedResult(
        mean_difference=mean_difference,
        sd_difference=sd_difference,
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        ci_low=mean_difference - t_critical * standard_error,
        ci_high=mean_difference + t_critical * standard_error,
        cohens_dz=mean_difference / sd_difference,
        permutation_p=float("nan"),
    )


def exact_sign_flip_permutation_p(differences: list[float]) -> float:
    """Return the exact two-sided sign-flip permutation p value.

    Enumerates all ``2 ** n`` sign assignments of the paired differences (an
    exact test, no randomness) and reports the fraction of assignments whose
    mean absolute difference is at least the observed one.

    Args:
        differences: Paired differences per unit.

    Returns:
        Exact two-sided permutation p value in ``(0, 1]``.
    """
    unit_count = len(differences)
    observed = abs(sum(differences) / unit_count)
    extreme_count = 0
    assignment_count = 0
    for signs in itertools.product((1.0, -1.0), repeat=unit_count):
        assignment_count += 1
        permuted_mean = abs(sum(sign * diff for sign, diff in zip(signs, differences)) / unit_count)
        if permuted_mean >= observed - 1e-12:
            extreme_count += 1
    return extreme_count / assignment_count


def monte_carlo_sign_flip_p(differences: list[float], n_resamples: int = SIGN_FLIP_RESAMPLES,
                             seed: int = SIGN_FLIP_SEED) -> float:
    """Return the two-sided p value of a Monte-Carlo sign-flip permutation test.

    For 29 paired frame pairs the exact enumeration (2**29 assignments) is
    infeasible, so signs are sampled uniformly at random with a fixed seed.

    Args:
        differences: Paired differences per frame pair.
        n_resamples: Number of random sign assignments.
        seed: RNG seed for reproducibility.

    Returns:
        Monte-Carlo two-sided permutation p value in ``(0, 1]``.
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(differences, dtype=float)
    observed = abs(values.mean())
    extreme_count = 0
    for _ in range(n_resamples):
        signs = rng.choice((-1.0, 1.0), size=len(values))
        if abs((signs * values).mean()) >= observed - 1e-12:
            extreme_count += 1
    return (extreme_count + 1) / (n_resamples + 1)


def bootstrap_ci_mean(differences: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Return the percentile bootstrap 95% CI of the mean paired difference.

    Args:
        differences: Paired differences per unit.
        n_resamples: Number of bootstrap resamples of the paired units.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple (ci_low, ci_high) percentile bounds of the bootstrap mean.
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(differences, dtype=float)
    count = len(values)
    indices = rng.randint(0, count, size=(n_resamples, count))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p values, preserving input order.

    Args:
        p_values: Raw p values of one family of comparisons.

    Returns:
        Step-down Holm-adjusted p values, clipped to ``[0, 1]``.
    """
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _has_probe_results(probe_json: dict, feature: str, probe: str) -> bool:
    """Return True when a feature set has fold results for the given probe.

    Some feature sets are recorded as ``None`` (e.g. ``cnn_frozen`` when its
    checkpoint is unavailable); those must be skipped rather than crashing the
    comparison.

    Args:
        probe_json: Parsed probe results file.
        feature: Short feature key.
        probe: Probe type, ``"mlp"`` or ``"linear"``.

    Returns:
        True when the feature/probe entry carries a non-empty ``fold_results``.
    """
    entry = probe_json.get("results", {}).get(feature)
    if not isinstance(entry, dict):
        return False
    probe_entry = entry.get(probe)
    return isinstance(probe_entry, dict) and bool(probe_entry.get("fold_results"))


def compute_results(probe_json: dict, reference: str, probe: str) -> dict:
    """Compute all fold-level paired comparisons against the reference feature.

    Args:
        probe_json: Parsed probe results file.
        reference: Short key of the reference feature set.
        probe: Probe type, ``"mlp"`` or ``"linear"``.

    Returns:
        ``{(feature, metric): PairedResult}`` for every usable feature set
        except the reference, with Holm adjustment applied per metric.
    """
    features = [
        key for key in probe_json["results"]
        if key != reference and _has_probe_results(probe_json, key, probe)
    ]
    results: dict[tuple[str, str], PairedResult] = {}
    for metric in METRIC_FIELDS:
        reference_values = load_fold_values(probe_json, reference, probe, metric)
        metric_results: dict[str, PairedResult] = {}
        for feature in features:
            feature_values = load_fold_values(probe_json, feature, probe, metric)
            metric_results[feature] = paired_statistics(reference_values, feature_values)
            metric_results[feature].unit_count = len(reference_values)
        adjusted = holm_adjust([metric_results[feature].p_value for feature in features])
        for feature, adjusted_p in zip(features, adjusted):
            metric_results[feature].holm_p = adjusted_p
            results[(feature, metric)] = metric_results[feature]
    return results


def compute_pair_results(probe_json: dict, reference: str, probe: str) -> dict | None:
    """Compute per-frame-pair paired comparisons against the reference feature.

    Frame pairs whose id is present for the reference but missing for a
    feature set (or vice versa) are counted and dropped, so every comparison
    runs on the matched subset.

    Args:
        probe_json: Parsed probe results file.
        reference: Short key of the reference feature set.
        probe: Probe type, ``"mlp"`` or ``"linear"``.

    Returns:
        ``{(feature, metric): PairedResult}`` computed on the matched
        frame-pair subset with a bootstrap CI and Monte-Carlo sign-flip
        permutation p value, or ``None`` when no feature set carries a dump.
    """
    reference_pairs = load_pair_values(probe_json, reference, probe, "bal_acc")
    if reference_pairs is None:
        return None
    features = [
        key for key in probe_json["results"]
        if key != reference and load_pair_values(probe_json, key, probe, "bal_acc") is not None
    ]
    if not features:
        return None

    results: dict[tuple[str, str], PairedResult] = {}
    for metric in METRIC_FIELDS:
        reference_values = load_pair_values(probe_json, reference, probe, metric)
        metric_results: dict[str, PairedResult] = {}
        for feature in features:
            feature_values = load_pair_values(probe_json, feature, probe, metric)
            matched = sorted(set(reference_values) & set(feature_values))
            dropped = len(set(reference_values) ^ set(feature_values))
            reference_aligned = [reference_values[p] for p in matched]
            feature_aligned = [feature_values[p] for p in matched]
            result = paired_statistics(reference_aligned, feature_aligned)
            result.ci_low, result.ci_high = bootstrap_ci_mean(
                [feature_values[p] - reference_values[p] for p in matched]
            )
            result.permutation_p = monte_carlo_sign_flip_p(
                [feature_values[p] - reference_values[p] for p in matched]
            )
            result.unit_count = len(matched)
            result.dropped_units = dropped
            metric_results[feature] = result
        adjusted = holm_adjust([metric_results[feature].p_value for feature in features])
        for feature, adjusted_p in zip(features, adjusted):
            metric_results[feature].holm_p = adjusted_p
            results[(feature, metric)] = metric_results[feature]
    return results


def check_expected(results: dict) -> bool:
    """Validate computed statistics against the SPEC-recorded values.

    Args:
        results: Output of :func:`compute_pair_results` (primary unit).

    Returns:
        True when every recorded statistic matches within
        :data:`CHECK_RTOL`; False otherwise, with mismatches printed.
    """
    if not EXPECTED:
        print("WARNING: no EXPECTED values recorded; skipping --check.")
        return True
    all_ok = True
    for key, expected in EXPECTED.items():
        if key not in results:
            print(f"MISSING comparison {key}")
            all_ok = False
            continue
        result = results[key]
        actual = (
            result.mean_difference,
            result.t_statistic,
            result.p_value,
            result.ci_low,
            result.ci_high,
            result.cohens_dz,
            result.permutation_p,
            result.holm_p,
        )
        for name, exp, act in zip(
            ("mean_difference", "t", "p", "ci_low", "ci_high", "dz", "perm_p", "holm_p"),
            expected,
            actual,
        ):
            if not math.isclose(exp, act, rel_tol=CHECK_RTOL, abs_tol=1e-6):
                print(f"MISMATCH {key} {name}: expected {exp}, got {act}")
                all_ok = False
    print("check passed" if all_ok else "check FAILED")
    return all_ok


def latex_table_row(feature: str, metric: str, result: PairedResult) -> str:
    """Return the report table row for one paired comparison.

    Args:
        feature: Human-readable feature label.
        metric: Metric key, ``"bal_acc"`` or ``"f1"``.
        result: Statistics of the comparison.

    Returns:
        A LaTeX tabular row with mean difference, CI, t, p and adjusted p.
    """
    return (
        f"{FEATURE_LABELS[feature]} & {metric} & {result.mean_difference:+.4f} & "
        f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}] & {result.t_statistic:.2f} & "
        f"{result.p_value:.4f} & {result.permutation_p:.4f} & {result.holm_p:.4f} \\\\"
    )


def result_to_dict(result: PairedResult) -> dict:
    """Serialize one PairedResult for the JSON output."""
    return {
        "mean_difference": result.mean_difference,
        "sd_difference": result.sd_difference,
        "t_statistic": result.t_statistic,
        "p_value": result.p_value,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "cohens_dz": result.cohens_dz,
        "permutation_p": result.permutation_p,
        "holm_p": result.holm_p,
        "unit_count": result.unit_count,
        "dropped_units": result.dropped_units,
    }


def main() -> int:
    """Run the paired significance tests and optionally validate them.

    Returns
    -------
    int
        Process exit status (0 on success, 1 on a failed ``--check``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/probes/unified_probe_results_cv10.json",
                        help="Probe results JSON (dump-enabled run)")
    parser.add_argument("--reference", default=REFERENCE_FEATURE,
                        help="Short key of the reference feature set")
    parser.add_argument("--probe", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--output", default="results/probes/probe_paired_significance.json",
                        help="Where to write the computed statistics")
    parser.add_argument("--check", action="store_true",
                        help="Validate against the SPEC-recorded values")
    args = parser.parse_args()

    probe_json = json.loads(Path(args.input).read_text())
    fold_results = compute_results(probe_json, args.reference, args.probe)
    pair_results = compute_pair_results(probe_json, args.reference, args.probe)

    fold_count = len(probe_json["results"][args.reference][args.probe]["fold_results"])
    print(f"Reference: {FEATURE_LABELS[args.reference]} | probe: {args.probe} | folds: {fold_count}")
    if pair_results is not None:
        print("-- per-frame-pair comparisons (primary) --")
        for (feature, metric), result in sorted(pair_results.items()):
            print(f"n={result.unit_count} dropped={result.dropped_units} "
                  + latex_table_row(feature, metric, result))
    print("-- per-fold comparisons (secondary) --")
    for (feature, metric), result in sorted(fold_results.items()):
        print(latex_table_row(feature, metric, result))

    if args.check:
        primary = pair_results if pair_results is not None else fold_results
        return 0 if check_expected(primary) else 1

    payload = {
        "reference": args.reference,
        "probe": args.probe,
        "input": args.input,
        "primary_unit": "pairs" if pair_results is not None else "folds",
        "args": probe_json.get("args", {}),
        "comparisons_pairs": (
            {f"{feature}|{metric}": result_to_dict(result)
             for (feature, metric), result in pair_results.items()}
            if pair_results is not None else {}
        ),
        "comparisons_folds": {
            f"{feature}|{metric}": result_to_dict(result)
            for (feature, metric), result in fold_results.items()
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
