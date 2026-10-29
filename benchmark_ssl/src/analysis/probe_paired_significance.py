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

Multi-seed pooling (SPEC 0021)
------------------------------
With more than one ``--input`` file (one probe artifact per training seed,
identical protocol and pair dumps), the per-frame-pair metrics are averaged
across seeds per feature set *before* the deltas are formed:

    d_i(f) = mean_s( m_f,s(i) - m_rp,s(i) ),  s in {42, 43, 44}

The frame pair stays the statistical unit (n = 29).  Seeds are repeated
measures on the same 29 pairs, NOT independent units, so averaging per pair
is the correct reduction; the cross-seed spread of each delta is stored
descriptively (per-seed deltas and their sd) and never tested.  The scope
limit from the seed audit (``docs/agent-state/research/13-edge-probe-seed-audit.md``)
survives: probe seeds sample only probe-head init / fold-shuffle /
batch-order noise; the frozen encoders are identical across seeds, so
encoder-level uncertainty is not addressed.

``--check`` validates whichever mode ran: the single-file path against the
SPEC 0006 ``EXPECTED`` table, the pooled path against ``EXPECTED_SEEDAVG``
(filled from the first verified run, SPEC 0021 implementation record).

Data provenance
---------------
Per-fold and per-frame-pair metrics of the unified edge probe, produced by
``python -m src.edge_probing.unified_edge_probe --features all --probe both
--cv-folds <k> --max-pairs 30 --seed <s> --dump-predictions`` for seeds
42/43/44 and stored in ``results/probes/unified_probe_results_cv10{,_s43,_s44}.json``
of this repository's ``results/probes/`` directory (seed 42 copy also in
``benchmark_ssl/results/probes/``).  The frozen encoders (DINOv2, CNN
NT-Xent checkpoint) are identical across folds, feature sets and seeds, so
the only paired quantity that varies between feature sets is the probe
training itself.

Usage
-----
    python -m src.analysis.probe_paired_significance [--check]
    python -m src.analysis.probe_paired_significance \
        --input results/probes/unified_probe_results_cv10.json \
               results/probes/unified_probe_results_cv10_s43.json \
               results/probes/unified_probe_results_cv10_s44.json \
        [--check]

Single ``--input`` reproduces the original single-seed analysis (SPEC 0006).
Multiple ``--input`` files run the seed-averaged analysis (SPEC 0021) and
write to ``results/probes/probe_paired_significance_multiseed.json`` by
default.  With ``--check`` the computed statistics are validated against the
values recorded in this file (``EXPECTED`` / ``EXPECTED_SEEDAVG``) and the
process exits non-zero on any deviation, so the reported numbers can be
verified mechanically.
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

# SPEC 0021 verified values for the seed-averaged (pooled) analysis, keyed by
# (feature, metric) with the same tuple layout as ``EXPECTED``.  Filled from
# the first verified multi-seed run over seeds 42/43/44; ``--check`` compares
# against this table when the pooled path runs.
EXPECTED_SEEDAVG: dict[tuple[str, str], tuple[float, ...]] = {
    ("cnn_e2e", "bal_acc"): (-0.233318, -9.477483, 0.00000000, -0.278977, -0.184258, -1.759924, 0.00001000, 0.00000000),
    ("cnn_e2e", "f1"): (-0.405032, -21.168545, 0.00000000, -0.443253, -0.370322, -3.930900, 0.00001000, 0.00000000),
    ("cnn_frozen", "bal_acc"): (0.063956, 1.710757, 0.09818520, -0.008618, 0.136615, 0.317680, 0.09804902, 0.19637040),
    ("cnn_frozen", "f1"): (0.093678, 2.519900, 0.01772109, 0.020822, 0.164919, 0.467934, 0.01916981, 0.01772109),
    ("dino", "bal_acc"): (0.163285, 7.815466, 0.00000002, 0.123450, 0.203098, 1.451296, 0.00001000, 0.00000008),
    ("dino", "f1"): (0.303493, 14.388782, 0.00000000, 0.263281, 0.344660, 2.671930, 0.00001000, 0.00000000),
    ("hoct2d_fourier", "bal_acc"): (0.131886, 6.044089, 0.00000163, 0.088462, 0.171721, 1.122359, 0.00003000, 0.00000488),
    ("hoct2d_fourier", "f1"): (0.265456, 7.878224, 0.00000001, 0.197986, 0.327074, 1.462950, 0.00001000, 0.00000004),
    ("hoct2d", "bal_acc"): (0.027556, 1.301423, 0.20372082, -0.013797, 0.067534, 0.241668, 0.20162798, 0.20372082),
    ("hoct2d", "f1"): (0.108283, 4.073663, 0.00034479, 0.057451, 0.159117, 0.756460, 0.00034000, 0.00068957),
    ("rp_fourier", "bal_acc"): (0.131607, 6.649934, 0.00000032, 0.093664, 0.170060, 1.234862, 0.00001000, 0.00000130),
    ("rp_fourier", "f1"): (0.266444, 8.959829, 0.00000000, 0.207552, 0.321004, 1.663798, 0.00001000, 0.00000000),
}


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


def seed_average_pair_values(probe_jsons: list[dict], feature: str, probe: str,
                             metric: str) -> dict[str, float] | None:
    """Average per-frame-pair metric values across seed artifacts by pair id.

    With a single artifact this is the identity map onto that artifact's
    values (SPEC 0006 behaviour).  With several artifacts (one per training
    seed, identical protocol), frame pairs missing from any artifact are
    dropped with a warning, so every comparison runs on the intersection.

    Args:
        probe_jsons: One or more parsed probe results files, one per seed.
        feature: Short feature key (see :data:`FEATURE_LABELS`).
        probe: Probe type, ``"mlp"`` or ``"linear"``.
        metric: Metric key, ``"bal_acc"`` or ``"f1"``.

    Returns:
        Mapping ``pair_id -> seed-averaged metric value``, or ``None`` when
        no artifact carries a per-pair dump for this feature set.
    """
    per_seed = [load_pair_values(probe_json, feature, probe, metric) for probe_json in probe_jsons]
    per_seed = [values for values in per_seed if values is not None]
    if not per_seed:
        return None
    common = set(per_seed[0])
    for values in per_seed[1:]:
        common &= set(values)
    dropped = set().union(*(set(values) for values in per_seed)) - common
    if dropped:
        print(f"WARNING: seed-averaging drops {len(dropped)} non-shared frame pairs "
              f"for {feature}/{probe}/{metric}")
    return {
        pair_id: sum(values[pair_id] for values in per_seed) / len(per_seed)
        for pair_id in sorted(common)
    }


def compute_pair_results(probe_jsons: list[dict], reference: str, probe: str
                         ) -> tuple[dict, dict] | None:
    """Compute per-frame-pair paired comparisons against the reference feature.

    Frame pairs whose id is present for the reference but missing for a
    feature set (or vice versa, or missing from any seed artifact) are
    counted and dropped, so every comparison runs on the matched subset.
    With more than one artifact the per-frame-pair metrics are seed-averaged
    per feature set before the deltas are formed (SPEC 0021); the frame pair
    stays the statistical unit and seeds are never tested independently.

    Args:
        probe_jsons: One or more parsed probe results files (one per seed).
        reference: Short key of the reference feature set.
        probe: Probe type, ``"mlp"`` or ``"linear"``.

    Returns:
        Tuple of (results, per_seed_deltas).  ``results`` maps
        ``(feature, metric)`` to a :class:`PairedResult` computed on the
        matched frame-pair subset with a bootstrap CI and Monte-Carlo
        sign-flip permutation p value and Holm-adjusted p.  ``per_seed_deltas``
        maps ``(feature, metric)`` to ``{pair_id: [per-seed delta]}`` and is
        empty for the single-artifact path.  ``None`` when no feature set
        carries a dump.
    """
    reference_pairs = seed_average_pair_values(probe_jsons, reference, probe, "bal_acc")
    if reference_pairs is None:
        return None
    features = [
        key for key in probe_jsons[0]["results"]
        if key != reference
        and seed_average_pair_values(probe_jsons, key, probe, "bal_acc") is not None
    ]
    if not features:
        return None

    results: dict[tuple[str, str], PairedResult] = {}
    per_seed_deltas: dict[tuple[str, str], dict[str, list[float]]] = {}
    for metric in METRIC_FIELDS:
        reference_values = seed_average_pair_values(probe_jsons, reference, probe, metric)
        metric_results: dict[str, PairedResult] = {}
        for feature in features:
            feature_values = seed_average_pair_values(probe_jsons, feature, probe, metric)
            matched = sorted(set(reference_values) & set(feature_values))
            dropped = len(set(reference_values) ^ set(feature_values))
            reference_aligned = [reference_values[p] for p in matched]
            feature_aligned = [feature_values[p] for p in matched]
            result = paired_statistics(reference_aligned, feature_aligned)
            differences = [feature_values[p] - reference_values[p] for p in matched]
            result.ci_low, result.ci_high = bootstrap_ci_mean(differences)
            result.permutation_p = monte_carlo_sign_flip_p(differences)
            result.unit_count = len(matched)
            result.dropped_units = dropped
            if len(probe_jsons) > 1:
                per_seed_reference = [load_pair_values(j, reference, probe, metric) for j in probe_jsons]
                per_seed_feature = [load_pair_values(j, feature, probe, metric) for j in probe_jsons]
                per_seed = {}
                for pair_id in matched:
                    deltas = [
                        seed_feature[pair_id] - seed_reference[pair_id]
                        for seed_reference, seed_feature in zip(per_seed_reference, per_seed_feature)
                        if seed_reference is not None and seed_feature is not None
                        and pair_id in seed_reference and pair_id in seed_feature
                    ]
                    if len(deltas) == len(probe_jsons):
                        per_seed[pair_id] = deltas
                per_seed_deltas[(feature, metric)] = per_seed
            metric_results[feature] = result
        adjusted = holm_adjust([metric_results[feature].p_value for feature in features])
        for feature, adjusted_p in zip(features, adjusted):
            metric_results[feature].holm_p = adjusted_p
            results[(feature, metric)] = metric_results[feature]
    return results, per_seed_deltas


def check_expected(results: dict, expected_table: dict | None = None) -> bool:
    """Validate computed statistics against the SPEC-recorded values.

    Args:
        results: Output of :func:`compute_pair_results` (primary unit).
        expected_table: Table to validate against — ``EXPECTED`` for the
            single-seed path (SPEC 0006) or ``EXPECTED_SEEDAVG`` for the
            seed-averaged path (SPEC 0021).  Defaults to ``EXPECTED``.

    Returns:
        True when every recorded statistic matches within
        :data:`CHECK_RTOL`; False otherwise, with mismatches printed.
    """
    if expected_table is None:
        expected_table = EXPECTED
    if not expected_table:
        print("WARNING: no EXPECTED values recorded; skipping --check.")
        return True
    all_ok = True
    for key, expected in expected_table.items():
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

    Single ``--input`` runs the original single-seed analysis (SPEC 0006);
    multiple ``--input`` artifacts run the seed-averaged analysis (SPEC 0021).

    Returns
    -------
    int
        Process exit status (0 on success, 1 on a failed ``--check``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+",
                        default=["results/probes/unified_probe_results_cv10.json"],
                        help="Probe results JSON file(s), one per training seed "
                             "(dump-enabled runs); multiple files trigger the "
                             "seed-averaged analysis")
    parser.add_argument("--reference", default=REFERENCE_FEATURE,
                        help="Short key of the reference feature set")
    parser.add_argument("--probe", default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--output", default=None,
                        help="Where to write the computed statistics "
                             "(default: probe_paired_significance.json, or "
                             "probe_paired_significance_multiseed.json for "
                             "the seed-averaged analysis)")
    parser.add_argument("--check", action="store_true",
                        help="Validate against the SPEC-recorded values of "
                             "the active mode (single-seed: EXPECTED, "
                             "seed-averaged: EXPECTED_SEEDAVG)")
    args = parser.parse_args()

    pooled = len(args.input) > 1
    if args.output is None:
        args.output = ("results/probes/probe_paired_significance_multiseed.json"
                       if pooled else "results/probes/probe_paired_significance.json")

    probe_jsons = [json.loads(Path(path).read_text()) for path in args.input]
    probe_seeds = [probe_json.get("args", {}).get("seed") for probe_json in probe_jsons]
    if pooled and any(seed is None for seed in probe_seeds):
        print("WARNING: an input artifact lacks args.seed; seed list contains None.")
    fold_results = compute_results(probe_jsons[0], args.reference, args.probe)
    pair_computation = compute_pair_results(probe_jsons, args.reference, args.probe)

    fold_count = len(probe_jsons[0]["results"][args.reference][args.probe]["fold_results"])
    seeds_label = "/".join(str(seed) for seed in probe_jsons)
    print(f"Reference: {FEATURE_LABELS[args.reference]} | probe: {args.probe} "
          f"| folds: {fold_count} | inputs: {len(probe_jsons)} | seeds: {seeds_label}")
    if pair_computation is not None:
        pair_results, per_seed_deltas = pair_computation
        print(f"-- per-frame-pair comparisons ({'seed-averaged' if pooled else 'single-seed'}, primary) --")
        for (feature, metric), result in sorted(pair_results.items()):
            print(f"n={result.unit_count} dropped={result.dropped_units} "
                  + latex_table_row(feature, metric, result))
    else:
        pair_results, per_seed_deltas = {}, {}

    print("-- per-fold comparisons (secondary) --")
    for (feature, metric), result in sorted(fold_results.items()):
        print(latex_table_row(feature, metric, result))

    if args.check:
        expected_table = EXPECTED_SEEDAVG if pooled else EXPECTED
        if pooled and not expected_table:
            print("WARNING: EXPECTED_SEEDAVG not yet recorded; cannot --check the pooled path.")
            return 1
        return 0 if check_expected(pair_results if pair_results else fold_results,
                                   expected_table) else 1

    payload = {
        "reference": args.reference,
        "probe": args.probe,
        "inputs": args.input,
        "seeds": probe_seeds,
        "pooled": pooled,
        "primary_unit": "pairs" if pair_results else "folds",
        "comparisons_pairs": (
            {f"{feature}|{metric}": result_to_dict(result)
             for (feature, metric), result in pair_results.items()}
        ),
        "comparisons_folds": {
            f"{feature}|{metric}": result_to_dict(result)
            for (feature, metric), result in fold_results.items()
        },
    }
    if not pooled:
        payload["args"] = probe_jsons[0].get("args", {})
    else:
        payload["per_seed_deltas"] = {
            f"{feature}|{metric}": per_seed
            for (feature, metric), per_seed in per_seed_deltas.items()
        }
        payload["seed_spread"] = {
            f"{feature}|{metric}": {
                "mean_pair_sd": float(np.mean([np.std(deltas, ddof=1)
                                               for deltas in per_seed.values() if len(deltas) > 1])),
                "max_pair_sd": float(np.max([np.std(deltas, ddof=1)
                                             for deltas in per_seed.values() if len(deltas) > 1])),
            }
            for (feature, metric), per_seed in per_seed_deltas.items()
        }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
