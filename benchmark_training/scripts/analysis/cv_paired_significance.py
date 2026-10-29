r"""Paired significance tests for the spatial-cutoff ablation.

Why this script exists: the report claims that masked attention (spatial
cutoff) and dense-flash attention (no cutoff) are statistically
indistinguishable in the 6-fold leave-one-condition-out cross-validation on
the vanvliet dataset.  That claim is only as good as the test behind it, so
every t value and p value printed in the report must come from this script.
``--check`` validates the computed statistics against the values recorded in
``docs/agent-state/specs/0004-cv-statistical-significance-argumentation/SPEC.md``,
so the .tex text can be verified mechanically at build time.

Data provenance: the per-fold paired results of the 6-fold LOCO-CV
(masked = spatial cutoff, dense flash = no cutoff; both trained with seed 42),
as published in ``report/appendix.tex``, table ``tab:cv_results``.  The raw
run artifacts are not tracked in this workspace; if a raw CSV of the fold
results becomes available, replace ``FOLDS`` with a loader for that file and
re-verify the expected values below.

Run::

    python report/scripts/cv_paired_significance.py [--check]

``--check`` exits non-zero if any statistic deviates from the SPEC values;
without it the script prints a human-readable summary plus the LaTeX table
row used in the appendix.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass

from scipy import stats


# Per-fold paired results (held-out condition, masked TRA, flash TRA,
# masked AOGM, flash AOGM) from report/appendix.tex tab:cv_results.
FOLDS: list[tuple[str, float, float, float, float]] = [
    ("rpsM", 0.9928, 0.9933, 174.4, 164.4),
    ("recA", 0.9203, 0.9184, 861.5, 882.4),
    ("pheA", 0.9987, 0.9986, 37.4, 38.9),
    ("metA", 0.9905, 0.9903, 213.8, 221.6),
    ("cib", 0.9998, 0.9998, 6.0, 7.5),
    ("trpL", 0.9947, 0.9948, 55.9, 58.0),
]

# SPEC 0004 verified values: metric -> (t, p, ci_low, ci_high, dz, perm_p).
EXPECTED: dict[str, tuple[float, float, float, float, float, float]] = {
    "TRA": (-0.781, 0.4699, -0.001144, 0.000611, -0.319, 0.6875),
    "AOGM": (0.960, 0.3811, -6.654, 14.588, 0.392, 0.40625),
}

# Relative tolerance for --check (t, p, CI and dz are compared relatively
# because TRA lives on a 1e-3 scale while AOGM lives on a 1e2 scale).
CHECK_RTOL = 2e-3


@dataclass
class PairedResult:
    """Statistics of a paired two-sided comparison (flash - masked).

    Attributes
    ----------
    mean_difference : float
        Mean of the paired differences (flash - masked).
    sd_difference : float
        Sample standard deviation of the paired differences (ddof=1).
    t_statistic : float
        Paired t statistic on n_folds - 1 degrees of freedom.
    p_value : float
        Two-sided p value of the paired t-test.
    ci_low, ci_high : float
        95% confidence interval of the mean paired difference.
    cohens_dz : float
        Paired effect size mean(diffs) / sd(diffs).
    """

    mean_difference: float
    sd_difference: float
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    cohens_dz: float


def paired_statistics(masked_values: list[float], flash_values: list[float]) -> PairedResult:
    """Return the paired two-sided t-test statistics (flash - masked).

    Parameters
    ----------
    masked_values : list[float]
        Per-fold metric of the masked (spatial-cutoff) configuration.
    flash_values : list[float]
        Per-fold metric of the dense-flash (no-cutoff) configuration.

    Returns
    -------
    PairedResult
        Mean, sd, t statistic, p value, 95% CI of the mean difference and
        Cohen's d_z for the paired sample.
    """
    differences = [flash - masked for masked, flash in zip(masked_values, flash_values)]
    fold_count = len(differences)
    mean_difference = sum(differences) / fold_count
    sd_difference = math.sqrt(
        sum((diff - mean_difference) ** 2 for diff in differences) / (fold_count - 1)
    )
    t_statistic, p_value = stats.ttest_rel(flash_values, masked_values)
    standard_error = sd_difference / math.sqrt(fold_count)
    t_critical = stats.t.ppf(0.975, fold_count - 1)
    return PairedResult(
        mean_difference=mean_difference,
        sd_difference=sd_difference,
        t_statistic=float(t_statistic),
        p_value=float(p_value),
        ci_low=mean_difference - t_critical * standard_error,
        ci_high=mean_difference + t_critical * standard_error,
        cohens_dz=mean_difference / sd_difference,
    )


def sign_flip_permutation_p(differences: list[float]) -> float:
    """Return the exact two-sided p value of the sign-flip permutation test.

    Enumerates all 2**n sign assignments of the paired differences (exact
    test, no randomness) and reports the fraction of assignments whose mean
    absolute difference is at least the observed one.

    Parameters
    ----------
    differences : list[float]
        Paired differences (flash - masked) per fold.

    Returns
    -------
    float
        Exact two-sided permutation p value in (0, 1].
    """
    fold_count = len(differences)
    observed = abs(sum(differences) / fold_count)
    extreme_count = 0
    assignment_count = 0
    for signs in itertools.product((1.0, -1.0), repeat=fold_count):
        assignment_count += 1
        permuted_mean = abs(sum(sign * diff for sign, diff in zip(signs, differences)) / fold_count)
        if permuted_mean >= observed - 1e-12:
            extreme_count += 1
    return extreme_count / assignment_count


def latex_table_row(tra: PairedResult, aogm: PairedResult) -> str:
    """Return the appendix table row for the paired t-test.

    Parameters
    ----------
    tra, aogm : PairedResult
        Results for the TRA and AOGM metrics respectively.

    Returns
    -------
    str
        One LaTeX table row (to be placed after the difference row in
        ``tab:cv_results``).
    """
    tra_part = f"$t = {tra.t_statistic:+.2f}$, $p = {tra.p_value:.2f}$"
    aogm_part = f"$t = {aogm.t_statistic:+.2f}$, $p = {aogm.p_value:.2f}$"
    return (
        "\\multicolumn{2}{l}{Paired $t$-test (flash $-$ masked)} & "
        f"\\multicolumn{{2}}{{c}}{{{tra_part}}} & "
        f"\\multicolumn{{2}}{{c}}{{{aogm_part}}} \\\\"
    )


def check_against_spec(tra: PairedResult, aogm: PairedResult, tra_perm_p: float, aogm_perm_p: float) -> list[str]:
    """Compare computed statistics against the SPEC 0004 expected values.

    Parameters
    ----------
    tra, aogm : PairedResult
        Computed results for both metrics.
    tra_perm_p, aogm_perm_p : float
        Computed exact permutation p values.

    Returns
    -------
    list[str]
        Human-readable descriptions of every mismatch (empty list = pass).
    """
    mismatches: list[str] = []
    computed = {
        "TRA": (tra.t_statistic, tra.p_value, tra.ci_low, tra.ci_high, tra.cohens_dz, tra_perm_p),
        "AOGM": (aogm.t_statistic, aogm.p_value, aogm.ci_low, aogm.ci_high, aogm.cohens_dz, aogm_perm_p),
    }
    for metric, expected_values in EXPECTED.items():
        for name, expected_value, actual_value in zip(
            ("t", "p", "CI low", "CI high", "d_z", "perm p"), expected_values, computed[metric]
        ):
            if not math.isclose(actual_value, expected_value, rel_tol=CHECK_RTOL, abs_tol=1e-9):
                mismatches.append(
                    f"{metric} {name}: expected {expected_value}, computed {actual_value:.6f}"
                )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    """Run the paired significance tests and optionally check against the SPEC.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (defaults to sys.argv).

    Returns
    -------
    int
        0 on success, 1 if ``--check`` found a deviation.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any statistic deviates from SPEC 0004",
    )
    arguments = parser.parse_args(argv)

    held_out_names = [fold[0] for fold in FOLDS]
    masked_tra = [fold[1] for fold in FOLDS]
    flash_tra = [fold[2] for fold in FOLDS]
    masked_aogm = [fold[3] for fold in FOLDS]
    flash_aogm = [fold[4] for fold in FOLDS]

    tra = paired_statistics(masked_tra, flash_tra)
    aogm = paired_statistics(masked_aogm, flash_aogm)
    tra_perm_p = sign_flip_permutation_p([f - m for m, f in zip(masked_tra, flash_tra)])
    aogm_perm_p = sign_flip_permutation_p([f - m for m, f in zip(masked_aogm, flash_aogm)])

    print("Paired two-sided tests, flash - masked, n = {} folds ({}):".format(
        len(FOLDS), ", ".join(held_out_names)))
    for metric, result, perm_p in (("TRA", tra, tra_perm_p), ("AOGM", aogm, aogm_perm_p)):
        print(
            f"  {metric}: mean diff {result.mean_difference:+.5f}, "
            f"sd {result.sd_difference:.5f}, t {result.t_statistic:+.3f}, "
            f"p {result.p_value:.4f}, 95% CI [{result.ci_low:+.5f}, {result.ci_high:+.5f}], "
            f"d_z {result.cohens_dz:+.3f}, exact sign-flip permutation p {perm_p:.4f}"
        )
    print("\nLaTeX row for tab:cv_results:")
    print(f"  {latex_table_row(tra, aogm)}")

    if arguments.check:
        sys.stdout.flush()
        mismatches = check_against_spec(tra, aogm, tra_perm_p, aogm_perm_p)
        if mismatches:
            print("\nCHECK FAILED:", file=sys.stderr)
            for mismatch in mismatches:
                print(f"  {mismatch}", file=sys.stderr)
            return 1
        print("\nCHECK PASSED: all statistics match SPEC 0004.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
