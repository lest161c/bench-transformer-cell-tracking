#!/usr/bin/env python3
"""Bootstrapped version of the multi-seed K-sweep metrics figure.

The figure shows how the tracking metrics change with the neighbour count
$K$, averaged over all six vanvliet conditions — the configuration-ranking
view, in which every configuration is evaluated on the same data. The band
is a 95% two-level cluster bootstrap CI (sequences within conditions,
conditions as clusters, stratified by training seed) computed from the
per-sequence evaluation files, not from the three per-seed aggregates.

Not plotted (but recorded in the sidecar): the in-sample (five training
conditions) and held-out trpL scopes and the paired
Δ(held-out − in-training) for the five matched configurations. Condition
difficulty dominates absolute cross-condition comparisons (trpL is among
the easiest conditions), so the generalization prior ("held-out performs
worse") is only expressible within trpL; the report states that test in
the text rather than the figure.

Data: ``results/knn_sweep_multiseed/runs/<run>/eval_<COND>.csv`` (JSON
``{sequence: {TRA, AOGM}}``). Held-out arm: newest post-fix run per
(config, seed) with the SPEC 0039 provenance gate enforced (same
resolution as ``aggregate_multiseed_eval.py``). In-training arm: newest
pre-fix (2026-09) run per (config, seed) — checkpoints trained WITH
trpL; these eval files predate the provenance gate, so a missing sidecar
is reported but not fatal there.

Output: ``report-shorten/figs/knn_sweep_metrics_bootstrap.{pdf,svg}``
plus a ``.meta.json`` sidecar with the point estimates, CI bounds, and
the bootstrap configuration. Deterministic: numpy default_rng(42),
B = 10000.

Run from the repository root::

    python3 report-shorten/scripts/generate_knn_sweep_bootstrap_figure.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "betterplots"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from betterplots import palette, set_style

set_style(usetex=False, colors="okabe-ito", despine=True,
          font_size=8.5, label_size=8.5, tick_size=7.5, legend_font_size=7.5)

import aggregate_multiseed_eval as agg

RUNS_DIR = REPO_ROOT / "results" / "knn_sweep_multiseed" / "runs"
SUMMARY_CSV = REPO_ROOT / "results" / "knn_sweep_multiseed" / "summary.csv"
FIGS = REPO_ROOT / "report-shorten" / "figs"

INSAMPLE_CONDITIONS = ["rpsM", "recA", "pheA", "metA", "cib"]
B = 10_000
RNG_SEED = 42
CI = (2.5, 97.5)
# Configurations with both arms (post-fix held-out and pre-fix in-training
# trpL evaluations); K=2/8 were only ever trained without trpL.
DELTA_CONFIGS = ["baseline", "k4", "k16", "k32", "k64"]

K_SWEEP_CONFIGS = ["k2", "k4", "k8", "k16", "k32", "k64", "baseline"]
K_SWEEP_TICKS = ["2", "4", "8", "16", "32", "64", "dense"]
OI_BLUE = palette.BLUE


def load_sequences(run_dir: str | Path, condition: str) -> dict[str, tuple[float, float]]:
    """Return {sequence: (TRA, AOGM)} for one eval file, owner exclusions applied."""
    data = json.load(open(Path(run_dir) / f"eval_{condition}.csv"))
    for sequence in agg.EXCLUDED_SEQUENCES:
        data.pop(sequence, None)
    return {seq: (v["TRA"], v["AOGM"]) for seq, v in data.items()}


def newest_prefx_run_dir(config: str, seed: int) -> str | None:
    """Newest pre-fix (2026-09) run dir for (config, seed) with eval files.

    The pre-fix checkpoints were trained WITH trpL (before the SPEC 0030
    launcher fix) and provide the in-training trpL reference. Date-filtered
    by the leading YYYY-MM of the run-dir name.
    """
    candidates = sorted(
        path for path in RUNS_DIR.glob(f"*_{config}_s{seed}")
        if path.name.split("_")[0] < "2026-10"
        and (path / f"eval_trpL.csv").is_file()
    )
    return str(candidates[-1]) if candidates else None


def load_arms() -> tuple[dict, dict]:
    """Load the held-out (clean) and in-training (pre-fix) per-sequence data.

    Returns
    -------
    tuple[dict, dict]
        ``clean[(config, seed)][scope]`` with scope ``"insample"`` mapping
        condition -> {sequence: (TRA, AOGM)} and scope ``"trpL"`` mapping
        sequence -> (TRA, AOGM); ``prefix[(config, seed)]`` mapping
        sequence -> (TRA, AOGM) for the pre-fix trpL evaluations.
    """
    clean: dict = {}
    prefix: dict = {}
    for config in agg.CONFIGS:
        for seed in agg.SEEDS:
            run_dir = agg.completed_run_dir(config, seed, RUNS_DIR)
            if run_dir is None:
                print(f"  ⚠ no eval files for {config} s{seed} — skipping")
                continue
            insample = {}
            for condition in INSAMPLE_CONDITIONS:
                eval_path = Path(run_dir) / f"eval_{condition}.csv"
                if agg.check_provenance(str(eval_path)) is None:
                    raise SystemExit(f"provenance gate failed: {eval_path}")
                insample[condition] = load_sequences(run_dir, condition)
            eval_path = Path(run_dir) / f"eval_trpL.csv"
            if agg.check_provenance(str(eval_path)) is None:
                raise SystemExit(f"provenance gate failed: {eval_path}")
            clean[(config, seed)] = {"insample": insample,
                                     "trpL": load_sequences(run_dir, "trpL")}
            pre_dir = newest_prefx_run_dir(config, seed)
            if pre_dir is None:
                continue
            if not (Path(pre_dir) / "eval_trpL.csv.provenance.json").is_file():
                print(f"  note: pre-fix run {Path(pre_dir).name} has no "
                      f"provenance sidecar (legacy arm, not gated)")
            prefix[(config, seed)] = load_sequences(pre_dir, "trpL")
    return clean, prefix


def validate_against_summary(clean: dict) -> None:
    """Cross-check the three-seed means of the clean arm against summary.csv."""
    if not SUMMARY_CSV.is_file():
        print("  note: summary.csv not found — skipping cross-check")
        return
    rows = _read_summary_rows()
    summary = {(r["config"], int(r["seed"])): r for r in rows}
    worst = 0.0
    for (config, seed), scopes in clean.items():
        row = summary.get((config, seed))
        if row is None:
            continue
        insample_tra = _scope_mean(
            [[v[0] for v in seqs.values()] for seqs in scopes["insample"].values()])
        insample_aogm = _scope_mean(
            [[v[1] for v in seqs.values()] for seqs in scopes["insample"].values()])
        worst = max(worst, abs(insample_tra - float(row["tra_insample"])),
                    abs(insample_aogm - float(row["aogm_insample"])),
                    abs(np.mean([v[0] for v in scopes["trpL"].values()])
                        - float(row["tra_trpl_heldout"])),
                    abs(np.mean([v[1] for v in scopes["trpL"].values()])
                        - float(row["aogm_trpl_heldout"])))
    if worst > 1e-6:
        print(f"  ⚠ clean-arm means deviate from summary.csv by {worst:.2e}")
    else:
        print(f"  ✓ per-sequence aggregation reproduces summary.csv "
              f"(max deviation {worst:.2e})")


def _read_summary_rows() -> list[dict]:
    """Read the K-sweep summary CSV into a list of row dicts."""
    import csv
    with open(SUMMARY_CSV) as handle:
        return list(csv.DictReader(handle))


def _scope_mean(sequence_lists: list[list[float]]) -> float:
    """Mean of per-condition means (the aggregator's in-sample scope)."""
    return float(np.mean([np.mean(values) for values in sequence_lists]))


def bootstrap_sequential(seed_draws: list[list[float]], rng: np.random.Generator,
                         b: int = B) -> np.ndarray:
    """Sequence bootstrap stratified by seed.

    Parameters
    ----------
    seed_draws : list[list[float]]
        One list of per-sequence values per seed (lengths may differ).
    rng : numpy Generator
    b : int
        Number of bootstrap replicates.

    Returns
    -------
    numpy.ndarray
        Shape (b,) bootstrap distribution of the across-seed mean.
    """
    arrays = [np.asarray(values) for values in seed_draws]
    per_seed = [rng.choice(a, size=(b, len(a)), replace=True).mean(axis=1)
                for a in arrays]
    return np.mean(per_seed, axis=0)


def bootstrap_cluster(conditions: dict[str, dict[str, list[float]]],
                      rng: np.random.Generator, b: int = B) -> np.ndarray:
    """Two-level cluster bootstrap for the in-sample scope.

    Resamples sequences within each condition, then resamples the
    conditions as clusters; stratified by seed (one independent draw per
    seed, averaged across seeds).
    """
    names = sorted(conditions)
    per_seed = []
    for seed_index in range(len(next(iter(conditions.values())))):
        # condition_draws[c][b]: bootstrap mean of condition c for this seed
        condition_draws = {}
        for name in names:
            values = conditions[name][seed_index]
            draws = rng.choice(values, size=(b, len(values)), replace=True).mean(axis=1)
            condition_draws[name] = draws
        clusters = rng.integers(0, len(names), size=(b, len(names)))
        gathered = np.stack([np.take(condition_draws[name], clusters[:, j])
                             for j, name in enumerate(names)])
        per_seed.append(gathered.mean(axis=0))
    return np.mean(per_seed, axis=0)


def bootstrap_paired(heldout: list[list[float]], training: list[list[float]],
                     rng: np.random.Generator, b: int = B) -> np.ndarray:
    """Paired sequence bootstrap within seed on shared trpL sequences.

    ``heldout[s]`` and ``training[s]`` are aligned per-sequence lists
    (same sequence order per seed); each replicate resamples the sequence
    indices once and applies them to both arms.
    """
    per_seed = []
    for h_values, t_values in zip(heldout, training):
        h = np.asarray(h_values)
        t = np.asarray(t_values)
        idx = rng.integers(0, len(h), size=(b, len(h)))
        per_seed.append(h[idx].mean(axis=1) - t[idx].mean(axis=1))
    return np.mean(per_seed, axis=0)


def ci(draws: np.ndarray) -> tuple[float, float]:
    """Percentile confidence interval of a bootstrap distribution."""
    return tuple(float(np.percentile(draws, p)) for p in CI)


def collect_statistics(clean: dict, prefix: dict) -> dict:
    """Compute point estimates and bootstrap CIs for every (config, scope)."""
    rng = np.random.default_rng(RNG_SEED)
    stats: dict = {}
    for config in agg.CONFIGS:
        seeds = [seed for seed in agg.SEEDS if (config, seed) in clean]
        if not seeds:
            continue
        for metric, index in (("tra", 0), ("aogm", 1)):
            # held-out trpL: sequence bootstrap within seed
            heldout = [[clean[(config, s)]["trpL"][seq][index]
                        for seq in sorted(clean[(config, s)]["trpL"])]
                       for s in seeds]
            point = float(np.mean([np.mean(v) for v in heldout]))
            draws = bootstrap_sequential(heldout, rng)
            stats[(config, "trpl_heldout", metric)] = {
                "mean": point, "ci": ci(draws)}
            # in-sample: two-level cluster bootstrap
            condition_values = {
                cond: [[clean[(config, s)]["insample"][cond][seq][index]
                        for seq in sorted(clean[(config, s)]["insample"][cond])]
                       for s in seeds]
                for cond in INSAMPLE_CONDITIONS}
            insample_draws = bootstrap_cluster(condition_values, rng)
            insample_point = float(np.mean([
                _scope_mean([[clean[(config, s)]["insample"][cond][seq][index]
                              for seq in sorted(clean[(config, s)]["insample"][cond])]
                             for cond in INSAMPLE_CONDITIONS])
                for s in seeds]))
            stats[(config, "insample", metric)] = {
                "mean": insample_point, "ci": ci(insample_draws)}
            # grand average over all six conditions ("avg on ALL"): the
            # configuration-ranking metric, recorded but not plotted
            all_conditions = dict(condition_values)
            all_conditions["trpL"] = heldout
            per_seed_all = [float(np.mean([np.mean(all_conditions[name][s])
                                           for name in all_conditions]))
                            for s in range(len(seeds))]
            stats[(config, "all_conds", metric)] = {
                "mean": float(np.mean(per_seed_all)),
                "ci": ci(bootstrap_cluster(all_conditions, rng))}
    # paired Δ for the matched configurations
    for config in DELTA_CONFIGS:
        seeds = [seed for seed in agg.SEEDS
                 if (config, seed) in clean and (config, seed) in prefix]
        if len(seeds) < len(agg.SEEDS):
            print(f"  ⚠ {config}: incomplete pre-fix arm "
                  f"({len(seeds)}/{len(agg.SEEDS)} seeds) — Δ skipped")
            continue
        for metric, index in (("tra", 0), ("aogm", 1)):
            clean_arm = {seed: clean[(config, seed)]["trpL"] for seed in seeds}
            prefix_arm = {seed: prefix[(config, seed)] for seed in seeds}
            heldout, _ = _aligned(clean_arm, index)
            training, _ = _aligned(prefix_arm, index)
            per_seed_delta = []
            for s, h_values, t_values in zip(seeds, heldout, training):
                point = float(np.mean(h_values) - np.mean(t_values))
                per_seed_delta.append(point)
            draws = bootstrap_paired(heldout, training, rng)
            stats[(config, "delta_heldout_minus_intraining", metric)] = {
                "mean": float(np.mean(per_seed_delta)),
                "ci": ci(draws),
                "per_seed": dict(zip(map(str, seeds), per_seed_delta)),
            }
            # in-training marginal series (plotted alongside held-out)
            stats[(config, "trpl_intraining", metric)] = {
                "mean": float(np.mean([np.mean(v) for v in training])),
                "ci": ci(bootstrap_sequential(training, rng))}
    return stats


def _aligned(per_seed: dict[int, dict[str, tuple[float, float]]],
             index: int) -> tuple[list[list[float]], list[str]]:
    """Align per-seed sequence maps on the shared sequence set."""
    shared = sorted(set.intersection(*(set(d) for d in per_seed.values())))
    values = [[per_seed[seed][seq][index] for seq in shared]
              for seed in sorted(per_seed)]
    return values, shared


def add_horizontal_grid(ax) -> None:
    """Only horizontal grid lines (guideline §5)."""
    ax.grid(True, axis="y", color="#cccccc", alpha=0.6, linewidth=0.4)
    ax.grid(False, axis="x")


def remove_top_right_spines(ax) -> None:
    """Remove top and right spines (belt-and-suspenders next to despine)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figsize_wide(aspect: float = 1.6) -> tuple[float, float]:
    """Compact two-panel size (guideline §7)."""
    width = 4.8
    return width, width / aspect


def plot(stats: dict) -> None:
    """Render the bootstrapped K-sweep figure into ``report-shorten/figs``."""
    positions = list(range(len(K_SWEEP_CONFIGS)))
    fig, axes = plt.subplots(1, 2, figsize=figsize_wide(2.4),
                             gridspec_kw={"wspace": 0.45})
    metrics = [("tra", "TRA"), ("aogm", "AOGM")]
    values_meta = []
    for ax, (metric, y_label) in zip(axes, metrics):
        xs, means, lows, highs = [], [], [], []
        for pos, config in zip(positions, K_SWEEP_CONFIGS):
            entry = stats.get((config, "all_conds", metric))
            if entry is None:
                continue
            low, high = entry["ci"]
            xs.append(pos)
            means.append(entry["mean"])
            lows.append(low)
            highs.append(high)
        ax.fill_between(xs, lows, highs, color=OI_BLUE, alpha=0.18,
                        linewidth=0, zorder=1)
        ax.plot(xs, means, color=OI_BLUE, marker="o", markersize=3.2,
                linewidth=1.1, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(K_SWEEP_TICKS, fontsize=7)
        ax.set_xlabel("Neighbour count $K$")
        ax.set_ylabel(y_label)
        add_horizontal_grid(ax)
        remove_top_right_spines(ax)
        lows = [stats[(c, "all_conds", metric)]["ci"][0]
                for c in K_SWEEP_CONFIGS if (c, "all_conds", metric) in stats]
        highs = [stats[(c, "all_conds", metric)]["ci"][1]
                 for c in K_SWEEP_CONFIGS if (c, "all_conds", metric) in stats]
        if metric == "tra":
            span = max(highs) - min(lows)
            ax.set_ylim(min(lows) - 0.15 * span, max(highs) + 0.15 * span)
        else:
            ax.set_ylim(0, max(highs) * 1.08)
    for config in DELTA_CONFIGS:
        for metric in ("tra", "aogm"):
            entry = stats.get((config, "delta_heldout_minus_intraining", metric))
            if entry is not None:
                values_meta.append({"config": config,
                                    "metric": f"delta_{metric}_heldout_minus_intraining",
                                    "mean": entry["mean"],
                                    "ci95_low": entry["ci"][0],
                                    "ci95_high": entry["ci"][1],
                                    "per_seed": entry["per_seed"]})
    for config in agg.CONFIGS:
        for metric in ("tra", "aogm"):
            for scope in ("insample", "all_conds"):
                entry = stats.get((config, scope, metric))
                if entry is not None:
                    values_meta.append({"config": config,
                                        "metric": f"{metric}_{scope}",
                                        "mean": entry["mean"],
                                        "ci95_low": entry["ci"][0],
                                        "ci95_high": entry["ci"][1]})
    fig.subplots_adjust(bottom=0.18)
    FIGS.mkdir(parents=True, exist_ok=True)
    meta = {
        "sources": [str(RUNS_DIR), str(SUMMARY_CSV)],
        "values": values_meta,
        "bootstrap": {
            "replicates": B,
            "rng_seed": RNG_SEED,
            "ci_percentiles": list(CI),
            "trpl_heldout": "sequence bootstrap within seed",
            "trpl_intraining": "sequence bootstrap within seed (pre-fix checkpoints)",
            "insample": ("two-level cluster bootstrap: sequences within "
                         "conditions, conditions as clusters; stratified by seed"),
            "all_conds": ("grand average over all six conditions; two-level "
                          "cluster bootstrap as for insample; recorded, not plotted"),
            "delta": ("paired sequence bootstrap within seed on the shared "
                      "trpL sequences (post-fix held-out vs pre-fix "
                      "in-training checkpoints)"),
            "excluded_sequences": list(agg.EXCLUDED_SEQUENCES),
        },
    }
    name = "knn_sweep_metrics_bootstrap"
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    sidecar = {"figure": name,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "generator": "report-shorten/scripts/generate_knn_sweep_bootstrap_figure.py",
               **meta}
    (FIGS / f"{name}.meta.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"  ✓ {name}.pdf + .svg + .meta.json written to {FIGS}")


def main() -> None:
    """Load the per-sequence evaluations and render the bootstrapped figure."""
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(
            f"mirrored eval tree not found: {RUNS_DIR} — rsync it as "
            "documented in scripts/aggregate_multiseed_eval.py")
    clean, prefix = load_arms()
    validate_against_summary(clean)
    stats = collect_statistics(clean, prefix)
    plot(stats)
    for (config, scope, metric), entry in sorted(stats.items()):
        low, high = entry["ci"]
        print(f"  {config:8s} {scope:30s} {metric}: {entry['mean']:.4f} "
              f"[{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
