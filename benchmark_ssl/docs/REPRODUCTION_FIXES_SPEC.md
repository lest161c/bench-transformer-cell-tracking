# SPEC.md — Low-Effort Reproducibility Fixes

*Status: DRAFT — Created 2026-08-13. Intended as a handoff spec for another agent.*

## Goal

Resolve the low-effort known issues listed in `docs/REPRODUCTION.md` §6
("Known discrepancies & reproducibility notes") so that the reproduction guide
is accurate and the output artifacts are self-documenting.

Each issue below is classified by effort level. This spec covers only the
**low-effort** items (estimated ≤ 30 min each).

---

## 1. Issue inventory

| # | Issue | Effort | Status |
|---|---|---|---|
| 1 | `ssl/pretrain_multi.py` references undefined `ROOT` | — | **Already fixed** (`_REPO_ROOT` in `ssl_trainer_multi.py`) |
| 2 | `ssl/downstream_compare.py` is a rewrite — on-disk CSV from old variant | Low | TODO |
| 3 | `ssl/pretrain.py` is a rewrite — `training_log.csv` from old variant | Low | TODO |
| 4 | Cluster artifacts not reproducible locally | High | Out of scope |
| 5 | DINOv2 full SSL never completed | N/A | Cannot fix without running experiment |
| 6 | Feature-set flags not persisted in `feature_signal*` output CSVs | Low | TODO |
| 7 | `generalization_test` / `hard_test` outputs from separate invocations | Low | TODO |

---

## 2. Fix #2 — Document which `downstream_compare.py` variant produced the artifact

### Problem
`REPRODUCTION.md` §4.3 says the on-disk `runs/downstream_compare/comparison.csv`
was produced by the *earlier* variant (fine-tuning, columns `epoch,model,train_loss,train_acc,val_loss,val_acc`).
The current `src/analysis/downstream_comparison.py` is a rewrite that performs
embedding + Hungarian tracking and writes different CSV columns.

### Goal
Make it impossible for a future reader to confuse which variant produced which artifact.

### Acceptance criteria
- [ ] `REPRODUCTION.md` §4.3 explicitly states that the on-disk `comparison.csv`
  was produced by the **old** variant and the current code is a rewrite
- [ ] The current `downstream_comparison.py` docstring or `main()` function
  documents the CSV column schema it produces
- [ ] If the old CSV is still on disk, rename or move it with a clear suffix
  (e.g. `comparison_legacy.csv`) to avoid accidental use

### Notes
- Check `runs/downstream_compare/comparison.csv` to confirm the column schema
  matches the old variant (if the file still exists)
- The current script is at `scripts/run_downstream_comparison.py` → `src/analysis/downstream_comparison.py`

---

## 3. Fix #3 — Document which `pretrain.py` variant produced `ssl_v1` artifacts

### Problem
`REPRODUCTION.md` §4.1 says `runs/ssl_v1/training_log.csv` and the TensorBoard
event were produced by the identity-BCE variant; the current `ssl/pretrain.py`
is an NT-Xent rewrite that writes neither `training_log.csv` nor acc/F1 metrics.
`results_analysis.md` states `d_model=64, nhead=2` but the archived
`runs/ssl_v1/config.yaml` says `d_model=128, nhead=4`.

### Goal
Make it clear which variant produced the archived artifacts so a reproducer
uses the right code version.

### Acceptance criteria
- [ ] `REPRODUCTION.md` §4.1 explicitly states that `ssl_v1` was produced by
  the identity-BCE variant (not the current NT-Xent rewrite)
- [ ] The archived `runs/ssl_v1/config.yaml` is cited as the authoritative
  config for the `ssl_v1` run (not `results_analysis.md`)
- [ ] The current `ssl_pretrainer.py` docstring documents which loss variant
  it implements (identity-BCE vs NT-Xent)

### Notes
- The archived config is at `benchmark_ssl/runs/ssl_v1/config.yaml`
- The current trainer is at `src/training/ssl_pretrainer.py`

---

## 4. Fix #6 — Persist `--feature-set` flag in `feature_signal*` output CSVs

### Problem
`REPRODUCTION.md` §6 note #6 says: "Feature-set flags for the `runs/feature_signal*`
runs are not persisted in the output files; the CSV columns confirm which flag was used."
This means a reader cannot tell which `--feature-set` produced which CSV without
inspecting the column names.

### Goal
Add the `feature_set` value as an explicit column or metadata field in the output
CSV so that the flag is self-documenting in the artifact.

### Acceptance criteria
- [ ] `src/analysis/signal_analyzer.py` writes `feature_set` as a column (or
  metadata row) in `feature_signal_results.csv`
- [ ] The same applies to any other output file produced by the signal analyzer
  (e.g. `feature_signal_report.html`)
- [ ] `REPRODUCTION.md` §6 note #6 is updated to reflect that the flag is now
  persisted

### Implementation hint
The `--feature-set` flag is already parsed at line 775 of `signal_analyzer.py`.
The value is available as `args.feature_set` in `main()`. Add it to the CSV
writer or as a column in the results DataFrame.

---

## 5. Fix #7 — Document the mapping between `--test` sub-runs and output dirs

### Problem
`REPRODUCTION.md` §6 note #7 says: "The `runs/generalization_test` and
`runs/hard_test` outputs were written by separate `ssl/ssl_convergence_test.py`
invocations (`--test generalization`, `--test hard`), not by the `--test all`
run that populated `runs/convergence_prediction`."

A reader trying to reproduce the convergence predictor results may not know which
output directory contains which data.

### Goal
Make the mapping between `--test` values and output directories explicit so a
reproducer knows exactly which command produces which artifacts.

### Acceptance criteria
- [ ] `REPRODUCTION.md` §4.5 (convergence predictor) includes a note mapping
  each `--test` value to its output directory:
  - `--test all` → `runs/convergence_prediction/`
  - `--test generalization` → `runs/generalization_test/`
  - `--test hard` → `runs/hard_test/`
- [ ] `src/analysis/convergence_predictor.py` (or the relevant module) documents
  this mapping in its docstring or `main()` function

### Implementation hint
The `--test` flag is already parsed in the convergence predictor script. Add a
docstring or comment that maps each test mode to its output directory.

---

## 6. Non-goals (out of scope)

- Fixing the cluster artifacts (issue #4) — requires H100 access
- Rerunning the DINOv2 full SSL experiment (issue #5) — requires 24h H100 time
- Refactoring the broader codebase (covered by the separate `SPEC.md` in `docs/`)
- Fixing the `ROOT` reference in `pretrain_multi.py` (already resolved)