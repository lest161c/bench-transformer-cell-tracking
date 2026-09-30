# OPEN_EXPERIMENTS.md

Inventory of every experiment still pending for report finalization (owner scope),
2026-09-25. Governance: anything entering the report needs (1) a SPEC, (2) an
implementation + commit, (3) measured reproducible results in the repo's
README/docs, (4) quantitative remarks in the report.

## A. Cluster experiments (in flight / queued)
- **A1 — Multi-seed sweep training, tasks 6-14 (k16/k32/k64 x seeds 42/43/44).**
  Job `4286428`: k32_s42 + k64_s43 COMPLETED; k16_s42/k32_s43/k32_s44 RUNNING
  (~20 h); k16_s43, k16_s44, k64_s44 FAILED = host-RAM OOM (SIGKILL, ~59 GiB RSS)
  -> resubmitted as `4293505_[7,8,14]`. If OOM recurs: --num-workers 8->4.
  Repo D1 (GRES one-liner) still pending.
- **A2 — In-sample eval of finished models.** First attempt (4286422) failed: the
  subproject venv lacked trackastra (bootstrap editable-install silently skipped).
  FIXED (uv pip install -e $TRK + traccuracy). Relaunched as `4293506` for EIGHT
  finished models (added k32_s42, k64_s43), skip-if-exists guards, resumable.
- **A3 — In-sample eval of the remaining models** (after A1 completes): same
  launcher, extended list.
- **A4 — Multi-seed table integration (report `\pending` P1).** Remains: aggregate
  A2+A3 -> mean+-std per config; re-verdict the "K=16 is best" claim; update
  `tab:knn_sweep`; update Limitations ("All runs used a single random seed...") and
  Future Work ("rest on single-seed runs") — both become stale once the table lands.
  Spec 0008. Do NOT mix per-epoch wall-times across builds.
- **A5 — DeepCell cross-dataset eval of all 15 sweep checkpoints (report `\pending`
  P4).** Depends on A1/A3. Remains: parameterize `cross_dataset_eval.py` hardcoded
  CHECKPOINTS (D4) + cluster run; one row/statement in report. Spec 0007.
- **A6 — Job-3920627 mask artifact.** SUPERSEDED by job 4281334 (isolated creation
  measured, integrated). No run needed; optional provenance search only.
- **A7 — Mask-knn dtype reconciliation** (full_bench ~1050 MB fp16-looking vs new
  creation CSV 2048 MB fp32). Documentation check only; no new run.

## B. Edge-probe statistics (spec 0006) — ALL DONE
- **B1 — DONE (2026-09-24):** probe dumps per-frame-pair metrics
  (--dump-predictions); 29/29 unique pair_ids verified; harness changes committed
  (bd6d736).
- **B2 — DONE (2026-09-24):** refresh run complete (10-fold, all 7 features,
  checkpoint fixed, dump enabled, seed 42). cnn_frozen now evaluated (0.775).
  Stale Aug numbers superseded; REPRODUCTION.md 4.10 rewritten (bd6d736).
- **B3 — DONE (2026-09-24):** per-pair tests computed and --check-verified
  (results/probes/probe_paired_significance.json); fold-level secondary included;
  Holm across feature sets; EXPECTED recorded in spec 0006.
- **B4 — DONE + COMMITTED (2026-09-25):** report text in `report/` 0eec600.
- **B5 — DONE + COMMITTED (0eec600, combined with 0005). \pending P2 removed.
- **B6 — DONE (2026-09-24):** S4.10 rewritten (10-fold + per-pair significance
  tables, commands, provenance note); inventory row updated (bd6d736).
- **B7 — Aug-15 probe variant**: declared SUPERSEDED (older feature family,
  expired-workspace paths, uncited). No action.

## C. Coordinate-free NoPE (spec 0005) — ALL DONE
- **C1 — DONE (ebcba3b + d2bad24):** REPRODUCTION.md NoPE section; tb numbers
  re-verified (best 0.094963, n=142, first 0.5024, final 0.1617).
- **C2 — DONE (0eec600, combined with 0006):** all three "was never run" hedges
  removed, mechanism list includes the confirmed coordinate shortcut,
  abstract/intro/conclusion re-pointed to positional dominance; PDF builds (26 pp).
- **C3 — DONE:** protocol note in REPRODUCTION.md (same val_loss definition, deeper
  fits, no cross-build wall-time comparisons).

## B+. Figure refresh (spec 0009) — DONE (2026-09-25, c45465a)
- edge_probe_bal_acc / edge_probe_f1 regenerated from unified_probe_results_cv10.json
  (7 features, 10-fold; hardcoded old values removed); NEW probe_significance forest
  figure from probe_paired_significance.json (per-pair deltas, bootstrap CIs, Holm p,
  n=29). PDF+SVG in results/figures/; provenance updated. Report inclusion of the new
  figures = owner call (report currently uses tables). Sweep/convergence figures
  deferred to 0008 results.
- **Metadata sidecars — DONE (c62d860):** every figure in results/figures/ has a
  committed <stem>.meta.json (created_utc, git commit, sources, plotted values);
  auto-written by savefig(); phase 2 = port to curated/generate_report.py
  (the report's 13 panels).

## D. Code/infra fixes pending commit
- **D1 — DONE (e75e19f):** repo sweep script now uses `--gres=gpu:1`.
- **D2 — DONE (e75e19f):** committed `run_eval_multiseed.slurm` (15-task array,
  glob-resolved run dirs, resumable per-condition eval, venv + no-cross-build
  notes; supersedes the stopgap script). Note: eval_cv.py writes JSON regardless
  of extension — the aggregator parses JSON.
- **D3 — DONE (2026-09-24):** None-feature hardening in probe_paired_significance.py
  (bd6d736).
- **D4 — DONE (c3dd13b):** `--checkpoints-file` added to cross_dataset_eval.py
  (default behavior unchanged) + slurm usage example. Cluster dry-run pending
  before the DeepCell run.
- **D5 — DONE (2026-09-25):** verifier PASS (backend recovered). Harness +
  significance math verified; report commits 0eec600/142f47d verified (hedges 0,
  numbers match artifacts, no job numbers, no duplication, exactly 2 \pendings
  remain: P1 + P4); --check executed green.

## E. Related pending (non-experiment, for completeness)
- **E1 — Labbook entry** for 2026-09-23/25: 500-epoch-cap finding, QOSMinGRES saga,
  eval-venv fix, OOM diagnosis, probe feature-family change.
- **E2 — Presentation deck** (`presentation/` holds only TUD templates; no talk yet).
- **E3 — DONE (2026-09-25):** vault doc corrected (`capella` + QOS note).
- **E4 — report/ dual-tracking cleanup** (parent repo mirrors report/* files).
- **E5 — DeepCell dataset stat gap** (`tab:app_dataset_stats` "Cell instances: --").
- **E6 — Overleaf sync coordination** for report commits (report repo is synced).
- **E7 — RESOLVED (2026-09-25):** `report/scripts/cv_paired_significance.py` is
  tracked in the authoritative nested `report/` repo; only the parent's mirror
  view is untracked (that is the E4 dual-tracking quirk, not a gap).
- **E8 — Parent push**: 4 commits ahead of origin (0da0bc1, bd6d736, ebcba3b,
  d2bad24); git push is permission-blocked in this environment -> owner runs it.
- **E9 — report/ Overleaf state (2026-09-25):** branch refactor/tud-template is
  IN SYNC with overleaf/main (report commits 0eec600/142f47d already on the
  Overleaf side).
