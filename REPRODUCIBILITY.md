# Reproducibility Index

Maps every figure, table, and headline number of the two report trees to the
data artifact it consumes and the exact command that produces it. The report
appendices point here.

- **`report-shorten/`** is the live deliverable (§1–§2). **`report/`** is the
  frozen full version; only its deltas are listed (§3).
- All commands run from the repository root. Interpreters: see Environments.
- Artifact status: ◇ = local-only (git-ignored, exists on this workstation) ·
  ☁ = produced on the cluster (re-run the job to recreate) · — = no data
  artifact (hand-written or formula-only).
- SPEC numbers and per-experiment narratives live in `labbook/` and
  `docs/agent-state/` (local-only by policy); this index does not repeat them.

## Environments

| Purpose | Interpreter / venv | Notes |
|---|---|---|
| Attention benchmarks (`benchmark_attn/`) | `benchmark_attn/.venv` | torch + CUDA; scripts need a GPU |
| SSL / edge probing / pretext (`benchmark_ssl/`) | `benchmark_ssl/.venv` | torch; CPU-ok for the pretext analyses |
| Training + eval (`benchmark_training/`) | `benchmark_training/.venv` (cluster) | Trackastra training/eval |
| Report build | `latexmk` | `cd report-shorten && latexmk -pdf main.tex` |
| Figures | `benchmark_attn/.venv` or `benchmark_ssl/.venv` | matplotlib/seaborn; palette from `betterplots/` |

SLURM jobs assume `$BENCH`, `$DATA_DIR`, and `~/.bench.env` (see each header).

## 1. Live report figures (`report-shorten/`)

| Figure | Data artifact | Generator | Status |
|---|---|---|---|
| `fig:speed_vs_n`, `fig:app_speed_detail` | `benchmark_attn/results/full_bench_v2_h100_cache_on.csv` | `python curated/generate_report.py` | ◇ |
| `fig:memory_vs_n` | `benchmark_attn/results/full_bench_v2_h100_cache_{on,off}.csv` | `python curated/generate_report.py` | ◇ |
| `fig:dist_cache_effect` | `benchmark_attn/results/full_bench_v2_h100_cache_{on,off}.csv` | `python curated/generate_report.py` | ◇ |
| `fig:mask_decomposition` | `benchmark_attn/results/cudnn_vs_fa2_mask_decomposition.csv` | `python curated/generate_report.py` | ◇ |
| `fig:knn_sweep_metrics_bootstrap` | `results/knn_sweep_multiseed/` per-sequence evals | `python benchmark_training/scripts/analysis/generate_knn_sweep_bootstrap_figure.py` | ◇ |
| `fig:app_masked_fraction` | vanvliet TRA masks | `python benchmark_training/scripts/analysis/cutoff_masked_fraction.py` | ◇ |
| `fig:probe_mlp` | — (TikZ schematic, no data) | hand-drawn in `methodology.tex` | — |

`curated/generate_report.py` regenerates all of its panels in one run; the
wrapper `python benchmark_training/scripts/analysis/generate_report_figures.py`
regenerates the corrected H100/memory/dist-cache set straight into
`report-shorten/figs`. Every written figure carries a `.meta.json` sidecar
with its exact sources.

## 2. Live report tables & headline numbers

| Element | Data artifact | Generator | Status |
|---|---|---|---|
| `tab:knn_sweep_multi` (§3.2) | `results/knn_sweep_multiseed/summary.csv` | train `benchmark_training/scripts/slurm/run_knn_sweep.slurm` → eval `run_eval_multiseed.slurm` → aggregate `python benchmark_training/scripts/analysis/aggregate_multiseed_eval.py` | ☁ + ◇ |
| `tab:cv_results` (§3.3) | `results/cv_rerun/cv500_decomp/collection.json` | `benchmark_training/scripts/slurm/run_cv.slurm` + `run_cv_reeval_decomp.slurm` | ☁ |
| `tab:edge_probe_knn` (§3.7) | `benchmark_ssl/results/probes/knn/cluster/{frame_cv,loco}/summary.csv` | `benchmark_ssl/slurm/run_knn_probe.slurm` (`src/analysis/feature_probe_knn.py`) | ☁ |
| `tab:feature_injection` (§3.8) | `benchmark_ssl/results/etultra_eval/*/summary_mean_std.csv` | `benchmark_ssl/slurm/train_etultra_dynaclr.slurm` + `slurm/eval/eval_etultra.slurm` | ☁ |
| `tab:app_pretext` (App. F) | `benchmark_ssl/results/pretext_coordinate_solvability.json` | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_coordinate_solvability.py` | ◇ |
| §3.5 corrected-pilot numbers | `benchmark_ssl/results/runs/pretext_sample_efficiency_test/result.json` | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_sample_efficiency_test.py` | ◇ |
| §3.5 bug-exclusion audit | `benchmark_ssl/results/pretext_collapse_bug_exclusion.json` (holds the 2026-10-29 rerun; the 2026-10-28 escape numbers are preserved in the labbook) | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_collapse_bug_exclusion.py` | ◇ |
| §3.5 per-anchor objective probe | `benchmark_ssl/results/pretext_peranchor_probe.json` (+ `_run1_2026-10-29.json` first run) | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_peranchor_probe.py` | ◇ |
| `tab:app_dataset_stats` (App.) | `results/dataset_stats/*.json` | `python benchmark_training/scripts/analysis/count_cell_instances.py` | ◇ |
| §3.4 inference attention share | `results/epoch_profile/dense_prof_s42/fit-profile-*.pt.trace.json` | `python benchmark_training/scripts/analysis/attn_share_inference.py --trace <trace.json>` | ◇ |
| §3.3 FlexAttention quadratic-time finding | `benchmark_attn/results/flex_reorder_b64{,_verdict}.json` ... `.csv` | `benchmark_attn/scripts/benchmarks/benchmark_flex_reorder.py` (slurm `benchmark_attn/scripts/slurm/run_flex_reorder.slurm`) | ☁ |
| §3.5 Amdahl cap (prose, no figure) | inputs are the CSVs above | `python benchmark_attn/analysis/generate_amdahl_figure.py` (writes `report-shorten/figs`; not `\includegraphics`'d) | ◇ |
| §3.5 forward/backward equivalence numbers | `benchmark_attn/results/sparse_backward_results_v2.csv` | `benchmark_attn/scripts/benchmarks/benchmark_backward.py` (H100) | ◇ |
| `tab:attn_names`, `tab:complexity`, `tab:app_impl_config`, `tab:app_flex_setup` | — | hand-written (naming/theory/config tables) | — |
| Published reference rows (Achard) | `papers/cyril_thesis.txt` Tables 3.4/3.5 | transcribed in `docs/agent-state/research/15-achard-thesis-deep-read.md` | ◇ |

## 3. Frozen full report (`report/`) — deltas only

- Extra figures: `19_b64_saturation` (`python curated/generate_report.py`,
  B=64 saturation CSVs) and `knn_sweep_metrics`
  (`python results/generate_plots.py`).
- Extra tables: `tab:app_edge` and `tab:app_feature_injection` — artifacts
  `benchmark_ssl/results/probes/unified_probe_results_cv10.json` (◇,
  2026-09-24, **legacy probe protocol**; the live report documents it as
  `sec:probe_protocol_legacy`) and
  `benchmark_ssl/results/etultra_eval/*/summary_mean_std.csv` (as §2).
- Panels removed from the report but still regenerable:
  `training_step_k_sweep*`, `backward_pass`, `epoch_decomposition`,
  `knn_sweep_epoch_time` (`python results/generate_plots.py`);
  `pipeline_breakdown`, `edge_probe_bal_acc/f1`, `05_convergence`–`13_ssl_negative`
  live in `curated/figures/` and `deprecated/legacy_figures/`.
- Legacy multi-seed probe rows referenced `results/probes/run_seed_reruns.sh`
  and `unified_probe_results_cv10_{s43,s44}.json` — **both are gone** (script
  missing, artifacts never created); the legacy rows rest on the single-seed
  `cv10.json` only.

## 4. Cluster jobs (SLURM)

| Job | Produces |
|---|---|
| `benchmark_attn/slurm/run_full_bench_v2.slurm` | forward benchmark CSVs (§3.1) |
| `benchmark_training/scripts/slurm/run_knn_sweep.slurm` | K-sweep training (§3.2) |
| `benchmark_training/scripts/slurm/run_eval_multiseed.slurm` | K-sweep evaluation |
| `benchmark_training/scripts/slurm/run_cv.slurm` | cutoff ablation (DenseFlash arm needs `install_dense_flash.py`) |
| `benchmark_ssl/slurm/run_knn_probe.slurm` | K=8 edge probing (§3.7) |
| `benchmark_ssl/slurm/unified_edge_probe.slurm` | legacy edge probing (frozen `report/` only) |
| `benchmark_ssl/slurm/train_etultra_dynaclr.slurm` + `slurm/eval/eval_etultra.slurm` | Et-Ultra comparison (§3.8) |
| `benchmark_ssl/slurm/cross_dataset_eval.slurm` | DeepCell cross-dataset (§3.9) |

## 5. Staleness & known gaps

- `benchmark_training/scripts/analysis/cv_paired_significance.py` is **stale**:
  it hardcodes the withdrawn 100-epoch table; do not use it for the current
  `tab:cv_results` (which comes from `cv500_decomp/collection.json`).
- `15_amdahl_law` has **no `.meta.json` sidecar** by design (pure formula
  plot, no external data) and is not `\includegraphics`'d — the Amdahl result
  is prose only.
- The projection-inspection (`ptfeat_proj`, effective rank 51/128) and
  sampled-cosine geometry diagnostics were computed ad hoc and have **no
  stored artifact**; the report cites only artifact-backed outcomes there.
- The DeepCell rows for the Et-Ultra arm (§3.8, node-loss artefact) and the
  K-sweep DeepCell subset (§3.9) are flagged/anomalous — do not read them as
  feature quality.
- `CachedDistAttention` and the `knn_neighbors=-2` dense-flash path depend on
  the Trackastra checkout; dense-flash support installs via
  `benchmark_training/scripts/slurm/install_dense_flash.py`.
- Almost all raw artifacts are **git-ignored** and cluster-produced;
  reproduction means re-running the jobs in §4, not copying files. The
  `.meta.json` sidecars record the exact source paths used at generation
  time.
