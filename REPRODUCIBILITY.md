# Reproducibility

This document maps **every figure and table in `report/`** to the script that
produces it, the command to run it, and the data artifact it consumes, so an
outsider can regenerate the report's evidence from the repository.

The report's appendix "Reproducibility" section points here.

## Environments

| Purpose | Interpreter / venv | Notes |
|---|---|---|
| Attention benchmarks (`benchmark_attn/`) | `benchmark_attn/.venv` | torch + CUDA; some scripts need a GPU |
| SSL / edge probing / pretext (`benchmark_ssl/`) | `benchmark_ssl/.venv` | torch; CPU-ok for the pretext analysis |
| Training + eval (`benchmark_training/`) | `benchmark_training/.venv` (on the cluster) | Trackastra training/eval |
| Report build | `latexmk` | `cd report && latexmk -pdf main.tex` |
| Figures | `benchmark_attn/.venv` or `benchmark_ssl/.venv` | matplotlib/seaborn; betterplots from `betterplots/` |

Cluster jobs are SLURM scripts (`*.slurm`); they assume `$BENCH`, `$DATA_DIR`,
and `~/.bench.env` (see the header of each). Raw run trees and result CSVs are
git-ignored; reproduce them by running the jobs, not by copying files.

## Report build

```bash
cd report && latexmk -pdf -g main.tex        # -> report/main.pdf
```

## Figures

Every figure in `report/figs/` has a `<name>.meta.json` sidecar recording its
generator and source data; regenerate with the generator below.

| Figure (report) | Generator script | Command |
|---|---|---|
| `01_speed_vs_n_h100{,_detail}` | `curated/generate_report.py` | `python curated/generate_report.py` |
| `04_memory_vs_n`, `16_dist_cache_effect` | `curated/generate_report.py` | same |
| `09_per_epoch_time`, `11_profiler_breakdown` | `curated/generate_report.py` | same |
| `15_amdahl_law` | `benchmark_attn/analysis/generate_amdahl_figure.py` | `python benchmark_attn/analysis/generate_amdahl_figure.py` |
| `training_step_k_sweep{,_summary,_memory}` | `results/generate_plots.py` | `python results/generate_plots.py` — **removed from the report (SPEC 0040, deprecated A500 era); artifact kept** |
| `backward_pass` | `results/generate_plots.py` | same |
| `knn_sweep_metrics`, `knn_sweep_epoch_time` | `results/generate_plots.py` | same |
| `11_profiler_breakdown`, `09_per_epoch_time` | `curated/generate_report.py` | same — **removed from the report (SPEC 0040); artifact kept** |
| `epoch_decomposition` | `results/generate_plots.py` | same; sources `results/epoch_profile/*/epoch_profile.json` + `trace_aggregation.json` (SPEC 0041, cluster job 4422737) |
| `pipeline_breakdown` | `results/generate_plots.py` | **removed from the report (2026-10-16): falsified premise, superseded by `epoch_decomposition`; artifact kept** |
| `edge_probe_bal_acc`, `edge_probe_f1`, `05_convergence`, `06_tra`, `07_aogm`, `08_edge_div_f1`, `13_ssl_negative` | `results/generate_plots.py` | same |
| `masked_fraction.png` | `benchmark_training/scripts/analysis/cutoff_masked_fraction.py` | `python benchmark_training/scripts/analysis/cutoff_masked_fraction.py` |
| `18_feature_injection_comparison` | `benchmark_ssl/src/analysis/generate_feature_injection_figure.py` | `python benchmark_ssl/src/analysis/generate_feature_injection_figure.py` |

`results/generate_plots.py` and `curated/generate_report.py` regenerate all
their panels in one run; each panel function has a docstring naming its data
source.

## Tables and headline numbers

| Table / claim | Data artifact | Producing script / job |
|---|---|---|
| `tab:knn_sweep_multi` (K-sweep, §3.2) | `results/knn_sweep_multiseed/summary.csv` | train: `benchmark_training/scripts/slurm/run_knn_sweep.slurm`; eval: `run_eval_multiseed.slurm`; aggregate: `python benchmark_training/scripts/analysis/aggregate_multiseed_eval.py` |
| `tab:cv_results` (cutoff ablation, §3.3) | `results/cv_rerun/cv500_decomp/collection.json` (per-run summaries incl. AOGM-A/AOGM-D decomposition; raw runs on the cluster) | train/eval: `run_cv.slurm`; decomposition re-eval: `run_cv_reeval_decomp.slurm`; paired stats: `python benchmark_training/scripts/analysis/cv_paired_significance.py [--check]` (**STALE: hardcodes the withdrawn 100-epoch table — do not use for the current table**) |
| §3.3 FlexAttention quadratic-time finding (block-mask density invariance, sorted-token control) | `benchmark_attn/results/flex_reorder_b64.csv` + `flex_reorder_b64_verdict.json` (cluster job 4484607) | `benchmark_attn/scripts/benchmarks/benchmark_flex_reorder.py` (slurm: `run_flex_reorder.slurm`) |
| §3.4 inference attention share (~60% of GPU kernel time, fused kernel 2.4%) | `results/epoch_profile/dense_prof_s42/fit-profile-{validation,training}_step.pt.trace.json` | `python benchmark_training/scripts/analysis/attn_share_inference.py --trace <trace.json>` |
| `tab:feature_injection`, `tab:app_feature_injection` (Et-Ultra, §3.8) | `benchmark_ssl/results/etultra_eval/*/summary_mean_std.csv` | train: `benchmark_ssl/slurm/train_etultra_dynaclr.slurm` (wrapper `run_train.py`); eval: `benchmark_ssl/slurm/eval/eval_etultra.slurm` (`eval_etultra.py`) |
| `tab:app_edge` (edge probing, §3.7) | `results/probes/unified_probe_results_cv10{,_s43,_s44}.json` (seeds 42/43/44; seed reruns via `results/probes/run_seed_reruns.sh`) | `benchmark_ssl/slurm/unified_edge_probe.slurm` (`src/edge_probing/unified_edge_probe.py`) |
| `tab:app_pretext` (§3.6) | `benchmark_ssl/results/pretext_coordinate_solvability.json` | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_coordinate_solvability.py` |
| `sec:pretext` corrected-pilot numbers (§3.6, shorten report) | `benchmark_ssl/results/runs/pretext_sample_efficiency_test/result.json` | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_sample_efficiency_test.py` |
| `sec:pretext` bug-exclusion audit (§3.5, shorten report; input-corruption metrics + trpL/rpsM differential SSL) | `benchmark_ssl/results/pretext_collapse_bug_exclusion.json` (**now holds the 2026-10-29 rerun**: geometry/appearance still green, but the rpsM BCE control arm collapsed (0.185/0.113) where the 2026-10-28 run had escaped (0.098/0.890, numbers preserved in `labbook/2026-10-23_pretext_failure_diagnosis.md` and `labbook/2026-10-28_pretext_collapse_bug_exclusion.md`); the original artifact was overwritten by the rerun) | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_collapse_bug_exclusion.py` |
| `sec:pretext` per-anchor objective probe (§3.5, shorten report; row-CE remedy vs BCE, 2x2 arms, argmax checkpoints; BCE escape bistability) | `benchmark_ssl/results/pretext_peranchor_probe.json` (+ `pretext_peranchor_probe_run1_2026-10-29.json`, first run) | `benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_peranchor_probe.py` |
| `tab:app_dataset_stats` | `results/dataset_stats/*.json` | `python benchmark_training/scripts/analysis/count_cell_instances.py` |
| `tab:app_per_epoch` | `results/.../epoch_times.json` | `python benchmark_training/scripts/analysis/extract_epoch_times.py` (run on the cluster) |
| Attention forward numbers (§3.1, §3.5) | `benchmark_attn/results/full_bench_v2_h100_cache_{on,off}.csv` | `benchmark_attn/slurm/run_full_bench_v2.slurm` (`scripts/benchmarks/benchmark_sweep.py`) |
| Backward numbers (§3.5) | `benchmark_attn/results/sparse_backward_results_v2.csv` | `benchmark_attn/scripts/benchmarks/benchmark_backward.py` (H100) |
| ~~K-sweep training-step numbers (appendix)~~ | `benchmark_attn/results/speed_mem_v2.csv` | removed from the report (SPEC 0040, deprecated A500 era); CSV kept |
| Cross-dataset DeepCell (K-sweep, §3.9) | `results/cross_dataset/deepcell/summary.json` | `benchmark_ssl/slurm/cross_dataset_eval.slurm` |
| Reference (his published) rows | `papers/cyril_thesis.txt` Tables 3.4/3.5 | transcribed in `research/15-achard-thesis-deep-read.md` |

## Cluster jobs (SLURM)

| Job | Produces |
|---|---|
| `benchmark_attn/slurm/run_full_bench_v2.slurm` | forward benchmark CSVs (§3.1) |
| `benchmark_training/scripts/slurm/run_knn_sweep.slurm` | K-sweep training (§3.2) |
| `benchmark_training/scripts/slurm/run_eval_multiseed.slurm` | K-sweep eval |
| `benchmark_training/scripts/slurm/run_cv.slurm` | cutoff ablation (DenseFlash arm needs DenseFlashAttention installed via `benchmark_training/scripts/slurm/install_dense_flash.py`) |
| `benchmark_ssl/slurm/unified_edge_probe.slurm` | edge probing (§3.7) |
| `benchmark_ssl/slurm/train_etultra_dynaclr.slurm` + `slurm/eval/eval_etultra.slurm` | Et-Ultra comparison (§3.8) |
| `benchmark_ssl/slurm/cross_dataset_eval.slurm` | DeepCell cross-dataset (§3.9) |

## Known gaps

- `report/figs/15_amdahl_law.pdf` has **no `.meta.json` sidecar** by design: it is
  a purely theoretical formula plot (`benchmark_attn/analysis/generate_amdahl_figure.py`,
  no external data). It is not referenced by an `\includegraphics` in the report
  body (the Amdahl result is prose only).
- The SPEC 0033 supporting diagnostics **`ptfeat_proj` projection inspection**
  (effective rank 51/128) and the **sampled-cosine geometry** were computed ad hoc
  during the SPEC 0033 session and have **no stored artifact**; the report
  therefore cites only the outcomes that do have artifacts (feature-usage
  controls, cosine-only AOGM 88332). Per-video node/edge counts are pulled locally
  (`benchmark_ssl/results/etultra_eval/*_summary_per_video.csv`).
- Most raw result artifacts (run trees, CSVs under `results/` and
  `benchmark_*results/`) are **git-ignored** and were produced on Capella;
  reproduction requires cluster access and re-running the jobs above. The
  `.meta.json` sidecars record the exact source paths used.
- Two DeepCell rows are **flagged/anomalous** and should not be read as feature
  quality: the Et-Ultra DeepCell arm (§3.8, node-loss artefact) and the
  K-sweep DeepCell subset note in §3.9.
- `CachedDistAttention` and the `knn_neighbors=-2` dense-flash path depend on the
  Trackastra checkout; the dense-flash support is installed via
  `install_dense_flash.py` (see the file's docstring).
