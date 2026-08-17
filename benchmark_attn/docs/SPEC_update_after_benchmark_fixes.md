# SPEC: Update README, REPRODUCTION, Results CSV, and Plots After Benchmark Fixes

**Date:** 2026-08-17
**Status:** READY FOR IMPLEMENTATION

## 1. Background — What Changed

Two bugs were found and fixed in the benchmark harness:

### Bug 1: `dense_masked` used wrong class (per-layer cdist)

**Old:** Registry mapped `dense_masked` → `RelativePositionalAttention`, which
recomputes `torch.cdist(spatial_coords, spatial_coords)` *per layer* — O(N²) ×
12 layers.

**Fix:** Registry now maps `dense_masked` → `CachedDistAttention`, which
receives the pre-computed `dist_2d` once (matching real
`TrackingTransformer.forward()` behavior).

**File:** `benchmark_attn/src/bench/registry.py`
**Change:** `"src.attention_modules.RelativePositionalAttention"` →
`"src.attention_modules.CachedDistAttention"`

### Bug 2: KNN index computation excluded from timing

**Old:** `benchmark_sweep.py:169` pre-computed KNN indices *before* the timing
loop, hiding the O(N²) `cdist + topk` cost that training incurs every forward
pass.

**Fix:** Added `--with-knn` flag. When set, KNN indices are recomputed inside
the timed closure.

**File:** `benchmark_attn/scripts/benchmarks/benchmark_sweep.py`

### Bug 3 (conceptual): `dense_flash` is not a realistic baseline

`dense_flash` uses no mask → gets FlashAttention-2 kernel. But in real
training, the spatial cutoff mask IS always enforced → forces fallback to
EfficientAttention. So `dense_flash` is an *upper bound* on speed, not a
reachable training configuration.

The realistic comparison is **`dense_masked` vs `mask_knn`** — both use
EfficientAttention (not FlashAttention-2).

## 2. Effect on Results

### Old (buggy) `full_bench_h100.csv` at N=2048:

| Method | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| dense_masked | 0.41 | 96 |
| mask_knn K=4 | 0.15 | 18 |
| gather_sdpa K=4 | 0.24 | 18 |
| dense_flash | 0.11 | 4 |

**Old conclusion:** mask_knn is 2.7× faster than dense_masked. Sparse wins.

### New (corrected) `full_bench_h100_with_knn.csv` at N=2048:

| Method | Time (ms) | Memory (MB) |
|--------|-----------|-------------|
| dense_masked | 0.344 | 143.8 |
| mask_knn K=4 | 0.396 | 70.9 |
| gather_sdpa K=4 | 0.480 | 19.9 |
| dense_flash | 0.111 | 6.8 |

**New conclusion:** At realistic N (≤2048), dense_masked is **faster** than
mask_knn. The old benchmark compared a strawman dense_masked (per-layer cdist)
against a cheating mask_knn (KNN cost hidden).

### Crossover analysis (dense_masked vs mask_knn):

| N | dense_masked (ms) | mask_knn K=4 (ms) | Winner |
|------|-------------------|-------------------|--------|
| 128 | 0.199 | 0.335 | dense_masked |
| 512 | 0.200 | 0.339 | dense_masked |
| 2048 | 0.344 | 0.396 | dense_masked |
| 4096 | 1.422 | 0.895 | **mask_knn** |
| 8192 | 5.113 | 2.914 | **mask_knn** |

Crossover at ~N=4000. Vanvliet training regime (window=4, ~35 cells/frame →
N≈140) is well below this threshold.

## 3. Files to Update

### 3.1 `results/full_bench_h100.csv`

**Action:** Replace with corrected data from cluster.

**Source:** The new CSV is at
`/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/benchmark_attn/results/full_bench_h100.csv`
on the cluster (job 3920627, completed).

**Important:** The new CSV has an additional `with_knn` column. The schema is:
`method,L,N,K,time_ms,memory_mb,error`

**Local copy:** Download the new CSV from the cluster and replace
`benchmark_attn/results/full_bench_h100.csv`.

**SSH command to fetch:**
```bash
scp capella:/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking/benchmark_attn/results/full_bench_h100.csv \
    benchmark_attn/results/full_bench_h100.csv
```

### 3.2 `README.md`

**Action:** Update the "Key Results" table and observations.

**Old text (lines 7-28):**
```
## Key Results

Full method sweep on NVIDIA A500 and NVIDIA H100 (fp16, single-layer forward pass, L=1). A500 numbers from `results/full_bench_a500.csv`, H100 numbers from `results/full_bench_h100.csv`. Sparse = KNN-gathered attention with K neighbors.

| N | Method | A500 time (ms) | A500 mem (MB) | H100 time (ms) | H100 mem (MB) | vs dense_masked |
|---|--------|----------------|---------------|----------------|---------------|-----------------|
| 2048 | dense_masked | 4.9 | 63 | 0.41 | 96 | 1× |
| 2048 | gather_sdpa K=4 | 3.1 | 15 | 0.24 | 18 | 1.6× (A500) / 1.7× (H100) faster |
| 8192 | dense_masked | 79.9 | 972 | 5.6 | 1487 | 1× |
| 8192 | dense_flash | 4.9 | 20 | 0.35 | 25 | — |
| 8192 | gather_sdpa K=4 | 12.2 | 60 | 0.54 | 70 | **6.5× (A500) → 10.3× (H100) faster**, 16× / 21× less mem |
| 8192 | gather_sdpa K=16 | 19.0 | 204 | 1.0 | 199 | 4.2× (A500) → 5.6× (H100) faster |

Key observations:

- **The sparse advantage grows on faster hardware.** gather_sdpa K=4 is 6.5× faster than dense_masked on the A500 (79.9 → 12.2 ms) but 10.3× faster on the H100 (5.6 → 0.54 ms). Dense attention is O(N²) work, gathered sparse attention O(NK); the faster GPU amplifies the gap.
- **Crossover confirmed for gather_matmul.** On the H100, gather_matmul K=4 ties dense_flash at N=8192 (0.36 ms vs 0.35 ms) — the predicted sparse/dense crossover is now confirmed for gather_matmul.
- **NSA timing is near-constant ~2.5 ms** up to N=4096 on the H100 (2.46–2.62 ms), jumping to 5.9 ms at N=8192.
- **minimax@8192 uses 44 GB** on the H100 — the largest single-layer footprint, closest to the 80 GB limit.
- **Zero OOMs across all 168 configurations** on the H100 (9 methods × 7 N values × up to 4 K values).
- Legacy A500-only finding (fp16, B=2, d=256, h=4): at L=4, N=8192 dense OOMs on the 4 GiB A500 while sparse K=4 fits (0.094 s / 385 MB).
```

**New text:**
```
## Key Results

Full method sweep on NVIDIA H100 (fp16, single-layer forward pass, L=1, KNN index computation included in timing via `--with-knn`). Numbers from `results/full_bench_h100.csv` (job 3920627).

Two benchmark bugs were fixed:
1. **`dense_masked` class fix:** Registry mapped `dense_masked` to `RelativePositionalAttention` (recomputes `cdist` per layer). Fixed to `CachedDistAttention` (uses pre-computed `dist_2d` once).
2. **KNN cost inclusion:** `--with-knn` flag now includes O(N²) `cdist + topk` in timing, matching training behavior.

| N | Method | H100 time (ms) | H100 mem (MB) | vs dense_masked |
|---|--------|----------------|---------------|-----------------|
| 2048 | dense_masked | 0.344 | 143.8 | 1× |
| 2048 | mask_knn K=4 | 0.396 | 70.9 | 0.87× (1.15× slower) |
| 2048 | gather_sdpa K=4 | 0.480 | 19.9 | 0.72× (1.39× slower) |
| 2048 | dense_flash | 0.111 | 6.8 | 3.1× faster (unrealistic — no mask) |
| 8192 | dense_masked | 5.113 | 1487.0 | 1× |
| 8192 | mask_knn K=4 | 2.914 | 1049.5 | **1.8× faster** |
| 8192 | gather_sdpa K=4 | 2.268 | 268.7 | **2.3× faster** |
| 8192 | dense_flash | 0.351 | 25.3 | 14.6× faster (unrealistic — no mask) |

Key observations:

- **`dense_flash` is not a realistic baseline.** It uses no mask → gets FlashAttention-2 kernel. But in training, the spatial cutoff mask IS always enforced → forces fallback to EfficientAttention. `dense_flash` is an upper bound on speed.
- **Realistic comparison: `dense_masked` vs `mask_knn`.** Both use EfficientAttention. At N≤2048, `dense_masked` is faster. At N≥4096, `mask_knn` is faster. Crossover at ~N=4000.
- **Sparse attention wins at high N (≥4096).** `mask_knn` is 1.8× faster than `dense_masked` at N=8192. `gather_sdpa` is 2.3× faster.
- **Vanvliet training regime (N≈140):** `dense_masked` (0.199 ms) is 1.7× faster than `mask_knn` (0.335 ms). Sparse attention is NOT faster at cell-tracking scale.
- **NSA timing is near-constant ~2.5 ms** up to N=4096 on the H100 (2.46–2.62 ms), jumping to 5.9 ms at N=8192.
- **minimax@8192 uses 44 GB** on the H100 — the largest single-layer footprint, closest to the 80 GB limit.
- **Zero OOMs across all 168 configurations** on the H100 (9 methods × 7 N values × up to 4 K values).
```

### 3.3 `REPRODUCTION.md`

**Action:** Update §4 (cluster H100 benchmark) and §6.7 (results index).

**Old text (§4, lines 149-186):**
```
## 4. `slurm/run_full_bench.slurm` — H100 full attention benchmark

SBATCH resources (as declared in the script): job `attn_full_bench`, account
`p_scads_celltracking`, partition `gpu-h100`, time `01:00:00`, 1 GPU, 8 CPUs,
90 GB mem, logs `logs/full_bench_%j.{out,err}`.

In-script env: `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`,
`MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `NCCL_DEBUG=WARN`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONNOUSERSITE=1`,
`unset PYTHONPATH PYTHONHOME`. The script sources `~/.bench.env` (gitignored,
set once via `slurm/setup_env.sh`), sources `slurm/load_cluster_env.sh`, then
invokes `.venv/bin/python` directly (no `uv sync` overhead — the `.venv/` was
created during one-time setup).

Submit:

```bash
sbatch slurm/run_full_bench.slurm
```

Equivalent manual command (also the one the slurm script should run):

```bash
cd "$BENCH/benchmark_attn" && .venv/bin/python scripts/benchmarks/benchmark_sweep.py \
    --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,dense_flash,dense_masked,nsa,knn_relpos,minimax \
    --d 320 --nhead 8 --warmup 10 --rep 50 \
    --out results/full_bench_h100.csv
```

(`$BENCH` comes from `~/.bench.env`.)

Purpose: full-method attention benchmark on the H100 (80 GB) as an
A500-vs-H100 reference; the largest N (8192) runs without the A500's 4 GB
limit.

Completed run: job `3898011` (logs `logs/full_bench_3898011.{out,err}`)
finished successfully. Output: `results/full_bench_h100.csv` — 168 rows, all
successful (zero OOMs). Largest single-layer memory: minimax@N=8192 = 44 GB,
i.e. 55 % of the 80 GB GPU. NSA timing is near-constant at ~2.5 ms from
N=128 up to N=4096, then doubles to 5.87 ms at N=8192.
```

**New text (§4):**
```
## 4. `slurm/run_full_bench.slurm` — H100 full attention benchmark

SBATCH resources (as declared in the script): job `attn_full_bench`, account
`p_scads_celltracking`, partition `gpu-h100`, time `01:00:00`, 1 GPU, 8 CPUs,
90 GB mem, logs `logs/full_bench_%j.{out,err}`.

In-script env: `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`,
`MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `NCCL_DEBUG=WARN`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONNOUSERSITE=1`,
`unset PYTHONPATH PYTHONHOME`. The script sources `~/.bench.env` (gitignored,
set once via `slurm/setup_env.sh`), sources `slurm/load_cluster_env.sh`, then
invokes `.venv/bin/python` directly (no `uv sync` overhead — the `.venv/` was
created during one-time setup).

Submit:

```bash
sbatch slurm/run_full_bench.slurm
```

Equivalent manual command (also the one the slurm script should run):

```bash
cd "$BENCH/benchmark_attn" && .venv/bin/python scripts/benchmarks/benchmark_sweep.py \
    --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,dense_flash,dense_masked,nsa,knn_relpos,minimax \
    --d 320 --nhead 8 --warmup 10 --rep 50 --with-knn \
    --out results/full_bench_h100.csv
```

(`$BENCH` comes from `~/.bench.env`.)

Purpose: full-method attention benchmark on the H100 (80 GB) with
KNN index computation included in timing (`--with-knn` flag). This
matches real training behavior where `TrackingTransformer.forward()`
recomputes KNN indices every forward pass.

### Benchmark bug fixes (2026-08-17)

Two bugs were found in the benchmark harness and fixed:

1. **`dense_masked` class fix.** The registry mapped `dense_masked` to
   `RelativePositionalAttention`, which recomputes `torch.cdist` per layer
   (O(N²) × 12 layers). Fixed to `CachedDistAttention`, which uses the
   pre-computed `dist_2d` once — matching real
   `TrackingTransformer.forward()` behavior.

2. **KNN cost inclusion.** The old benchmark pre-computed KNN indices
   *before* the timing loop (`benchmark_sweep.py:169`), hiding the O(N²)
   `cdist + topk` cost. The `--with-knn` flag now includes this cost in
   timing, matching training behavior.

**Effect at N=2048:** `dense_masked` dropped from 0.41 ms → 0.344 ms (correct
class). `mask_knn` rose from 0.15 ms → 0.396 ms (KNN cost included). The
ranking flipped: `dense_masked` is now faster than `mask_knn` at N≤2048.

Completed run: job `3920627` (logs `logs/full_bench_3920627.{out,err}`)
finished successfully. Output: `results/full_bench_h100.csv` — 169 rows, all
successful (zero OOMs). Largest single-layer memory: minimax@N=8192 = 44 GB,
i.e. 55 % of the 80 GB GPU. NSA timing is near-constant at ~2.5 ms from
N=128 up to N=4096, then doubles to 5.87 ms at N=8192.
```

**Old text (§6.7, lines 258-274):**
```
### 6.7 `results/full_bench_h100.csv` — H100 full attention benchmark

Data: [full_bench_h100.csv](results/full_bench_h100.csv).

168 rows: 9 methods × 7 N values × up to 4 K values, single-layer (L=1), fp16 on H100 (80 GB).
**All configurations completed with zero OOMs** (job `3898011`). Key observations:

| Method | N=8192 time | N=8192 memory | Notes |
|--------|-------------|---------------|-------|
| dense_flash | 0.35 ms | 25 MB | Fastest at all N |
| gather_matmul K=4 | 0.36 ms | 72 MB | Effectively tied with dense_flash (0.36 vs 0.35 ms) |
| gather_sdpa K=4 | 0.55 ms | 70 MB | |
| knn_relpos K=4 | 0.55 ms | 71 MB | Near-identical to gather_sdpa |
| mask_knn K=4 | 1.18 ms | 1050 MB | K-independent memory (N×N mask) |
| nsa | 5.86 ms | 1714 MB | Near-constant ~2.5 ms up to N=4096, then doubles to 5.86 ms at N=8192 |
| minimax | 294.9 ms | 44053 MB | Largest single-layer footprint; 55 % of the 80 GB GPU |
```

**New text (§6.7):**
```
### 6.7 `results/full_bench_h100.csv` — H100 full attention benchmark

Data: [full_bench_h100.csv](results/full_bench_h100.csv).

169 rows: 9 methods × 7 N values × up to 4 K values, single-layer (L=1), fp16
on H100 (80 GB), with KNN index computation included in timing
(`--with-knn` flag, job `3920627`). **All configurations completed with zero
OOMs.**

**Two benchmark bugs were fixed** (see §4 for details):
1. `dense_masked` used wrong class (per-layer cdist) → fixed to `CachedDistAttention`
2. KNN index computation excluded from timing → fixed with `--with-knn` flag

**Corrected results at N=8192:**

| Method | N=8192 time | N=8192 memory | Notes |
|--------|-------------|---------------|-------|
| dense_flash | 0.351 ms | 25 MB | **Not realistic** — no mask, FlashAttention-2 |
| dense_masked | 5.113 ms | 1487 MB | Realistic dense baseline (spatial cutoff mask) |
| mask_knn K=4 | 2.914 ms | 1050 MB | **1.8× faster** than dense_masked at N=8192 |
| gather_sdpa K=4 | 2.268 ms | 269 MB | **2.3× faster** than dense_masked at N=8192 |
| nsa | 5.861 ms | 1714 MB | Near-constant ~2.5 ms up to N=4096 |
| minimax | 294.9 ms | 44053 MB | Largest single-layer footprint; 55 % of the 80 GB GPU |

**Crossover analysis (dense_masked vs mask_knn):**

| N | dense_masked (ms) | mask_knn K=4 (ms) | Winner |
|------|-------------------|-------------------|--------|
| 128 | 0.199 | 0.335 | dense_masked |
| 2048 | 0.344 | 0.396 | dense_masked |
| 4096 | 1.422 | 0.895 | mask_knn |
| 8192 | 5.113 | 2.914 | mask_knn |

Crossover at ~N=4000. Vanvliet training regime (N≈140) is well below this
threshold — sparse attention is NOT faster at cell-tracking scale.
```

### 3.4 `analysis/plot_sparse.py`

**Action:** Update the plot script to use the corrected CSV data and
reflect the new conclusions.

The script currently reads `results/benchmark_sparse_results.csv`. It needs
to also read `results/full_bench_h100.csv` (the corrected H100 data) and
generate updated plots.

**Changes needed:**

1. Add a function `load_h100_results()` that reads
   `results/full_bench_h100.csv` and returns a DataFrame with columns
   `method`, `N`, `K`, `time_ms`, `memory_mb`, and a `label` column.

2. Add a function `plot_h100_time_vs_n(df)` that generates a log-log
   Time vs N figure for the H100 data, with methods color-coded:
   - `dense_flash` (tab:cyan)
   - `dense_masked` (tab:blue)
   - `mask_knn` (tab:orange)
   - `gather_sdpa` (tab:green)
   - `nsa` (tab:purple)
   - `minimax` (tab:red)

3. Add a function `plot_h100_memory_vs_n(df)` that generates a log-log
   Memory vs N figure for the H100 data.

4. Add a function `plot_crossover_analysis(df)` that generates a figure
   showing the crossover point where `mask_knn` becomes faster than
   `dense_masked`. Plot both methods on the same axes, with a vertical
   line at the crossover N (~4000).

5. Update `main()` to call the new functions and add the new figures to
   the HTML output.

6. Update the HTML title to reflect the corrected analysis.

### 3.5 `analysis/make_report.py`

**Action:** Update the comprehensive report generator to use corrected
data and reflect the new conclusions.

**Changes needed:**

1. Update `load_knn_methods_csv()` to handle the corrected CSV schema.

2. Update `make_knn_speedup_figure()` to use the corrected data and
   show the new crossover analysis.

3. Update the HTML report title and introduction to reflect the
   corrected analysis.

4. Add a note about the benchmark bug fixes in the report.

### 3.6 `src/bench/html_report.py`

**Action:** No changes needed. This is a utility module.

### 3.7 `docs/README_knn_methods.md`

**Action:** Update to reflect the corrected benchmark results.

### 3.8 `docs/README_system_impact.md`

**Action:** Update to reflect the corrected benchmark results.

## 4. Implementation Order

1. Download corrected `full_bench_h100.csv` from cluster
2. Update `README.md` (§3.2)
3. Update `REPRODUCTION.md` (§3.3)
4. Update `analysis/plot_sparse.py` (§3.4)
5. Update `analysis/make_report.py` (§3.5)
6. Update `docs/README_knn_methods.md` (§3.7)
7. Update `docs/README_system_impact.md` (§3.8)
8. Run `analysis/plot_sparse.py` to regenerate plots
9. Run `analysis/make_report.py` to regenerate report
10. Commit and push all changes

## 5. Acceptance Criteria

- [ ] `results/full_bench_h100.csv` contains the corrected data (169 rows,
      `--with-knn` flag applied, `dense_masked` uses `CachedDistAttention`)
- [ ] `README.md` "Key Results" table shows corrected numbers
- [ ] `README.md` observations mention that `dense_flash` is unrealistic
- [ ] `REPRODUCTION.md` §4 documents the bug fixes and `--with-knn` flag
- [ ] `REPRODUCTION.md` §6.7 shows corrected results table and crossover
- [ ] `analysis/plot_sparse.py` generates updated plots from corrected data
- [ ] `analysis/make_report.py` generates updated report
- [ ] All changes committed and pushed
