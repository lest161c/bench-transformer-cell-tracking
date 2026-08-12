# REPRODUCTION.md — Attention / Memory Benchmarks (`benchmark_attn/`)

Reproduction guide for every experiment in this directory: exact CLI invocations,
editable hardcoded lists, required environments, output files, and the measured
results ledger.

Authoritative sources:
- The experiment readiness inventory in `labbook/` — experiment inventory,
  feasibility verdicts, measured numbers.
- The benchmark scripts themselves.
- `run_full_bench.slurm` SBATCH directives.
- `benchmark_attn/.venv` (Python 3.14.4, torch 2.12.0+cu130).

---

## 1. Local (A500) vs Cluster (H100) split

Two distinct hardware environments are involved. Almost all benchmarks in this
directory run **locally on the RTX A500 Laptop** (4096 MiB, compute cap 8.6,
Ampere). Only `run_full_bench.slurm` is a H100 (80 GB, TU Dresden Capella) job.

| Environment | Hardware | GPU VRAM | Venv | Used for |
|---|---|---|---|---|
| Local (this repo) | NVIDIA RTX A500 Laptop | 4 GiB | `benchmark_attn/.venv/bin/python` | `scripts/benchmarks/benchmark_sweep.py` (consolidated sweep; replaces the four standalone benchmark scripts: dense-vs-gather, full method set, gather variants, small-N), `benchmarks/benchmark_mask_vs_gather.py`, `benchmarks/benchmark_all_spatial_methods.py`, `benchmarks/benchmark_knn_methods.py`, `benchmarks/benchmark_cached_dist.py`, `analysis/plot_sparse.py` |
| Cluster (Capella) | NVIDIA H100 | 80 GiB | `$TRK/.venv` (see §3) | `run_full_bench.slurm` → `scripts/benchmarks/benchmark_sweep.py` → `full_bench_h100.csv` |

All local benchmarks use fp16 on CUDA (`dtype = torch.float16`). Scripts assert
`torch.cuda.is_available()` and are not intended to run on CPU.

Local venv verified packages (`benchmark_attn/.venv`):
`torch 2.12.0+cu130`, `native_sparse_attention_pytorch`, `seaborn 0.13.2`,
`pandas 3.0.3`, `matplotlib 3.11.0`, `wandb 0.28.1`.

```bash
V=benchmark_attn/.venv/bin/python
```

Run every local benchmark from inside `benchmark_attn/` so that
`src/attention_modules.py` and the CSV paths resolve:

```bash
cd benchmark_attn
$V scripts/benchmarks/benchmark_sweep.py
```

---

## 2. Local (A500) experiments

### 2.1 `scripts/benchmarks/benchmark_sweep.py` — dense vs gather-sparse sweep (incl. N=1024/4096, NSA)

Purpose: forward time + incremental peak GPU memory for `dense_masked`
(RelativePositionalAttention), `dense_flash`, `nsa` (Native Sparse Attention),
`gather-sdpa` (GatherSparseAttention), `gather-fused` (GatherSparseFusedAttention),
each over N and K, at L = 1 and L = 4 layers.

Sweep parameters are CLI flags (batch_size=1, d, nhead, coord_dim):

```bash
# scripts/benchmarks/benchmark_sweep.py
--layers 1,4                           # L values (L_vals)
--Ns 128,256,512,2048,8192             # <-- EDIT for full N range
--Ks 4,16,64                           # K values (K_vals)
# NSA sel_blocks is not a CLI knob; the class default num_selected_blocks=4 applies
```

Reproduction of the **on-disk superset CSV** (`benchmark_sparse_results.csv`,
covering N = {128, 256, 512, 1024, 2048, 4096, 8192} and NSA
`sel_blocks` = {2, 4, 8, 16, 64}) requires passing the full sweep flags:

- `--Ns 128,256,512,1024,2048,4096,8192`
- `--Ks 4,16,64` (NSA `sel_blocks` is not exposed by `benchmark_sweep.py`;
  the NSA class default `num_selected_blocks=4` applies)

**Important:** the CSV on disk was produced by the run with the superset values.
If you re-run with a smaller sweep, it will overwrite the superset results.

NSA semantics: the NSA `K` column is `num_selected_blocks` (block selection), it
is **not** a KNN-K. NSA memory at small N: N=512 sel=8 ≈ 284 MB ⇒ sel=16 ≈ 570 MB,
sel=64 ≈ 1.5–2 GB (borderline on A500). Keep N ≤ 512 for NSA K=16/64.

Invocation and outputs:

```bash
cd benchmark_attn && $V scripts/benchmarks/benchmark_sweep.py \
    --methods dense_flash,dense_masked,nsa,gather-sdpa,gather-fused \
    --Ns 128,256,512,1024,2048,4096,8192 --Ks 4,16,64 --layers 1,4
# writes benchmark_sparse_results.csv  (columns: method,L,N,K,time_ms,memory_mb,error)
```

OOM rows are recorded as `status=oom` with empty `time_s`/`mem_mb` (per-cell
try/except in `try_bench`).

Measured results: see §6.2 (N=1024/4096 rows) and §6.3 (NSA sel 16/64).

### 2.2 `scripts/benchmarks/benchmark_sweep.py` — full method set × full N range (incl. CachedDist-adjacent methods)

Purpose: unified benchmark of all standalone attention methods at
N = [128 .. 8192]:

- `dense_masked` = RelativePositionalAttention (per-call cdist, explicit N×N mask)
- `dense_flash` = DenseFlashAttention (no mask, no spatial cutoff)
- `gather-sdpa` = GatherSparseAttention
- `mask-knn` = KNNMaskSparseAttention
- `nsa` = NSASparseAttention
- `knn-relpos` = KNNRelativePositionalAttention
- `minimax` = MiniMaxSparseAttention

Note (from the script docstring): CachedDistAttention is **not** in this
benchmark — it needs the full Trackastra `forward()` that amortizes cdist across
layers; the timing difference is ~1.5× at L=12 (see `benchmark_cached_dist.py`).

CLI:

```
python scripts/benchmarks/benchmark_sweep.py [--methods ...] [--d 320] [--nhead 8]
                          [--warmup 5] [--rep 30] [--out sweep_results.csv]
                          [--mode none|bias|rope] [--dist-mode v0|v1]
                          [--Ns 128,256,512,1024,2048,4096,8192] [--Ks 4,16,64] [--layers 1]
```

Sweep defaults (previously hardcoded at line 167):

```bash
--Ns 128,256,512,1024,2048,4096,8192   # default
--Ks 4,16,64                           # default (the old sweep used K = 4..128)
```

Default `--mode none`, `--dist-mode v1` (matches Trackastra's CachedDistAttention
config used in `benchmark_cached_dist.py`). fp16, d=320, nhead=8, coord_dim=2,
single N×N pass per method.

The local A500 run used Ks limited to [4, 16] and produced:

```bash
cd benchmark_attn && $V scripts/benchmarks/benchmark_sweep.py \
    --methods gather-sdpa,gather-fused,gather-matmul,mask-knn,dense_flash,dense_masked,nsa,knn-relpos,minimax \
    --Ks 4,16 --out benchmark_attn/full_bench_a500.csv
# writes full_bench_a500.csv  (columns: method,L,N,K,time_ms,memory_mb,error)
```

`full_bench_a500.csv` is the **source of truth** for mask-knn / gather-sdpa /
knn-relpos / nsa / dense_masked at N = 1024, 2048, 4096, 8192 on the A500 — including
mask-knn at N ≥ 2048, which `benchmark_mask_vs_gather.py` (limited to N ≤ 512)
does not cover. Measured values: see §6.1.

### 2.3 `benchmark_cached_dist.py` — CachedDistAttention real measurement

Purpose: real per-layer forward measurement of Trackastra's dense baseline
`RelativePositionalAttention` (per-layer 2D cdist) vs `CachedDistAttention`
(uses a precomputed `dist_2d`), plus the one-time 2D `torch.cdist` cost. Computes
the amortized L-layer (default 12: 6 enc + 6 dec) totals:

```
total_baseline = L * t_dense_masked
total_cached   = t_cdist_2d + L * t_cached_dist
```

The classes are self-contained copies in the local `src/attention_modules.py`
(`CachedDistAttention`, `RelativePositionalAttention`) — `benchmark_cached_dist.py`
has NO trackastra dependency.

CLI (defaults from the script):

```
python benchmarks/benchmark_cached_dist.py [--d 320] [--nhead 8] [--warmup 5] [--rep 30]
    [--out benchmark_attn/cached_dist_results.csv] [--layers 12]
    [--Ns 128,256,512,1024,2048,4096,8192] [--cutoff 256] [--dist-mode v1]
```

Setup notes:
- fp16, d=320, h=8, B=1 (synthetic `x`), coord_dim=2, `cutoff_spatial=256`,
  `mode="none"`, `attn_dist_mode="v1"`.
- Dtype requirement (real-model nuance): `dist_2d` must be cast to the model
  dtype (fp16) before the v1 decay add (`dist_2d_m = dist_2d.to(dtype)` in the
  script); the real model computes it in fp32 which breaks fp16 SDPA mask dtype
  on torch 2.12+cu130.

```bash
cd benchmark_attn && $V benchmark_cached_dist.py
# writes cached_dist_results.csv  (columns: method,N,time_ms,memory_mb,error)
```

Measured results: see §6.4. N=8192 OOMs on the A500 for the cached variant
(dense_masked@8192 fits: 190.0 ms / 1487 MB).

### 2.4 `scripts/benchmarks/benchmark_sweep.py` — gather variants (`--methods gather-sdpa,gather-fused,gather-matmul`)

Purpose: compare the three gather variants head-to-head:
- SDPA = `GatherSparseAttention` (SDPA on flattened (B*N, nH, 1, Dh)) — `gather-sdpa`
- Fused = `GatherSparseFusedAttention` (spatial reorder) — `gather-fused`
- Matmul = `GatherSparseMatmulAttention` (manual `torch.matmul`; `src/attention_modules.py:607`)

CLI (defaults from the script):

```
python scripts/benchmarks/benchmark_sweep.py [--d 320] [--nhead 8] [--warmup 5] [--rep 30]
    --methods gather-sdpa,gather-fused,gather-matmul
    [--out benchmark_attn/sweep_results.csv]
    [--Ns 128,256,512,1024,2048,4096,8192] [--Ks 4,16,64]
```

fp16, d=320, h=8, B=1 (synthetic), coord_dim=2, KNN indices computed from
`coords[..., 1:]` (2D cdist + topk). Output includes matmul/sdpa and matmul/fused
ratios (< 1 means gather-matmul faster).

```bash
cd benchmark_attn && $V scripts/benchmarks/benchmark_sweep.py \
    --methods gather-sdpa,gather-fused,gather-matmul
# writes sweep_results.csv  (columns: method,L,N,K,time_ms,memory_mb,error)
```

Measured results: see §6.5. gather-matmul is consistently fastest (up to 5× vs
gather-sdpa at N ≥ 1024).

### 2.5 `benchmark_knn_methods.py` — KNN method comparison (analytical)

Purpose: theoretical/calibrated comparison of `cached_dense` (CachedDistAttention
baseline), `dense_flash`, `mask-knn`, `gather-sdpa`, `minimax`. This is an
**analytical model only** — calibrated to labbook relative speeds at N=256
— it does not touch the GPU. The `--gpu` mode mentioned in
`README_knn_methods.md` is **not implemented** in the current script
(`main()` only calls `run_analytical()`).

CLI:

```
python benchmarks/benchmark_knn_methods.py [--out benchmark_attn/knn_methods_results.csv]
                                [--outdir benchmark_attn]
```

```bash
cd benchmark_attn && $V benchmark_knn_methods.py
# writes knn_methods_results.csv  (columns: method,N,K,time_ms,spatial_cutoff,data_source)
# writes knn_methods_comparison.png, knn_crossover_analysis.png,
#        speedup_heatmap_gather-KNN.png, speedup_heatmap_mask-KNN.png
```

`data_source = analytical_calibrated`. **This is NOT a real measurement** — the
real CachedDistAttention measurement is `benchmark_cached_dist.py` (§2.3).

### 2.6 `benchmark_mask_vs_gather.py` — mask vs gather

Purpose: compare `KNNMaskSparseAttention` vs `GatherSparseAttention` vs
`DenseFlashAttention` at N = [128, 256, 512], K = 16, L = 1, plus an SDPA
backend profile section. No CLI args; B=2, d=256, h=4, coord_dim=3 hardcoded.

```bash
cd benchmark_attn && $V benchmark_mask_vs_gather.py
# writes benchmark_mask_vs_gather.csv  (columns: method,N,K,time_s,mem_mb,speedup_vs_gather)
```

The script has no CLI args and writes `benchmark_mask_vs_gather.csv` to the
current working directory; run it from inside `benchmark_attn/` so the output
lands alongside the other benchmark CSVs. **Status: DONE.** mask-knn data from
this script is limited to N ≤ 512; real mask-knn data at N ≥ 2048 lives in
`full_bench_a500.csv` (§2.2, §6.1).

### 2.7 `benchmark_all_spatial_methods.py` — all spatial-cutoff methods

Purpose: FlexAttention vs gather-sdpa vs mask-knn vs cuDNN hard mask, head-to-head
on the A500. d=320, nhead=8, d_head=40, fp16, d_max=256, lam=5, knn_k=16
(default). Verifies spatial cutoff equivalence via cosine similarity against the
hard-mask baseline.

CLI:

```
python benchmarks/benchmark_all_spatial_methods.py [--outdir benchmark_attn] [--knn-k 16]
```

```bash
cd benchmark_attn && $V benchmark_all_spatial_methods.py
# writes all_spatial_methods.csv  (columns: N, hard_cudnn_ms, mask_knn_ms,
#   gather_knn_ms, gather_knn_error, flex_ms, cos_mask_vs_hard, cos_flex_vs_hard)
```

Hardcoded N sweep (line 149): `Ns = [128, 256, 512, 1024, 2048, 4096]`.

**Caveat:** the `gather_knn_ms` column is not populated (unsupported) — see the
`gather_knn_error` column. Do NOT cite gather numbers from this CSV.

### 2.8 `validate_mask_vs_gather.py` — numerical equivalence of the two KNN variants

Purpose: verifies that `GatherSparseAttention` and `KNNMaskSparseAttention`
produce numerically identical forward/backward outputs. It measures forward
output cosine similarity, forward maximum absolute difference $|\Delta|$, and
gradient relative error across parameters with significant gradient norms
($\|\nabla\| > 10^{-6}$), across $18$ configurations ($N \in \{128, 256, 512\}$,
$K \in \{4, 16\}$, $3$ seeds each) in fp16 on CUDA. It writes no CSV (console
output only).

Result: cosine similarity $0.9995$–$1.0010$, forward
$|\Delta| < 5 \times 10^{-4}$, gradient relative error $< 8 \times 10^{-4}$
across all $18$ configurations. The whole validation costs roughly $2$
GPU-minutes versus $600{+}$ for a full training run — a $300\times$ reduction —
so the faster mask-knn inherits the gather-sdpa tracking accuracy without
retraining.

```bash
cd benchmark_attn && $V benchmarks/validate_mask_vs_gather.py
```

### 2.9 `plot_sparse.py` — visualization of the sparse sweep

Purpose: seaborn/matplotlib figures from `benchmark_sparse_results.csv`,
combined into a single self-contained HTML report (figures embedded as base64
PNGs). No CLI args, no GPU.

```bash
cd benchmark_attn && $V plot_sparse.py
# reads  benchmark_sparse_results.csv
# writes benchmark_sparse.html
```

Figures: time-vs-N per L, reorder speedup heatmap, incremental memory per L,
feasibility map (OK/OOM), and KNN speedup/memory-savings heatmaps vs dense.

---

## 3. Cluster requirements

### 3.1 Path constants (verify against scripts)

| Constant | Path |
|---|---|
| `WS` | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601` |
| `TRK` (trackastra checkout + venv + runs) | `$WS/trackastra` |
| `BENCH` (bench repo on cluster) | `$WS/bench-transformer-cell-tracking` |
| `ENV_DIR` | `$TRK/.venv` |
| `DATA_DIR` | `/data/cat/ws/mawe985g-data/data/celltracking` |

These paths are **unreachable locally**; cluster access requires SSH host
`capella` (VPN required). `run_full_bench.slurm` defines `WS`, `REPO`,
`VENV` identically.

### 3.2 Partition / resources

- SLURM partition: `gpu-h100`
- SLURM account: `p_scads_celltracking`
- GPU: 1 per node (`--gpus-per-node=1`)
- Cluster venv: `$TRK/.venv` (activated inside the slurm script)

### 3.3 SSH access

```bash
ssh capella        # VPN required; paths /data/cat/ws/... and /data/cat/ws/mawe985g-data/...
```

---

## 4. `run_full_bench.slurm` — H100 full attention benchmark

### 4.1 Exact SBATCH resources

| Directive | Value |
|---|---|
| `--job-name` | `attn_full_bench` |
| `--account` | `p_scads_celltracking` |
| `--partition` | `gpu-h100` |
| `--time` | `01:00:00` |
| `--gpus-per-node` | `1` |
| `--cpus-per-task` | `8` |
| `--mem` | `90G` |
| `--output` | `benchmark_attn/logs/full_bench_%j.out` |
| `--error` | `benchmark_attn/logs/full_bench_%j.err` |

Env vars set in-script:
`PIP_REQUIRE_VIRTUALENV=false`, `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`,
`MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `NCCL_DEBUG=WARN`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONNOUSERSITE=1`, `unset PYTHONPATH
PYTHONHOME`.

The script activates `$VENV` (`$WS/trackastra/.venv`), installs torch from the
cu124 wheel index (`python -m pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu124 -q`), then `cd "$REPO"`.

### 4.2 Command executed

```bash
python benchmark_attn/scripts/benchmarks/benchmark_sweep.py \
    --methods gather-sdpa,gather-fused,gather-matmul,mask-knn,dense_flash,dense_masked,nsa,knn-relpos,minimax \
    --d 320 --nhead 8 \
    --warmup 10 --rep 50 \
    --out benchmark_attn/full_bench_h100.csv
```

Config echoed by the script header: d_model=320, nhead=8, mode=none,
dist_mode=v1, N = [128, 256, 512, 1024, 2048, 4096, 8192], K = 4 16.

### 4.3 Purpose

Produce the full-method attention benchmark on the H100 (80 GB) so the A500
numbers (`full_bench_a500.csv`) can be compared against an H100 reference and the
largest N (8192) can run without the A500's 4 GB limit.

### 4.4 How to run

There is **no TEST=1 mode** in this script. Submit from the cluster bench repo
root (paths inside the script are relative to `$REPO`):

```bash
sbatch benchmark_attn/run_full_bench.slurm
```

Logs: `benchmark_attn/logs/full_bench_<jobid>.{out,err}` (relative to `$REPO`).
On success the script `cat`s the output CSV to the log.

### 4.5 Outputs

- `benchmark_attn/full_bench_h100.csv` on the cluster (columns:
  `method,N,time_ms,memory_mb,error`).
- Local copy does not exist yet — this experiment is PENDING.

### 4.6 Discrepancy (K values)

The script's echo block states "K = 4 16", but `scripts/benchmarks/benchmark_sweep.py`
defaults `--Ks` to `4,16,64`. The actual submitted run will therefore sweep
K = {4, 16, 64}, not just {4, 16}. If only K={4,16} is desired, pass `--Ks 4,16`
when submitting.

---

## 5. Experiment status summary

Status legend: DONE = measured artifact exists on disk (locally and/or on the
cluster); PENDING = code ready, not yet run / H100-only / cluster capacity
needed.

### 5.1 `benchmark_attn` experiments

| Experiment | Script / Slurm | Status | Resources | Outputs |
|---|---|---|---|---|
| Dense vs sparse sweep (N incl. 1024/4096, NSA sel_blocks 16/64) | `scripts/benchmarks/benchmark_sweep.py` | **DONE** | A500, `benchmark_attn/.venv` | `benchmark_sparse_results.csv`, `benchmark_sparse.html` |
| Full method set × N (dense_masked, dense_flash, gather-sdpa/mask-knn, nsa, knn-relpos) | `scripts/benchmarks/benchmark_sweep.py` | **DONE** | A500 | `full_bench_a500.csv` |
| H100 full benchmark (same script, K sweep) | `run_full_bench.slurm` | **PENDING** | H100, 90G, 8 CPUs, 1 h | `full_bench_h100.csv` (cluster) |
| CachedDistAttention real measurement | `benchmark_cached_dist.py` | **DONE** | A500 (classes in local `src/attention_modules.py`) | `cached_dist_results.csv` |
| GatherSparseAttention SDPA/Fused/Matmul | `scripts/benchmarks/benchmark_sweep.py` | **DONE** | A500 | `sweep_results.csv` |
| KNN method comparison (analytical) | `benchmark_knn_methods.py` | **DONE** | CPU only | `knn_methods_results.csv` + PNGs |
| Mask vs gather (N≤512) | `benchmark_mask_vs_gather.py` | **DONE** | A500 | `benchmark_mask_vs_gather.csv` |
| All spatial-cutoff methods | `benchmark_all_spatial_methods.py` | **DONE** | A500 | `all_spatial_methods.csv` (gather_knn_ms not populated) |
| Sparse sweep visualization | `plot_sparse.py` | **DONE** | CPU only | `benchmark_sparse.html` |
| Backward pass at N=8192 | `benchmark_backward/benchmark_sparse_backward.py` | **DONE** | A500 | `benchmark_backward/sparse_backward_results_8192.csv` |
| Full-model speed/mem K=64 | `benchmark_training/benchmarks/benchmark_speed_mem.py` | **DONE** | A500 | `benchmark_training/results/training_speed_memory_by_N_K.csv`, copy at `benchmark_attn/speed_mem.csv` |
| DINOv3 comparison | `benchmark_dinov3_comparison*.py` | **PENDING** (requires gated weights) | internet | `dinov3_comparison*.csv` |

### 5.2 Cluster experiments

| Experiment | Status | Notes |
|---|---|---|
| Multi-seed K-sweep (5 K × seeds 42/43/44, 15 jobs via `run_single_config.slurm`) | **PENDING** | 48 h budget each; expect 6.4–11 h (early-stopped) |
| DeepCell cross-dataset eval incl. KNN checkpoints | **PENDING** | edit `CHECKPOINTS` in `cross_dataset_eval.py`; inference-only, fits 1 GPU |
| Pull 5 clean K-sweep run dirs from `$TRK/runs/` | **PENDING** | needed for K=4/K=64 checkpoint reproducibility |

---

## 6. Measured results ledger (DONE experiments)

Numbers verbatim from the experiment readiness inventory in `labbook/`
(§1.x / §5), cross-checked against the CSVs on disk. All A500 fp16 unless stated.

### 6.1 `full_bench_a500.csv` — mask-knn at N ≥ 2048 (labbook §1.2)

| N | mask-knn K=4 | mask-knn K=16 |
|---|---|---|
| 2048 | 1.41 ms / 37 MB | 1.48 ms / 37 MB |
| 4096 | 5.19 ms / 138 MB | 5.35 ms / 138 MB |
| 8192 | 24.9 ms / 532 MB | 26.1 ms / 532 MB |

Fits the A500 at N=8192.

### 6.2 `benchmark_sparse_results.csv` — N=1024/4096 sweep (labbook §5.5)

L=1, N=4096: `dense_masked` 0.0392 s / 556 MB; `dense_flash` 0.0029 s;
`gather-sdpa` K16 0.0204 s / 204 MB; `gather-fused` K16 0.0205 s.
L=4, N=4096, K=64: **OOM** for both `gather-sdpa` and `gather-fused`.
L=1, N=8192: `dense_masked` OOM; `dense_flash` 0.0107 s / 40 MB; `gather-sdpa` K4/K16/K64 =
0.0265/0.0414/0.0963 s (120/408/1560 MB); `gather-fused` similar.

### 6.3 NSA sel_blocks 16/64 (labbook §5.7, §1.3)

- sel_blocks ∈ {16, 64} at N ≤ 512 all OK — e.g. L=1, N=512, sel=16:
  0.0197 s / 546.8 MB.
- L=4, N=512: sel ≥ 16 → **OOM**.
- All sel ∈ {16, 64} at N ≥ 2048 → **OOM**.
- NSA K column = `num_selected_blocks` (NOT KNN-K) — do not mix.

### 6.4 `cached_dist_results.csv` — CachedDistAttention (labbook §1.4)

Per-layer speedup / end-to-end (L=12) speedup, `dense_masked` vs `cached_dist`:

| N | per-layer | end-to-end (L=12) |
|---|---|---|
| 128 | 1.68× | 1.61× |
| 256 | 1.90× | 1.82× |
| 512 | 1.03× | 1.02× |
| 1024 | 1.07× | 1.05× |
| 2048 | 1.07× | 1.06× |
| 4096 | 1.22× | 1.20× |
| 8192 | **OOM** | **OOM** |

Nuance: the "~2× speedup" figure only holds at small N (≤ 256); at N = 512–2048
it collapses to ~1.05×, and N=8192 OOMs on the A500. The "~2×" CachedDistAttention
figure must be qualified to the small-N regime (the dominant regime in vanvliet).
Raw CSV nuance: `dense_masked`@8192 actually succeeded (190.0 ms / 1487 MB);
only `cached_dist`@8192 OOM'd in the on-disk CSV.

### 6.5 `sweep_results.csv` — GatherSparseMatmulAttention (labbook §1.5)

matmul/sdpa ratio (< 1 means gather-matmul faster), fp16, d=320, h=8, B=2:

| N | K=4 | K=16 | K=64 |
|---|---|---|---|
| 128 | 0.83× | 0.55× | 0.60× |
| 256 | 0.46× | 0.38× | 0.58× |
| 512 | 0.21× | 0.36× | 0.57× |
| 1024 | 0.20× | 0.34× | 0.59× |
| 2048 | 0.20× | 0.35× | 0.56× |

gather-matmul is consistently faster than gather-sdpa and gather-fused everywhere — up to **5× at N ≥ 1024**
(0.20×) and 1.2–2.6× at small N.

### 6.6 Backward pass at N=8192 (labbook §5.6) — `benchmark_backward/sparse_backward_results_8192.csv`

All methods fit, **NO OOM**:

| method | forward | backward | peak mem |
|---|---|---|---|
| dense_masked | 165.2 ms | 118.5 ms | 1615 MB |
| dense_flash | 10.8 ms | 24.9 ms | 66.5 MB |
| mask_knn | 62.8 ms | 133.2 ms | 1090 MB |
| gather_knn K16 | 33.2 ms | 81.0 ms | 2430 MB |
| sparse_softmax K4 | 4.6 ms | 10.8 ms | 81 MB (fastest) |

d=320, nhead=8, B=1, fp16. Side effect: `benchmark_backward/sparse_backward.png`
was overwritten by the run.

### 6.7 speed_mem (labbook §5.8) — `benchmark_attn/speed_mem.csv`

Full-model fp32 fwd+bwd+AdamW table for K {0,4,8,16,32,64} × N {128,256,512}.
Headline: **K=64, N=512 = 194 ms / 1604 MB** (fits A500). Note:
`benchmark_training/benchmarks/benchmark_speed_mem.py` inlines
`AgentCentricNormalization` (rather than importing it from `track_encoder`) and
requires `wandb` (installed in `benchmark_attn/.venv`).

---

## 7. Early-stopping guidance

**Only relevant for training runs** (e.g. the multi-seed K-sweep in §5.2).
The attention benchmarks in this directory are single-shot measurements and do
**not** need early stopping — there is no training loop to stop.

For the cluster training runs: never run to 500 epochs if `val_loss` has
plateaued — patience 83 already stops near-optimal; `--epochs 500` is only an
upper bound.

---

## 8. Known discrepancies / notes

1. The on-disk `benchmark_sparse_results.csv` was produced by the superset sweep
   (N = 128…8192, NSA sel ∈ {2,4,8,16,64}). Reproduce it with
   `scripts/benchmarks/benchmark_sweep.py` by passing the full sweep flags
   (`--Ns 128,256,512,1024,2048,4096,8192 --Ks 4,16,64 --layers 1,4`; NSA
   `sel_blocks` is not a CLI knob — class default `num_selected_blocks=4`
   applies); otherwise the smaller sweep overwrites the superset data.
2. `run_full_bench.slurm` echoes "K = 4 16" but
   `scripts/benchmarks/benchmark_sweep.py` defaults `--Ks` to `4,16,64`; the H100
   run will actually sweep all three K values.
3. `README_knn_methods.md` documents a `--gpu` mode that the current
   `benchmark_knn_methods.py` does not implement (analytical only).
4. `cached_dist_results.csv`: `dense_masked`@8192 succeeded (190.0 ms/1487 MB)
   while `cached_dist`@8192 OOM'd; the labbook writes "OOM (both variants)" —
   the on-disk CSV records OOM only for the cached variant.
5. `all_spatial_methods.csv` `gather_knn_ms` is not populated (unsupported) —
   do not cite.
6. `benchmark_mask_vs_gather.py` writes its output CSV to the current working
   directory; run it from inside `benchmark_attn/` so the output lands at
   `benchmark_attn/benchmark_mask_vs_gather.csv`.
7. The cluster slurm script installs torch from the cu124 index while the local
   venv is torch 2.12.0+cu130 — a mismatch to be aware of when comparing
   A500 (cu130) vs H100 (cu124) timings.
