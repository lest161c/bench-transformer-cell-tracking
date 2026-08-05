# REPRODUCTION.md — Attention / Memory Benchmarks (`benchmark_attn/`)

Reproduction guide for every experiment in this directory: exact CLI invocations,
editable hardcoded lists, required environments, output files, and the measured
results ledger.

Authoritative sources:
- `labbook/2026-08-03_experiment_readiness_inventory.md` (incl. the
  "Curation decision (2026-08-04)" section) — experiment inventory, feasibility
  verdicts, measured numbers.
- The benchmark scripts themselves (read at commit state of 2026-08-04).
- `run_full_bench.slurm` SBATCH directives.
- `benchmark_attn/.venv` (Python 3.14.4, torch 2.12.0+cu130).

---

## 1. Local (A500) vs Cluster (H100) split

Two distinct hardware environments are involved. Almost all benchmarks in this
directory run **locally on the RTX A500 Laptop** (4096 MiB, compute cap 8.6,
Ampere). Only `run_full_bench.slurm` is a H100 (80 GB, TU Dresden Capella) job.

| Environment | Hardware | GPU VRAM | Venv | Used for |
|---|---|---|---|---|
| Local (this repo) | NVIDIA RTX A500 Laptop | 4 GiB | `benchmark_attn/.venv/bin/python` | `benchmarks/benchmark_sparse.py`, `benchmarks/benchmark_full.py`, `benchmarks/benchmark_mask_vs_gather.py`, `benchmarks/benchmark_all_spatial_methods.py`, `benchmarks/benchmark_knn_methods.py`, `benchmarks/benchmark_cached_dist.py`, `benchmarks/benchmark_gather_v3.py`, `analysis/plot_sparse.py` |
| Cluster (Capella) | NVIDIA H100 | 80 GiB | `$TRK/.venv` (see §3) | `run_full_bench.slurm` → `benchmark_full.py` → `full_bench_h100.csv` |

All local benchmarks use fp16 on CUDA (`dtype = torch.float16`). Scripts assert
`torch.cuda.is_available()` and are not intended to run on CPU.

Local venv verified packages (`benchmark_attn/.venv`):
`torch 2.12.0+cu130`, `native_sparse_attention_pytorch`, `seaborn 0.13.2`,
`pandas 3.0.3`, `matplotlib 3.11.0`, `wandb 0.28.1`.

```bash
V=benchmark_attn/.venv/bin/python
```

Run every local benchmark from inside `benchmark_attn/` so that `model_parts.py`
and the CSV paths resolve:

```bash
cd benchmark_attn
$V benchmarks/benchmark_sparse.py
```

---

## 2. Local (A500) experiments

### 2.1 `benchmarks/benchmark_sparse.py` — dense vs gather-sparse sweep (incl. N=1024/4096, NSA)

Purpose: forward time + incremental peak GPU memory for `dense`
(RelativePositionalAttention), `dense_flash`, `nsa` (Native Sparse Attention),
`sparse` (GatherSparseAttention), `sparse_v2` (GatherSparseAttentionV2), each
over N and K, at L = 1 and L = 4 layers.

Hardcoded config (B=2, d=256, h=4, coord_dim=3):

```python
# benchmarks/benchmark_sparse.py
L_vals = [1, 4]                        # line 213
N_vals = [128, 256, 512, 2048, 8192]   # line 214  <-- EDIT for full N range
K_vals = [4, 16, 64]                   # line 215
...
for sel_blocks in [2, 4, 8]:           # line 244  <-- EDIT for NSA K=16/64
```

Reproduction of the **on-disk superset CSV** (`benchmark_sparse_results.csv`,
written 2026-08-04, covering N = {128, 256, 512, 1024, 2048, 4096, 8192} and NSA
`sel_blocks` = {2, 4, 8, 16, 64}) requires temporarily editing:

- line 214 → `N_vals = [128, 256, 512, 1024, 2048, 4096, 8192]`
- line 244 → `for sel_blocks in [2, 4, 8, 16, 64]:`

**Important:** the working tree currently has the un-edited lists (N without
1024/4096, sel_blocks {2,4,8}); the CSV on disk was produced by the run with the
edited values. If you re-run with the current file as-is, you will get a smaller
CSV that overwrites the superset results.

NSA semantics: the NSA `K` column is `num_selected_blocks` (block selection), it
is **not** a KNN-K. NSA memory at small N: N=512 sel=8 ≈ 284 MB ⇒ sel=16 ≈ 570 MB,
sel=64 ≈ 1.5–2 GB (borderline on A500). Keep N ≤ 512 for NSA K=16/64.

Invocation and outputs:

```bash
cd benchmark_attn && $V benchmarks/benchmark_sparse.py
# writes benchmark_sparse_results.csv  (columns: method,L,N,K,reorder,time_s,mem_mb,status)
```

OOM rows are recorded as `status=oom` with empty `time_s`/`mem_mb` (per-cell
try/except in `try_bench`).

Measured results: see §6.2 (N=1024/4096 rows) and §6.3 (NSA sel 16/64).

### 2.2 `benchmark_full.py` — full method set × full N range (incl. CachedDist-adjacent methods)

Purpose: unified benchmark of all standalone attention methods at
N = [128 .. 8192]:

- `dense_masked` = RelativePositionalAttention (per-call cdist, explicit N×N mask)
- `dense_flash` = DenseFlashAttention (no mask, no spatial cutoff)
- `gather-KNN` = GatherSparseAttention
- `mask-KNN` = KNNMaskSparseAttention
- `NSA` = NSASparseAttention
- `KNN-RelPos` = KNNRelativePositionalAttention
- `MiniMax` = MiniMaxSparseAttention

Note (from the script docstring): CachedDistAttention is **not** in this
benchmark — it needs the full Trackastra `forward()` that amortizes cdist across
layers; the timing difference is ~1.5× at L=12 (see `benchmark_cached_dist.py`).

CLI:

```
python benchmarks/benchmark_full.py [--d 320] [--nhead 8] [--warmup 5] [--rep 30]
                          [--out benchmark_full_results.csv]
                          [--mode none|bias|rope] [--dist-mode v0|v1]
```

Hardcoded sweep (line 167):

```python
Ns = [128, 256, 512, 1024, 2048, 4096, 8192]; Ks = [4, 16, 32, 64, 128]
```

Default `--mode none`, `--dist-mode v1` (matches Trackastra's CachedDistAttention
config used in `benchmark_cached_dist.py`). fp16, d=320, nhead=8, coord_dim=2,
single N×N pass per method.

The local A500 run (2026-06-17) used Ks limited to [4, 16] and produced:

```bash
cd benchmark_attn && $V benchmark_full.py --out benchmark_attn/full_bench_a500.csv
# writes full_bench_a500.csv  (columns: method,N,time_ms,memory_mb,error)
```

`full_bench_a500.csv` is the **source of truth** for mask-KNN / gather-KNN /
KNN-RelPos / NSA / dense at N = 1024, 2048, 4096, 8192 on the A500 — including
mask-KNN at N ≥ 2048, which the outdated `benchmark_mask_vs_gather.py` never
covered. Measured values: see §6.1.

### 2.3 `benchmark_cached_dist.py` — CachedDistAttention REAL measurement (NEW)

Purpose: real per-layer forward measurement of Trackastra's dense baseline
`RelativePositionalAttention` (per-layer 2D cdist) vs `CachedDistAttention`
(uses a precomputed `dist_2d`), plus the one-time 2D `torch.cdist` cost. Computes
the amortized L-layer (default 12: 6 enc + 6 dec) totals:

```
total_baseline = L * t_dense_masked
total_cached   = t_cdist_2d + L * t_cached_dist
```

The classes are self-contained copies in the local `model_parts.py`
(`CachedDistAttention`, `RelativePositionalAttention`) — `benchmark_cached_dist.py`
has NO trackastra dependency (the earlier importlib shim that loaded the classes
from the trackastra source was removed 2026-08-05).

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

### 2.4 `benchmark_gather_v3.py` — GatherSparseAttentionV3 (cached KNN indices) (NEW)

Purpose: compare the three gather variants head-to-head:
- V1 = `GatherSparseAttention` (SDPA on flattened (B*N, nH, 1, Dh))
- V2 = `GatherSparseAttentionV2` (spatial reorder)
- V3 = `GatherSparseAttentionV3` (manual `torch.matmul`; `model_parts.py:583`)

CLI (defaults from the script):

```
python benchmarks/benchmark_gather_v3.py [--d 320] [--nhead 8] [--warmup 5] [--rep 30]
    [--out benchmark_attn/gather_v3_results.csv]
    [--Ns 128,256,512,1024,2048,4096,8192] [--Ks 4,16,64]
```

fp16, d=320, h=8, B=1 (synthetic), coord_dim=2, KNN indices computed from
`coords[..., 1:]` (2D cdist + topk). Output includes V3/V1 and V3/V2 ratios
(< 1 means V3 faster).

```bash
cd benchmark_attn && $V benchmark_gather_v3.py
# writes gather_v3_results.csv  (columns: method,N,K,time_ms,memory_mb,error)
```

Measured results: see §6.5. V3 is consistently fastest (up to 5× vs V1 at
N ≥ 1024).

### 2.5 `benchmark_knn_methods.py` — KNN method comparison (analytical)

Purpose: theoretical/calibrated comparison of `cached_dense` (CachedDistAttention
baseline), `dense_flash`, `mask-KNN`, `gather-KNN`, `MiniMax`. This is an
**analytical model only** — calibrated to labbook relative speeds at N=256
(2026-06-08) — it does not touch the GPU. The `--gpu` mode mentioned in
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

### 2.6 `benchmark_mask_vs_gather.py` — mask vs gather (OUTDATED, superseded)

Purpose: compare `KNNMaskSparseAttention` vs `GatherSparseAttention` vs
`DenseFlashAttention` at N = [128, 256, 512], K = 16, L = 1, plus an SDPA
backend profile section. No CLI args; B=2, d=256, h=4, coord_dim=3 hardcoded.

```bash
cd benchmark_attn && $V benchmark_mask_vs_gather.py
# writes benchmark_mask_vs_gather.csv  (columns: method,N,K,time_s,mem_mb,speedup_vs_gather)
```

The historical run wrote the CSV to the repo root
(`/home/.../research-proj/benchmark_mask_vs_gather.csv`, 2026-05-29) because it
was executed from there. **Status: superseded.** The report claim "mask-KNN only
measured at N ≤ 512" applies only to this script; real mask-KNN data at
N ≥ 2048 lives in `full_bench_a500.csv` (§2.2, §6.1).

### 2.7 `benchmark_all_spatial_methods.py` — all spatial-cutoff methods

Purpose: FlexAttention vs gather-KNN vs mask-KNN vs cuDNN hard mask, head-to-head
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

**Caveat:** the `gather_knn_ms` column is **broken** (−1 everywhere, shape bug —
the gather reshape is invalid for the given tensor sizes, see
`gather_knn_error`). Do NOT cite gather numbers from this CSV.

### 2.8 `validate_mask_vs_gather.py` — numerical equivalence of the two KNN variants

Purpose: this is the **source of the report's equivalence-validation claim**
(`report/implementation.tex` §8.4, `report/argumentation.tex`): that
`GatherSparseAttention` and `KNNMaskSparseAttention` produce numerically
identical forward/backward outputs. It measures forward output cosine
similarity, forward maximum absolute difference $|\Delta|$, and gradient
relative error across parameters with significant gradient norms
($\|\nabla\| > 10^{-6}$), across $18$ configurations ($N \in \{128, 256, 512\}$,
$K \in \{4, 16\}$, $3$ seeds each) in fp16 on CUDA. This is a load-bearing
script for the paper; it writes no CSV (console output only).

Result (verbatim from the report): cosine similarity $0.9995$–$1.0010$, forward
$|\Delta| < 5 \times 10^{-4}$, gradient relative error $< 8 \times 10^{-4}$
across all $18$ configurations. The whole validation costs roughly $2$
GPU-minutes versus $600{+}$ for a full training run — a $300\times$ reduction —
so the faster mask-KNN inherits the gather-KNN tracking accuracy without
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
python benchmark_attn/benchmark_full.py \
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

The script's echo block states "K = 4 16", but `benchmark_full.py` line 167
hardcodes `Ks = [4, 16, 32, 64, 128]`. The actual submitted run will therefore
sweep K = {4, 16, 32, 64, 128}, not just {4, 16}. If only K={4,16} is desired,
edit `benchmark_full.py:167` before submitting.

---

## 5. Experiment status summary

Status legend: DONE = measured artifact exists on disk (locally and/or on the
cluster); PENDING = code ready, not yet run / H100-only / cluster capacity
needed; NOT-NEEDED = de-prioritized by the Curation decision (2026-08-04).

### 5.1 `benchmark_attn` experiments

| Experiment | Script / Slurm | Status | Resources | Outputs |
|---|---|---|---|---|
| Dense vs sparse sweep (N incl. 1024/4096, NSA sel_blocks 16/64) | `benchmarks/benchmark_sparse.py` | **DONE** (2026-08-04, superset run) | A500, `benchmark_attn/.venv` | `benchmark_sparse_results.csv`, `benchmark_sparse.html` |
| Full method set × N (dense_masked, dense_flash, gather/mask-KNN, NSA, KNN-RelPos) | `benchmark_full.py` | **DONE** (2026-06-17) | A500 | `full_bench_a500.csv` |
| H100 full benchmark (same script, K sweep) | `run_full_bench.slurm` | **PENDING** (H100-only; not run) | H100, 90G, 8 CPUs, 1 h | `full_bench_h100.csv` (cluster) |
| CachedDistAttention real measurement | `benchmark_cached_dist.py` | **DONE** (2026-08-03) | A500 (classes in local `model_parts.py`) | `cached_dist_results.csv` |
| GatherSparseAttention V1/V2/V3 | `benchmark_gather_v3.py` | **DONE** (2026-08-03) | A500 | `gather_v3_results.csv` |
| KNN method comparison (analytical) | `benchmark_knn_methods.py` | **DONE** (analytical only) | CPU only | `knn_methods_results.csv` + PNGs |
| Mask vs gather (N≤512) | `benchmark_mask_vs_gather.py` | **DONE** (superseded) | A500 | `benchmark_mask_vs_gather.csv` |
| All spatial-cutoff methods | `benchmark_all_spatial_methods.py` | **DONE** | A500 | `all_spatial_methods.csv` (gather_knn_ms broken) |
| Sparse sweep visualization | `plot_sparse.py` | **DONE** | CPU only | `benchmark_sparse.html` |
| Backward pass at N=8192 | `benchmark_backward/benchmark_sparse_backward.py` | **DONE** (2026-08-04) | A500 | `benchmark_backward/sparse_backward_results_8192.csv` |
| Full-model speed/mem K=64 | `benchmark_training/benchmarks/benchmark_speed_mem.py` | **DONE** (2026-08-04) | A500 | `benchmark_training/results/training_speed_memory_by_N_K.csv`, copy at `benchmark_attn/speed_mem.csv` |
| DINOv3 comparison | `benchmark_dinov3_comparison*.py` | **NOT-NEEDED** (HTTP 403 — weights download blocked) | internet | `dinov3_comparison*.csv` |

### 5.2 Cluster experiments per the Curation decision (2026-08-04)

| Experiment | Status | Notes |
|---|---|---|
| Multi-seed K-sweep (5 K × seeds 42/43/44, 15 jobs via `run_single_config.slurm`) | **PENDING / NECESSARY** | closes the single-seed limitation; 48 h budget each, expect 6.4–11 h (early-stopped) |
| DeepCell cross-dataset eval incl. KNN checkpoints | **PENDING / NECESSARY** | edit `CHECKPOINTS` in `cross_dataset_eval.py`; inference-only, fits 1 GPU |
| Pull 5 clean K-sweep run dirs from `$TRK/runs/2026-06-01_23-04-*_clean/` | **PENDING** | needed for K=4/K=64 checkpoint reproducibility |
| DINOv2 full SSL pretraining (24 h) | **NOT-NEEDED** | micro-tests + edge-probe ceiling already support the conclusion |
| Low-label regime sweep (12 h) | **NOT-NEEDED** | SSL showed zero improvement at 10% labels |
| CNN extended/race/lambda re-runs | **NOT-NEEDED** | already done (jobs 3768419, 3719326, 3710363) |
| `phase1_*`, `dist_ablation`, `diag2`, `quick_bench`, `ssl_only`, `ssl_d2d` | **NOT-NEEDED** | diagnostic/earlier exploratory; not required for report claims |
| CNN feature-injection training (8 variants) | **NOT-NEEDED** unless re-running | H100-only (A500 OOM measured) |

---

## 6. Measured results ledger (DONE experiments)

Numbers verbatim from `labbook/2026-08-03_experiment_readiness_inventory.md`
§1.x / §5, cross-checked against the CSVs on disk. All A500 fp16 unless stated.

### 6.1 `full_bench_a500.csv` — mask-KNN at N ≥ 2048 (labbook §1.2)

| N | mask-KNN K=4 | mask-KNN K=16 |
|---|---|---|
| 2048 | 1.41 ms / 37 MB | 1.48 ms / 37 MB |
| 4096 | 5.19 ms / 138 MB | 5.35 ms / 138 MB |
| 8192 | 24.9 ms / 532 MB | 26.1 ms / 532 MB |

Fits the A500 at N=8192. This resolves the `\pending{mask-KNN benchmark at
N≥2048}` marker in `argumentation.tex`.

### 6.2 `benchmark_sparse_results.csv` — N=1024/4096 sweep (labbook §5.5)

L=1, N=4096: `dense` 0.0392 s / 556 MB; `dense_flash` 0.0029 s;
`sparse` K16 0.0204 s / 204 MB; `sparse_v2` K16 0.0205 s.
L=4, N=4096, K=64: **OOM** for both `sparse` and `sparse_v2`.
L=1, N=8192: `dense` OOM; `dense_flash` 0.0107 s / 40 MB; `sparse` K4/K16/K64 =
0.0265/0.0414/0.0963 s (120/408/1560 MB); `sparse_v2` similar.

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

Nuance for the report: the "~2× speedup" claim only holds at small N (≤ 256);
at N = 512–2048 it collapses to ~1.05×, and N=8192 OOMs on the A500. The report's
"~2×" CachedDistAttention claim must be qualified to the small-N regime (the
dominant regime in vanvliet). Raw CSV nuance: `dense_masked`@8192 actually
succeeded (190.0 ms / 1487 MB); only `cached_dist`@8192 OOM'd in the on-disk CSV.

### 6.5 `gather_v3_results.csv` — GatherSparseAttentionV3 (labbook §1.5)

V3/V1 ratio (< 1 means V3 faster), fp16, d=320, h=8, B=2:

| N | K=4 | K=16 | K=64 |
|---|---|---|---|
| 128 | 0.83× | 0.55× | 0.60× |
| 256 | 0.46× | 0.38× | 0.58× |
| 512 | 0.21× | 0.36× | 0.57× |
| 1024 | 0.20× | 0.34× | 0.59× |
| 2048 | 0.20× | 0.35× | 0.56× |

V3 is consistently faster than V1/V2 everywhere — up to **5× at N ≥ 1024**
(0.20×) and 1.2–2.6× at small N. This contradicts the report's "V3 does not
exist" narrative.

### 6.6 Backward pass at N=8192 (labbook §5.6) — `benchmark_backward/sparse_backward_results_8192.csv`

All methods fit, **NO OOM** (contrary to expectation):

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
Headline: **K=64, N=512 = 194 ms / 1604 MB** (fits A500). FIX APPLIED to
`benchmark_training/benchmarks/benchmark_speed_mem.py`: the broken
`from track_encoder import AgentCentricNormalization` import was resolved by
inlining the class; `wandb` was installed into `benchmark_attn/.venv`.

---

## 7. Early-stopping guidance

**Only relevant for training runs** (e.g. the NECESSARY multi-seed K-sweep).
The attention benchmarks in this directory are single-shot measurements and do
**not** need early stopping — there is no training loop to stop.

For the cluster training runs: never run to 500 epochs if `val_loss` has
plateaued — patience 83 already stops near-optimal; `--epochs 500` is only an
upper bound.

---

## 8. Known discrepancies / notes (found 2026-08-04)

1. `benchmarks/benchmark_sparse.py` working tree currently has `N_vals` without 1024/4096
   (line 214) and `sel_blocks = [2, 4, 8]` (line 244), but the on-disk
   `benchmark_sparse_results.csv` (Aug 4) covers N = 128…8192 and
   sel ∈ {2,4,8,16,64}. Reproduce the superset CSV by editing those two lines
   before running; otherwise the smaller sweep overwrites the superset data.
2. `run_full_bench.slurm` echoes "K = 4 16" but `benchmark_full.py:167`
   hardcodes `Ks = [4, 16, 32, 64, 128]`; the H100 job will actually sweep all
   five K values.
3. `README_knn_methods.md` documents a `--gpu` mode that the current
   `benchmark_knn_methods.py` does not implement (analytical only).
4. `cached_dist_results.csv`: `dense_masked`@8192 succeeded (190.0 ms/1487 MB)
   while `cached_dist`@8192 OOM'd; the labbook writes "OOM (both variants)" —
   the on-disk CSV records OOM only for the cached variant.
5. `all_spatial_methods.csv` `gather_knn_ms` is broken (−1 everywhere, shape
   bug) — do not cite.
6. `benchmark_mask_vs_gather.py`'s output CSV historically landed at the repo
   root (`research-proj/benchmark_mask_vs_gather.csv`) because it was run from
   there; run from `benchmark_attn/` to keep outputs together.
7. The cluster slurm script installs torch from the cu124 index while the local
   venv is torch 2.12.0+cu130 — a mismatch to be aware of when comparing
   A500 (cu130) vs H100 (cu124) timings.
