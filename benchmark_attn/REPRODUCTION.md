# REPRODUCTION.md — Attention / Memory Benchmarks (`benchmark_attn/`)

Reproduction guide for every experiment in this directory: exact CLI invocations, required
environments, output files, and the measured-results index.

Authoritative sources: the benchmark scripts, `slurm/run_full_bench.slurm`, the CSVs under
`results/`, and the local environment managed by `uv sync` (Python 3.14.4, torch 2.12.0+cu130).

---

## 1. Local (A500) vs Cluster (H100) split

Two hardware environments are involved. All benchmarks in this directory run **locally on the
RTX A500 Laptop** (4096 MiB, compute cap 8.6, Ampere) except `slurm/run_full_bench.slurm`,
which targets an H100 (80 GB) on the TU Dresden Capella cluster.

| Environment | Hardware | GPU VRAM | Invocation | Used for |
|---|---|---|---|---|
| Local (this repo) | NVIDIA RTX A500 Laptop | 4 GiB | `uv run python <script>` | `scripts/benchmarks/benchmark_sweep.py`, `scripts/benchmarks/benchmark_cached_dist.py`, `tests/validate_equivalence.py`, `analysis/plot_sparse.py` |
| Cluster (Capella) | NVIDIA H100 | 80 GiB | `uv run python <script>` | `slurm/run_full_bench.slurm` → `scripts/benchmarks/benchmark_sweep.py` → `full_bench_h100.csv` |

All local benchmarks use fp16 on CUDA (`dtype = torch.float16`). Scripts assert
`torch.cuda.is_available()` and are not intended to run on CPU.

Verified local packages: `torch 2.12.0+cu130`, `native_sparse_attention_pytorch`,
`seaborn 0.13.2`, `pandas 3.0.3`, `matplotlib 3.11.0`.

```bash
cd benchmark_attn
uv sync   # one-time setup; creates .venv/ and installs all deps from pyproject.toml
```

Run every local benchmark from inside `benchmark_attn/` so `src/attention_modules.py` and the
CSV paths resolve.

---

## 2. Local (A500) experiments

### 2.1 `scripts/benchmarks/benchmark_sweep.py` — method × L × N × K sweep

Purpose: forward time + incremental peak GPU memory for every registered method
(`dense_masked`, `dense_flash`, `nsa`, `gather_sdpa`, `gather_fused`, `gather_matmul`,
`mask_knn`, `knn_relpos`, `minimax`) over L × N × K. Writes `method,L,N,K,time_ms,memory_mb,error`
rows; OOM and other failures land in the `error` column.

```bash
cd benchmark_attn && uv run python scripts/benchmarks/benchmark_sweep.py \
    --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,dense_flash,dense_masked,nsa,knn_relpos,minimax \
    --Ns 128,256,512,1024,2048,4096,8192 --Ks 4,16,64 --layers 1,4 \
    --mode none --dist-mode v1 --d 320 --nhead 8 --coord-dim 2 --seed 42 \
    --out results/full_bench_a500.csv
```

Notes:
- NSA `K` column is `num_selected_blocks` (block selection), **not** a KNN-K. On the A500 keep
  N ≤ 512 for NSA K=16/64 (sel≥16 OOMs at N≥2048; L=4 N=512 sel≥16 OOMs).
- `--warmup`/`--rep` are accepted for CLI compatibility; the harness governs effective timing
  (5 warmup iterations, 0.3 s autorange).
- Reproducing the on-disk superset CSV `results/benchmark_sparse_results.csv`
  (N=128..8192, NSA sel_blocks={2,4,8,16,64}) requires the full `--Ns`/`--Ks`/`--layers` flags
  above; a smaller sweep overwrites it.

### 2.2 Gather variants (same script)

```bash
cd benchmark_attn && uv run python scripts/benchmarks/benchmark_sweep.py \
    --methods gather_sdpa,gather_fused,gather_matmul \
    --out results/sweep_results.csv
```

Pre-consolidation measurements live in `results/gather_v3_results.csv` (V1=SDPA, V2=Fused,
V3=Matmul); see §6.4.

### 2.3 `scripts/benchmarks/benchmark_cached_dist.py` — CachedDistAttention measurement

Purpose: per-layer forward of `dense_masked` (RelativePositionalAttention) vs
`CachedDistAttention` plus the one-time 2D `torch.cdist` cost; reports the amortized L=12
(6 enc + 6 dec) totals. Classes are self-contained copies in `src/attention_modules.py` — no
trackastra dependency.

```bash
cd benchmark_attn && uv run python scripts/benchmarks/benchmark_cached_dist.py
# writes results/cached_dist_results.csv  (method,N,time_ms,memory_mb,error)
```

Note: `dist_2d` must be cast to the model dtype before the v1 decay add (the real model
computes it in fp32, which breaks the fp16 SDPA mask dtype on torch 2.12+cu130).

### 2.4 `tests/validate_equivalence.py` — numerical equivalence

Purpose: verifies `GatherSparseAttention` vs `KNNMaskSparseAttention` produce numerically
identical outputs (forward cosine similarity, forward max abs diff, gradient relative error)
across 18 configs (N∈{128,256,512}, K∈{4,16}, 3 seeds each) in fp16. Console output only.

```bash
cd benchmark_attn && uv run python tests/validate_equivalence.py
```

Result: cosine similarity 0.9995–1.0010, forward |Δ| < 5e-4, gradient relative error < 8e-4.

### 2.7 `analysis/plot_sparse.py` — visualization

Purpose: seaborn/matplotlib figures from `results/benchmark_sparse_results.csv` into a single
self-contained HTML (figures embedded as base64 PNGs). No CLI args, no GPU.

```bash
cd benchmark_attn && uv run python analysis/plot_sparse.py
# reads  results/benchmark_sparse_results.csv
# writes results/benchmark_sparse.html
```

---

## 3. Cluster requirements

### 3.1 One-time setup (login node)

```bash
git clone <repo-url> bench-transformer-cell-tracking
cd bench-transformer-cell-tracking
bash slurm/setup_env.sh          # creates ~/.bench.env, runs uv sync
```

`setup_env.sh` will prompt you to edit `~/.bench.env` — replace the
`/path/to/...` placeholders with your actual cluster paths.  Then re-run
the script to complete the setup.

This creates `.venv/` in each subproject and installs all dependencies
from `pyproject.toml`.  Slurm jobs use `.venv/bin/python` directly —
**no `uv sync` runs inside the job**, so the full time budget is
available for the actual benchmark.

### 3.2 Updating dependencies

If `pyproject.toml` changes after a `git pull`:

```bash
cd benchmark_attn && uv sync    # re-syncs .venv/
```

### 3.3 Cluster access

Access: SSH host `capella` (VPN required). Partition `gpu-h100`, account
`p_scads_celltracking`, 1 GPU per node.

---

## 4. `slurm/run_full_bench.slurm` — H100 full attention benchmark

SBATCH resources (as declared in the script): job `attn_full_bench`, account
`p_scads_celltracking`, partition `gpu-h100`, time `01:00:00`, 1 GPU, 8 CPUs, 90 GB mem,
logs `logs/full_bench_%j.{out,err}`.

In-script env: `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`,
`MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK`, `NCCL_DEBUG=WARN`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONNOUSERSITE=1`, `unset PYTHONPATH PYTHONHOME`.
The script sources `~/.bench.env` (gitignored, set once via `slurm/setup_env.sh`),
sources `slurm/load_cluster_env.sh`, then invokes `.venv/bin/python` directly
(no `uv sync` overhead — the `.venv/` was created during one-time setup).

Submit:

```bash
sbatch slurm/run_full_bench.slurm
```

Equivalent manual command (also the one the slurm script should run):

```bash
cd "$BENCH/benchmark_attn" && .venv/bin/python scripts/benchmarks/benchmark_sweep.py \
    --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,dense_flash,dense_masked,nsa,knn_relpos,minimax \
    --d 320 --nhead 8 --warmup 10 --rep 50 --Ks 4,16,32,64 \
    --out results/full_bench_h100.csv
```

(`$BENCH` comes from `~/.bench.env`.)

Purpose: full-method attention benchmark on the H100 (80 GB) as an A500-vs-H100 reference; the
largest N (8192) runs without the A500's 4 GB limit.

Completed run: job `3898011` (logs `logs/full_bench_3898011.{out,err}`) finished successfully.
Output: `results/full_bench_h100.csv` — 168 rows, all successful (zero OOMs).
Largest single-layer memory: minimax@N=8192 = 44 GB, i.e. 55 % of the 80 GB GPU.
NSA timing is near-constant at ~2.5 ms from N=128 up to N=4096, then doubles to 5.87 ms at N=8192.

---

## 5. Experiment inventory

Artifacts on disk are the source of truth; this table maps each experiment to its entrypoint
and output.

| Experiment | Entrypoint | Output artifact |
|---|---|---|
| Dense/sparse sweep incl. NSA sel_blocks (superset) | `scripts/benchmarks/benchmark_sweep.py` | `results/benchmark_sparse_results.csv`, `results/benchmark_sparse.html` |
| Full method set × N (A500) | `scripts/benchmarks/benchmark_sweep.py` | `results/full_bench_a500.csv` |
| Gather SDPA/Fused/Matmul | `scripts/benchmarks/benchmark_sweep.py` | `results/gather_v3_results.csv` |
| CachedDistAttention measurement | `scripts/benchmarks/benchmark_cached_dist.py` | `results/cached_dist_results.csv` |
| Mask vs gather (N ≤ 512) | `scripts/benchmarks/benchmark_mask_vs_gather.py` | `benchmark_mask_vs_gather.csv` (CWD) |
| Spatial-cutoff methods | `scripts/benchmarks/benchmark_all_spatial_methods.py` | `results/all_spatial_methods.csv` |
| KNN equivalence validation | `tests/validate_equivalence.py` | console only |
| Sparse sweep visualization | `analysis/plot_sparse.py` | `results/benchmark_sparse.html` |
| H100 full benchmark | `slurm/run_full_bench.slurm` | `results/full_bench_h100.csv` (cluster) |
| Backward pass at N=8192 | `../deprecated/benchmark_backward/benchmark_sparse_backward.py` | `../deprecated/benchmark_backward/sparse_backward_results_8192.csv` |
| Full-model speed/mem (K × N) | `../benchmark_training/scripts/benchmarks/benchmark_speed_mem.py` | `results/speed_mem.csv` |
| DINOv3 representation comparison | `exploratory/benchmark_dinov3_comparison_v2.py` | `results/dinov3_comparison.csv`, `results/dinov3_comparison_v2.csv` |

---

## 6. Measured results index

CSVs render as interactive tables on GitHub.

### 6.1 `results/full_bench_a500.csv` — full method set on the A500

Data: [full_bench_a500.csv](results/full_bench_a500.csv).

Legacy schema (`method,N,time_ms,memory_mb,error`; names like `mask-KNN_K=16`). mask_knn fits
the A500 at N=8192 (26.1 ms / 532 MB at K=16).

### 6.2 `results/benchmark_sparse_results.csv` — dense vs sparse sweep, incl. NSA

Data: [benchmark_sparse_results.csv](results/benchmark_sparse_results.csv).

Legacy schema (`method,L,N,K,reorder,time_s,mem_mb,status`; methods `dense`, `sparse`,
`sparse_v2`, `nsa`). NSA `K` = `num_selected_blocks` (not KNN-K): L=1 N=512 sel=16 is OK
(0.0197 s / 546.8 MB); L=4 N=512 sel≥16 and N≥2048 sel∈{16,64} OOM on the A500.

### 6.3 `results/cached_dist_results.csv` — CachedDistAttention per-layer & L=12

Data: [cached_dist_results.csv](results/cached_dist_results.csv).

Speedup is ~1.7–1.9× at N ≤ 256, ~1.05× at N=512–2048, 1.22× at N=4096. In the on-disk CSV
`dense_masked`@8192 fits (190.0 ms / 1487 MB) while `cached_dist`@8192 OOMs.

### 6.4 `results/gather_v3_results.csv` — gather SDPA/Fused/Matmul

Data: [gather_v3_results.csv](results/gather_v3_results.csv).

Legacy V1/V2/V3 naming (V1=SDPA, V2=Fused, V3=Matmul). gather_matmul is up to 5× faster than
gather_sdpa at N ≥ 1024.

### 6.5 Backward pass at N=8192

Data: [sparse_backward_results_8192.csv](../deprecated/benchmark_backward/sparse_backward_results_8192.csv).

All methods fit with no OOM (d=320, nhead=8, B=1, fp16); fastest is `sparse_softmax` K4
(4.6 ms fwd / 10.8 ms bwd / 81 MB).

### 6.6 `results/speed_mem.csv` — full-model speed/memory

Data: [speed_mem.csv](results/speed_mem.csv).

Full-model fp32 fwd+bwd+AdamW over K {0,4,8,16,32,64} × N {128,256,512}. Headline:
K=64, N=512 = 194 ms / 1604 MB (fits the A500).

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

---

## 7. Early-stopping guidance

Only relevant to training runs (e.g. a multi-seed K-sweep), not to the single-shot attention
benchmarks in this directory. For training: never run to 500 epochs once `val_loss` plateaus —
patience 83 already stops near-optimal; `--epochs 500` is only an upper bound.

---

## 8. Known discrepancies / notes

1. **Legacy superset CSV.** `results/benchmark_sparse_results.csv` was produced by the superset
   sweep (N=128..8192, NSA sel_blocks={2,4,8,16,64}). Reproduce with the full flags in §2.1
   (`--Ns 128,256,512,1024,2048,4096,8192 --Ks 4,16,64 --layers 1,4`; NSA `sel_blocks` is not a
   CLI knob — the class default `num_selected_blocks=4` applies); a smaller sweep overwrites the
   superset data.
