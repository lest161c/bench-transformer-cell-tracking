# SPEC — `benchmark_training` Refactor

## Goal

Refactor `benchmark_training` to comply with `AGENTS.md`, adopt a
tractable directory layout, migrate benchmarks that belong to other
repositories, and document the hardware required to reproduce each
experiment.

## Scope

This repository benchmarks **training performance** of Trackastra —
both resource usage (time, memory) and evaluation results (loss,
accuracy). Isolated attention benchmarks live in `benchmark_attn/`;
isolated SSL proof-of-concept benchmarks live in `benchmark_ssl/`.

**In scope:**
- Reorganise the directory tree to the AGENTS.md standard layout.
- Extract duplicated code from the benchmark scripts into a shared
  library under `src/`.
- Reduce the eight SLURM scripts to two: one unified training script
  with selectable variants, one diagnostic / smoke-test script.
- Move the three analysis (plotting) scripts under `scripts/analysis/`
  and give them proper `main()` functions.
- Migrate benchmarks that are more concerned with isolated attention
  performance or SSL transfer quality to their appropriate repositories.
- Rewrite `REPRODUCTION.md` to be short, CLI-focused, and free of
  hard-coded cluster paths.
- Rewrite `README.md` to document the new structure and the hardware
  required for each class of experiment.

**Out of scope:**
- Changing the underlying model code (`GatherSparseAttention`, SSL
  trainer, etc.) — those live in `benchmark_attn` / `benchmark_ssl`.
- Changing the SLURM job accounting — they are environment facts.
- Re-running any experiments.

## Target directory layout

```
benchmark_training/
├── src/                                # shared library
│   ├── __init__.py
│   ├── models.py                       # SinusoidalPE, AgentCentricNormalization,
│   │                                   #   DenseEncoder, SparseEncoder, KNN helper
│   ├── data.py                         # synthetic batch + real-vanvliet pair loaders
│   ├── ssl_transfer.py                 # init_from_ssl (dense + sparse paths)
│   └── bench_utils.py                  # timing / memory measurement, CSV I/O
├── scripts/
│   ├── benchmarks/                     # training performance benchmarks
│   │   ├── benchmark_speed_mem.py      # N=8192 training resource benchmark
│   │   ├── benchmark_combined.py       # 2x2 factorial convergence benchmark
│   │   └── benchmark_ablation.py       # convergence ablation: attention × K × L
│   ├── analysis/                       # plot / report generators
│   │   ├── plot_ablation.py
│   │   ├── plot_combined.py
│   │   └── plot_comprehensive.py
│   └── slurm/                          # HPC submission scripts
│       ├── run_training.slurm          # unified training, --export=VARIANT=<name>
│       └── run_diagnostic.slurm        # smoke tests + quick-bench
├── configs/                            # training YAML configs
│   ├── vanvliet_baseline.yaml
│   └── vanvliet_sparse_k16_ssl.yaml
├── results/                            # generated CSVs and HTML reports (reproducible)
├── docs/                               # documentation
│   ├── SPEC.md                         # this file
│   ├── REPRODUCTION.md                 # reproduction guide
│   ├── analysis_ablation.md            # analysis notes (migrated from analysis/)
│   ├── analysis_n1024.md               # analysis notes (migrated from analysis/)
│   └── results_analysis.md             # analysis notes (migrated from analysis/)
└── README.md                           # this file
```

## Migration decisions

The original `benchmark_training` contained 8 benchmark scripts. After
auditing each against the criterion "clearly relevant to benchmarking
the new training performance of Trackastra", the following migrations
were made:

| Script | Migrated to | Reason |
|---|---|---|
| Speed sweep mode (`--mode sweep`) | `benchmark_attn/` | Pure attention layer timing; redundant with `benchmark_sweep.py` |
| Ablation Phase 1 (1-epoch speed scaling) | `benchmark_attn/` | Isolated attention scaling; redundant with `benchmark_sweep.py` |
| `benchmark_downstream.py` (SSL transfer evaluation) | `benchmark_ssl/scripts/run_downstream_matched.py` | Isolated SSL proof-of-concept; tests SSL transfer quality |

The remaining 3 benchmark scripts are clearly about training performance:

1. **`benchmark_speed_mem.py`** — Measures training step time + peak
   memory at N=8192 for five attention+SSL variants. Tests the memory
   ceiling of the full Trackastra association model.
2. **`benchmark_combined.py`** — 2×2 factorial convergence benchmark:
   dense/sparse × random/SSL init. Measures wall-clock convergence.
3. **`benchmark_ablation.py`** — Convergence ablation: dense/sparseK4/
   sparseK16 × L=1,4 × N=512 over 15 epochs.

## Consolidation rules

The original 8 benchmark scripts shared ~70% of their code. The
duplicated components were extracted into the shared `src/` library:

| Component | Source files | Destination |
|---|---|---|
| `SinusoidalPositionalEncoding` / `PE` | 3 copies | `src/models.py` |
| `AgentCentricNormalization` | 4 copies | `src/models.py` |
| `DenseEncoder` | 4 copies | `src/models.py` |
| `SparseEncoder` | 5 copies | `src/models.py` |
| KNN-from-coordinates helper | 5 copies | `src/models.py` |
| `load_real_pairs` / `create_tracking_pairs` | 3 copies | `src/data.py` |
| `make_synthetic_batch` / `make_synthetic_data` | 2 copies | `src/data.py` |
| `init_from_ssl` (dense + sparse variants) | 3 copies | `src/ssl_transfer.py` |
| `measure_timing` / CSV row writing | 4 copies | `src/bench_utils.py` |

## Unified SLURM script

`scripts/slurm/run_training.slurm` replaces the seven original training
scripts with a single `--export=VARIANT=<name>` interface:

```bash
sbatch --export=VARIANT=baseline scripts/slurm/run_training.slurm
sbatch --export=VARIANT=sparse_k16 scripts/slurm/run_training.slurm
sbatch --export=VARIANT=sparse_k16_ssl scripts/slurm/run_training.slurm
```

`run_diagnostic.slurm` covers the smoke-test path and quick-bench runs.

## Acceptance criteria

1. Every Python file in `benchmark_training/` passes `ast.parse()`.
2. Every module, class, and function in `src/` has a docstring.
3. No script under `scripts/` runs work at import time.
4. `scripts/slurm/` contains exactly two files.
5. `docs/REPRODUCTION.md` is ≤ 200 lines and contains no hard-coded
   cluster paths.
6. `README.md` documents the A500-vs-H100 split per experiment.
7. `results/` contains only files whose producer is in the new
   `scripts/`.
8. No dangling references to removed or migrated scripts.

## Risks and assumptions

- **Assumption:** The `benchmark_attn/` `benchmark_sweep.py` already
  provides the isolated attention layer timing that the removed sweep
  mode was doing. If more isolated attention benchmarks are needed,
  they should be added to `benchmark_attn/`, not `benchmark_training/`.
- **Assumption:** The `benchmark_ssl/scripts/run_downstream_matched.py`
  script imports shared library code from `benchmark_training/src/` via
  `sys.path` manipulation. This cross-repo dependency is acceptable
  because the downstream benchmark tests SSL transfer quality (an SSL
  concern) but uses Trackastra's model components.
- **Assumption:** The remaining benchmark scripts use synthetic data
  (not real vanvliet data). This is acceptable for measuring training
  performance (time, memory, convergence) because the synthetic data
  controls for cell count and association complexity. Real data
  experiments are handled by the SLURM training scripts.
