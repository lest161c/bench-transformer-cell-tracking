# Gather-Sparse Attention Benchmark

Standalone benchmarks for sparse attention mechanisms in transformer-based cell tracking, part of the **Trackastra** research project.

**Core idea:** Replace N×N dense attention with KNN-gathered N×K sparse attention. Precompute KNN indices, gather K/V, run SDPA on N×K tensors. Complexity drops from O(N²) to O(NK).

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
| 8192 | gather_matmul K=4 | — | — | 0.36 | 72 | ties dense_flash (0.36 vs 0.35 ms) |

Key observations:

- **The sparse advantage grows on faster hardware.** gather_sdpa K=4 is 6.5× faster than dense_masked on the A500 (79.9 → 12.2 ms) but 10.3× faster on the H100 (5.6 → 0.54 ms). Dense attention is O(N²) work, gathered sparse attention O(NK); the faster GPU amplifies the gap.
- **Crossover confirmed for gather_matmul.** On the H100, gather_matmul K=4 ties dense_flash at N=8192 (0.36 ms vs 0.35 ms) — the predicted sparse/dense crossover is now confirmed for gather_matmul.
- **NSA timing is near-constant ~2.5 ms** up to N=4096 on the H100 (2.46–2.62 ms), jumping to 5.9 ms at N=8192.
- **minimax@8192 uses 44 GB** on the H100 — the largest single-layer footprint, closest to the 80 GB limit.
- **Zero OOMs across all 168 configurations** on the H100 (9 methods × 7 N values × up to 4 K values).
- Legacy A500-only finding (fp16, B=2, d=256, h=4): at L=4, N=8192 dense OOMs on the 4 GiB A500 while sparse K=4 fits (0.094 s / 385 MB).

**Spatial reorder** (Wu et al., Point Transformer V3, CVPR 2024): negligible speedup on random data (~1-3%). Real data may differ.

**Attention backends:** FlashAttention/CuDNN vs Math backend ~1.5-2× throughput difference at N≥512.

## Setup

```bash
uv sync
```

Requires Python ≥3.14, PyTorch ≥2.11 with CUDA.

## Usage

All scripts run from the `benchmark_attn/` directory. After `uv sync`,
invoke any script via `uv run python <script>`:

```bash
uv run python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,dense_flash --Ks 4,16
```

### Core benchmarks

```bash
# Main sparse vs dense sweep (N up to 8192, K 4-64; extend via --Ns/--Ks flags)
uv run python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,dense_flash,dense_masked --Ks 4,16,64

# Unified benchmark: all methods × all N, forward + peak memory (mask_knn at N>=2048)
uv run python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,knn_relpos,minimax,nsa,dense_flash,dense_masked

# CachedDistAttention real measurement (classes in local src/attention_modules.py)
uv run python scripts/benchmarks/benchmark_cached_dist.py

# Gather variants: gather_sdpa (GatherSparseAttention), gather_fused (GatherSparseFusedAttention), gather_matmul (GatherSparseMatmulAttention)
uv run python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,gather_fused,gather_matmul

# Profiler (per-operation CUDA time breakdown) → profiler_out/profile_gather_results.json
# Merged profiler: general mode (dense/sparse sweep) or gather mode (GatherSparseAttention)
uv run python scripts/benchmarks/benchmark_operator_breakdown.py --mode general
uv run python scripts/benchmarks/benchmark_operator_breakdown.py --mode gather

# Mask vs gather numerical equivalence across 18 configs (console output)
uv run python tests/validate_equivalence.py

# Plot main results → benchmark_sparse.html
uv run python analysis/plot_sparse.py
```

### Additional benchmarks

```bash
# Per-operator CUDA time/memory breakdown
uv run python scripts/benchmarks/benchmark_operator_breakdown.py

# Small-N behavior (N ≤ 512)
uv run python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,gather_fused,gather_matmul --Ns 128,256,512 --Ks 16 --layers 1
```

**Not included:** FlexAttention (PyTorch's `torch.nn.attention.flex_attention`) is not benchmarked here because the sweep's claim (which sparse attention variant is fastest at cell-tracking scale) does not depend on it. FlexAttention with spatial cutoff has been compared informally against the gathered/masked variants; the gathered/masked KNN approaches dominate at the N ranges used in cell tracking.

### Exploratory / one-off

```bash
uv run python exploratory/benchmark_pure_attn.py
uv run python exploratory/benchmark_soft_decay_measure.py
uv run python exploratory/benchmark_no_bias_flashattn.py
uv run python exploratory/benchmark_no_mask_soft_decay.py
uv run python exploratory/benchmark_flex_spatial.py
uv run python exploratory/benchmark_spatial_block_partition.py
uv run python exploratory/benchmark_spatial_flash.py
uv run python exploratory/benchmark_spatial_cutoff_verification.py
uv run python exploratory/benchmark_data_pipeline.py
uv run python exploratory/benchmark_sdpa_backends.py
uv run python exploratory/benchmark_system_impact.py
uv run python exploratory/benchmark_trackastra_inference.py
uv run python exploratory/benchmark_dinov3_comparison_v2.py
uv run python exploratory/benchmark_reorder_real_vs_random.py
uv run python exploratory/profiler_recipe.py
```

### Analysis / visualization

```bash
# Tile size analysis
uv run python analysis/tile_size_analysis.py

# Verify CUDA attention backends
uv run python analysis/verify_cudnn.py

# Generate the HTML dashboard (aggregates results/ artifacts)
uv run python analysis/make_report.py
```

## Structure

```
benchmark_attn/
├── src/                            # library / implementation code
│   ├── attention_modules.py        # core attention classes
│   ├── positional_encoding.py      # RoPE + bin helpers
│   └── bench/                      # shared benchmark harness
│       ├── timing.py               # measure(), try_bench()
│       ├── data.py                 # make_inputs(), compute_knn_indices()
│       ├── registry.py             # METHOD_REGISTRY (9 methods)
│       ├── io.py                   # write_results_csv()
│       └── cli.py                  # add_common_args(), parse_int_list()
│
├── scripts/                        # runnable entry points
│   └── benchmarks/
│       ├── benchmark_sweep.py      # consolidated method × L × N × K sweep
│       ├── benchmark_cached_dist.py # CachedDistAttention measurement
│       └── benchmark_operator_breakdown.py  # per-operator CUDA time/memory breakdown
│
├── tests/                          # validation scripts
│   └── validate_equivalence.py     # mask vs gather numerical equivalence
│
├── exploratory/                    # one-off research scripts
├── docs/                           # REPRODUCTION.md, sidecar READMEs
├── results/                        # generated artifacts
├── config.yaml
├── pyproject.toml
└── README.md
```

## Reference

- \citep{gallusser2024trackastra} — Gallusser, B., & Weigert, M. (2024). *Trackastra: Transformer-based cell tracking for live-cell microscopy.* ECCV 2024. arXiv:2405.15700. — [github.com/weigertlab/trackastra](https://github.com/weigertlab/trackastra)
- \citep{hassani2023neighborhood} — Hassani, A., Walton, S., Li, J., Li, S., & Shi, H. (2023). *Neighborhood Attention Transformer.* CVPR 2023. arXiv:2204.07143.
- \citep{hassani2024faster} — Hassani, A., Hwu, W.-M., & Shi, H. (2024). *Faster Neighborhood Attention: Reducing the O(n²) Cost of Self Attention at the Threadblock Level.* NeurIPS 2024. arXiv:2403.04690.
- \citep{wu2024ptv3} — Wu, X., Jiang, L., Wang, P.-S., Liu, Z., Liu, X., Qiao, Y., Ouyang, W., He, T., & Zhao, H. (2024). *Point Transformer V3: Simpler, Faster, Stronger.* CVPR 2024. arXiv:2312.10035.
- \citep{dao2023flashattention2} — Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691.
- \citep{yuan2025nsa} — Yuan, J., Gao, H., Dai, D., et al. (2025). *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention.* arXiv:2502.11089.
- \citep{lai2026minimax} — Lai, X., Xu, W., Yang, Y., et al. (2026). *MiniMax Sparse Attention.* arXiv:2606.13392.

## Further reading

- [KNN Methods Benchmark](docs/README_knn_methods.md) — gather-KNN, mask-KNN, MiniMax, dense flash comparison
- [Pure Attention Kernel Benchmark](docs/README_pure_attn.md) — raw attention kernel isolation
- [SDPA Backend Dispatch](docs/README_sdpa_backends.md) — which SDPA backend PyTorch selects
- [System Impact Benchmark](docs/README_system_impact.md) — FLOP and memory breakdown
