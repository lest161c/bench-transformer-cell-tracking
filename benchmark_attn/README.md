# Gather-Sparse Attention Benchmark

Standalone benchmarks for sparse attention mechanisms in transformer-based cell tracking, part of the **Trackastra** research project.

**Core idea:** Replace N×N dense attention with KNN-gathered N×K sparse attention. Precompute KNN indices, gather K/V, run SDPA on N×K tensors. Complexity drops from O(N²) to O(NK).

## Key Results

Tested on NVIDIA A500 (fp16, B=2, d=256, h=4). All numbers single-layer forward pass.

| N | Method | Time (s) | GPU Mem (MB) | vs Dense |
|---|--------|----------|-------------|----------|
| 2048 | Dense | 0.0085 | 142 | 1× |
| 2048 | Sparse K=16 | 0.0089 | 102 | 0.96× speed, 1.4× mem savings |
| 8192 | Dense | 0.1391 | 2200 | 1× |
| 8192 | Sparse K=4 | 0.0237 | 120 | **5.9× faster**, 18× less mem |
| 8192 | Sparse K=16 | 0.0361 | 408 | **3.9× faster**, 5.4× less mem |
| 8192 L=4 | Dense | OOM | OOM | — |
| 8192 L=4 | Sparse K=4 | 0.0935 | 385 | Fits when dense OOMs |

**Spatial reorder** (Wu et al., Point Transformer V3, CVPR 2024): negligible speedup on random data (~1-3%). Real data may differ.

**Attention backends:** FlashAttention/CuDNN vs Math backend ~1.5-2× throughput difference at N≥512.

## Setup

```bash
uv sync
```

Requires Python ≥3.14, PyTorch ≥2.11 with CUDA.

## Usage

All scripts run from the `benchmark_attn/` directory with the project venv:
`cd benchmark_attn && .venv/bin/python <script>`.

### Core benchmarks

```bash
# Main sparse vs dense sweep (N up to 8192, K 4-64; extend via --Ns/--Ks flags)
python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,dense_flash,dense_masked --Ks 4,16,64

# Unified benchmark: all methods × all N, forward + peak memory (mask_knn at N>=2048)
python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,gather_fused,gather_matmul,mask_knn,knn_relpos,minimax,nsa,dense_flash,dense_masked

# CachedDistAttention real measurement (classes in local src/attention_modules.py)
python scripts/benchmarks/benchmark_cached_dist.py

# Gather variants: gather_sdpa (GatherSparseAttention), gather_fused (GatherSparseFusedAttention), gather_matmul (GatherSparseMatmulAttention)
python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,gather_fused,gather_matmul

# Profiler (per-operation CUDA time breakdown) → profiler_out/profile_gather_results.json
# Merged profiler: general mode (dense/sparse sweep) or gather mode (GatherSparseAttention)
python scripts/benchmarks/benchmark_profiler.py --mode general
python scripts/benchmarks/benchmark_profiler.py --mode gather

# Mask vs gather numerical equivalence across 18 configs (console output)
python tests/validate_equivalence.py

# Plot main results → benchmark_sparse.html
python analysis/plot_sparse.py
```

### Additional benchmarks

```bash
# Mask vs gather at small N (N ≤ 512)
python scripts/benchmarks/benchmark_mask_vs_gather.py

# Generic profiler wrapper
python scripts/benchmarks/benchmark_profiler.py

# All spatial methods (NOTE: gather_knn_ms column is not populated; do not cite gather)
python scripts/benchmarks/benchmark_all_spatial_methods.py

# Small-N behavior (N ≤ 512)
python scripts/benchmarks/benchmark_sweep.py --methods gather_sdpa,gather_fused,gather_matmul --Ns 128,256,512 --Ks 16 --layers 1
```

### Exploratory / one-off

```bash
python exploratory/benchmark_pure_attn.py
python exploratory/benchmark_soft_decay_measure.py
python exploratory/benchmark_no_bias_flashattn.py
python exploratory/benchmark_no_mask_soft_decay.py
python exploratory/benchmark_flex_spatial.py
python exploratory/benchmark_spatial_block_partition.py
python exploratory/benchmark_spatial_flash.py
python exploratory/benchmark_spatial_cutoff_verification.py
python exploratory/benchmark_data_pipeline.py
python exploratory/benchmark_sdpa_backends.py
python exploratory/benchmark_system_impact.py
python exploratory/benchmark_trackastra_inference.py
python exploratory/benchmark_dinov3_comparison_v2.py
python exploratory/benchmark_reorder_real_vs_random.py
python exploratory/profiler_recipe.py
```

### Analysis / visualization

```bash
# Tile size analysis
python analysis/tile_size_analysis.py

# Verify CUDA attention backends
python analysis/verify_cudnn.py

# Tra-AOGM visualization
python analysis/tra_aogm_visualization.py

# Collapse visualization
python analysis/vis_collapse.py

# Generate the HTML dashboard (aggregates results/ artifacts)
python analysis/make_report.py
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
│       ├── benchmark_mask_vs_gather.py # mask vs gather + SDPA dispatch
│       ├── benchmark_all_spatial_methods.py # FlexAttention comparison
│       └── benchmark_profiler.py   # merged profiler (general + gather)
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
