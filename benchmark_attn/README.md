# Gather-Sparse Attention Benchmark

Standalone benchmarks for sparse attention mechanisms in transformer-based cell tracking, part of the **Trackastra** research project.

**Core idea:** Replace N×N dense attention with KNN-gathered N×K sparse attention. Precompute KNN indices, gather K/V, run SDPA on N×K tensors. Complexity drops from O(N²) to O(NK).

## Key Results

Tested on NVIDIA RTX 4090 (fp16, B=2, d=256, h=4). All numbers single-layer forward pass.

| N | Method | Time (s) | GPU Mem (MB) | vs Dense |
|---|--------|----------|-------------|----------|
| 2048 | Dense | 0.0085 | 142 | 1× |
| 2048 | Sparse K=16 | 0.0089 | 102 | 0.96× speed, 1.4× mem savings |
| 8192 | Dense | 0.1391 | 2200 | 1× |
| 8192 | Sparse K=4 | 0.0237 | 120 | **5.9× faster**, 18× less mem |
| 8192 | Sparse K=16 | 0.0361 | 408 | **3.9× faster**, 5.4× less mem |
| 8192 L=4 | Dense | OOM | OOM | — |
| 8192 L=4 | Sparse K=4 | 0.0935 | 385 | Fits when dense OOMs |

**Spatial reorder** (Hassani et al. 2024): negligible speedup on random data (~1-3%). Real data may differ.

**Attention backends:** FlashAttention/CuDNN vs Math backend ~1.5-2× throughput difference at N≥512.

## Setup

```bash
uv sync
```

Requires Python ≥3.14, PyTorch ≥2.11 with CUDA.

## Usage

### Core benchmarks

```bash
# Main sparse vs dense benchmark (N up to 16384, K 4-128)
python benchmarks/benchmarks/benchmark_sparse.py

# Unified benchmark: all methods × all N, forward + peak memory
python benchmarks/benchmark_full.py

# Reorder ablation with real centroids (requires data/vanvliet)
python benchmarks/benchmark_reorder_real_vs_random.py

# Plot main results → benchmark_sparse.html
python analysis/plot_sparse.py
```

### Attention mechanism benchmarks

```bash
# Pure attention variants (no spatial component)
python benchmarks/benchmark_pure_attn.py

# KNN method comparison (gather vs mask vs NSA)
python benchmarks/benchmark_knn_methods.py

# FlashAttention with learned cutoff distance
python benchmarks/benchmark_flash_with_cutoff.py

# Soft decay vs hard cutoff equivalence
python benchmarks/benchmark_soft_decay.py

# No-bias FlashAttention verification
python benchmarks/benchmark_no_bias_flashattn.py

# No-mask soft decay variant
python benchmarks/benchmark_no_mask_soft_decay.py
```

### Spatial benchmarks

```bash
# All spatial methods comparison
python benchmarks/benchmark_all_spatial_methods.py

# Spatial block partition strategies
python benchmarks/benchmark_spatial_block_partition.py

# Spatial FlashAttention solutions
python benchmarks/benchmark_spatial_flash.py

# Flexible spatial attention
python benchmarks/benchmark_flex_spatial.py

# Spatial cutoff verification
python benchmarks/benchmark_spatial_cutoff_verification.py
```

### System & profiling benchmarks

```bash
# System impact breakdown
python benchmarks/benchmark_system_impact.py

# Blockwise norm measurement
python benchmarks/benchmark_blockwise_norm.py

# Data pipeline breakdown
python benchmarks/benchmark_data_pipeline.py

# FFN vs attention ratio sweep
python benchmarks/benchmark_ffn_attention_ratio.py

# SDPA backend comparison (Math vs FlashAttention vs CuDNN)
python benchmarks/benchmark_sdpa_backends.py

# Small-N behavior
python benchmarks/benchmark_small_n.py
```

### Additional tools

```bash
# Mask vs gather equivalence verification
python analysis/validate_mask_vs_gather.py
python benchmarks/benchmark_mask_vs_gather.py

# Tile size analysis
python analysis/tile_size_analysis.py

# Cosine histogram analysis
python analysis/cosine_histogram.py

# Verify CUDA attention backends
python analysis/verify_cudnn.py

# Profiler recipe
python analysis/profiler_recipe.py
python benchmarks/benchmark_profiler.py

# Trackastra inference benchmark
python benchmarks/benchmark_trackastra_inference.py

# Dinov3 comparison
python benchmarks/benchmark_dinov3_comparison.py
python benchmarks/benchmark_dinov3_comparison_v2.py

# Tra-AOGM visualization
python analysis/tra_aogm_visualization.py

# Collapse visualization
python analysis/vis_collapse.py

# Generate HTML reports
python analysis/make_report.py
```

## Structure

```
├── model_parts.py               # Core attention modules (GatherSparseAttention, RelativePositionalAttention, etc.)
├── config.yaml                  # Default hyperparameters
├── pyproject.toml               # Dependencies
├── README.md                    # Project overview
├── README_knn_methods.md        # KNN methods documentation
├── README_pure_attn.md          # Pure attention documentation
├── README_sdpa_backends.md      # SDPA backends documentation
├── README_system_impact.md      # System impact documentation
├── REPRODUCTION.md              # Benchmark reproduction guide
├── results_analysis.md          # Results analysis notes
├── uv.lock                      # Lockfile (gitignored)
├── validate_prompt.md           # Prompt validation notes (pending removal)
│
├── benchmarks/                           # 31 benchmark scripts (benchmark_*.py + profiler/validation helpers)
│   ├── benchmark_all_spatial_methods.py            # Unified spatial methods benchmark
│   ├── benchmark_blockwise_norm.py                 # Blockwise norm
│   ├── benchmark_cached_dist.py                    # Cached distance benchmark
│   ├── benchmark_data_pipeline.py                  # Data pipeline
│   ├── benchmark_dinov3_comparison.py              # Dinov3 comparison
│   ├── benchmark_dinov3_comparison_v2.py           # Dinov3 comparison v2
│   ├── benchmark_ffn_attention_ratio.py            # FFN vs attention ratio
│   ├── benchmark_flash_with_cutoff.py              # FlashAttention with learned cutoff
│   ├── benchmark_flex_spatial.py                   # Flexible spatial
│   ├── benchmark_full.py                           # Unified benchmark: all methods × all N
│   ├── benchmark_gather_v3.py                      # Gather v3 benchmark
│   ├── benchmark_knn_methods.py                    # KNN method comparison
│   ├── benchmark_mask_vs_gather.py                 # Mask vs gather benchmark
│   ├── benchmark_no_bias_flashattn.py              # No-bias FlashAttention
│   ├── benchmark_no_mask_soft_decay.py             # No-mask soft decay
│   ├── benchmark_profiler.py                       # Profiler benchmark
│   ├── benchmark_pure_attn.py                      # Pure attention variants
│   ├── benchmark_reorder_real_vs_random.py         # Reorder speedup: real centroids vs random
│   ├── benchmark_sdpa_backends.py                  # SDPA backend comparison
│   ├── benchmark_small_n.py                        # Small-N behavior
│   ├── benchmark_soft_decay.py                     # Soft decay vs hard cutoff
│   ├── benchmark_soft_decay_measure.py             # Soft decay measurement
│   ├── benchmark_sparse.py                         # Dense vs sparse benchmark (N up to 16384, K 4-128)
│   ├── benchmark_spatial_block_partition.py        # Block partition strategies
│   ├── benchmark_spatial_cutoff_verification.py    # Cutoff verification
│   ├── benchmark_spatial_flash.py                  # Spatial FlashAttention
│   ├── benchmark_system_impact.py                  # System impact breakdown
│   ├── benchmark_trackastra_inference.py           # Trackastra inference
│   ├── profile_gather_sparse.py                    # Gather sparse profiler
│   ├── profiler_recipe.py                          # Profiler recipe
│   └── validate_mask_vs_gather.py                  # Equivalence verification
│
├── analysis/                             # Plotting / report / verification scripts
│   ├── plot_sparse.py                  # Visualization → HTML
│   ├── make_report.py                  # HTML report generator
│   ├── verify_cudnn.py                 # CUDA backend verification
│   ├── tile_size_analysis.py           # Tile size analysis
│   ├── cosine_histogram.py             # Cosine histogram
│   ├── tra_aogm_visualization.py       # AOGM visualization
│   └── vis_collapse.py                 # Collapse visualization
│
├── slurm/                                # SLURM batch scripts
│   └── run_full_bench.slurm       # SLURM batch script
│
└── results/                              # Generated artifacts (gitignored)
    ├── benchmark_sparse_results.csv       # Benchmark output data
    ├── ..._results.csv                    # Various benchmark result CSVs
    ├── ..._analysis.png                   # Benchmark visualization PNGs
    ├── comprehensive_report.html          # Full HTML report
    └── profiling_report.html              # Profiling HTML report
```

## Reference

- Gallusser & Weigert, *Trackastra*, ECCV 2024 — [github.com/weigertlab/trackastra](https://github.com/weigertlab/trackastra)
- Hassani et al., *Neighborhood Attention*, ECCV 2024
