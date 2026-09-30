# SPEC: Integrate Backward Benchmark into benchmark_attn

## Problem

The sparse backward pass benchmark lives in `deprecated/benchmark_backward/`.
It has its own `model_parts.py` that duplicates attention implementations
already present in `benchmark_attn/src/attention_modules.py`. The benchmark
needs to be refactored to use the canonical attention modules and integrated
into `benchmark_attn/scripts/benchmarks/`.

## Changes

### 1. Add `SparseSoftmaxAttention` to `attention_modules.py`

The deprecated `model_parts.py` defines a `SparseSoftmaxAttention` class
that computes softmax only over K neighbors (O(N·K) backward instead of
O(N²)). This class does not exist in `benchmark_attn/src/attention_modules.py`
and must be added.

### 2. Write `benchmark_backward.py` in `scripts/benchmarks/`

Move `deprecated/benchmark_backward/benchmark_sparse_backward.py` to
`benchmark_attn/scripts/benchmarks/benchmark_backward.py`. Update imports
to use `src.attention_modules` instead of the local `model_parts`. The
script measures forward, backward, and total time for each attention
variant across N and K, and saves results to CSV.

### 3. Move CSV results to `benchmark_attn/results/`

Copy `sparse_backward_results.csv` and `sparse_backward_results_8192.csv`
to `benchmark_attn/results/`.

### 4. Register `sparse_softmax` method

Add `sparse_softmax` to `METHOD_REGISTRY` in `src/bench/registry.py` so
it can be used by other benchmark scripts (e.g. `benchmark_sweep.py`).

## Acceptance Criteria

- `SparseSoftmaxAttention` exists in `attention_modules.py` with the same
  forward signature as the other KNN-based attention classes.
- `benchmark_backward.py` runs from `benchmark_attn/` and imports from
  `src.attention_modules` (no local `model_parts.py`).
- CSV results are in `benchmark_attn/results/`.
- `sparse_softmax` is registered in the method registry.
