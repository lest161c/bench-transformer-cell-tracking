# Sparse Backward Pass Optimization — Research Plan

**Date:** 2026-06-23
**Status:** Planning

## Motivation

The backward pass dominates 57% of training step time (15.6ms out of 27.4ms at N=200). PyTorch's standard autograd computes gradients through the full N×N softmax even though only K=16 neighbors are used in KNN sparse attention. The softmax partition function couples all N tokens, making the backward O(N²) despite O(N·K) forward sparsity.

## Goal

Implement a custom sparse backward kernel that computes attention gradients using only the K=16 sparse indices, reducing backward pass time by ~3× and overall step time by ~25%.

## Starting Points

### Key papers (downloaded to papers/sparse_backward/)

1. **LLSA** (Zhou et al., 2025) — Sparse-index forward+backward, 6.09× training speedup
2. **SPARSEK** (Lou et al., 2024) — Fused top-k backward kernel, linear time
3. **SUS Backprop** (Pankov & Harik, 2025) — Validates K=16 is above minimum gradient threshold
4. **FlashAttn Binary Block Masking** (Sharma & Geiping, 2024) — 9× improvement for sparse masks
5. **SSA** (Shen et al., 2025) — Caveat: sparse-only training can hurt gradient flow

### Existing code to modify

- `trackastra/trackastra/model/model_parts.py` — CachedDistAttention and GatherSparseAttention classes
- `benchmark_attn/model_parts.py` — standalone attention implementations for benchmarking

### Key technical insight

Replace full softmax (over all N tokens) with sparse softmax (over K=16 selected keys only). This single change eliminates the need to touch all N tokens in the backward pass, reducing the gradient computation from O(N²) to O(N·K).

## Implementation Steps

### Phase 1: Sparse Softmax (low effort, immediate gain)
- [ ] Implement `sparse_softmax(logits, indices)` — softmax over only K=16 entries per row
- [ ] Verify numerical equivalence with full softmax for the selected K entries
- [ ] Measure speedup on synthetic benchmarks (benchmark_attn/)
- [ ] Validate no convergence degradation on vanvliet data

### Phase 2: Custom Triton Backward Kernel (medium effort, 3-5× gain)
- [ ] Write Triton kernel for sparse attention forward + backward
- [ ] Fuse QK^T, softmax, and V multiplication using only (N, K) indices
- [ ] Backward: compute dQ, dK, dV without materializing N×N matrix
- [ ] Integrate into CachedDistAttention as a new attention variant

### Phase 3: Integration and Validation
- [ ] Full training run with sparse backward on vanvliet (6 conditions)
- [ ] Compare TRA/AOGM vs baseline — ensure no accuracy degradation
- [ ] Measure wall-clock training speedup on H100 (Capella cluster)
- [ ] Profile to confirm backward pass share drops from 57% to target ~25%

## Expected Outcomes

| Metric | Current | Target |
|--------|---------|--------|
| Backward pass time | 15.6ms (57%) | ~5-7ms (~25%) |
| Training step time | 27.4ms | ~20ms |
| Training speedup | 1× | ~1.25-1.35× |

## Risks

- **SSA caveat**: Sparse-only gradient flow may hurt convergence. Mitigation: validate KNN task-specific neighbors provide sufficient gradient signal.
- **Implementation complexity**: Triton kernel development requires CUDA expertise. Fallback: start with PyTorch-level sparse softmax (Phase 1 only).
- **Softmax approximation**: Sparse softmax changes the partition function. Verify no accuracy loss on vanvliet.
