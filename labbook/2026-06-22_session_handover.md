# Session Handover — 2026-06-22

**State:** All benchmarks created. Key open problems identified. Context needs clearing.

---

## 1. Priors and Assumptions (What We Believe)

| Prior | Status | Evidence |
|-------|--------|----------|
| mask-KNN K=16 is best valid method at N≤512 | **Confirmed** | 2.2× vs dense_masked at N=256 (GPU measured). TRA 0.9972, AOGM 37.7. |
| Attention is NOT the compute bottleneck | **Confirmed** | blockwise_causal_norm 13% of step, FFN 5%, backward 57%. Data loading CPU-bound at N<200. |
| MiniMax is not viable at our scale | **Confirmed** | 10-30× slower than mask-KNN at N=128-512 (analytical, calibrated to labbook). |
| gather-KNN is slower than mask-KNN at N<2000 | **Confirmed** | Per-token gather overhead ~23µs dominates. Labbook 2026-06-08. |
| NSA is not viable | **Confirmed** | 27× slower than dense_flash at N=256 (GPU measured). |
| DINOv3 should be better than DINOv2 for SSL | **UNKNOWN** | DINOv3 weights download blocked (HTTP 403). Code ready in dino_encoder.py. |
| Soft decay + FlashAttention = speedup | **DISPROVEN** | See section 2 below. |

---

## 2. BIGGEST OPEN PROBLEM: FlashAttention Does Not Dispatch With Spatial Constraints

### What we tried

Replacing the hard spatial mask with soft distance decay:

```
score_ij = QK^T - λ·dist_ij/d_max    (no -inf, just soft penalty)
```

Goal: remove the explicit N×N mask so that FlashAttention dispatches.

### What we measured (benchmark_soft_decay_measure.py, d=320, nhead=8, CUDA)

| N | hard mask (cuDNN) | soft decay (SDPA mask) | dense_flash (no mask) |
|---|:---:|:---:|:---:|
| 256 | 0.043ms | 0.039ms (1.1×) | 0.026ms (1.65×) |
| 512 | 0.107ms | 0.089ms (1.2×) | 0.058ms (1.85×) |
| 1024 | 0.326ms | 0.303ms (1.1×) | 0.157ms (2.1×) |

Numerical equivalence: cos_sim = 1.0000 — soft decay output = hard mask output ✓

### The problem

`F.scaled_dot_product_attention(q, k, v, attn_mask=soft_bias)` dispatches cuDNN — NOT FlashAttention — even though the mask values are finite (not -inf). PyTorch's dispatch logic: **ANY attn_mask argument disables FlashAttention**, regardless of mask values.

- Without `attn_mask`: FlashAttention dispatches (dense_flash = 0.026ms)
- With `attn_mask` (even soft values): cuDNN dispatches (soft = 0.039ms, hard = 0.043ms)
- The soft mask is 1.1× faster than hard mask (cuDNN handles soft finite values better than -inf)
- But it's 1.5× SLOWER than pure FlashAttention (0.039ms vs 0.026ms)

### Potential solutions (none verified)

1. **FlexAttention score_mod** (`torch.nn.attention.flex_attention`): apply `score_mod(q,k,dist)` inside FlashAttn kernel. Requires PyTorch 2.5+ with `torch.compile`. Not dispatching in our PT 2.12 environment. Needs Capella test.

2. **Spatial block partition** (benchmark_spatial_blocks.py): Hilbert-sort → split into blocks → FlashAttn per block with overlap. NO mask — FlashAttn dispatches. But analytical only, never GPU-measured. Overhead from sort + cross-block attention unknown.

3. **Pre-bias Q/K**: Multiply K by `exp(-λ·dist/(2·d_max))` so that `Q·K' ≈ Q·K - λ·dist/d_max`. This is multiplicative, not additive — changes the math. Unknown accuracy impact.

### What needs to happen

Run on Capella with latest PyTorch nightly that has FlexAttention working. Test score_mod approach. Measure actual FlashAttention dispatch.

---

## 3. Open TODOs

### High Priority

- [ ] **Fix FlashAttention dispatch with spatial cutoff.** Biggest open problem. See section 2.
- [ ] **Update progress_roadmap.html** with measured soft decay data (replace analytical estimates). Current HTML still shows old data in some sections.
- [ ] **Resolve DINOv3 download.** HTTP 403 from dl.fbaipublicfiles.com/dinov3/. Code ready in dino_encoder.py. Benchmark script at benchmark_dinov3_comparison.py.

### Medium Priority

- [ ] **Run all benchmarks on Capella GPU.** Current measurements: only A500 (d=256, nhead=4) for attention, and local GPU (d=320, nhead=8) for soft decay. Need full sweep on H100.
- [ ] **Verify spatial block partition on GPU.** Analytical model claims 5-64× speedup at large N. Never measured.
- [ ] **TRA/AOGM verification for soft decay.** Phase 2: train Trackastra on vanvliet with soft decay only (no hard mask). Compare TRA vs baseline 0.9972.

### Low Priority

- [ ] **Implement FlexAttention score_mod in Trackastra model.** Requires PyTorch 2.5+ and torch.compile.
- [ ] **Vectorize blockwise_causal_norm.** Analytical model claims 1.06-2.3× speedup. Needs actual implementation in train.py.
- [ ] **Gradient checkpointing for FFN.** Saves 67% activation memory. Needs implementation.

---

## 4. Key Files

| File | Purpose | Status |
|------|---------|--------|
| `progress_roadmap.html` | Main project report, Chart.js interactive | Needs update with measured data |
| `benchmark_attn/benchmark_soft_decay_measure.py` | GPU measurement of soft decay | Working, shows 1.1× not 3.6× |
| `benchmark_attn/benchmark_dinov3_comparison.py` | DINOv2 vs v3 comparison | DINOv3 blocked (HTTP 403) |
| `benchmark_pipeline/` | Non-attention bottleneck benchmarks | Analytical, needs GPU verification |
| `benchmark_attn/benchmark_knn_methods.py` | KNN methods analytical model | Calibrated to labbook, needs GPU |
| `benchmark_attn/benchmark_flash_with_cutoff.py` | FlashAttn dispatch analysis | Fixed dist_cost bug, analytical |
| `trackastra/trackastra/model/dino_encoder.py` | DINOv2/v3 support | Code ready, v3 untestable |
| `labbook/2026-06-22_session_summary.md` | Today's session summary | Complete |

---

## 5. What Was Corrected Today

| Claim | Was | Is |
|-------|-----|----|
| Soft decay is 3.6× faster than hard mask | Claimed (analytical) | **1.1×** (GPU measured) |
| Soft decay dispatches FlashAttention | Claimed | **Does NOT dispatch** — attn_mask blocks it |
| DINOv3 access works | Claimed | **HTTP 403** — weights download blocked |
| FFN is 6-9× more FLOPs than attention | Claimed (wrong formula) | **At N=256: attention 58%, FFN 42%** |
| The 4× FlashAttn claim | Claimed vs CachedDist | **vs mask-KNN: only 1.3× at N=256** |

---

## 6. Working Environment

- **GPU:** NVIDIA with CUDA 13.0 (nvidia-smi available)
- **PyTorch venv:** `benchmark_attn/.venv/` — torch 2.12.0+cu130, CUDA works
- **Run benchmarks:** `benchmark_attn/.venv/bin/python benchmark_attn/<script>.py`
- **Capella H100:** Slurm scripts in `benchmark_combined/`, SSH via `ssh capella`
- **Git branch:** `cached-dist-attn` (17 commits ahead of origin)
