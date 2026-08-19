# Verification Phase — Findings & Corrections Needed

**Date:** 2026-08-19
**Session:** Verification of CV results, benchmark CSV, and crop/cutoff distinction

---

## CRITICAL FINDING: Benchmark Speedup Figure Is Wrong

### The Error

The report claims **3.16× speedup** at N=128 for dense_flash vs dense_masked. The actual benchmark CSV (`full_bench_h100.csv`) shows:

| Method | N=128 time (ms) | N=2048 time (ms) |
|--------|-----------------|------------------|
| dense_flash | 0.112 | 0.111 |
| dense_masked | 0.199 | 0.344 |

- **Actual speedup at N=128: 1.78×** (0.199 / 0.112)
- The 3.16× figure was computed by comparing the **N=2048 masked** time (0.344ms) with the **N=128 flash** time (0.109ms) — a **mismatched-N comparison**

### Speedup Across All N (corrected)

| N | Flash (ms) | Masked (ms) | Speedup |
|------|-----------|------------|---------|
| 128 | 0.112 | 0.199 | **1.78×** |
| 256 | 0.113 | 0.200 | **1.77×** |
| 512 | 0.114 | 0.200 | **1.75×** |
| 1024 | 0.115 | 0.201 | **1.75×** |
| 2048 | 0.111 | 0.344 | **3.10×** |
| 4096 | 0.113 | 1.311 | **11.60×** |
| 8192 | 0.351 | 5.113 | **14.57×** |

### Amdahl's Law Correction

With the correct speedup figure:

| Parameter | Report (wrong) | Corrected |
|-----------|---------------|-----------|
| Attention speedup (k) | 3.16× | 1.78× |
| Attention share (s) | 5% | 5% |
| **Theoretical training speedup** | **3.5%** | **2.2%** |
| Measured training speedup | ~3% | ~3% |

The measured ~3% is still consistent with the corrected 2.2% prediction (within noise).

### Files Needing Correction

- `evaluation.tex:85`: "3.16× faster than masked EfficientAttention at N=128" → "1.78×"
- `evaluation.tex:90`: Amdahl formula uses k=3.16 → should use k=1.78
- `evaluation.tex:93`: "3.5%" → "2.2%"
- `argumentation.tex:23`: "3.7--15× over masked dense" — verify this range
- `abstract.tex`: Check if 3.16× is mentioned
- `conclusion.tex`: Check if 3.16× is mentioned
- `scripts/generate_amdahl_figure.py`: Update k=3.16 to k=1.78 in the figure
- `resources/figures/15_amdahl_law.pdf`: Regenerate with corrected figure

---

## CRITICAL FINDING: No Crop Is Used in Our Experiments

### The Error

The appendix (`appendix.tex:29-31`) presents "after 320×320 crop" statistics (median ~53, mean ~87) as if they describe our training data. **They don't.**

- None of the slurm scripts pass `--crop_size`
- Trackastra defaults to `crop_size=None` → **no cropping is applied**
- The "after crop" statistics are misleading

### What Our Experiments Actually Use

Our experiments use **full-frame** data (no crop):

| Statistic | Full Frame (actual) | After 320×320 Crop (not used) |
|-----------|---------------------|-------------------------------|
| Median N | 62 | ~53 |
| Mean N | 165.7 | ~87 |
| Max N | 1618 | ~672 |
| Frac. N > 500 | 9.3% | ~0.4% |

### The Crop vs. Spatial Cutoff Distinction

These are **different things** that the report conflates:

| Parameter | What It Does | Default | Our Experiments |
|-----------|-------------|---------|-----------------|
| **Crop** (`crop_size`) | Randomly crops a sub-region of the field of view during data loading | `None` (no crop) | No crop |
| **Spatial cutoff** (`d_max`) | Boolean attention mask M_ij = 0 if ‖p_i - p_j‖₂ ≤ d_max, -∞ otherwise | 256px | 256px (masked) / removed (dense_flash) |

- The **crop** is data preprocessing (reduces N)
- The **spatial cutoff** is an attention mechanism choice (forces EfficientAttention)
- They are independent parameters
- The CV experiment kept the full frame (no crop) and varied only the spatial cutoff

### The "Should We Remove the Crop?" Question

**We don't use a crop.** Our experiments run on full frames. The question is moot.

The real question is: **should we ADD a crop?** A crop would:
1. Reduce mean N from ~166 to ~87
2. Make attention faster (N² scaling: ~3.6× fewer tokens → ~13× less attention work)
3. But throw away ~48% of cells per sample

Since attention is only ~5% of step time, the training speedup from cropping would be minimal (~2-3% at most). The crop is more interesting as an augmentation strategy than as an efficiency strategy.

---

## CV Results: Verified ✓

### All 12 Folds Completed Successfully

| Fold | Cond | Masked TRA | Flash TRA | ΔTRA | Masked AOGM | Flash AOGM | ΔAOGM |
|------|------|-----------|-----------|------|------------|------------|-------|
| 0 | rpsM | 0.9928 | 0.9933 | +0.0005 | 174.4 | 164.4 | -10.1 |
| 1 | recA | 0.9203 | 0.9184 | -0.0018 | 861.5 | 882.4 | +20.9 |
| 2 | pheA | 0.9987 | 0.9986 | -0.0001 | 37.4 | 38.9 | +1.5 |
| 3 | metA | 0.9905 | 0.9903 | -0.0002 | 213.8 | 221.6 | +7.8 |
| 4 | cib | 0.9998 | 0.9998 | -0.0000 | 6.0 | 7.5 | +1.5 |
| 5 | trpL | 0.9947 | 0.9948 | +0.0001 | 55.9 | 58.0 | +2.1 |
| **Mean** | | **0.9828** | **0.9825** | **-0.0003** | **224.8** | **228.8** | **+4.0** |

### Verification Summary

- ✅ All 12 folds completed (6 masked + 6 dense_flash)
- ✅ All per-fold TRA/AOGM match report values exactly
- ✅ ΔTRA = -0.0003 (verified from raw eval_results.json)
- ✅ ΔAOGM = +4.0 (verified from raw eval_results.json)
- ✅ All folds used window=4, dropout=0.05, seed=42, 100 epochs
- ✅ No crop_size passed → full-frame data
- ✅ Configs differ only in knn_neighbors (-1 masked vs -2 dense_flash)

---

## HOCT vs. Trackastra: Different Sparsification Strategies

### HOCT (Bragantini et al., 2026)

HOCT uses `distance_threshold` (default 200px in code, 300px in CLI) to construct the **candidate graph**:
- Only creates edges between cells within `distance_threshold`
- The GNN never sees distant cell pairs
- This is **pre-attention sparsification** — the graph itself is sparse
- HOCT also uses `n_neighbors` (default 5) to further limit connectivity

### Trackastra (Gallusser et al., 2024)

Trackastra uses a **dense N×N attention matrix** with a Boolean mask:
- Computes full N×N attention scores
- Applies mask M_ij = 0 if ‖p_i - p_j‖₂ ≤ d_max, -∞ otherwise
- The mask forces PyTorch's `check_for_attn_mask` gate → falls back to EfficientAttention
- This is **post-attention sparsification** — the computation is dense, only the result is sparse

### Key Difference

HOCT's pre-attention sparsification is structurally efficient: the GNN only processes edges that exist. Trackastra's post-attention sparsification is structurally inefficient: it computes the full N×N matrix and then masks it, paying O(N²) cost for an O(Nk) problem.

The DenseFlashAttention variant removes the mask entirely, enabling FlashAttention-2. This is the optimal strategy at vanvliet scale (N≈166) where:
- N² is small enough to fit in memory
- The mask costs more (EfficientAttention fallback) than it saves (no cells masked out)

---

## Summary of Corrections Needed

### 1. Benchmark Speedup (CRITICAL)

- **Wrong:** 3.16× at N=128
- **Correct:** 1.78× at N=128
- **Impact:** Amdahl prediction changes from 3.5% to 2.2% training speedup
- **Files:** `evaluation.tex`, `argumentation.tex`, `abstract.tex`, `conclusion.tex`, `scripts/generate_amdahl_figure.py`

### 2. Crop Statistics (MODERATE)

- **Wrong:** Appendix presents "after 320×320 crop" statistics as if they describe our training data
- **Correct:** Our experiments don't use a crop; the full-frame statistics are what matters
- **Impact:** The "N≈87 after crop" figure is wrong; actual mean N is ~166
- **Files:** `appendix.tex`, `evaluation.tex`, `argumentation.tex`

### 3. Amdahl's Law Figure (MINOR)

- The figure (`15_amdahl_law.pdf`) uses k=3.16 as the "FlashAttention-2" curve
- Should be updated to k=1.78 (or include both for comparison)
- **Files:** `scripts/generate_amdahl_figure.py`, `resources/figures/15_amdahl_law.pdf`

### 4. N≈140 vs N≈166 (MINOR)

- The report uses "N≈140" as the vanvliet operating point
- The actual mean N (full-frame, no crop) is **166**
- The median N is **62**
- The benchmark point N=128 is between median and mean
- **Files:** `evaluation.tex`, `argumentation.tex`, `abstract.tex`, `conclusion.tex`

---

## Next Steps

1. **Fix the benchmark speedup figure** (3.16× → 1.78×) throughout the report
2. **Fix the Amdahl prediction** (3.5% → 2.2%) and regenerate the figure
3. **Clarify the crop situation** — remove the "after crop" statistics from the appendix, or clearly state that our experiments don't use a crop
4. **Update the N≈140 figure** to N≈166 (mean full-frame) or N≈62 (median full-frame)
5. **Recompile the report** and verify all corrections
6. **Continue with remaining verification items** (per-epoch times from sacct, edge probing, profiler data)
