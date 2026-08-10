# SPEC: Feature Normalization Rework and Verification Histograms

## Status: Draft
**Date**: 2026-08-09

---

## 1. Problem

The probing pipeline already implements leakage-safe z-score standardization
(`fit_feature_standardizer` / `apply_feature_standardizer` in
`unified_edge_probe.py`). However, there is no way for the user to visually
verify that normalization was correctly applied before running probes.

Additionally, the cosine similarity histogram code
(`benchmark_attn/analysis/cosine_histogram.py`) has a bug: the
`compute_histograms` function is called but its return values are discarded,
and the same L2 normalization + similarity computation is done again manually
in `main()`, doubling the work.

---

## 2. Goals

1. **Fix the wasted computation bug** in `cosine_histogram.py`.
2. **Rework histogram generation** so the user can run it as a sanity check
   tool, showing:
   - Per-feature distributions before and after z-score standardization
   - Cosine similarity distributions (intra-cell vs inter-cell) after L2
     normalization
3. **Keep it simple** — the histograms are an optional manual analysis tool,
   not an automated verification pipeline.

---

## 3. Bug Fix: `cosine_histogram.py`

### Bug Description

In `main()`, line 270:
```python
intra_rp, inter_rp, _, _ = compute_histograms(f1, f2, l1, l2, nbins=args.nbins)
```

The return values `intra_rp` and `inter_rp` are never used. The actual
similarity computation is done manually at lines 273–284, which duplicates
the L2 normalization and similarity matrix computation already done inside
`compute_histograms`.

### Fix

Remove the redundant `compute_histograms` call and the duplicate manual
computation. Instead, have `compute_histograms` return the raw similarity
values (intra and inter) in addition to the histogram counts, and use those
directly.

---

## 4. Histogram Rework

### 4.1 What the Histograms Show

**Panel A: Raw Feature Distributions**
- One subplot per feature dimension (up to 12; for DINO 384D, show first 12
  dimensions or PCA-reduce to 12)
- Shows the raw, un-normalized feature values

**Panel B: Standardized Feature Distributions**
- Same layout as Panel A
- After z-score standardization (mean=0, std=1)
- Overlay a standard normal N(0,1) PDF for reference

**Panel C: Per-Feature Statistics After Standardization**
- Bar chart: per-feature mean (should be ≈ 0 on train split)
- Bar chart: per-feature std (should be ≈ 1 on train split)

**Panel D: Cosine Similarity Distributions**
- Histogram of intra-cell cosine similarities (same cell, different view)
- Histogram of inter-cell cosine similarities (different cells)
- Vertical lines at means; annotate separation gap

### 4.2 How to Run

```bash
# Generate verification histograms for regionprops features
python benchmark_attn/analysis/cosine_histogram.py --verify

# Generate verification histograms for DINO features
python benchmark_attn/analysis/cosine_histogram.py --verify --dino
```

The `--verify` flag generates the 4-panel verification figure. Without
`--verify`, the script generates the original cosine similarity histograms
(as before, but with the bug fixed).

### 4.3 Output Files

```
benchmark_attn/cosine_similarity_histogram.png     (existing, bug fixed)
benchmark_attn/normalization_verification.png      (new, with --verify)
benchmark_attn/cosine_similarity_results.csv       (existing)
```

---

## 5. Acceptance Criteria

- [ ] `compute_histograms` bug fixed (no wasted computation)
- [ ] `--verify` flag generates 4-panel normalization verification figure
- [ ] Verification figure shows raw and standardized feature distributions
- [ ] Verification figure shows per-feature mean/std after standardization
- [ ] Verification figure shows cosine similarity distributions
- [ ] All new functions have docstrings (AGENTS.md hard rule)
- [ ] Seeds are exposed as `--seed` parameters (AGENTS.md reproducibility rule)

---

## 6. Open Questions

1. **PCA for high-dimensional features**: Should DINO (384D) features be
   PCA-reduced to 12 components before generating per-feature histograms?
   - **Recommendation**: Yes, PCA to 12 components, preserving 95%+ variance.

2. **Which CV fold to visualize**: Should histograms show fold 0, or
   aggregate across all folds?
   - **Recommendation**: Fold 0 (deterministic with seed).

3. **Feature naming**: Should the histograms use feature names from
   `FEATURE_NAMES` in `rich_features.py`, or generic names like "dim_0"?
   - **Recommendation**: Use `FEATURE_NAMES` when available (regionprops),
     generic names for DINO/CNN.
