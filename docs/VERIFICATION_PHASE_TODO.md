# Verification Phase TODO

**Date:** 2026-08-18
**Purpose:** Comprehensive verification of all claims, measurements, and arguments before final submission.

---

## 1. Claims That Require Verification

### 1.1 Spatial Cutoff Ablation (NEW — needs full verification)

**Claim:** Removing the spatial cutoff (DenseFlashAttention) produces identical tracking accuracy to the masked baseline (ΔTRA = -0.0003, ΔAOGM = +4.0).

**Evidence needed:**
- [ ] Verify all 6 CV folds completed successfully (check exit codes)
- [ ] Verify eval_results.json files are correct (check n_sequences, check for errors)
- [ ] Verify the CV protocol was properly executed (train on 5, test on 6th)
- [ ] Verify that the "masked" config actually uses the spatial cutoff mask
- [ ] Verify that the "dense_flash" config actually removes the mask
- [ ] Cross-check: does the CV use the same code branch for both configs?
- [ ] Cross-check: are the hyperparameters truly identical between configs?

**Risk:** If the CV protocol has a bug (e.g., data leakage, wrong config), the ΔTRA ≈ 0 claim is invalid.

### 1.2 Amdahl's Law Analysis (NEW — needs verification)

**Claim:** FlashAttention-2 is 3.16× faster than masked EfficientAttention at N=128, but this yields only 3.5% training speedup because attention is ~5% of step time.

**Evidence needed:**
- [ ] Verify the 3.16× figure from the benchmark CSV (dense_flash vs dense_masked at N=128)
- [ ] Verify the 5% attention share figure (from profiler, labbook 2026-05-29)
- [ ] Verify the Amdahl formula is correctly applied
- [ ] Verify the measured 3% training speedup from CV per-epoch times
- [ ] Cross-check: is the 3% measured speedup consistent with the 3.5% theoretical prediction?

**Risk:** If the 3.16× figure is wrong (e.g., from a different N), the Amdahl analysis is invalid.

### 1.3 Per-Epoch Training Time (NEW — needs verification)

**Claim:** Vanilla (3.6 min/epoch) ≈ optimized (3.7 min/epoch) ≈ CV masked (3.6 min/epoch) ≈ CV dense_flash (3.5 min/epoch).

**Evidence needed:**
- [ ] Verify vanilla_train per-epoch time from sacct (12h timeout, 202 epochs)
- [ ] Verify optimal_train per-epoch time from sacct (12h timeout, 197 epochs)
- [ ] Verify CV masked per-epoch time from sacct (6 folds, 100 epochs each)
- [ ] Verify CV dense_flash per-epoch time from sacct (6 folds, 100 epochs each)
- [ ] Cross-check: are the per-epoch times consistent across folds?

**Risk:** If the per-epoch times are confounded (e.g., different batch sizes, different data loading), the speedup comparison is invalid.

### 1.4 Other Optimizations (NEW — needs verification)

**Claim:** Other optimizations (fast-regionprops, vectorized blockwise_norm, CachedDistAttention) provide negligible training speedup at N≈140.

**Evidence needed:**
- [ ] Verify fast-regionprops runs in the data loader (CPU, parallel with GPU)
- [ ] Verify vectorized blockwise_norm's 1.96× isolated speedup (from blockwise_norm_results.csv)
- [ ] Verify CachedDistAttention's 1.5× isolated speedup (from labbook)
- [ ] Cross-check: is the combined theoretical speedup (~9.5%) consistent with the measured speedup (~3%)?

**Risk:** If the isolated speedups are wrong, the theoretical analysis is invalid.

---

## 2. Existing Claims That Need Re-Verification

### 2.1 Attention Benchmark (corrected H100)

**Claim:** Dense FlashAttention is 3.16× faster than masked EfficientAttention at N=128.

**Evidence needed:**
- [ ] Re-verify the benchmark CSV (full_bench_h100.csv) is the corrected version
- [ ] Re-verify the --with-knn flag was used
- [ ] Re-verify the dense_flash and dense_masked methods are present
- [ ] Re-verify the N=128 row has the correct time_ms values

### 2.2 KNN Equivalence (18 configurations)

**Claim:** Gather-KNN and mask-KNN are numerically equivalent in both forward and backward passes.

**Evidence needed:**
- [ ] Verify the equivalence test was run with the correct configurations
- [ ] Verify the forward and backward passes were both tested
- [ ] Verify the float16 SDPA kernel rounding tolerance was correctly applied

### 2.3 Edge Probing Results

**Claim:** DINOv2 achieves 0.896 balanced accuracy, 7D regionprops 0.860, HOCT 19D 0.880.

**Evidence needed:**
- [ ] Verify the edge probing results are from 5-fold cross-validation
- [ ] Verify the feature types are correctly labeled
- [ ] Verify the standardization protocol was correctly applied

### 2.4 DeepCell Cross-Dataset Evaluation

**Claim:** Baseline generalizes best (TRA 0.990, cHOTA 0.956, AOGM 592).

**Evidence needed:**
- [ ] Verify the DeepCell evaluation was run with the correct checkpoint
- [ ] Verify the traccuracy pipeline was used correctly
- [ ] Verify the TRA, cHOTA, and AOGM values are correct

### 2.5 CNN Feature Injection (8 variants)

**Claim:** All CNN-augmented models were outperformed by the 7D regionprops-only baseline.

**Evidence needed:**
- [ ] Verify the CNN feature injection results are from the H100
- [ ] Verify the 8 variants are correctly labeled
- [ ] Verify the validation loss values are correct

### 2.6 SSL Pretraining Results

**Claim:** Identity BCE reached validation loss 0.404 vs 0.416 for from-scratch baseline; NT-Xent collapsed to cosine similarity 0.89.

**Evidence needed:**
- [ ] Verify the SSL pretraining results are from the correct runs
- [ ] Verify the identity BCE and NT-Xent objectives were correctly implemented
- [ ] Verify the cosine similarity collapse figure (0.89) is correct

---

## 3. Figures That Need Regeneration

### 3.1 New Figures (from CV results)

- [ ] `14_cv_comparison.pdf`: TRA and AOGM comparison across 6 folds
- [ ] `15_amdahl_law.pdf`: Training speedup vs attention share of step time
- [ ] `16_per_epoch_time.pdf`: Per-epoch training times across configurations
- [ ] `17_attention_share.pdf`: Attention share of step time vs N

### 3.2 Existing Figures (need re-verification)

- [ ] `01_speed_vs_n.pdf`: Verify it includes all measured configurations
- [ ] `05_convergence.pdf`: Verify it shows all 5 KNN configurations
- [ ] `06_tra.pdf`, `07_aogm.pdf`: Verify tracking accuracy figures
- [ ] `11_profiler_breakdown.pdf`: Verify profiler breakdown figure
- [ ] `13_ssl_negative.pdf`: Verify SSL negative results figure
- [ ] `edge_probe_bal_acc.pdf`: Verify edge probing balanced accuracy figure

---

## 4. Report Sections That Need Final Review

### 4.1 Abstract

- [ ] Verify all numbers in the abstract are correct
- [ ] Verify the abstract accurately reflects the report's findings

### 4.2 Introduction

- [ ] Verify the contributions list is accurate and complete
- [ ] Verify the introduction accurately motivates the work

### 4.3 Evaluation

- [ ] Verify all tables have correct values
- [ ] Verify all figures are referenced correctly
- [ ] Verify the CV ablation section is accurate
- [ ] Verify the Amdahl's law section is accurate
- [ ] Verify the per-epoch training time section is accurate

### 4.4 Conclusion

- [ ] Verify the conclusion accurately summarizes the findings
- [ ] Verify the future work section is accurate and complete
- [ ] Verify the limitations section is accurate and complete

### 4.5 Argumentation

- [ ] Verify the argumentation section is consistent with the new findings
- [ ] Verify the "Dense FlashAttention" rejection is still valid (or needs updating)
- [ ] Verify the "Decision" paragraph is consistent with the CV results

---

## 5. Open Questions

### 5.1 Does the CV protocol have data leakage?

The CV protocol trains on 5 conditions and tests on the 6th. But the training data includes all sequences from the 5 conditions, and the test data includes all sequences from the 6th condition. Is there any overlap between training and test sequences?

### 5.2 Is the 3% measured speedup statistically significant?

The CV gives 6 paired per-epoch measurements. The 3% speedup is consistent across folds, but is it statistically significant? A paired t-test would answer this.

### 5.3 Does the spatial cutoff affect generalization?

The CV results show that removing the spatial cutoff doesn't degrade tracking accuracy on the held-out condition. But does it affect generalization to completely different datasets (e.g., DeepCell)?

### 5.4 Is the Amdahl's law analysis complete?

The Amdahl's law analysis considers only the attention kernel speedup. But the CV also includes other optimizations (fast-regionprops, vectorized blockwise_norm, CachedDistAttention). Are these optimizations' speedups correctly accounted for in the Amdahl analysis?

### 5.5 Does the report accurately reflect the proposal's goals?

The proposal asks:
1. Can we make transformer-based cell tracking more efficient?
2. Can we pretrain the association head from unlabeled data?

The report answers:
1. Yes, but the speedup is negligible at vanvliet scale (3%).
2. No, SSL pretraining fails to improve over from-scratch baseline.

Are these answers accurately reflected in the report?

---

## 6. Verification Phase Protocol

### 6.1 Data Verification

1. Download all CV results from HPC
2. Verify all eval_results.json files are correct
3. Verify all summary.json files are correct
4. Cross-check with sacct data

### 6.2 Figure Verification

1. Generate all new figures (14-17)
2. Verify all existing figures (01-13)
3. Compile LaTeX report and verify all figures render correctly

### 6.3 Report Verification

1. Read the full report end-to-end
2. Verify all numbers, tables, and figures
3. Verify all claims are supported by evidence
4. Verify all limitations are accurately stated

### 6.4 Proposal Alignment

1. Read the proposal end-to-end
2. Verify the report addresses all proposal questions
3. Verify the report's conclusions align with the proposal's goals
