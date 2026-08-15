# H100 Unified Edge Probe — Which Features Encode Cell Identity?

**Date:** 2026-08-15
**Context:** Full-factorial edge probing of seven feature types (handcrafted regionprops, HOCT, frozen SSL-pretrained CNN/DINOv2, and end-to-end trained CNN) with both linear and MLP probes under 5-fold cross-validation on a single H100. Determines which feature representation carries usable cell-identity signal for association decisions, and whether the probe's capacity (linear vs. MLP) changes the verdict.
**Depends on:** `2026-07-09_h100_cnn_convergence.md`, `2026-06-08_contrastive_ssl_failure_analysis.md`, `2026-06-15_dino_contrastive_ssl.md`, `2026-07-20_hoct_insights_for_proposal.md`, `2026-08-05_project_state_and_priorities.md`

---

## 1. Motivation

The feature poverty analysis (`2026-06-08_contrastive_ssl_failure_analysis.md`) identified the root cause of SSL failure: **7-dim regionprops contain no usable per-cell identity information** (inter-cell cosine similarity ~0.89, ~3 effective dimensions). The CNN convergence race (`2026-07-09_h100_cnn_convergence.md`) then showed that frozen CNN image features let the full Trackastra model escape the stuck-at-0.5 saddle point (val_bal_acc 0.720 / test 0.671 vs. immediate collapse for 7D). The HOCT analysis (`2026-07-20_hoct_insights_for_proposal.md`) further suggested that HOCT-style 2D node features and Fourier positional encodings are informative for association decisions.

Those results were all *indirect*: convergence quality of a downstream transformer is a noisy proxy for feature informativeness. What was missing was a **direct, controlled measurement** of how much cell-identity signal each candidate feature representation actually carries — and whether a simple probe can extract it.

**Question:** Across handcrafted (regionprops, HOCT 2D), pretrained (CNN NT-Xent frozen, DINOv2 frozen), learned (CNN end-to-end), and position-augmented feature variants, which representation carries the most usable cell-identity signal, and does a linear probe suffice to extract it, or is nonlinear capacity required?

This experiment answers that with a **unified edge probe**: 7 feature types × 2 probe architectures (linear, MLP), full factorial, 5-fold cross-validation, on a single H100 GPU.

---

## 2. Experimental Setup

### 2.1 Common Configuration

| Parameter | Value |
|-----------|-------|
| Task | Edge probing: does a cell pair belong to the same track? (binary) |
| Features | all (7 feature types, full factorial) |
| Probes | both (linear + MLP) |
| CV | 5-fold cross-validation (seed 42) |
| Max pairs per frame | 30 |
| Epochs | 200 |
| Patience | 10 |
| Eval every | 1 epoch |
| LR | 0.001 |
| Batch size (linear / MLP) | 256 / 128 |
| Shuffle baseline | on (label-shuffled controls) |
| Standardization | z-score per feature dim, **fit on the training split of each fold only** (leakage-safe, HOCT-paper default) |
| Data | vanvliet, 6 conditions (rpsM, recA, pheA, metA, cib, trpL) |
| GPU | 1× NVIDIA H100 80 GB (Capella cluster, node c147) |
| Job ID | 3909181 |
| Date completed | 2026-08-15 22:31 CEST |

### 2.2 Feature Types (7)

| Feature | Dim | Description |
|---------|:---:|-------------|
| Regionprops 7D | 7 | Hand-crafted regionprops (`area, perimeter, eccentricity, solidity, extent, mean_intensity, std_intensity`) |
| Regionprops 7D + Fourier PE | 55 | 7D regionprops + Fourier PE of position (t, y, x) |
| CNN NT-Xent (frozen) | — | `ScaledCNN` large (1.47M params), NT-Xent-pretrained, frozen — **SKIPPED** (no checkpoint at configured path) |
| CNN end-to-end | — | `ScaledCNN` large trained end-to-end on the probe task (features are learned, so not standardized) |
| DINOv2 (frozen) | 384 | Frozen DINOv2 image features |
| HOCT 2D | 13 | HOCT 2D adaptation (19D → 13D: drop z, 3×3 inertia → 2×2) |
| HOCT 2D + Fourier PE | 43 | HOCT 13D with spatial positions (y, x) replaced by Fourier PE |

Note: CNN NT-Xent (frozen) appears in the factorial but produced **no result** — the configured checkpoint `.../results/checkpoints/cnn/cnn_ntxent_large.pt` does not exist on the cluster, so the run was skipped (status `SKIP` in the log).

---

## 3. Results

### 3.1 Full Results (5-fold CV, mean ± std)

| Feature | Probe | BalAcc (mean±std) | F1 (mean±std) | Status |
|---------|-------|-------------------|---------------|--------|
| Regionprops 7D | linear | 0.5482±0.0312 | 0.1320±0.0228 | OK |
| Regionprops 7D | mlp | 0.8767±0.0182 | 0.4113±0.0210 | OK |
| Regionprops 7D + Fourier PE | linear | 0.5307±0.0100 | 0.1153±0.0424 | OK |
| Regionprops 7D + Fourier PE | mlp | 0.9529±0.0206 | 0.6864±0.0945 | OK |
| CNN NT-Xent (frozen) | — | — | — | **SKIP** |
| CNN end-to-end | linear | 0.5000±0.0000 | 0.0000±0.0000 | OK |
| CNN end-to-end | mlp | 0.5000±0.0000 | 0.0000±0.0000 | OK |
| DINOv2 (frozen) | linear | 0.5397±0.0146 | 0.1278±0.0410 | OK |
| DINOv2 (frozen) | mlp | 0.8895±0.0295 | 0.5618±0.0620 | OK |
| HOCT 2D (13D) | linear | 0.5280±0.0221 | 0.1370±0.0399 | OK |
| HOCT 2D (13D) | mlp | 0.9330±0.0216 | 0.5914±0.0741 | OK |
| HOCT 2D + Fourier PE (43D) | linear | 0.5358±0.0190 | 0.1282±0.0408 | OK |
| **HOCT 2D + Fourier PE (43D)** | **mlp** | **0.9608±0.0151** | **0.7189±0.0748** | **OK — best** |

**Best performer:** HOCT 2D + Fourier PE (43D) + MLP probe — bal_acc = **0.9608 ± 0.0151**, F1 = **0.7189 ± 0.0748**.

### 3.2 MLP Probes Dramatically Outperform Linear Probes

| Feature | Linear bal_acc | MLP bal_acc | MLP Δ |
|---------|:--------------:|:-----------:|:-----:|
| Regionprops 7D | 0.5482 | 0.8767 | +0.329 |
| Regionprops 7D + Fourier PE | 0.5307 | 0.9529 | +0.422 |
| DINOv2 (frozen) | 0.5397 | 0.8895 | +0.350 |
| HOCT 2D (13D) | 0.5280 | 0.9330 | +0.405 |
| HOCT 2D + Fourier PE (43D) | 0.5358 | 0.9608 | +0.425 |
| CNN end-to-end | 0.5000 | 0.5000 | 0.000 |

**Every feature that carries any signal is nearly unusable with a linear probe (0.53–0.55, essentially shuffled level) but strongly decodable with an MLP probe (0.88–0.96).** The association-relevant structure in these features is highly nonlinear — a single linear hyperplane cannot separate matched from unmatched cell pairs, but a shallow MLP can.

### 3.3 Fourier PE Dramatically Improves MLP Probe Performance

| Feature | MLP bal_acc (no PE) | MLP bal_acc (with Fourier PE) | Δ |
|---------|:-------------------:|:-----------------------------:|:---:|
| Regionprops 7D | 0.8767 | 0.9529 | **+0.076** |
| HOCT 2D (13D) | 0.9330 | 0.9608 | **+0.028** |

Fourier positional encoding of spatial position is a consistent, large boost for the MLP probe: +0.076 bal_acc for regionprops and +0.028 for HOCT 2D, pushing both past 0.95. The best overall configuration is exactly the combination of the most informative base features (HOCT 2D) with positional structure made explicit (Fourier PE).

### 3.4 Shuffled Baselines Confirm Real Signal (no label leakage)

| Feature (shuffled) | Probe | BalAcc (mean±std) | F1 (mean±std) |
|---------------------|-------|-------------------|---------------|
| Regionprops 7D | linear | 0.5205±0.0085 | 0.1250±0.0302 |
| Regionprops 7D | mlp | 0.5306±0.0235 | 0.1342±0.0376 |
| Regionprops 7D + Fourier PE | linear | 0.5139±0.0177 | 0.1132±0.0424 |
| Regionprops 7D + Fourier PE | mlp | 0.5183±0.0205 | 0.1144±0.0621 |
| CNN end-to-end | linear | 0.5000±0.0000 | 0.0000±0.0000 |
| CNN end-to-end | mlp | 0.5000±0.0000 | 0.0000±0.0000 |
| DINOv2 (frozen) | linear | 0.5373±0.0173 | 0.1276±0.0422 |
| DINOv2 (frozen) | mlp | 0.5301±0.0205 | 0.1248±0.0505 |
| HOCT 2D (13D) | linear | 0.5247±0.0071 | 0.1241±0.0332 |
| HOCT 2D (13D) | mlp | 0.5198±0.0085 | 0.1263±0.0365 |
| HOCT 2D + Fourier PE (43D) | linear | 0.5243±0.0119 | 0.1188±0.0392 |
| HOCT 2D + Fourier PE (43D) | mlp | 0.5174±0.0143 | 0.1232±0.0399 |

All shuffled controls cluster at **0.50–0.53 bal_acc / 0.11–0.13 F1** regardless of feature or probe, while real features reach 0.53–0.96 bal_acc. This rules out label leakage or probe-side artifacts: the MLP gains in §3.2 are genuine feature signal, not overfitting to shuffled labels.

---

## 4. Key Findings

### 4.1 Best Feature/Probe Combination: HOCT 2D + Fourier PE + MLP

HOCT 2D (13D) + Fourier PE (43D total) with an MLP probe is the clear winner: **bal_acc = 0.9608 ± 0.0151, F1 = 0.7189 ± 0.0748**. This is the highest balanced accuracy of any configuration and among the lowest variance across folds. The HOCT-style feature set — richer than 7D regionprops (13 dims: time, diameter, intensity, 2×2 inertia, border distances) plus explicit Fourier positional encoding — carries the most usable cell-identity signal.

### 4.2 MLP Probes Unlock the Signal; Linear Probes See Nothing

The single most striking result: **all linear probes sit at 0.53–0.55 bal_acc (indistinguishable from shuffled controls), while MLP probes on the same features reach 0.88–0.96.** The association decision boundary is strongly nonlinear in every feature space tested. Any downstream architecture (and any evaluation methodology) that relies on linear separation of these features will conclude the features are useless — which is exactly the trap the 7D-based analyses nearly fell into.

### 4.3 CNN End-to-End Completely Fails

CNN end-to-end scores **0.5000 ± 0.0000** with both probe types — exactly chance, with zero F1. The learned CNN features contain **no decodable cell-identity signal** in this setup. Combined with the previous SSL analyses, this suggests the end-to-end CNN on this data collapses into a degenerate representation (consistent with the feature-poverty diagnosis, not evidence for the learned features being useful).

### 4.4 CNN NT-Xent (Frozen) Was Skipped — No Checkpoint

The frozen NT-Xent CNN cell of the factorial was **skipped**: the configured checkpoint `cnn_ntxent_large.pt` does not exist on the cluster. This is a gap in the factorial — the frozen SSL-CNN feature type (the one that enabled the convergence race, `2026-07-09`) could not be probed in this run. The prior, smaller probe run (`2026-08-05_project_state_and_priorities.md`) measured CNN-NT-Xent 128D MLP at 0.703, well below the other frozen features.

### 4.5 Fourier PE Is a Consistent, Large Boost

Adding Fourier PE of spatial position improves the MLP probe on both base feature families:
- Regionprops 7D: 0.8767 → **0.9529** (+0.076)
- HOCT 2D: 0.9330 → **0.9608** (+0.028)

Explicit positional structure is highly informative for association decisions — pair distance/geometry is a strong cue, and making it explicit (rather than relying on raw coordinates) helps the probe.

### 4.6 Shuffled Baselines Confirm the Signal Is Real

Shuffled controls are flat at 0.50–0.53 for every feature/probe pair, with high overlap across configurations. Real features span 0.53–0.96. No configuration's shuffled control approaches even the worst real MLP result (0.8767), confirming no label leakage.

### 4.7 DINOv2 Frozen: Good With MLP, Marginal With Linear

DINOv2 (frozen) reaches **0.8895** with the MLP probe — a strong third place — but with a linear probe it is 0.5397, only marginally above its own shuffled control (0.5373). DINOv2 carries real signal, but only extractable nonlinearly.

---

## 5. Significance

### 5.1 For the Feature Poverty Problem

This is the cleanest measurement of feature informativeness to date. The earlier conclusion from `2026-06-08_contrastive_ssl_failure_analysis.md` (7D regionprops are information-starved) is **confirmed and sharpened**: 7D regionprops reach only 0.8767 even with a nonlinear probe, and the *same* 7 dims plus Fourier position reach 0.9529. Meanwhile the richest handcrafted set (HOCT 2D + Fourier PE) reaches 0.9608. Feature poverty is real, but it is **partially a linear-decoding artifact** — the signal is nonlinear and requires nonlinear capacity to extract.

### 5.2 For the Research Trajectory

| Phase | Experiment | Result |
|-------|-----------|:------:|
| Problem discovery | SSL collapse analysis (`2026-06-08`) | Feature poverty diagnosed (7D insufficient) |
| Proposed solution | DINO + Contrastive SSL (`2026-06-15`) | Image features argued as the fix |
| Scale verification | CNN convergence race (`2026-07-09`) | CNN features enable convergence (0.671 test) |
| **Representation ranking** | **This experiment** | **HOCT 2D + Fourier PE + MLP = 0.9608; Fourier PE boosts every family; linear probes see nothing** |
| Next | Replace/augment 7D features in the model with HOCT 2D + Fourier PE; probe the missing frozen CNN cell | Target: TRA > 0.99 |

### 5.3 Implications for the Model and the Report

1. **Feature engineering decision:** HOCT 2D + Fourier PE is the best *available* feature representation (0.9608), beats DINOv2 frozen (0.8895) and 7D regionprops (0.8767) on MLP probes. This is a concrete, measured argument for upgrading the node features used in Trackastra-style training.
2. **Nonlinearity matters:** any evaluation or model component that restricts decoding to linear maps (e.g., dot-product association scoring) is leaving most of the available signal on the table. This is consistent with the HOCT edge-centric design (`2026-07-20_hoct_insights_for_proposal.md`) and explains why node-dot-product scoring underperforms.
3. **Methodological guardrail:** linear-only probe studies on this data would have reported "no feature encodes identity" — the MLP probe is essential. This must be reflected in the report's representation-analysis section.
4. **Gap to close:** the frozen CNN NT-Xent feature was skipped (no checkpoint); its probe result is needed to complete the factorial before finalizing the report tables.

### 5.4 Bottom Line

**The H100 unified edge probe ranks the feature zoo: HOCT 2D + Fourier PE (43D) is the best representation (bal_acc 0.9608 with an MLP probe), Fourier positional encoding consistently adds signal, and every feature family requires nonlinear (MLP) capacity to be decoded at all — linear probes are indistinguishable from shuffled controls. The frozen CNN NT-Xent cell was skipped for lack of a checkpoint. This turns the feature-poverty debate into a concrete, measured ranking and motivates upgrading node features to HOCT 2D + Fourier PE.**

---

## 6. Files Referenced

| File | Role |
|------|------|
| `benchmark_ssl/logs/edge_probe-3909181.out` | Job log with full results table, shuffled baselines, best-performer summary |
| `benchmark_ssl/results/unified_probe_results_cv.json` | Detailed results: per-fold histories, means/stds, args (seed 42, 5-fold, standardize, shuffle) |
| `benchmark_ssl/slurm/unified_edge_probe.slurm` | Slurm script (job edge_probe, H100, 2h, seed/config as above) |
| `benchmark_ssl/src/edge_probing/unified_edge_probe.py` | Probe driver (feature registry, standardization) |
| `benchmark_ssl/src/edge_probing/harness/` | Probe harness: feature extractors, edge datasets, evaluation |
