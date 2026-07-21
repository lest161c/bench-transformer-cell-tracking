# Reproduction Guide: Trackastra Sparse Attention + CNN Feature Injection

Comprehensive guide to reproduce all experimental findings in this project.
Covers sparse attention benchmarks, CNN feature injection, edge probing, and pipeline optimization — across A500 (micro) and H100 (production) scales.

---

## 1. Prerequisites

### 1.1 Environment Setup

**Python version:** 3.12+

**Conda environment (recommended):**
```bash
conda create -n trackastra python=3.12
conda activate trackastra
```

**Core dependencies:**
```bash
pip install torch>=2.5.0 torchvision>=0.20.0
pip install seaborn matplotlib pandas numpy scipy scikit-learn tifffile
pip install wandb pyyaml einops lightning
```

**CUDA:** CUDA 12.4+ (tested on A5000 24 GB and H100 80 GB)

### 1.2 Hardware Requirements

| Experiment | Required GPU | Est. Runtime | Est. Cost |
|---|---|---|---|
| Standalone sparse benchmarks | Any NVIDIA GPU (4 GB+) | 5–15 min | Free (local) |
| KNN ablation (L=4) | A5000 24 GB (or similar) | 4–6 hours | ~$1 |
| CNN convergence race | **H100 80 GB** | ~32 min | ~$5 |
| CNN extended experiments | **H100 80 GB** | ~4 hours | ~$35 |
| Edge probing | A5000 24 GB | 30 min–2 hours | ~$1 |
| Pipeline benchmarks | Any NVIDIA GPU (4 GB+) | 10–30 min | Free (local) |

> **NOTE:** CNN experiments require H100 due to 250M-parameter transformer at full scale (d_model=320, 6L+6L). A5000 (24 GB) will OOM.

### 1.3 Dataset Requirements

Primary dataset: **Vanvliet** (brightfield microscopy, 6 conditions, C. elegans embryos)

**Obtaining the data:**
```bash
# The dataset path is configured via DATA_ROOT in configs/*.yaml
# Currently stored at (example):
export DATA_ROOT=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/data/vanvliet
```

**Cell Tracking Challenge (CTC) datasets** (for downstream evaluation):
- Download from: http://www.celltrackingchallenge.net/
- Required: `Fluo-N2DH-GOWT1`, `Fluo-N2DL-HeLa`, `PhC-C2DL-PSC`, `BF-C2DL-HSC`

> **Note:** Dataset paths are hard-coded in config files. Update `DATA_ROOT` and `data_root` fields before running.

---

## 2. Repository Setup

```bash
# Clone main working repo (this repo)
git clone https://github.com/lest161c/bench-transformer-cell-tracking.git
cd bench-transformer-cell-tracking

# Clone Trackastra fork with cached-dist-attn branch
git clone --branch cached-dist-attn https://github.com/lest161c/trackastra.git
pip install -e trackastra/

# (Optional) Set up W&B logging
wandb login
```

**Verify git commits:**
```bash
# Main repo (this project):
#   HEAD SHA is visible via: git log -1 --format=%H
#   Key commits:
#     515886b  — lambda(t) scheduled mixing
#     f4ec71c  — CONCAT injection + unified edge probing
#     dd9a111  — 5-fold CV edge probe
#     07ed360  — HOCT 19D feature extraction
#     e0625b8  — Edge probe SLURM script fix
```

---

## 3. Experiment 1: Sparse Attention Benchmarks

### 3.1 Standalone Benchmark

**Purpose:** Measure forward pass time and memory for dense FlashAttention vs. sparse KNN attention.

**Command:**
```bash
# Run the combined speed+memory benchmark
python bench-transformer-cell-tracking/benchmark_combined/benchmark_speed_mem.py

# Expected output file:
#   bench-transformer-cell-tracking/benchmark_combined/results/speed_mem.csv
```

**Config:** `bench-transformer-cell-tracking/benchmark_attn/config.yaml`

**Expected results (`speed_mem.csv`):**

| K | N | time_ms | mem_mb |
|---|---|---|---|
| 0 | 128 | 8.3 | 80.3 |
| 0 | 256 | 23.1 | 290.2 |
| 0 | 512 | 80.9 | 1104.1 |
| 4 | 128 | 16.9 | 86.3 |
| 4 | 256 | 39.4 | 302.1 |
| 4 | 512 | 106.0 | 1123.0 |
| 16 | 128 | 19.3 | 110.4 |
| 16 | 256 | 44.8 | 350.2 |
| 16 | 512 | 116.8 | 1219.4 |

**To reproduce the speed plot:**
```bash
python results/generate_plots.py
# → results/figures/sparse_attention_speed.pdf
# → results/figures/sparse_attention_memory.pdf
```

### 3.2 Additional Standalone Benchmarks

```bash
# Full ablation (dense vs sparse, L=1 and L=4, random and SSL init)
python bench-transformer-cell-tracking/benchmark_combined/benchmark_combined.py

# Downstream matched (K=0,4,8,16,32 at 100%/50%/10% data)
python bench-transformer-cell-tracking/benchmark_combined/downstream_matched.py

# Mask vs gather benchmark
# See: benchmark_mask_vs_gather.csv, benchmark_small_n_results.csv
```

### 3.3 Backward Pass Benchmark

**Purpose:** Measure forward + backward pass time for each attention method.

**Command:**
```bash
python benchmark_backward/profile_backward.py  # if available
# or the raw data is at:
#   benchmark_backward/sparse_backward_results.csv
```

**Reproduce figure:**
```bash
python results/generate_plots.py
# → results/figures/backward_pass.pdf
```

**Key finding:** `mask_knn` is the fastest sparse method. `gather_knn` is ~3–6× slower and uses ~6–20× more memory.

### 3.4 KNN Ablation (TRA Parity)

**Purpose:** Compare dense vs sparse convergence at scale (L=4, N=512).

**SLURM scripts:** `bench-transformer-cell-tracking/benchmark_combined/`
- `run_dist_ablation.slurm` — distributed ablation (all K values)
- `run_diag2.slurm` — diagnostic runs

**Command:**
```bash
# Run full ablation (needs A5000 or similar)
sbatch bench-transformer-cell-tracking/benchmark_combined/run_dist_ablation.slurm

# Or run individual combinations:
python bench-transformer-cell-tracking/benchmark_combined/benchmark_combined.py \
    --mode converge --attn dense --L 4 --N 512 --init rand
python bench-transformer-cell-tracking/benchmark_combined/benchmark_combined.py \
    --mode converge --attn sparseK4 --L 4 --N 512 --init rand
```

**Data:** `bench-transformer-cell-tracking/benchmark_combined/results/ablation_full.csv`

**Reproduce figure:**
```bash
python results/generate_plots.py
# → results/figures/knn_tra_parity.pdf
```

**Key finding:** Sparse attention (K=4, K=16) converges at the same rate as dense attention. SSL pretraining converges faster initially but dense and sparse reach the same floor. At K=4 (L=4), the model overfits after Epoch 8 due to insufficient neighbors.

**Expected runtime:** 4–6 hours on A5000 for full ablation.

---

## 4. Experiment 2: CNN Feature Injection

> **IMPORTANT:** All CNN experiments require H100 80 GB GPU. They will **OOM** on A5000.

**Common config parameters:**
- Model: Mini Trackastra, full scale (d_model=320, nhead=8, 6 encoder + 6 decoder)
- Data: ALL vanvliet frames (6 conditions), 301 consecutive frame pairs
- Train/Val/Test: 241/30/30 (178/20/20 after filtering)
- pos_weight: 100.0
- Optimizer: AdamW, Cosine LR (lr=1e-4)
- Max epochs: 500, Patience: 50

### 4.1 Vanilla CNN Baseline

**Config:** `configs/vanvliet_baseline_cnn.yaml`

**Train command:**
```bash
python trackastra/trackastra/model/train.py \
    --config configs/vanvliet_baseline_cnn.yaml \
    --wandb-project trackastra-cnn-convergence
```

**Expected result:**
- Best val_loss: **0.035** (epoch 51)
- CNN frozen, 7D regionprops + 64D CNN features → 71D input

**Runtime:** ~30 min on H100 (job 3719326).

### 4.2 CNN + Feature Dropout

**Config:** `configs/vanvliet_cnn_dropout.yaml`

**Train command:**
```bash
python trackastra/trackastra/model/train.py \
    --config configs/vanvliet_cnn_dropout.yaml \
    --wandb-project trackastra-cnn-convergence
```

**Key difference:** `cnn_feat_dropout=0.2` — randomly zeros CNN patches and first 4 regionprops dims during training.

**Expected result:**
- Best val_loss: **0.016** (epoch 53)
- 2.2× better than vanilla CNN (0.035).

**Runtime:** ~30 min on H100.

### 4.3 CNN + Trainable (Fine-Tuning)

**Config:** `configs/vanvliet_cnn_trainable.yaml`

**Train command:**
```bash
python trackastra/trackastra/model/train.py \
    --config configs/vanvliet_cnn_trainable.yaml \
    --wandb-project trackastra-cnn-convergence
```

**Key difference:** `cnn_trainable=True` — CNN fine-tuned via BCE gradient (1.47M params adapt).

**Expected result:**
- Best val_loss: **0.036** (epoch 8)
- Then monotonic degradation to 0.083 (epoch 91).
- **Catastrophic forgetting** of SSL-pretrained visual features.

**Runtime:** ~30 min on H100.

### 4.4 CNN + Both (Trainable + Dropout)

**Config:** `configs/vanvliet_cnn_both.yaml`

**Train command:**
```bash
python trackastra/trackastra/model/train.py \
    --config configs/vanvliet_cnn_both.yaml \
    --wandb-project trackastra-cnn-convergence
```

**Expected result:**
- Best val_loss: **0.024** (epoch 48)
- Middle ground: dropout helps but trainable still degrades.

### 4.5 CNN CONCAT + Dropout (Best Overall)

**Config:** `configs/vanvliet_cnn_concat.yaml`

**Train command:**
```bash
python trackastra/trackastra/model/train.py \
    --config configs/vanvliet_cnn_concat.yaml \
    --wandb-project trackastra-cnn-convergence
```

**Expected result:**
- Best val_loss: **0.01131** (best of all CNN variants)
- Uses CONCAT mode (features concatenated before projection, not added as residual).

### 4.6 λ(t) Decay (Negative Result)

**Configs:**
- `configs/vanvliet_lambda_decay.yaml` — λ only
- `configs/vanvliet_lambda_decay_dropout.yaml` — λ + dropout
- `configs/vanvliet_lambda_half.yaml` — λ halved

**Schedule:** λ(t) = λ₀ · exp(−t/τ) where λ₀=1.0, τ=20 epochs.

**Why it failed:** The continuous λ decay mechanism had no direct literature precedent. Equivalent approaches in HOCT use fundamentally different mechanisms (auxiliary loss, stochastic dropout). Our additive residual λ(t) schedule was unable to escape the hybrid-representation saddle point.

**Expected results:**
- λ only: val_loss = **0.077** (2.2× worse than vanilla CNN baseline)
- λ + dropout: val_loss = **0.040** (1.14× worse than vanilla)
- Both worse than doing nothing (0.035).

### 4.7 Regionprops-Only Baseline (No CNN)

**Config:** `configs/vanvliet_baseline.yaml`

**Train command:**
```bash
python trackastra/trackastra/model/train.py \
    --config configs/vanvliet_baseline.yaml \
    --wandb-project trackastra-cnn-convergence
```

**Expected result:**
- Best val_loss: **0.003** (epoch 152)
- **11.7× better than best CNN variant.** 7D regionprops alone is sufficient — CNN features are actively harmful.

### 4.8 cnn_proj Weight Forensics

**Finding:** Across ALL four CNN variants, `cnn_proj.weight` norm is invariant at **10.81** (vs `proj.weight` norm = 11.74). Per-dim norm ratio cnn/proj = **1.0034**. The model cannot dynamically adjust CNN vs regionprops reliance.

**To reproduce:**
```bash
python scripts/checkpoint_forensics.py \
    --checkpoint checkpoints/h100_conv_race/model_C_best.pt
```

---

## 5. Experiment 3: Edge Probing

### 5.1 Feature Comparison

**Script:** `benchmark_ssl/probe/unified_edge_probe.py`

**Commands:**
```bash
# Test all features with both linear and MLP probes
python benchmark_ssl/probe/unified_edge_probe.py \
    --features all --probe both --epochs 200

# Individual feature types:
python benchmark_ssl/probe/unified_edge_probe.py --features rp    --probe linear --epochs 50
python benchmark_ssl/probe/unified_edge_probe.py --features dino  --probe both   --epochs 200
python benchmark_ssl/probe/unified_edge_probe.py --features hoct19 --probe both  --epochs 200
python benchmark_ssl/probe/unified_edge_probe.py --features cnn_frozen --probe both --epochs 200
```

**SLURM script:** `benchmark_ssl/cnn_encoder/run_full_probe.slurm` (if available)

**Expected results (5-fold CV):**

| Feature + Probe | Balanced Acc | F1 Score |
|---|---|---|
| DINOv2 MLP | 0.896 ± 0.044 | 0.571 ± 0.121 |
| DINOv2 linear | 0.541 ± 0.018 | 0.138 ± 0.037 |
| HOCT 19D MLP | 0.873 ± 0.021 | 0.394 ± 0.052 |
| HOCT 19D linear | 0.575 ± 0.050 | 0.145 ± 0.052 |
| Regionprops MLP | 0.860 ± 0.014 | 0.396 ± 0.056 |
| Regionprops linear | 0.557 ± 0.030 | 0.131 ± 0.029 |
| CNN NT-Xent MLP | 0.703 ± 0.056 | 0.289 ± 0.127 |
| CNN NT-Xent linear | 0.543 ± 0.015 | 0.129 ± 0.035 |
| CNN e2e MLP | 0.500 ± 0.000 | 0.000 ± 0.000 |
| CNN e2e linear | 0.500 ± 0.000 | 0.000 ± 0.000 |

**Key finding:** DINOv2 MLP (0.896) is the clear winner. CNN end-to-end collapses to random (0.500). Regionprops MLP (0.860) performs nearly as well as DINOv2 at zero cost.

### 5.2 5-Fold Cross-Validation

```bash
python benchmark_ssl/probe/unified_edge_probe.py \
    --features all --probe both --cv-folds 5 --epochs 200
```

The `--cv-folds 5` flag splits the 30/30 val/test pairs into 5 folds. Results are reported as mean ± std.

### 5.3 Reproduce Probe Figures

```bash
python results/generate_plots.py
# → results/figures/edge_probe_bal_acc.pdf
# → results/figures/edge_probe_f1.pdf
```

> **Note:** These plots use hard-coded data from the progress dashboard (progress_roadmap.html §Charts 19–20) because the unified probe JSON output was not saved. Re-running the probe script with `--output results.json` will produce the raw data.

---

## 6. Experiment 4: Pipeline Bottleneck Analysis

### 6.1 FFN Checkpointing

**Data:** `benchmark_pipeline/ffn_checkpoint_results.csv`

**Command:**
```bash
python benchmark_pipeline/benchmark_ffn_checkpoint.py  # if available
```

**Key finding:** Checkpointing every 3 layers reduces memory 67% with only 27% recompute overhead.

### 6.2 Normalization Vectorization

**Data:** `benchmark_pipeline/blockwise_norm_results.csv`

**Command:**
```bash
python benchmark_pipeline/benchmark_blockwise_norm.py  # if available
```

**Key finding:** Vectorized norm is 1.13–10.54× faster than serial loop depending on batch size.

### 6.3 Spatial Blocks

**Data:** `benchmark_pipeline/spatial_blocks_results.csv`

**Key finding:** Spatial blocks with cached distances achieve up to **5425× speedup** vs naive baseline at N=8192.

### 6.4 Regionprops CPU/GPU Split

**Data:** `benchmark_pipeline/regionprops_results.csv`

**Key finding:** CPU dominates at N≤200 (80%+ of inference time). GPU takes over at N≈600.

### 6.5 Reproduce Pipeline Figures

```bash
python results/generate_plots.py
# → results/figures/pipeline_breakdown.pdf
```

---

## 7. Figure Generation (Complete)

```bash
cd /home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj
mkdir -p results/figures
pip install seaborn matplotlib pandas numpy scipy  # if not installed
python results/generate_plots.py
```

### Expected output files in `results/figures/`:

| File | Content | Size (approx) |
|---|---|---|
| `cnn_convergence.pdf` | Baseline vs Vanilla CNN val_loss trajectories (log y) | 24 KB |
| `cnn_variant_ranking.pdf` | 8 CNN variants sorted by val_loss (horizontal bar) | 19 KB |
| `edge_probe_bal_acc.pdf` | Edge prediction balanced accuracy (horizontal bar, grouped) | 20 KB |
| `edge_probe_f1.pdf` | Edge prediction F1 scores (horizontal bar, grouped) | 19 KB |
| `sparse_attention_speed.pdf` | N vs time for different K values | 18 KB |
| `sparse_attention_memory.pdf` | N vs memory for different K values | 19 KB |
| `pipeline_breakdown.pdf` | Training step breakdown + regionprops CPU/GPU split | 22 KB |
| `backward_pass.pdf` | Forward/backward time for attention methods | 16 KB |
| `knn_tra_parity.pdf` | Dense vs sparse convergence curves (log y) | 21 KB |

---

## 8. Cross-Dataset Evaluation

### 8.1 Held-Out Dataset: HOCT19

The HOCT 19D features were added in commit `07ed360`. To evaluate on HOCT19:

```bash
python benchmark_ssl/probe/unified_edge_probe.py \
    --features hoct19 --probe both --epochs 200
```

**Data source:** HOCT 19D features are derived from:
- Centroid (2D)
- Equivalent diameter (1D)
- Intensity (4D: mean, std, skew, kurtosis)
- Inertia tensor (4D: 2 eigenvalues + 2 orientations)
- Border distance (1D)
- Plus temporal features extracted from 2-frame windows

### 8.2 CTC Datasets

**Configs for downstream evaluation:**
- `configs/base-config.yaml` — base config with CTC dataset paths
- `configs/dummy.yaml` — minimal test config

**Commands:**
```bash
# Evaluate a trained model on CTC metrics
python bench-transformer-cell-tracking/benchmark_combined/full_eval.py \
    --checkpoint checkpoints/<model>.pt \
    --dataset Fluo-N2DH-GOWT1
```

**SLURM scripts:**
- `bench-transformer-cell-tracking/benchmark_combined/run_eval_all.slurm` — evaluate all checkpoints
- `bench-transformer-cell-tracking/benchmark_combined/eval_one_model.slurm` — single model evaluation

---

## 9. Complete Results Tables

### 9.1 CNN Convergence Race (H100)

| Config | Best val_loss | Best epoch | vs Baseline (0.003) | vs Vanilla CNN (0.035) |
|---|---|---|---|---|
| CNN CONCAT + dropout | **0.01131** | — | 3.8× worse | **3.1× better** |
| CNN + dropout | **0.016** | 53 | 5.3× worse | **2.2× better** |
| CNN + dropout p=0.5 | 0.023 | — | 7.7× worse | 1.5× better |
| CNN + both | 0.024 | 48 | 8× worse | 1.5× better |
| Vanilla CNN | 0.035 | 51 | 11.7× worse | — |
| CNN + trainable | 0.036 | 8 | 12× worse | tied |
| λ + dropout | 0.040 | — | 13.3× worse | 1.14× worse |
| λ only | 0.077 | — | 25.7× worse | 2.2× worse |
| **Baseline (no CNN)** | **0.003** | **152** | — | **11.7× better** |

### 9.2 KNN Ablation (Final val_loss, L=1, baseline model)

| Model | Best val_loss | KNN neighbors |
|---|---|---|
| sparse_k64_clean | **0.0019** | 64 |
| sparse_k16_clean | **0.0021** | 16 |
| sparse_k4_clean | 0.0024 | 4 |
| baseline_clean | 0.0026 | 0 (dense) |
| sparse_k32_clean | 0.0032 | 32 |

*(Source: `results_final/results_table.csv`)*

### 9.3 Edge Probing Summary

| Feature | Linear bal_acc | MLP bal_acc | Linear F1 | MLP F1 |
|---|---|---|---|---|
| DINOv2 | 0.541 | **0.896** | 0.138 | **0.571** |
| HOCT 19D | 0.575 | 0.873 | 0.145 | 0.394 |
| Regionprops (7D) | 0.557 | 0.860 | 0.131 | 0.396 |
| CNN NT-Xent | 0.543 | 0.703 | 0.129 | 0.289 |
| CNN e2e | 0.500 | 0.500 | 0.000 | 0.000 |

### 9.4 Sparse Attention Speed Summary (N=512)

| Method | Time (ms) | Memory (MB) | vs dense_flash |
|---|---|---|---|
| dense_flash | 0.10 | 4.16 | 1.00× |
| mask_knn (K=16) | 0.20 | 8.16 | 0.49× |
| sparse_softmax (K=16) | 0.63 | 16.5 | 0.16× |
| gather_knn (K=16) | 1.77 | 151.9 | 0.06× |

*(Source: `benchmark_backward/sparse_backward_results.csv`, N=512, K=16)*

---

## 10. Failed Experiments (Documented)

### 10.1 λ(t) Decay

- **Hypothesis:** Gradually reducing CNN feature weight would force the model to transition from CNN-reliant to regionprops-reliant.
- **Result:** λ only (0.077) and λ+dropout (0.040) both worse than vanilla CNN (0.035).
- **Root cause:** SGD cannot willingly increase loss to down-weight CNN features. The λ decay schedule was ad-hoc with no literature precedent. Equivalent mechanisms (HOCT auxiliary loss, LINet stochastic regularization) use fundamentally different approaches.
- **Configs:** `configs/vanvliet_lambda_decay.yaml`, `configs/vanvliet_lambda_decay_dropout.yaml`, `configs/vanvliet_lambda_half.yaml`

### 10.2 Identity BCE SSL

- **Experiment:** Using BCE with identity-matrix targets for SSL pretraining.
- **Config:** Not separately saved; parameters in `configs/base-config.yaml` under SSL settings.
- **Result:** The identity BCE SSL strategy collapsed. All models predicted all-negative (balanced accuracy = 0.500 ± 0.000).
- **Root cause:** BCE identity objective has no gradient signal — the optimal solution is trivial (all zeros) and the model finds it immediately.
- **Labbook:** `labbook/2026-06-01_identity_bce_negative_result.md`

### 10.3 CNN Fine-Tuning (Trainable)

- **Result:** Best val_loss = 0.036 at epoch 8, then monotonic degradation to 0.083.
- **Root cause:** Catastrophic forgetting of NT-Xent features during BCE fine-tuning. The 1.47M CNN params overfit to 1098 training windows.

### 10.4 CNN Injection After LayerNorm (Architectural Flaw)

- **Root cause:** In `model.py:_embed()`, CNN features are added **after** LayerNorm:
  ```python
  features = self.proj(features)
  features = self.norm(features)           # LayerNorm applied FIRST
  features = features + self.cnn_proj(cnn_out)  # CNN added AFTER norm
  ```
- **Consequence:** CNN features bypass normalization entirely. The norm layers cannot disentangle CNN noise from regionprops signal.
- **Suggested fix:** Move CNN injection before LayerNorm:
  ```python
  features = self.proj(features) + self.cnn_proj(cnn_out)  # blend before norm
  features = self.norm(features)                            # normalize together
  ```

---

## Appendix A: SLURM Scripts

| Script | Purpose | GPU Required |
|---|---|---|
| `bench-transformer-cell-tracking/benchmark_combined/run_dist_ablation.slurm` | Distributed KNN ablation (all K values) | A5000 |
| `bench-transformer-cell-tracking/benchmark_combined/run_diag2.slurm` | Diagnostic convergence runs | A5000 |
| `bench-transformer-cell-tracking/benchmark_combined/run_baseline.slurm` | Baseline dense training | A5000 |
| `bench-transformer-cell-tracking/benchmark_combined/run_low_label_exp.slurm` | Low-label fraction experiments | A5000 |
| `bench-transformer-cell-tracking/benchmark_combined/run_eval_all.slurm` | Evaluate all checkpoints | A5000 |
| `bench-transformer-cell-tracking/benchmark_combined/eval_one_model.slurm` | Single model evaluation | A5000 |
| `bench-transformer-cell-tracking/benchmark_combined/run_dense_vs_sparse_clean.slurm` | Clean dense vs sparse comparison | A5000 |
| `benchmark_ssl/run_ssl_dino_pretrain.slurm` | DINOv2 SSL pretraining | A5000 |
| `benchmark_ssl/cnn_encoder/run_full_cnn_extended.slurm` | CNN extended experiments (dropout, trainable, both) | **H100** |

## Appendix B: Config File Reference

| Config | Experiment | Key Parameters |
|---|---|---|
| `configs/vanvliet_baseline.yaml` | Regionprops-only baseline (no CNN) | `cnn_encoder: null` |
| `configs/vanvliet_baseline_cnn.yaml` | Vanilla CNN (frozen, no dropout) | `cnn_encoder.use: true`, `cnn_feat_dropout: 0`, `cnn_trainable: false` |
| `configs/vanvliet_cnn_dropout.yaml` | CNN + feature dropout | `cnn_feat_dropout: 0.2` |
| `configs/vanvliet_cnn_trainable.yaml` | CNN trainable (fine-tune) | `cnn_trainable: true` |
| `configs/vanvliet_cnn_both.yaml` | CNN + dropout + trainable | `cnn_feat_dropout: 0.2`, `cnn_trainable: true` |
| `configs/vanvliet_cnn_concat.yaml` | CNN CONCAT (best variant) | `cnn_injection: concat` |
| `configs/vanvliet_lambda_decay.yaml` | λ(t) decay | `cnn_lambda_decay: true` |
| `configs/vanvliet_lambda_decay_dropout.yaml` | λ(t) + dropout | `cnn_lambda_decay: true`, `cnn_feat_dropout: 0.2` |
| `configs/vanvliet_lambda_half.yaml` | λ halved | `cnn_lambda_init: 0.5` |
| `configs/base-config.yaml` | Base config with paths | All shared parameters |
| `configs/dummy.yaml` | Minimal test config | Quick smoke test |

## Appendix C: Directory Structure Reference

```
.
├── benchmark_attn/                    # Attention micro-benchmarks
├── benchmark_backward/                # Backward pass benchmarks
│   └── sparse_backward_results.csv    # 76-row CSV, 5 methods × 3 K values × 5 N values
├── benchmark_pipeline/                # Pipeline component benchmarks
│   ├── ffn_checkpoint_results.csv     # FFN checkpointing memory/time tradeoff
│   ├── regionprops_results.csv        # CPU vs GPU inference split
│   ├── blockwise_norm_results.csv     # Serial vs vectorized normalization
│   └── spatial_blocks_results.csv     # Spatial block efficiency
├── benchmark_mask_vs_gather.csv       # 13-row: mask_knn vs gather vs dense
├── benchmark_small_n_results.csv      # 13-row: small-scale benchmark
├── benchmark_sparse_results.csv       # 81-row: full sparse benchmark (2 methods × 2 L × 7 N × 3 K)
├── bench-transformer-cell-tracking/   # Submodule: core benchmark framework
│   └── benchmark_combined/results/
│       ├── speed_mem.csv              # 16-row: N vs time/memory for K=0,4,8,16,32
│       ├── ablation_full.csv          # 205-row: full convergence ablation
│       └── downstream_matched.csv     # 301-row: downstream matched evaluation
├── benchmark_ssl/                     # SSL benchmarks + probe experiments
│   └── probe/unified_edge_probe.py    # 1599-line: unified edge probing framework
├── configs/                           # 12 YAML configs for all experiments
├── labbook/                           # 50+ experiment log files
├── results/
│   ├── generate_plots.py              # THIS REPRODUCTION SCRIPT
│   └── figures/                       # Output directory (9 PDFs)
├── results_final/                     # Final convergence tables
│   ├── results_table.csv              # 6-row summary
│   └── convergence.csv                # 800+ row per-epoch convergence
├── progress_roadmap.html              # Master progress dashboard (all chart data)
└── reproduction.md                    # THIS DOCUMENT
```

---

*Generated on 2026-07-21. For questions, contact leonard.starke@mediainterface.de*
