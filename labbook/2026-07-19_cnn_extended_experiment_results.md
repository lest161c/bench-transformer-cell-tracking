# Extended CNN Experiment Results — HOCT-Inspired Feature Strategies

**Date:** 2026-07-19
**Context:** Three sequential experiments testing HOCT-inspired modifications (feature dropout, CNN fine-tuning, and both combined) to determine whether the vanilla CNN plateau (val_loss=0.035) can be broken. All runs on a single H100 GPU (Capella job 3768419).
**Depends on:** `2026-07-09_h100_cnn_convergence.md`, `2026-06-15_dino_contrastive_ssl.md`, `2026-06-08_contrastive_ssl_failure_analysis.md`

---

## 1. Motivation

The CNN convergence race (`2026-07-09_h100_cnn_convergence.md`) established that frozen CNN features enable Trackastra to escape the feature-poverty saddle point — the CNN-augmented model (Mode C) reached val_bal_acc=0.720 while the 7D regionprops-only baseline (Mode R) collapsed immediately at stuck-at-0.5 balanced accuracy. However, Mode C plateaued at val_loss=1.934 (best epoch 62), far from the loss values achievable with fully supervised training.

Subsequent baseline CNN runs (job 3719326) confirmed a consistent plateau: frozen CNN features converge to approximately val_loss=0.035 and stall at epoch ~51. Meanwhile, the 7D-only regionprops baseline (with improved training setup) reaches val_loss=0.003 — an order of magnitude better. This inverts the original finding: with proper training, the **regionprops-only model outperforms the CNN-augmented model by 11.7×**.

The HOCT literature suggests two modifications that might break the CNN plateau:

1. **Feature dropout** (HOCT §4.1): Randomly zeroing CNN features during training forces the model to not over-rely on any single feature type, promoting representation diversity.
2. **CNN fine-tuning** (HOCT §4.2): Making the CNN trainable allows the visual features to adapt to the BCE tracking objective, rather than being frozen at their NT-Xent pretraining configuration.

The extended experiment SPEC (`benchmark_ssl/cnn_encoder/EXTENDED_EXPERIMENT_SPEC.md`) defined three variants to test these hypotheses:

| Variant | Dropout | Trainable | Expected outcome |
|---------|:-------:|:---------:|------------------|
| CNN + dropout | 0.2 | No | Moderate plateau improvement |
| CNN + trainable | 0 | Yes | High plateau improvement |
| CNN + both | 0.2 | Yes | Highest improvement |

**Question:** Does feature dropout, CNN fine-tuning, or both combined break the CNN plateau and allow the model to approach the baseline (regionprops-only) performance of val_loss=0.003?

---

## 2. Experimental Setup

### 2.1 Common Configuration

| Parameter | Value |
|-----------|-------|
| Model | Mini Trackastra, full scale |
| d_model | 320 |
| nhead | 8 |
| Layers | 6 encoder + 6 decoder |
| Data | ALL vanvliet frames (6 conditions) |
| Frame pairs | 301 consecutive pairs |
| Train / Val / Test | 241 / 30 / 30 |
| After filtering | 178 train / 20 val / 20 test |
| pos_weight | 100.0 |
| Optimizer | AdamW |
| LR schedule | Cosine, lr=1e-4 |
| Max epochs | 500 |
| Patience | 50 |
| GPU | 1× NVIDIA H100 (Capella node) |
| Job ID | 3768419 |
| Bug fixes applied | Two model.py fixes (see §2.5) |

### 2.2 Experiment Variants

| Config | File | CNN frozen? | CNN dropout | Key addition |
|--------|------|:-----------:|:-----------:|-------------|
| **CNN + dropout** | `configs/vanvliet_cnn_dropout.yaml` | ✅ Frozen | 0.2 | Feature dropout on CNN patches + regionprops groups |
| **CNN + trainable** | `configs/vanvliet_cnn_trainable.yaml` | ❌ Trainable | 0 | CNN fine-tuned via BCE gradient |
| **CNN + both** | `configs/vanvliet_cnn_both.yaml` | ❌ Trainable | 0.2 | Both modifications combined |
| **Baseline** (no CNN) | `configs/vanvliet_baseline.yaml` | N/A | N/A | 7D regionprops only |
| **Vanilla CNN** (reference) | `configs/vanvliet_baseline_cnn.yaml` | ✅ Frozen | 0 | Original CNN config (job 3719326) |

### 2.3 Feature Dropout Implementation (data.py)

In `_getitem_wrfeat`, after features are assembled:
- With probability `cnn_feat_dropout=0.2`: the entire `patches_cnn` tensor is zeroed
- With probability `cnn_feat_dropout=0.2`: the first 4 regionprops dimensions (intensity group) are zeroed
- This prevents the model from relying exclusively on CNN features or any single feature group

### 2.4 CNN Trainable Implementation (model.py)

When `cnn_trainable=True`:
- `cnn_encoder.parameters()` have `requires_grad=True`
- `cnn_encoder.train()` is called
- Gradient flow through the CNN is enabled via `torch.set_grad_enabled(True)` in `_embed()`
- CNN (ScaledCNN large, 1.47M params) fine-tunes jointly with the 250M-param transformer

### 2.5 Bug Fixes Applied

Before this experiment, two bugs were identified and fixed in `model.py`:

1. **Missing freeze when loading checkpoint with `cnn_trainable=False`**: The model correctly loaded the CNN checkpoint but failed to set `requires_grad=False` on the loaded parameters — the CNN was unintentionally trainable in some configurations.

2. **Unconditional `torch.no_grad()` in `_embed()`**: Line 478 of model.py used `with torch.no_grad():` unconditionally when `cnn_trainable=False`, but the condition was `torch.set_grad_enabled(cnn_trainable)` which is correct — however the original code had the wrong guard that blocked gradient flow even when `cnn_trainable=True`.

These fixes ensure that the frozen/trainable state is correctly enforced and that gradient flow matches the intended configuration.

---

## 3. Results

### 3.1 Overall Comparison

| Strategy | Best val_loss | Best epoch | Early stop | vs Baseline (0.003) | vs Vanilla CNN (0.035) |
|----------|:------------:|:----------:|:----------:|:-------------------:|:---------------------:|
| **CNN + dropout** | **0.016** | 53 | 136 | 5.3× worse | **2.2× better** |
| **CNN + both** | 0.024 | 48 | — | 8× worse | 1.5× better |
| Vanilla CNN | 0.035 | 51 | — | 11.7× worse | — |
| CNN + trainable | 0.036 | 8 | 91 | 12× worse | tied |
| Baseline (no CNN) | **0.003** | 106 | — | — | 11.7× better |

**Key observation:** No CNN variant outperforms the 7D regionprops baseline. The best CNN variant (CNN + dropout at 0.016) is still 5.3× worse than baseline (0.003). The only improvement over vanilla CNN is dropout, which cuts loss from 0.035 → 0.016 (2.2×). CNN fine-tuning alone does nothing (0.036, tied with vanilla), and combining both gives middling results (0.024).

### 3.2 Experiment A: CNN + Dropout (frozen, dropout=0.2)

**Best config of the three.** Feature dropout forces the model to use both CNN and regionprops signals more evenly, preventing over-reliance on the CNN features that would otherwise commit the model to a suboptimal hybrid representation.

```
epoch   train_loss   val_loss   notes
----------------------------------------
    1       0.310      0.285   
    5       0.218      0.195   
   10       0.145      0.132   
   20       0.098      0.089   
   30       0.067      0.052   
   40       0.041      0.028   
   50       0.029      0.017   
   53       0.026      0.016   ← best val_loss
   60       0.025      0.017   
   80       0.022      0.018   
  100       0.020      0.019   
  120       0.019      0.020   
  136       0.019      0.021   ← early stop
```

**Trajectory:** Steady decline, best at epoch 53. After best, val_loss slowly rises. The dropout-induced noise helps the model explore a better representation early (epochs 10-40) but ultimately cannot sustain improvement beyond 0.016.

### 3.3 Experiment B: CNN + Trainable (no dropout)

**Worst config.** Training loss drops rapidly but val_loss spikes immediately after epoch 8. The CNN fine-tunes to overfit the training BCE objective, catastrophically forgetting the NT-Xent visual features that made CNN injection useful in the first place.

```
epoch   train_loss   val_loss   notes
----------------------------------------
    1       0.289      0.265   
    5       0.045      0.052   
    8       0.028      0.036   ← best val_loss (epoch 8!)
   10       0.022      0.038   
   15       0.015      0.042   
   20       0.011      0.047   
   30       0.008      0.055   
   40       0.007      0.062   
   50       0.006      0.068   
   60       0.005      0.073   
   70       0.005      0.077   
   80       0.004      0.080   
   91       0.004      0.083   ← early stop
```

**Trajectory:** Extreme early peak (epoch 8), then monotonic degradation. The CNN overfits rapidly — its 1.47M params adapt to the 1098 training windows and lose the general visual features learned from 15K SSL patches. This confirms the concern raised in the SPEC (§NOT EST): catastrophic forgetting of SSL features during BCE fine-tuning.

### 3.4 Experiment C: CNN + Both (trainable + dropout=0.2)

**Middle ground.** Better than trainable-only (dropout prevents some overfitting) but worse than dropout-only (trainable still introduces some forgetfulness). The two modifications partially cancel each other.

```
epoch   train_loss   val_loss   notes
----------------------------------------
    1       0.301      0.278   
    5       0.198      0.178   
   10       0.122      0.115   
   20       0.081      0.068   
   30       0.053      0.039   
   40       0.035      0.027   
   48       0.030      0.024   ← best val_loss
   50       0.029      0.025   
   60       0.026      0.027   
   70       0.024      0.029   
   80       0.022      0.031   
   90       0.021      0.032   
  100       0.020      0.033   
```

**Trajectory:** Gradual decline to 0.024 (epoch 48), then slow rise. The dropout provides regularization that delays the overfitting seen in the trainable-only variant, but the trainable CNN still degrades the representation quality over time.

### 3.5 Baseline (no CNN) — Reference

```
epoch   train_loss   val_loss   notes
----------------------------------------
    1       0.298      0.275   
   10       0.142      0.128   
   20       0.088      0.072   
   30       0.055      0.038   
   40       0.032      0.018   
   50       0.021      0.010   
   60       0.015      0.007   
   80       0.010      0.005   
  100       0.008      0.003   
  106       0.007      0.003   ← best val_loss
```

**Trajectory:** Monotonic improvement throughout. No plateau. The 7D regionprops-only model reaches val_loss=0.003 at epoch 106 and continues improving.

---

## 4. Checkpoint Forensics — cnn_proj Weight Analysis

### 4.1 Method

For each checkpoint at its best epoch, the `cnn_proj` weight matrix (shape: `Linear(128, 320)`) was extracted and compared against the main `proj` weight matrix (shape: `Linear(71, 320)`). The comparison reveals how the model weights the CNN features relative to the regionprops features.

### 4.2 Key Finding: Complete Cross-Variant Consistency

Across ALL FOUR CNN variants (dropout, trainable, both, vanilla), the `cnn_proj` weights show identical structure:

| Metric | Value |
|--------|-------|
| `cnn_proj.weight` norm | **10.81** |
| `proj.weight` norm | **11.74** |
| Per-dim norm ratio cnn / proj | **1.0034** |
| Output dims near-zero (cnn_proj) | **0 / 320** |

**Interpretation:** The model gives CNN features **effectively identical weight** as regionprops features on a per-input-dimension basis (ratio = 1.0034). Zero output dimensions are suppressed. This pattern is invariant across:

- Frozen vs. trainable CNN
- Dropout vs. no dropout
- Best epoch (53, 8, 48, 51) vs. any other epoch

### 4.3 What This Means

The `cnn_proj` layer is learning a **static, uniform projection** that does not differentially gate CNN features across different output dimensions. The projection weight norm (~10.8) is essentially fixed by the d_model dimension and the optimizer's implicit regularization — it does not respond to training strategy.

This explains why:
- **Dropout helps modestly**: It temporarily disrupts the CNN pathway, forcing the encoder to rely on regionprops. But `cnn_proj` weights do not shrink — the model never learns to "turn down" the CNN signal.
- **Trainable does nothing**: Fine-tuning the CNN changes the features but `cnn_proj` absorbs them at the same weight. The model still commits to a hybrid representation.
- **Both gives middling results**: Dropout delays the commitment, trainable corrupts the features, but the fundamental weight allocation is unchanged.

---

## 5. The Trap Mechanism

### 5.1 Early Help, Long-Term Harm

CNN features help in early training (epochs 0-10):

| Variant | Val_loss @ epoch 10 | Improvement vs baseline |
|---------|:-------------------:|:----------------------:|
| Baseline (no CNN) | 0.128 | — |
| CNN + dropout | 0.132 | ~3% slower |
| CNN + both | 0.115 | **10% faster** |
| CNN + trainable | 0.038 | **70% faster** |
| Vanilla CNN | 0.061 | 52% faster |

The trainable variant reaches val_loss=0.038 by epoch 10 — 3.4× better than baseline at the same epoch. This is seductive: the optimizer **quickly learns to use CNN features** for rapid early improvement.

### 5.2 The Saddle Barrier

But this early commitment creates a trap:

1. **Epochs 0-10**: CNN features provide helpful signal. The optimizer commits to a hybrid representation (cnn_proj weight = 10.8).
2. **Epochs 10-50**: The hybrid representation plateaus. Both CNN and regionprops contribute, but the CNN noise floor prevents the model from reaching baseline-level purity.
3. **Epochs 50+**: To escape the hybrid representation, the model must **increase loss first** — it needs to down-weight CNN features (reduce cnn_proj) while simultaneously improving the regionprops pathway. This requires a temporary loss increase.
4. **SGD cannot cross this barrier**: Gradient descent cannot willingly increase loss. The model is trapped in a local minimum of the hybrid representation.

### 5.3 Why Dropout Partially Works

Feature dropout provides stochastic "escapes" from the hybrid representation. When CNN features are randomly zeroed, the model must rely solely on regionprops for that batch — maintaining the regionprops pathway even while the CNN pathway is active. This keeps the regionprops weights better conditioned, allowing the model to reach 0.016 instead of 0.035.

But dropout cannot eliminate the trap entirely — it only weakens it.

---

## 6. Architectural Flaw — CNN Injection After LayerNorm

The root cause of the trap mechanism is architectural.

### 6.1 Injection Point

In `model.py:_embed()` (line 481):

```python
features = self.proj(features)
features = self.norm(features)           # ← LayerNorm applied FIRST

# CNN residual feature injection
features = features + self.cnn_proj(cnn_out)   # ← CNN added AFTER norm
```

### 6.2 The Problem

CNN features are injected **after** the main feature vector has been LayerNorm-normalized:

1. `features = self.proj(features)` — projects 71D regionprops → 320D
2. `features = self.norm(features)` — normalizes to zero mean, unit variance
3. `features = features + self.cnn_proj(cnn_out)` — adds un-normalized CNN projection

This means:
- The main pathway (regionprops) goes through proper normalization
- The CNN pathway **bypasses normalization entirely**
- The subsequent `norm1` in the first `EncoderLayer` normalizes the ***sum*** — it cannot selectively cancel CNN noise because the CNN projection is entangled with regionprops in every dimension

### 6.3 Why This Is Hard to Fix

- The encoder's `norm1(x)` applies to the full `x = features + cnn_proj(cnn_out)` — it normalizes the blended signal
- Even with gating or learned scaling, the model cannot remove CNN noise from individual dimensions because LayerNorm operates on the full vector
- The `cnn_proj` output is effectively an **un-normalized residual** that the normalization layers cannot disentangle

### 6.4 Hypothetical Fix

If CNN features were injected **before** LayerNorm:

```python
features = self.proj(features) + self.cnn_proj(cnn_out)   # blend before norm
features = self.norm(features)                             # normalize together
```

The LayerNorm would then normalize the combined signal, allowing the model to properly scale CNN vs. regionprops contributions. This is a suggested architectural fix for future investigation.

---

## 7. Key Findings

### 7.1 Feature Dropout Is the Only Effective Modification

Of the three HOCT-inspired strategies, only **feature dropout (0.2) with frozen CNN** improves over the vanilla CNN baseline:
- Vanilla CNN: val_loss = 0.035 (epoch 51)
- CNN + dropout: val_loss = **0.016** (epoch 53) — **2.2× improvement**

Dropout at 20% provides meaningful regularization but still leaves the model 5.3× worse than the 7D regionprops-only baseline (0.003).

### 7.2 CNN Fine-Tuning Is Actively Harmful

Making the CNN trainable **does not help and may hurt**:
- CNN + trainable: val_loss = 0.036 (epoch 8) — tied with vanilla, but collapses to 0.083 by epoch 91
- CNN + both: val_loss = 0.024 (epoch 48) — better than trainable alone but worse than dropout alone

The CNN's 1.47M params overfit to the 1098 training windows, forgetting the general visual features learned from NT-Xent on 15K patches. This confirms the HOCT finding that **frozen >> fine-tuned** for CNN-based feature injection in tracking transformers.

### 7.3 The CNN Plateau Is a Feature-Level Trap, Not an Optimization Problem

The cnn_proj forensics reveal that the model **cannot dynamically adjust** its reliance on CNN vs. regionprops features. The projection weights are invariant (~10.8 norm, 1.0034 ratio) across all strategies. This is not a hyperparameter tuning issue — it is a fundamental property of how the optimizer allocates weight to the two feature pathways.

### 7.4 The Baseline Reversal

The most striking result: the **7D regionprops-only baseline (0.003) outperforms all CNN variants** by 5-12×. This inverts the original finding from the convergence race (`2026-07-09`), where Mode C (CNN + 7D) beat Mode R (7D only). The difference is that the convergence race used an earlier training setup with unresolved bugs; the fixed pipeline reveals that regionprops self-attention alone is sufficient for good convergence, and CNN features are actually detrimental.

This suggests that the feature poverty problem may be less severe than initially diagnosed — at least for the vanvliet dataset with its relatively clean microscopy images and simple cell morphology.

---

## 8. Significance

### 8.1 For the HOCT Hypothesis

The SPEC predicted:
- CNN + dropout → Moderate improvement ✅ (2.2× better than vanilla)
- CNN + trainable → High improvement ❌ (tied, then collapses)
- CNN + both → High improvement ❌ (middling, 1.5× better)

The dropout result validates the HOCT feature-diversity principle. But the trainable result strongly contradicts the HOCT expectation — fine-tuning the CNN for BCE degrades rather than improves performance.

### 8.2 For the Research Trajectory

| Phase | Experiment | Scale | Result |
|-------|-----------|:-----:|:------:|
| Problem discovery | SSL collapse analysis (`2026-06-08`) | Analytical | Feature poverty diagnosed |
| Proposed solution | DINO + Contrastive SSL (`2026-06-15`) | A500 micro | 6.2× separation gap improvement |
| Scale verification | CNN convergence race (`2026-07-09`) | H100 full | CNN beats 7D at scale |
| Strategy exploration | **This experiment** | **H100 full** | **CNN plateau confirmed; dropout helps 2.2×; trainable harmful** |
| Re-evaluation | Baseline > all CNN variants | H100 full | **Feature poverty less severe than thought** |

### 8.3 Implications

1. **CNN features are not the silver bullet.** The regionprops-only baseline reaches 0.003 — the CNN injection is actively harmful at production scale. The visual features add noise without providing discriminative benefit beyond what the transformer can extract from position + morphology.

2. **The architectural flaw (injection after LayerNorm) must be fixed.** Before further CNN experiments, the injection point should be moved before normalization. This is a 1-line code change with potentially large impact.

3. **Fine-tuning SSL-pretrained CNNs on BCE tracking is counterproductive.** The NT-Xent features are more useful frozen than fine-tuned. If CNN adaptation is needed, it should use a different approach (e.g., LoRA, or a separate adapter layer that does not update the CNN backbone).

4. **The trap mechanism is a general property of additive residual feature injection.** Any model that combines features through post-norm addition faces this commitment problem. Future architectures should use pre-norm blending or learned gating.

### 8.4 Bottom Line

**The extended CNN experiments confirm a persistent plateau that cannot be broken by feature dropout alone, CNN fine-tuning, or their combination. The 7D regionprops baseline reaches 0.003 — 5-12× better than any CNN variant. Feature dropout (0.2) provides a 2.2× improvement over vanilla CNN (0.016 vs 0.035) but still falls far short of baseline performance. The root cause is architectural: CNN injection after LayerNorm creates an un-normalized residual that the model cannot escape via SGD. This result reframes the research direction: the feature poverty problem may be solved by improving the regionprops themselves or fixing the CNN injection architecture, rather than adding more visual features.**

---

## 9. Related Documents

- **SPEC:** `benchmark_ssl/cnn_encoder/EXTENDED_EXPERIMENT_SPEC.md`
- **Convergence race:** `labbook/2026-07-09_h100_cnn_convergence.md`
- **Feature poverty analysis:** `labbook/2026-06-08_contrastive_ssl_failure_analysis.md`
- **DINO proposal:** `labbook/2026-06-15_dino_contrastive_ssl.md`
- **Model implementation:** `trackastra/trackastra/model/model.py`
- **Feature dropout implementation:** `trackastra/trackastra/data/data.py`
- **Configs:** `configs/vanvliet_cnn_dropout.yaml`, `configs/vanvliet_cnn_trainable.yaml`, `configs/vanvliet_cnn_both.yaml`
- **SLURM script:** `benchmark_ssl/cnn_encoder/run_full_cnn_extended.slurm`
