# Extended H100 Experiment — SPEC 2: Scheduled Mixing λ(t) Decay

## Goal
Fix the CNN feature injection trap by implementing **scheduled mixing** (λ(t) cosine decay) — a training-only change that lets CNN features shape early representation geometry then gracefully fade away, producing a model that is **identical to the pure-regionprops baseline at inference time**.

## Background

### The Trap (from Experiment 1 results)

Trackastra injects frozen CNN features additively **after** LayerNorm:
```python
# model.py line 481
features = self.norm(features) + self.cnn_proj(cnn_out)
```

This creates a trap mechanism (see labbook `2026-07-19_cnn_extended_experiment_results.md` §5.2):

1. **Epochs 0–10**: CNN features provide helpful signal. The optimizer commits to a hybrid representation — `cnn_proj.weight` norm converges to ~10.8.
2. **Epochs 10–50**: The hybrid representation plateaus. The model cannot escape because SGD cannot willingly increase loss.
3. **Result**: One of our best CNN variants (dropout-only) reaches val_loss=0.016 — still **5.3× worse** than the pure regionprops baseline (0.003).

The `cnn_proj` weight norm is invariant (~10.8) across all tested strategies (dropout, trainable, both, vanilla). The model **never learns to turn down the CNN signal**.

### The Fix: Scheduled Mixing λ(t)

Replace the hard additive injection:
```python
features = self.norm(features) + self.cnn_proj(cnn_out)
```
with a cosine-decayed version:
```python
lambda_t = 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
features = self.norm(features) + lambda_t * self.cnn_proj(cnn_out)
```

**Behaviour:**
- λ(0) = 1.0: Full CNN signal at start — shapes representation geometry
- λ(T/2) = 0.5: Half-weight at midpoint
- λ(T) = 0.0: CNN features completely removed at end — model is pure regionprops

### Theoretical Validation

Two concurrent papers provide independent support:

1. **TextTeacher (Nauen et al., 2026, TMLR)**: Shows that auxiliary features (analogous to our CNN features) act as **feature-space preconditioners** — they shape the optimization landscape during early training but are not needed at convergence. TextTeacher attaches a text encoder to an image model, mixes the text features into the image encoder with a learned gating parameter, and demonstrates that the gating parameter naturally decays to near-zero by convergence. Our λ(t) schedule achieves the same effect deterministically.

2. **LINet — Continuous Linear Integration (Clinger, 2026, arXiv:2606.31135)**: Proposes **progressive modality dropout** where the dropout probability for an auxiliary modality increases linearly from 0→1 over the course of training. This is structurally equivalent to our λ(t) going from 1→0. LINet shows this prevents modality competition and allows the primary modality to fully take over. Our scheduled mixing is the continuous (non-stochastic) analogue of LINet's progressive dropout.

3. **HOCT (Bragantini et al., 2026)**: Context — HOCT's edge-centric architecture avoids this trap entirely because it uses explicit edge tokens rather than node-centric dot-product similarity. Our λ(t) approach is a training-only patch for the node-centric architecture, but the HOCT framing explains why this trap exists: node-centric representations fuse feature sources at the embedding level, making it impossible for the model to selectively ignore a feature source at inference time.

## Prerequisites
- Previous experiment complete (`EXTENDED_EXPERIMENT_SPEC.md`, jobs 3768419)
- Branch `cached-dist-attn` with all previous modifications
- Smoke test `smoke_test_cnn_trackastra.py` must pass before SLURM submission
- Requires understanding of `global_step` propagation in Lightning (see M2)

## Modifications

### M1: Lambda Decay in model.py (+6 lines)

**File:** `trackastra/trackastra/model/model.py`  
**Location:** `_embed()` method, after line 480 (before the existing `features = features + self.cnn_proj(cnn_out)` on line 481)

**Current code (line 481):**
```python
features = features + self.cnn_proj(cnn_out)
```

**New code:**
```python
# Scheduled mixing: λ(t) cosine decay from 1→0
if self.config.get("lambda_decay", False):
    global_step = self.config.get("global_step", 0)
    total_steps = self.config.get("total_steps", 1)
    lambda_t = 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
    features = features + lambda_t * self.cnn_proj(cnn_out)
else:
    features = features + self.cnn_proj(cnn_out)
```

**Also need to add `import math`** at top of file if not already present.

**Config keys added to `self.config` dict (in `__init__`):**
- `lambda_decay` (bool, default `False`)
- `global_step` (int, default `0`)
- `total_steps` (int, default `1`)

**Smoke test:** Verify that with `lambda_decay=True`, `lambda_t` starts at ~1.0 at step 0 and reaches ~0.0 at step = total_steps.

### M2: Getting total_steps and global_step into the Model

The model needs to know the current training progress. There are two design options; we recommend the **Lightning Callback approach** because it is non-invasive and does not require changing train.py's data flow.

**Option A (Recommended): Lightning Callback**  
File: `trackastra/scripts/train.py`  

1. Create a `LambdaDecayCallback` that updates the model's config at each training step:
```python
class LambdaDecayCallback(pl.pytorch.callbacks.Callback):
    def on_train_start(self, trainer, pl_module):
        """Called at the beginning of training, before the first step."""
        # Compute total_steps = epochs * steps_per_epoch
        if hasattr(trainer, 'estimated_stepping_batches') and trainer.estimated_stepping_batches > 0:
            total_steps = int(trainer.estimated_stepping_batches)
        else:
            # Fallback: approximate from datamodule
            total_steps = trainer.max_epochs * len(trainer.train_dataloader)
        pl_module.model.config["total_steps"] = total_steps
        pl_module.model.config["global_step"] = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Update global_step before each forward pass."""
        pl_module.model.config["global_step"] = trainer.global_step
```

2. Add this callback to the `callbacks` list where other callbacks are appended (around line 890 in train.py).

**Option B (Simpler): Pre-compute total_steps in train.py**  
Before creating the model, compute:
```python
total_steps = args.epochs * len(datamodule.train_dataloader())
```
Then pass it through: `model = TrackingTransformer(..., total_steps=total_steps)` and store in `self.config["total_steps"]`. The `global_step` still needs updating — either via callback or by modifying `WrappedLightningModule.training_step()` to call `self.model.config["global_step"] = self.global_step` before the forward pass.

We recommend Option A for minimum diff. Document both.

### M3: Config Changes (3 new YAML files)

Copy `configs/vanvliet_baseline_cnn.yaml` and add the key `lambda_decay: true`.

| Config | Base | Added Fields |
|--------|------|-------------|
| `configs/vanvliet_lambda_decay.yaml` | vanvliet_baseline_cnn | `lambda_decay: true` |
| `configs/vanvliet_lambda_decay_dropout.yaml` | vanvliet_baseline_cnn | `lambda_decay: true`, `cnn_feat_dropout: 0.2` |
| `configs/vanvliet_lambda_half.yaml` | vanvliet_baseline_cnn | `lambda_decay: true` + code modification for λ=0.5 constant |

**Note on `lambda_half`**: This is the **ablation control** — it uses a modified model where `lambda_t = 0.5` (constant, no decay). This tests: is the benefit from (a) having some λ<1, or (b) specifically from the decay schedule? If constant λ=0.5 performs identically to λ(t) decay, the decay isn't the mechanism — pure down-weighting is.

You can implement this with a separate config key `lambda_constant: 0.5` that overrides the schedule:
```python
lambda_constant = self.config.get("lambda_constant", None)
if lambda_constant is not None:
    lambda_t = lambda_constant
else:
    lambda_t = 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
```

### M4: SLURM Runs (3 sequential experiments)

**File:** `benchmark_ssl/cnn_encoder/run_lambda_decay.slurm`  

Run 3 sequential experiments on a single H100 (72h wall time, 500 epochs each):

1. **Lambda decay only** — `configs/vanvliet_lambda_decay.yaml`  
   Frozen CNN + λ(t) cosine decay 1→0. The primary hypothesis test.

2. **Lambda decay + dropout** — `configs/vanvliet_lambda_decay_dropout.yaml`  
   Frozen CNN + λ(t) decay + feature dropout (0.2). Tests whether combining both partial fixes produces cumulative gains.

3. **Ablation: constant λ=0.5** — `configs/vanvliet_lambda_half.yaml`  
   Frozen CNN + constant λ=0.5 (no decay). Tests whether the decay schedule matters or just reducing CNN weight.

**Reference runs for comparison (already completed):**
- `configs/vanvliet_baseline.yaml` — pure regionprops (val_loss=0.003)
- `configs/vanvliet_baseline_cnn.yaml` — vanilla frozen CNN (val_loss=0.035)
- `configs/vanvliet_cnn_dropout.yaml` — CNN + dropout (val_loss=0.016)

## Expected Results

| Variant | Expected val_loss @ epoch 10 | Expected @ epoch 100 | Expected @ epoch 500 | vs Baseline (0.003) |
|---------|:---:|:---:|:---:|:---:|
| **Baseline (no CNN)** | 0.070 | 0.010 | **0.003** | 1× |
| **Vanilla CNN** (ref) | 0.061 | 0.035 | 0.035 (plateau) | **11.7× worse** |
| **CNN + dropout** (ref) | 0.065 | 0.020 | 0.016 (plateau) | **5.3× worse** |
| **λ(t) decay only** | 0.061 | **0.015** | **0.005** | **~1.7× worse** |
| **λ(t) decay + dropout** | 0.063 | **0.012** | **0.004** | **~1.3× worse** |
| **Constant λ=0.5** | 0.065 | 0.025 | 0.020 | **~6.7× worse** |

**Rationale for λ(t) decay predictions:**
- By epoch 100 (~20% of training), λ ≈ cos(0.2π) ≈ 0.65 — CNN still at 65% weight but decaying. The model is already shifting to regionprops.
- By epoch 250 (~50%), λ ≈ 0.0 — CNN features fully removed. The model continues training as pure regionprops.
- Final convergence should approach the baseline (0.003) because the model is **identical** to the baseline at inference.
- The gap vs baseline represents the cost of early CNN contamination that may leave residual artifacts in the transformer weights.
- Constant λ=0.5 should perform similarly to vanilla CNN (half the CNN signal = half the damage) — providing a strong control.

### Key Metrics to Monitor

1. **Early convergence speed** (epochs 0-20): Does λ(t) preserve the 15-43% speedup from CNN injection?
2. **Late convergence** (epochs 100-500): Does the model approach baseline after λ→0?
3. **λ(t) trajectory**: Log `lambda_t` to confirm cosine decay. Verify that after λ<0.01 (≈epoch 230), val_loss trajectory matches the baseline run.
4. **cnn_proj weight norm**: Measure at initialization, mid-training (λ=0.5), and end (λ=0). Expect it to remain ~10.8 early but become irrelevant at inference.

## Confidence Assessment

| Outcome | Likelihood | Why |
|---------|:---:|------|
| **λ(t) decay reaches val_loss < 0.010** | 60% | This is the primary hypothesis. TextTeacher and LINet both independently validate the mechanism. The model becomes pure regionprops at inference — it should approach baseline performance. The 40% uncertainty is whether early CNN contamination leaves irreversible damage in the transformer weights (e.g., attention patterns biased toward CNN-derived representations). |
| **λ(t) + dropout reaches val_loss < 0.005** | 30% | Dropout provides stochastic regularization during the early phase when λ is high, potentially reducing CNN contamination. But dropout was already tested and reached only 0.016 — its marginal benefit on top of λ(t) may be small. |
| **Constant λ=0.5 beats vanilla CNN** | 80% | Reducing CNN weight from 1.0 to 0.5 should roughly halve the damage. Expect ~0.020 vs 0.035. This confirms the mechanism is simply CNN signal strength. |
| **λ(t) fully recovers baseline (0.003)** | 15% | Optimistic case. Requires that CNN contamination during early training is fully reversible. More likely the transformer retains some representation bias even after CNN features are removed. |
| **No improvement over CNN + dropout (0.016)** | 15% | Pessimistic case. If the transformer's attention patterns have already learned to exploit CNN features and cannot unlearn them even after the features are removed, λ(t) would not help. Partial evidence: the model was trained with dropout for 136 epochs and never recovered — the damage was cumulative. But in that experiment, CNN features were always present; in λ(t), they are fully removed. |

## NOT EST — Will the λ(t) schedule interfere with the learning rate schedule?
The existing training uses a `WarmupCosineLRScheduler` (warmup_epochs=5, max_epochs=500). λ(t) uses its own cosine schedule (no warmup, full 0→500 range). These two cosine schedules operate on different timescales and are independent — the LR schedule controls optimizer step size, while λ(t) controls feature mixing. No interference expected, but logging both `lr` and `lambda_t` per epoch for verification is recommended.

## Validation Checklist
- [ ] `math` imported at top of `model.py`
- [ ] `lambda_decay`, `global_step`, `total_steps` added to `self.config` in `model.py:TrackingTransformer.__init__`
- [ ] Lambda decay code added at line 481 of `model.py` (guarded by `lambda_decay` config flag)
- [ ] `LambdaDecayCallback` implemented in `train.py` (or alternative M2 approach)
- [ ] Three configs created and parseable by `train.py`
- [ ] With `lambda_decay=false`, original behaviour is preserved (no regression)
- [ ] Smoke test: verify `lambda_t=1.0` at step 0, `lambda_t≈0.0` at step = total_steps
- [ ] Smoke test: verify `lambda_t` is logged during training
- [ ] SLURM script passes test mode: `sbatch --export=TEST=1 run_lambda_decay.slurm`
- [ ] Push to `cached-dist-attn`
- [ ] Pull on Capella
- [ ] Submit

## Implementation Order
1. Modify `model.py` (M1: lambda decay logic, config keys, import math) — 6 lines of logic
2. Create `LambdaDecayCallback` in `train.py` (M2) — 15 lines
3. Add callback to callbacks list in `train.py` — 1 line
4. Create 3 config YAMLs (M3) — copy baseline, add `lambda_decay: true`
5. Create SLURM script `run_lambda_decay.slurm` (M4) — copy extended template, update configs
6. Update `smoke_test_cnn_trackastra.py` to test lambda decay config
7. Run smoke test → local validation
8. Commit, push, pull on Capella, submit

## References

- Nauen, et al. (2026). "TextTeacher: Auxiliary Features as Feature-Space Preconditioners." *Transactions on Machine Learning Research (TMLR)*.
- Clinger (2026). "LINet: Progressive Modality Dropout for Continuous Linear Integration." arXiv:2606.31135.
- Bragantini, J., Theodoro, I., & Royer, L. A. (2026). "Higher-Order Cell Tracking Transformer." arXiv:2607.11754v1.
