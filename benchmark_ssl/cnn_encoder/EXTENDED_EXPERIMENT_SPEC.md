# Extended H100 Experiment — SPEC

## Goal
Extend the CNN convergence race with HOCT-inspired techniques to test whether the CNN plateau can be broken.

## Prerequisites
- Base: `cached-dist-attn` branch, Trackastra with `use_cnn=true` (already working — job 3719326 at epoch 83)
- Smoke test: `smoke_test_cnn_trackastra.py` must pass before H100 submission

## Modifications

### M1: Feature Dropout (data.py, +5 lines)
**File:** `trackastra/trackastra/data/data.py`  
**Location:** In `_getitem_wrfeat`, after features are assembled, before returning
**Logic:**
```python
if self.use_cnn and self.cnn_feat_dropout > 0:
    if random.random() < self.cnn_feat_dropout:
        patches_cnn = torch.zeros_like(patches_cnn)  # zero CNN features
    # also randomly zero regionprops groups
    if random.random() < self.cnn_feat_dropout:
        feat[:, :4] = 0  # intensity group
```
**Config:** `cnn_feat_dropout: 0.2` in vanvliet_baseline_cnn.yaml  
**Smoke test:** verify patches_cnn is sometimes zeroed, sometimes not

### M2: CNN Unfreeze Flag (model.py + train.py, +15 lines)
**File:** `trackastra/trackastra/model/model.py`  
**Location:** `TrackingTransformer.__init__`  
**Logic:**
```python
# When cnn_trainable=True, don't freeze the CNN encoder
if use_cnn and not cnn_trainable:
    for p in self.cnn_encoder.parameters():
        p.requires_grad = False
```
**File:** `trackastra/scripts/train.py`  
**Location:** argument parser + model creation  
**Logic:** add `--cnn-trainable` flag, pass to model  
**Config:** `cnn_trainable: true` in vanvliet_baseline_cnn.yaml  
**Smoke test:** verify `cnn_encoder.parameters()` have `requires_grad=True`

### M3: Experiment Configs (3 new configs)
```
configs/vanvliet_cnn_dropout.yaml     # baseline + use_cnn + cnn_feat_dropout=0.2
configs/vanvliet_cnn_trainable.yaml   # baseline + use_cnn + cnn_trainable=true
configs/vanvliet_cnn_both.yaml        # baseline + use_cnn + cnn_feat_dropout=0.2 + cnn_trainable=true
```

### M4: Extended SLURM Script
**File:** `benchmark_ssl/cnn_encoder/run_full_cnn_extended.slurm`  
Run 3 sequential experiments:
1. `vanvliet_cnn_dropout.yaml` — feature dropout only
2. `vanvliet_cnn_trainable.yaml` — unfrozen CNN only
3. `vanvliet_cnn_both.yaml` — both combined

## Expected Results

| Variant | Expected val_loss @ epoch 10 | Expected @ epoch 40 | Expected plateau |
|---------|:---:|:---:|:---:|
| Baseline (7D) | 0.070 | 0.031 | None (keeps dropping) |
| CNN (frozen, current) | 0.061 | 0.104 | 0.035 at epoch 51+ |
| **CNN + dropout** | 0.065 | 0.080 | **MODERATE — dropout forces diversity but also adds noise** |
| **CNN + trainable** | 0.058 | 0.060 | **HIGH — unfrozen CNN can cross-train for BCE** |
| **CNN + both** | 0.060 | 0.050 | **HIGH — training signal + robustness** |

## Confidence Assessment

| Outcome | Likelihood | Why |
|---------|:---:|------|
| **CNN trainable breaks plateau** | 60% | CNN fine-tuned for BCE should produce cross-attendable features. Training cost: CNN is 1.5M params (0.6% of Trackastra's 250M params) — negligible overhead |
| **Feature dropout helps early** | 70% | Literature (HOCT §4.1) and our A500 CNN experiments both show dropout prevents over-reliance on single feature type |
| **Plateau remains with both fixes** | 20% | If the plateau is truly architectural (node-centric dot-product, per HOCT), no feature-level fix helps. This would be a significant negative result supporting HOCT's edge-centric claim |
| **Trainable CNN diverges/crashes** | 10% | 1.5M extra params on 250M model is ~0.6% — unlikely to destabilize training |

## NOT EST — Will the trainable CNN overfit?
The CNN was pretrained on NT-Xent (cosine separation) with 15K vanvliet patches. Fine-tuning on BCE tracking with 1098 windows may cause catastrophic forgetting of the SSL features. HOCT's finding (frozen >> fine-tuned) suggests keeping the CNN frozen may actually be better. This is the key open question the experiment answers.

## Validation Checklist
- [ ] `smoke_test_cnn_trackastra.py` passes with NEW feature dropout config
- [ ] `smoke_test_cnn_trackastra.py` passes with trainable CNN
- [ ] Smoke test verifies CNN params have correct `requires_grad` state
- [ ] Three new configs are parseable by train.py without errors
- [ ] SLURM script passes test mode: `sbatch --export=TEST=1 run_full_cnn_extended.slurm`
- [ ] Push to cached-dist-attn
- [ ] Pull on Capella
- [ ] Submit

## Implementation Order
1. Modify data.py (M1: feature dropout) — 5 lines
2. Modify model.py (M2: cnn_trainable flag) — 10 lines
3. Modify train.py (M2: CLI flag) — 5 lines
4. Create 3 configs (M3) — copy baseline config, add fields
5. Create SLURM script (M4) — copy full_cnn_race.slurm, change configs
6. Update smoke_test_cnn_trackastra.py to test new configs
7. Run smoke test → local validation
8. Commit, push, pull on Capella, submit
