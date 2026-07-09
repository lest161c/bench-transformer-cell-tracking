# benchmark_ssl — SSL Pretraining Analysis & Signal Tests

Diagnostic tools for self-supervised pretraining on vanvliet cell tracking data.

## Contents

| File | Purpose |
|------|---------|
| `ssl_pipeline.py` | Data loading, SSLDataset, collator for contrastive SSL |
| `distortions.py` | 6 distortion families (affine, elastic, jitter, dropout, photometric, feature_noise) |
| `track_encoder.py` | ASCENT-inspired CellEmbedder |
| `pretrain.py` | NT-Xent contrastive SSL training with TensorBoard |
| `analyze_signal.py` | **Data-level signal analysis** (no model). Tests if features carry sufficient signal for contrastive learning. **Run first.** |
| `test_dino.py` | DINOv2 feature separation analysis. Tests DINO patch features as regionprops replacement. |
| `ssl_convergence_test.py` | **Convergence predictor.** Micro-SSL training + generalization + hard-pipeline tests. Predicts if full-scale training will converge before HPC submission. |
| `downstream_convergence.py` | Simulates fine-tuning: SSL-pretrained vs random init on real adjacent-frame pairs. |
| `rich_features.py` | Extended feature extraction (shape descriptors, Hu moments, local image patches via PCA). |
| `config.yaml` | Hyperparameters for distortion pipeline + SSL training. |
| `run_ssl_dino_pretrain.slurm` | HPC submission script for full DINO+SSL pretraining (24h, H100). |

## Setup

```bash
uv sync
```

## Usage

### 1. Data-level signal check (fast, no GPU needed)
```bash
uv run python3 analyze_signal.py --conditions rpsM,recA,pheA,metA,cib,trpL --max-frames 500
```
Reports separation gap between same-cell and different-cell features. `gap > 0.3` → strong signal for contrastive learning.

### 2. DINO feature test (GPU)
```bash
uv run python3 test_dino.py
```
Tests if DINOv2 patch features provide better separation than 7D regionprops (gap improved from 0.047 → 0.29).

### 3. Convergence predictor (GPU, ~30min)
```bash
uv run python3 ssl_convergence_test.py --max-frames 200
```
Runs 4 tests: gap-vs-distortion, micro-SSL training, distortion ranking, PCA. Outputs HTML + CSVs.

### 4. Downstream convergence test (GPU, ~15min)
```bash
uv run python3 downstream_convergence.py --max-pairs 60 --epochs 20
```
Compares SSL-pretrained vs random init on real tracking pairs.

### 5. HPC full pretraining
```bash
sbatch run_ssl_dino_pretrain.slurm
```

## Key findings (vanvliet)

- **7D regionprops**: gap=0.047 — insufficient for contrastive learning (confirmed by prior NT-Xent collapse)
- **DINOv2 patch features (384D)**: gap=0.29 — 6.2x improvement
- **Micro-SSL on hard pipeline** (affine+jitter+dropout): gap improved 0.04 → 0.36 (+0.32) in 200 steps
- **Generalization**: val gap +0.095 on held-out frames (genuine learning, not memorization)
- Fine-tuning convergence speedup expected from DINO+SSL pretraining

## Distortion Families

| Family | Effect | Preserves cell identity? |
|--------|--------|-------------------------|
| Affine | Rotation, scale, shear | Modifies shape features proportionally |
| Elastic | Non-linear local warp | Slightly perturbs features |
| Jitter | Per-cell random shift | Features unchanged (hardest for identity) |
| Dropout | Random cell removal | Teaches robustness to missing cells |
| Photometric | Intensity scale/shift | Only affects intensity features |
| Feature noise | Gaussian feature noise | Prevents shortcut memorization |

# References

- ASCENT (Han & Lu 2025): CVF — contrastive SSL for C. elegans neuron tracking
- Cell-DINO (Moutakanni et al. 2025): PLOS Comp Bio — DINOv2 SSL on cell microscopy
- Trackastra (Gallusser & Weigert 2024): ECCV — transformer-based cell tracking
