# benchmark_ssl — SSL Pretraining Analysis & Signal Tests

Diagnostic tools for self-supervised pretraining on vanvliet cell tracking data.

## Project layout

```
benchmark_ssl/
├── src/                  # Python source (importable as src.*)
│   ├── data/             # Feature extraction, frame loading, SSL dataset
│   ├── distortions/      # Augmentation pipeline (6 distortion families)
│   ├── models/           # CellEmbedder, attention modules, ScaledCNN
│   ├── training/         # SSL pretrainer, multi-config trainer
│   ├── analysis/         # Signal, convergence, downstream, diagnostics
│   ├── edge_probing/     # Edge probing harness (unified_edge_probe, etc.)
│   ├── cnn_encoder/      # CNN pretraining, convergence, cross-dataset eval
│   └── mini_trackastra/  # Mini trackastra experiments
├── slurm/                # HPC submission scripts + cluster_env.sh
├── configs/              # YAML configuration
├── tests/                # Unit and integration tests
└── docs/                 # SPEC.md, REPRODUCTION.md, results_analysis.md
```

## Setup

```bash
uv sync
```

## Usage

All entry points are invoked via `python -m src.<module>`:

### 1. Data-level signal check (fast, no GPU needed)
```bash
python -m src.analysis.signal_analyzer --conditions rpsM,recA,pheA,metA,cib,trpL --max-frames 500
```
Reports separation gap between same-cell and different-cell features. `gap > 0.3` → strong signal for contrastive learning.

### 2. DINO feature test (GPU)
```bash
python -m src.analysis.dino_test
```
Tests if DINOv2 patch features provide better separation than 7D regionprops (gap improved from 0.047 → 0.29).

### 3. Convergence predictor (GPU, ~30min)
```bash
python -m src.analysis.convergence_predictor --max-frames 200
```
Runs 4 tests: gap-vs-distortion, micro-SSL training, distortion ranking, PCA. Outputs HTML + CSVs.

### 4. Downstream convergence test (GPU, ~15min)
```bash
python -m src.analysis.downstream_convergence --max-pairs 60 --epochs 20
```
Compares SSL-pretrained vs random init on real tracking pairs.

### 5. Unified edge probe (GPU, ~50min)
```bash
python -m src.edge_probing.unified_edge_probe --features all --probe both --epochs 200 --cv-folds 5
```
5-fold CV edge probe over regionprops, HOCT, CNN-NT-Xent, CNN-e2e, and DINOv2 features.

### 6. HPC experiments
```bash
sbatch slurm/h100_conv_race.slurm        # H100 convergence race
sbatch slurm/cross_dataset_eval.slurm    # Cross-dataset evaluation
sbatch slurm/unified_edge_probe.slurm    # Unified edge probe (H100)
```

### 7. Run tests
```bash
pytest tests/
```

## Subpackage overview

| Subpackage | Key symbols | Purpose |
|---|---|---|
| `src.data` | `SSLDataset`, `collate_ssl`, `features_from_frame`, `load_experiment_frames`, `extract_*` | Data loading and feature extraction |
| `src.distortions` | `DistortionPipeline`, `AffineDistortion`, `JitterDistortion`, etc. | Augmentation pipeline (6 families) |
| `src.models` | `CellEmbedder`, `ScaledCNN`, `RotaryPositionalEncoding`, `GatherSparseAttention`, etc. | Model architectures |
| `src.training` | `ssl_pretrainer`, `ssl_trainer_multi`, `ssl_trainer` | SSL training scripts |
| `src.analysis` | `signal_analyzer`, `convergence_predictor`, `dino_test`, etc. | Analysis and diagnostic scripts |
| `src.edge_probing` | `unified_edge_probe`, `train_cnn_probe`, `verify_normalization` | Edge probing framework |
| `src.cnn_encoder` | `cnn_ssl`, `h100_convergence`, `cross_dataset_eval`, `visualize_embeddings` | CNN-specific experiments |
| `src.mini_trackastra` | `end_to_end`, `temporal_ssl`, `compare_backbones` | Mini trackastra experiments |

## Key findings (vanvliet)

- **7D regionprops**: gap=0.047 — insufficient for contrastive learning (confirmed by prior NT-Xent collapse)
- **DINOv2 patch features (384D)**: gap=0.29 — 6.2x improvement
- **Micro-SSL on hard pipeline** (affine+jitter+dropout): gap improved 0.04 → 0.36 (+0.32) in 200 steps
- **Generalization**: val gap +0.095 on held-out frames (genuine learning, not memorization)
- Fine-tuning convergence speedup expected from DINO+SSL pretraining

## Distortion Families

| Family | Effect | Preserves cell identity? |
|--------|--------|--------------------------|
| Affine | Rotation, scale, shear | Modifies shape features proportionally |
| Elastic | Non-linear local warp | Slightly perturbs features |
| Jitter | Per-cell random shift | Features unchanged (hardest for identity) |
| Dropout | Random cell removal | Teaches robustness to missing cells |
| Photometric | Intensity scale/shift | Only affects intensity features |
| Feature noise | Gaussian feature noise | Prevents shortcut memorization |

## Documentation

- `docs/SPEC.md` — refactor specification (design decisions, acceptance criteria)
- `docs/REPRODUCTION.md` — complete reproduction guide for all experiments
- `docs/results_analysis.md` — analysis of historical results

## References

- ASCENT (Han & Lu 2025): CVF — contrastive SSL for C. elegans neuron tracking
- Cell-DINO (Moutakanni et al. 2025): PLOS Comp Bio — DINOv2 SSL on cell microscopy
- Trackastra (Gallusser & Weigert 2024): ECCV — transformer-based cell tracking
