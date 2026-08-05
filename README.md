# Bench Transformer Cell Tracking

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ECCV 2024](https://img.shields.io/badge/ECCV-2024-4b44ce.svg)](https://github.com/weigertlab/trackastra)
[![arXiv](https://img.shields.io/badge/arXiv-2405.15700-b31b1b.svg)](https://arxiv.org/abs/2405.15700)

Sparse attention benchmarks and CNN feature injection experiments for transformer-based cell tracking (Trackastra, ECCV 2024).
**Bottom line:** Dense attention outperforms all CNN-injection variants; sparse attention matches dense convergence at K=16 with up to 5.9× speedup at long sequences.

---

## Benchmark Code

| Directory | Description |
|---|---|
| `benchmark_attn/` | Standalone sparse attention micro-benchmarks (dense vs gather vs mask-KNN vs spatial-flash) |
| `benchmark_ssl/` | CNN encoder training (DINO, MAE), edge probing, cross-dataset downstream evaluation |
| `benchmark_combined/` | End-to-end training benchmarks, convergence experiments, local SLURM launchers (the clean K-sweep harness and multi-seed configs live in the separate `bench-transformer-cell-tracking/` checkout) |
| `configs/` | YAML configs for all training runs (baseline, CNN variants, lambda-schedule) |

---

## Results

Detailed results are kept in the per-experiment documentation next to the code:

| Area | Where results live |
|---|---|
| Sparse attention micro-benchmarks (dense vs gather vs mask-KNN, CachedDistAttention, GatherV3, backward, speed/memory) | `benchmark_attn/REPRODUCTION.md` |
| CNN feature injection, edge probing, DeepCell cross-dataset | `benchmark_ssl/REPRODUCTION.md` and `benchmark_ssl/cnn_encoder/REPRODUCTION.md` |
| Clean K-sweep training (convergence, TRA/AOGM) | `benchmark_combined/REPRODUCTION.md` |

---

## Installation

```bash
git clone https://github.com/lest161c/bench-transformer-cell-tracking.git
cd bench-transformer-cell-tracking
```

**Dependencies:**

```bash
pip install torch>=2.5.0 torchvision>=0.20.0
pip install seaborn matplotlib pandas numpy scipy scikit-learn tifffile
pip install wandb pyyaml einops lightning
```

---

## AI Disclosure

This repository was created with substantial assistance from the opencode agent
(https://opencode.ai) running DeepSeek-V4 (model `deepseek/deepseek-v4-flash`).
All AI-generated output was reviewed and verified by the author, who takes full
responsibility for the code, results, and text.

## Citation

If you use Trackastra in your research, cite the original work:

```bibtex
@inproceedings{gallusser2024trackastra,
  title={Trackastra: Transformer-based cell tracking for live-cell microscopy},
  author={Gallusser, Benjamin and Weigert, Martin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

Key references for this project:

- **Trackastra** — Gallusser & Weigert, ECCV 2024 ([arXiv:2405.15700](https://arxiv.org/abs/2405.15700))
- **DINOv2** — Oquab et al., arXiv 2023 ([2304.07193](https://arxiv.org/abs/2304.07193))
- **HOCT** — Bragantini et al., arXiv 2026 ([2607.11754](https://arxiv.org/abs/2607.11754), higher-order cell tracking transformer)
- **NSA** — Native Sparse Attention (deep learning sparse attention kernel)
- **flash-attention** — Dao et al., 2022 (fast attention with IO awareness)

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv:2304.07193},
  year={2023}
}
```

The code and analysis in this repository were produced with the assistance of
the DeepSeek-V4 model (via the opencode agent). Cite DeepSeek as follows:

```bibtex
@misc{deepseekai2024deepseekv3technicalreport,
  title={DeepSeek-V3 Technical Report},
  author={DeepSeek-AI},
  year={2024},
  eprint={2412.19437},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2412.19437},
}
```

## License

The benchmark code in this repository is provided for research purposes. Trackastra itself is
BSD-3-Clause licensed by its authors (weigertlab/trackastra).
