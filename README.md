# Bench Transformer Cell Tracking

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![ECCV 2024](https://img.shields.io/badge/ECCV-2024-4b44ce.svg)](https://github.com/weigertlab/trackastra)
[![arXiv](https://img.shields.io/badge/arXiv-2405.15700-b31b1b.svg)](https://arxiv.org/abs/2405.15700)

A benchmark suite for efficient attention and feature-injection methods in transformer-based cell tracking, built on [Trackastra](https://github.com/weigertlab/trackastra) (ECCV 2024). It evaluates sparse attention architectures (dense, gather-/mask-KNN, FlashAttention, NSA) and CNN feature encoders across tracking accuracy, convergence, and throughput.

---

## Benchmark Code

| Directory | Description |
|---|---|
| `benchmark_attn/` | Standalone sparse attention micro-benchmarks (dense vs gather vs mask-KNN vs spatial-flash) |
| `benchmark_ssl/` | SSL pretraining analysis, signal tests, edge probing, cross-dataset downstream evaluation |
| `benchmark_training/` | End-to-end training benchmarks, convergence experiments, K-sweep harness, multi-seed configs |
| `configs/` | YAML configs for all training runs (baseline, CNN variants, lambda-schedule) |

---

## Canonical Configuration Names

Every attention configuration has exactly **one canonical name: the CamelCase
class name** in `benchmark_attn/src/attention_modules.py`. Use it in code,
configs, report text, figures, and commit messages. The snake_case registry
keys (`benchmark_attn/src/bench/registry.py`) exist only as CLI/CSV
identifiers and map 1:1 to the canonical names — do not mix the two systems
in the same document. Do not invent prose synonyms for configurations
(e.g. "dense masked baseline", "gather family", "KNN-restricted mask
variant", "dense flash"); name the configuration canonically and, on first
use, add a plain-language description next to it.

| Canonical name | Registry key | Meaning |
|---|---|---|
| `RelativePositionalAttention` | — | Stock Trackastra baseline: dense masked SDPA, pairwise distances recomputed per layer |
| `CachedDistAttention` | `dense_masked` | Dense masked SDPA, pairwise distances computed once and shared across layers |
| `DenseFlashAttention` | `dense_flash` | Dense SDPA without any mask → FlashAttention-2 backend |
| `SparseGatherRelativePositionalAttention` | `sparse_gather_relpos` | Dense masked SDPA restricted to the K nearest neighbours, with relative-position bias |
| `SparseGatherAttention` | `sparse_gather_sdpa` | SparseGather: gathered K/V rows, SDPA on 1×K tensors |
| `SparseGatherFusedAttention` | `sparse_gather_fused` | SparseGather, fused-kernel backend |
| `SparseGatherMatmulAttention` | `sparse_gather_matmul` | SparseGather, explicit matmul backend |
| `SparseMaskAttention` | `sparse_mask` | SparseMask: scatter-built KNN mask, single masked SDPA call |
| `SparseSoftmaxAttention` | `sparse_softmax` | Softmax restricted to the K gathered neighbours (backward benchmark only) |
| `MiniMaxSparseAttention` | `minimax` | Block-level index → top-k blocks → matmul on selected KV |
| `NSASparseAttention` | `nsa` | Native Sparse Attention port |
| `FlexAttentionModule` | `flex_attn` | PyTorch FlexAttention (`torch.nn.attention.flex_attention`, compiled block mask) |

Vendor and product names keep their official spelling and are not aliases of
our configurations: **FlashAttention**, **FlashAttention-2**, **FlexAttention**,
**Native Sparse Attention (NSA)**, and **EfficientAttention** (the PyTorch
memory-efficient SDPA backend).

---

## Results

Detailed results are kept in the per-experiment documentation next to the code:

| Area | Where results live |
|---|---|
| Sparse attention micro-benchmarks (DenseFlashAttention vs SparseGatherAttention vs SparseMaskAttention, CachedDistAttention, backward, speed/memory) | `benchmark_attn/REPRODUCTION.md` |
| CNN feature injection, edge probing, DeepCell cross-dataset | `benchmark_ssl/docs/REPRODUCTION.md` |
| Clean K-sweep training (convergence, TRA/AOGM) | `benchmark_training/docs/REPRODUCTION.md` |

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
Code refactoring and repository restructuring were additionally assisted by
GLM-5.2 (model `scads/zai-org/GLM-5.2-FP8`), hosted on the SCADS.AI server. Later
report rework and the final experimental campaign (experiments, statistical
significance testing, report integration) were assisted by DeepSeek-V4.1
(model `deepseek/deepseek-v4.1-flash`) and GLM-5.3-Flash (model
`scads/zai-org/GLM-5.3-Flash`), the latter also hosted on the SCADS.AI server.
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
- **DINOv2** — Oquab et al., TMLR 2024 ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193))
- **HOCT** — Bragantini et al., arXiv 2026 ([2607.11754](https://arxiv.org/abs/2607.11754), higher-order cell tracking transformer)
- **NSA** — Native Sparse Attention (deep learning sparse attention kernel)
- **flash-attention** — Dao et al., 2022 (fast attention with IO awareness)

```bibtex
@article{oquab2024dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2024},
  note={arXiv:2304.07193}
}
```

The code and analysis in this repository were produced with the assistance of
the DeepSeek-V4 model (via the opencode agent) and the GLM-5.2 model (hosted on
the SCADS.AI server). Cite DeepSeek as follows:

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
