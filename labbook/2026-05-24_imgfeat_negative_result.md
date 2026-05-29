# Image Feature SSL: Negative Result

**Date:** 2026-05-24

## Motivation

ASCENT paper (Han & Lu 2025) uses a DINO-pretrained ChannelViT on 64×64×3 image patches to encode visual identity. We tested whether similar approaches work for bacterial tracking with vanvliet data.

## Experiments

All tests on A500 laptop, 200 frames (rpsM condition), 20 epochs, batch=4:

| Backbone | Type | Output dim | Loss (epoch 1→20) | Inter-cell sim | Notes |
|----------|------|------------|---------------------|----------------|-------|
| Regionprops (7-dim MLP) | Shallow | 128 | 3.34→3.22 (flat) | 0.90 | Complete collapse |
| Scratch CNN (5 conv, 1.8M params) | Trainable | 128 | 3.58→**2.88** (↓) | 0.66-0.86 | Modest learning, best result |
| ResNet18 (ImageNet) | Frozen | 512→128 | 3.81→3.39 (flat) | 0.73-0.94 | No transfer |
| DINOv2 ViT-S/14 (ImageNet) | Frozen | 384→256 | 3.54→3.82 (flat) | 0.81-0.97 | No transfer |

## Key finding

**ImageNet-pretrained backbones do not transfer to bacterial microscopy.** This is expected — ImageNet contains natural photographs (dogs, cars, landscapes), not grayscale bacteria colonies. The visual features are domain-irrelevant.

**A trainable CNN from scratch shows modest learning** (loss 3.58→2.88, inter-sim dropping to 0.66) but is severely undertrained at 143 frames × 20 epochs. With all 6 conditions (~2600 frames) and 100+ epochs, this direction should work.

## Comparison to ASCENT

ASCENT's ChannelViT was pretrained with **DINO on ImageNet**, but their domain (C. elegans neurons) is qualitatively different from bacteria:
- Neurons: round fluorescent spots in 3D tissue, homogeneous appearance, distinctive local neighborhood of other neurons
- Bacteria: elongated cells in dense colonies, growing and dividing, less distinctive local context

ASCENT's success comes from: (1) DINO backbone for basic visual feature extraction, (2) contrastive SSL for domain-specific refinement. Step (1) works for neurons but not bacteria.

## Implication for project

The current project scope (1 semester, 1-2 days/week, regionprops features as baseline) cannot deliver working image-feature SSL. However:

1. **Architecture is correct** — synthetic test proves NT-Xent + CellEmbedder works
2. **Sparse attention is the primary contribution** — solid, measured, reproducible
3. **Image features are correctly identified as the path forward** — proposal §4.4 lists this as optional, we demonstrated the approach and its current limitations
