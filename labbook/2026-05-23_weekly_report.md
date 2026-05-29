# Weekly Report: May 19-23

**KNN attention benchmark.** Swept N∈[128,8192], K∈[4,64] on RTX 4090. KNN slower than dense at N<1024 due to cdist+topk overhead. Crossover at N≈2048. At N=8192 dense OOMs, KNN fits. Token reordering re-checked — gain stays <3%.

**Combined ablation.** Tested KNN vs dense on synthetic identity association (d=128, L=4, A500). KNN K=4 gives lower val_loss (0.108 vs 0.504 dense) — but this is a toy model, unclear if it translates to full Trackastra (d=320, L=12). Old SSL pipeline adds nothing on top of KNN at this scale.

**SSL pipeline.** Current implementation uses wrong ASCENT paper. Needs rewrite to contrastive InfoNCE. Whether SSL actually helps downstream remains unproven — the synthetic proxy is encouraging but the real test is on Trackastra training.

**Feature concern.** 7-dim regionprops may be too shallow for contrastive SSL — cells have near-identical features, little signal to discriminate. Image patches tested briefly but inconclusive. Needs verification.

**Next:** Rewrite SSL architecture, measure on H100, verify with Trackastra training.
