"""benchmark_ssl: SSL pretraining benchmarks for transformer-based cell tracking.

Subpackages:
    data: Feature extraction, frame loading, and SSL dataset utilities.
    distortions: Augmentation pipeline (affine, elastic, jitter, dropout, photometric, feature_noise).
    models: CellEmbedder, attention modules (RoPE, KNN-sparse, gather-sparse, relative-bias).
    training: SSL pretraining and multi-config SSL training.
    analysis: Signal analysis, convergence prediction, downstream convergence, diagnostics.
    edge_probing: Edge probing framework (unified 5-fold CV edge probe, CNN probe training).
    cnn_encoder: CNN-specific pretraining, convergence tests, cross-dataset evaluation.
    mini_trackastra: Mini trackastra experiments (end-to-end SSL, temporal SSL, backbone tests).
"""
