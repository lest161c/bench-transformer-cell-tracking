"""Shared harness for edge-probing benchmarks.

Re-exports the public API of the registry, frame_data,
feature_extractors, edge_datasets, probes, training, evaluation,
and results submodules so benchmark scripts can import everything
from ``src.edge_probing.harness`` in one statement.

Adding a new feature type or probe architecture requires:
1. Implementing the extractor / forward function in the
   appropriate module.
2. Registering a new ``FeatureEntry`` / ``ProbeEntry`` in
   ``registry.py``.
No other harness module needs to change.
"""

from .registry import (
    FeatureEntry,
    ProbeEntry,
    FEATURE_REGISTRY,
    PROBE_REGISTRY,
    FEATURE_ALIASES,
    list_features,
    list_probes,
)
from .frame_data import (
    load_frame,
    load_tracklets,
    scan_consecutive_pairs,
    extract_patches,
    PATCH_SIZE,
)
from .feature_extractors import (
    extract_regionprops_7d, extract_regionprops_7d_fourier,
    extract_hoct19, extract_hoct19_fourier,
    compute_dino_embs,
    load_dino, ScaledCNN,
    PROJECT_ROOT, CACHE_DIR,
)
from .edge_datasets import (
    EdgePairDataset,
    EdgePairDatasetPatches,
    fit_feature_standardizer,
    apply_feature_standardizer,
    make_balanced_dataloader,
    flatten_to_pairs,
    concatenate_datasets,
    shuffle_edge_data_features,
)
from .probes import (
    LinearProbe,
    MLPProbe,
    CNNProbeE2E,
)
from .training import (
    compute_metrics,
    train_probe,
)
from .evaluation import (
    build_frame_pairs,
    evaluate_feature,
    run_cross_validation,
    TRAIN_CONDITIONS,
    VAL_CONDITIONS,
)
from .results import (
    print_results_table,
    save_results_json,
    EM_DASH,
)

__all__ = [
    # Registry
    "FeatureEntry", "ProbeEntry", "FEATURE_REGISTRY", "PROBE_REGISTRY",
    "FEATURE_ALIASES", "list_features", "list_probes",
    # Frame data
    "load_frame", "load_tracklets", "scan_consecutive_pairs",
    "extract_patches", "PATCH_SIZE",
    # Feature extractors
    "extract_regionprops_7d", "extract_regionprops_7d_fourier",
    "extract_hoct19", "extract_hoct19_fourier",
    "compute_dino_embs",
    "load_dino", "ScaledCNN", "PROJECT_ROOT", "CACHE_DIR",
    # Edge datasets
    "EdgePairDataset", "EdgePairDatasetPatches",
    "fit_feature_standardizer", "apply_feature_standardizer",
    "make_balanced_dataloader", "flatten_to_pairs",
    "concatenate_datasets", "shuffle_edge_data_features",
    # Probes
    "LinearProbe", "MLPProbe", "CNNProbeE2E",
    # Training
    "compute_metrics", "train_probe",
    # Evaluation
    "build_frame_pairs", "evaluate_feature", "run_cross_validation",
    "TRAIN_CONDITIONS", "VAL_CONDITIONS",
    # Results
    "print_results_table", "save_results_json", "EM_DASH",
]
