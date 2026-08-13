"""Edge-probing benchmarks for cell tracking.

Subpackages:
    harness: Shared infrastructure (registries, frame loading, feature
        extraction, edge datasets, probes, training, evaluation, results).

Modules:
    unified_edge_probe: Unified edge-probing benchmark orchestrator.
    verify_normalization: Per-feature z-score standardization verification.
    feature_distribution_histograms: Per-feature distribution histograms.
"""

from src.edge_probing.harness import (
    FEATURE_REGISTRY,
    PROBE_REGISTRY,
    FEATURE_ALIASES,
    list_features,
    list_probes,
    EdgePairDataset,
    EdgePairDatasetPatches,
    LinearProbe,
    MLPProbe,
    CNNProbeE2E,
    compute_metrics,
    train_probe,
    evaluate_feature,
    run_cross_validation,
    build_frame_pairs,
    print_results_table,
    save_results_json,
)

__all__ = [
    "FEATURE_REGISTRY",
    "PROBE_REGISTRY",
    "FEATURE_ALIASES",
    "list_features",
    "list_probes",
    "EdgePairDataset",
    "EdgePairDatasetPatches",
    "LinearProbe",
    "MLPProbe",
    "CNNProbeE2E",
    "compute_metrics",
    "train_probe",
    "evaluate_feature",
    "run_cross_validation",
    "build_frame_pairs",
    "print_results_table",
    "save_results_json",
]
