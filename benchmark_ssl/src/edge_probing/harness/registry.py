"""Registries for feature extractors and probe architectures.

The benchmark_attn-style harness separates ``what to benchmark`` (the
registry) from ``how to benchmark`` (the timing/data/io helpers). For
edge probing, the registry has two dimensions:

- **Feature extractors** — what per-cell representation to use as
  the probe input (regionprops, HOCT, frozen CNN, end-to-end CNN,
  DINOv2). New feature types are added by registering a new entry
  here; no other harness module needs to change.
- **Probe architectures** — the classifier head on top of the
  concatenated (feat_anchor, feat_query) edge representation (Linear, MLP).
  New probe architectures are registered the same way.

Adding a new feature type or probe to the harness is a 3-step
operation: define the entry dataclass, add it to the registry dict,
and implement the extraction / forward function in the appropriate
module (``feature_extractors.py`` or ``probes.py``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureEntry:
    """Metadata for one feature extractor registered with the harness.
    
    Attributes:
        key: Short identifier used in CLI ``--features`` and in result
            keys (e.g. ``"rp"``, ``"dino"``, ``"hoct19"``).
        display_name: Human-readable name for result tables and logs.
        feat_dim: Dimensionality of the per-cell feature vector
            produced by this extractor. Used to size the probe head.
        needs_patches: True if the extractor consumes raw image
            patches (e.g. ``cnn_e2e`` trains on patches directly).
        is_learned: True if the features are learned during probing
            (``cnn_e2e``). Learned features are exempt from z-score
            standardization and from the shuffle-baseline check.
    """
    
    key: str
    display_name: str
    feat_dim: int
    needs_patches: bool
    is_learned: bool


@dataclass(frozen=True)
class ProbeEntry:
    """Metadata for one probe architecture registered with the harness.
    
    Attributes:
        key: Short identifier used in CLI ``--probe`` (e.g. ``"linear"``,
            ``"mlp"``).
        display_name: Human-readable name for result tables and logs.
        default_lr: Default learning rate for this probe. Linear probes
            typically use a higher LR (1e-3) than MLPs (1e-4).
        default_batch_size: Default batch size for this probe.
    """
    
    key: str
    display_name: str
    default_lr: float
    default_batch_size: int


FEATURE_REGISTRY: dict[str, FeatureEntry] = {
    "rp":            FeatureEntry("rp",            "Regionprops 7D",            7,   False, False),
    "rp_fourier":    FeatureEntry("rp_fourier",    "Regionprops 7D + Fourier",  55,  False, False),
    "cnn_frozen":    FeatureEntry("cnn_frozen",    "CNN NT-Xent (frozen)",     128, False, False),
    "cnn_e2e":      FeatureEntry("cnn_e2e",      "CNN end-to-end",           128, True,  True),
    "dino":          FeatureEntry("dino",          "DINOv2 (frozen)",          384, False, False),
    "hoct19":        FeatureEntry("hoct19",        "HOCT 19D (2D \u2192 13D)", 13,  False, False),
    "hoct19_fourier": FeatureEntry("hoct19_fourier", "HOCT 13D + Fourier PE",  43,  False, False),
}


PROBE_REGISTRY: dict[str, ProbeEntry] = {
    "linear": ProbeEntry("linear", "Linear", 1e-3, 256),
    "mlp":    ProbeEntry("mlp",    "MLP",    1e-4, 128),
}


FEATURE_ALIASES: dict[str, list[str]] = {
    "all":     list(FEATURE_REGISTRY.keys()),
    "shallow": ["rp"],
    "deep":    ["cnn_frozen", "cnn_e2e", "dino"],
}


def list_features() -> list[str]:
    """Return the registry keys of all available feature extractors.
    
    Returns:
        List of feature keys in insertion order (matches ``FEATURE_REGISTRY``).
    """
    return list(FEATURE_REGISTRY.keys())


def list_probes() -> list[str]:
    """Return the registry keys of all available probe architectures.
    
    Returns:
        List of probe keys in insertion order (matches ``PROBE_REGISTRY``).
    """
    return list(PROBE_REGISTRY.keys())
