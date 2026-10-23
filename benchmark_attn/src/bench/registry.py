"""Registry mapping benchmark method keys to attention module classes.

The registry deliberately stores dotted ``class_path`` strings instead of
importing the classes at module load time, so this module has no hard import
dependency on ``src.attention_modules``. Classes are resolved lazily (and only
on demand) via :func:`resolve_class`.
"""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodEntry:
    """Metadata for one benchmarked attention method.

    Attributes:
        registry_key: CSV-facing name used in benchmark result files.
        class_path: Dotted import path of the attention class, e.g.
            "src.attention_modules.SparseGatherFusedAttention".
        needs_knn: True if the method requires precomputed KNN indices.
        needs_coords: True if the method requires spatial coordinates.
    """

    registry_key: str          # CSV-facing name
    class_path: str            # dotted import path, e.g. "src.attention_modules.SparseGatherFusedAttention"
    needs_knn: bool
    needs_coords: bool


METHOD_REGISTRY: dict[str, MethodEntry] = {
    "dense_masked": MethodEntry("dense_masked", "src.attention_modules.CachedDistAttention", needs_knn=False, needs_coords=True),
    "dense_flash": MethodEntry("dense_flash", "src.attention_modules.DenseFlashAttention", needs_knn=False, needs_coords=False),
    "sparse_gather_sdpa": MethodEntry("sparse_gather_sdpa", "src.attention_modules.SparseGatherAttention", needs_knn=True, needs_coords=True),
    "sparse_gather_fused": MethodEntry("sparse_gather_fused", "src.attention_modules.SparseGatherFusedAttention", needs_knn=True, needs_coords=True),
    "sparse_gather_matmul": MethodEntry("sparse_gather_matmul", "src.attention_modules.SparseGatherMatmulAttention", needs_knn=True, needs_coords=True),
    "sparse_mask": MethodEntry("sparse_mask", "src.attention_modules.SparseMaskAttention", needs_knn=True, needs_coords=True),
    "sparse_softmax": MethodEntry("sparse_softmax", "src.attention_modules.SparseSoftmaxAttention", needs_knn=True, needs_coords=False),
    "sparse_gather_relpos": MethodEntry("sparse_gather_relpos", "src.attention_modules.SparseGatherRelativePositionalAttention", needs_knn=True, needs_coords=True),
    "nsa": MethodEntry("nsa", "src.attention_modules.NSASparseAttention", needs_knn=False, needs_coords=False),
    "minimax": MethodEntry("minimax", "src.attention_modules.MiniMaxSparseAttention", needs_knn=False, needs_coords=False),
    "flex_attn": MethodEntry("flex_attn", "src.attention_modules.FlexAttentionModule", needs_knn=False, needs_coords=False),
}


def resolve_class(class_path: str) -> type:
    """Import and return the class at the given dotted path.

    Splits the trailing class name off ``class_path``, imports the module,
    and returns the attribute named after the class.

    Args:
        class_path: Dotted import path, e.g.
            "src.attention_modules.DenseFlashAttention".

    Returns:
        The class object. Raises ``ImportError`` (unknown module) or
        ``AttributeError`` (unknown class) if the path does not resolve.
    """
    module_name, _, class_name = class_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def list_methods() -> list[str]:
    """Return the registry keys of all available benchmark methods.

    Returns:
        List of registry keys in insertion order.
    """
    return list(METHOD_REGISTRY.keys())
