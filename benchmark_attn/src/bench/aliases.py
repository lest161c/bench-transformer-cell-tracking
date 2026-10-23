"""Old -> new attention-config name aliases (SPEC 0031 dense/sparse rename).

Why: the rename to the dense/sparse vocabulary (DenseFlash, DenseMasked,
SparseGather, SparseMask, with backend suffixes) changes identifiers, but frozen
artifacts (result CSVs, run directories, archived `.v1` CSVs) keep the old names.
Parsers and figure generators normalize old names through this map instead of
migrating those files, so no series is silently dropped or grey-fallbacked.

Usage::

    from src.bench.aliases import normalize_method
    method = normalize_method(row["method"])   # old and new both resolve

`CachedDistAttention` and the `knn_neighbors` flag semantics are unchanged and
are not aliased; external methods (flex/minimax/nsa) are unchanged.
"""

# Old registry keys -> new registry keys (SPEC 0031 §1).
OLD_TO_NEW_METHOD = {
    "gather_sdpa": "sparse_gather_sdpa",
    "gather_fused": "sparse_gather_fused",
    "gather_matmul": "sparse_gather_matmul",
    "mask_knn": "sparse_mask",
    "knn_relpos": "sparse_gather_relpos",
}

# Old class names -> new class names (code identifiers).
OLD_TO_NEW_CLASS = {
    "GatherSparseAttention": "SparseGatherAttention",
    "GatherSparseFusedAttention": "SparseGatherFusedAttention",
    "GatherSparseMatmulAttention": "SparseGatherMatmulAttention",
    "KNNMaskSparseAttention": "SparseMaskAttention",
    "KNNRelativePositionalAttention": "SparseGatherRelativePositionalAttention",
}


def normalize_method(name: str) -> str:
    """Return the canonical (new) method key for an old or new name.

    Unknown names are returned unchanged, so callers that then look the name up
    in a colour/label table should treat a miss explicitly rather than silently.
    """
    return OLD_TO_NEW_METHOD.get(name, name)
