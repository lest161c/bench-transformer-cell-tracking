"""Shared benchmark harness package for ``benchmark_attn``.

Re-exports the public API of the timing, data, registry, io and cli
submodules so benchmark scripts can import everything from ``src.bench`` in
one statement.
"""

from .timing import measure, try_bench
from .data import make_inputs, compute_knn_indices
from .registry import METHOD_REGISTRY, MethodEntry, resolve_class, list_methods
from .io import write_results_csv
from .cli import add_common_args, parse_int_list

__all__ = [
    "measure",
    "try_bench",
    "make_inputs",
    "compute_knn_indices",
    "METHOD_REGISTRY",
    "MethodEntry",
    "resolve_class",
    "list_methods",
    "write_results_csv",
    "add_common_args",
    "parse_int_list",
]
