"""CSV writing helpers for the benchmark harness.

Extracted from the duplicated result-writing loops in the benchmark scripts
(``benchmarks/benchmark_full.py``, ``benchmarks/benchmark_gather_v3.py``,
``benchmarks/benchmark_cached_dist.py``), which all opened the output file
with ``newline=""``, wrote a header row, and padded short result rows to a
uniform column count.
"""

import csv
from pathlib import Path


def write_results_csv(rows: list[list], out_path: Path, header: list[str]) -> None:
    """Write benchmark results to a CSV file with consistent formatting.

    Writes the header row followed by each data row. Short rows are padded
    with empty cells up to the header width so every row has a uniform column
    count; cells beyond the header width are kept as-is.

    Args:
        rows: List of result rows, each a list of cell values.
        out_path: Destination file path (str or pathlib.Path).
        header: Column names for the first row.
    """
    with open(out_path, "w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(header)
        for row in rows:
            padded_row = list(row) + [""] * max(0, len(header) - len(row))
            writer.writerow(padded_row)
