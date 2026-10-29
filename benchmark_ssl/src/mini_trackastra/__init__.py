"""Mini trackastra experiments: the end-to-end SSL prototypes and pretext probes.

`end_to_end` backs the live pretext probe chain (sample-efficiency test,
bug-exclusion audit, per-anchor objective probe); `end_to_end_bce` +
`src/analysis/pretext_failure_diagnosis` back the SPEC 0048 failure
diagnosis. The superseded prototypes (temporal_ssl, compare_backbones) were
removed on 2026-10-29.
"""

from src.mini_trackastra.end_to_end import main as run_end_to_end
from src.mini_trackastra.end_to_end_bce import main as run_end_to_end_bce

