"""Analysis scripts: signal/diagnostic era (bannered historical) and the live
pretext probe family.

The May-August diagnostic scripts (signal_analyzer, convergence_predictor,
downstream_convergence, coord_shortcut_diagnostic, downstream_comparison,
dino_test, cnn_embedding_pca, reproduce_f04, projection_weight_analysis)
were removed on 2026-10-29; their numbers live in the labbook and the
bannered `docs/REPRODUCTION.md` §4.
"""

from src.analysis.end_to_end_diagnostic import main as run_end_to_end_diagnostic
from src.analysis.dino_backbone_comparison import main as run_dino_backbone_comparison
from src.analysis.feature_probe import main as run_feature_probe

# Plot scripts migrated by the ancillary agent (T6); may not exist yet.
from src.analysis.plot_ssl import main as run_plot_ssl
from src.analysis.plot_expected_metrics import main as run_plot_expected_metrics
