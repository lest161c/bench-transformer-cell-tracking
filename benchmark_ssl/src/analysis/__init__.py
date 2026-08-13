"""Analysis and diagnostic scripts: signal, convergence, downstream, shortcuts, DINO comparison."""

from src.analysis.signal_analyzer import main as run_signal_analyzer
from src.analysis.convergence_predictor import main as run_convergence_predictor
from src.analysis.downstream_convergence import main as run_downstream_convergence
from src.analysis.coord_shortcut_diagnostic import main as run_coord_shortcut_diagnostic
from src.analysis.end_to_end_diagnostic import main as run_end_to_end_diagnostic
from src.analysis.dino_backbone_comparison import main as run_dino_backbone_comparison
from src.analysis.downstream_comparison import main as run_downstream_comparison
from src.analysis.feature_probe import main as run_feature_probe
from src.analysis.dino_test import main as run_dino_test

# Plot scripts migrated by the ancillary agent (T6); may not exist yet.
from src.analysis.plot_ssl import main as run_plot_ssl
from src.analysis.plot_expected_metrics import main as run_plot_expected_metrics
