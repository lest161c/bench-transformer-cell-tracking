"""Mini trackastra experiments: end-to-end SSL, temporal SSL, backbone tests."""

from src.mini_trackastra.compare_backbones import main as run_compare_backbones
from src.mini_trackastra.end_to_end import main as run_end_to_end
from src.mini_trackastra.end_to_end_bce import main as run_end_to_end_bce
from src.mini_trackastra.temporal_ssl import main as run_temporal_ssl
from src.mini_trackastra.test_barlow import main as run_test_barlow
from src.mini_trackastra.test_cell_dino import main as run_test_cell_dino
from src.mini_trackastra.test_freeze import main as run_test_freeze
