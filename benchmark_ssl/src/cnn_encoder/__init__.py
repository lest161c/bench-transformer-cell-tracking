"""CNN-specific pretraining, convergence tests, cross-dataset evaluation."""

from src.cnn_encoder.cnn_ssl import main as run_cnn_ssl
from src.cnn_encoder.cnn_convergence import main as run_cnn_convergence
from src.cnn_encoder.h100_convergence import main as run_h100_convergence
from src.cnn_encoder.cross_dataset_eval import main as run_cross_dataset_eval
from src.cnn_encoder.scale_sweep import main as run_scale_sweep
from src.cnn_encoder.visualize_embeddings import main as run_visualize_embeddings
from src.cnn_encoder.smoke_test import main as run_smoke_test
