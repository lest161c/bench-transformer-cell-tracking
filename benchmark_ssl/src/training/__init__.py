"""SSL training scripts: pretrainer, multi-config trainer, single-config trainer."""

from src.training.ssl_pretrainer import main as run_ssl_pretrain
from src.training.ssl_trainer_multi import main as run_ssl_trainer_multi
from src.training.ssl_trainer import main as run_ssl_train
