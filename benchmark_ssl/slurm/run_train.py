"""Register the DynaCLR backbone, then run the fork's training script.

Why: the fork's ``scripts/train.py`` builds its ``--pretrained_feats_model``
argparse ``choices`` from ``AVAILABLE_PRETRAINED_BACKBONES``, so our bridge must
be imported before that script runs. Running ``train.py`` through this wrapper
guarantees the registration without editing the fork checkout.

Usage: run this from the fork checkout with this repo's ``src`` on ``PYTHONPATH``:
    python benchmark_ssl/slurm/run_train.py -c <config> --seed <n>
"""

from pathlib import Path

import etultra_dynaclr.fork_registry_bridge  # noqa: F401  registers dynaclr-scaledcnn

_TRAIN_SCRIPT = Path("scripts/train.py")


def main() -> None:
    """Run the fork's trainer in-process after the bridge has registered."""
    if not _TRAIN_SCRIPT.exists():
        raise SystemExit(
            f"{_TRAIN_SCRIPT} not found; run this wrapper from the fork checkout "
            f"(cwd={Path.cwd()})."
        )
    import runpy

    runpy.run_path(str(_TRAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
