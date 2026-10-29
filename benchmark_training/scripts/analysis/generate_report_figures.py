"""Regenerate the corrected report figures directly into ``report-shorten/figs``.

Run from the repository root with the project Python environment::

    python3 report-shorten/scripts/generate_report_figures.py

The panel functions live in :mod:`curated.generate_report` and read the
benchmark CSVs from the repository data directories. This wrapper redirects
the generator's ``FIGS`` output directory to ``report-shorten/figs`` and
regenerates only the three panels corrected by ticket W6.4: the H100 speed
and detail figures, the H100 memory figure, and the distance-cache effect
figure, plus the bootstrapped multi-seed K-sweep metrics figure
(:mod:`generate_knn_sweep_bootstrap_figure`, skipped with a warning when the
mirrored per-sequence eval tree is absent). Every figure writes its PDF, SVG,
and ``.meta.json`` sidecar directly into the target directory (no copy or
move step). Running the script twice with the same inputs reproduces the same
figure content apart from the sidecar timestamp.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from curated import generate_report

REPORT_FIGURES_DIR = REPOSITORY_ROOT / 'report-shorten' / 'figs'


def main():
    """Regenerate the W6.4 figure set into ``report-shorten/figs``."""
    generate_report.FIGS = REPORT_FIGURES_DIR
    REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_report.panel_speed_vs_n_h100()
    generate_report.panel_memory_vs_n()
    generate_report.panel_dist_cache_effect()
    try:
        import generate_knn_sweep_bootstrap_figure
        generate_knn_sweep_bootstrap_figure.main()
    except FileNotFoundError as error:
        print(f'  ⚠ skipping knn_sweep_metrics_bootstrap: {error}')


if __name__ == '__main__':
    main()
