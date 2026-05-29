#!/usr/bin/env python3
import argparse
import logging
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval")

sys.path.insert(0, str(Path(__file__).parent / "trackastra"))

import numpy as np
import pandas as pd

from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, BasicMetrics, DivisionMetrics


def find_experiments(data_dir):
    exps = []
    for gene_dir in sorted(data_dir.iterdir()):
        if not gene_dir.is_dir():
            continue
        for exp_dir in sorted(gene_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if "_BAD" in exp_dir.name:
                continue
            if (exp_dir / "img").is_dir() and (exp_dir / "TRA").is_dir():
                exps.append(exp_dir)
    return exps


def flatten_results(metric_results, ckpt_name, exp_name):
    row = {"checkpoint": ckpt_name, "experiment": exp_name}
    for r in metric_results:
        mname = r["metric"]["name"]
        res = r["results"]
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        row[f"{mname}_{k}_{k2}"] = v2
                else:
                    row[f"{mname}_{k}"] = v
    return row


def evaluate_checkpoint(ckpt_dir, experiments, device="cuda"):
    logger.info(f"Loading model from {ckpt_dir.name}")
    model = Trackastra.from_folder(ckpt_dir, device=device)
    model.transformer.eval()

    rows = []
    for exp_dir in experiments:
        exp_name = f"{exp_dir.parent.name}/{exp_dir.name}"
        logger.info(f"  Run {exp_name}")
        try:
            track_graph, masks_tracked = model.track_from_disk(
                imgs_path=exp_dir / "img",
                masks_path=exp_dir / "TRA",
                mode="greedy",
            )

            with tempfile.TemporaryDirectory() as tmp:
                graph_to_ctc(track_graph, masks_tracked, outdir=tmp)

                gt_data = load_ctc_data(
                    str(exp_dir / "TRA"), name="GT"
                )
                pred_data = load_ctc_data(tmp, name=ckpt_dir.name)

                res_list, _matched = run_metrics(
                    gt_data=gt_data,
                    pred_data=pred_data,
                    matcher=CTCMatcher(),
                    metrics=[CTCMetrics(), BasicMetrics(), DivisionMetrics()],
                )

                rows.append(flatten_results(res_list, ckpt_dir.name, exp_name))
        except Exception as e:
            logger.error(f"  FAIL {exp_name}: {e}")
            rows.append({"checkpoint": ckpt_dir.name, "experiment": exp_name, "error": str(e)})

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--html", default="results.html")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    ckpt_dir = Path(args.checkpoints) if args.checkpoints else base / "checkpoints"
    data_dir = Path(args.data) if args.data else base / "data" / "vanvliet"

    logger.info(f"Checkpoints: {ckpt_dir}")
    logger.info(f"Data: {data_dir}")

    if (ckpt_dir / "model.pt").exists():
        ckpt_dirs = [ckpt_dir]
    else:
        ckpt_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and (d / "model.pt").exists()])
    logger.info(f"Found {len(ckpt_dirs)} checkpoints: {[d.name for d in ckpt_dirs]}")

    exps = find_experiments(data_dir)
    logger.info(f"Found {len(exps)} experiments")

    all_rows = []
    for ckpt in ckpt_dirs:
        logger.info(f"\n=== {ckpt.name} ===")
        all_rows.extend(evaluate_checkpoint(ckpt, exps, device=args.device))

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    logger.info(f"CSV saved: {args.output}")

    # Seaborn HTML
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import io
        import base64

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        key_cols = [c for c in numeric_cols if any(k in c for k in ["TRA", "DET", "LNK", "Node F1", "Edge F1", "Division F1"])]

        if key_cols:
            n = len(key_cols)
            fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
            if n == 1:
                axes = [axes]
            for ax, col in zip(axes, key_cols):
                sns.barplot(data=df, x="experiment", y=col, hue="checkpoint", ax=ax)
                ax.tick_params(axis="x", rotation=90)
                ax.set_ylabel(col, fontsize=12)
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                vmin = df[col].min()
                vmax = df[col].max()
                margin = max((vmax - vmin) * 0.1, 0.0005)
                ax.set_ylim(vmin - margin, min(vmax + margin, 1.0))
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            plt.close()

            table_html = df.round(4).to_html()
            html = f"""<!DOCTYPE html>
<html><head><title>Eval Results</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #eee; }}
img {{ max-width: 100%; }}
</style></head>
<body>
<h1>Tracking Evaluation Results</h1>
<img src="data:image/png;base64,{img_b64}">
<hr><h2>Full Table</h2>
{table_html}
</body></html>"""
            Path(args.html).write_text(html)
            logger.info(f"HTML saved: {args.html}")
    except Exception as e:
        logger.warning(f"Could not generate HTML: {e}")


if __name__ == "__main__":
    main()
