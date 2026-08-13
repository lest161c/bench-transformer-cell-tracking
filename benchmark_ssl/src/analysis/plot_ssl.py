"""Seaborn viz → single HTML. Sources SSL training log CSV.

Reads the per-epoch SSL pretraining metrics from ``runs/ssl_v1/training_log.csv``,
renders loss / accuracy / F1 / precision-recall / timing figures plus an
improvement summary table, and appends a downstream SSL-vs-random-init
comparison when ``runs/downstream_compare/comparison.csv`` exists.
Writes a single self-contained ``src.html`` with base64-embedded PNGs.
"""

import csv, io, base64
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")


def fig_to_b64(fig):
    """Encode a matplotlib Figure as a base64 PNG payload.

    Args:
        fig: matplotlib Figure to serialize.

    Returns:
        str: base64-encoded PNG bytes (without any data-URI prefix).
        Called by main() to embed each figure into the HTML report.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def main():
    """Build the SSL benchmark HTML report from the training log CSV.

    Loads ``runs/ssl_v1/training_log.csv``, creates the six per-epoch metric
    figures plus an improvement summary table, appends the downstream
    SSL-vs-random-init comparison figures if the comparison CSV exists, and
    writes ``src.html`` with all figures embedded as base64 PNGs.
    Called by the __main__ guard so that importing the module has no side
    effects.
    """
    # --- load ---
    rows = []
    with open("runs/ssl_v1/training_log.csv") as file_handle:
        for row in csv.DictReader(file_handle):
            for column in row:
                try:
                    row[column] = float(row[column])
                except ValueError:
                    pass
            rows.append(row)
    df = pd.DataFrame(rows)
    epochs = df["epoch"].values

    figures = []

    # ===== 1. Loss convergence (train + val) =====
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="epoch", y="train_loss", marker="o", label="Train Loss", ax=ax)
    sns.lineplot(data=df, x="epoch", y="val_loss", marker="s", label="Val Loss", ax=ax)
    ax.set_title("SSL Pretraining: Loss Convergence (5 epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss (pos_weight=10)")
    ax.legend()
    ax.set_xticks(epochs)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # ===== 2. Accuracy convergence =====
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="epoch", y="train_acc", marker="o", label="Train Acc", ax=ax)
    sns.lineplot(data=df, x="epoch", y="val_acc", marker="s", label="Val Acc", ax=ax)
    ax.set_title("SSL Pretraining: Association Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_xticks(epochs)
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # ===== 3. F1 score convergence =====
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=df, x="epoch", y="train_f1", marker="o", label="Train F1", ax=ax)
    sns.lineplot(data=df, x="epoch", y="val_f1", marker="s", label="Val F1", ax=ax)
    ax.set_title("SSL Pretraining: F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend()
    ax.set_xticks(epochs)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # ===== 4. Precision vs Recall =====
    if "val_prec" in df.columns and "val_rec" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=df, x="epoch", y="val_prec", marker="o", label="Val Precision", ax=ax)
        sns.lineplot(data=df, x="epoch", y="val_rec", marker="s", label="Val Recall", ax=ax)
        ax.set_title("SSL Pretraining: Precision vs Recall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_xticks(epochs)
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)

    # ===== 5. Epoch time =====
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df, x="epoch", y="time_s", color="steelblue", ax=ax)
    ax.set_title("Training Time per Epoch (RTX A500 Laptop, 4GB)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.grid(True, ls="--", alpha=0.3, axis="y")
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # ===== 6. Improvement summary table =====
    first = df.iloc[0]
    last = df.iloc[-1]
    summary = pd.DataFrame({
        "Metric": ["Train Loss", "Val Loss", "Train Acc", "Val Acc", "Train F1", "Val F1"],
        "Epoch 1": [
            f"{first['train_loss']:.4f}", f"{first['val_loss']:.4f}",
            f"{first['train_acc']:.4f}", f"{first['val_acc']:.4f}",
            f"{first['train_f1']:.4f}", f"{first['val_f1']:.4f}",
        ],
        f"Epoch {int(last['epoch'])}": [
            f"{last['train_loss']:.4f}", f"{last['val_loss']:.4f}",
            f"{last['train_acc']:.4f}", f"{last['val_acc']:.4f}",
            f"{last['train_f1']:.4f}", f"{last['val_f1']:.4f}",
        ],
        "Change": [
            f"{last['train_loss']-first['train_loss']:+.4f}",
            f"{last['val_loss']-first['val_loss']:+.4f}",
            f"{last['train_acc']-first['train_acc']:+.4f}",
            f"{last['val_acc']-first['val_acc']:+.4f}",
            f"{last['train_f1']-first['train_f1']:+.4f}",
            f"{last['val_f1']-first['val_f1']:+.4f}",
        ],
    })
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    table = ax.table(cellText=summary.values, colLabels=summary.columns,
                     cellLoc="center", loc="center", colWidths=[0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for j, val in enumerate(summary["Change"]):
        cell = table[(j + 1, 3)]
        if val.startswith("-"):
            cell.set_facecolor("#c8e6c9")  # green = improvement
        elif val.startswith("+"):
            cell.set_facecolor("#ffcdd2")  # red = regression
    ax.set_title("SSL Pretraining: Improvement Summary", fontsize=13, pad=20)
    fig.tight_layout()
    figures.append(fig)
    plt.close(fig)

    # --- combine into HTML ---
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>SSL Pretraining Benchmark (ASCENT-inspired)</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#fafafa}",
        "h1{color:#333} h2{color:#555} img{max-width:100%;margin:20px 0;border:1px solid #ddd;border-radius:6px}</style>",
        "</head><body>",
        "<h1>SSL Pretraining via Geometric Distortion</h1>",
        "<p>ASCENT-inspired encoder on vanvliet dataset (rpsM/recA/pheA). "
        "5 epochs, RTX A500 Laptop (4GB).</p>",
    ]

    for i, fig in enumerate(figures):
        b64 = fig_to_b64(fig)
        html_parts.append(f"<figure><figcaption>Figure {i+1}</figcaption>")
        html_parts.append(f'<img src="data:image/png;base64,{b64}" /></figure>')

    # ===== 7. Downstream: SSL vs Random init — Val Loss =====
    try:
        cmp = pd.read_csv("runs/downstream_compare/comparison.csv")
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in ["ssl", "rand"]:
            sub = cmp[cmp["model"] == model]
            sns.lineplot(data=sub, x="epoch", y="val_loss", marker="o",
                         label=f"{'SSL-pretrained' if model == 'ssl' else 'Random init'}", ax=ax)
        ax.set_title("Downstream: SSL-Pretrained vs Random Init (Val Loss)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Loss")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)

        # ===== 8. Downstream: Train Loss =====
        fig, ax = plt.subplots(figsize=(9, 5))
        for model in ["ssl", "rand"]:
            sub = cmp[cmp["model"] == model]
            sns.lineplot(data=sub, x="epoch", y="train_loss", marker="o",
                         label=f"{'SSL-pretrained' if model == 'ssl' else 'Random init'}", ax=ax)
        ax.set_title("Downstream: SSL-Pretrained vs Random Init (Train Loss)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)

        # ===== 9. Downstream: Final loss comparison bar =====
        final = cmp[cmp["epoch"] == cmp["epoch"].max()]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(final))
        colors = ["#2ecc71", "#e74c3c"]
        labels = {"ssl": "SSL-pretrained", "rand": "Random init"}
        for i, (_, row) in enumerate(final.iterrows()):
            ax.bar(i, row["val_loss"], color=colors[i], label=labels[row["model"]], width=0.5)
        ax.set_xticks([])
        ax.set_ylabel("Final Val Loss")
        ax.set_title(f"Final Val Loss after {int(final['epoch'].iloc[0])} epochs")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3, axis="y")
        # Annotate values
        for i, (_, row) in enumerate(final.iterrows()):
            ax.text(i, row["val_loss"] + 0.001, f"{row['val_loss']:.4f}", ha="center", fontsize=11)
        fig.tight_layout()
        figures.append(fig)
        plt.close(fig)
    except FileNotFoundError:
        pass

    html_parts.append("</body></html>")

    with open("src.html", "w") as file_handle:
        file_handle.write("\n".join(html_parts))

    print(f"Saved src.html with {len(figures)} figures")


if __name__ == "__main__":
    main()
