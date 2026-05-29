import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os

sns.set_theme(style="whitegrid")

base = os.path.dirname(os.path.abspath(__file__))
csvs = sorted(glob.glob(os.path.join(base, "eval_*.csv")))
df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
df["short_name"] = df["model"].str.replace(r"^2026-05-\d+_\d+-\d+-\d+_", "", regex=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=df, x="short_name", y="edge_accuracy", ax=axes[0], hue="short_name", legend=False)
axes[0].set_title("Edge Accuracy")
axes[0].set_xlabel("")
axes[0].set_ylim(0.97, 1.0)
for i, v in enumerate(df["edge_accuracy"]):
    axes[0].text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=9)

sns.barplot(data=df, x="short_name", y="bce_loss", ax=axes[1], hue="short_name", legend=False)
axes[1].set_title("BCE Loss")
axes[1].set_xlabel("")
for i, v in enumerate(df["bce_loss"]):
    axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

fig.suptitle("Evaluation Results", fontsize=14)
plt.tight_layout()
out = os.path.join(base, "eval_plot.png")
plt.savefig(out, dpi=150)
print(f"Saved plot to {out}")
