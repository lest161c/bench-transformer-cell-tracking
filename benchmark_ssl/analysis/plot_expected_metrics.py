"""Generate **expected / fake** performance metrics for sparse-attention + SSL run.

Purpose: provides reference estimates to compare against real run.
All curves are synthetic — clearly labeled as "expected" throughout.
"""

import argparse
import io, base64
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

OUT = Path(__file__).resolve().parents[1] / "expected_metrics.html"
N_EPOCHS = 100
N_SSL = 5


def main(seed: int = 42) -> None:
    """Generate the synthetic "expected metrics" HTML report.

    Builds fake loss, timing, and VRAM curves using a numpy RNG seeded by
    `seed`, renders the six matplotlib figures, and writes a single
    self-contained HTML page (base64-embedded PNGs) to OUT.
    Called by the __main__ guard so that importing the module has no side
    effects; pass `--seed` to vary the synthetic data.
    """
    np.random.seed(seed)

    # ============================================================
    # 1. Expected loss convergence
    # ============================================================
    # Typical Trackastra loss: starts ~1-2, converges to ~0.05-0.2
    # SSL pretraining gives lower initial loss
    rng = np.random.default_rng(seed)

    epochs = np.arange(1, N_EPOCHS + 1)

    # Expected train loss: exponential decay + noise
    base_train = 1.8 * np.exp(-0.035 * epochs) + 0.08
    noise_train = rng.normal(0, 0.03 * np.exp(-0.02 * epochs), N_EPOCHS)
    train_loss = base_train + noise_train
    train_loss = np.clip(train_loss, 0.01, None)

    # Expected val loss: slightly higher, more noise
    base_val = 2.0 * np.exp(-0.032 * epochs) + 0.12
    noise_val = rng.normal(0, 0.04 * np.exp(-0.015 * epochs), N_EPOCHS)
    val_loss = base_val + noise_val
    val_loss = np.clip(val_loss, 0.02, None)

    # SSL pre-training loss (first 5 epochs)
    ssl_epochs = np.arange(1, N_SSL + 1)
    ssl_loss = 2.5 * np.exp(-0.4 * ssl_epochs) + 0.15
    ssl_noise = rng.normal(0, 0.015, N_SSL)
    ssl_loss = ssl_loss + ssl_noise

    # ============================================================
    # 2. Timing estimates
    # ============================================================
    # Based on A100, sparse k=16, batch=48, max_tokens=4096
    # SSL epochs: faster (single-frame pairs with distortion)
    ssl_time_per_epoch = rng.normal(180, 15, N_SSL)  # ~3 min
    ssl_time_per_epoch = np.clip(ssl_time_per_epoch, 150, 250)

    # Main training: slower as model learns more complex associations
    time_per_epoch_base = 420 * np.exp(-0.008 * epochs) + 300  # ~5-7 min
    time_per_epoch = time_per_epoch_base + rng.normal(0, 15, N_EPOCHS)
    time_per_epoch = np.clip(time_per_epoch, 240, 600)

    # ============================================================
    # 3. VRAM estimates
    # ============================================================
    # Sparse (k=16): ~28-35GB peak on A100 with batch=48
    # SSL: smaller peak due to fewer tokens per sample
    ssl_vram = rng.normal(22, 1.5, N_SSL)
    ssl_vram = np.clip(ssl_vram, 18, 28)

    vram_base = 32 - 4 * np.exp(-0.02 * epochs)  # slight decrease as model stabilizes
    vram_peak = vram_base + rng.normal(0, 1.5, N_EPOCHS)
    vram_peak = np.clip(vram_peak, 24, 40)

    # ============================================================
    # Build figures
    # ============================================================
    figures = []

    # ---- 1. Loss convergence ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, color="tab:blue", lw=1.5, label="Expected train loss")
    ax.plot(epochs, val_loss, color="tab:orange", lw=1.5, label="Expected val loss")
    ax.axvspan(0, N_SSL, alpha=0.08, color="green", label=f"SSL pretrain ({N_SSL} epochs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Expected loss convergence   [FAKE / synthetic estimate]")
    ax.legend()
    ax.set_xlim(1, N_EPOCHS)
    fig.text(0.5, -0.02, "Fake data — do not cite. Use as reference only.",
             ha="center", fontsize=9, fontstyle="italic", color="gray",
             transform=ax.transAxes)
    fig.tight_layout()
    figures.append(("Loss convergence", fig))

    # ---- 2. SSL loss detail ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ssl_epochs, ssl_loss, color="tab:green", lw=2, marker="o")
    ax.set_xlabel("SSL epoch")
    ax.set_ylabel("SSL loss")
    ax.set_title("Expected SSL pretraining loss   [FAKE / synthetic]")
    ax.set_xticks(ssl_epochs)
    fig.text(0.5, -0.02, "Fake data — reference only",
             ha="center", fontsize=9, fontstyle="italic", color="gray",
             transform=ax.transAxes)
    fig.tight_layout()
    figures.append(("SSL loss", fig))

    # ---- 3. Time per epoch ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, time_per_epoch / 60, color="tab:purple", lw=1.5)
    ax.axvspan(0, N_SSL, alpha=0.08, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time per epoch (min)")
    ax.set_title("Expected time per epoch   [FAKE / synthetic]")
    # add SSL epoch overlay
    ssl_t_min = ssl_time_per_epoch / 60
    for i, time_min in enumerate(ssl_t_min):
        ax.annotate(f"{time_min:.1f}", (ssl_epochs[i], time_min), fontsize=7,
                    ha="center", va="bottom", color="green")
    fig.text(0.5, -0.02, "Fake data — reference only",
             ha="center", fontsize=9, fontstyle="italic", color="gray",
             transform=ax.transAxes)
    fig.tight_layout()
    figures.append(("Time per epoch", fig))

    # ---- 4. Total time breakdown ----
    total_ssl_s = ssl_time_per_epoch.sum()
    total_main_s = time_per_epoch.sum()
    total_s = total_ssl_s + total_main_s
    labels = [f"SSL pretrain ({total_ssl_s/60:.0f} min)", f"Main training ({total_main_s/60:.0f} min)"]
    colors = ["tab:green", "tab:blue"]

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        [total_ssl_s, total_main_s], labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90, explode=(0.05, 0),
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    ax.set_title(f"Expected total time: {total_s/3600:.1f} hours   [FAKE / synthetic]")
    fig.text(0.5, -0.02, "Fake — reference only",
             ha="center", fontsize=9, fontstyle="italic", color="gray",
             transform=ax.transAxes)
    fig.tight_layout()
    figures.append(("Total time breakdown", fig))

    # ---- 5. VRAM usage ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, vram_peak, color="tab:red", lw=1.5, label="Expected GPU VRAM (peak)")
    ax.axhline(y=vram_peak.mean(), color="tab:red", ls="--", alpha=0.5,
               label=f"Mean: {vram_peak.mean():.0f} GB")
    ax.axvspan(0, N_SSL, alpha=0.08, color="green", label="SSL pretrain")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VRAM (GB)")
    ax.set_title("Expected GPU memory usage (A100 80GB)   [FAKE / synthetic]")
    ax.legend()
    fig.text(0.5, -0.02, "Fake data — reference only",
             ha="center", fontsize=9, fontstyle="italic", color="gray",
             transform=ax.transAxes)
    fig.tight_layout()
    figures.append(("VRAM usage", fig))

    # ---- 6. Summary table ----
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    summary = [
        ["Metric", "Expected value", "Notes"],
        ["SSL epochs", str(N_SSL), ""],
        ["SSL loss (final)", f"{ssl_loss[-1]:.4f}", ""],
        ["Train loss (final)", f"{train_loss[-1]:.4f}", ""],
        ["Val loss (final)", f"{val_loss[-1]:.4f}", ""],
        ["Total SSL time", f"{total_ssl_s/60:.0f} min", ""],
        ["Total main time", f"{total_main_s/60:.0f} min", ""],
        ["Total wall time", f"{total_s/3600:.1f} hours", ""],
        ["Avg epoch time", f"{time_per_epoch.mean()/60:.1f} min", ""],
        ["Peak VRAM", f"{vram_peak.max():.0f} GB", "A100 sparse k=16"],
        ["Mean VRAM", f"{vram_peak.mean():.0f} GB", ""],
    ]
    table = ax.table(cellText=summary[1:], colLabels=summary[0],
                     loc="center", cellLoc="left",
                     colWidths=[0.25, 0.25, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(3):
        table[0, j].set_facecolor("#404040")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(summary)):
        table[i, 0].set_facecolor("#f0f0f0")
    ax.set_title("Expected metrics summary   [ALL VALUES SYNTHETIC]",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    figures.append(("Summary table", fig))

    # ============================================================
    # Render HTML
    # ============================================================
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Expected metrics — sparse+SSL</title>
<style>
body { font-family: sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; }
.banner { background: #fff3cd; border: 1px solid #ffc107; padding: 1em; border-radius: 6px;
           text-align: center; font-weight: bold; font-size: 1.1em; margin-bottom: 2em; }
img { width: 100%; margin-bottom: 1.5em; border: 1px solid #ddd; border-radius: 4px; }
h1 { border-bottom: 2px solid #ddd; padding-bottom: 0.3em; }
</style></head><body>
<h1>[FAKE] Expected / Synthetic Performance Metrics</h1>
<div class="banner">
    ALL VALUES ON THIS PAGE ARE SYNTHETIC — FAKE DATA FOR REFERENCE ONLY.<br>
    Generated by <code>plot_expected_metrics.py</code>. Do not cite or report.
</div>
"""

    for title, fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        html += f'<h2>{title}</h2>\n'
        html += f'<img src="data:image/png;base64,{b64}" alt="{title}">\n'
        plt.close(fig)

    html += """
<hr>
<p style="color:gray;font-style:italic;font-size:0.9em">
    Rendering: <code>python plot_expected_metrics.py</code><br>
    Last generated: {date}<br>
    Config: sparse attention k=16, SSL pretrain, A100, mixed precision fp16
</p>
</body></html>
"""

    OUT.write_text(html)
    print(f"Written {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the expected/synthetic metrics HTML report."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the numpy RNG that produces the synthetic data.",
    )
    args = parser.parse_args()
    main(seed=args.seed)
