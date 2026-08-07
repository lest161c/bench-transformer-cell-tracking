"""Trackastra inference benchmark — timing breakdown for get_features() and predict_windows().

Measures wall time and memory across varying cell counts (N) using
example_data_bacteria and synthetic masks of known sizes.

Addresses meeting_18_06_26.txt lines 12-18:
  - measure get_features() separately
  - measure predict_windows() for different N
  - inference time vs N scaling
  - memory vs N scaling

Usage:
    python benchmark_trackastra_inference.py [--out results.csv]
"""

import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_example_data():
    """Load example bacteria data (63 frames, trpL/150310-11)."""
    from trackastra.data import example_data_bacteria
    return example_data_bacteria()


def _get_model():
    """Load pretrained Trackastra model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from trackastra import Trackastra
    model = Trackastra.from_pretrained("general_2d", device=device)
    return model, device


def _synthesize_data(N_cells_per_frame: int, T: int = 10, H: int = 320, W: int = 320):
    """Generate synthetic masks/images with controlled cell count.

    Args:
        N_cells_per_frame: Average cells per frame (uniform across frames).
        T: Number of frames.
        H, W: Image dimensions.

    Returns:
        masks: (T, H, W) uint16 array
        imgs: (T, H, W) uint16 array
    """
    rng = np.random.RandomState(42)
    masks = []
    imgs = []
    for t in range(T):
        mask = np.zeros((H, W), dtype=np.uint16)
        img = np.zeros((H, W), dtype=np.uint16)
        for i in range(N_cells_per_frame):
            # Random ellipse
            cy = rng.randint(20, H - 20)
            cx = rng.randint(20, W - 20)
            ry = rng.randint(5, 12)
            rx = rng.randint(5, 12)
            yy, xx = np.ogrid[:H, :W]
            region = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            if mask[region].sum() == 0:  # no overlap
                mask[region] = i + 1
                img[region] = rng.randint(100, 200)
        masks.append(mask)
        imgs.append(img)
    return np.stack(masks, axis=0), np.stack(imgs, axis=0)


def _measure_get_features(masks, imgs, n_workers=0):
    """Measure get_features() wall time and peak memory."""
    from trackastra.data import get_features

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    start_time = time.perf_counter()
    features = get_features(masks, imgs, features="wrfeat", n_workers=n_workers)
    end_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = (mem_peak - mem_before) / (1024 ** 2)
    else:
        mem_used = 0

    total_cells = sum(len(feature.labels) for feature in features)
    return end_time - start_time, mem_used, total_cells


def _measure_predict_windows(features, model, batch_size=4):
    """Measure predict_windows() wall time and peak memory.

    Returns:
        wall_time: seconds
        mem_mb: peak GPU memory increment
        n_windows: number of windows processed
        mean_cells_per_window: average N in each window
    """
    from trackastra.data import build_windows
    from trackastra.model.predict import predict_windows

    windows = build_windows(features, window_size=4)
    n_cells = [len(window["labels"]) for window in windows]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    start_time = time.perf_counter()
    result = predict_windows(windows, features, model, batch_size=batch_size)
    end_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated()
        mem_used = (mem_peak - mem_before) / (1024 ** 2)
    else:
        mem_used = 0

    return end_time - start_time, mem_used, len(windows), np.mean(n_cells) if n_cells else 0, max(n_cells) if n_cells else 0


def run_benchmark():
    """Run the full Trackastra inference benchmark.

    Executes two test suites:
      1. Example bacteria data (real microscopy frames).
      2. Synthetic scaling with varying cell counts (N = 10..1000).

    For each test, measures ``get_features()`` and ``predict_windows()``
    wall time and peak GPU memory.

    Returns:
        Tuple of (rows, device) where *rows* is a list of per-test
        result dictionaries and *device* is the compute device string.
    """
    rows = []

    print("Loading model...")
    model, device = _get_model()
    gc.collect()

    # ── Test 1: Example data (bacteria) ──
    print("\n=== Test 1: Example bacteria data ===")
    masks, imgs = _get_example_data()
    T, H, W = masks.shape
    total_cells = len(np.unique(masks)) - 1

    # get_features timing
    time_feat, memory_mb_feat, n_feat = _measure_get_features(masks, imgs)
    print(f"  get_features: {time_feat:.3f}s, {memory_mb_feat:.1f} MB, {n_feat} cells")

    rows.append({
        "test": "example_bacteria", "N_mean": n_feat // T, "N_max": -1,
        "T": T, "stage": "get_features",
        "time_s": time_feat, "mem_mb": memory_mb_feat, "n_total": n_feat,
    })

    # predict_windows timing
    from trackastra.data import get_features
    features = get_features(masks, imgs, features="wrfeat")

    time_pred, memory_mb_pred, n_windows, mean_cells, max_cells = _measure_predict_windows(features, model)
    print(f"  predict_windows: {time_pred:.3f}s, {memory_mb_pred:.1f} MB, {n_windows} windows, "
          f"mean {mean_cells:.0f} cells/win, max {max_cells} cells/win")

    rows.append({
        "test": "example_bacteria", "N_mean": mean_cells, "N_max": max_cells,
        "T": T, "stage": "predict_windows",
        "time_s": time_pred, "mem_mb": memory_mb_pred, "n_total": n_windows,
    })

    # ── Test 2: Synthetic scaling with varying N ──
    print("\n=== Test 2: Synthetic N scaling ===")
    gc.collect()

    N_values = [10, 25, 50, 100, 200, 500, 1000]
    for N_cells in N_values:
        try:
            masks_syn, imgs_syn = _synthesize_data(N_cells, T=10)
            T_syn = 10

            time_feat, memory_mb_feat, n_feat = _measure_get_features(masks_syn, imgs_syn)
            print(f"  N={N_cells:>4d}  get_features: {time_feat:.4f}s  "
                  f"cells={n_feat}")

            rows.append({
                "test": f"synthetic_N={N_cells}", "N_mean": N_cells, "N_max": N_cells,
                "T": T_syn, "stage": "get_features",
                "time_s": time_feat, "mem_mb": memory_mb_feat, "n_total": n_feat,
            })

            from trackastra.data import get_features
            features_syn = get_features(masks_syn, imgs_syn, features="wrfeat")
            time_pred, memory_mb_pred, n_windows, mean_cells, max_cells = _measure_predict_windows(
                features_syn, model
            )
            print(f"  N={N_cells:>4d}  predict_win: {time_pred:.4f}s  mem={memory_mb_pred:.1f}MB  "
                  f"windows={n_windows}  mean_N={mean_cells:.0f}  max_N={max_cells}")

            rows.append({
                "test": f"synthetic_N={N_cells}", "N_mean": mean_cells, "N_max": max_cells,
                "T": T_syn, "stage": "predict_windows",
                "time_s": time_pred, "mem_mb": memory_mb_pred, "n_total": n_windows,
            })

        except Exception as error:
            print(f"  N={N_cells:>4d}  ERROR: {error}")
            rows.append({
                "test": f"synthetic_N={N_cells}", "N_mean": N_cells, "N_max": N_cells,
                "T": 10, "stage": "ERROR",
                "time_s": -1, "mem_mb": -1, "n_total": -1,
                "error": str(error)[:200],
            })

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return rows, device


def save_results(rows, path="trackastra_inference_benchmark.csv"):
    """Save benchmark rows to *path* as CSV.

    Args:
        rows: List of dictionaries to write.
        path: Output CSV file path.
    """
    fieldnames = ["test", "N_mean", "N_max", "T", "stage", "time_s", "mem_mb", "n_total"]
    with open(path, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {path}")


def generate_figure(rows, save_path="trackastra_inference_scaling.png"):
    """Generate scaling plots: time vs N, memory vs N, and % breakdown."""
    get_feat = [row for row in rows if row["stage"] == "get_features" and "synthetic" in row["test"]]
    pred_win = [row for row in rows if row["stage"] == "predict_windows" and "synthetic" in row["test"]]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Time vs N (log-log)
    ax = axes[0, 0]
    if get_feat:
        ns_f = [row["N_mean"] for row in get_feat if row["time_s"] > 0]
        ts_f = [row["time_s"] * 1000 for row in get_feat if row["time_s"] > 0]
        ax.plot(ns_f, ts_f, "o-", color="#e74c3c", label="get_features()",
                markersize=8, linewidth=2)
    if pred_win:
        ns_p = [row["N_mean"] for row in pred_win if row["time_s"] > 0]
        ts_p = [row["time_s"] * 1000 for row in pred_win if row["time_s"] > 0]
        ax.plot(ns_p, ts_p, "s-", color="#3498db", label="predict_windows()",
                markersize=8, linewidth=2)
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Time vs Cell Count")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Memory vs N
    ax = axes[0, 1]
    if get_feat:
        ns_f = [row["N_mean"] for row in get_feat if row["mem_mb"] > 0]
        ms_f = [row["mem_mb"] for row in get_feat if row["mem_mb"] > 0]
        ax.plot(ns_f, ms_f, "o-", color="#e74c3c", label="get_features()",
                markersize=8, linewidth=2)
    if pred_win:
        ns_p = [row["N_mean"] for row in pred_win if row["mem_mb"] > 0]
        ms_p = [row["mem_mb"] for row in pred_win if row["mem_mb"] > 0]
        ax.plot(ns_p, ms_p, "s-", color="#3498db", label="predict_windows()",
                markersize=8, linewidth=2)
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("GPU Memory (MiB)")
    ax.set_title("GPU Memory vs Cell Count")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Time breakdown (% of total inference)
    ax = axes[1, 0]
    categories = []
    feat_pct, pred_pct = [], []
    for row in get_feat:
        matching_pred = [pred_row for pred_row in pred_win if pred_row["test"] == row["test"]]
        if matching_pred and row["time_s"] > 0 and matching_pred[0]["time_s"] > 0:
            total = row["time_s"] + matching_pred[0]["time_s"]
            categories.append(f"N={row['N_mean']}")
            feat_pct.append(row["time_s"] / total * 100)
            pred_pct.append(matching_pred[0]["time_s"] / total * 100)

    if categories:
        bar_positions = np.arange(len(categories))
        bar_width = 0.35
        ax.bar(bar_positions - bar_width/2, feat_pct, bar_width, color="#e74c3c", label="get_features()", alpha=0.8)
        ax.bar(bar_positions + bar_width/2, pred_pct, bar_width, color="#3498db", label="predict_windows()", alpha=0.8)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(categories)
        ax.set_ylabel("% of total time")
        ax.set_title("Time Breakdown by Stage")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Time per cell vs N
    ax = axes[1, 1]
    if pred_win:
        ns_p = [row["N_mean"] for row in pred_win if row["time_s"] > 0]
        ts_per_cell = [row["time_s"] / (row["N_mean"] * row["T"]) * 1e6
                       for row in pred_win if row["time_s"] > 0]
        ax.plot(ns_p, ts_per_cell, "D-", color="#2ecc71",
                markersize=8, linewidth=2)
    ax.set_xlabel("Cells per frame (N)")
    ax.set_ylabel("Time per cell (μs)")
    ax.set_title("Per-Cell Inference Time")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Trackastra Inference Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure → {save_path}")


def main():
    """Run the Trackastra inference benchmark and save CSV/figure."""
    parser = argparse.ArgumentParser(description="Trackastra inference benchmark")
    parser.add_argument("--out-csv", default="trackastra_inference_benchmark.csv")
    parser.add_argument("--out-png", default="trackastra_inference_scaling.png")
    args = parser.parse_args()

    rows, device = run_benchmark()

    save_results(rows, args.out_csv)
    generate_figure(rows, args.out_png)

    # Console summary
    print("\n" + "=" * 60)
    print(f"Benchmark complete on {device.upper()}")
    for stage in ["get_features", "predict_windows"]:
        sub = [row for row in rows if row["stage"] == stage and row["time_s"] > 0]
        if sub:
            avg_t = np.mean([row["time_s"] for row in sub])
            avg_m = np.mean([row["mem_mb"] for row in sub if row["mem_mb"] > 0])
            print(f"  {stage}: avg {avg_t*1000:.2f}ms, avg mem {avg_m:.1f}MB "
                  f"({len(sub)} measurements)")


if __name__ == "__main__":
    main()
