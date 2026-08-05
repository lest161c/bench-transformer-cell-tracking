"""Feature signal analysis for contrastive SSL on vanvliet regionprops.

Surgical data-level tests to determine if augmented features carry sufficient
cell-identity signal for contrastive learning — before any model training.

Key questions answered here:
  Q1: After distortion, are same-cell features more similar than different-cell?
  Q2: What is the effective dimensionality of the augmented feature space?
  Q3: Which features survive which distortion families?
  Q4: Can we match cells across views using features alone? (recall@k)
  Q5: Is the contrastive signal strong enough at the data level?

Interpretation guide for Q5 (run `python analyze_signal.py --interpret`):
  - PASS (mean intra-sim - mean inter-sim > 0.3): Strong signal, contrastive
    learning should work directly with current features.
  - BORDERLINE (0.1 < gap < 0.3): Weak but present signal. Needs richer
    features (shape descriptors, image patches) per proposal §4.4.
  - FAIL (gap < 0.1): Regionprops are fundamentally insufficient. Must add
    perceptual features (CNN crops, SAM embeddings) before attempting
    contrastive learning.

Usage:
    python analyze_signal.py                          # Run all analyses on rpsM
    python analyze_signal.py --conditions rpsM,recA   # Multiple conditions
    python analyze_signal.py --interpret              # Print pass/fail verdict

Output:
    benchmark_ssl/feature_signal_report.html  — interactive HTML report
    benchmark_ssl/feature_signal_results.csv  — numerical summary
"""

import argparse
import csv
import io
import base64
import logging
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tifffile import imread

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_signal")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark_ssl root (sibling modules)

# --- reuse ssl_pipeline + distortions from benchmark_ssl ---
from ssl_pipeline import load_experiment_frames
from rich_features import extract as extract_rich, FEATURE_DIMS, FEATURE_NAMES
from distortions import (
    AffineDistortion,
    ElasticDistortion,
    JitterDistortion,
    DropoutDistortion,
    PhotometricDistortion,
    FeatureNoise,
    DistortionPipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ─── test batteries ───────────────────────────────────────────────────────────


def compute_intra_inter_similarity(
    feats1: np.ndarray, feats2: np.ndarray, labels1: np.ndarray, labels2: np.ndarray
) -> dict:
    """Q1: Intra-cell (same cell, diff views) vs inter-cell (diff cells) similarity.

    Returns dict with:
      intra_sim:  mean cosine sim of same-cell pairs
      inter_sim:  mean cosine sim of different-cell pairs
      sep_gap:    intra_sim - inter_sim  (key decision metric)
    """
    if len(feats1) == 0 or len(feats2) == 0:
        return {"intra_sim": np.nan, "inter_sim": np.nan, "sep_gap": np.nan}

    f1 = feats1 / (np.linalg.norm(feats1, axis=1, keepdims=True) + 1e-12)
    f2 = feats2 / (np.linalg.norm(feats2, axis=1, keepdims=True) + 1e-12)
    sim = f1 @ f2.T  # (N1, N2)

    intra, inter = [], []
    for i, lbl_i in enumerate(labels1):
        matches = np.where(labels2 == lbl_i)[0]
        for j in range(len(labels2)):
            if j in matches:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])

    intra_sim = np.mean(intra) if intra else np.nan
    inter_sim = np.mean(inter) if inter else np.nan
    return {
        "intra_sim": intra_sim,
        "inter_sim": inter_sim,
        "sep_gap": (intra_sim - inter_sim) if not (np.isnan(intra_sim) or np.isnan(inter_sim)) else np.nan,
    }


def compute_effective_rank(feats: np.ndarray, var_threshold: float = 0.95) -> dict:
    """Q2: Effective dimensionality — PCA components needed to explain var_threshold variance."""
    if len(feats) < 3:
        return {"n_dim": np.nan, "explained_var_ratio": []}
    feats_c = feats - feats.mean(axis=0, keepdims=True)
    n_components = min(len(feats), feats.shape[1])
    if n_components < 2:
        return {"n_dim": 1, "explained_var_ratio": [1.0]}
    pca = PCA(n_components=n_components).fit(feats_c)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_dim = int(np.searchsorted(cumvar, var_threshold) + 1)
    return {
        "n_dim": n_dim,
        "explained_var_ratio": pca.explained_variance_ratio_.tolist(),
    }


def compute_destructiveness(
    feats_orig: np.ndarray, feats_dist: np.ndarray, feat_names: list
) -> dict:
    """Q3: Per-feature normalized MSE after distortion.

    e^2_i = mean((f_dist[:,i] - f_orig[:,i])^2) / var(f_orig[:,i])
    Values > 1.0  → feature is effectively destroyed by distortion.
    Values < 0.1  → feature is nearly preserved.
    """
    destruction = {}
    for i, name in enumerate(feat_names):
        orig = feats_orig[:, i]
        dist = feats_dist[:, i]
        var = np.var(orig)
        if var < 1e-12:
            destruction[name] = np.nan
        else:
            destruction[name] = float(np.mean((dist - orig) ** 2) / var)
    return destruction


def compute_recall_at_k(
    feats1: np.ndarray, feats2: np.ndarray, labels1: np.ndarray, labels2: np.ndarray, ks=(1, 3, 5)
) -> dict:
    """Q4: Feature-only cell matching across views.

    For each cell in view1, rank all cells in view2 by cosine similarity.
    Is the correct match in top-k?
    """
    if len(feats1) == 0 or len(feats2) == 0:
        return {f"recall@{k}": np.nan for k in ks}

    f1 = feats1 / (np.linalg.norm(feats1, axis=1, keepdims=True) + 1e-12)
    f2 = feats2 / (np.linalg.norm(feats2, axis=1, keepdims=True) + 1e-12)
    sim = f1 @ f2.T

    results = {}
    for topk in ks:
        hits = 0
        total = 0
        for i, lbl in enumerate(labels1):
            if lbl not in set(labels2):
                continue
            true_j = np.where(labels2 == lbl)[0][0]
            top_k = np.argsort(-sim[i])[:topk]
            if true_j in top_k:
                hits += 1
            total += 1
        results[f"recall@{topk}"] = hits / total if total > 0 else np.nan
    return results


# ─── data loading ─────────────────────────────────────────────────────────────


def collect_samples(frames, distortion_pipeline, max_frames=200, feature_set="basic"):
    """Collect distortion pairs from frames for all analyses.

    Returns list of dicts with feats_orig, feats1, feats2, labels1, labels2, distortion_names.
    """
    samples = []
    n_loaded = 0
    for idx in range(min(len(frames), max_frames)):
        _, _, _, mask_path, img_path = frames[idx]
        mask = imread(mask_path)
        img = imread(img_path).astype(np.float32)
        p1, p998 = np.percentile(img, (1, 99.8))
        img = np.clip((img - p1) / (p998 - p1 + 1e-8), 0, 1)

        result = extract_rich(feature_set, mask, img, ndim=2)
        if result is None:
            continue
        coords_src, labels_src, feats_dict_src = result
        if len(labels_src) < 2:
            continue

        feats_orig = np.concatenate(list(feats_dict_src.values()), axis=-1).astype(np.float32)

        c1, f1_dict, l1, c2, f2_dict, l2 = distortion_pipeline(
            coords_src, feats_dict_src, labels_src
        )
        f1 = np.concatenate(list(f1_dict.values()), axis=-1).astype(np.float32)
        f2 = np.concatenate(list(f2_dict.values()), axis=-1).astype(np.float32)

        idx1 = np.argsort(l1)
        idx2 = np.argsort(l2)

        samples.append({
            "feats_orig": feats_orig,
            "feats1": f1[idx1],
            "feats2": f2[idx2],
            "labels1": l1[idx1],
            "labels2": l2[idx2],
            "labels_orig": labels_src.copy(),
        })
        n_loaded += 1
        if n_loaded >= max_frames:
            break

    logger.info(f"Collected {len(samples)} samples with cells")
    return samples


def collect_per_distortion_samples(frames, cfg, max_frames=100, feature_set="basic"):
    """Collect samples for each distortion family individually."""
    import copy
    feat_names = FEATURE_NAMES.get(feature_set)
    if feat_names is None:
        # patch set: determine dynamically from first sample
        feat_names = ["patch_features"]

    single_distortion_names = [
        "affine", "elastic", "jitter", "dropout", "photometric", "feature_noise",
    ]

    per_dist = {}
    for dname in single_distortion_names:
        dcfg = copy.deepcopy(cfg)
        dcfg["distortions"] = [dname]
        dist = DistortionPipeline.from_config(dcfg)
        samples = collect_samples(frames, dist, max_frames=max_frames, feature_set=feature_set)
        per_dist[dname] = {"samples": samples, "feat_names": feat_names}
        logger.info(f"  {dname}: {len(samples)} frames")
    return per_dist


# ─── all tests runner ─────────────────────────────────────────────────────────


def run_all_tests(samples, feat_names):
    """Run all analysis tests on collected samples.

    Returns dict of aggregated results across all frames.
    """
    # Q1: intra/inter similarity
    intra_all, inter_all, sep_all = [], [], []
    for s in samples:
        r = compute_intra_inter_similarity(s["feats1"], s["feats2"], s["labels1"], s["labels2"])
        if not (np.isnan(r.get("sep_gap", np.nan))):
            intra_all.append(r["intra_sim"])
            inter_all.append(r["inter_sim"])
            sep_all.append(r["sep_gap"])

    q1 = {
        "intra_sim_mean": float(np.mean(intra_all)) if intra_all else np.nan,
        "intra_sim_std": float(np.std(intra_all)) if intra_all else np.nan,
        "inter_sim_mean": float(np.mean(inter_all)) if inter_all else np.nan,
        "inter_sim_std": float(np.std(inter_all)) if inter_all else np.nan,
        "sep_gap_mean": float(np.mean(sep_all)) if sep_all else np.nan,
        "sep_gap_std": float(np.std(sep_all)) if sep_all else np.nan,
    }

    # Q2: effective rank on pooled post-distortion features
    all_feats = np.concatenate([s["feats1"] for s in samples] + [s["feats2"] for s in samples], axis=0)
    orig_feats = np.concatenate([s["feats_orig"] for s in samples], axis=0)
    if len(all_feats) > 2:
        q2_dist = compute_effective_rank(all_feats, var_threshold=0.95)
        q2_orig = compute_effective_rank(orig_feats, var_threshold=0.95)
    else:
        q2_dist = {"n_dim": np.nan, "explained_var_ratio": []}
        q2_orig = {"n_dim": np.nan, "explained_var_ratio": []}
    q2 = {
        "orig_n_dim_95": q2_orig["n_dim"],
        "distorted_n_dim_95": q2_dist["n_dim"],
        "orig_explained_var": q2_orig.get("explained_var_ratio", []),
        "distorted_explained_var": q2_dist.get("explained_var_ratio", []),
    }

    # Q3: destructiveness per-frame then averaged
    dest_frames = []
    for s in samples:
        # align: feats_orig may have more cells than feats1 (dropout)
        # match by label to compare same cells
        orig_labels = s.get("labels_orig", s["labels1"])
        # use feats_orig but compare only cells that survived
        # simple approach: compare feats1 to feats2 (both post-distortion) — which
        # measures cross-view consistency, not destructiveness.
        # Instead, for cells that appear in both orig and dist1, compare features.
        labels_orig = s.get("labels_orig")
        if labels_orig is None:
            continue
        # map orig -> dist1 by label
        feat_map_orig = {int(l): s["feats_orig"][i] for i, l in enumerate(labels_orig)}
        feat_map_dist = {}
        for i, l in enumerate(s["labels1"]):
            # first occurrence wins
            if int(l) not in feat_map_dist:
                feat_map_dist[int(l)] = s["feats1"][i]
        shared = set(feat_map_orig) & set(feat_map_dist)
        if len(shared) < 2:
            continue
        orig_aligned = np.stack([feat_map_orig[l] for l in sorted(shared)])
        dist_aligned = np.stack([feat_map_dist[l] for l in sorted(shared)])
        dest_frames.append(compute_destructiveness(orig_aligned, dist_aligned, feat_names))
    if dest_frames:
        q3 = {name: float(np.mean([d[name] for d in dest_frames if not np.isnan(d.get(name, np.nan))]))
              for name in feat_names}
    else:
        q3 = {name: np.nan for name in feat_names}

    # Q4: recall@k
    recall_list = {f"recall@{k}": [] for k in (1, 3, 5)}
    for s in samples:
        r = compute_recall_at_k(s["feats1"], s["feats2"], s["labels1"], s["labels2"])
        for k in (1, 3, 5):
            v = r.get(f"recall@{k}", np.nan)
            if not np.isnan(v):
                recall_list[f"recall@{k}"].append(v)
    q4 = {k: float(np.mean(v)) if v else np.nan for k, v in recall_list.items()}

    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}


def run_per_distortion_tests(per_dist_data):
    """Run Q1 + Q4 per distortion family to isolate impact.

    Returns {distortion_name: {q1: ..., q4: ...}}
    """
    results = {}
    for dname, data in per_dist_data.items():
        r = run_all_tests(data["samples"], data["feat_names"])
        results[dname] = {"q1": r["q1"], "q4": r["q4"]}
    return results


# ─── interpretation ───────────────────────────────────────────────────────────


def interpret(results: dict) -> str:
    """Verdict on whether current features can support contrastive SSL."""
    gap = results["all"]["q1"]["sep_gap_mean"]
    recall1 = results["all"]["q4"]["recall@1"]
    recall3 = results["all"]["q4"]["recall@3"]
    eff_dim = results["all"]["q2"]["distorted_n_dim_95"]

    lines = []
    lines.append("=" * 65)
    lines.append("FEATURE SIGNAL ANALYSIS :: VERDICT FOR CONTRASTIVE SSL")
    lines.append("=" * 65)

    if np.isnan(gap):
        lines.append("  INCONCLUSIVE — no valid frames analyzed.")
        return "\n".join(lines)

    lines.append(f"  Separation gap (intra - inter):  {gap:.4f}")
    lines.append(f"  Recall@1 (feature-only match):   {recall1:.4f}")
    lines.append(f"  Recall@3:                        {recall3:.4f}")
    lines.append(f"  Effective dim (95% var):         {eff_dim}")
    lines.append("")

    reasons = []
    verdict = "UNDETERMINED"

    # Rule 1: Separation gap
    if gap > 0.3:
        reasons.append(f"✓ Gap={gap:.3f} > 0.3 — strong class separation")
    elif gap > 0.15:
        reasons.append(f"~ Gap={gap:.3f} — moderate separation (borderline)")
    else:
        reasons.append(f"✗ Gap={gap:.3f} < 0.15 — poor separation")

    # Rule 2: Recall
    if recall1 > 0.5:
        reasons.append(f"✓ Recall@1={recall1:.3f} > 0.5 — feature matching is effective")
    elif recall1 > 0.2:
        reasons.append(f"~ Recall@1={recall1:.3f} — moderate feature matching")
    else:
        reasons.append(f"✗ Recall@1={recall1:.3f} < 0.2 — features alone can't match cells")

    # Rule 3: Effective dimensionality
    if eff_dim >= 4:
        reasons.append(f"✓ eff_dim={eff_dim} — ample dimensions for contrastive")
    elif eff_dim >= 2:
        reasons.append(f"~ eff_dim={eff_dim} — low but workable")
    else:
        reasons.append(f"✗ eff_dim={eff_dim} — collapse risk (1D = no contrastive signal)")

    lines.extend(reasons)
    lines.append("")

    # Composite verdict
    if gap > 0.3 and recall1 > 0.5 and eff_dim >= 4:
        verdict = "PASS — regionprops carry sufficient signal for contrastive SSL"
        lines.append(f"✓ VERDICT: {verdict}")
    elif gap > 0.15 and recall1 > 0.2 and eff_dim >= 2:
        verdict = "BORDERLINE — contrastive learning may work with careful tuning (lower temp, projection head)"
        lines.append(f"~ VERDICT: {verdict}")
    else:
        verdict = "FAIL — regionprops insufficient. Add perceptual features (proposal §4.4: shape descriptors, CNN image crops)"
        lines.append(f"✗ VERDICT: {verdict}")

    lines.append("=" * 65)

    # Per-distortion breakdown
    if results.get("per_distortion"):
        lines.append("")
        lines.append("PER-DISTORTION BREAKDOWN:")
        lines.append(f"  {'Distortion':<15} {'Gap':>8} {'Recall@1':>10} {'Recall@3':>10}")
        lines.append("  " + "-" * 45)
        for dname, dr in results["per_distortion"].items():
            dg = dr["q1"]["sep_gap_mean"]
            dr1 = dr["q4"]["recall@1"]
            dr3 = dr["q4"]["recall@3"]
            if not (np.isnan(dg) or np.isnan(dr1)):
                lines.append(f"  {dname:<15} {dg:>8.4f} {dr1:>10.4f} {dr3:>10.4f}")
        lines.append("")

    return "\n".join(lines)


# ─── HTML report ──────────────────────────────────────────────────────────────


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _clean_var(v, default="N/A", fmt=".4f"):
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return default
    if isinstance(v, (float, np.floating)):
        return f"{v:{fmt}}"
    if isinstance(v, np.integer):
        return f"{v}"
    return str(v)


def _make_color(val, thresholds, colors):
    for t, c in zip(thresholds, colors):
        if val >= t:
            return c
    return colors[-1]


def _make_table_html(rows, header):
    lines = ["<table>", "<tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr>"]
    for row in rows:
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def generate_report(all_results, per_dist_results, interpret_output, save_path="feature_signal_report.html"):
    """Generate an HTML report with all figures and tables."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    figures = []

    # ── Figure 1: Intra vs Inter similarity distribution ──
    q1 = all_results["q1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Intra-cell\n(same cell, diff views)", "Inter-cell\n(diff cells)"]
    means = [q1["intra_sim_mean"], q1["inter_sim_mean"]]
    stds = [q1["intra_sim_std"], q1["inter_sim_std"]]
    bars = ax.bar(categories, means, yerr=stds, capsize=5,
                  color=["#2ecc71", "#e74c3c"], alpha=0.8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Q1: Feature Similarity: Same-Cell vs Different-Cell")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    gap = q1["sep_gap_mean"]
    ax.text(0.5, 0.02, f"Separation gap = {gap:.4f}", transform=fig.transFigure,
            ha="center", fontsize=11, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    figures.append(("Q1: Same-Cell vs Different-Cell Separability", fig))
    plt.close(fig)

    # ── Figure 2: Effective dimensionality (PCA) ──
    q2 = all_results["q2"]
    var_dist = q2.get("distorted_explained_var", [])
    if var_dist:
        fig, ax = plt.subplots(figsize=(8, 5))
        cumvar = np.cumsum(var_dist)
        dims = np.arange(1, len(var_dist) + 1)
        ax.plot(dims, cumvar, "o-", color="#3498db", linewidth=2, markersize=6)
        ax.axhline(0.95, color="gray", ls="--", alpha=0.6, label="95% variance")
        ax.axvline(q2["distorted_n_dim_95"], color="#e74c3c", ls="--", alpha=0.6,
                   label=f"{q2['distorted_n_dim_95']} components")
        ax.set_xlabel("Number of PCA Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("Q2: Effective Dimensionality of Augmented Features")
        ax.legend()
        ax.set_xticks(dims)
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        figures.append(("Q2: PCA Explained Variance", fig))
        plt.close(fig)

    # ── Figure 3: Per-distortion destructiveness ──
    if per_dist_results:
        dist_names = list(per_dist_results.keys())
        gaps = [per_dist_results[d]["q1"]["sep_gap_mean"] for d in dist_names]
        r1s = [per_dist_results[d]["q4"]["recall@1"] for d in dist_names]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(dist_names))
        w = 0.35
        bars1 = ax1.bar(x - w/2, gaps, w, label="Separation gap", color="#3498db", alpha=0.8)
        ax1.set_ylabel("Separation gap", color="#3498db")
        ax1.tick_params(axis="y", labelcolor="#3498db")
        ax1.axhline(0.15, color="#3498db", ls=":", alpha=0.5, label="borderline threshold")

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + w/2, r1s, w, label="Recall@1", color="#e67e22", alpha=0.8)
        ax2.set_ylabel("Recall@1", color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22")

        ax1.set_xticks(x)
        ax1.set_xticklabels(dist_names)
        ax1.set_title("Q3: Per-Distortion Signal Breakdown")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        figures.append(("Q3: Distortion Impact on Signal", fig))
        plt.close(fig)

    # ── Figure 4: Destructiveness per feature dimension ──
    q3 = all_results.get("q3", {})
    if q3:
        feat_names = list(q3.keys())
        dest_vals = [q3[k] for k in feat_names]
        colors = ["#e74c3c" if v > 1.0 else ("#f39c12" if v > 0.5 else "#2ecc71")
                  for v in dest_vals]
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(feat_names, dest_vals, color=colors, alpha=0.8)
        ax.axvline(1.0, color="#e74c3c", ls="--", alpha=0.5, label="destroyed")
        ax.axvline(0.5, color="#f39c12", ls=":", alpha=0.5, label="degraded")
        ax.set_xlabel("Destructiveness (normalized MSE)")
        ax.set_title("Q3b: Per-Feature Destructiveness (all distortions)")
        ax.legend()
        for bar, val in zip(bars, dest_vals):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        figures.append(("Q3b: Feature Destructiveness", fig))
        plt.close(fig)

    # ── Figure 5: Recall@k ──
    q4 = all_results["q4"]
    ks = [1, 3, 5]
    recalls = [q4.get(f"recall@{k}", np.nan) for k in ks]
    valid = [(k, r) for k, r in zip(ks, recalls) if not np.isnan(r)]
    if valid:
        fig, ax = plt.subplots(figsize=(7, 5))
        k_vals, r_vals = zip(*valid)
        colors = ["#e74c3c" if r < 0.3 else ("#f39c12" if r < 0.5 else "#2ecc71")
                  for r in r_vals]
        bars = ax.bar([f"k={k}" for k in k_vals], r_vals, color=colors, alpha=0.8)
        ax.set_ylabel("Recall")
        ax.set_title("Q4: Feature-Only Cell Matching")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.4, label="random baseline")
        ax.axhline(0.2, color="#e74c3c", ls=":", alpha=0.4, label="poor threshold")
        ax.legend()
        for bar, r in zip(bars, r_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{r:.3f}", ha="center", fontsize=10, fontweight="bold")
        fig.tight_layout()
        figures.append(("Q4: Cell Matching Recall", fig))
        plt.close(fig)

    # ── Summary verdict box ──
    gap = all_results["q1"]["sep_gap_mean"]
    if not np.isnan(gap):
        if gap > 0.3:
            verdict_color = "#2ecc71"
            short_v = "PASS"
        elif gap > 0.15:
            verdict_color = "#f39c12"
            short_v = "BORDERLINE"
        else:
            verdict_color = "#e74c3c"
            short_v = "FAIL"
    else:
        verdict_color = "#95a5a6"
        short_v = "N/A"

    # Assemble HTML
    html_parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Feature Signal Analysis for Contrastive SSL</title>",
        "<style>",
        "body{font-family:sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#fafafa}",
        "h1{color:#333} h2{color:#555;border-bottom:2px solid #ddd;padding-bottom:5px}",
        "img{max-width:100%;margin:15px 0;border:1px solid #ddd;border-radius:6px}",
        "table{border-collapse:collapse;width:100%;margin:15px 0}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:center}",
        "th{background:#f5f5f5;font-weight:bold}",
        "tr:nth-child(even){background:#f9f9f9}",
        ".verdict{padding:20px;border-radius:8px;margin:20px 0;font-size:1.1em;font-weight:bold}",
        ".verdict-box{padding:15px;border-radius:8px;border:2px solid}",
        ".fail{background:#ffeaea;border-color:#e74c3c;color:#c0392b}",
        ".pass{background:#eaffea;border-color:#2ecc71;color:#27ae60}",
        ".borderline{background:#fef9e7;border-color:#f39c12;color:#d68910}",
        ".pre{font-family:monospace;white-space:pre;background:#f8f8f8;padding:15px;border-radius:5px;overflow-x:auto;font-size:0.9em;line-height:1.5}",
        "</style></head><body>",
        "<h1>Feature Signal Analysis for Contrastive SSL</h1>",
        f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M')} &mdash; "
        f"surgical data-level tests before model training.</p>",

        f"<div class='verdict-box {short_v.lower() if short_v != 'N/A' else ''}' style='border-left:5px solid {verdict_color}'>",
        f"<h2 style='margin-top:0;color:{verdict_color}'>Verdict: {short_v}</h2>",
        f"<p style='font-size:1em'>{interpret_output.split('VERDICT:')[-1].strip().split(chr(10))[0] if 'VERDICT:' in interpret_output else interpret_output[:200]}</p>",
        "</div>",
    ]

    # Summary table
    html_parts.append("<h2>Key Metrics</h2>")
    rows = [
        ["Intra-cell similarity (mean)", _clean_var(q1["intra_sim_mean"])],
        ["Inter-cell similarity (mean)", _clean_var(q1["inter_sim_mean"])],
        ["Separation gap (intra - inter)", _clean_var(gap),
         _make_color(gap, [0.3, 0.15], ["#2ecc71", "#f39c12", "#e74c3c"]) if not np.isnan(gap) else ""],
        ["Effective dim (orig, 95% var)", _clean_var(q2.get("orig_n_dim_95"), fmt="d")],
        ["Effective dim (distorted, 95% var)", _clean_var(q2.get("distorted_n_dim_95"), fmt="d")],
        ["Recall@1 (feature-only)", _clean_var(q4.get("recall@1"))],
        ["Recall@3", _clean_var(q4.get("recall@3"))],
        ["Recall@5", _clean_var(q4.get("recall@5"))],
    ]
    for i in range(1, len(rows)):
        if len(rows[i]) == 2:
            rows[i].append("")
    html_parts.append(_make_table_html(
        [[r[0], r[1], ""] for r in rows],
        ["Metric", "Value", ""]
    ))

    # Figures
    for title, fig in figures:
        b64 = _fig_to_b64(fig)
        html_parts.append(f"<figure><figcaption><b>{title}</b></figcaption>"
                          f'<img src="data:image/png;base64,{b64}" /></figure>')

    # Full interpretation output
    html_parts.append("<h2>Detailed Interpretation</h2>")
    html_parts.append(f"<div class='pre'>{interpret_output}</div>")

    # Per-distortion details table
    if per_dist_results:
        html_parts.append("<h2>Per-Distortion Breakdown</h2>")
        dist_rows = []
        for dname, dr in per_dist_results.items():
            dq1, dq4 = dr["q1"], dr["q4"]
            dist_rows.append([
                dname,
                _clean_var(dq1["sep_gap_mean"]),
                _clean_var(dq4["recall@1"]),
                _clean_var(dq4.get("recall@3")),
            ])
        html_parts.append(_make_table_html(dist_rows,
            ["Distortion", "Separation Gap", "Recall@1", "Recall@3"]))

    html_parts.append("</body></html>")
    html = "\n".join(html_parts)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(html)
    logger.info(f"Report saved to {save_path}")
    return html


def save_csv(all_results, per_dist_results, csv_path):
    """Save numerical results to CSV."""
    rows = []
    q1 = all_results["q1"]
    q2 = all_results["q2"]
    q3 = all_results.get("q3", {})
    q4 = all_results["q4"]

    row = {"scope": "all"}
    row.update({f"q1_{k}": v for k, v in q1.items() if isinstance(v, (int, float))})
    row.update({f"q2_{k}": v for k, v in q2.items() if isinstance(v, (int, float))})
    row.update({f"q3_{k}": v for k, v in q3.items()})
    row.update({f"q4_{k}": v for k, v in q4.items()})
    rows.append(row)

    if per_dist_results:
        for dname, dr in per_dist_results.items():
            row = {"scope": dname}
            row.update({f"q1_{k}": v for k, v in dr["q1"].items() if isinstance(v, (int, float))})
            row.update({f"q4_{k}": v for k, v in dr["q4"].items()})
            rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")


# ─── main ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Feature signal analysis for contrastive SSL")
    p.add_argument("--config", default="config.yaml", help="benchmark_ssl config")
    p.add_argument("--conditions", default="rpsM", help="comma-separated conditions")
    p.add_argument("--max-frames", type=int, default=200, help="frames per condition")
    p.add_argument("--outdir", default="runs/feature_signal", help="output dir")
    p.add_argument("--feature-set", default="basic", choices=["basic", "shape", "hu", "patch"],
                   help="Feature richness level (proposal §4.4)")
    p.add_argument("--interpret", action="store_true",
                   help="Print interpretation verdict only (no analysis run)")
    args = p.parse_args()

    if args.interpret:
        csv_path = Path(args.outdir) / "feature_signal_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_row = df[df["scope"] == "all"].iloc[0]
            gap = all_row.get("q1_sep_gap_mean", np.nan)
            recall1 = all_row.get("q4_recall@1", np.nan)
            recall3 = all_row.get("q4_recall@3", np.nan)
            eff_dim = all_row.get("q2_distorted_n_dim_95", np.nan)
            dummy_result = {
                "all": {
                    "q1": {"sep_gap_mean": gap, "intra_sim_mean": np.nan, "inter_sim_mean": np.nan},
                    "q4": {"recall@1": recall1, "recall@3": recall3},
                    "q2": {"distorted_n_dim_95": eff_dim},
                }
            }
            per_dist_data = {}
            for _, row in df[df["scope"] != "all"].iterrows():
                dname = row["scope"]
                per_dist_data[dname] = {
                    "q1": {"sep_gap_mean": row.get("q1_sep_gap_mean", np.nan)},
                    "q4": {"recall@1": row.get("q4_recall@1", np.nan), "recall@3": row.get("q4_recall@3", np.nan)},
                }
            dummy_result["per_distortion"] = per_dist_data
            print(interpret(dummy_result))
        else:
            print(f"No results found at {csv_path}. Run without --interpret first.")
        return

    logger.info(f"Conditions: {args.conditions}")
    conditions = [c.strip() for c in args.conditions.split(",")]

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["conditions"] = conditions

    # Load frames
    frames = load_experiment_frames(cfg["data_root"], conditions=conditions)
    logger.info(f"Total frames: {len(frames)}")

    feature_set = args.feature_set
    logger.info(f"Feature set: {feature_set} ({FEATURE_DIMS.get(feature_set, '?')}D)")

    # ── Full distortion pipeline ──
    logger.info("Building full distortion pipeline...")
    dist = DistortionPipeline.from_config(cfg)
    samples = collect_samples(frames, dist, max_frames=args.max_frames, feature_set=feature_set)
    feature_names = FEATURE_NAMES.get(feature_set)
    if feature_names is None and samples:
        feature_names = [f"feat_{i}" for i in range(samples[0]["feats1"].shape[-1])]

    all_results = run_all_tests(samples, feature_names)

    # ── Per-distortion ──
    logger.info("Building per-distortion pipelines...")
    per_dist_data = collect_per_distortion_samples(frames, cfg, max_frames=min(args.max_frames // 2, 100), feature_set=feature_set)
    per_dist_results = run_per_distortion_tests(per_dist_data)

    full_results = {"all": all_results, "per_distortion": per_dist_results}

    # ── Interpret & report ──
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    interpret_output = interpret(full_results)
    print(interpret_output)

    save_csv(all_results, per_dist_results, str(outdir / "feature_signal_results.csv"))
    generate_report(all_results, per_dist_results, interpret_output,
                    save_path=str(outdir / "feature_signal_report.html"))

    return full_results


if __name__ == "__main__":
    main()
