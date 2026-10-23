#!/usr/bin/env python3
"""Why did the §4.2 identity-BCE pretext pretraining fail? (SPEC 0048)

Diagnosis harness for the negative pretraining result recorded in
``labbook/2026-06-01_identity_bce_negative_result.md``. It rebuilds the SSL
phase of ``src/mini_trackastra/end_to_end_bce.py`` faithfully (same data path,
model, loss, steps) while recording the information that run discarded, and
trains controlled variants to test the ranked hypotheses of SPEC 0048:

- V0 ``as_published``: view-2 labels taken as the first n2 base labels (the
  dropout-survivor labels are discarded there), view-2 appearance sampled from
  the clean image at warped coordinates — the exact published behaviour.
- V1 ``fixed_labels``: true survivor labels, appearance still as published.
- V2 ``fixed_labels_warped_appearance``: true survivor labels and view-2
  appearance extracted from an image warped by the same transform.

Modes: G = FourierPE (coordinates visible), F = NoPE (appearance only).

Diagnostics per (variant, mode): SSL loss trajectory; pretext accuracy against
the trained-with (diagonal) and the true (survivor) labels; output mappings
(argmax partner per view-1 cell) stratified by view-2 index position, dropout
status and true-partner displacement; partner-distribution concentration
(collapse check); positive-vs-negative logit spread. Plus a downstream
early-transfer probe (val accuracy at steps 0/10/30 for each pretrained init
vs random init) and one 5x-longer SSL run (undertraining check).

Reproduce (from the repository root)::

    benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_failure_diagnosis.py

Outputs ``benchmark_ssl/results/pretext_failure_diagnosis/diagnosis.json`` and
``diagnosis.png``. Deterministic (numpy seed 42; torch seeded per run — the
original script never seeded torch, which is itself recorded as a finding).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

BENCHMARK_SSL_ROOT = Path(__file__).resolve().parents[2]
if str(BENCHMARK_SSL_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SSL_ROOT))

from src.mini_trackastra.end_to_end import (  # noqa: E402
    SEED,
    device,
    scan_frames,
    load_frame,
    extract_patches,
    compute_dino_embs,
    free_dino,
    FourierPE,
    NoPE,
    MiniEncoder,
    MiniDecoder,
    MiniTrackingTransformer,
    bce_assoc_loss,
    assoc_accuracy,
    balanced_accuracy,
)

from scipy.ndimage import affine_transform  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "pretext_failure_diagnosis"

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "vanvliet"
CONDITIONS = ["rpsM"]
N_SSL_FRAMES = 30
SSL_STEPS = 200
LONG_SSL_STEPS = 1000
DOWNSTREAM_PAIRS = 15
DOWNSTREAM_STEPS = 30
D_MODEL = 128
NHEAD = 4
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
D_HEAD = 32
POS_PER_DIM = 16
PE_DIM = POS_PER_DIM * 2 * 2
SSL_LR = 1e-3
DOWNSTREAM_LR = 1e-3
MIN_CELLS = 8

VARIANTS = ("as_published", "fixed_labels", "fixed_labels_warped_appearance",
            "warped_appearance_as_published_labels")
MODES = ("G", "F")


# ─── Distortion (numerically identical to end_to_end.apply_affine/jitter/
#     dropout for mode "full", but returning the transform and true labels) ──

def distort_full_with_transform(coords: np.ndarray, labels: np.ndarray):
    """Apply the pretext's "full" distortion; return warp, survivor labels.

    Draw order matches ``end_to_end.py`` exactly (angle, scale_x, scale_y,
    jitter noise, dropout mask) so the distorted coordinates are identical to
    the pretext pipeline's for the same random stream.
    """
    theta = np.random.uniform(-10, 10) / 180 * np.pi
    scale_x, scale_y = np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)
    transform = np.array(
        [[scale_x * np.cos(theta), -scale_x * np.sin(theta)],
         [scale_y * np.sin(theta), scale_y * np.cos(theta)]]
    )
    warped = coords @ transform
    warped = warped + np.random.randn(*warped.shape).astype(np.float32) * 4
    keep = np.random.rand(len(labels)) > 0.1
    return warped[keep], labels[keep], transform


def warp_image(img: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Resample the image under the inverse coordinate warp.

    The coordinate warp is ``q = p @ M`` (row vectors), i.e. ``q_col = M^T p``;
    the consistent image warp is ``W(q) = img(M^{-T} q)``, which
    ``scipy.ndimage.affine_transform`` expresses with matrix ``inv(M).T``.
    A patch extracted from ``W`` at the warped coordinate ``q`` therefore shows
    the same cell content as the clean patch at ``p`` (resampled under the
    inverse warp).
    """
    return affine_transform(img, np.linalg.inv(transform).T, order=1,
                            mode="constant", cval=0.0, prefilter=False)


# ─── SSL data rebuild (records what the published run discarded) ─────────────

def build_ssl_frames(data_root: Path, conditions: list[str],
                     max_frames: int) -> list[dict]:
    """Rebuild the SSL frame pairs with true survivor labels and transforms.

    Per frame this records: clean view (features, coords, labels), the
    distorted view's features sampled the published way (clean image at warped
    coordinates) and the corrected way (warped image at warped coordinates),
    the true survivor labels, the survivor map (base index -> view-2 index,
    i.e. the position of the base cell's label among the survivors; -1 if
    dropped), and the per-cell displacement of the true partner.
    """
    np.random.seed(SEED)
    frame_paths = scan_frames(str(data_root), conditions, max_frames)
    frames = []
    for mask_path, img_path in frame_paths:
        loaded = load_frame(mask_path, img_path)
        if loaded is None:
            continue
        coords, labels, img = loaded
        warped, survivor_labels, transform = distort_full_with_transform(
            coords.copy(), labels.copy())
        warped_img = warp_image(img, transform)

        patches_clean = extract_patches(img, coords)
        patches_warped_published = extract_patches(img, warped)
        patches_warped_correct = extract_patches(warped_img, warped)
        features_clean = compute_dino_embs(patches_clean)
        features_warped_published = compute_dino_embs(patches_warped_published)
        features_warped_correct = compute_dino_embs(patches_warped_correct)

        n = min(len(features_clean), len(features_warped_published))
        if n < 2:
            continue
        # The published run truncates both views to n and keeps the first n
        # cells of each; the survivor map is relative to the truncated view-1.
        survivor_of = np.array(
            [int(np.where(survivor_labels == labels[j])[0][0])
             if labels[j] in survivor_labels else -1
             for j in range(n)])
        true_partner_displacement = np.array([
            float(np.linalg.norm(warped[survivor_of[i]] - coords[i]))
            if survivor_of[i] >= 0 else float("nan")
            for i in range(n)])
        frames.append({
            "features_clean": features_clean[:n],
            "features_warped_published": features_warped_published[:n],
            "features_warped_correct": features_warped_correct[:n],
            "coords_clean": coords[:n],
            "coords_warped": warped[:n],
            "labels_clean": labels[:n],
            "labels_warped_true": survivor_labels,
            "survivor_of": survivor_of,
            "true_partner_displacement": true_partner_displacement,
            "transform": transform,
        })
    free_dino()
    return frames


def analytic_label_corruption(frames: list[dict]) -> dict:
    """Quantify the published label bug from the recorded survivor maps.

    A sample is corrupted when the first dropped base cell lies before the
    truncated view length (then every view-2 cell after that index carries the
    wrong positive partner). ``corrupted_positive_fraction`` is the fraction of
    view-2 cells whose survivor map is not the identity.
    """
    corrupted_samples, corrupted_positives, total_positives = 0, 0, 0
    for frame in frames:
        survivor_of = frame["survivor_of"]
        total_positives += len(survivor_of)
        mismatches = int((survivor_of != np.arange(len(survivor_of))).sum())
        corrupted_positives += mismatches
        if mismatches > 0:
            corrupted_samples += 1
    return {
        "n_frames": len(frames),
        "corrupted_sample_fraction": corrupted_samples / max(1, len(frames)),
        "corrupted_positive_fraction": corrupted_positives / max(1, total_positives),
    }


# ─── SSL training (faithful to run_ssl_bce_phase, parameterised) ─────────────

def features_for_variant(frame: dict, variant: str) -> np.ndarray:
    """View-2 features for a variant: published (clean image) or warped image."""
    if variant in ("fixed_labels_warped_appearance",
                   "warped_appearance_as_published_labels"):
        return frame["features_warped_correct"]
    return frame["features_warped_published"]


def labels_for_variant(frame: dict, variant: str) -> np.ndarray:
    """View-2 labels for a variant: first-n diagonal (published) or survivors."""
    if variant in ("as_published", "warped_appearance_as_published_labels"):
        return frame["labels_clean"][: len(frame["features_warped_published"])]
    return frame["labels_warped_true"][: len(frame["features_warped_published"])]


def build_model() -> tuple:
    """Construct (encoder, decoder, model, pe) exactly as the published run."""
    fourier_pe = FourierPE(pos_per_dim=POS_PER_DIM).to(device)
    noise_pe = NoPE(PE_DIM).to(device)
    encoder = MiniEncoder(d_model=D_MODEL, nhead=NHEAD,
                          num_layers=N_ENCODER_LAYERS, pe_dim=PE_DIM).to(device)
    decoder = MiniDecoder(d_model=D_MODEL, nhead=NHEAD,
                          num_layers=N_DECODER_LAYERS).to(device)
    model = MiniTrackingTransformer(encoder=encoder, decoder=decoder,
                                    d_head=D_HEAD).to(device)
    return encoder, decoder, model, {"G": fourier_pe, "F": noise_pe}


def train_ssl(frames: list[dict], mode: str, variant: str,
              steps: int) -> dict:
    """Train the mini model on the pretext exactly like run_ssl_bce_phase.

    Faithful loop: per step, a random frame order; per frame, BCE on the
    association logits with the variant's labels; grad clip 1.0; Adam at 1e-3.
    Torch is seeded per run (the published script never seeded torch).
    """
    torch.manual_seed(SEED)
    encoder, decoder, model, pe_modules = build_model()
    pe = pe_modules[mode]
    optimizer = torch.optim.Adam(model.parameters(), lr=SSL_LR)

    gpu_frames = []
    for frame in frames:
        features_2 = features_for_variant(frame, variant)
        labels_2 = labels_for_variant(frame, variant)
        gpu_frames.append((
            torch.from_numpy(frame["features_clean"]).float().to(device),
            torch.from_numpy(features_2).float().to(device),
            torch.from_numpy(frame["coords_clean"]).float().unsqueeze(0).to(device),
            torch.from_numpy(frame["coords_warped"]).float().unsqueeze(0).to(device),
            frame["labels_clean"],
            labels_2,
        ))

    losses = []
    for step in range(steps):
        model.train()
        loss_total, n_batches = 0.0, 0
        perm = torch.randperm(len(gpu_frames))
        for idx in perm:
            feats_1, feats_2, coords_1, coords_2, labels_1, labels_2 = gpu_frames[idx]
            pe_1 = pe(coords_1).squeeze(0)
            pe_2 = pe(coords_2).squeeze(0)
            logits = model(feats_1, pe_1, feats_2, pe_2)
            loss = bce_assoc_loss(logits, labels_1, labels_2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item()
            n_batches += 1
        losses.append(loss_total / max(1, n_batches))
    return {"encoder": encoder, "decoder": decoder, "model": model,
            "pe": pe, "losses": losses}


# ─── Output-mapping diagnostics ──────────────────────────────────────────────

def output_mapping_diagnostics(trained: dict, frames: list[dict],
                               variant: str) -> dict:
    """What does the trained model actually predict on the pretext?

    Reports accuracy against the trained-with target (diagonal for
    ``as_published``, survivor labels otherwise), accuracy against the TRUE
    identity (argmax partner's survivor equals the view-1 index), the
    prefix/suffix accuracy profile (H1 signature), displacement stratification,
    partner-distribution concentration (collapse), and logit separation.
    """
    trained_model = trained["model"]
    pe = trained["pe"]
    trained_model.eval()
    diag_correct = diag_total = 0
    cellwise_correct = cellwise_total = 0
    true_correct_by_position = {third: [0, 0] for third in ("first", "middle", "last")}
    true_correct_by_displacement = {bucket: [0, 0] for bucket in ("near", "mid", "far")}
    true_correct_by_dropped = {"survivor": [0, 0], "dropped": [0, 0]}
    true_correct = true_total = 0
    partner_counts: dict[int, int] = {}
    positive_logits, negative_logits = [], []

    with torch.no_grad():
        for frame in frames:
            features_2 = features_for_variant(frame, variant)
            labels_2 = labels_for_variant(frame, variant)
            n = len(features_2)
            logits = trained_model(
                torch.from_numpy(frame["features_clean"]).float().to(device),
                pe(torch.from_numpy(frame["coords_clean"]).float()
                   .unsqueeze(0).to(device)).squeeze(0),
                torch.from_numpy(features_2).float().to(device),
                pe(torch.from_numpy(frame["coords_warped"]).float()
                   .unsqueeze(0).to(device)).squeeze(0),
            )
            logits_cpu = logits.cpu().numpy()
            prediction = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            diagonal = np.eye(n, dtype=bool)
            diag_correct += int((prediction == diagonal).sum())
            diag_total += prediction.size

            # Cellwise accuracy against the TRUE survivor-label target,
            # built directly from label equality (base label i must pair
            # with the view-2 cell carrying the same label).
            labels_base = frame["labels_clean"][:n]
            labels_survivor = frame["labels_warped_true"][:n]
            target_true = labels_base[:, None] == labels_survivor[None, :]
            cellwise_correct += int((prediction == target_true).sum())
            cellwise_total += prediction.size

            argmax_partners = logits_cpu.argmax(axis=1)
            sigmoid = 1.0 / (1.0 + np.exp(-logits_cpu))
            labels_base = frame["labels_clean"][:n]
            labels_survivor = frame["labels_warped_true"][:n]
            survivor_set = set(int(label) for label in labels_survivor)
            for i in range(n):
                partner = int(argmax_partners[i])
                partner_counts[partner] = partner_counts.get(partner, 0) + 1
                # Correctness by LABEL EQUALITY: view-1 cell i matches the
                # view-2 cell carrying the same base label. (Index-based
                # checks via the survivor map are error-prone: survivor_of is
                # the FORWARD map base->view-2.)
                if int(labels_base[i]) in survivor_set:
                    # Survivor row: top-1 must hit the cell's true partner.
                    correct = int(labels_survivor[partner] == labels_base[i])
                    true_correct += correct
                    true_total += 1
                    third = ("first" if i < n / 3
                             else "last" if i >= 2 * n / 3 else "middle")
                    true_correct_by_position[third][0] += correct
                    true_correct_by_position[third][1] += 1
                    displacement = frame["true_partner_displacement"][i]
                    if not np.isnan(displacement):
                        bucket = ("near" if displacement < 30
                                  else "far" if displacement > 100 else "mid")
                        true_correct_by_displacement[bucket][0] += correct
                        true_correct_by_displacement[bucket][1] += 1
                    true_correct_by_dropped["survivor"][0] += correct
                    true_correct_by_dropped["survivor"][1] += 1
                else:
                    # Dropped row: correct behaviour is rejecting every
                    # candidate (argmax always points somewhere, so the fair
                    # metric is whether the best sigmoid stays below 0.5).
                    rejected = int(sigmoid[i].max() < 0.5)
                    true_correct_by_dropped["dropped"][0] += rejected
                    true_correct_by_dropped["dropped"][1] += 1
                positive_logits.append(float(logits_cpu[i, partner]))
                off = np.delete(logits_cpu[i], partner)
                negative_logits.append(float(off.max()))

    def rate(entry) -> float | None:
        return entry[0] / entry[1] if entry[1] else None

    counts = np.array(sorted(partner_counts.values(), reverse=True))
    return {
        "accuracy_trained_target": diag_correct / max(1, diag_total),
        "accuracy_true_identity_cellwise": cellwise_correct / max(1, cellwise_total),
        "accuracy_true_identity_argmax": true_correct / max(1, true_total),
        "accuracy_by_position": {third: rate(entry)
                                 for third, entry in true_correct_by_position.items()},
        "accuracy_by_displacement": {bucket: rate(entry)
                                     for bucket, entry in true_correct_by_displacement.items()},
        "accuracy_survivor_rows": rate(true_correct_by_dropped["survivor"]),
        "accuracy_dropped_rows": rate(true_correct_by_dropped["dropped"]),
        "partner_concentration_top1_share": float(counts[0] / counts.sum())
            if counts.size else None,
        "partner_distinct_fraction": (len(partner_counts) / true_total)
            if true_total else None,
        "mean_logit_true_partner": float(np.mean(positive_logits)),
        "mean_logit_best_off_target": float(np.mean(negative_logits)),
    }


# ─── Downstream early-transfer probe ─────────────────────────────────────────

def build_downstream_pairs(data_root: Path, max_pairs: int) -> dict:
    """Rebuild the downstream pair data exactly like end_to_end_bce.main()."""
    from src.mini_trackastra.end_to_end import scan_consecutive_pairs

    np.random.seed(SEED + 1)
    pairs = scan_consecutive_pairs(str(data_root), CONDITIONS, max_pairs + 5)
    idx = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * 0.2))
    splits = {"train": [pairs[i] for i in idx[:-n_val]],
              "val": [pairs[i] for i in idx[-n_val:]]}
    pair_data = {"train": [], "val": []}
    for split_name, split_pairs in splits.items():
        for mask_t, mask_n, img_t, img_n in split_pairs:
            result_t = load_frame(mask_t, img_t)
            result_n = load_frame(mask_n, img_n)
            if result_t is None or result_n is None:
                continue
            coords_t, labels_t, image_t = result_t
            coords_n, labels_n, image_n = result_n
            shared = set(labels_t) & set(labels_n)
            if len(shared) < MIN_CELLS or len(labels_t) < MIN_CELLS \
                    or len(labels_n) < MIN_CELLS:
                continue
            idx_t = [i for i, l in enumerate(labels_t) if l in shared]
            idx_n = [i for i, l in enumerate(labels_n) if l in shared]
            coords_t_s, labels_t_s = coords_t[idx_t], labels_t[idx_t]
            coords_n_s, labels_n_s = coords_n[idx_n], labels_n[idx_n]
            pair_data[split_name].append({
                "dino_t": torch.from_numpy(
                    compute_dino_embs(extract_patches(image_t, coords_t_s))).float(),
                "dino_n": torch.from_numpy(
                    compute_dino_embs(extract_patches(image_n, coords_n_s))).float(),
                "coords_t": torch.from_numpy(coords_t_s).float(),
                "coords_n": torch.from_numpy(coords_n_s).float(),
                "labels_t": torch.from_numpy(labels_t_s).long(),
                "labels_n": torch.from_numpy(labels_n_s).long(),
            })
    free_dino()
    return pair_data


def downstream_probe(init: dict | None, pair_data: dict, steps: int) -> dict:
    """Val accuracy at steps 0/10/... for a pretrained (or random) init.

    Replicates run_downstream_bce's loop (Adam 1e-3, BCE) for a short probe;
    step 0 evaluates the untouched init — the cleanest transfer signal.
    """
    torch.manual_seed(SEED)
    encoder, decoder, model, _ = build_model()
    pe = init["pe"] if init is not None else FourierPE(pos_per_dim=POS_PER_DIM).to(device)
    if init is not None:
        encoder.load_state_dict(init["encoder"].state_dict())
        decoder.load_state_dict(init["decoder"].state_dict())
        model.load_state_dict(init["model"].state_dict(), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=DOWNSTREAM_LR)

    def evaluate() -> tuple[float, float, float]:
        model.eval()
        losses, accs, bal_accs = [], [], []
        with torch.no_grad():
            for item in pair_data["val"]:
                logits = model(
                    item["dino_t"].to(device),
                    pe(item["coords_t"].unsqueeze(0).to(device)).squeeze(0),
                    item["dino_n"].to(device),
                    pe(item["coords_n"].unsqueeze(0).to(device)).squeeze(0),
                )
                losses.append(bce_assoc_loss(
                    logits, item["labels_t"], item["labels_n"]).item())
                accs.append(assoc_accuracy(logits, item["labels_t"], item["labels_n"]))
                bal_accs.append(balanced_accuracy(
                    logits, item["labels_t"], item["labels_n"]))
        return float(np.mean(losses)), float(np.mean(accs)), float(np.mean(bal_accs))

    trajectory = {0: evaluate()}
    for step in range(1, steps + 1):
        model.train()
        for item in pair_data["train"]:
            logits = model(
                item["dino_t"].to(device),
                pe(item["coords_t"].unsqueeze(0).to(device)).squeeze(0),
                item["dino_n"].to(device),
                pe(item["coords_n"].unsqueeze(0).to(device)).squeeze(0),
            )
            loss = bce_assoc_loss(logits, item["labels_t"], item["labels_n"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % 10 == 0 or step == steps:
            trajectory[step] = evaluate()
    return {str(step): {"val_loss": values[0], "val_acc": values[1],
                        "val_bal_acc": values[2]}
            for step, values in trajectory.items()}


# ─── Orchestration ───────────────────────────────────────────────────────────

def make_figure(results: dict, out_path: Path) -> None:
    """Render the diagnosis summary figure from the collected results.

    Panel 1: final SSL loss per (variant, mode). Panel 2: true-identity argmax
    accuracy per (variant, mode) with the all-negative floor visible as the
    ~0.1 plateau of the broken variants. Panel 3: displacement stratification
    of the corrected variant (accuracy decays with warp displacement).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = results["runs"]
    keys = [key for key in runs if "/long" not in key]
    labels = [key.replace("fixed_labels_warped_appearance", "V2")
               .replace("warped_appearance_as_published_labels", "V3")
               .replace("fixed_labels", "V1").replace("as_published", "V0")
              for key in keys]
    final_losses = [runs[key]["final_ssl_loss"] for key in keys]
    true_accs = [runs[key]["diagnostics"]["accuracy_true_identity_argmax"]
                 for key in keys]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].bar(range(len(keys)), final_losses, color="#4472c4")
    axes[0].set_xticks(range(len(keys)), labels, rotation=45, ha="right",
                       fontsize=7)
    axes[0].axhline(0.13, color="crimson", ls="--", lw=1,
                    label="achievable (0.13)")
    axes[0].set_ylabel("final SSL BCE loss")
    axes[0].legend(fontsize=7)
    axes[0].set_title("SSL loss plateau", fontsize=9)

    axes[1].bar(range(len(keys)), true_accs, color="#ed7d31")
    axes[1].set_xticks(range(len(keys)), labels, rotation=45, ha="right",
                       fontsize=7)
    axes[1].set_ylabel("true-identity argmax accuracy")
    axes[1].axhline(1.0 / 40, color="grey", ls=":", lw=1, label="chance (~1/40)")
    axes[1].legend(fontsize=7)
    axes[1].set_title("pretext output mappings", fontsize=9)

    displacement = runs["fixed_labels_warped_appearance/G"]["diagnostics"][
        "accuracy_by_displacement"]
    buckets = ["near", "mid", "far"]
    axes[2].bar(buckets, [displacement[b] for b in buckets], color="#70ad47")
    axes[2].set_ylabel("true-identity accuracy")
    axes[2].set_xlabel("true-partner displacement (px)")
    axes[2].set_title("V2/G: accuracy vs displacement", fontsize=9)

    fig.suptitle("Identity-BCE pretext failure diagnosis (SPEC 0048)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    """CLI: --figure-only rebuilds the PNG from an existing diagnosis.json."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Identity-BCE pretext failure diagnosis (SPEC 0048)")
    parser.add_argument("--figure-only", action="store_true",
                        help="rebuild diagnosis.png from diagnosis.json "
                             "without retraining")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run all variants, collect diagnostics, write JSON + figure, print verdict."""
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        results = json.loads((RESULTS_DIR / "diagnosis.json").read_text())
        make_figure(results, RESULTS_DIR / "diagnosis.png")
        print(f"Wrote {RESULTS_DIR / 'diagnosis.png'}")
        return 0
    print(f"device={device}")
    print("Building SSL frames (DINO embedding)...")
    frames = build_ssl_frames(DATA_ROOT, CONDITIONS, N_SSL_FRAMES)
    print(f"frames={len(frames)}")
    corruption = analytic_label_corruption(frames)
    print(f"label corruption (published target): {corruption}")

    results = {"corruption": corruption, "runs": {}, "downstream": {}}
    trained_runs: dict[tuple[str, str], dict] = {}
    for variant in VARIANTS:
        for mode in MODES:
            print(f"Training SSL {variant}/{mode} ({SSL_STEPS} steps)...")
            trained = train_ssl(frames, mode, variant, SSL_STEPS)
            trained_runs[(variant, mode)] = trained
            results["runs"][f"{variant}/{mode}"] = {
                "final_ssl_loss": trained["losses"][-1],
                "loss_first5": trained["losses"][:5],
                "diagnostics": output_mapping_diagnostics(trained, frames, variant),
            }
            print(f"  final loss {trained['losses'][-1]:.4f}  "
                  f"true-identity argmax acc "
                  f"{results['runs'][f'{variant}/{mode}']['diagnostics']['accuracy_true_identity_argmax']:.3f}")

    # Undertraining check (H4): same variant, 5x longer.
    print(f"Training SSL fixed_labels/G ({LONG_SSL_STEPS} steps)...")
    long_trained = train_ssl(frames, "G", "fixed_labels", LONG_SSL_STEPS)
    results["runs"]["fixed_labels/G/long"] = {
        "steps": LONG_SSL_STEPS,
        "final_ssl_loss": long_trained["losses"][-1],
        "diagnostics": output_mapping_diagnostics(long_trained, frames, "fixed_labels"),
    }

    # Downstream early-transfer probe (H3).
    print("Building downstream pairs...")
    pair_data = build_downstream_pairs(DATA_ROOT, DOWNSTREAM_PAIRS)
    print(f"pairs: {len(pair_data['train'])} train / {len(pair_data['val'])} val")
    if not pair_data["val"] or not pair_data["train"]:
        results["downstream"]["note"] = (
            "skipped: no downstream pairs passed the min-cells filter")
        print(results["downstream"]["note"])
    else:
        results["downstream"]["random_init"] = downstream_probe(
            None, pair_data, DOWNSTREAM_STEPS)
        for variant in VARIANTS:
            key = f"{variant}/G"
            if (variant, "G") in trained_runs:
                results["downstream"][key] = downstream_probe(
                    trained_runs[(variant, "G")], pair_data, DOWNSTREAM_STEPS)

    out_json = RESULTS_DIR / "diagnosis.json"
    out_json.write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {out_json}")

    # Console verdict per hypothesis.
    runs = results["runs"]
    print("\n=== VERDICT INPUTS (SPEC 0048 H1-H5) ===")
    print(f"H1 corruption: {corruption}")
    print(f"H1 (labels, appearance corrupted): as_published/G true-acc "
          f"{runs['as_published/G']['diagnostics']['accuracy_true_identity_argmax']:.3f} "
          f"vs fixed_labels/G "
          f"{runs['fixed_labels/G']['diagnostics']['accuracy_true_identity_argmax']:.3f}")
    print(f"H1 (labels, appearance fixed): fixed_labels_warped_appearance/G "
          f"{runs['fixed_labels_warped_appearance/G']['diagnostics']['accuracy_true_identity_argmax']:.3f} "
          f"vs warped_appearance_as_published_labels/G "
          f"{runs['warped_appearance_as_published_labels/G']['diagnostics']['accuracy_true_identity_argmax']:.3f}")
    print(f"H2 (appearance): fixed_labels/F "
          f"{runs['fixed_labels/F']['diagnostics']['accuracy_true_identity_argmax']:.3f} "
          f"vs fixed_labels_warped_appearance/F "
          f"{runs['fixed_labels_warped_appearance/F']['diagnostics']['accuracy_true_identity_argmax']:.3f}")
    print(f"H4 (undertraining): fixed_labels/G {SSL_STEPS} steps "
          f"{runs['fixed_labels/G']['diagnostics']['accuracy_true_identity_argmax']:.3f} "
          f"vs {LONG_SSL_STEPS} steps "
          f"{runs['fixed_labels/G/long']['diagnostics']['accuracy_true_identity_argmax']:.3f}")
    for key, trajectory in results["downstream"].items():
        step0 = trajectory["0"]
        print(f"H3 downstream {key}: step0 val_acc {step0['val_acc']:.3f} "
              f"bal_acc {step0['val_bal_acc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
