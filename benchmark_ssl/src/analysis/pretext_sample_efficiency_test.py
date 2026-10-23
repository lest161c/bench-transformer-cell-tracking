#!/usr/bin/env python3
"""ONE decisive run: does corrected pretext pretraining improve sample efficiency?

Question (owner, 2026-10-23): a single run that proves the corrected §4.2
identity-BCE pretext either improves low-label downstream association or does
not; if not, the pretext line is discarded.

Pre-registered design and verdict rule (fixed before running):

- Downstream condition: ``trpL`` (excluding the quality-flagged experiment
  ``150303-08_BAD``), consecutive frame pairs with >= 8 shared track labels.
  Label budget K = 5 train pairs; up to 10 further pairs are validation
  (labels used for scoring only, never for training).
- Pretraining pool: unlabeled DENSE frames from the SAME condition (the
  realistic deployment: the videos exist, only annotations are scarce),
  EXCLUDING the exact mask files used as downstream train/val pairs and the
  quality-flagged experiment. Frames are free (segmentation masks exist, no
  track annotations needed); annotated windows are the scarce resource. This
  answers the "just train more epochs on the same data" objection: the
  pretraining consumes zero annotations.
- Corrected recipe (SPEC 0048 fixes): view-2 appearance from the warped
  image, dropout survivor labels as ground truth, affine included.
- Arms:
  A  random init   + finetune K=5 pairs, 2000 steps  (label-matched baseline)
  B  SSL init      + finetune K=5 pairs, 2000 steps  (the claim)
  C  random init   + finetune K=5 pairs, 2300 steps  (compute-matched control:
     total updates ~= B's 300 SSL + 2000 finetune)
- VERDICT (pre-registered): the pretext WORKS iff
      bal_acc(B, final) - bal_acc(A, final) >= +0.10   (clear low-label gain)
  AND bal_acc(B, final) >= bal_acc(C, final) - 0.02   (not explained by extra
  labeled-update epochs). Otherwise it does NOT justify full training and the
  line is discarded.

Single seed (numpy 42; torch seeded per arm), single run — no sweeps by owner
directive. Reproduce (from the repository root)::

    benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_sample_efficiency_test.py

Outputs ``benchmark_ssl/results/runs/pretext_sample_efficiency_test/result.json``
and prints the verdict. Runs on the local GPU in well under 30 minutes.
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
    distort_with_transform,
    warp_image,
    FourierPE,
    MiniEncoder,
    MiniDecoder,
    MiniTrackingTransformer,
    bce_assoc_loss,
    assoc_accuracy,
    balanced_accuracy,
)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "vanvliet"
RESULTS_DIR = (Path(__file__).resolve().parents[2] / "results" / "runs"
               / "pretext_sample_efficiency_test")

ALL_CONDITIONS = ["trpL"]
DOWNSTREAM_CONDITION = "trpL"
EXCLUDED_EXPERIMENTS = {"150303-08_BAD"}

POOL_FRAMES_PER_CONDITION = 400
POOL_MAX_FRAMES = 60
MIN_POOL_CELLS = 8
SSL_STEPS = 300
FINETUNE_STEPS = 2000
COMPUTED_MATCHED_STEPS = SSL_STEPS + FINETUNE_STEPS
K_TRAIN_PAIRS = 5
MAX_VAL_PAIRS = 10
MIN_CELLS = 8

D_MODEL = 128
NHEAD = 4
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
D_HEAD = 32
POS_PER_DIM = 16
PE_DIM = POS_PER_DIM * 2 * 2
SSL_LR = 1e-3
DOWNSTREAM_LR = 1e-3


def build_model():
    """Construct (encoder, decoder, model) with a fresh FourierPE outside."""
    encoder = MiniEncoder(d_model=D_MODEL, nhead=NHEAD,
                          num_layers=N_ENCODER_LAYERS, pe_dim=PE_DIM).to(device)
    decoder = MiniDecoder(d_model=D_MODEL, nhead=NHEAD,
                          num_layers=N_DECODER_LAYERS).to(device)
    model = MiniTrackingTransformer(encoder=encoder, decoder=decoder,
                                    d_head=D_HEAD).to(device)
    return encoder, decoder, model


def build_ssl_pool(excluded_masks: set[str]) -> list[dict]:
    """Unlabeled pretraining frames from all conditions, corrected recipe.

    Skips frames with fewer than MIN_POOL_CELLS cells (sparse early-lag frames
    make the pretext trivial and dominate the loss average — the first run's
    pool had a median of 3 cells/frame and the pretext never learned), skips
    the quality-flagged experiments, and skips every frame whose mask file is
    used as a downstream train/val pair. Each remaining frame contributes a
    (clean view, warped view) pair: view-2 appearance from the warped image,
    survivor labels as ground truth (SPEC 0048 fixes).
    """
    np.random.seed(SEED)
    pool = []
    for condition in ALL_CONDITIONS:
        if len(pool) >= POOL_MAX_FRAMES:
            break
        frames = scan_frames(str(DATA_ROOT), [condition],
                             POOL_FRAMES_PER_CONDITION)
        for mask_path, img_path in frames:
            if len(pool) >= POOL_MAX_FRAMES:
                break
            if str(mask_path) in excluded_masks:
                continue
            experiment = Path(mask_path).parent.parent.name
            if experiment in EXCLUDED_EXPERIMENTS:
                continue
            loaded = load_frame(mask_path, img_path)
            if loaded is None or len(loaded[0]) < MIN_POOL_CELLS:
                continue
            coords, labels, img = loaded
            warped, survivor_labels, transform = distort_with_transform(
                coords.copy(), labels.copy(), "full")
            warped_img = warp_image(img, transform)
            features_clean = compute_dino_embs(extract_patches(img, coords))
            features_warped = compute_dino_embs(extract_patches(warped_img, warped))
            n = min(len(features_clean), len(features_warped))
            if n < 2:
                continue
            pool.append({
                "mask_path": str(mask_path),
                "features_clean": features_clean[:n],
                "features_warped": features_warped[:n],
                "coords_clean": coords[:n],
                "coords_warped": warped[:n],
                "labels_clean": labels[:n],
                "labels_warped": survivor_labels,
            })
    free_dino()
    return pool


def train_ssl(pool: list[dict], steps: int) -> dict:
    """Train the mini model on the corrected pretext over the pool."""
    torch.manual_seed(SEED)
    encoder, decoder, model = build_model()
    pe = FourierPE(pos_per_dim=POS_PER_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SSL_LR)
    gpu_pool = [(torch.from_numpy(f["features_clean"]).float().to(device),
                 torch.from_numpy(f["features_warped"]).float().to(device),
                 torch.from_numpy(f["coords_clean"]).float().unsqueeze(0).to(device),
                 torch.from_numpy(f["coords_warped"]).float().unsqueeze(0).to(device),
                 f["labels_clean"], f["labels_warped"]) for f in pool]
    losses = []
    for step in range(steps):
        model.train()
        loss_total, n_batches = 0.0, 0
        for idx in torch.randperm(len(gpu_pool)):
            feats_1, feats_2, coords_1, coords_2, labels_1, labels_2 = gpu_pool[idx]
            logits = model(feats_1, pe(coords_1).squeeze(0),
                           feats_2, pe(coords_2).squeeze(0))
            loss = bce_assoc_loss(logits, labels_1, labels_2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item()
            n_batches += 1
        losses.append(loss_total / max(1, n_batches))
        if (step + 1) % 50 == 0:
            print(f"  SSL step {step + 1}: loss {losses[-1]:.4f}")
    return {"encoder": encoder, "decoder": decoder, "model": model, "pe": pe,
            "losses": losses}


def build_downstream_pairs() -> tuple[list[dict], list[dict], set[str]]:
    """K labeled train pairs + validation pairs from the held-out condition.

    Also returns the mask paths of the used frames so the pretext pool can
    exclude them.
    """
    from src.mini_trackastra.end_to_end import scan_consecutive_pairs

    np.random.seed(SEED + 1)
    candidates = []
    for mask_t, mask_n, img_t, img_n in scan_consecutive_pairs(
            str(DATA_ROOT), [DOWNSTREAM_CONDITION], 60):
        experiment = Path(mask_t).parent.parent.name
        if experiment in EXCLUDED_EXPERIMENTS:
            continue
        candidates.append((mask_t, mask_n, img_t, img_n))

    train_pairs, val_pairs = [], []
    excluded_masks: set[str] = set()
    for mask_t, mask_n, img_t, img_n in candidates:
        if len(train_pairs) >= K_TRAIN_PAIRS and len(val_pairs) >= MAX_VAL_PAIRS:
            break
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
        item = {
            "mask_t": str(mask_t), "mask_n": str(mask_n),
            "dino_t": torch.from_numpy(compute_dino_embs(
                extract_patches(image_t, coords_t))).float(),
            "dino_n": torch.from_numpy(compute_dino_embs(
                extract_patches(image_n, coords_n))).float(),
            "coords_t": torch.from_numpy(coords_t).float(),
            "coords_n": torch.from_numpy(coords_n).float(),
            "labels_t": labels_t, "labels_n": labels_n,
        }
        if len(train_pairs) < K_TRAIN_PAIRS:
            train_pairs.append(item)
        else:
            val_pairs.append(item)
        excluded_masks.add(str(mask_t))
        excluded_masks.add(str(mask_n))
    free_dino()
    return train_pairs, val_pairs, excluded_masks


def finetune_and_evaluate(init: dict | None, train_pairs: list[dict],
                          val_pairs: list[dict], steps: int,
                          eval_points: tuple[int, ...]) -> dict:
    """Finetune on the K labeled pairs; evaluate bal-acc/loss on val."""
    torch.manual_seed(SEED)
    encoder, decoder, model = build_model()
    pe = FourierPE(pos_per_dim=POS_PER_DIM).to(device)
    if init is not None:
        encoder.load_state_dict(init["encoder"].state_dict())
        decoder.load_state_dict(init["decoder"].state_dict())
        model.load_state_dict(init["model"].state_dict(), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=DOWNSTREAM_LR)

    def evaluate() -> tuple[float, float]:
        model.eval()
        bal_accs, losses = [], []
        with torch.no_grad():
            for item in val_pairs:
                logits = model(
                    item["dino_t"].to(device),
                    pe(item["coords_t"].unsqueeze(0).to(device)).squeeze(0),
                    item["dino_n"].to(device),
                    pe(item["coords_n"].unsqueeze(0).to(device)).squeeze(0),
                )
                losses.append(bce_assoc_loss(
                    logits, item["labels_t"], item["labels_n"]).item())
                bal_accs.append(balanced_accuracy(
                    logits, item["labels_t"], item["labels_n"]))
        return float(np.mean(bal_accs)), float(np.mean(losses))

    trajectory = {}
    for step in range(steps + 1):
        if step in eval_points:
            bal_acc, loss = evaluate()
            trajectory[str(step)] = {"val_bal_acc": bal_acc, "val_loss": loss}
        if step == steps:
            break
        model.train()
        for item in train_pairs:
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
    return trajectory


def main() -> int:
    """Run arms A/B/C and print the pre-registered verdict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={device}")

    print("Building downstream pairs (held-out condition "
          f"{DOWNSTREAM_CONDITION})...")
    train_pairs, val_pairs, excluded_masks = build_downstream_pairs()
    print(f"train pairs={len(train_pairs)} (label budget K={K_TRAIN_PAIRS}), "
          f"val pairs={len(val_pairs)}")
    if len(train_pairs) < K_TRAIN_PAIRS or len(val_pairs) < 2:
        print("INSUFFICIENT downstream pairs; aborting without a verdict.")
        return 1

    print("Building SSL pool (unlabeled frames, all conditions, "
          "downstream frames excluded)...")
    pool = build_ssl_pool(excluded_masks)
    cells = [len(f["labels_clean"]) for f in pool]
    print(f"pool frames={len(pool)}  cells/frame min/median/max = "
          f"{min(cells)}/{int(np.median(cells))}/{max(cells)}")
    if len(pool) < 20:
        print("INSUFFICIENT pool frames; aborting without a verdict.")
        return 1

    print(f"Pretraining (corrected recipe, {SSL_STEPS} steps)...")
    ssl_init = train_ssl(pool, SSL_STEPS)
    print(f"  SSL final loss {ssl_init['losses'][-1]:.4f} "
          f"(first {ssl_init['losses'][0]:.4f})")

    results = {"n_pool": len(pool), "k_train_pairs": len(train_pairs),
               "n_val_pairs": len(val_pairs),
               "ssl_final_loss": ssl_init["losses"][-1], "arms": {}}
    print("Arm A: random + finetune 2000 steps (label-matched baseline)")
    results["arms"]["A_random"] = finetune_and_evaluate(
        None, train_pairs, val_pairs, FINETUNE_STEPS,
        (0, 500, 1000, FINETUNE_STEPS))
    print("Arm B: SSL init + finetune 2000 steps (the claim)")
    results["arms"]["B_ssl"] = finetune_and_evaluate(
        ssl_init, train_pairs, val_pairs, FINETUNE_STEPS,
        (0, 500, 1000, FINETUNE_STEPS))
    print(f"Arm C: random + finetune {COMPUTED_MATCHED_STEPS} steps "
          "(compute-matched control)")
    results["arms"]["C_random_compute_matched"] = finetune_and_evaluate(
        None, train_pairs, val_pairs, COMPUTED_MATCHED_STEPS,
        (0, FINETUNE_STEPS, COMPUTED_MATCHED_STEPS))

    b_final = results["arms"]["B_ssl"][str(FINETUNE_STEPS)]["val_bal_acc"]
    a_final = results["arms"]["A_random"][str(FINETUNE_STEPS)]["val_bal_acc"]
    c_final = (results["arms"]["C_random_compute_matched"]
               [str(COMPUTED_MATCHED_STEPS)]["val_bal_acc"])
    margin = b_final - a_final
    works = (margin >= 0.10) and (b_final >= c_final - 0.02)
    results["verdict"] = {
        "B_final": b_final, "A_final": a_final, "C_final": c_final,
        "margin_B_minus_A": margin,
        "rule": "WORKS iff B-A >= +0.10 AND B >= C - 0.02",
        "works": bool(works),
    }

    out = RESULTS_DIR / "result.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out}")
    print("\n=== PRE-REGISTERED VERDICT ===")
    print(f"B (SSL init)  final val bal_acc: {b_final:.3f}")
    print(f"A (random)    final val bal_acc: {a_final:.3f}")
    print(f"C (random, {COMPUTED_MATCHED_STEPS} steps) final val bal_acc: {c_final:.3f}")
    print(f"margin B-A = {margin:+.3f} (rule: >= +0.10)")
    print(f"B vs C     = {b_final - c_final:+.3f} (rule: >= -0.02)")
    print(f"VERDICT: {'WORKS — justify full training' if works else 'DOES NOT WORK — discard the pretext line'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
