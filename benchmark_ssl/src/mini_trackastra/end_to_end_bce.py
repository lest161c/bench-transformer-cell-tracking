#!/usr/bin/env python3
"""Full-model BCE SSL -> downstream benchmark with MiniTrackingTransformer.

Tests whether training the FULL model (encoder + decoder + heads) with
BCE identity loss during SSL speeds up downstream convergence.

Key difference from end_to_end.py:
  - end_to_end.py: NT-Xent on encoder only (decoder gap + loss mismatch)
  - THIS script: BCE identity on full model (no decoder gap, matching loss)

Modes:
  F: NoPE + full-model BCE SSL -> downstream with NoPE
  G: FourierPE + full-model BCE SSL -> downstream with FourierPE
  R: Random init -> downstream with FourierPE [baseline]

Architecture (shared with end_to_end.py):
  DINO patches -> DINOv2 frozen -> Linear(384, d_model) -> (+ PE) -> LayerNorm
    -> Encoder x2 (MultiheadSelfAttention + FFN)
    -> Decoder x2 (MultiheadCrossAttention: tgt=enc_all, memory=enc_all + FFN)
    -> head_x(enc_t), head_y(dec_n) -> assoc = head_x @ head_y^T -> BCE

Usage:
  cd src/mini_trackastra
  .venv/bin/python end_to_end_bce.py
  .venv/bin/python end_to_end_bce.py --ssl-steps 100 --downstream-steps 100 \\
      --max-ssl-frames 30 --max-downstream-pairs 15 --d-model 128
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared components from end_to_end.py
from src.mini_trackastra.end_to_end import (
    device,
    SEED,
    DINO_DIM,
    PATCH_SIZE,
    get_dino,
    compute_dino_embs,
    extract_patches,
    load_frame,
    scan_frames,
    scan_consecutive_pairs,
    apply_jitter,
    apply_affine,
    apply_dropout,
    distort,
    FourierPE,
    NoPE,
    MiniEncoder,
    MiniDecoder,
    MiniTrackingTransformer,
    bce_assoc_loss,
    assoc_accuracy,
    balanced_accuracy,
    free_dino,
    _build_target_matrix,
    make_report,  # reuse plotting
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mini_bce")


# ═══════════════════════════════════════════════════════════════════════════════
# Full-model BCE SSL Phase
# ═══════════════════════════════════════════════════════════════════════════════

def run_ssl_bce_phase(ssl_data, args, pe_dim):
    """Train full MiniTrackingTransformer with BCE identity loss on distorted views.

    For each single frame, we create two distorted views. The model predicts
    associations between views. Target: identity matrix (cell i in view1
    matches cell i in view2 because they're the same cell under different
    distortions).

    This trains encoder + decoder + heads jointly, closing the decoder gap
    and using the same loss function as downstream (BCE).

    Args:
        ssl_data: list of (e1, e2, c1, c2, lbl) with precomputed DINO features
        args: parsed arguments
        pe_dim: positional encoding dimension

    Returns:
        dict: {mode: {"encoder": MiniEncoder, "decoder": MiniDecoder,
                       "head_x": nn.Linear, "head_y": nn.Linear,
                       "pos_enc": PE_module}}
    """
    logger.info(f"SSL BCE data: {len(ssl_data)} frames loaded")

    # Build PE modules
    fourier_pe = FourierPE(pos_per_dim=args.pos_per_dim).to(device)
    noise_pe = NoPE(pe_dim).to(device)

    configs = [
        ("F", noise_pe, "NoPE + full-model BCE"),
        ("G", fourier_pe, "FourierPE + full-model BCE"),
    ]

    ssl_results = {}
    for mode, pe, desc in configs:
        # Build full model
        enc = MiniEncoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_encoder_layers,
            pe_dim=pe_dim,
        ).to(device)
        dec = MiniDecoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_decoder_layers,
        ).to(device)
        model = MiniTrackingTransformer(
            encoder=enc, decoder=dec, d_head=args.d_head,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.ssl_lr)
        logger.info(f"  SSL Mode {mode} ({desc}): {args.ssl_steps} steps")
        logger.info(f"    Params: enc={sum(p.numel() for p in enc.parameters()):,}, "
                    f"dec={sum(p.numel() for p in dec.parameters()):,}, "
                    f"total={sum(p.numel() for p in model.parameters()):,}")

        # Move data to GPU once
        gpu_data = []
        for e1, e2, c1, c2, lbl in ssl_data:
            n = min(len(e1), len(e2))
            if n < 2:
                continue
            gpu_data.append((
                torch.from_numpy(e1[:n]).float().to(device),
                torch.from_numpy(e2[:n]).float().to(device),
                torch.from_numpy(c1[:n]).float().unsqueeze(0).to(device),
                torch.from_numpy(c2[:n]).float().unsqueeze(0).to(device),
                lbl[:n],
            ))

        if len(gpu_data) == 0:
            logger.warning(f"  Mode {mode}: no valid SSL frames, skipping")
            continue

        log_interval = max(1, args.ssl_steps // 10)
        for step in range(args.ssl_steps):
            model.train()
            loss_total = 0.0
            n_batches = 0
            perm = torch.randperm(len(gpu_data))
            for idx in perm:
                de1, de2, c1, c2, lbl = gpu_data[idx]
                n1, n2 = de1.shape[0], de2.shape[0]
                # Labels for both views (may differ due to dropout)
                lbl1 = lbl  # view1 labels (all cells present)
                lbl2 = lbl[:n2] if len(lbl) == n1 else lbl[:n2]

                # Build positional encodings
                pe1 = pe(c1).squeeze(0)  # (N1, pe_dim)
                pe2 = pe(c2).squeeze(0)  # (N2, pe_dim)

                # Full model forward: view1 queries view2
                logits = model(de1, pe1, de2, pe2)  # (N1, N2)

                # Target: label-based matching (handles dropout correctly)
                # Cell i in view1 matches cell j in view2 iff same label
                loss = bce_assoc_loss(logits, lbl1, lbl2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                loss_total += loss.item()
                n_batches += 1

            if (step + 1) % log_interval == 0 or step == 0:
                avg_loss = loss_total / max(1, n_batches)
                logger.info(f"    [{mode}] step {step+1:4d}: loss={avg_loss:.4f}")

        # Save final state
        ssl_results[mode] = {
            "encoder": enc,
            "decoder": dec,
            "head_x": model.head_x,
            "head_y": model.head_y,
            "pos_enc": pe,
            "model": model,
        }
        final_loss = loss_total / max(1, n_batches)
        logger.info(f"    [{mode}] done: final_loss={final_loss:.6f}")

    return ssl_results


# ═══════════════════════════════════════════════════════════════════════════════
# Downstream Phase (adapted for full-model SSL)
# ═══════════════════════════════════════════════════════════════════════════════

def run_downstream_bce(ssl_results, pe_dim, pair_data, args):
    """Train full MiniTrackingTransformer with BCE on real frame pairs.

    Tests:
      F: NoPE full-model BCE SSL -> downstream NoPE
      G: FourierPE full-model BCE SSL -> downstream FourierPE
      R: Random init -> downstream FourierPE [baseline]

    Args:
        ssl_results: dict from run_ssl_bce_phase (modes F, G)
        pe_dim: positional encoding dimension
        pair_data: dict with "train" and "val" lists of precomputed items
        args: parsed arguments

    Returns:
        dict: {mode: pd.DataFrame}
    """
    if len(pair_data.get("train", [])) < 2:
        logger.error("Not enough training pairs.")
        return None

    test_modes = [
        ("R", "Random init baseline", None),
        ("F", "NoPE + full-model BCE SSL", ssl_results.get("F")),
        ("G", "FourierPE + full-model BCE SSL", ssl_results.get("G")),
    ]

    results = {}
    for mode, mode_name, ssl_info in test_modes:
        if mode in ("F", "G") and ssl_info is None:
            logger.warning(f"  Mode {mode}: SSL results missing, skipping")
            continue

        # Build model
        enc = MiniEncoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_encoder_layers,
            pe_dim=pe_dim,
        ).to(device)
        dec = MiniDecoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_decoder_layers,
        ).to(device)
        model = MiniTrackingTransformer(
            encoder=enc, decoder=dec, d_head=args.d_head,
        ).to(device)

        if ssl_info is not None:
            # Load SSL-pretrained weights
            enc.load_state_dict(ssl_info["encoder"].state_dict())
            dec.load_state_dict(ssl_info["decoder"].state_dict())
            model.head_x.load_state_dict(ssl_info["head_x"].state_dict())
            model.head_y.load_state_dict(ssl_info["head_y"].state_dict())
            pe_mod = ssl_info["pos_enc"]
        else:
            pe_mod = FourierPE(pos_per_dim=args.pos_per_dim).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.downstream_lr)
        logger.info(f"\n  Downstream [{mode}] {mode_name}")

        rows = []
        for step in range(args.downstream_steps):
            model.train()
            train_losses, train_accs, train_bal_accs = [], [], []
            for pd_item in pair_data["train"]:
                de_t = pd_item["dino_t"].to(device)
                de_n = pd_item["dino_n"].to(device)
                ct = pd_item["coords_t"].unsqueeze(0).to(device)
                cn = pd_item["coords_n"].unsqueeze(0).to(device)
                lt = pd_item["labels_t"]
                ln = pd_item["labels_n"]

                pe_t = pe_mod(ct).squeeze(0)
                pe_n = pe_mod(cn).squeeze(0)

                logits = model(de_t, pe_t, de_n, pe_n)
                loss = bce_assoc_loss(logits, lt, ln)
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_losses.append(loss.item())
                sd = logits.detach()
                train_accs.append(assoc_accuracy(sd, lt, ln))
                train_bal_accs.append(balanced_accuracy(sd, lt, ln))

            if step % 10 == 0 or step == args.downstream_steps - 1:
                model.eval()
                val_losses, val_accs, val_bal_accs = [], [], []
                with torch.no_grad():
                    for pd_item in pair_data["val"]:
                        de_t = pd_item["dino_t"].to(device)
                        de_n = pd_item["dino_n"].to(device)
                        ct = pd_item["coords_t"].unsqueeze(0).to(device)
                        cn = pd_item["coords_n"].unsqueeze(0).to(device)
                        lt = pd_item["labels_t"]
                        ln = pd_item["labels_n"]

                        pe_t = pe_mod(ct).squeeze(0)
                        pe_n = pe_mod(cn).squeeze(0)

                        logits = model(de_t, pe_t, de_n, pe_n)
                        val_losses.append(bce_assoc_loss(logits, lt, ln).item())
                        val_accs.append(assoc_accuracy(logits, lt, ln))
                        val_bal_accs.append(balanced_accuracy(logits, lt, ln))

                rows.append({
                    "step": step,
                    "train_loss": float(np.mean(train_losses)),
                    "train_acc": float(np.mean(train_accs)),
                    "val_loss": float(np.mean(val_losses)),
                    "val_acc": float(np.mean(val_accs)),
                    "val_bal_acc": float(np.mean(val_bal_accs)),
                })

        results[mode] = pd.DataFrame(rows)
        if rows:
            final = rows[-1]
            logger.info(
                f"    [{mode}] done: val_loss={final['val_loss']:.4f}  "
                f"val_acc={final['val_acc']:.4f}  "
                f"bal_acc={final['val_bal_acc']:.4f}"
            )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict (custom for BCE modes)
# ═══════════════════════════════════════════════════════════════════════════════

def make_bce_report(downstream_results, outdir, args):
    """Generate convergence plot and verdict for BCE SSL modes."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    labels_map = {
        "F": "Mode F: NoPE + full-model BCE (SSL)",
        "G": "Mode G: FourierPE + full-model BCE (SSL)",
        "R": "Mode R: Random init (baseline)",
    }
    colors = {"F": "#3498db", "G": "#e67e22", "R": "#95a5a6"}
    plot_order = ["F", "G", "R"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Val loss
    ax = axes[0]
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        ax.plot(df["step"], df["val_loss"], color=colors.get(mode, "#333"),
                label=labels_map.get(mode, mode), linewidth=2)
    ax.set_xlabel("Downstream training step")
    ax.set_ylabel("Val BCE Loss")
    ax.set_title("Full-Model BCE SSL: Downstream Val Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Balanced accuracy
    ax = axes[1]
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        ax.plot(df["step"], df.get("val_bal_acc", df["val_acc"]),
                color=colors.get(mode, "#333"), linewidth=2)
    ax.set_xlabel("Downstream training step")
    ax.set_ylabel("Val Balanced Accuracy")
    ax.set_title("Downstream Convergence: Balanced Acc (baseline=50%)")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Convergence speed (steps to bal_acc > 0.65)
    ax = axes[2]
    mode_names, conv_steps, bar_colors = [], [], []
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        acc_col = "val_bal_acc" if "val_bal_acc" in df.columns else "val_acc"
        hit = df[df[acc_col] > 0.65]
        s = hit["step"].iloc[0] if len(hit) else args.downstream_steps
        mode_names.append(labels_map.get(mode, mode).split("(")[0].strip())
        conv_steps.append(s)
        bar_colors.append(colors.get(mode, "#333"))
    bars = ax.bar(mode_names, conv_steps, color=bar_colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Steps to bal_acc > 0.65")
    ax.set_title("Convergence Speed (lower = faster)")
    for bar, v in zip(bars, conv_steps):
        label = str(v) if v < args.downstream_steps else "never"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                label, ha="center", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(
        f"MiniTrackastra: Full-Model BCE SSL -> Downstream\n"
        f"{args.max_ssl_frames} SSL frames, {args.ssl_steps} SSL steps, "
        f"{len(downstream_results.get('R', pd.DataFrame()))} data",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = outdir / "convergence_bce.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {fig_path}")

    # Save CSVs
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        df.to_csv(outdir / f"downstream_{mode}.csv", index=False)
        logger.info(f"CSV saved: {outdir / f'downstream_{mode}.csv'}")

    # Verdict
    lines = ["=" * 65, "FULL-MODEL BCE SSL: VERDICT", "=" * 65, ""]
    for mode in plot_order:
        if mode not in downstream_results:
            continue
        df = downstream_results[mode]
        frow = df.iloc[-1]
        irow = df.iloc[0]
        acc_col = "val_bal_acc" if "val_bal_acc" in df.columns else "val_acc"
        lines.append(f"  {labels_map[mode]}:")
        lines.append(f"    Val loss:  {irow['val_loss']:.3f} -> {frow['val_loss']:.3f}")
        lines.append(f"    Val {acc_col}: {irow[acc_col]:.3f} -> {frow[acc_col]:.3f}")

        # Check convergence speed
        hit = df[df[acc_col] > 0.65]
        conv = hit["step"].iloc[0] if len(hit) else args.downstream_steps
        lines.append(f"    Steps to {acc_col}>0.65: {conv}")
        lines.append("")

    # Key comparison: Mode F vs Mode R
    if "F" in downstream_results and "R" in downstream_results:
        df_f = downstream_results["F"]
        df_r = downstream_results["R"]
        acc_col = "val_bal_acc" if "val_bal_acc" in df_f.columns else "val_acc"
        f_final = df_f.iloc[-1][acc_col]
        r_final = df_r.iloc[-1][acc_col]
        delta = f_final - r_final
        lines.append("  KEY QUESTION: Does full-model BCE SSL beat random init?")
        lines.append(f"    Mode F final {acc_col}: {f_final:.4f}")
        lines.append(f"    Mode R final {acc_col}: {r_final:.4f}")
        if delta > 0.03:
            lines.append(f"    *** YES: SSL improves downstream by +{delta:.4f} ***")
            lines.append("    Full-model BCE SSL with NoPE closes BOTH the decoder gap")
            lines.append("    AND the coordinate shortcut.")
        elif delta > 0.00:
            lines.append(f"    ~ MARGINAL: +{delta:.4f} (noise-level)")
            lines.append("    Closing decoder gap + coord shortcut gives minimal benefit")
            lines.append("    at this scale. Bottleneck may be DINO feature quality")
            lines.append("    (domain mismatch: ImageNet -> bacteria).")
        else:
            lines.append(f"    *** NO: SSL is WORSE than random by {delta:.4f} ***")
            lines.append("    Full-model BCE pretraining harms convergence.")

    # Mode F vs Mode G: does coordinate shortcut hurt in full-model setting?
    if "F" in downstream_results and "G" in downstream_results:
        df_f = downstream_results["F"]
        df_g = downstream_results["G"]
        acc_col = "val_bal_acc" if "val_bal_acc" in df_f.columns else "val_acc"
        f_final = df_f.iloc[-1][acc_col]
        g_final = df_g.iloc[-1][acc_col]
        delta = f_final - g_final
        lines.append("")
        lines.append("  COORDINATE SHORTCUT: NoPE (F) vs FourierPE (G)")
        lines.append(f"    Mode F final {acc_col}: {f_final:.4f}")
        lines.append(f"    Mode G final {acc_col}: {g_final:.4f}")
        if delta > 0.03:
            lines.append(f"    NoPE wins by +{delta:.4f} — coordinate shortcut still hurts")
            lines.append("    even with full-model BCE SSL. Removing coordinates is")
            lines.append("    beneficial regardless of decoder gap closure.")
        elif delta > 0.00:
            lines.append(f"    ~ Marginal: +{delta:.4f} — coordinate shortcut has minimal")
            lines.append("    impact when decoder gap is closed.")
        else:
            lines.append(f"    FourierPE wins by {-delta:.4f} — with full-model BCE,")
            lines.append("    coordinates may actually help (cells usually near their")
            lines.append("    previous position, downstream benefits from PE).")

    lines.append("")
    lines.append(f"Full results: {outdir}")
    lines.append(f"  {outdir}/convergence_bce.png")
    for mode in plot_order:
        if mode in downstream_results:
            lines.append(f"  {outdir}/downstream_{mode}.csv")
    lines.append("=" * 65)

    verdict = "\n".join(lines)
    print("\n" + verdict)
    with open(outdir / "verdict_bce.txt", "w") as f:
        f.write(verdict)
    logger.info(f"Verdict saved: {outdir / 'verdict_bce.txt'}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Full-model BCE SSL -> downstream benchmark"
    )
    p.add_argument("--data-root", default="../../data/vanvliet")
    p.add_argument("--conditions", default="rpsM")
    p.add_argument("--max-ssl-frames", type=int, default=30)
    p.add_argument("--max-downstream-pairs", type=int, default=15)
    p.add_argument("--min-cells", type=int, default=8)
    p.add_argument("--ssl-steps", type=int, default=200)
    p.add_argument("--ssl-lr", type=float, default=1e-3)
    p.add_argument("--downstream-steps", type=int, default=100)
    p.add_argument("--downstream-lr", type=float, default=1e-3)

    # Architecture
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--num-decoder-layers", type=int, default=2)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--pos-per-dim", type=int, default=16)
    p.add_argument("--pe-dim-coord", type=int, default=2)

    # Distortion
    p.add_argument("--distortion", default="full",
                   choices=["jitter4", "full"])

    # Output
    p.add_argument("--outdir", default="runs/mini_trackastra_bce")

    return p.parse_args(argv)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]
    pe_dim = args.pos_per_dim * args.pe_dim_coord * 2

    logger.info(
        f"Full-Model BCE SSL benchmark\n"
        f"  Architecture: d_model={args.d_model}, nhead={args.nhead}, "
        f"enc={args.num_encoder_layers}L, dec={args.num_decoder_layers}L\n"
        f"  PE dim: {pe_dim}, d_head: {args.d_head}\n"
        f"  SSL: {args.ssl_steps} steps, lr={args.ssl_lr}, "
        f"frames={args.max_ssl_frames}\n"
        f"  Downstream: {args.downstream_steps} steps, lr={args.downstream_lr}, "
        f"pairs={args.max_downstream_pairs}\n"
        f"  Distortion: {args.distortion}\n"
        f"  Device: {device}"
    )

    # ─── Collect SSL data ─────────────────────────────────────────────────
    logger.info("Scanning single frames for SSL...")
    frames = scan_frames(args.data_root, conditions, args.max_ssl_frames)
    logger.info(f"Found {len(frames)} frames")

    if len(frames) < 2:
        logger.error("Need >= 2 frames for SSL.")
        sys.exit(1)

    ssl_data = []
    for mp, ip in frames:
        r = load_frame(mp, ip)
        if r is None:
            continue
        c1, lbl, img = r
        c2, _ = distort(c1.copy(), lbl.copy(), args.distortion)
        p1 = extract_patches(img, c1)
        p2 = extract_patches(img, c2)
        e1 = compute_dino_embs(p1)
        e2 = compute_dino_embs(p2)
        ssl_data.append((e1, e2, c1, c2, lbl))
    logger.info(f"SSL precomputed: {len(ssl_data)} frames")

    # ─── Collect downstream data ──────────────────────────────────────────
    logger.info("Scanning consecutive frame pairs...")
    pairs = scan_consecutive_pairs(
        args.data_root, conditions, args.max_downstream_pairs + 5
    )

    np.random.seed(SEED + 1)
    idx = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * 0.2))
    train_pairs = [pairs[i] for i in idx[:-n_val]]
    val_pairs = [pairs[i] for i in idx[-n_val:]]
    logger.info(f"Downstream: {len(train_pairs)} train + {len(val_pairs)} val")

    pair_data = {"train": [], "val": []}
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for mt, mn, it, i_n in split_pairs:
            rt = load_frame(mt, it)
            rn = load_frame(mn, i_n)
            if rt is None or rn is None:
                continue
            ct, lt, imgt = rt
            cn, ln, imgn = rn
            shared = set(lt) & set(ln)
            if len(shared) < args.min_cells:
                continue
            if len(lt) < args.min_cells or len(ln) < args.min_cells:
                continue
            idx_t = [i for i, l in enumerate(lt) if l in shared]
            idx_n = [i for i, l in enumerate(ln) if l in shared]
            ct_s, lt_s = ct[idx_t], lt[idx_t]
            cn_s, ln_s = cn[idx_n], ln[idx_n]
            pt_s = extract_patches(imgt, ct_s)
            pn_s = extract_patches(imgn, cn_s)
            pair_data[split_name].append({
                "dino_t": torch.from_numpy(compute_dino_embs(pt_s)).float(),
                "dino_n": torch.from_numpy(compute_dino_embs(pn_s)).float(),
                "coords_t": torch.from_numpy(ct_s).float(),
                "coords_n": torch.from_numpy(cn_s).float(),
                "labels_t": torch.from_numpy(lt_s).long(),
                "labels_n": torch.from_numpy(ln_s).long(),
            })
    logger.info(
        f"Downstream precomputed: {len(pair_data['train'])} train, "
        f"{len(pair_data['val'])} val"
    )

    free_dino()

    # ─── SSL Phase (full-model BCE) ───────────────────────────────────────
    ssl_results = run_ssl_bce_phase(ssl_data, args, pe_dim)

    # ─── Downstream Phase ─────────────────────────────────────────────────
    downstream_results = run_downstream_bce(ssl_results, pe_dim, pair_data, args)

    if downstream_results is None:
        logger.error("Downstream phase failed.")
        sys.exit(1)

    # ─── Report ───────────────────────────────────────────────────────────
    make_bce_report(downstream_results, outdir, args)
    logger.info(f"Done. All outputs in {outdir}/")


if __name__ == "__main__":
    main()
