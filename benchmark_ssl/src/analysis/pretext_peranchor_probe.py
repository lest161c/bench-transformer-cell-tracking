#!/usr/bin/env python3
"""Pretext objective probe: per-anchor row-CE vs BCE on the two pools (2026-10-29).

Question (owner, 2026-10-29): the MOST BASIC local experiment that can HINT at
whether a longer/corrected pretext run would be valuable. The 2026-10-28 bug
exclusion (``pretext_collapse_bug_exclusion.py``, BUG_EXCLUDED=true)
established that the BCE-over-NxN pretext collapses on trpL (loss = all-negative
entropy floor 0.1606, true-identity argmax at chance) while the identical code
learns on rpsM (loss ~0.10 vs floor 0.188, argmax ~0.89). The candidate remedy
is objective-level: normalize per anchor instead of over the full matrix
(row-wise cross-entropy with one positive per row = InfoNCE over in-row
candidates), which cannot collapse to all-negative by construction — every
scored row has exactly one positive target, so ranking the partner above
distractors always decreases the loss. This probe swaps ONLY the loss and
re-runs the identical 300-step SSL protocol on both pools; the downstream /
transfer chain stays out of scope regardless of outcome (owner directive of
2026-10-23 stands; this probe tests pretext learnability only).

Pre-registered design and verdict rule (fixed before running; deterministic,
seed 42, local A500):

- Pools: exactly the 2026-10-28 probe's pools (provenance-locked: 5 train / 8
  val downstream pairs; 60 trpL frames; rpsM pool via ALL_CONDITIONS=["rpsM"]).
  No input re-audit: inputs were verified clean on 2026-10-28; pool identity is
  cross-checked by frame count plus the BCE control arm's signature.
- Arms (2x2): {BCE, PerAnchorCE} x {trpL, rpsM}, 300 steps each, identical
  model architecture/seed/batch order/optimizer (each arm re-seeds
  torch.manual_seed(SEED) exactly as ``P.train_ssl`` does; neither loss draws
  RNG, so arms are bit-comparable within a pool).
- Metrics: loss trajectory; BCE floor H(pos_frac) and per-anchor floor log(N2)
  (uniform predictor) per pool; true-identity argmax accuracy and
  partner-vs-best-off-target logit gap at steps 100/200/300 (state snapshots,
  evaluated post hoc, same k=12 frames as the 2026-10-28 probe); last-100-step
  argmax slope.

VERDICT (pre-registered):
- INVALID (probe says nothing): the BCE control arm fails to reproduce the
  2026-10-28 signature (trpL argmax >= 0.2, or rpsM argmax <= 0.5), OR the
  per-anchor objective fails on the control pool (rpsM argmax < 0.5 — the
  remedy must at least learn where BCE learns).
- ESCALATE (a longer run is valuable): PerAnchorCE on trpL reaches argmax
  >= 0.5 at any checkpoint (escapes where BCE could not, same budget), OR
  argmax in [0.2, 0.5) at step 300 AND still rising
  (argmax300 > argmax200 + 0.02).
- NO_HINT: PerAnchorCE on trpL argmax < 0.2 at step 300 AND flat
  (argmax300 <= argmax200 + 0.02) — the pool lacks per-cell discriminative
  signal at prototype scale; longer runs unlikely to help on trpL, and the
  report's "our interpretation" hedge is promoted to a measured finding.
- Else: INCONCLUSIVE (report the numbers).

Reproduce (from the repository root)::

    benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_peranchor_probe.py

Smoke test (tiny, no verdict)::

    benchmark_ssl/.venv/bin/python benchmark_ssl/src/analysis/pretext_peranchor_probe.py --smoke

Outputs ``benchmark_ssl/results/pretext_peranchor_probe.json`` and prints the
verdict. Full run on the local GPU in well under 20 minutes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BENCHMARK_SSL_ROOT = Path(__file__).resolve().parents[2]
if str(BENCHMARK_SSL_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_SSL_ROOT))
ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

import src.mini_trackastra.end_to_end as e2e  # noqa: E402
import pretext_sample_efficiency_test as P  # noqa: E402
from pretext_collapse_bug_exclusion import output_mappings  # noqa: E402

RESULTS_DIR = BENCHMARK_SSL_ROOT / "results"

SMOKE = "--smoke" in sys.argv
SSL_STEPS = 3 if SMOKE else 300
CHECKPOINTS = (100, 200) if not SMOKE else (1, 2)
EVAL_K = 12  # same frames as the 2026-10-28 probe's argmax numbers
SNAPSHOT_EVERY = tuple(CHECKPOINTS) + (SSL_STEPS,)

# Decisive-artifact anchors for the sanity check (2026-10-28 probe).
BCE_TRP_L_REFERENCE_LOSS = 0.1605545369287332
BCE_TRP_L_LOSS_TOL = 0.01


def entropy_floor(pos_frac: float) -> float:
    p = min(max(pos_frac, 1e-9), 1 - 1e-9)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def pool_floors(pool: list[dict]) -> dict:
    """BCE floor H(pos_frac) and per-anchor uniform-predictor floor log(N2)."""
    bce, pa = [], []
    for f in pool:
        pos = (np.asarray(f["labels_clean"])[:, None]
               == np.asarray(f["labels_warped"])[None, :])
        bce.append(entropy_floor(float(pos.mean())))
        pa.append(float(np.log(max(pos.shape[1], 2))))
    return {"bce_floor_mean": float(np.mean(bce)),
            "peranchor_floor_mean": float(np.mean(pa))}


def per_anchor_assoc_loss(logits, labels_t, labels_n):
    """Row-wise cross-entropy: one positive per anchor row (survivor labels).

    For each anchor i in view 1 whose label survives in view 2 (exactly one
    match by label equality; argmax picks the first on duplicates), the loss
    is -log softmax(logits[i])[j*(i)]. Rows without a surviving partner carry
    no association signal and are skipped. A non-discriminative constant
    predictor sits at log(N2); the all-negative direction does not exist
    because every scored row has exactly one positive target.
    """
    target = e2e._build_target_matrix(labels_t, labels_n, logits.device)
    row_pos = target.sum(dim=1)
    has = row_pos > 0
    if not has.any():
        return logits.sum() * 0.0
    idx = target.argmax(dim=1)
    return F.cross_entropy(logits[has], idx[has])


LOSSES = {"bce": e2e.bce_assoc_loss, "peranchor": per_anchor_assoc_loss}


def train_ssl_local(pool: list[dict], steps: int, objective: str) -> dict:
    """Mirror of ``P.train_ssl`` with a parameterized loss + state snapshots.

    Identical seeding, model construction, batch order, optimizer, and grad
    clipping; only the loss call differs (neither loss draws RNG).
    """
    loss_fn = LOSSES[objective]
    torch.manual_seed(e2e.SEED)
    encoder, decoder, model = P.build_model()
    pe = P.FourierPE(pos_per_dim=P.POS_PER_DIM).to(e2e.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P.SSL_LR)
    gpu_pool = [(torch.from_numpy(f["features_clean"]).float().to(e2e.device),
                 torch.from_numpy(f["features_warped"]).float().to(e2e.device),
                 torch.from_numpy(f["coords_clean"]).float().unsqueeze(0).to(e2e.device),
                 torch.from_numpy(f["coords_warped"]).float().unsqueeze(0).to(e2e.device),
                 f["labels_clean"], f["labels_warped"]) for f in pool]
    losses, snapshots = [], {}
    for step in range(steps):
        model.train()
        loss_total, n_batches = 0.0, 0
        for idx in torch.randperm(len(gpu_pool)):
            feats_1, feats_2, coords_1, coords_2, labels_1, labels_2 = gpu_pool[idx]
            logits = model(feats_1, pe(coords_1).squeeze(0),
                           feats_2, pe(coords_2).squeeze(0))
            loss = loss_fn(logits, labels_1, labels_2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item()
            n_batches += 1
        losses.append(loss_total / max(1, n_batches))
        if (step + 1) in SNAPSHOT_EVERY:
            snapshots[step + 1] = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
        if (step + 1) % 50 == 0:
            print(f"    step {step + 1}: loss {losses[-1]:.4f}")
    return {"model": model, "pe": pe, "losses": losses, "snapshots": snapshots}


def argmax_trajectory(arm: dict, pool: list[dict]) -> list[dict]:
    """True-identity argmax acc + partner gap at each checkpoint (post hoc)."""
    model, pe = arm["model"], arm["pe"]
    traj = []
    for step in sorted(arm["snapshots"], reverse=True):
        if step != SSL_STEPS:
            model.load_state_dict(arm["snapshots"][step])
        rows = output_mappings({"model": model, "pe": pe}, pool, k=EVAL_K)
        traj.append({
            "step": step,
            "argmax_acc": float(np.mean([r["true_identity_argmax_acc"]
                                         for r in rows])) if rows else None,
            "partner_gap_mean": float(np.mean(
                [r["partner_minus_best_offtarget_mean"] for r in rows
                 if r["partner_minus_best_offtarget_mean"] is not None]))
            if rows else None,
        })
    traj.sort(key=lambda d: d["step"])
    return traj


def main() -> int:
    if SMOKE:
        P.POOL_MAX_FRAMES = 4
        P.MIN_POOL_CELLS = 8
    torch.manual_seed(e2e.SEED)
    np.random.seed(e2e.SEED)
    print(f"device={e2e.device}  smoke={SMOKE}")

    # 1) Provenance: downstream pairs + trpL pool must match the decisive run.
    train_pairs, val_pairs, excluded_masks = P.build_downstream_pairs()
    print(f"downstream: train={len(train_pairs)} val={len(val_pairs)} "
          f"(decisive artifact: 5 / 8)")
    if not SMOKE:
        assert len(train_pairs) == 5 and len(val_pairs) == 8, \
            "downstream-pair selection diverged from decisive run"

    pools = {}
    pools["trpL"] = P.build_ssl_pool(excluded_masks)
    print(f"trpL pool: {len(pools['trpL'])} frames (decisive artifact: 60)")
    if not SMOKE:
        assert len(pools["trpL"]) == 60, "pool selection diverged from decisive run"
    P.ALL_CONDITIONS = ["rpsM"]
    pools["rpsM"] = P.build_ssl_pool(set())
    print(f"rpsM pool: {len(pools['rpsM'])} frames")

    floors = {name: pool_floors(pool) for name, pool in pools.items()}

    # 2) The 2x2: {bce, peranchor} x {trpL, rpsM}, identical protocol.
    arms = {}
    for pool_name in ("trpL", "rpsM"):
        for objective in ("bce", "peranchor"):
            print(f"[{objective} x {pool_name}] SSL {SSL_STEPS} steps...")
            arm = train_ssl_local(pools[pool_name], SSL_STEPS, objective)
            traj = argmax_trajectory(arm, pools[pool_name])
            arms[f"{objective}_{pool_name}"] = {
                "final_loss": arm["losses"][-1],
                "first_loss": arm["losses"][0],
                "trajectory": traj,
            }
            for t in traj:
                print(f"    step {t['step']}: argmax {t['argmax_acc']:.3f} "
                      f"gap {t['partner_gap_mean']:+.2f}")

    # 3) Pre-registered verdict.
    bce_trpL = arms["bce_trpL"]
    bce_rpsM = arms["bce_rpsM"]
    pa_trpL = arms["peranchor_trpL"]
    pa_rpsM = arms["peranchor_rpsM"]
    acc = lambda arm, step: next(t["argmax_acc"] for t in arm["trajectory"]
                                 if t["step"] == step)
    sanity = (
        abs(bce_trpL["final_loss"] - BCE_TRP_L_REFERENCE_LOSS) <= BCE_TRP_L_LOSS_TOL
        and acc(bce_trpL, SSL_STEPS) < 0.2
        and acc(bce_rpsM, SSL_STEPS) > 0.5
    )
    remedy_valid = acc(pa_rpsM, SSL_STEPS) > 0.5
    pa_trpL_final = acc(pa_trpL, SSL_STEPS)
    pa_trpL_prev = acc(pa_trpL, CHECKPOINTS[-1])
    escaped = pa_trpL_final >= 0.5
    rising = pa_trpL_final >= 0.2 and pa_trpL_final > pa_trpL_prev + 0.02
    flat_and_low = pa_trpL_final < 0.2 and pa_trpL_final <= pa_trpL_prev + 0.02

    if not sanity or not remedy_valid:
        verdict_name = "INVALID"
    elif escaped or rising:
        verdict_name = "ESCALATE"
    elif flat_and_low:
        verdict_name = "NO_HINT"
    else:
        verdict_name = "INCONCLUSIVE"

    results = {
        "smoke": SMOKE,
        "ssl_steps": SSL_STEPS,
        "eval_k": EVAL_K,
        "floors": floors,
        "arms": arms,
        "checks": {
            "bce_trpL_reproduces_collapse": bool(sanity),
            "peranchor_learns_control_pool": bool(remedy_valid),
            "peranchor_trpL_final_argmax": pa_trpL_final,
            "peranchor_trpL_prev_argmax": pa_trpL_prev,
            "peranchor_trpL_slope_100": pa_trpL_final - pa_trpL_prev,
        },
        "verdict": {
            "name": verdict_name,
            "rules": (
                "INVALID: BCE arm fails 2026-10-28 signature OR peranchor "
                "fails on rpsM (argmax < 0.5). "
                "ESCALATE: peranchor trpL argmax >= 0.5 at any checkpoint, "
                "OR in [0.2, 0.5) at final step AND rising (+0.02 over last "
                "checkpoint). NO_HINT: peranchor trpL argmax < 0.2 AND flat. "
                "Else INCONCLUSIVE."
            ),
        },
    }
    out_path = RESULTS_DIR / "pretext_peranchor_probe.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}")
    if SMOKE:
        print("SMOKE OK (no verdict in smoke mode)")
        return 0
    print("\n=== PRE-REGISTERED VERDICT ===")
    print(f"BCE  trpL: loss {bce_trpL['final_loss']:.4f} "
          f"(reference {BCE_TRP_L_REFERENCE_LOSS:.4f})")
    for arm_name, arm in arms.items():
        steps_txt = ", ".join(f"{t['step']}:{t['argmax_acc']:.3f}"
                              for t in arm["trajectory"])
        print(f"{arm_name:22s} argmax {steps_txt}")
    print(f"VERDICT: {verdict_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
