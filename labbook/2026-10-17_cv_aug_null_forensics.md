# CV cutoff ablation: the Aug-2026 "within noise" result was a code fallback

**Date:** 2026-10-17 · **Related:** SPEC 0043, tickets 29/33/19; artifacts `CV-aug-null-forensics.md`, `CV-flip-analysis.md`, `CV-rerun-results.md`, `CV-seed43-replicate.md`.

## What the Aug experiment was meant to test
6-fold leave-one-condition-out (LOCO) on vanvliet: masked baseline (`knn_neighbors=-1`, CachedDistAttention) vs DenseFlash (`knn_neighbors=-2`, spatial cutoff removed), seed 42; Slurm array `3926370` (2026-08-18). Purpose: ablate the spatial cutoff, which the Trackastra paper fixes at $d_{max}=256$px but never ablates.

## Why it produced "accuracy within noise" (the cause)
`knn_neighbors=-2` had **no dispatch branch**. At the last commit before the array (`a5764f0`, 2026-08-17) and at the then-HEAD, the factory was `if knn_neighbors > 0: KNNMaskSparseAttention else: CachedDistAttention`; `class DenseFlashAttention` had **no commit anywhere** in history. The `-2→DenseFlashAttention` branch existed only as uncommitted working-tree edits dated **2026-10-14** (`*.pre_denseflash.bak`). So **both arms instantiated CachedDistAttention** — the same architecture, differing only by seed/init. The paired difference was therefore pure training jitter: small, symmetric, mean≈0 (old ΔTRA −0.00027 [−0.0019,+0.0005]; ΔAOGM +4.0 [−10.0,+20.9]), and it read as "within noise". Corroboration: the old "DenseFlash" column matches the new **masked** run on 5/6 folds.

## Errors
- **E1 code (root cause):** silent fallback of an unknown selector value; no raise.
- **E2 process:** no arm-identity guard (the loud DenseFlash guard was only added 2026-10-17, commit `2f30075`).
- **E3 process:** the null was never provenance-checked; the raw runs were later lost (SPEC 0038 could not find them; SPEC 0004 already noted they existed only in `appendix.tex`).
- **E4 reporting:** a scientific claim ("cutoff purely computational / accuracy indistinguishable / zero cost") was inferred from an unverified null.
- **E5 earlier confound:** an earlier attempt compared baseline `window=4,dropout=0.05` vs "dense_flash" `window=10,dropout=0.01` — not one-factor.
- **E6 stats:** the (later-removed) tests were not the bug; they correctly found no difference between identical arms.

## How we fixed it
1. Implemented + installed native `DenseFlashAttention` (`model_parts.py:823`) and the `knn_neighbors == -2` branch (`model.py:422`).
2. Hardened `run_cv.slurm` (`2f30075`): a loud guard that aborts if `-2` does not map to DenseFlashAttention; `capella` partition + `--gres=gpu:1`; traccuracy eval venv; per-task `--cachedir`; `150303-08_BAD` excluded.
3. Re-ran: `4428930` (seed 42) and `4429851` (seed 43, independent replicate).
4. Result: removing the cutoff is **worse**, not neutral — DenseFlash worse on **6/6 folds at both seeds (12/12 fold-level signs)** for TRA and AOGM; pooled AOGM penalty **+51.2** [+11.9,+129.2] of ~225 (s42 +45.5, s43 +56.8), pooled ΔTRA −0.0023. Reported descriptively (no inferential test, GATE-1).
5. Report updated: the invalid table withdrawn (`61a5187`), then the seed-43 two-seed text (`8aaa9da`).

## Lesson
An ablation must **prove its arms differ** (attention class, mask, dispatched kernel) before a null can be read as evidence; unknown selector values must raise, not fall through to a default. And archive the raw runs with the table.
