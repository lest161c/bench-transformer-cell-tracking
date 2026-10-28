#!/bin/bash
#SBATCH --job-name=cv_decomp_reeval
#SBATCH --account=p_scads_celltracking
#SBATCH --partition=capella
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --output=logs/cv_decomp_reeval.out
#SBATCH --error=logs/cv_decomp_reeval.err
#
# Eval-only re-run of the cv500 CV-cutoff campaign (2026-10-23, 24 runs) with
# the CTC decomposition (AOGM-A / AOGM-D) added to eval_cv.py.
#
# Why: the published tab:cv_results stores only TRA + AOGM per run. The
# decomposition answers WHY DenseFlash loses AOGM: missed divisions (NS,
# node errors -> AOGM_D) vs dropped/added links (edge errors -> AOGM_A).
# No retraining: each eval re-runs track_from_disk on the held-out condition
# with the SAME protocol (greedy, max_distance 50, GT detections) and the
# SAME checkpoint (md5 recorded in the provenance sidecar), so the new
# TRA/AOGM must reproduce the published values (consistency check below).
#
# Outputs: <run_dir>/cv_eval_decomp/{eval_results.json,summary.json,
# eval_results.json.provenance.json,eval_log.txt} per run, plus the aggregate
# $BENCH/results/cv_rerun/cv500_decomp/collection.json (per-run records,
# per-fold-seed paired deltas for TRA/AOGM/AOGM_A/AOGM_D, and the
# consistency check against the published tab:cv_results values).
#
# Idempotent: runs whose cv_eval_decomp/eval_results.json already exists are
# skipped unless FORCE=1: sbatch --export=FORCE=1 (combine with other exports
# comma-separated; sbatch does NOT accumulate repeated --export flags).
#
# Eval-only cost: ~17 s per run measured on the 100-epoch campaign
# (provenance started/finished); model loading dominates. Budget 2 h for 24.

source "$HOME/.bench.env"
source "$REPO_ROOT/slurm/load_cluster_env.sh"

DATA_DIR="${DATA_DIR}/vanvliet"
cd "$TRK"

# eval_cv.py imports traccuracy, which is NOT in $TRK/.venv (see run_cv.slurm).
EVAL_PY="$BENCH/benchmark_training/.venv/bin/python"
if [ ! -x "$EVAL_PY" ]; then
    echo "ERROR: eval venv missing: $EVAL_PY"
    exit 1
fi

# ─── Published tab:cv_results values (consistency check) ─────────────
# (arm, seed, fold, condition) -> TRA, AOGM from report/appendix.tex
# tab:cv_results. The re-eval must reproduce these within rounding
# (TRA 1e-6, AOGM 1e-3); a mismatch means the re-eval diverged from the
# original eval and the decomposition must not be trusted.
declare -A PUBLISHED_TRA PUBLISHED_AOGM
PUBLISHED_TRA[42-0-masked]=0.993382;  PUBLISHED_AOGM[42-0-masked]=159.750
PUBLISHED_TRA[42-0-dense_flash]=0.991533; PUBLISHED_AOGM[42-0-dense_flash]=214.688
PUBLISHED_TRA[42-1-masked]=0.932559;  PUBLISHED_AOGM[42-1-masked]=690.929
PUBLISHED_TRA[42-1-dense_flash]=0.925800; PUBLISHED_AOGM[42-1-dense_flash]=774.643
PUBLISHED_TRA[42-2-masked]=0.998673;  PUBLISHED_AOGM[42-2-masked]=38.100
PUBLISHED_TRA[42-2-dense_flash]=0.998612; PUBLISHED_AOGM[42-2-dense_flash]=40.500
PUBLISHED_TRA[42-3-masked]=0.990498;  PUBLISHED_AOGM[42-3-masked]=189.200
PUBLISHED_TRA[42-3-dense_flash]=0.987935; PUBLISHED_AOGM[42-3-dense_flash]=264.700
PUBLISHED_TRA[42-4-masked]=0.999949;  PUBLISHED_AOGM[42-4-masked]=2.000
PUBLISHED_TRA[42-4-dense_flash]=0.998963; PUBLISHED_AOGM[42-4-dense_flash]=35.250
PUBLISHED_TRA[42-5-masked]=0.995139;  PUBLISHED_AOGM[42-5-masked]=40.800
PUBLISHED_TRA[42-5-dense_flash]=0.994618; PUBLISHED_AOGM[42-5-dense_flash]=66.100
PUBLISHED_TRA[43-0-masked]=0.993371;  PUBLISHED_AOGM[43-0-masked]=161.375
PUBLISHED_TRA[43-0-dense_flash]=0.991207; PUBLISHED_AOGM[43-0-dense_flash]=226.500
PUBLISHED_TRA[43-1-masked]=0.931841;  PUBLISHED_AOGM[43-1-masked]=718.571
PUBLISHED_TRA[43-1-dense_flash]=0.924681; PUBLISHED_AOGM[43-1-dense_flash]=797.643
PUBLISHED_TRA[43-2-masked]=0.998707;  PUBLISHED_AOGM[43-2-masked]=37.300
PUBLISHED_TRA[43-2-dense_flash]=0.998406; PUBLISHED_AOGM[43-2-dense_flash]=48.100
PUBLISHED_TRA[43-3-masked]=0.990106;  PUBLISHED_AOGM[43-3-masked]=208.400
PUBLISHED_TRA[43-3-dense_flash]=0.987833; PUBLISHED_AOGM[43-3-dense_flash]=275.000
PUBLISHED_TRA[43-4-masked]=0.999797;  PUBLISHED_AOGM[43-4-masked]=8.000
PUBLISHED_TRA[43-4-dense_flash]=0.998318; PUBLISHED_AOGM[43-4-dense_flash]=60.750
PUBLISHED_TRA[43-5-masked]=0.995133;  PUBLISHED_AOGM[43-5-masked]=41.000
PUBLISHED_TRA[43-5-dense_flash]=0.994665; PUBLISHED_AOGM[43-5-dense_flash]=61.500

mkdir -p "$BENCH/results/cv_rerun/cv500_decomp"
mkdir -p "$BENCH/logs"

N_DONE=0; N_SKIP=0; N_FAIL=0
for run_dir in "$TRK"/runs/2026-*_cv500_*_s42 "$TRK"/runs/2026-*_cv500_*_s43; do
    [ -d "$run_dir" ] || continue
    [ -f "$run_dir/model.pt" ] || { echo "SKIP (no model.pt): $run_dir"; continue; }
    name=$(basename "$run_dir")
    # name: <ts>_cv500_<arm>_fold<N>_<cond>_s<seed>; arm may contain an
    # underscore ("dense_flash"), so split on the fixed cv500_ prefix.
    if [[ "$name" == *_cv500_dense_flash_* ]]; then arm="dense_flash"; else arm="masked"; fi
    rest="${name#*_cv500_${arm}_}"
    fold=$(echo "${rest%%_*}" | tr -d 'fold')
    rest2="${rest#*_}"
    cond="${rest2%%_*}"
    seed=$(echo "${rest2##*_}" | tr -d 's')
    outdir="$run_dir/cv_eval_decomp"
    echo "=== $name (arm=$arm fold=$fold cond=$cond seed=$seed) ==="
    if [ -f "$outdir/eval_results.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "  already evaluated, skipping (FORCE=1 to redo)"
        N_SKIP=$((N_SKIP+1))
        continue
    fi
    mkdir -p "$outdir"
    "$EVAL_PY" "$BENCH/benchmark_training/scripts/eval_cv.py" \
        --run-dir "$run_dir" \
        --heldout "$cond" \
        --data-root "$DATA_DIR" \
        --output "$outdir/eval_results.json" \
        --device cuda 2>&1 | tee "$outdir/eval_log.txt"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "  EVAL FAILED rc=$rc"
        N_FAIL=$((N_FAIL+1))
        continue
    fi
    # consistency check against the published values
    new_tra=$("$EVAL_PY" -c "import json;print(json.load(open('$outdir/summary.json'))['mean_TRA'])")
    new_aogm=$("$EVAL_PY" -c "import json;print(json.load(open('$outdir/summary.json'))['mean_AOGM'])")
    pub_tra=${PUBLISHED_TRA[$seed-$fold-$arm]}
    pub_aogm=${PUBLISHED_AOGM[$seed-$fold-$arm]}
    ok_tra=$("$EVAL_PY" -c "print(abs($new_tra - $pub_tra) <= 1e-6)")
    ok_aogm=$("$EVAL_PY" -c "print(abs($new_aogm - $pub_aogm) <= 1e-3)")
    if [ "$ok_tra" = "True" ] && [ "$ok_aogm" = "True" ]; then
        echo "  CONSISTENCY PASS (TRA $new_tra vs $pub_tra; AOGM $new_aogm vs $pub_aogm)"
    else
        echo "  CONSISTENCY FAIL (TRA $new_tra vs $pub_tra; AOGM $new_aogm vs $pub_aogm)"
        echo "  -> the re-eval diverged from the original eval; do not trust the decomposition"
    fi
    N_DONE=$((N_DONE+1))
done

echo ""
echo "=== evaluated=$N_DONE skipped=$N_SKIP failed=$N_FAIL ==="

# ─── Aggregate ────────────────────────────────────────────────────────
"$EVAL_PY" - <<'PYEOF'
import json
from pathlib import Path

runs_root = Path("runs")  # cwd is $TRK; run dirs live in $TRK/runs
out_dir = Path(__import__("os").environ["BENCH"]) / "results/cv_rerun/cv500_decomp"
records = []
for run_dir in sorted(runs_root.glob("2026-*_cv500_*_s42")) + \
               sorted(runs_root.glob("2026-*_cv500_*_s43")):
    summ = run_dir / "cv_eval_decomp/summary.json"
    res = run_dir / "cv_eval_decomp/eval_results.json"
    if not summ.is_file():
        continue
    name = run_dir.name
    # arm may contain an underscore ("dense_flash"); split on the fixed
    # cv500_ prefix instead of naive _-splitting.
    rest = name.split("_cv500_", 1)[1]
    arm = "dense_flash" if rest.startswith("dense_flash") else "masked"
    rest2 = rest[len(arm) + 1:]          # fold<N>_<cond>_s<seed>
    fold = int(rest2.split("_")[0].replace("fold", ""))
    cond = rest2.split("_")[1]
    seed = int(rest2.split("_")[2].replace("s", ""))
    s = json.load(open(summ))
    per_seq = json.load(open(res)) if res.is_file() else {}
    records.append({
        "run_dir": name, "arm": arm, "fold": fold, "condition": cond,
        "seed": seed, "summary": s, "per_sequence": per_seq,
    })

# paired deltas per (fold, seed): dense_flash - masked
by_key = {(r["arm"], r["fold"], r["seed"]): r for r in records}
paired = []
for seed in (42, 43):
    for fold in range(6):
        m = by_key.get(("masked", fold, seed))
        d = by_key.get(("dense_flash", fold, seed))
        if not m or not d:
            continue
        cond = m["condition"]
        row = {"fold": fold, "condition": cond, "seed": seed}
        for metric in ("TRA", "AOGM", "AOGM_A", "AOGM_D"):
            mv = m["summary"].get(f"mean_{metric}")
            dv = d["summary"].get(f"mean_{metric}")
            row[f"masked_{metric}"] = mv
            row[f"dense_flash_{metric}"] = dv
            row[f"delta_{metric}"] = (dv - mv) if (mv is not None and dv is not None) else None
        # AOGM-A rate: errors per GT edge (mean over the fold's sequences)
        def rate(rec):
            seqs = [v for v in rec["per_sequence"].values()
                    if isinstance(v, dict) and v.get("AOGM_A") is not None]
            gt = [v.get("gt_edge_count") for v in seqs]
            if not seqs or not all(gt):
                return None
            return sum(v["AOGM_A"] for v in seqs) / sum(gt)
        row["masked_AOGM_A_rate"] = rate(m)
        row["dense_flash_AOGM_A_rate"] = rate(d)
        paired.append(row)

collection = {
    "campaign": "cv500_eval_decomp",
    "note": "eval-only re-run of the 2026-10-23 cv500 campaign with the CTC "
            "decomposition (AOGM_A/AOGM_D) added to eval_cv.py; same checkpoints, "
            "same eval protocol; consistency check against the published "
            "tab:cv_results in the job log",
    "n_runs": len(records),
    "per_run": records,
    "paired_dense_minus_masked": paired,
}
dest = out_dir / "collection.json"
dest.write_text(json.dumps(collection, indent=2, default=float))
print(f"Wrote {dest}")

print("\n=== paired differences (dense_flash - masked) ===")
print("fold cond     seed | d_AOGM  d_AOGM_A  d_AOGM_D | masked_rate dense_rate")
def fmt(x, spec):
    return format(x, spec) if x is not None else "--"
for r in paired:
    print(f"{r['fold']:4d} {r['condition']:<8s} {r['seed']} | "
          f"{fmt(r['delta_AOGM'], '+8.2f')} {fmt(r['delta_AOGM_A'], '+9.2f')} "
          f"{fmt(r['delta_AOGM_D'], '+9.2f')} | "
          f"{fmt((r['masked_AOGM_A_rate'] or 0)*100 if r['masked_AOGM_A_rate'] is not None else None, '6.3f')}%  "
          f"{fmt((r['dense_flash_AOGM_A_rate'] or 0)*100 if r['dense_flash_AOGM_A_rate'] is not None else None, '6.3f')}%")
PYEOF
