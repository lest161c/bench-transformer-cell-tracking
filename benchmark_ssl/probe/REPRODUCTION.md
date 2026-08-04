# REPRODUCTION.md — Edge Probing Experiments (`benchmark_ssl/probe/`)

This document describes how to reproduce every experiment in
`benchmark_ssl/probe/`. The directory contains three experiments:

| Experiment | Script | Status | Run where |
|---|---|---|---|
| Unified 5-fold CV edge probe | `unified_edge_probe.py` (1599 lines) | **DONE** (2026-08-04, local A500) | Local (A500), runnable on H100 |
| Same probe via SLURM | `run_unified_probe.slurm` | **PENDING** (not yet run on cluster) | Cluster (H100) |
| Legacy single-split edge probe | `edge_probe.py` (972 lines) | **DONE / SUPERSEDED** by unified probe | Local |

Authoritative source for feasibility verdicts and measured numbers is
`labbook/2026-08-03_experiment_readiness_inventory.md` (§2.1 and §5, item 1).

---

## Repository and cluster path constants

Verified against the scripts (see `run_unified_probe.slurm` and
`unified_edge_probe.py`).

| Constant | Path |
|---|---|
| Repo root (local) | `/home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj` |
| Local SSL venv | `benchmark_ssl/.venv` (Python 3.14.4, torch 2.12.0+cu130, sklearn 1.9.0, scipy, scikit-image, tifffile, pandas, numpy) |
| Local vanvliet data | `data/vanvliet/` (repo root; 6 conditions `rpsM, recA, pheA, metA, cib, trpL`, complete 3.1 GB) |
| Local results output (repo root) | `results/unified_probe_results_cv.json` |
| TRK (cluster, trackastra checkout + venv + runs) | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra` |
| BENCH (cluster bench repo) | `/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking` |
| DATA_DIR (cluster data) | `/data/cat/ws/mawe985g-data/data/celltracking` (vanvliet subdir: `.../data/celltracking/vanvliet`) |
| Cluster venv | `$TRK/.venv` (activated inside the SLURM script) |
| Cluster SLURM partition / account | partition `gpu-h100`, account `p_scads_celltracking`, 1 GPU per node |
| NT-Xent checkpoint (used by `cnn_frozen`) | `benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt` (5.9 MB; default in script) |
| DINOv2 weights cache (local) | `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` (88 MB, `dinov2_vits14`) |

---

## 1. Local (A500) run — unified 5-fold CV edge probe `unified_edge_probe.py`

**Status: DONE (2026-08-04).** This is the authoritative edge-probe experiment and
the backing artifact for the report's edge-probe table. It is **runnable locally**
(measured wall time 50 min 53 s; GPU peak ~3.35 GiB on a 4 GB RTX A500 Laptop).
The labbook (§2.1) estimated 30–90 min; the measured run was 50 min 53 s.

### 1.1 Requirements (local)

- GPU: NVIDIA RTX A500 Laptop, 4096 MiB, compute capability 8.6 (Ampere).
  The run fits comfortably: peak VRAM ~3.35 GiB. CPU-only fallback exists
  (`torch.cuda.is_available()`), but is not recommended.
- Environment: `benchmark_ssl/.venv` (torch 2.12.0+cu130, sklearn, scipy,
  scikit-image, tifffile, pandas, numpy).
- Data: `data/vanvliet/` at the repo root (all 6 conditions).
- DINOv2 weights: cached at
  `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` (88 MB);
  `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` requires this
  file or internet access on the machine.
- NT-Xent checkpoint for `cnn_frozen`:
  `benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt` (5.9 MB). The script's
  default `--checkpoint` resolves to this path. If it is missing, the script
  logs a warning and sets `checkpoint_path = None`, which makes `cnn_frozen`
  produce random-initialized (not NT-Xent) features — do not reproduce with a
  missing checkpoint.

### 1.2 Exact command

Run from the **repo root** (`research-proj`). All paths below are relative to
the repo root; `--output` is written relative to the current working directory
(so it lands at `<repo>/results/unified_probe_results_cv.json`).

```bash
cd /home/leonard.starke@mediainterface.de/Dokumente/Uni/research-proj
benchmark_ssl/.venv/bin/python benchmark_ssl/probe/unified_edge_probe.py \
    --features all --probe both --epochs 200 --cv-folds 5 --shuffle-baseline \
    --data-root data/vanvliet --output results/unified_probe_results_cv.json
```

This is the exact invocation that produced
`results/unified_probe_results_cv.json` (recorded `args` in the JSON confirm:
`data_root="data/vanvliet"`, `features="all"`, `probe="both"`, `epochs=200`,
`cv_folds=5`, `shuffle_baseline=true`, `seed=42`, `max_pairs=30`, `lr=0.001`,
`batch_size_linear=256`, `batch_size_mlp=128`, `patience=10`,
`eval_every=1`).

### 1.3 What it does

- Scans up to `--max-pairs 30` **consecutive frame pairs** across all 6
  conditions (`rpsM,recA,pheA,metA,cib,trpL`).
- For each feature type, builds per-frame-pair cell-cell edge data: a sample is
  `(feat_i, feat_j, label)` where `label = 1` for same-tracklet cells and for
  parent->child division edges, `0` otherwise (BCE task).
- Runs **KFold(5)** (sklearn `KFold`, `shuffle=True`, `random_state=42`) over
  frame pairs — all conditions, train on K-1 folds, validate on 1 held-out fold.
- Trains `Linear` and `MLP` probes (`--probe both`); MLP uses
  `nn.Sequential(Linear(2*feat_dim, 128), ReLU, Linear(128, 1))`.
- With `--shuffle-baseline`, repeats the CV on per-pair shuffled features
  (labels untouched) as a label-leakage / overfit check.
- Prints results to the console and, when `--output` is given, writes JSON.

### 1.4 Output

- JSON: `results/unified_probe_results_cv.json` (336 KB, created 2026-08-04).
  Top-level keys: `args`, `results`, `rows`.
- Console log (not saved to a file): per-fold progress, per-feature CV summary,
  and the shuffled-baseline block. **Shuffled rows are printed only in the
  console log — they are NOT written into the JSON** (see §4.4).

---

## 2. Cluster (H100) run — `run_unified_probe.slurm`

**Status: PENDING** (script is ready; the local run already produced the same
JSON artifact, so a cluster run is only needed to re-verify on H100 or to
regenerate with different args).

The SLURM script runs the exact same unified probe with the cluster data root
and an H100 partition. Time budget 2 h, memory 64 G — generous for this probe
(peak VRAM ~3.35 GiB even on the A500).

### 2.1 SBATCH directives (verbatim)

```bash
#!/bin/bash
#SBATCH --account=p_scads_celltracking
#SBATCH --job-name=edge_probe
#SBATCH --output=logs/edge_probe-%j.out
#SBATCH --error=logs/edge_probe-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
```

### 2.2 Environment setup inside the script (verbatim)

```bash
export PIP_REQUIRE_VIRTUALENV=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
BENCH=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking
ENV_DIR="$TRK/.venv"

. "$ENV_DIR/bin/activate"
cd "$BENCH"
mkdir -p logs results
```

### 2.3 Exact command run by the job

```bash
python benchmark_ssl/probe/unified_edge_probe.py \
    --features all \
    --probe both \
    --epochs 200 \
    --cv-folds 5 \
    --shuffle-baseline \
    --data-root /data/cat/ws/mawe985g-data/data/celltracking/vanvliet \
    --output results/unified_probe_results_cv.json
```

(`cd "$BENCH"` makes `--output results/unified_probe_results_cv.json` resolve to
`$BENCH/results/unified_probe_results_cv.json`.)

### 2.4 Outputs

- `$BENCH/results/unified_probe_results_cv.json` (identical schema to the local
  run).
- `$BENCH/logs/edge_probe-%j.out` / `$BENCH/logs/edge_probe-%j.err` (SLURM
  stdout/stderr; the JSON is written even if the console log is lost).

### 2.5 Known issue in the script

The post-run "Print results summary" block checks for
`results/unified_probe_results.json` (no `_cv`), but the run writes
`results/unified_probe_results_cv.json`. Because the check is for the wrong
filename, the summary block never fires (it is dead code). This does **not**
affect the probe run itself or the JSON output — only the extra console
summary. Fix: change `results/unified_probe_results.json` to
`results/unified_probe_results_cv.json` in both the `[ -f ... ]` test and the
`open(...)` call.

---

## 3. Legacy single-split probe — `edge_probe.py`

**Status: DONE / SUPERSEDED.** This is the older probe, kept for reference. It
was superseded by the unified 5-fold CV probe (which is the source of truth for
the report).

### 3.1 Differences from the unified probe

| Aspect | `edge_probe.py` (legacy) | `unified_edge_probe.py` |
|---|---|---|
| Feature sets | Regionprops 7D (area/eccentricity/perimeter/solidity/extent/orientation/intensity_mean), Hu moments 20D, combined 27D | rp 7D, hoct19 12D, cnn_frozen 128D, cnn_e2e 128D, dino 384D |
| Edge targets | `target_all` and `target_div` (division-only, harder task) | single all-edges target (same-cell + divisions) |
| Train/val split | single random **80/20** split (`np.random.permutation`, seed 42) | **KFold(5)** over frame pairs, seed 42 |
| Training | `steps` (default 200), patience 20, eval every 20 | `epochs` (default 200), patience 10, eval every 1 |
| Output | always writes `results_edge_probe.txt` | JSON only when `--output` given |
| Shuffle baseline | none | `--shuffle-baseline` |

### 3.2 Command

```bash
cd benchmark_ssl/probe
../.venv/bin/python edge_probe.py --max-pairs 30 --steps 200
```

### 3.3 Output

`benchmark_ssl/probe/results_edge_probe.txt` (created 2026-07-23). Measured
values on the single split (from the on-disk file):

- Regionprops 7D: all-edges Linear 0.5524, all-edges MLP 0.5000; divisions
  Linear 0.8946, divisions MLP 0.9000.
- Hu Moments 20D: all-edges Linear 0.4882, all-edges MLP 0.5149; divisions
  Linear 0.9000, divisions MLP 0.8438.
- Both 27D: all-edges Linear 0.5619, all-edges MLP 0.5000; divisions Linear
  0.8601, divisions MLP 0.9000.

Because it uses a single split and different feature sets, these numbers are
**not directly comparable** to the unified CV results and are not used in the
report. Do not use this script for new measurements.

---

## 4. Key semantics (unified probe)

### 4.1 Features

| Key | Display name | Dimensionality | Notes |
|---|---|---|---|
| `rp` | Regionprops 7D | 7 | eq_diam, intensity_mean, inertia_tensor(4), border_dist |
| `hoct19` | HOCT 19D (2D -> 12D) | 12 | centroid(2), eq_diam, intensity(4: min/max/mean/std), inertia(4), border(1); 2D adaptation of HOCT's 19D |
| `cnn_frozen` | CNN NT-Xent (frozen) | 128 | ScaledCNN (scale=large) frozen, weights from `cnn_ntxent_large.pt` |
| `cnn_e2e` | CNN end-to-end | 128 | ScaledCNN trained jointly with the probe (patches input, `is_e2e=True`) |
| `dino` | DINOv2 (frozen) | 384 | DINOv2 `dinov2_vits14` (small ViT, 21M params), frozen; 64x64 patches resized to 224x224, ImageNet normalization |

`--features all` runs all five. Aliases: `shallow` = `rp`,
`deep` = `cnn_frozen,cnn_e2e,dino`.

### 4.2 Probes

- `Linear`: `nn.Linear(2 * feat_dim, 1)` over `concat(feat_t, feat_n)`.
- `MLP`: `nn.Sequential(Linear(2 * feat_dim, 128), ReLU, Linear(128, 1))`.
- Hyper-parameters: linear lr = `--lr` (1e-3), MLP lr = `--lr / 10` (1e-4);
  batch 256 (linear) / 128 (MLP); Adam; BCEWithLogitsLoss; early stopping
  patience 10 epochs on validation balanced accuracy; best state restored;
  balanced sampling (`WeightedRandomSampler`) for feature-based loaders.
- For `cnn_e2e`, the probe is `CNNProbeE2E` and the whole model trains
  end-to-end.

### 4.3 CV protocol

- `--cv-folds 5` = sklearn `KFold(n_splits=5, shuffle=True, random_state=42)`
  over the frame-pair datasets (all conditions together).
- Report per-feature/per-probe: `bal_acc_mean`, `bal_acc_std`, `bal_acc_min`,
  `bal_acc_max`, `f1_*`, plus per-fold `fold_results` (each with
  `final_bal_acc`, `final_f1`, `final_precision`, `final_recall`, and a full
  `history`).
- `--seed 42` (default) seeds torch, numpy and the KFold RNG.
- Without `--cv-folds` (default 0), the script falls back to a condition split
  (`train_conditions rpsM,recA,pheA,metA` / `val_conditions cib,trpL`), and to
  a random 80/20 split only if the val conditions yield no pairs.

### 4.4 Shuffle baseline and JSON serialization

- `--shuffle-baseline` runs the same CV on shuffled features (per frame pair,
  independent permutation of rows, seed 42), labels unchanged. Its purpose is a
  label-leakage / overfit check. Result: bal_acc ~0.50–0.55, i.e. **no label
  leakage**.
- **Shuffled rows are NOT persisted in the JSON.** The serialization code
  (`main()`/`--output`) writes only the `linear` and `mlp` keys (with
  `history` stripped); `linear_shuffled` / `mlp_shuffled` results exist only in
  the console log. To re-verify the shuffle baseline you must re-run with
  `--shuffle-baseline` and read the console output.
- JSON is written **only when `--output` is given** (default: no save).

### 4.5 Feature cache

`cnn_frozen` and `dino` features are cached per frame under
`benchmark_ssl/probe/feature_cache/` (`<cond>_<exp>_t<frame>_<feat_type>.npy`).
The 2026-08-04 run left 30 `dino` + 30 `cnn_frozen` cache files. On a fresh
machine the cache is rebuilt automatically. `--no-cache` skips
loading/saving.

---

## 5. Cluster requirements

To run the unified probe on the H100 cluster (via `run_unified_probe.slurm`):

1. **Cluster access / SLURM**: submit with `sbatch run_unified_probe.slurm`
   from inside `$BENCH`. Partition `gpu-h100`, account
   `p_scads_celltracking`, 1 GPU per node. No VPN-free path to the cluster
   paths — the run happens on the cluster itself.
2. **Code**: the probe must be present at
   `$BENCH/benchmark_ssl/probe/unified_edge_probe.py` (and `edge_probe.py` if
   needed). `run_unified_probe.slurm` runs `python benchmark_ssl/probe/...`
   after `cd "$BENCH"`.
3. **Environment**: `$TRK/.venv` with torch (CUDA build), sklearn, scipy,
   scikit-image, tifffile, pandas, numpy. The script sets
   `PIP_REQUIRE_VIRTUALENV=false`, thread env vars, `PYTHONNOUSERSITE=1`, and
   unsets `PYTHONPATH`/`PYTHONHOME` to avoid environment shadowing.
4. **Data**: vanvliet at
   `/data/cat/ws/mawe985g-data/data/celltracking/vanvliet` (all 6 conditions,
   `img/*.tif`, `TRA/man_track*.tif`, `TRA/man_track.txt`).
5. **NT-Xent checkpoint**: for `cnn_frozen`, the default `--checkpoint`
   resolves to `$BENCH/benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt`
   (relative to the script location). If the checkpoint lives elsewhere on the
   cluster (e.g. under `$TRK`), pass `--checkpoint` explicitly.
6. **DINOv2 weights**: `torch.hub.load("facebookresearch/dinov2",
   "dinov2_vits14")` downloads/caches into `~/.cache/torch/hub/`. The node
   needs the cached `dinov2_vits14_pretrain.pth` (88 MB) or outbound internet.
7. **Resources**: wall time 2 h (measured local run was 50 min 53 s on a much
   slower GPU), 64 G RAM, 8 CPUs, 1 GPU. Outputs go to
   `$BENCH/results/unified_probe_results_cv.json` and `$BENCH/logs/`.
8. **Known script quirk**: the summary-print block at the end of
   `run_unified_probe.slurm` references `results/unified_probe_results.json`
   instead of `results/unified_probe_results_cv.json`, so it never prints (see
   §2.5). This does not affect the probe or the JSON output.

---

## 6. Experiment status summary

| Experiment | Status | Resources | Outputs |
|---|---|---|---|
| Unified 5-fold CV edge probe (`unified_edge_probe.py`, local A500) | **DONE** (2026-08-04) | RTX A500 Laptop 4 GB (peak ~3.35 GiB), `benchmark_ssl/.venv`, `data/vanvliet`, DINOv2 vits14 cached weights, `cnn_ntxent_large.pt`; 50 min 53 s | `results/unified_probe_results_cv.json` |
| Unified 5-fold CV edge probe via SLURM (`run_unified_probe.slurm`, H100) | **PENDING** (script ready, not submitted) | H100, partition `gpu-h100`, account `p_scads_celltracking`, 1 GPU, 8 CPUs, 64 G, 2 h; cluster venv `$TRK/.venv`; data root `/data/cat/ws/mawe985g-data/data/celltracking/vanvliet` | `$BENCH/results/unified_probe_results_cv.json`, `$BENCH/logs/edge_probe-%j.{out,err}` |
| Legacy edge probe (`edge_probe.py`, single 80/20 split) | **DONE / SUPERSEDED** (2026-07-23) | local; same venv and data | `benchmark_ssl/probe/results_edge_probe.txt` |
| `cnn_probe.py` / `train_cnn_probe.py` (legacy CNN probe artifacts) | **DONE / SUPERSEDED** (medium-scale; all 0.500) | local | `cnn_probe_results.txt` (not in this dir); source of truth for the 0.703 CNN number is the unified `cnn_frozen` (large + 5-fold), NOT this legacy file |

NOT-NEEDED: no further edge-probe experiments are required — the unified run
already reproduces the report's edge-probe table (see §7). The multi-seed
K-sweep and DeepCell cross-dataset experiments (cluster-only) are tracked in
the labbook inventory and are outside this directory.

---

## 7. Measured results ledger (verbatim from `labbook/2026-08-03...` §5, item 1)

Re-run unified 5-fold edge probe (2026-08-04) -> `results/unified_probe_results_cv.json`.
MLP CV values reproduce the report's edge-probe table — DINOv2 **0.8958±0.0362**
(report 0.896), HOCT19 **0.8799±0.0201** (report 0.873, run-to-run), 7D
regionprops **0.8595±0.0137** (report 0.860), CNN-NT-Xent **0.7031±0.0554**
(report 0.703), CNN-e2e **0.5000±0.0000** (report 0.500). Linear probes
0.50–0.57 (report only cites MLP). Shuffle baseline 0.50–0.55 (no label
leakage). KFold(5) + shuffle baseline confirmed in log. 50 min 53 s, GPU peak
~3.35 GiB. NOTE: shuffled rows are NOT persisted in the JSON (serialization
filters to linear/mlp) — they exist only in the run log.

These numbers reproduce the report's edge-probe table (0.896 / 0.873 / 0.860 /
0.703 / 0.500).

Verified against `results/unified_probe_results_cv.json` (bal_acc mean +/- std,
5-fold CV):

| Feature | Probe | bal_acc (mean +/- std) | F1 (mean +/- std) |
|---|---|---|---|
| DINOv2 (frozen, 384D) | MLP | **0.8958 +/- 0.0362** | 0.5659 +/- 0.0903 |
| HOCT 19D -> 12D | MLP | **0.8799 +/- 0.0201** | 0.4235 +/- 0.0763 |
| Regionprops 7D | MLP | **0.8595 +/- 0.0137** | 0.3963 +/- 0.0562 |
| CNN NT-Xent (frozen, 128D) | MLP | **0.7031 +/- 0.0554** | 0.2895 +/- 0.1263 |
| CNN end-to-end (128D) | MLP | **0.5000 +/- 0.0000** | 0.0000 +/- 0.0000 |
| HOCT 19D -> 12D | Linear | 0.5729 +/- 0.0121 | 0.1441 +/- 0.0372 |
| Regionprops 7D | Linear | 0.5574 +/- 0.0304 | 0.1310 +/- 0.0289 |
| CNN NT-Xent (frozen) | Linear | 0.5432 +/- 0.0149 | 0.1288 +/- 0.0350 |
| DINOv2 (frozen) | Linear | 0.5276 +/- 0.0059 | 0.1129 +/- 0.0440 |
| CNN end-to-end | Linear | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 |

Notes:

- The primary report metric is **MLP balanced accuracy** (the linear probes are
  0.50–0.57 and are not cited in the report).
- The report's HOCT value 0.873 differs slightly from the measured 0.8799 —
  this is run-to-run variation (labbook notation "report 0.873, run-to-run").
- `cnn_e2e` at exactly 0.5000 means the end-to-end CNN never separates classes
  (balanced accuracy of a constant predictor).
- Shuffle baseline (label-leakage check) was ~0.50–0.55 and is visible only in
  the console log of the run, not in the JSON.

---

## Appendix A. Resource file inventory (what you need to have in place)

| Resource | Local path | Cluster path |
|---|---|---|
| Probe script | `benchmark_ssl/probe/unified_edge_probe.py` | `$BENCH/benchmark_ssl/probe/unified_edge_probe.py` |
| SLURM script | `benchmark_ssl/probe/run_unified_probe.slurm` | `$BENCH/benchmark_ssl/probe/run_unified_probe.slurm` |
| Legacy probe | `benchmark_ssl/probe/edge_probe.py` | `$BENCH/benchmark_ssl/probe/edge_probe.py` |
| Legacy result | `benchmark_ssl/probe/results_edge_probe.txt` | — |
| Data (vanvliet) | `data/vanvliet/` (6 conditions) | `/data/cat/ws/mawe985g-data/data/celltracking/vanvliet` |
| NT-Xent checkpoint | `benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt` | `$BENCH/benchmark_ssl/cnn_encoder/probe/cnn_ntxent_large.pt` (or pass `--checkpoint`) |
| DINOv2 vits14 weights | `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` | `~/.cache/torch/hub/` on the node (or download) |
| Feature cache | `benchmark_ssl/probe/feature_cache/` (30 dino + 30 cnn_frozen `.npy`) | rebuilt automatically |
| Results JSON | `results/unified_probe_results_cv.json` (repo root) | `$BENCH/results/unified_probe_results_cv.json` |

## Appendix B. Discrepancies found

1. **SLURM summary bug**: `run_unified_probe.slurm` lines 46–55 check
   `results/unified_probe_results.json` but the run writes
   `results/unified_probe_results_cv.json` — the post-run summary never prints
   (cosmetic; does not affect the probe or JSON).
2. **Shuffle baseline not in JSON**: by design, serialization filters to
   `linear`/`mlp`; `linear_shuffled`/`mlp_shuffled` live only in the console
   log. Any later reproduction that needs the shuffle numbers must re-run with
   `--shuffle-baseline` and capture stdout.
3. **HOCT report value**: measured 0.8799 vs report 0.873 — run-to-run
   variation, explicitly annotated in the labbook; all other values match the
   report within rounding.
4. **Results location**: the local JSON is at the **repo root**
   `results/unified_probe_results_cv.json`, not under
   `benchmark_ssl/results/` — because the run executed from the repo root with
   `--output results/...`. `benchmark_ssl/results/` does not exist.
