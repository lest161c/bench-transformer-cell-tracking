# SPEC — Replace cluster_env.sh with shared .env + loader

Status: draft (pending user approval before implementation).

## 1. Problem

Cluster-specific HPC paths are currently hardcoded in
`benchmark_ssl/slurm/cluster_env.sh` and duplicated inline in two other
slurm scripts. Three issues:

1. **Cluster paths leak into git**: the values
   (`/data/cat/ws/lest161c-cell_tracking/...`) are workspace-specific and
   do not belong in version control.
2. **Setup is duplicated** across three locations:
   `benchmark_ssl/slurm/cluster_env.sh`,
   `benchmark_attn/slurm/run_full_bench.slurm` (inline),
   `benchmark_training/scripts/slurm/{run_training,run_diagnostic}.slurm`
   (inline).
3. **Scope creep in `cluster_env.sh`**: it does env setup **and** git
   branch checkout **and** `pip install` **and** `mkdir logs`. These are
   unrelated concerns mixed into one file.

## 2. Goal

Replace the three duplicates with a single source of truth:

- one tracked template (`.env.example`) at the repo root
- one tiny shared loader (`slurm/load_cluster_env.sh`) at the repo root
- one local gitignored copy (`.env`) per developer / per cluster

The loader's responsibility is narrow:
source `.env`, validate paths, export runtime env vars, activate the
uv-managed venv. **Nothing else.**

## 3. Non-goals

- Migrating dependency management from `pip` to `uv` (already in
  progress elsewhere; treat the venv as already populated).
- Removing the `pip install -e .` calls in slurm scripts that install
  the local `trackastra` package — those install local source, not
  third-party deps, and stay.
- Touching the existing inline `pip install native-sparse-attention-pytorch`
  in `benchmark_attn/slurm/run_full_bench.slurm` — that is a
  project-specific dep pending pyproject migration, outside this refactor.
- Adding a Python-dotenv / envdir / direnv dependency. Pure bash only.

## 4. Design

### 4.1 File layout (additions / changes)

| Path | Status | Purpose |
|---|---|---|
| `/.env.example` | add, tracked | Template for cluster paths (TRK, BENCH, DATA_DIR) |
| `/.env.secrets.example` | add, tracked | Template for secrets (HF_TOKEN) |
| `/.env` | already gitignored | Local cluster paths |
| `/.env.secrets` | add gitignore entry | Local secrets |
| `/slurm/load_cluster_env.sh` | add, tracked | Shared loader |
| `/benchmark_ssl/slurm/cluster_env.sh` | delete | Replaced by loader |
| `/benchmark_ssl/slurm/{cross_dataset_eval,h100_conv_race,unified_edge_probe}.slurm` | edit | Source loader instead |
| `/benchmark_attn/slurm/run_full_bench.slurm` | edit | Source loader, drop inline env setup |
| `/benchmark_training/scripts/slurm/{run_training,run_diagnostic}.slurm` | edit | Source loader, drop inline env setup |
| `/.gitignore` | edit | Add `.env.secrets` |
| `/benchmark_ssl/docs/REPRODUCTION.md` | edit | Update §2.3 to point at new layout |
| `/docs/SPEC_cluster_env_refactor.md` | add | This file |

### 4.2 `.env.example` contents

Plain `KEY=value` lines (no `export`, dotenv convention). Comments
explain each variable. Example shape:

```sh
# Cluster paths — replace with your own values.
TRK=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/trackastra
BENCH=/data/cat/ws/lest161c-cell_tracking/lest161c-ssl_cell_tracking-1778979601/bench-transformer-cell-tracking
DATA_DIR=/data/cat/ws/mawe985g-data/data/celltracking

# Derived — defaults to $TRK/.venv if unset. Uncomment to override.
# ENV_DIR=/custom/venv/path
```

### 4.3 `.env.secrets.example` contents

```sh
# Secrets — never commit the populated .env.secrets file.
HF_TOKEN=hf_your_token_here
```

### 4.4 Loader responsibilities (`slurm/load_cluster_env.sh`)

1. **Discover `.env`**: at `<repo_root>/.env`, where `<repo_root>` is
   the directory containing this script's parent (`slurm/`).
   Implementation: `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"`.
2. **Source `.env`** with `set -a; source "$REPO_ROOT/.env"; set +a`.
3. **If `.env` is missing**: print a clear error telling the user to
   `cp .env.example .env` and exit non-zero.
4. **Apply defaults** for derived variables:
   `ENV_DIR="${ENV_DIR:-$TRK/.venv}"`.
5. **Validate required paths**: `$TRK`, `$ENV_DIR/bin/activate`. Fail
   with a clear error if any are missing.
   `$BENCH` is validated only if set (some scripts do not need it).
6. **Export runtime env vars** (identical to current `cluster_env.sh`):
   - `PIP_REQUIRE_VIRTUALENV=false`
   - `OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}`
   - `MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}`
   - `NCCL_DEBUG=WARN`
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
   - `PYTHONNOUSERSITE=1`
   - `unset PYTHONPATH PYTHONHOME`
7. **Activate venv**: `. "$ENV_DIR/bin/activate"`.

The loader does **not**: run `pip install`, run `git` commands, or
`mkdir logs`. Each slurm script owns those for its own job.

### 4.5 Per-script invocation pattern

Each slurm script gets one new line near the top (after `#SBATCH`
directives):

```bash
source "$(dirname "${BASH_SOURCE[0]}")/../../slurm/load_cluster_env.sh"
```

with the relative depth adjusted to the script's location:

- `benchmark_ssl/slurm/*.slurm` → `/../../slurm/load_cluster_env.sh`
- `benchmark_attn/slurm/*.slurm` → `/../../slurm/load_cluster_env.sh`
- `benchmark_training/scripts/slurm/*.slurm` → `/../../../slurm/load_cluster_env.sh`

`${BASH_SOURCE[0]}` is used (not `$0`) so the path resolves correctly
even when the script is `sbatch`'d from a different cwd.

### 4.6 Each slurm script's remaining job-specific steps

After sourcing the loader, each script still owns:

- `cd "$TRK"` (or wherever its working dir should be)
- `git fetch && git checkout <branch> && git pull` for its branch
- `pip install -e .` (or `uv pip install -e .`) for local trackastra —
  only in scripts that train it
- `mkdir -p logs`
- the actual `python -m ...` invocation

### 4.7 DATA_DIR in `benchmark_training/run_training.slurm`

This script currently redefines `DATA_DIR` to
`/data/cat/ws/.../celltracking/vanvliet` (one level deeper than the
`.env` value). After refactor:

- `.env` carries `DATA_DIR=/data/cat/ws/.../celltracking` (matches
  `cluster_env.sh`).
- The script does `DATA_DIR="$DATA_DIR/vanvliet"` immediately after
  sourcing the loader, mirroring the pattern in `benchmark_ssl`.

## 5. Acceptance criteria

Each item must hold after the refactor.

1. `git grep -n "cluster_env.sh"` returns zero hits in tracked files.
2. `git grep -n "OMP_NUM_THREADS=" benchmark_*/slurm` returns zero
   hits outside the loader (env var definition lives in one place).
3. `git grep -n "pip install torch" benchmark_*/slurm` returns zero
   hits — dep installs move out of slurm scripts.
4. `git grep -n "git fetch" benchmark_*/slurm` returns zero hits
   inside `cluster_env.sh` — git is script-specific, not env.
5. `.env.example` is tracked and contains TRK, BENCH, DATA_DIR as
   placeholder values matching today's cluster (so a fresh clone runs).
6. `.env.secrets.example` is tracked; `.env.secrets` is gitignored.
7. `.env` and `.env.secrets` are both gitignored.
8. Each of the six slurm scripts:
   - has exactly one `source ... load_cluster_env.sh` line;
   - has no inline `OMP_NUM_THREADS=`, `NCCL_DEBUG=`, `CUBLAS_WORKSPACE_CONFIG=`,
     `unset PYTHONPATH`, or `. "$VENV/bin/activate"` lines;
   - still has its `#SBATCH` directives, `cd`, `git checkout`,
     `pip install -e .`, `mkdir -p logs`, and the python invocation.
9. `slurm/load_cluster_env.sh`:
   - has a docstring explaining its purpose and scope;
   - prints a helpful error and exits non-zero if `.env` is missing;
   - validates `$TRK` and `$ENV_DIR/bin/activate` exist;
   - does NOT run `pip install`, `git`, or `mkdir`.
10. `bash -n slurm/load_cluster_env.sh` exits 0 (syntax check).

## 6. Migration steps (for the user, after the refactor lands)

1. `git pull` to fetch the new layout.
2. `cp .env.example .env` and fill in real paths (or keep the example
   values if running on the same cluster).
3. `cp .env.secrets.example .env.secrets` and paste the existing
   `HF_TOKEN` (previously in `.env`).
4. Existing `.env` (with HF_TOKEN) can be deleted, or kept as a
   reference — it is gitignored either way.

## 7. Out of scope / not addressed

- Validation that the venv actually has the expected packages
  (trust uv; load-time smoke test is the script's responsibility).
- Per-job env-var overrides (e.g. `TRK=/x sbatch script.slurm`). The
  refactor replaces the override mechanism with "edit `.env`". If
  per-submission overrides are needed later, the loader can be extended
  with a `${VAR:-default}`-style override layer.
- Any changes to Python code that reads `HF_TOKEN`.
