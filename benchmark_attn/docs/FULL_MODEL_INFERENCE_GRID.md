# Full-model inference grid (SPEC 0062)

**Script:** `benchmark_attn/scripts/benchmarks/benchmark_full_model_inference_grid.py`
**Runner:** `benchmark_attn/slurm/run_full_model_inference_grid.slurm`
**Rerun runner:** `benchmark_attn/slurm/run_full_model_inference_grid_rerun.slurm`
**Real-sequence runner:** `benchmark_attn/scripts/benchmarks/benchmark_real_sequence_pipeline.py`
**Spec:** `docs/agent-state/specs/0062-full-model-inference-grid/SPEC.md`
**Tickets:** F1–F8; report integration under SPEC 0066 R2 (2026-11-04); this
document is the F5 protocol and is extended with the job record in F6.

## Purpose and scope

This benchmark measures the **complete Trackastra inference chain** with one
of five attention implementations swapped into every encoder and decoder
layer. It exists because isolated attention timings do not show how much of
the full forward pass an attention variant actually controls once embedding,
the encoder, the decoder, the association head, output normalisation, and the
linker are included.

**This is a timing-only experiment.** It makes no accuracy, tracking-quality,
or out-of-distribution-generalisation claim. The checkpoint was trained at
roughly `N≈50–200` cells per frame; the large-`N` grid is a synthetic stress
test of the inference path, not a quality evaluation. No report or
claim-bearing source is edited by F1–F6.

**Rerun v2 (2026-11-05).** The rerun runner fixes the `build_graph` hot loop,
applies the paper's `0.05` association threshold in the linker lookup, and adds
the `flex_attn` variant. It writes
`full_model_inference_grid_v2*.csv` and `full_model_inference_grid_b64_v2*.csv`
so the original SPEC 0062 results remain untouched.

**Rerun-v2 prep validation (SPEC 0068, 2026-11-09).** All flex plumbing is
committed and validated locally: `DRY_RUN=1` echoes the flex-inclusive v2
commands; the CPU smoke (`--smoke`, N=128) passes with all five variants
`status=ok` including `flex_attn` — flex runs its `torch.compile` path on CPU
under torch ≥ 2.12 (verified 2.12.1 local, 2.14.0 cluster), so the CPU smoke
variant set is now all five variants
(`CPU_SMOKE_VARIANT_KEYS` in `tests/test_full_model_inference_grid.py`);
the fp32 CPU sanity passes (3/3 comparisons); and both grid test files pass
(27 passed). Launch with:

```bash
sbatch benchmark_attn/slurm/run_full_model_inference_grid_rerun.slurm
```

## Real-sequence aggregate at N=32768

The synthetic N grid is a kernel stress test: at `N=32768` it places ~3277
detections in each of 10 frames, whereas vanvliet has median 16, mean 45,
p95 190 and max 460 cells/frame (DeepCell max 235). Candidate-graph
construction therefore sees an unrealistically dense graph and can dominate
the total.

For a natural end-to-end number, use the separate real-sequence runner:

```bash
cd benchmark_attn
python scripts/benchmarks/benchmark_real_sequence_pipeline.py \
  --checkpoint ../checkpoints/k16 \
  --data-root ../data/vanvliet \
  --target-detections 32768 \
  --device cuda --dtype fp16 --batch-size 4 \
  --out results/real_sequence_pipeline_v2.csv \
  --env-json results/real_sequence_pipeline_v2_env.json \
  --summary results/real_sequence_pipeline_v2_summary.csv
```

It runs Trackastra's actual inference path per real sequence
(`get_features` → `build_windows` → `predict_windows` → `build_graph` →
`track_greedy`) and sums timings over enough real sequence prefixes to reach
`N=32768` total detections. The final frame may be truncated to hit `N`
exactly; that is recorded in the CSV and env JSON. `predict_windows` applies
the paper's `0.05` association threshold before `build_graph`, so the graph
uses realistic sparse weights. This is a throughput measurement across real
videos, not a single-video latency claim; keep it separate from the synthetic
N grid.

## Variants and checkpoint

Default checkpoint: `checkpoints/k16` (`config.yaml`, `train_config.yaml`,
`model.pt`). The selected path is recorded in the raw CSV and the environment
JSON. The architecture comes from the checkpoint config, not from the
benchmark script. For k16 the expected values are `coord_dim=2`, `feat_dim=7`,
`d_model=320`, `nhead=4`, 6 encoder + 6 decoder layers, `window=10`,
`dropout=0.01`, `attn_positional_bias=rope`,
`attn_positional_bias_n_spatial=16`, `attn_dist_mode=v0`,
`spatial_pos_cutoff=256`, `knn_neighbors=16`, `causal_norm=quiet_softmax`.

| CSV key | Trackastra class | Positional mode | KNN | Model-level `dist_2d` |
|---|---|---|---|---|
| `relative_pos` | `RelativePositionalAttention` | `rope` | ignored | passed but ignored; own per-layer `cdist` |
| `cached_dist` | `CachedDistAttention` | `rope` | ignored | consumed for the spatial cutoff |
| `sparse_mask_k16` | `KNNMaskSparseAttention` | `rope` | 16 | passed but ignored |
| `sparse_gather_k16` | `GatherSparseAttention` | `rope` | 16 | passed but ignored |
| `flex_attn` | `TrackastraFlexAttention` | `rope` | ignored | passed but ignored; block-mask cutoff |

Each variant gets a fresh model instance loaded from the checkpoint so a swap
cannot contaminate the next variant. Attention weights are copied by exact
key (`q_pro.*`, `k_pro.*`, `v_pro.*`, `proj.*`, `rot_pos_enc.*`) with a strict
load for the default k16 rope checkpoint. `DenseFlash` and other attention
classes remain out of scope. The `flex_attn` adapter reuses the checkpoint
RoPE weights and applies the paper's `0.05` edge threshold and spatial cutoff.

## Synthetic data recipe

Inputs are generated once per `(B, N)` with seed 42 and reused across
variants. No dataset, image, mask, or feature extraction is used.

- `coords`: `(B, N, 3)` float. Column 0 is time; columns 1–2 are spatial.
  Timepoints are `torch.arange(N) % 10`, so every one of the `window=10`
  frames is populated for every grid `N`.
- `features`: `(B, N, 7)` float, standard normal (wrfeat-shaped synthetic
  features).
- Spatial layout `uniform` (default): a seeded uniform draw over a square
  whose side is `10 * sqrt(ceil(N / window))`, keeping nominal density roughly
  constant as `N` grows. Optional `tiled`: a deterministic jittered lattice
  with the same side. Both are synthetic, not real coordinates.
- `knn_indices`: `(B, N, 16)` `int64` from
  `compute_knn_indices(coords, knn_neighbors=16, query_chunk=512)` in
  `benchmark_attn/src/bench/data.py`, computed **outside** the timed region.
- `timepoints` for `normalize_output`: `coords[..., 0].long()`, `(B, N)`.
- `padding_mask=None`.

For `B > 1`, the `B=1` sample is replicated across the batch and its KNN
indices are expanded, so batch scaling is measured rather than data variation.
`B=1` is the primary production-large-`N` case.

### KNN self-inclusion caveat

`compute_knn_indices` includes the query token itself; the model's internal
fallback uses `topk(K+1)[..., 1:]` and excludes self. Both sparse variants
receive the same precomputed indices, so the sparse-vs-sparse equivalence check
is unaffected and timing is unaffected. This convention is not an accuracy
statement and no tracking-quality claim is made from it.

## Shared `N×N` costs and the Gather speedup cap

Several costs are paid by every variant, so attention-only speedups overstate
end-to-end gains:

- **Model-level `dist_2d`.** `TrackingTransformer.forward` computes a
  `torch.cdist` over all `N` nodes unconditionally, even for
  `sparse_gather_k16` which ignores it. It is inside `embed_ms` and
  `full_forward_ms`. `dist_2d_standalone_ms` reports a standalone estimate of
  this already-included cost; it is **never subtracted** from any column.
- **Logits `einsum`.** `A = einsum("bnd,bmd->bnm", ...)` produces an `N×N`
  association matrix for every variant. `einsum_ms` isolates it and
  `head_logits_ms` includes it.
- **`normalize_output`.** It computes a spatial `cdist`, builds the invalid
  mask, and runs `blockwise_causal_norm`, another shared `N×N` cost.
  `normalize_ms` and peak memory report it.

`GatherSparseAttention` therefore cannot eliminate these shared costs. Its
end-to-end speedup is capped by them even when its isolated attention kernel is
much faster; the summary reports both `attention_share` and
`attention_share_of_forward` so the remaining headroom is visible.

## Timing protocol

For each `(variant, N, B)`:

1. Load a fresh model, build inputs, and run `warmup` untimed full-chain
   passes (`5` on CUDA, `1` for CPU smoke).
2. Register component timers around the transformer forward,
   `transformer.proj`, every `encoder[i]` / `encoder[i].attn`, every
   `decoder[i]` / `decoder[i].attn`, `head_x`, and `head_y`.
3. For each of `repeats` (`3` primary):
   - synchronise, run `gc.collect()` plus `torch.cuda.empty_cache()`
     (`_clear_device_cache`), synchronise again, and reset peak-memory
     statistics — the same pre-baseline protocol as the isolated attention
     benchmark (`src/bench/timing.py::measure`);
   - use CUDA events on GPU and `time.perf_counter` at the same boundaries on
     CPU;
   - run `model.transformer(...)` and `normalize_output(...)` under
     `torch.no_grad()`;
   - measure memory per repeat (SPEC 0062 §12): synchronise and capture the
     allocated/reserved forward peaks after the forward; reset the peak
     statistics, run `normalize_output`, synchronise, and capture the
     allocated/reserved normalization peaks; report the total peaks
     (`max` of the two stages), the per-stage peaks, the pre-repeat
     allocated baseline, and keep `peak_mem_mb` as the incremental delta
     (all `0.0` on CPU; empty cells on `oom`/`err`);
   - time the CPU linker separately;
   - in the `finally` block, outside the timed region, remove the hooks,
     release the repeat's `association`/`normalized_association` tensors
     (set to `None`), and call `_clear_device_cache` once, so no `N×N`
     tensor survives into the next repeat; then write one raw row.
4. On OOM or any exception, record a row with status `oom` or
   `err: <message>` and continue with the next configuration; the same
   `finally` cleanup (hooks, tensors, cache) runs for success and failure.

`total_ms = full_forward_ms + normalize_ms + linker_ms` for `B=1`. The forward
and normalisation are GPU-timed while the linker is CPU-timed, so `total_ms` is
a synthetic end-to-end **sum**, not a wall-clock overlap measurement. For
`B>1`, linker timing is skipped and `total_ms = full_forward_ms +
normalize_ms`. `forward_closure_error_ms = full_forward_ms - (embed_ms +
encoder_ms + decoder_ms + head_logits_ms)` is a diagnostic only; a large value
indicates hook-boundary gaps and is warned on, never used to correct a row.

Sanity checks run in fp32 on CPU with TF32 disabled; the primary H100 timing
uses fp16. `torch.cdist`/SDPA numerics differ by dtype, the timing dtype is
recorded in every row and the environment JSON, and no fp16 result is presented
as an accuracy measurement. If fp16 `cdist` fails on the cluster, fall back to
fp32 and record the changed dtype in the environment JSON and docs.

## Linker timing (F4)

The greedy linker is the default: `build_graph` followed by `track_greedy`.
Nodes are one per token with `{"id": i, "time": int(timepoints[0, i]),
"label": i, "coords": [y, x]}`. The normalised association matrix is copied to
CPU once outside the timed linker region and exposed through a read-only
consecutive-frame weight lookup, so no `N×N` Python dictionary is
materialised. The lookup drops associations below the Trackastra edge
threshold `0.05`, matching `predict_windows`. `build_graph` uses
`use_distance=False`, `max_distance=256`,
`max_neighbors=10`, `delta_t=1`; `track_greedy` uses `allow_divisions=True`,
`threshold=0.5`. `linker_build_graph_ms` and `linker_greedy_ms` are CPU wall
times and `linker_ms` is their sum.

ILP is optional. With `--linker-mode ilp`, `track_ilp(candidate_graph,
ilp_config="gt")` runs only when `motile` imports; otherwise the row keeps its
forward status and records `linker_status="ilp_unavailable"` with empty linker
cells. Linker timing is skipped for `B>1` and recorded as
`linker_status="skipped_b_gt_1"`.

## Output schemas

All artifact paths are relative to `benchmark_attn/` when running from that
directory.

### Raw CSV

`results/full_model_inference_grid.csv` has one row per
`(variant, N, B, repeat_idx)`. The v2 B=1 grid has `5 × 7 × 1 × 3 = 105`
planned rows (`4 × 7 × 1 × 3 = 84` in the original four-variant run),
including `oom`/`err` rows.

```
variant,N,B,repeat_idx,status,checkpoint,dtype,seed,knn_neighbors,
coord_layout,linker_mode,linker_status,
full_forward_ms,embed_ms,encoder_ms,encoder_attn_ms,decoder_ms,
decoder_attn_ms,head_logits_ms,head_x_ms,head_y_ms,einsum_ms,
embed_proj_ms,normalize_ms,linker_build_graph_ms,linker_greedy_ms,
linker_ms,total_ms,peak_mem_mb,baseline_mem_mb,peak_mem_total_mb,
peak_mem_reserved_mb,forward_peak_mem_total_mb,forward_peak_mem_reserved_mb,
normalize_peak_mem_total_mb,normalize_peak_mem_reserved_mb,
dist_2d_standalone_ms,forward_closure_error_ms,error
```

| Column | Meaning |
|---|---|
| `variant`, `N`, `B`, `repeat_idx` | Grid coordinates and repeat index. |
| `status` | `ok`, `oom`, or `err: <message>`. |
| `checkpoint`, `dtype`, `seed`, `knn_neighbors`, `coord_layout`, `linker_mode` | Provenance fields copied into every row. |
| `linker_status` | `ok`, `skipped_b_gt_1`, `ilp_unavailable`, or `oom`/`err` when only the linker fails. |
| `full_forward_ms` | Event pair around `model.transformer(...)`, including embedding, model-level `dist_2d`, all layers, heads, and the logits `einsum`. |
| `embed_ms` | Forward start to entry of `encoder[0]`; includes `_embed` and the unconditional model-level `dist_2d`. |
| `encoder_ms` | Entry of `encoder[0]` to exit of `encoder[-1]`. |
| `encoder_attn_ms` | Sum of the `encoder[i].attn` hook intervals. |
| `decoder_ms` | Entry of `decoder[0]` to exit of `decoder[-1]`. |
| `decoder_attn_ms` | Sum of the `decoder[i].attn` hook intervals. |
| `head_logits_ms` | Exit of `decoder[-1]` to exit of the transformer forward; includes `head_x`, `head_y`, and the `einsum`. |
| `head_x_ms`, `head_y_ms` | Hook intervals for the two heads. |
| `einsum_ms` | Exit of `head_y` to exit of the transformer forward. |
| `embed_proj_ms` | `transformer.proj` hook interval; diagnostic only. |
| `normalize_ms` | Event pair around `normalize_output(A, timepoints, coords)`. |
| `linker_build_graph_ms`, `linker_greedy_ms`, `linker_ms` | CPU linker stages and their sum. |
| `total_ms` | `full_forward_ms + normalize_ms + linker_ms` for `B=1`; forward + normalize for `B>1`. Mixed GPU/CPU sum. |
| `peak_mem_mb` | Incremental CUDA peak for one repeat (`peak_mem_total_mb - baseline_mem_mb`, floored at 0); `0.0` on CPU. Kept for compatibility with earlier runs. |
| `baseline_mem_mb` | CUDA memory allocated before the repeat (model weights plus persistent tensors), in MiB; `0.0` on CPU. |
| `peak_mem_total_mb` | **Total peak VRAM allocated** across forward + normalize (`max` of the two stage peaks), in MiB; `0.0` on CPU. |
| `peak_mem_reserved_mb` | **Total peak VRAM reserved** by the caching allocator across forward + normalize, in MiB; `0.0` on CPU. |
| `forward_peak_mem_total_mb`, `forward_peak_mem_reserved_mb` | Allocated/reserved peaks captured after the forward, before `normalize_output`. |
| `normalize_peak_mem_total_mb`, `normalize_peak_mem_reserved_mb` | Allocated/reserved peaks captured after `normalize_output` (peak statistics are reset between the two stages). |

Memory semantics (SPEC 0062 §12): **allocated** bytes are live tensor
payload; **reserved** bytes are caching-allocator segments, including cached
free blocks (reserved ≥ allocated, and reserved is what can hit the VRAM
limit). Peaks are read after `torch.cuda.synchronize()`; peak statistics are
reset once before the repeat and again between the forward and
`normalize_output` so the two stages are separable. The pre-repeat protocol
(synchronise, `gc.collect()` + `empty_cache()` via `_clear_device_cache`,
synchronise, reset peak statistics, capture baseline) matches the isolated
attention benchmark (`src/bench/timing.py::measure`), and each repeat's
`association`/`normalized_association` tensors are released after the repeat
so the baseline reflects only the model weights plus persistent inputs. All
memory cells are empty on `oom`/`err` rows and `0.0` on CPU.
| `dist_2d_standalone_ms` | Standalone `torch.cdist` estimate on the same `(B, N)` shape, repeated in each row; already included in the forward, never subtracted. |
| `forward_closure_error_ms` | Diagnostic closure residual; not a correction. |
| `error` | Empty on success, otherwise the exception type/message. |

### Summary CSV (F6 owns it)

`results/full_model_inference_grid_summary.csv` has one row per
`(variant, N, B, linker_mode)`, aggregating only `status="ok"` repeats (F6
replaces the explicit F3 placeholder):

- `n_ok`, `n_oom`, `n_err`;
- mean and sample standard deviation (`n-1`) for `full_forward_ms`,
  `total_ms`, `embed_ms`, `encoder_ms`, `encoder_attn_ms`, `decoder_ms`,
  `decoder_attn_ms`, `head_logits_ms`, `einsum_ms`, `normalize_ms`,
  `linker_build_graph_ms`, `linker_greedy_ms`, `linker_ms`;
- `peak_mem_mb_max`;
- mean and sample standard deviation (`n-1`) for `baseline_mem_mb`,
  `peak_mem_total_mb`, `peak_mem_reserved_mb`, `forward_peak_mem_total_mb`,
  `forward_peak_mem_reserved_mb`, `normalize_peak_mem_total_mb`,
  `normalize_peak_mem_reserved_mb` (SPEC 0062 §12);
- total-VRAM maxima over the `ok` repeats: `peak_mem_total_mb_max`,
  `peak_mem_reserved_mb_max`, `forward_peak_mem_total_mb_max`,
  `normalize_peak_mem_total_mb_max`;
- `speedup_vs_relative_pos = mean(total_ms of relative_pos) / mean(total_ms of
  variant)` at the same `(N, B)`;
- `attention_share = (encoder_attn_ms + decoder_attn_ms) / total_ms`;
- `attention_share_of_forward = (encoder_attn_ms + decoder_attn_ms) /
  full_forward_ms`;
- `forward_speedup_vs_relative_pos = mean(full_forward_ms of relative_pos) /
  mean(full_forward_ms of variant)`.

At `N=32768` the summary report prints total speedup and attention share for
every `ok` variant and marks `n/a` where the baseline or variant OOMed. The F6
analysis script owns the summary CSV/PNG and the printed table; this document
only defines the contract.

### Environment JSON

`results/full_model_inference_grid_env.json` contains exactly these named
fields, with no environment dump and no secrets:

```
timestamp_utc, command, python_version, platform,
torch_version, cuda_version, cudnn_version, gpu_name, device, dtype,
git_commit, git_dirty, checkpoint_path, checkpoint_sha256,
model_config, variants, Ns, batch_sizes, repeats, warmup, seed,
coord_layout, knn_neighbors, linker_mode, motile_available
```

The checkpoint SHA-256, model config, dtype/device, torch/CUDA versions, git
commit, and full grid definition make the run traceable. The F3 CPU smoke
recorded checkpoint SHA-256
`cc36f8a3776f475af057def227276b34dbdc60bfb099b390e92ec78eacdad2b0`; the H100
run re-records its own value.

## Exact run commands

CPU smoke (from `benchmark_attn/`; writes only the smoke CSV and never
overwrites the primary CSV; runs all five registered variants including
`flex_attn`, which exercises its `torch.compile` path on CPU):

```bash
cd benchmark_attn
.venv/bin/python scripts/benchmarks/benchmark_full_model_inference_grid.py \
  --checkpoint ../checkpoints/k16 \
  --smoke
```

Local (non-cluster) note: the `benchmark_attn/.venv` (Python 3.14) lacks
`dask`/`trackastra`/`lz4`/`numba`; the smoke and sanity paths were validated
with the system Python 3.12 interpreter, which has `lz4`/`numba` but needs
`dask` (pure-Python) plus the trackastra checkout on `PYTHONPATH`:

```bash
cd benchmark_attn
PYTHONPATH="$PWD/../trackastra:<dir-with-dask>" python3 \
  scripts/benchmarks/benchmark_full_model_inference_grid.py \
  --checkpoint ../checkpoints/k16 --smoke    # or --sanity
```

On the cluster, `$TRK/.venv` already has the full stack and neither note
applies.

fp32 CPU sanity/equivalence (from the repository root):

```bash
python3 benchmark_attn/scripts/benchmarks/benchmark_full_model_inference_grid.py \
  --checkpoint checkpoints/k16 \
  --sanity
```

Primary H100 invocation (from `benchmark_attn/`):

```bash
cd benchmark_attn
.venv/bin/python scripts/benchmarks/benchmark_full_model_inference_grid.py \
  --checkpoint ../checkpoints/k16 \
  --variants relative_pos,cached_dist,sparse_mask_k16,sparse_gather_k16 \
  --Ns 512,1024,2048,4096,8192,16384,32768 \
  --batch-sizes 1 \
  --repeats 3 \
  --warmup 5 \
  --seed 42 \
  --device cuda \
  --dtype fp16 \
  --coord-layout uniform \
  --linker-mode greedy \
  --out results/full_model_inference_grid.csv \
  --env-json results/full_model_inference_grid_env.json \
  --summary results/full_model_inference_grid_summary.csv
```

Slurm runner (from the repository root; logs go to `benchmark_attn/logs/`, so
create that directory once before the first submission):

```bash
mkdir -p benchmark_attn/logs
bash -n benchmark_attn/slurm/run_full_model_inference_grid.slurm
DRY_RUN=1 bash benchmark_attn/slurm/run_full_model_inference_grid.slurm
sbatch benchmark_attn/slurm/run_full_model_inference_grid.slurm
```

The runner sources `$HOME/.bench.env` and
`$REPO_ROOT/slurm/load_cluster_env.sh`, uses 1 GPU / 8 CPUs / 90 GB, and skips
the primary grid when `results/full_model_inference_grid.csv` already exists.

**Cluster interpreter (environment note).** Every python invocation in the
runner uses `"$TRK/.venv/bin/python"` (the Trackastra checkout's
virtualenv), not `benchmark_attn/.venv`: the benchmark venv lacks
`dask`/`trackastra`, while `$TRK/.venv` has both, with trackastra installed
as an editable install. `$TRK` is defined in `$HOME/.bench.env` and validated
by `load_cluster_env.sh`. The checkpoint stays at `../checkpoints/k16`
relative to `benchmark_attn/` (the absolute cluster path is recorded in the
environment JSON). The local command blocks above keep `.venv/bin/python` as
written; on the cluster, the runner substitutes `$TRK/.venv/bin/python`. The
optional `B=8/16` up-to-`N=8192` invocation is commented out in the runner and
writes separate `*_batch.*` files if it is ever enabled.

Summary (F6, from `benchmark_attn/`):

```bash
cd benchmark_attn
.venv/bin/python analysis/summarize_full_model_inference_grid.py
```

All-metrics plot (additive; from `benchmark_attn/`):

```bash
cd benchmark_attn
.venv/bin/python analysis/summarize_full_model_inference_grid.py \
  --plot-all results/full_model_inference_grid_all_metrics.png
```

`--plot-all` (optional argument, default
`results/full_model_inference_grid_all_metrics.png`) writes a nine-panel
PNG of every measured summary metric against `N` on a log2 axis, one
line+marker per variant with a shared legend: total and forward speedup vs
`relative_pos`, `total_ms`, `full_forward_ms`, the attention module time
(`encoder_attn_ms + decoder_attn_ms`), `normalize_ms`, `linker_ms`
(`B=1` greedy only — the linker is not timed for `B>1`),
`attention_share` of `total_ms`, and the total peak VRAM
(`peak_mem_reserved_mb_max`, y-label `Total peak VRAM (reserved, MB)`).
Timings use log y-axes; NaN/missing values
are skipped per variant and panel, and an empty panel is annotated
"no data". The two-panel `--plot` summary PNG is unchanged. This figure is
an analysis artifact for reading the F6 numbers; it makes no accuracy or
tracking-quality claim and is not a report figure.

Parts plots (additive; from `benchmark_attn/`):

```bash
cd benchmark_attn
.venv/bin/python analysis/summarize_full_model_inference_grid.py \
  --plot-parts results/full_model_inference_grid_parts
```

B=64 batch run (the B=64 grid writes `results/full_model_inference_grid_b64.csv`
and `_b64_summary.csv`; linker skipped, see caveat below):

```bash
cd benchmark_attn
.venv/bin/python analysis/summarize_full_model_inference_grid.py \
  --raw results/full_model_inference_grid_b64.csv \
  --env-json results/full_model_inference_grid_b64_env.json \
  --out results/full_model_inference_grid_b64_summary.csv \
  --plot-batch 64 \
  --plot-parts results/full_model_inference_grid_b64_parts
```

**Run status:** the `B=64` grid was executed under Slurm job `4525463`
(2026-11-04, H100; first executed under job `4525220` the same day, which is
superseded — see the memory note in the "H100 total-VRAM rerun" section
below) — 60 raw rows (54 `ok`, 6 `oom`: `relative_pos` and `cached_dist` at
`N=8192`), 20 summary rows, all rows
`linker_status=skipped_b_gt_1`; see the "H100 total-VRAM rerun" section
below for the memory and speedup tables.

`--plot-parts` (optional argument, default prefix
`results/full_model_inference_grid_parts`) writes three decomposition PNGs
from one prefix, selecting the rows of the batch size given by `--plot-batch`
(int, default 1): `<prefix>_time_fractions.png` — 2×2 panels, one per
variant (`relative_pos`, `cached_dist`, `sparse_mask_k16`,
`sparse_gather_k16`, `flex_attn`), each a 100% stacked area over `N` of the attention
(`encoder_attn_ms + decoder_attn_ms`), `normalize_ms`, `linker_ms`, and
remaining "other" fraction of `total_ms`, with a shared legend and
missing/NaN cells dropped per variant; `<prefix>_forward_time.png` —
log-log forward time (`full_forward_ms`) vs sequence length; and
`<prefix>_memory.png` — log-log **total peak VRAM**
(`peak_mem_reserved_mb_max`, y-label `Total peak VRAM (reserved, MB)`;
not the `peak_mem_mb` delta) vs sequence length
with a grey dashed 94 GB H100 VRAM reference line. The forward-time
and memory figures are separate single panels styled after the attention-only
H100 report panels (`curated/generate_report.py::panel_speed_vs_n_h100` and
`panel_memory_vs_n`): one solid measured line per variant in Okabe-Ito
colours (black, vermilion, sky blue, orange; `o` markers, size 2.2,
linewidth 0.8), a frameless method legend in the upper left, y major ticks
labelled as base-10 powers, and no projections (all plotted `N` are
measured). All three figures space `N` logarithmically (base 2) but label
the x ticks with the plain integer sequence lengths (512, 1024, 2048, …),
never power notation. The matplotlib fallback behaviour matches
`--plot`/`--plot-all`. `--plot-batch` affects `--plot-parts` only;
`--plot`/`--plot-all` are unchanged. All three are analysis artifacts for
reading the F6 numbers, not report figures.

**Linker-skipped caveat (`B>1`).** For `B>1` the CPU linker is not timed
(`linker_status=skipped_b_gt_1`), `linker_ms` is empty, and
`total_ms = full_forward_ms + normalize_ms`. The `--plot-batch 64`
time-fraction figure therefore treats the missing linker as 0 — the "linker"
band and legend entry appear at zero — and the "other" fraction keeps its
residual meaning. Do not compare a B=64 `total_ms` decomposition against the
B=1 one as if both included linker time.

Component-ratio plot (additive; from `benchmark_attn/`):

```bash
cd benchmark_attn
.venv/bin/python analysis/summarize_full_model_inference_grid.py \
  --plot-ratios results/full_model_inference_grid_ratios
```

B=64 batch run:

```bash
cd benchmark_attn
.venv/bin/python analysis/summarize_full_model_inference_grid.py \
  --raw results/full_model_inference_grid_b64.csv \
  --env-json results/full_model_inference_grid_b64_env.json \
  --out results/full_model_inference_grid_b64_summary.csv \
  --plot-batch 64 \
  --plot-ratios results/full_model_inference_grid_b64_ratios
```

`--plot-ratios` (optional argument, default prefix
`results/full_model_inference_grid_ratios`) writes
`<prefix>_component_ratios.png`, a small-multiple figure with one panel per
timed component — `total_ms`, `full_forward_ms`, attention
(`encoder_attn_ms + decoder_attn_ms`), `normalize_ms`, and `linker_ms`
(only when the selected batch has linker timings, i.e. `B=1`; the linker
panel is omitted for `B>1` because the linker is skipped) — showing per
non-baseline variant (`cached_dist`, `sparse_mask_k16`, `sparse_gather_k16`,
`flex_attn`;
the Okabe-Ito colours of the other parts figures) the component time ratio
variant / `relative_pos` at the same `(N, B)` on a logarithmic y-axis with a
grey dashed parity line at 1.0, so **ratios below 1.0 mean the variant's
component is faster**. The x ticks are the plain integer sequence lengths of
the batch grid (`512, 1024, 2048, 4096, 8192, 16384, 32768` for the primary
B=1 grid; `512–8192` for B=64), never power notation. Missing or
non-positive ratios are skipped per variant and panel — e.g. in the B=64 run
`relative_pos` and `cached_dist` OOM at `N=8192`, so no variant shows a
ratio there. The B=1 figure is a 2×3 grid (five panels plus one slot
annotating the ratio direction); a B>1 batch gives a 2×2 grid of the four
linker-free panels. `--plot-batch` selects the batch (default 1);
`--plot`/`--plot-all`/`--plot-parts` are unchanged. The figure is an
analysis artifact for reading the F6 numbers, not a report figure.

## Tests

From the repository root:

```bash
python3 -m pytest benchmark_attn/tests/test_full_model_inference_grid.py benchmark_attn/tests/test_summarize_full_model_inference_grid.py -q
```

The checkpoint-backed construction/equivalence/smoke tests skip when
`checkpoints/k16` is absent.

## Caveats

- **Training range.** The k16 checkpoint was trained at roughly `N≈50–200`
  cells per window. `N` up to 32768 is a synthetic timing stress test.
- **No accuracy or OOD claim.** The synthetic inputs, the tiled layout, and
  the KNN self-inclusion convention are benchmark fixtures, not a quality
  evaluation.
- **B=1 primary.** The production-large-`N` case is `B=1`; the optional
  `B=8/16` grid is prepared but not run under the approved scope.
- **fp16 timing vs fp32 sanity.** Equivalence is checked in fp32; the H100
  timing uses fp16 and is labelled as such. Do not compare dtype-mixed rows as
  if they were the same numerics.
- **Mixed GPU/CPU total.** `total_ms` adds a GPU forward to a CPU linker; it is
  not a wall-clock overlap measurement.
- **Total peak VRAM vs incremental delta (SPEC 0062 §12).** `peak_mem_mb`
  remains the incremental delta (total peak minus the pre-repeat allocated
  baseline) for compatibility; `peak_mem_total_mb` and `peak_mem_reserved_mb`
  report the absolute total peak VRAM, where **allocated** counts live tensor
  bytes and **reserved** counts caching-allocator segments including cached
  free blocks (reserved ≥ allocated; reserved is what can hit the VRAM
  limit). The memory figures plot the reserved total against the 94 GB H100
  line, never the delta.
- **Shared `N×N` costs.** `dist_2d`, the logits `einsum`, and
  `normalize_output` cap end-to-end Gather speedups and are reported explicitly
  rather than hidden.
- **F4/F6 ownership.** F4 owns the linker timing implementation; F6 owns the
  H100 execution, the summary script/CSV/PNG, and the `N=32768` table. F7's
  report integration is DONE under SPEC 0066 R2 (2026-11-04); see
  `docs/agent-state/specs/0066-inference-scaling-narrative/evidence/R2-INTEGRATION.md`
  and `REPRODUCIBILITY.md` §1a.

## H100 job record (F6)

**Job:** Slurm job `4522664` (partition `capella`, NVIDIA H100), completed
2026-11-03, submitted through the owner/runner workflow. Logs:
`benchmark_attn/logs/full_model_inference_grid_4522664.out` and
`.err`; the `.out` records the grid echo and
`wrote 84 raw rows to results/full_model_inference_grid.csv`.

**Environment (from `results/full_model_inference_grid_env.json`):**
Python 3.10.12, torch 2.14.0+cu130, CUDA 13.0, cuDNN 92400, GPU
`NVIDIA H100`, device `cuda`, dtype `fp16`, seed 42, repeats 3, warmup 5,
coord layout `uniform`, `knn_neighbors=16`, linker `greedy`,
`motile_available=false`, git commit
`2a826dcfa286e0f9ef97c810b61376f7dd7a1e1c` (`git_dirty: true`), checkpoint
`../checkpoints/k16` (absolute cluster path
`/data/cat/ws/lest161c-ssl_cell_tracking/bench-transformer-cell-tracking/checkpoints/k16`),
checkpoint SHA-256
`cc36f8a3776f475af057def227276b34dbdc60bfb099b390e92ec78eacdad2b0` —
identical to the F3 CPU smoke value, and the model config matches the
expected k16 values (`d_model=320`, `nhead=4`, 6+6 layers, `window=10`,
`rope`, `spatial_pos_cutoff=256`, `knn_neighbors=16`,
`causal_norm=quiet_softmax`).

**Row/status counts:** the raw CSV contains 84 data rows
(4 variants × 7 `N` × `B=1` × 3 repeats), every row `status=ok` and
`linker_status=ok` — no `oom`, no `err` rows. The summary CSV has 28 rows.
This run predates the total-peak-VRAM columns of SPEC 0062 §12: its raw CSV
and summary carried only `peak_mem_mb` (the incremental delta — its
identical 27136 MB `peak_mem_mb_max` values are **not** total memory). The
owner reran the grid with the new columns — first as job `4525220`
(2026-11-04, superseded the same day, see below), then as the FINAL
memory-cleanup rerun **job `4525463`**; the current
`results/full_model_inference_grid{,_summary}.csv` come from job `4525463`,
and the corrected total-VRAM tables are in the "H100 total-VRAM rerun"
section below.

**`N=32768` result table** (`B=1`, greedy linker; means over the 3 ok
repeats, from `results/full_model_inference_grid_summary.csv`):

| Variant | total_ms | forward_ms | attn share of total | attn share of forward | total speedup | forward speedup |
|---|---:|---:|---:|---:|---:|---:|
| `relative_pos` | 9027.19 | 617.93 | 0.0663 | 0.9684 | 1.00x | 1.00x |
| `cached_dist` | 8996.22 | 568.05 | 0.0610 | 0.9656 | 1.0034x | 1.0878x |
| `sparse_mask_k16` | 8558.63 | 169.73 | 0.0176 | 0.8887 | 1.0547x | 3.6406x |
| `sparse_gather_k16` | 8486.26 | 80.57 | 0.0072 | 0.7589 | 1.0637x | 7.6693x |

For `sparse_gather_k16` the CPU linker alone is 8043.49 ms
(`build_graph` 7575.69 + `track_greedy` 467.81) and `normalize_output` adds
362.20 ms — both dwarf the 80.57 ms forward. `total_ms` is
`full_forward_ms + normalize_ms + linker_ms` and therefore **includes the
CPU linker**, which is timed on the CPU while the forward/normalisation are
GPU-timed (a synthetic sum, not a wall-clock overlap measurement).

**Key finding:** at `N=32768` the run is linker-bound, not attention-bound.
`Gather` makes the forward 7.67x faster, but total throughput is only 1.06x
faster because ~8.0 s of the ~8.5 s total is the CPU linker plus the shared
`N×N` costs (`dist_2d`, logits `einsum`, `normalize_output`). Timing-only
result on synthetic data; no accuracy or tracking-quality claim.

## H100 total-VRAM rerun (job 4525463, FINAL — F6 correction + F8 `B=64` comparison)

> **Supersedes job `4525220` (same day, 2026-11-04).** The first total-VRAM
> rerun produced impossible values because the repeat loop did not
> `gc.collect()` before `empty_cache()` and did not release the repeat's
> `association`/`normalized_association` tensors between repeats — previous
> repeats' `N×N` tensors were counted inside the next repeat's baselines and
> peaks. All job-`4525220` memory values (e.g. the `B=1` `N=32768` reserved
> 29978/40236/30826/32300 MiB table and the `B=64` tables quoted from it) are
> contaminated history and must not be quoted. The tables below are the
> final, clean measurement.

**Job:** Slurm job `4525463` (partition `capella`, NVIDIA H100), completed
2026-11-04, submitted through the owner/runner workflow. Logs:
`benchmark_attn/logs/full_model_memclean_4525463.out` and `.err`. One job
produced **both** grids: the full `B=1` grid (rerun so the CSVs carry the
SPEC 0062 §12 total-peak-VRAM columns, superseding the job-`4522664` CSV)
and the `B=64` grid (`N=512..8192`).

**Memory-cleanup protocol (cluster commit `9da9520` / local commit
`7a87679`, "fix(bench): flush memory between full-model repeats").** The
repeat loop now:

1. synchronises, runs `gc.collect()` **plus** `torch.cuda.empty_cache()`
   (through `_clear_device_cache`), and synchronises again **before each
   repeat's baseline capture** — the same protocol as the isolated attention
   benchmark (`src/bench/timing.py::measure`);
2. releases the repeat's `association` / `normalized_association` tensors
   (`= None`) and clears the cache in the repeat's `finally` block —
   previously the cache clear ran only on the failure path, so after a
   successful repeat the previous repeat's `N×N` tensors stayed referenced
   and contaminated the next repeat's baseline and peaks;
3. only then resets the peak-memory statistics and captures the pre-repeat
   baseline, which is now the true model footprint (~52–63 MiB allocated,
   `baseline_mem_mb` mean 62.24 MiB at `N=32768`).

**Environment (from the job log, echoed for both grids):** device `cuda`,
dtype `fp16`, GPU `NVIDIA H100`, torch `2.14.0+cu130`, CUDA `13.0`, seed 42,
repeats 3, warmup 5, greedy linker, `motile_available=false`, checkpoint
SHA-256 `cc36f8a3…dad2b0` (unchanged), git commit
`9da9520cbcbb89c0a6560836bca979dbbe52504b`. Note: the local
`results/full_model_inference_grid{,_b64}_env.json` files were not re-pulled
after this rerun and still carry the superseded job-`4525220` environment;
the job log is the authoritative `4525463` environment record.

**Row/status counts:** `B=1` — 84 raw rows, all `ok`, 28 summary rows.
`B=64` — 60 raw rows (**54 `ok`**, **6 `oom`**), 20 summary rows; all 60
rows `linker_status=skipped_b_gt_1`, so `total_ms = full_forward_ms +
normalize_ms` there. The 6 OOM rows (`relative_pos` and `cached_dist` at
`N=8192`) are recorded, not dropped.

**Memory-quantity correction.** The job-`4522664` run recorded only
`peak_mem_mb`, the **incremental delta** (total peak minus the pre-repeat
allocated baseline). Its identical 27136 MB values for every variant at
`N=32768` are that delta — dominated by the shared `N×N` allocations — and
must **not** be quoted as total memory. Total peak VRAM is
`peak_mem_total_mb` (allocated) / `peak_mem_reserved_mb` (reserved by the
caching allocator, reserved ≥ allocated) and includes the model-weight
baseline. The memory figures plot `peak_mem_reserved_mb_max` with the
y-label `Total peak VRAM (reserved, MB)` against the 94 GB H100 line —
**reserved total, never the delta**.

**`B=1` `N=32768` memory table** (maxima over the 3 `ok` repeats, MiB):

| Variant | total allocated | total reserved | forward reserved | normalize reserved | incremental delta |
|---|---:|---:|---:|---:|---:|
| `relative_pos` | 27710 | 27930 | 23834 | 27930 | 27648 |
| `cached_dist` | 27710 | 36140 | 36140 | 36140 | 27648 |
| `sparse_mask_k16` | 27710 | 28972 | 20780 | 28972 | 27648 |
| `sparse_gather_k16` | 27710 | 29234 | 7730 | 29234 | 27648 |

The total **allocated** peak is now uniform across variants (27710 MiB — the
shared `N×N` allocations dominate); the pre-repeat baseline is the true
model footprint (`baseline_mem_mb` mean 62.24 MiB), and at the smallest grid
point (`B=1` `N=512`) the reserved totals are 82 / 100 / 100 / 126 MiB
(`relative_pos` / `cached_dist` / mask / gather) — no multi-GB residue.
Timing at `N=32768` (total speedup: gather 1.090x, mask 1.076x, cached
1.02x; gather forward 7.570x) keeps the linker-bound finding of the
job-`4522664` record standing.

**`B=64` memory and speedup tables** (`peak_mem_reserved_mb_max` /
`peak_mem_total_mb_max` over the `ok` repeats, MiB; speedups vs
`relative_pos` at the same `(N, B)`):

`N=4096` (all variants `ok`):

| Variant | total reserved | total allocated | total speedup | forward speedup |
|---|---:|---:|---:|---:|
| `relative_pos` | 25378 | 22852 | 1.00x | 1.00x |
| `cached_dist` | 37698 | 26963 | 1.061x | 1.087x |
| `sparse_mask_k16` | 22338 | 13971 | **2.019x** | **3.082x** |
| `sparse_gather_k16` | 16194 | 13139 | **1.356x** | **1.523x** |

`N=8192` (`relative_pos` and `cached_dist` OOM in all 3 repeats; speedups
`n/a` because the baseline OOMed):

| Variant | total reserved | total allocated | total speedup | forward speedup |
|---|---:|---:|---:|---:|
| `relative_pos` | n/a (oom) | n/a (oom) | n/a | n/a |
| `cached_dist` | n/a (oom) | n/a (oom) | n/a | n/a |
| `sparse_mask_k16` | 85638 | 52436 | n/a | n/a |
| `sparse_gather_k16` | 48774 | 36946 | n/a | n/a |

Because the linker is skipped for `B>1`, a `B=64` `total_ms` is not
comparable with the `B=1` one (linker-skipped caveat above). Timing-only
result on synthetic data; no accuracy or tracking-quality claim. Report
integration is DONE under SPEC 0066 R2; see docs/agent-state/specs/0066-inference-scaling-narrative/evidence/R2-INTEGRATION.md and REPRODUCIBILITY.md §1a.
