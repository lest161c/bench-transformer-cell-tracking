# Full-model inference grid (SPEC 0062)

**Script:** `benchmark_attn/scripts/benchmarks/benchmark_full_model_inference_grid.py`
**Runner:** `benchmark_attn/slurm/run_full_model_inference_grid.slurm`
**Spec:** `docs/agent-state/specs/0062-full-model-inference-grid/SPEC.md`
**Tickets:** F1–F6; this document is the F5 protocol and is extended with the
job record in F6.

## Purpose and scope

This benchmark measures the **complete Trackastra inference chain** with one
of four attention implementations swapped into every encoder and decoder
layer. It exists because isolated attention timings do not show how much of
the full forward pass an attention variant actually controls once embedding,
the encoder, the decoder, the association head, output normalisation, and the
linker are included.

**This is a timing-only experiment.** It makes no accuracy, tracking-quality,
or out-of-distribution-generalisation claim. The checkpoint was trained at
roughly `N≈50–200` cells per frame; the large-`N` grid is a synthetic stress
test of the inference path, not a quality evaluation. No report or
claim-bearing source is edited by F1–F6.

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

Each variant gets a fresh model instance loaded from the checkpoint so a swap
cannot contaminate the next variant. Attention weights are copied by exact
key (`q_pro.*`, `k_pro.*`, `v_pro.*`, `proj.*`, `rot_pos_enc.*`) with a strict
load for the default k16 rope checkpoint. `DenseFlash`, `FlexAttention`, and
other attention classes are out of scope.

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
   - synchronise, clear the CUDA cache, reset peak-memory statistics;
   - use CUDA events on GPU and `time.perf_counter` at the same boundaries on
     CPU;
   - run `model.transformer(...)` and `normalize_output(...)` under
     `torch.no_grad()`;
   - read incremental peak memory as `max_memory_allocated()` minus the
     pre-repeat allocated baseline (`0.0` on CPU);
   - time the CPU linker separately and write one raw row.
4. On OOM or any exception, record a row with status `oom` or
   `err: <message>`, clear caches, and continue with the next configuration.

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
materialised. `build_graph` uses `use_distance=False`, `max_distance=256`,
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
`(variant, N, B, repeat_idx)`; the primary grid has
`4 × 7 × 1 × 3 = 84` planned rows, including `oom`/`err` rows.

```
variant,N,B,repeat_idx,status,checkpoint,dtype,seed,knn_neighbors,
coord_layout,linker_mode,linker_status,
full_forward_ms,embed_ms,encoder_ms,encoder_attn_ms,decoder_ms,
decoder_attn_ms,head_logits_ms,head_x_ms,head_y_ms,einsum_ms,
embed_proj_ms,normalize_ms,linker_build_graph_ms,linker_greedy_ms,
linker_ms,total_ms,peak_mem_mb,dist_2d_standalone_ms,
forward_closure_error_ms,error
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
| `peak_mem_mb` | Incremental CUDA peak for one repeat; `0.0` on CPU. |
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
overwrites the primary CSV):

```bash
cd benchmark_attn
.venv/bin/python scripts/benchmarks/benchmark_full_model_inference_grid.py \
  --checkpoint ../checkpoints/k16 \
  --smoke
```

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
`$REPO_ROOT/slurm/load_cluster_env.sh`, runs the existing
`benchmark_attn/.venv`, uses 1 GPU / 8 CPUs / 90 GB, and skips the primary
grid when `results/full_model_inference_grid.csv` already exists. The optional
`B=8/16` up-to-`N=8192` invocation is commented out in the runner and writes
separate `*_batch.*` files if it is ever enabled.

Summary (F6, from `benchmark_attn/`):

```bash
cd benchmark_attn
.venv/bin/python analysis/summarize_full_model_inference_grid.py
```

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
- **Shared `N×N` costs.** `dist_2d`, the logits `einsum`, and
  `normalize_output` cap end-to-end Gather speedups and are reported explicitly
  rather than hidden.
- **F4/F6 ownership.** F4 owns the linker timing implementation; F6 owns the
  H100 execution, the summary script/CSV/PNG, and the `N=32768` table. F7
  report integration remains blocked on owner review of F6.
