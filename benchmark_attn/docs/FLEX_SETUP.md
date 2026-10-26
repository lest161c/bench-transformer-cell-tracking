# FlexAttention setup cost (SPEC 0051 H2)

## Purpose

The main attention sweep warms every method up before timing, so the
FlexAttention CSV reports kernel-only time.  Two expensive one-time steps are
absorbed by that warmup and never appear in the report:

1. `FlexAttentionModule.prepare(coords)` builds the shared block mask with
   `create_block_mask` (`B=1`, `H=1`) and creates the
   `torch.compile(flex_attention, dynamic=False)` wrapper.
2. The first forward call compiles `flex_attention` for the current `(B, N)`
   shape and executes it once.

`scripts/benchmarks/benchmark_flex_setup.py` measures those steps separately
for each `(B, N)` without changing what the steady-state columns mean.

## CSV schema

The script writes these columns in exactly this order:

| Column | Unit | Meaning |
| --- | --- | --- |
| `batch` | – | Batch dimension `B`. |
| `N` | – | Sequence length. |
| `prepare_ms` | ms | Wall time of `module.prepare(coords)`: block-mask construction plus creation of the compiled callable. |
| `first_call_ms` | ms | Wall time of one forward call on the freshly prepared module. Includes `torch.compile` **and one forward execution**. |
| `compile_overhead_ms` | ms | Signed `first_call_ms - steady_ms`. Because the first call also executes the model, a positive value approximates compile cost; a value `<= 0` means the first call was not slower than steady state. |
| `total_setup_ms` | ms | `prepare_ms + max(0.0, compile_overhead_ms)`. Setup excluding the one execution already represented in `steady_ms`. |
| `prepare_peak_mem_mb` | MiB | Incremental peak device memory during `prepare`. |
| `first_call_peak_mem_mb` | MiB | Incremental peak device memory during the first forward call. |
| `steady_ms` | ms | Mean steady-state forward time from `src.bench.timing.measure`. |
| `steady_std_ms` | ms | Standard deviation across timing repeats (0.0 when `repeats == 1`). |
| `steady_mem_mb` | MiB | Incremental peak device memory of one steady-state call. |
| `status` | – | `ok`, `oom`, or `err`. |

Times are formatted with three decimals; memory values with one decimal.  A
failed row leaves every numeric cell empty and records `oom` or `err` in
`status`; the run continues with the next shape and clears the CUDA cache.

## One-time-per-(B, N)-shape semantics

The setup measured here is paid **once per `(B, N)` shape per process**.  The
block mask depends on `N` (the shared single-batch graph), and Inductor
compiles `flex_attention` for a concrete `(B, N)` because the wrapper uses
`dynamic=False`.  A long-running training or inference job therefore pays this
cost at the first call for each new shape, not on every forward pass.
`steady_ms` is the cost after that first call.

## Fresh-Inductor-cache rule

`torch._dynamo.reset()` clears only the in-process Dynamo cache.  It does **not**
clear the on-disk Inductor cache in `TORCHINDUCTOR_CACHE_DIR`.  To measure a
truly cold compile:

- run each batch size in a separate Python process, and
- point that process at a fresh `TORCHINDUCTOR_CACHE_DIR`.

`slurm/run_flex_setup.slurm` follows this rule: B=1 and B=64 run in separate
processes, each with its own `mktemp -d /tmp/flex_inductor_*` cache that is
removed afterwards.  Within a process the script still calls
`torch._dynamo.reset()` before every shape so no in-memory graph is reused.

## `total_setup_ms` definition

```
total_setup_ms = prepare_ms + max(0.0, compile_overhead_ms)
```

The `max(0.0, ...)` clamps the signed compile-overhead estimate at zero when
the first call happened to be as fast as or faster than steady state.  The
formula deliberately excludes one forward execution because that execution is
already represented in `steady_ms`.  The raw signed value remains visible in
`compile_overhead_ms`.

## Caveat: the first call also contains one forward execution

`first_call_ms` is not pure compile time.  It includes compilation **and** one
full forward pass.  That is why `compile_overhead_ms` subtracts `steady_ms`
rather than reporting `first_call_ms` directly, and why `total_setup_ms` uses
the clamped difference.

## How to run

### Local smoke

From `benchmark_attn/`:

```bash
TORCHINDUCTOR_CACHE_DIR=$(mktemp -d) \
  .venv/bin/python scripts/benchmarks/benchmark_flex_setup.py \
  --smoke --device cuda
```

`--smoke` forces `B=2`, `N in {128, 256}`, `--warmup 1 --rep 50 --repeats 1`
and writes `results/flex_setup_smoke.csv`.  It never overwrites the H100 CSV.

### Cluster

After the owner pushes the runner and script:

```bash
cd benchmark_attn
sbatch slurm/run_flex_setup.slurm
```

The runner writes:

- `results/flex_setup_h100_b1.csv`
- `results/flex_setup_h100_b64.csv`

Logs go to `logs/flex_setup_%j.out` and `logs/flex_setup_%j.err`.
