# Refactor Rework Remarks

Open rework items raised during the `benchmark_attn` refactor discussion.
Each item is grouped by the file it concerns, with action notes added during implementation.

## `benchmark_attn/README.md`

- [ ] **Cosine histogram analysis.** Verify whether `analysis/cosine_histogram.py`
      is still maintained. If deprecated, remove it from the README structure
      listing.
- [ ] **Hassani citation is wrong for one entry.** The README credits Hassani et al.
      for both *Neighborhood Attention* and *spatial reorder*. One of these is
      incorrect — check the paper and correct the attribution.
- [ ] **Citation style.** Rewrite references as proper LaTeX-style citations
      (`\citep{...}` / `\bibliography`). There are more than two sources to cite
      in the attention benchmarks section; gather the full list from the
      paper/notebooks before editing.

## `benchmark_attn/REPRODUCTION.md`

- [ ] **Remove `labbook` references.** `labbook/` is not pushed; remove every
      mention and any sentence that depends on it.
- [ ] **"Important notice" resolution.** The deprecated/unresolved notice must
      either be resolved immediately or deleted.
- [ ] **Reduce verbosity.** Section 1 is hundreds of lines. Decide whether
      detailed reproduction prose is standard practice; if not, restructure
      toward CLI invocations + minimal commentary.
- [ ] **Drop non-scientific hedges.** Phrases like "**This is NOT a real
      measurement**" are quick-fix notes, not scientific prose — replace with
      neutral language or move to a private note.
- [ ] **Drop `STATUS: DONE` lines.** They confuse the reader; the existence of
      the artifact on disk is sufficient.
- [ ] **Reassess CAVEATs.** Each caveat must either (a) be fixed or (b) be
      rephrased as a quantified uncertainty in the measurement methodology.
- [ ] **Remove hardcoded cluster paths from the author's account.** Sections
      3, 4, 5 currently contain paths like `/data/cat/ws/lest161c-...` that
      are specific to one user. Rewrite as generic setup instructions for a
      reader who has no prior access to that cluster.
- [ ] **De-duplicate with README.md.** Material that already lives in
      `README.md` should be referenced, not repeated.
- [ ] **Section 6 (measured results ledger).** Either render the CSVs into a
      separate results-only markdown, or remove the section entirely (the CSVs
      are the source of truth).
- [ ] **Section 8 (known discrepancies).** Triage each item: resolve, delete,
      or convert to a tracked issue.

## Other README files (`README_knn_methods.md`, etc.)

- [ ] **Move to `docs/`.** If the sidecar READMEs add value, relocate them to
      `benchmark_attn/docs/` and reference them from the central `README.md`.
- [ ] **Strip duplicates and stale content.** Each sidecar must be checked for
      overlap with the central README and outdated numbers.

## `benchmark_attn/model_parts.py`

- [ ] **Rename file.** `model_parts` is a trackastra internal name and is not
      appropriate for a standalone benchmark repo. Propose a better name (e.g.
      `attention_modules`) and update every import.
- [ ] **Extract helper functions.** The leading helpers
      (`_init_exponential_bins`, `_init_linear_bins`, `_init_fourier_frequencies`,
      `_rotate_half`, `ATTN_IGNORE_VALUE`) distract from the main attention
      modules. Move them to a separate file and import them.

## Combined directory structure (AGENTS.md standard + harness package)

Adopt the AGENTS.md standard layout and integrate the shared benchmark
harness as a library subpackage:

```
benchmark_attn/
├── src/                                # library / implementation code
│   ├── __init__.py
│   ├── attention_modules.py            # core attention classes (renamed from model_parts.py)
│   ├── positional_encoding.py          # bin init + RoPE helpers (extracted)
│   └── bench/                          # shared benchmark harness
│       ├── __init__.py
│       ├── timing.py                   # measure(), try_bench()
│       ├── data.py                     # make_inputs(), compute_knn_indices()
│       ├── registry.py                 # METHOD_REGISTRY (9 methods)
│       ├── io.py                       # write_results_csv()
│       └── cli.py                      # add_common_args(), parse_int_list()
├── scripts/                            # runnable entry points
│   └── benchmarks/                     # consolidated benchmark_sweep.py
├── docs/                               # REPRODUCTION.md, sidecar READMEs, REFACTOR_REMARKS.md
├── exploratory/                        # one-off research scripts (unchanged)
└── results/                            # generated artifacts
```

## Naming convention for attention variants

Replace the versioned names with mechanism-descriptive names:

| Old class | New class | Registry key |
|---|---|---|
| `GatherSparseAttention` (V1, SDPA) | `GatherSparseAttention` (keep) | `gather_sdpa` |
| `GatherSparseAttentionV2` | `GatherSparseFusedAttention` | `gather_fused` |
| `GatherSparseAttentionV3` | `GatherSparseMatmulAttention` | `gather_matmul` |

Other descriptive names to keep: `dense_masked`, `dense_flash`, `mask_knn`,
`knn_relpos`, `nsa`, `minimax`.

## Completed work

- [x] **Extract shared harness.** `src/bench/` package with `timing.py`
      (`measure`, `try_bench`), `data.py` (`make_inputs`,
      `compute_knn_indices`), `registry.py` (`METHOD_REGISTRY`,
      `resolve_class`, `list_methods`), `io.py` (`write_results_csv`),
      `cli.py` (`add_common_args`, `parse_int_list`). Eliminates 5 copies
      of `measure()` and 3 copies of `knn_indices()`.
- [x] **Rename `model_parts.py` → `src/attention_modules.py`.** Updated all
      imports. Backward-compatible alias kept for external code.
- [x] **Extract helper functions.** `src/positional_encoding.py` contains
      `ATTN_IGNORE_VALUE`, `_init_exponential_bins`, `_init_linear_bins`,
      `_init_fourier_frequencies`, `_rotate_half`, `RotaryPositionalEncoding`.
- [x] **Rename V2/V3 classes.** `GatherSparseAttentionV2` →
      `GatherSparseFusedAttention`, `GatherSparseAttentionV3` →
      `GatherSparseMatmulAttention`. Backward-compatible aliases kept.
- [x] **Consolidate benchmark scripts.** `scripts/benchmarks/benchmark_sweep.py`
      replaces `benchmark_sparse.py`, `benchmark_full.py`,
      `benchmark_gather_v3.py`, `benchmark_small_n.py`. Single CLI with
      `--methods`, `--Ns`, `--Ks`, `--layers`, `--mode`, `--dist-mode`.
- [x] **Fix `.gitignore`.** Anchored `scripts/` to `/scripts/` so
      `benchmark_attn/scripts/` is trackable.
- [x] **Update method naming in documentation.** All `.md` files now use the
      new registry keys (`gather_sdpa`, `gather_fused`, `gather_matmul`,
      `mask_knn`, `knn_relpos`, `nsa`, `minimax`, `dense_flash`,
      `dense_masked`) instead of `sparse`, `sparse_v2`, `V1/V2/V3`, etc.
- [x] **Update script references in documentation.** `README.md` and
      `REPRODUCTION.md` now reference `scripts/benchmarks/benchmark_sweep.py`
      instead of the four old scripts.
- [x] **Remove `labbook` references.** All `labbook/` mentions removed from
      `REPRODUCTION.md`.
- [x] **Remove `STATUS: DONE`/`PENDING` lines.** All status lines removed from
      `REPRODUCTION.md`.
- [x] **Remove hardcoded cluster paths.** All `/data/cat/ws/lest161c-...` paths
      removed from `REPRODUCTION.md`.
- [x] **Remove non-scientific hedges.** All `**This is NOT a real measurement**`
      and similar phrases removed from `REPRODUCTION.md`.
- [x] **Fix Hassani citation.** Corrected from "*Neighborhood Attention*, ECCV
      2024" to "*Neighborhood Attention Transformer*, CVPR 2023" and
      "*Faster Neighborhood Attention*, NeurIPS 2024".
- [x] **Rewrite references as LaTeX-style citations.** All references now use
      `\citep{key}` format with full bibliographic details.
- [x] **Move sidecar READMEs to `docs/`.** All four sidecar READMEs relocated
      to `benchmark_attn/docs/` and referenced from central `README.md`.
- [x] **Remove deprecated scripts.** Deleted `benchmark_blockwise_norm.py`,
      `benchmark_ffn_attention_ratio.py`, `benchmark_flash_with_cutoff.py`,
      `benchmark_knn_methods.py`.
- [x] **Merge profiler scripts.** `benchmark_profiler.py` + `profile_gather_sparse.py`
      merged into single `benchmark_profiler.py` with `--mode general|gather`.
- [x] **Move TRA/AOGM and collapse scripts to `exploratory/`.** These are
      reporting scripts, not attention benchmarks.
- [x] **Remove refactor-process notes from docstrings.** Removed all references
      to "extracted from", "former model_parts.py", "merges two previously
      separate tools", etc.
- [x] **Rename cryptic vars in `validate_equivalence.py`.** All single-letter
      and abbreviated variable names replaced with descriptive names.
- [x] **Unify registry key separators.** All method keys now use underscores
      consistently: `gather_sdpa`, `gather_fused`, `gather_matmul`,
      `mask_knn`, `knn_relpos`, `nsa`, `minimax`, `dense_flash`,
      `dense_masked`.

## Remaining work (from user's rework list)

- [ ] **`README.md`**: Update structure listing to reflect new `src/`/`scripts/`/`docs/` layout. Fix Hassani citation (wrong for one entry). Rewrite references as proper LaTeX-style citations. Verify whether `cosine_histogram.py` is deprecated and remove from README if so.
- [ ] **`REPRODUCTION.md`**: Remove `labbook` references (not pushed). Resolve or delete the "Important notice". Reduce verbosity — restructure toward CLI invocation + minimal commentary. Drop non-scientific hedges like "**This is NOT a real measurement**". Drop `STATUS: DONE` lines. Reassess CAVEATs — each must be fixed or rephrased as quantified uncertainty. Remove hardcoded cluster paths from author's account (sections 3, 4, 5) — rewrite as generic setup instructions. De-duplicate with `README.md`. Section 6 (results ledger): either render CSVs into a separate results-only markdown, or remove. Section 8 (discrepancies): triage each item — resolve, delete, or convert to tracked issue.
- [ ] **Other README files** (`README_knn_methods.md`, etc.): Move to `docs/`. Strip duplicates and stale content. Reference from central `README.md`.
- [ ] **`model_parts.py` (now `attention_modules.py`)**: Rename complete. Helper extraction complete. Verify no remaining references to old `model_parts.py` name in scripts or documentation.
- [ ] **Directory reorganization**: Move existing benchmark scripts to `scripts/benchmarks/`. Move analysis scripts to `scripts/analysis/`. Move SLURM scripts to `scripts/slurm/`. Move validation scripts to `tests/`. Update all import paths.
