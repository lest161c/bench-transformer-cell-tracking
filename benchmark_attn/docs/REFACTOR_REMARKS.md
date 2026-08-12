# Refactor Rework Remarks

Open rework items raised during the `benchmark_attn` refactor discussion.
Each item is grouped by the file it concerns, with action notes added during implementation.

## `benchmark_attn/README.md`

- [ ] **Script proliferation.** Reduce the number of benchmark scripts; add
      argparse / `--methods` filtering to select configs at runtime instead of
      one script per sweep. (Resolves `benchmark_full.py` vs `benchmark_sparse.py`
      duplication.)
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
│       ├── registry.py                 # METHOD_REGISTRY
│       ├── io.py                       # write_results_csv()
│       └── cli.py                      # add_common_args()
├── scripts/                            # runnable entry points
│   ├── benchmarks/                     # consolidated benchmark scripts
│   ├── analysis/                       # visualization / report scripts
│   └── slurm/                          # SLURM batch scripts
├── tests/                              # validation scripts
├── exploratory/                        # one-off research scripts (unchanged)
├── results/                            # generated outputs
├── docs/                               # SPEC.md, REPRODUCTION.md, sidecar READMEs
├── README.md
├── pyproject.toml
└── config.yaml
```

## Naming convention for attention variants

Replace the versioned names with mechanism-descriptive names:

| Old class | New class | Registry key |
|---|---|---|
| `GatherSparseAttention` (V1, SDPA) | `GatherSparseAttention` (keep) | `gather-sdpa` |
| `GatherSparseAttentionV2` | `GatherSparseFusedAttention` | `gather-fused` |
| `GatherSparseAttentionV3` | `GatherSparseMatmulAttention` | `gather-matmul` |

Other descriptive names to keep: `dense_masked`, `dense_flash`, `mask-knn`,
`knn-relpos`, `nsa`, `minimax`.

## Status

- [x] Rework items documented.
- [ ] Implementation in progress (parallel agents launched).
- [ ] Documentation updated to reflect new structure.
</content>
</invoke>