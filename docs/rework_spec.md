# Rework Spec — Recolor Report Figures

Goal: apply the central scientific color scheme from
`docs/vis_guidelines.md` to **every** figure that appears in
`report/main.pdf`.

## Inputs
- `docs/vis_guidelines.md` — palette assignment, neutrals,
  forbidden colours.
- `docs/figure_provenance.md` — list of scripts that generate
  the 8 figures used by `main.pdf`.
- `report/layout_notes.md` — the 1-to-1 hex replacement map.

## Scripts to modify
1. `curated/generate_report.py` — generates figures 01, 11, 13
   used in `main.pdf` (plus 10 orphans).
2. `results/generate_plots.py` — generates the 4 appendix
   figures used in `main.pdf` (plus 5 orphans).
3. `report/scripts/generate_amdahl_figure.py` — generates
   `15_amdahl_law.pdf`.

## Color changes (one-line summary per script)

### `curated/generate_report.py`
- Replace `COLORS` dict: Flat UI hex → Okabe–Ito hex.
- Replace `sns.set_theme(palette='muted', ...)` with a custom
  Okabe–Ito cycle.
- Replace `cmap='RdYlGn'` (heatmap) with `cmap='viridis'`.
- Keep `sns.set_theme(style='whitegrid')`.
- Keep all functional code, save paths, file names unchanged.

### `results/generate_plots.py`
- Replace `TRACKASTRA_*` constants with Okabe–Ito hex.
- Remove the existing 10-colour `TRACKASTRA_CYCLE`.
- Keep font, dpi, size, save paths unchanged.

### `report/scripts/generate_amdahl_figure.py`
- Replace the per-curve hex codes with Okabe–Ito slots.
- Use `#000000` for theory, `#0072B2` for measured, `#D55E00`
  for ablation, `#999999` for grid.

## Outputs
- Re-rendered PDF + SVG files in `curated/figures/`,
  `results/figures/`, and `report/resources/figures/`.
- Same filenames, same dimensions, same axes — only colours
  change.
- `report/main.pdf` rebuilt via `latexmk main`.

## Acceptance criteria
- [ ] No Flat UI hex (`#1abc9c`, `#2ecc71`, `#3498db`,
      `#9b59b6`, `#e74c3c`, `#e67e22`, `#f39c12`, `#27ae60`)
      appears in any output SVG.
- [ ] No `tab10` hex (`#4878d0`, `#ee854a`, `#956cb4`,
      `#6acc64`, `#d65f5f`, `#8c613c`) appears in any output
      SVG.
- [ ] No `RdYlGn` colormap in the heatmap output
      (`10_speedup_heatmap`); `viridis` only.
- [ ] Only one red, one blue, one green across all figures
      (Okabe–Ito slots).
- [ ] `report/main.pdf` rebuilds without errors and contains
      the recoloured figures.
- [ ] Each modified script still runs from its expected
      workdir and produces a file of identical dimensions.

## Non-goals
- No new figures, no new scripts, no new tests.
- No layout / typography changes; colours only.
- No removal of orphan figures in this commit (separate task).
