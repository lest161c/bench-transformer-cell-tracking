# Visualization Guidelines

Single source of truth for all figures in this report.
Derived from the audit in `report/layout_notes.md`.

## 1. One palette per role

| Role                       | Palette       | Source                                  |
| -------------------------- | ------------- | --------------------------------------- |
| Categorical / line / bar   | **Okabe–Ito** | matplotlib `okabeIto`, *Nature Methods* 2011 |
| Sequential heatmap         | **`viridis`** | matplotlib default, perceptually uniform |
| Diverging heatmap (if needed) | **`RdBu`** | ColorBrewer, cblind-safe                |

Colorblind-safe and print-safe are mandatory, not optional.

### Okabe–Ito slots — fixed assignment

```
#000000  theory / baseline curve
#E69F00  variant A  (primary method, "ours")
#56B4E9  variant B
#009E73  variant C
#F0E442  variant D
#0072B2  secondary metric
#D55E00  ablation / loss
#CC79A7  attention / map overlay
```

Same slot → same colour across **every** figure in the report.
No re-mapping per figure.

### Reserved neutrals (do not change)

```
#262626  text / axes labels
#cccccc  gridlines
#ffffff  background
#999999  axes (if a 2nd grey is unavoidable)
```

## 2. What is forbidden

- Flat UI / web-design palettes (`#3498db`, `#e74c3c`, `#1abc9c`,
  …) — replaced by Okabe–Ito.
- Matplotlib `tab10` default — replaced by Okabe–Ito.
- Red-green diverging maps (`RdYlGn`) for heatmaps — replaced by
  `viridis`.
- Pure `#FF0000` or `#00FF00` for line plots.
- More than 2 different greys for the same axes role.
- 3D effects, drop shadows, gradients on bars.
- Two-tone hatching to compensate for missing colours.

## 3. Format and resolution

| Output                      | Rule                                       |
| --------------------------- | ------------------------------------------ |
| Source                      | one Python / matplotlib script per figure  |
| Exported files              | PDF **and** SVG from the same script        |
| SVG only if heatmap/colormap| both PDF (for `\includegraphics`) and SVG |
| Raster fallback             | ≥300 dpi                                    |
| Aspect ratio                | text-width by default; `\textwidth` for full |
| Font in figure              | match body font (Latin Modern / sans)        |

## 4. Typography in figures

- One font family, same as body text.
- Title above plot, axis labels horizontal, units in label.
- Legend text same size as axis ticks.
- No boldface for emphasis; use colour slot instead.
- Numerics: same precision as the table they replace.

## 5. Axes and gridlines

- Gridlines in `#cccccc`, ≤0.5 pt, on by default.
- Axis line in `#262626`, 0.8 pt.
- Tick marks outside, same colour as axis line.
- Y-axis starts at 0 for bar charts and loss curves.
- No scientific notation unless range ≥10⁴.
- One scale per axis (no broken axes).

## 6. Legends

- One legend per figure, never one legend per subplot.
- Position: top or right of the axes, outside the data area.
- Order in legend = order of plotting.
- Frame: optional, never filled, line in `#999999`.
- Label text matches the axis-label style.

## 7. Reproducibility

- Every figure must be regenerable from one script.
- Script lives under `scripts/figures/` and is named after the
  figure (`figure_01_speed_vs_n.py`).
- Script writes both PDF and SVG to `resources/figures/`.
- Script exposes a fixed seed parameter (default seed = 0).
- Script prints the file paths it wrote, so the LaTeX build can
  pick them up.

## 8. Pre-commit checklist for a new figure

- [ ] Uses Okabe–Ito slots, not Flat UI or tab10.
- [ ] Heatmap uses `viridis` (or `RdBu` if diverging).
- [ ] Gridlines in `#cccccc`, text in `#262626`.
- [ ] Same font as body text.
- [ ] No 3D, no shadow, no gradient.
- [ ] PDF **and** SVG exported from the same script.
- [ ] Script in `scripts/figures/`, has `--seed` flag.
- [ ] Legend matches plotted order.
- [ ] ≥300 dpi for any raster fallback.
- [ ] Description in commit message names the figure file.
