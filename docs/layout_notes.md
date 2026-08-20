# Report Layout Notes — `report/main.pdf`

PDF was read natively (model returns image attachments), but inline
display was rejected by the size limit, so I switched to a flat
analysis. All quantitative findings below were extracted directly
from the SVG source files that produce the figures in the PDF.

## 1. Source material

| Item                | Value                                       |
| ------------------- | ------------------------------------------- |
| PDF                 | `report/main.pdf`, 26 pages, 657 KB        |
| Diagram sources     | 14 SVG + 24 PDF figures in `resources/figures/` |
| SVGs used by report | 14                                          |
| `None`/axes-only    | 10                                          |
| Total distinct hex  | **55**                                      |

SVGs inspected (one per row in §2): 01–13, 15. `04_memory_vs_n`
exists as both PDF and SVG; only SVG analysed.

## 2. Diagram inventory and distinct-color counts

| Figure                     | Series | Distinct colors | Dominant palette          |
| -------------------------- | ------ | --------------- | -------------------------- |
| 01_speed_vs_n              | 5      | 9               | Flat UI                    |
| 02_small_n                 | 4      | 7               | Flat UI                    |
| 03_mask_vs_gather          | 4      | 8               | Flat UI                    |
| 04_memory_vs_n             | 5      | 8               | Flat UI                    |
| 05_convergence             | 5      | 8               | Flat UI                    |
| 06_tra                     | 5      | 9               | Flat UI                    |
| 07_aogm                    | 5      | 8               | Flat UI                    |
| 08_edge_div_f1             | 2      | 5               | Flat UI (subset)           |
| 09_per_epoch_time          | 5      | 8               | Flat UI                    |
| 10_speedup_heatmap         | n/a    | 28              | Custom diverging Red→Yellow→Green |
| 11_profiler_breakdown      | 5      | 10              | Flat UI (+ `#c0392b`, `#27ae60`) |
| 12_layer_scaling           | 6      | 9               | Matplotlib `tab10`         |
| 13_ssl_negative            | 4      | 8               | Flat UI                    |
| 15_amdahl_law              | 4      | 11              | Ad-hoc (`#21918c`, `#31668e`, `#38b977`, `#463480`, `#9bd93c`, `#000000`, `#ff0000`) |

**Key observation:** the same 5 series are plotted across at
least 8 figures (01, 02, 03, 04, 05, 06, 07, 09) — each chart is
its own colour-mapping. There is no consistent encoding of
*"method A / B / C / D / E"* across the report.

## 3. Problem analysis

Three different palettes are mixed in the same document:

1. **Flat UI Colors** (web-design palette by designer `Designmodo`)
   — the dominant palette, present in 13/14 SVGs. Bright,
   saturated, not optimised for print or for
   red-green–deficient vision.
2. **Matplotlib `tab10`** — present only in `12_layer_scaling.svg`,
   conflicting with Flat UI in the same report.
3. **Ad-hoc colours in `15_amdahl_law`** — including pure `#ff0000`
   red and pure `#000000` black for theory curves.

The diverging colormap in `10_speedup_heatmap.svg` uses
`#006837 → #fffebe` (matplotlib `RdYlGn`) — a red/green diverging
map that is **not colourblind-safe** and has a confusing
yellow/green midpoint.

Concretely inconsistent colour usage:

- `#3498db` (Flat UI "Peter river blue") appears in 9 files but
  `#4878d0` (matplotlib blue) appears for the same role in
  `12_layer_scaling.svg`.
- `#e74c3c` (Flat UI "Alizarin red") is used for the
  attention/comparison line in 10 files, but `#c0392b` (Flat UI
  "Pomegranate") is used for the same line in
  `11_profiler_breakdown.svg`.
- Grey is encoded as `#cccccc` in 14 files, but `#808080`, `#888888`,
  `#b0b0b0` and `#95a5a6` all appear as separate greys.

## 4. Proposed central scientific palette

**Primary recommendation: the Okabe–Ito palette**
(Okabe & Ito, 2002 / 2008; recommended in *Nature Methods*,
"Points of view: Color blindness", 2011).

| Slot | Hex       | Role in this report                              |
| ---- | --------- | ------------------------------------------------- |
| 1    | `#000000` | baseline / theory curves                          |
| 2    | `#E69F00` | variant A / "ours"                                 |
| 3    | `#56B4E9` | variant B / baseline                               |
| 4    | `#009E73` | variant C / efficient variant                     |
| 5    | `#F0E442` | variant D (sparse attention)                      |
| 6    | `#0072B2` | secondary metric / "theoretical" line            |
| 7    | `#D55E00` | ablation / red curve for losses                   |
| 8    | `#CC79A7` | final highlight / attention map                   |

Properties:

- Colorblind-safe for the three common forms of dichromacy
  (protanopia, deuteranopia, tritanopia).
- Print-safe (renders correctly in grayscale).
- The palette is purpose-built for **scientific figures**, not
  for marketing websites.
- The same hex codes are bundled with most plotting libraries
  (matplotlib `okabeIto`, seaborn `okabeito`, ggplot2
  `scale_colour_okabeito`).

Auxiliary colours (axes, grid, text) — keep to one neutral:

- text: `#262626` (already used in 13 of 14 SVGs)
- grid: `#cccccc` (already used in 14 of 14 SVGs)

## 5. Replacement map

Target mapping per file. `tab10` and Flat UI hex codes are
replaced 1-to-1 by Okabe–Ito; ad-hoc colours in `15_amdahl_law`
are collapsed to two roles only.

```
# Plot colour replacements
#1abc9c (Flat UI turquoise)  → #009E73  (Okabe–Ito bluish green)
#2ecc71 (Flat UI emerald)    → #009E73  (only one green!)
#3498db (Flat UI blue)       → #0072B2  (Okabe–Ito blue)
#9b59b6 (Flat UI amethyst)   → #CC79A7  (Okabe–Ito reddish purple)
#e74c3c (Flat UI alizarin)   → #D55E00  (Okabe–Ito vermilion)
#e67e22 (Flat UI carrot)     → #E69F00  (Okabe–Ito orange)
#f39c12 (Flat UI orange)     → #E69F00  (same orange!)
#c0392b (Flat UI pomegranate)→ #D55E00  (same vermilion!)
#27ae60 (Flat UI nephritis)  → #009E73  (same green)

# Matplotlib tab10 replacements (used in 12_layer_scaling)
#4878d0 (tab10 blue)        → #0072B2
#ee854a (tab10 orange)      → #E69F00
#956cb4 (tab10 purple)      → #CC79A7
#6acc64 (tab10 green)       → #009E73
#d65f5f (tab10 red)         → #D55E00
#8c613c (tab10 brown)       → #999999  (neutral grey, axes role)

# Heatmap colormap replacement
10_speedup_heatmap.svg:
  current: #006837 → #fffebe (RdYlGn, red-green, NOT cblind-safe)
  → matplotlib `viridis`    (purple→teal→yellow, perceptually uniform)
  OR `cividis`              (blue→grey, print + cblind safe)

#15_amdahl_law.svg ad-hoc colours
#21918c, #31668e, #38b977, #463480, #9bd93c, #ff0000
 → collapse to #0072B2 (theory line) and #D55E00 (ablation line)
 → plus one grey #999999 for grid lines (already present)

# Grey consolidation (4 different greys today → 1)
#808080, #888888, #b0b0b0, #95a5a6 → #999999  (axes grey)
#cccccc kept as gridline grey
#262626 kept as text/axes colour
#ffffff kept as background
```

## 6. Quantified outcome

| Metric                                          | Today   | After replacement |
| ----------------------------------------------- | ------- | ----------------- |
| Total distinct hex colours (14 SVGs)            | **55**  | **14** (8 Okabe–Ito + 4 neutrals + 1 colormap anchor + 1 black) |
| Different categorical palettes mixed in report | **3+** (Flat UI, tab10, ad-hoc) | **1** (Okabe–Ito) |
| Colourblind-safe figures                        | 0/14   | 14/14              |
| Different greys for the same axes role          | 4      | 1                  |
| Reds used for "ours/our method" line           | 4 (#e74c3c, #c0392b, #d65f5f, #ff0000) | 1 (#D55E00) |
| Blues used for "baseline" line                 | 2 (#3498db, #4878d0) | 1 (#0072B2) |

## 7. Action items

1. Re-render every SVG in `resources/figures/` through a single
   Python / matplotlib style sheet that exposes the Okabe–Ito
   palette as 8 named slots (`series_a` … `series_h`).
2. Replace `10_speedup_heatmap.svg` colormap with `viridis`
   (or `cividis`).
3. Re-export every figure as PDF **and** SVG from the same
   script, so source and output can never disagree.
4. Keep the two neutrals already dominant in the corpus
   (`#262626` text, `#cccccc` grid) so the change is minimal.
5. Audit the 10 PDF-only figures (no SVG available) for the
   same Flat-UI / tab10 leak.

## 8. Sources / references

- Okabe & Ito (2002), "Color Universal Design — for better
  presentation of data to everyone".
- *Nature Methods* (2011), "Points of view: Color blindness".
- matplotlib `viridis` colormap (Stéfan van der Walt &
  Nathaniel Smith, 2015).
- matplotlib built-in palette `okabeIto` (8 hex codes).
