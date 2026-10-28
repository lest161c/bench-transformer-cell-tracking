# STYLE.md: expert-talk and report-derived material

**Status:** binding for `presentation/beamer/` and reusable for any
report-derived talk material.
**Ground truth:** `report-shorten/*.tex` and `report-shorten/figs/*`.
No number, claim, or figure may come from a superseded campaign or an older
deck.

This file captures the style used for the SPEC 0055 insight-first deck. It
extends the report typography rules in `docs/report_typography_rules.md`.

## 1. Audience and framing

- Assume Trackastra / ML lab experts. Do not teach attention, masked SDPA,
  cross-validation, k-nearest neighbours, contrastive learning, or
  self-supervised pretraining from first principles.
- Frame the talk as an audit of what was learned, not as a result catalogue.
  Put the non-obvious findings and the process that produced them first.
- One claim per slide. The title carries the takeaway; the visual carries the
  evidence; the speaker carries the interpretation.
- Name failures, pivots, and open branches explicitly. Do not fill a gap with
  a proxy.

## 2. Slide density

- One compact visual plus one short line is the target.
- If a beat needs a second visual, a second table, or a paragraph, split it
  into another frame or move the detail to a backup.
- Result slides lead with a figure or a compact table. Do not write prose to
  compensate for a missing visual.
- Backup frames may be denser, but must remain readable at Q&A distance and
  must not contradict the main deck.
- A minor or out-of-distribution control does not get a main-deck slide. Put it
  in a backup with the caveat; the main deck carries only results that can bear
  a headline claim.
- Do not present an OOD ablation (for example, zeroing an input the model was
  trained with) as evidence that the input is useful. The proper control is a
  matched trained arm without that input.

## 3. Language and terminology

- Use the report's canonical names: `CachedDistAttention`, `DenseFlash`,
  `SparseGather`, `SparseMask`, `FlexAttention`, `NSA`,
  `RelativePositionalAttention`.
- DynaCLR is a training recipe, not an architecture. The architecture is
  `ScaledCNN`. Write `ScaledCNN (DynaCLR recipe)` on first use; never label a
  row or slide just `DynaCLR`.
- Use the report's terms: cutoff, AOGM, AOGM-A, AOGM-D, TRA, LOCO, frame-CV.
- Use "to our knowledge" for literature claims. Never write a bare "first".
- Prefer "state of the art" over the abbreviation "SOTA".
- Present the current state. Do not mention superseded experiments, old
  versions, or corrected campaigns as history.
- Do not use "essentially", "approximately" without a number, or "=~" as a
  substitute for a bound. State the number and the bound.
- No em dashes as punctuation in slide text. Use a comma pair, a colon,
  parentheses, or a sentence split. En dashes are allowed for ranges
  (`0.992--1.023$\times$`, `$K{=}4$--$64$`).
- Hedging must be explicit: name the seed count, the held-out condition, the
  confidence interval, or the threat to validity.

## 4. Evidence and process

Every experiment, result, or design slide must be able to answer:

- **(a) Why** this experiment or design choice?
- **(b) What exactly was tried?** Architectures, data, splits, seeds, metrics,
  and the controlled variable.
- **(c) What is novel versus prior work?** Be specific and cite.
- **(d) Interpretation and threat to validity.** What the result licenses and
  what it does not.

Keep the full answers in source comments and in the speaker cues. Do not put
four labelled bullets on the slide. The title usually carries (d), the visual
carries (b), and a short badge or line carries (a)/(c).

## 5. Figures, tables, colours

- Reuse the report figures in place. Do not generate new talk figures unless
  the owner asks.
- Use a report figure at most once in the main deck. If two results share a
  figure, use the two-panel figure once and put the second result in a compact
  table or a second panel.
- Do not crop or annotate a report figure unless the crop is recorded and the
  source remains visible.
- Keep the TUD theme, fonts, and logos unchanged.
- Prefer the report's values to a new derived number. If a number is not in
  the report, stop and report the conflict rather than deriving it on the fly.
- Do not place text as separate absolute nodes over an empty rectangle. Put the
  text inside a `TikZ` node with `text width` and let the node size itself;
  fixed-size boxes with absolute text overflow at the first font or line change.

## 6. Timing and deck shape

- Target 28:30 (1,710 s) for a 30-minute slot, leaving 1:30 before Q&A.
- A 30-minute expert talk is not a page-count contest. Cut framing,
  repetition, and claims the room already accepts before cutting evidence.
- Put detailed tables, kernel verification, occupancy, and full decompositions
  in backups.
- `TALK_TIMING.md` is the timing contract. It also carries short speaker
  process cues so the process is spoken, not only shown.

## 7. Verification

Run from `presentation/beamer/`:

```bash
make
pdfinfo main.pdf | grep Pages
grep -c '^!' main.log
grep -i 'overfull' main.log
grep -i 'undefined' main.log
```

Then check:

- the main-frame count and the backup-frame count match
  `docs/agent-state/specs/<spec>/SLIDE_OUTLINE.md`;
- the timing rows sum to the stated total;
- no stale claim or superseded number appears;
- no em dash appears in slide text;
- every figure is from the allowed report set;
- `pdftotext -bbox` reports no word outside the page and no overlapping word
  boxes.

## 8. Bookkeeping

- The contract lives in `docs/agent-state/specs/<NNNN-slug>/SPEC.md`.
- The work breakdown lives in `TASKS.md` beside it.
- The frame-by-frame contract lives in `SLIDE_OUTLINE.md`.
- `docs/agent-state/` is local-only and is not committed.
- Do not edit `report-shorten/` or any benchmark tree to make a slide pass.
