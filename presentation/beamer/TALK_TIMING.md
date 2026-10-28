# TALK_TIMING: SPEC 0055

Speaker timing script for the 39-frame main deck (49-page deck with 9 untimed
backups and 1 reference page). Seconds and frame order follow
`SLIDE_OUTLINE.md`. Main-deck total: **28:30 (1,710 s)**. The 30-minute slot
leaves **1:30** as buffer before Q&A; B1–B9 are untimed Q&A material.

## Main deck

**Opening (frames 1–3)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 1 | Title: Efficient sparse attention and self-supervised learned features for transformer-based cell tracking | 10 | 0:10 |
| 2 | What we set out to learn | 55 | 1:05 |
| 3 | Five things we did not expect | 60 | 2:05 |
| | **Opening subtotal** | **125** | **2:05** |

**§4.1 Efficient sparse attention (frames 4–15)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 4 | §4.1: What does the mask actually cost? | 5 | 2:10 |
| 5 | The proposal's premise was only half right | 70 | 3:20 |
| 6 | What we benchmarked and why | 35 | 3:55 |
| 7 | Speed at production batch: DenseFlash to 16k; SparseGather only at 32k | 55 | 4:50 |
| 8 | Memory sets the cliff, not just time | 45 | 5:35 |
| 9 | Where the mask cost lives: construction, not the kernel | 55 | 6:30 |
| 10 | If you keep the cutoff, use SparseMask | 50 | 7:20 |
| 11 | Removing the cutoff is not accuracy-neutral | 75 | 8:35 |
| 12 | SparseMask inherits SparseGather accuracy | 30 | 9:05 |
| 13 | Token reordering does not help | 25 | 9:30 |
| 14 | Attention is not the training bottleneck | 80 | 10:50 |
| 15 | NSA is rejected at these N | 30 | 11:20 |
| | **§4.1 subtotal** | **555** | **9:15** |

**§4.2 SSL association pretraining (frames 16–23)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 16 | §4.2: Can warps manufacture association labels? | 5 | 11:25 |
| 17 | Why the warp pretext is not just another augmentation | 40 | 12:05 |
| 18 | What we tried: trpL held out, 5 labeled pairs, compute-matched control | 55 | 13:00 |
| 19 | Result: all three arms 0.500 | 30 | 13:30 |
| 20 | Why it collapsed: ~1/N positives and an all-negative optimum | 65 | 14:35 |
| 21 | Coordinate solvability: 0.982 of warp labels recoverable from coordinates | 55 | 15:30 |
| 22 | What we did instead: DynaCLR encoder 1.47M, frozen | 45 | 16:15 |
| 23 | Deprioritized, not disproven; two branches remain open | 45 | 17:00 |
| | **§4.2 subtotal** | **340** | **5:40** |

**§4.3 / §4.4 Features (frames 24–33)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 24 | §4.3/§4.4: What was actually trained? What do features add? | 5 | 17:05 |
| 25 | What was actually trained: frozen injection, not end-to-end fine-tuning | 40 | 17:45 |
| 26 | KNN probe: K=8 covers 99.3%, candidates by centroid, shared MLP | 50 | 18:35 |
| 27 | Position and geometry dominate: DynaCLR 0.826, nearest-centroid 0.811 | 60 | 19:35 |
| 28 | What the probe means: features add little over geometry | 40 | 20:15 |
| 29 | Injection protocol: projection head + concat; encoder frozen | 35 | 20:50 |
| 30 | Small encoder is competitive, not state of the art | 60 | 21:50 |
| 31 | The head uses features, but not their identity | 60 | 22:50 |
| 32 | Raw embedding geometry near-uniform: cosine-only fails (~88332) | 30 | 23:20 |
| 33 | §4.4 interpretation: small encoder competitive; position is the signal | 50 | 24:10 |
| | **§4.3/§4.4 subtotal** | **430** | **7:10** |

**Synthesis and close (frames 34–39)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 34 | What changed, and how we caught it | 65 | 25:15 |
| 35 | What we did not do, and why | 45 | 26:00 |
| 36 | Recommendations for the Trackastra team | 55 | 26:55 |
| 37 | What I would do next | 40 | 27:35 |
| 38 | What these results do not license | 45 | 28:20 |
| 39 | Thank you | 10 | 28:30 |
| | **Synthesis subtotal** | **260** | **4:20** |

**Main-deck total: 28:30 (1,710 s).**

Section totals: Opening 125 s (2:05); §4.1 555 s (9:15); §4.2 340 s (5:40);
§4.3/§4.4 430 s (7:10); Synthesis 260 s (4:20).

## Backup slides (untimed)

| # | Frame title | Timing |
|---|---|---|
| B1 | Full speed detail | Q&A |
| B2 | Full mask decomposition | Q&A |
| B3 | Cutoff occupancy | Q&A |
| B4 | Kernel verification and equivalence | Q&A |
| B5 | K-sweep and K=16 | Q&A |
| B6 | Probe table | Q&A |
| B7 | Injection table | Q&A |
| B8 | Pretext and coordinate solvability | Q&A |
| B9 | Amdahl and complexity | Q&A |

Backups are after frame 39 and are not counted in the 28:30 total.

## Speaker process cues

These are the sentences that carry the process, not the result. Keep them short;
the slide already shows the number.

- **Frame 2:** "We fixed the test before the result: kernel-verified dispatch,
  production batch, held-out condition, pre-registered rule, coordinates-only
  candidates."
- **Frame 5:** "The premise was testable, so we tested it. Forced-backend arms
  and per-row kernel names showed masked SDPA reaches fused cuDNN on H100 and
  torch 2.14; the caveat is the dispatchable head dimension."
- **Frame 9:** "The cost is caller-side: 4.41 ms to construct the mask at
  N=8192, 14x the kernel call, plus 1032 MB when a bool mask is materialized to
  a float bias."
- **Frame 11:** "This is the one result that changes the default: removing the
  cutoff is worse in 12/12 fold-seed cells. It is a paired, one-signed effect
  with a two-seed bound, not a significance claim."
- **Frame 14:** "The Amdahl trap: a 1.75x kernel speedup predicts 1.044
  end-to-end because attention is 10%. The norm chain is 38.7% and host-side:
  240 synchronizations per step, GPU kernels only 1.6 to 2.1%."
- **Frame 18:** "The low-label rule was fixed before the run. We did not tune
  it after seeing 0.500."
- **Frame 20:** "We ruled out a bug before accepting the null: loss at the
  entropy floor, targets match the true correspondence, the same code learns on
  another pool."
- **Frame 26:** "K=8 covers 99.3% of true matches and the candidates come from
  coordinates only, so the feature cannot choose its own shortlist."
- **Frame 30:** "The honest comparator is Original Trackastra at 11.9, not only
  SAM2.1. We are competitive among frozen-feature extractors, not state of the art, and the
  remaining DeepCell gap is detection."
- **Frame 31:** "Zeroing breaks the head, so features are required. Shuffling
  identity costs only 0.94 AOGM, so the head is not using learned identity."
- **Frame 34:** "These are the beliefs that changed. Each row names the
  measurement that changed it."
- **Frame 35:** "These are the things we did not do. The port was optional, the
  reorder was discarded, NSA loses at small N, and the distortion families
  remain open."
- **Frame 38:** "This is the boundary: no state-of-the-art claim, no large-N accuracy, no
  association-head pretraining, no low-label gain, no significance test."
