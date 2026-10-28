# TALK_TIMING: SPEC 0055

Speaker timing script for the 37-frame main deck (48-page deck with 10 untimed
backups and 1 reference page). Seconds and frame order follow
`SLIDE_OUTLINE.md`. Main-deck total: **28:30 (1,710 s)**. The 30-minute slot
leaves **1:30** as buffer before Q&A; B1–B10 are untimed Q&A material.

## Main deck

**Opening (frames 1–3)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 1 | Title: Efficient sparse attention and self-supervised learned features for transformer-based cell tracking | 10 | 0:10 |
| 2 | What we set out to learn | 55 | 1:05 |
| 3 | Five things we did not expect | 75 | 2:20 |
| | **Opening subtotal** | **140** | **2:20** |

**§4.1 Efficient sparse attention (frames 4–15)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 4 | §4.1: What does the mask actually cost? | 5 | 2:25 |
| 5 | The proposal's premise was only half right | 70 | 3:35 |
| 6 | What we benchmarked and why | 35 | 4:10 |
| 7 | Speed at production batch: DenseFlash to 16k; SparseGather only at 32k | 65 | 5:15 |
| 8 | Memory sets the cliff, not just time | 45 | 6:00 |
| 9 | Where the mask cost lives: construction, not the kernel | 55 | 6:55 |
| 10 | If you keep the cutoff, use SparseMask | 50 | 7:45 |
| 11 | Removing the cutoff is not accuracy-neutral | 90 | 9:15 |
| 12 | SparseMask inherits SparseGather accuracy | 30 | 9:45 |
| 13 | Token reordering does not help | 25 | 10:10 |
| 14 | Attention is not the training bottleneck | 95 | 11:45 |
| 15 | NSA is rejected at these N | 30 | 12:15 |
| | **§4.1 subtotal** | **595** | **9:55** |

**§4.2 SSL association pretraining (frames 16–23)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 16 | §4.2: Can warps manufacture association labels? | 5 | 12:20 |
| 17 | Why the warp pretext is not just another augmentation | 40 | 13:00 |
| 18 | What we tried: trpL held out, 5 labeled pairs, compute-matched control | 55 | 13:55 |
| 19 | Result: all three arms 0.500 | 30 | 14:25 |
| 20 | Why it collapsed: ~1/N positives and an all-negative optimum | 65 | 15:30 |
| 21 | Coordinate solvability: 0.982 of warp labels recoverable from coordinates | 55 | 16:25 |
| 22 | What we did instead: ScaledCNN 1.47M, frozen | 45 | 17:10 |
| 23 | Deprioritized, not disproven; two branches remain open | 45 | 17:55 |
| | **§4.2 subtotal** | **340** | **5:40** |

**§4.3 / §4.4 Features (frames 24–31)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 24 | §4.3/§4.4: What was actually trained? What do features add? | 5 | 18:00 |
| 25 | What was actually trained: frozen injection, not end-to-end fine-tuning | 40 | 18:40 |
| 26 | KNN probe: K=8 covers 99.3%, candidates by centroid, shared MLP | 50 | 19:30 |
| 27 | Position and geometry dominate: ScaledCNN 0.826, nearest-centroid 0.811 | 75 | 20:45 |
| 28 | What the probe means: features add little over geometry | 40 | 21:25 |
| 29 | Injection protocol: projection head + concat; encoder frozen | 35 | 22:00 |
| 30 | Small encoder is competitive, not state of the art | 70 | 23:10 |
| 31 | §4.4 interpretation: small encoder competitive; position is the signal | 50 | 24:00 |
| | **§4.3/§4.4 subtotal** | **365** | **6:05** |

**Synthesis and close (frames 32–37)**

| # | Frame title | Seconds | Cumulative |
|---:|---|---:|---|
| 32 | What changed, and how we caught it | 75 | 25:15 |
| 33 | What we did not do, and why | 45 | 26:00 |
| 34 | Recommendations for the Trackastra team | 55 | 26:55 |
| 35 | What I would do next | 40 | 27:35 |
| 36 | What these results do not license | 45 | 28:20 |
| 37 | Thank you | 10 | 28:30 |
| | **Synthesis subtotal** | **270** | **4:30** |

**Main-deck total: 28:30 (1,710 s).**

Section totals: Opening 140 s (2:20); §4.1 595 s (9:55); §4.2 340 s (5:40);
§4.3/§4.4 365 s (6:05); Synthesis 270 s (4:30).

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
| B10 | Feature-use and cosine controls (report prose only) | Q&A |

Backups are after frame 37 and are not counted in the 28:30 total.

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
  SAM2.1. We are competitive among frozen-feature extractors, not state of the
  art, and the remaining DeepCell gap is detection."
- **Frame 32:** "These are the beliefs that changed. Each row names the
  measurement that changed it."
- **Frame 33:** "These are the things we did not do. The port was optional, the
  reorder was discarded, NSA loses at small N, and the distortion families
  remain open."
- **Frame 36:** "This is the boundary: no state-of-the-art claim, no large-N
  accuracy, no association-head pretraining, no low-label gain, no significance
  test."
- **Backup B10:** "The feature-use and cosine controls are report prose only.
  The zero row is an out-of-distribution probe, not a no-feature baseline; the
  cosine value predates the aggregation correction."
