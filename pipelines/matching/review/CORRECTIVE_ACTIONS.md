# Corrective actions — response to REVIEW.md + the abstraction gap (2026-07-19)

Response to the 2026-07-11 adversarial review, plus the abstraction gap raised
separately (figurative signs drawn as stick figures, e.g. `A39_abstracted.jpg`).

Honesty note up front: the abstraction fix below is an **untested bet**, not a
demonstrated win. We have exactly **one** real abstracted drawing (A39) and it
already *passes*, so nothing here can yet be shown to move the product gap. See
[§Validation](#validation--what-is-still-tbd). The review's own measurement
discipline (don't grade a fix on the transform you trained on; treat pjb deltas
≲6 pt as ties) is applied throughout.

---

## 1. What was done

| # | Action | Status | Notes |
|---|---|---|---|
| 0 | **Repo unblocked for training** | done | `egyptian_hiero`→`seshat` rename left the venv symlink and all 20 `slurm/*.sbatch` paths dangling; the venv python was broken. Added a compat symlink `egyptian_hiero → seshat` (reversible). See [§bridge](#a-note-on-the-symlink-bridge). |
| 1 | **Abstraction bank + training integration** | done | `make_abstractions.py` (reuses `skeleton_simplify`); `--abstract` flag in `train_encoder.py` and `review/train_grouped.py`. |
| 2 | **Isolated abstraction training run** | done — job `9114107`, see [§7](#7-results-2026-07-22) | `slurm/matcher_abstract.sbatch`: `review_grouped` recipe with abstraction as the **single** added variable. |
| 2b | **Isolated frame-aug run (F2)** | done — job `9116693` (resubmit of `9114535`, which died at epoch 44 to a transient DataLoader worker crash), see [§7](#7-results-2026-07-22) | `slurm/matcher_frames.sbatch`: baseline + `--p-frame 0.5` only. Also re-ran the baseline stress with the new `frame`/`half_top` corruptions (`runs/review_grouped/stress_new.json`) for an apples-to-apples F2/F4 baseline. |
| 2c | **Isolated partial/half-draw run (F4)** | done — job `9114536`, see [§7](#7-results-2026-07-22) | `slurm/matcher_partial.sbatch`: baseline + `--p-partial 0.5` only. |
| 3 | **F8 security fix** | done, verified | `load_encoder` now `weights_only=True` (was arbitrary-code-exec on a swapped checkpoint). Loads production ckpt cleanly. |
| 4 | **F8 data fix** | done, verified | Added missing `D36` (forearm, 𓂝 U+1309D) to `gardiner_hieroglyphs.csv` — codepoint verified against `unicodedata` (the source JSON's D-series hex is truncated/corrupt, so it was cross-checked, not copied). |
| 5 | **Review recommendations triaged** | see [§6](#6-review-findings--disposition) | rec #1 (grouped split) is live in the running job; frames/flip/partial deliberately deferred to separate runs (below). |

---

## 2. The abstraction gap — what we actually know

The claim "the encoder can't handle abstractions" is **too strong** and our own
evidence contradicts it. Separating the two things that were being conflated:

**(a) The one real human abstraction works.** `A39_abstracted.jpg` (cropped to
the canvas drawing → `review/examples/abstraction_query_A39.png`) matches **A39
at cos 0.617, rank 1, margin 0.10 over G6** — above the app's 0.60/0.03 gate.
The current production encoder handles this human stick-figure fine. The user
confirms ("we were successful this time").

**(b) Machine `skeleton_simplify` renders are out-of-distribution to the current
encoder.** Embedding-space cosine of the A39 abstraction render to A39 itself:

| render | cos(real A39 query, render) | cos(detailed A39 proto, render) |
|---|---|---|
| detailed canonical A39 | **0.617** | 1.000 |
| skeleton render, keep 40% strokes | 0.001 | 0.002 |
| keep 70% | 0.197 | 0.253 |
| keep 100% (skeletonize+simplify only) | 0.318 | — |

Two takeaways that drove the design:

1. **Low stroke-budgets are garbage, not abstraction.** "Keep the N longest
   strokes" drops the *shortest* strokes first — which for a compound figure
   deletes the small salient element. On A39 the ~40% render is *two giraffes,
   no human* (`examples/abstraction_sweep_A39.png`): cos 0.001, **below the
   impostor median (0.381, review P5)** — literally less like A39 than a random
   sign. Training on those is label noise. The bank therefore ships a single
   **salience-preserving `frac=1.0`** level (skeletonize + Douglas-Peucker +
   uniform thin stroke — the double-outline→centerline transform that *is* most
   of the canonical↔hand-drawn gap), which keeps the A39 loop-head stick figure.
   Stroke-*dropping* variety is already covered by the on-the-fly `p_dropstroke`
   aug, so it isn't baked in as static noise.

2. **This is a training signal, not an index prototype.** Because the renders
   are OOD, adding them to the index does **nothing**: rebuilding the production
   index with abstract prototypes leaves A39 at 0.617 with the *detailed*
   prototype still winning (verified — the zero-GPU lever is a no-op on the
   current encoder). The fix has to be at the **encoder**: cross-entropy pulls
   the abstract renders back onto their class, teaching skeletal-invariance.
   Hence the training run. (Whether abstract prototypes help *after* retraining
   is tested separately — the sbatch builds both index variants.)

---

## 3. Immediate corrective action (shipped as code, running now)

- **`make_abstractions.py`** — for each of the 769 canonical glyphs, one skeletal
  stick-figure render (`hiero_data/abstractions/lvl1/<CLASS>.png`, ~2 s for the
  whole set on 24 cores). Drops straight into `build_index --canonical` or
  `train_* --abstract`.
- **Training integration** — abstraction renders enter as `source="canon"` items
  (canonical heavy-aug path, so `_handwrite` wobble lands on top = a *hand-drawn*
  stick figure), train-only, never in val. In the running job: `abstract_repeat
  6` → 4614 abstraction items (~50% of detailed-canonical weight); the val split
  is byte-identical to `review_grouped` (7344 files / 564 groups), so
  **abstraction is the only variable** — a clean A/B against the honest baseline.
- **`slurm/matcher_abstract.sbatch`** (job `9114107`) — train → index → evaluate
  (grouped-val + pjb + the real A39 probe) → stress; then a second index *with*
  abstract prototypes, evaluated the same way. Self-validating on the pjb
  regression gate.

---

## 4. Brainstorm — ways to reduce the abstraction gap

Ranked by product-impact / effort. #1 is done; the rest are proposed.

1. **Abstraction augmentation (done).** Skeletal renders as canonical-style
   training data. Cheapest lever, reuses existing code, generalizes to every
   figurative sign, and doubles as the review's rec-#2b "canvas-style rendering
   branch" (skeleton + uniform stroke). Also lifts the 204 canonical-only
   classes, which today have *zero* real drawings behind them.
2. **Real stick-figure data from the collection pump (highest quality, slow).**
   Already implemented client-side (`app/js/collect.mjs`, confirm-tap = label).
   This is the only source of *true* human abstractions. Action: promote
   confirmed low-score/misranked drawings first into a probe, then into training
   — exactly the review's rec #4. Needs users to export (nothing is uploaded).
3. **Salience-aware simplification (better than length-ranked).** The A39 failure
   shows "keep longest strokes" is wrong for compound figures — it discards the
   human. A budget that keeps at least one stroke per connected component, or
   weights central/small components up, would produce human-like abstractions at
   *lower* stroke counts (more aggressive, still correct). Medium effort; only if
   frac=1.0 proves insufficient.
4. **Category-targeted weighting.** Abstraction is a figure phenomenon
   (Gardiner A/B/C/D/E/F = people, gods, animals, parts). The category is free in
   the class name — weight abstraction aug up for those, leave geometric signs
   (N/O/Q/S/Aa) alone. Trivial once #1's effect is measured.
5. **Abstract prototypes in the index (after retrain only).** Once the encoder is
   skeleton-invariant, an abstract prototype *may* catch queries the detailed one
   misses. Gate on the P5 separability probe (don't let abstract prototypes steal
   rank). The running job measures this (`index_abstract.npz`). Reversible A/B.
6. **Progressive/partial-stroke training (also fixes F4's 0.09 half-drawing).**
   People abstract *and* draw incrementally; both are "fewer strokes present."
   One augmentation (keep a random contiguous region / stroke prefix) serves
   both. Deliberately **not** in the abstraction run (keeps the variable clean) —
   proposed as the next isolated run.

Rejected: a match-time skeletonized-query branch (review F3 already shows
match-time normalization can't fix *completeness* problems, only width) and
figure-specific pose skeletons (over-engineered for the payoff).

---

## 5. Validation — what is still TBD

**We cannot yet claim the gap is closed, and the report will not pretend
otherwise.** Reasons:

- The only real abstracted sample (A39) already passes, so the retrain has no
  *rank* headroom to prove on it — only score/margin on n=1, which is noise.
- There are **zero reproduced failures**. "Other abstracted queries failed" is
  reported but none were captured; the collection store is browser-local and
  currently empty on disk.

So the go/no-go on the running job is, honestly:

- **Primary gate (fully honest):** the abstraction run must **not regress** the
  `review_grouped` baseline — pjb 0.840 top-1 / 0.984 top-5, grouped-val 0.935 /
  0.982, and the stress ordering. A win on abstraction that costs unseen-writer
  accuracy is a net loss.
- **Directional only:** A39 stays ≥ its 0.617 / rank-1, and the abstract-index
  variant doesn't hurt separability.
- **Real signal is blocked on data** → see below.

**A green regression gate is NOT "abstraction handled."** Transfer risk: the real
A39 human abstraction is cos 0.617 to the *detailed* canonical but only 0.318 to
our frac=1.0 render — the human drew the giraffes as necks+legs, `skeleton_simplify`
keeps their *body outlines* as centerlines, a structurally different transform. So
the encoder is being trained toward a proxy that may not be where real human
abstractions actually live. `9114107` finishing with pjb held and A39 still passing
means only that — the abstraction benefit stays unmeasured.

**Discriminating check when failing abstractions arrive** — for each, compute (1)
its rank (misranked past 5, or just sub-threshold?), and (2) cos to
detailed-canonical vs cos to the frac=1.0 render:
- consistently closer to the **detailed** prototype (as A39 is) → the skeleton
  bank is the *wrong* bridge; the lever is real collected data / heavier
  `_handwrite`, not more skeleton renders.
- closer to the **render** → the bank is on-target and `abstract_repeat` is the
  knob to turn.

---

## 6. Review findings — disposition

| Finding | Disposition |
|---|---|
| **F1** file-level leakage | rec #1 adopted. Group-disjoint split is now the **default** in `hieromatch/data.py` (`build_items(group_val=True)`, user-approved); legacy leaky split only via `--file-level-split`. `review/train_grouped.py::build_items_grouped` is a thin delegate (verified item-identical split: 769 classes / 45969 train / 7344 val). |
| **F2** frame reliance | deferred to a separate isolated run (frame aug + strip). Bundling it here would confound the abstraction A/B (advisor's call). |
| **F3** canvas domain | partially covered — frac=1.0 abstraction *is* the skeleton+uniform-stroke branch. App-side width normalization already shipped. |
| **F4** mirror/rot/half-draw | half-drawing (0.09) is the worst; folded into brainstorm #6 (next run). Mirror/flip deferred — needs the mirror-distinct sign pairs curated first (don't corrupt directional labels). |
| **F5** rejection threshold | already shipped in-app (0.60 + margin 0.03). No action. |
| **F6** centroids add ~nothing | measured again here incidentally; leaving as-is (A/B flag is the review's suggestion, low priority). |
| **F7** selection/metric hygiene | not addressed this pass (best.pt-by-retrieval, CIs). Low risk. |
| **F8** engineering nits | `weights_only=True` ✅, `D36` CSV ✅. `demo_server` size cap not done (dev-only tool). |

Sequenced next runs (each an isolated variable, per the review's "pjb deltas ≲6
pt are ties" discipline): frame aug (F2) and progressive/partial-stroke (F4) —
both now **done**, results below.

---

## 7. Results (2026-07-22)

All three isolated A/B runs completed. Shared baseline: `review_grouped`
(identical 7344-file / 564-group val split for every run). pjb n=256 over 6
easy classes — deltas ≲6 pt are ties.

| metric | baseline | + abstraction (`9114107`) | + frame (`9116693`) | + partial (`9114536`) |
|---|---|---|---|---|
| grouped-val top-1 | **0.935** | 0.920 (−1.5) | 0.923 (−1.3) | 0.913 (−2.2) |
| pjb unseen-writer top-1 | 0.840 | **0.973 (+13.3)** | 0.906 (+6.6, borderline) | 0.813 (−2.7, tie) |
| pjb top-5 | 0.984 | 0.992 | 0.965 | 0.953 |
| A39 abstraction probe | rank 1 (prod, 0.617) | rank 1 | rank 1 | rank 2–5 |
| stress `frame` top-1 | 0.911 | 0.904 | 0.918 (tie) | 0.891 |
| stress `half_top` top-1 | 0.109 | 0.109 | 0.091 | **0.345 (~3×)** |

(Baseline `frame`/`half_top` from `runs/review_grouped/stress_new.json` —
re-stressed with the new corruption set, so apples-to-apples.)

**Reads:**

- **Abstraction — the standout.** pjb 0.840 → 0.973 (+13.3, well past the tie
  band) at −1.5 val, A39 stays rank 1. Coherent generalization story (skeletal
  invariance transfers to unseen writers), but pjb's 6 easy classes mean this
  needs confirmation on the real failing abstractions before promotion. The
  *abstraction gap itself* remains unmeasured (§5 stands unchanged).
- **Abstract prototypes in the index — still a no-op, even after retrain.**
  `eval_abstract_index.json`: pjb 0.969 vs 0.973, val 0.921 vs 0.920 —
  within noise. Ship the plain index; brainstorm #5 is closed (measured, no
  benefit).
- **Frame — F2 largely a non-problem at query time.** The *baseline* already
  scores 0.911 under the `frame` corruption; frame aug moves it to 0.918 (tie),
  shrinks the clean→frame drop from −2.6 to −0.9 pt, costs −1.3 val. pjb +6.6
  sits right on the tie boundary. Verdict: not worth adopting alone; the
  review's frame-reliance concern was real in *training* signal terms but does
  not translate into a query-time failure worth a dedicated aug.
- **Partial — does its one job, at a price.** `half_top` 0.109 → 0.345 (~3× on
  half-finished drawings) but −2.2 val, −2.7 pjb, and A39 drops out of rank 1
  (still top-5). A robustness-vs-accuracy trade to take only if half-draw
  matters more than ~2 pt everywhere else. Not a default-recipe candidate.
- **Candidate recipe going forward:** abstraction aug alone (pending real-data
  confirmation). Frame and partial stay available as flags
  (`--p-frame`/`--p-partial`) but off by default.

Infra note: frames job `9114535` failed at epoch 44 with "Pin memory thread
exited unexpectedly" (transient DataLoader worker death, not the aug — 44 clean
epochs, val 0.920). Resubmitted as `9116693` with `--mem=96G`, workers 16→12
(results-identical: aug rng is seeded per (seed, epoch, index)); completed
cleanly.

---

## What I need from you

To turn the abstraction fix from a bet into a measured result, I need the
**failing** abstractions, not the passing one:

1. In the app's About dialog, **Export** your collected drawings (the JSON has
   embedded PNGs + labels + the rank each got). Drop it anywhere in the repo.
2. Or point me at the specific signs whose abstractions failed.

I'll build a real abstraction probe from those and re-run evaluation against both
the baseline and the new model — that's the only honest way to say whether this
worked.

---

### A note on the symlink bridge

`egyptian_hiero → seshat` is a **bridge**, not a real fix: the venv's internal
symlinks and every `slurm/*.sbatch` still hardcode the old name. It works today
because the symlink makes both paths resolve. The clean fix (when convenient) is
to rebuild the venv under `seshat` and `sed` the sbatch paths; until then, don't
delete the symlink or jobs stop resolving their interpreter.
