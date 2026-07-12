# Adversarial review — matching pipeline (2026-07-11)

Scope: `pipelines/matching/` (encoder training, index, retrieval, evaluation,
stress testing, demo server), reviewed as the backend of the dictionary app.
Method: code + data inspection, then targeted GPU experiments on Ginsburg
(probe suite job **8927902**, leakage-corrected retrain job **8927903**; both
under account `crew`). All quantitative claims below are reproducible:

- probe suite: `review/adversarial_probe.py` → `review/probe_results.json`
  (production model `runs/default` = `a100_drop10`, resnet34@160)
- corrected retrain: `review/train_grouped.py` → `runs/review_grouped/`
  (identical production recipe, only the val split changed)
- qualitative evidence: `review/examples/`

## Executive summary

The pipeline is well-engineered (deterministic splits, val exclusion at index
time, frozen-gate synthetic filter, clean separation of train/index/eval), but
**the headline metric is inflated by dataset leakage, and the evaluation domain
is not the app's domain**. The honest numbers for the product are:

| What | Reported | Measured in this review |
|---|---|---|
| Held-out handwriting top-1 | **0.971** | **0.935** with a group-disjoint split (same recipe) — ~3.6 pt was leakage |
| Same, but frameless canvas-style queries (the app's input) | — | **0.78 top-1 / 0.85 top-5** (canvas-sim, 800 queries, 565 classes) |
| Unseen writer (pjb, 6 classes, n=256) | 0.863 | confirmed 0.863 (±4.2 pt 95% CI) — this, not 0.971, predicts app behavior |
| CPU latency (production config) | "≈15 ms (resnet18@112)" | **52 ms** 1-thread / 22 ms 4-thread (resnet34@160) |

None of this makes the matcher unusable — top-5 stays high everywhere that
matters (0.85–0.99), which is what the dictionary app actually needs. But the
README's numbers should be restated, and three trainable weaknesses (frames,
canvas domain, mirrored signs) have cheap fixes.

---

## Findings

### F1 — Train/val leakage: the 0.971 is substantially memorization (HIGH)

The Hand-drawn Hieroglyph Dataset is shipped **pre-augmented**: filenames
`<src>_<drawing>_<variant>_<CLASS>-x-y.png` carry ~12 near-identical variants
per source drawing (median 6 drawings/class × 12 variants). `build_items()`
splits at *file* level, so **100% of val images have same-drawing siblings in
train** (P1: `val_leaked_frac = 1.0`). Pixel-space nearest-train-neighbour
distances confirm it is genuine duplication, not just writer overlap: ≥10% of
val images have an *exact* duplicate in train (q10 = 0.000), 25% are within
normalized L2 0.001, 70% within 0.05; 81% of nearest neighbours are the same
source drawing.

Quantified impact: retraining with the **identical production recipe** but a
group-disjoint split (whole source drawings held out) gives held-out top-1
**0.935** (top-5 0.982, n=7344 files / 564 groups) vs 0.971 — and pjb is
unchanged (0.840 vs 0.863, well within the ±4 pt CI at n=256), i.e. the model
is equally good; only the measurement was optimistic. Note 0.935 is still a
*same-writer, same-scan-pipeline* number; the unseen-writer/unseen-capture
numbers below are lower.

Downstream contamination: every decision made on file-level val inherits the
bias — `best.pt` epoch selection, the `--p-dropstroke` sweep, and the synthetic
feedback accept/reject gate all optimized partially for memorization.

### F2 — The model leans on scan-frame borders that app queries won't have (HIGH)

Nearly every dataset image contains the scanned grid-cell border (frame
detected in 74% of val images; see `examples/p2_val0_orig.png`). Canonical
glyphs and app canvas queries have no frame. Measured (P2, n=800):

- val with frames stripped: 0.966 → **0.918** top-1 (−4.9 pt; stripping is
  imperfect — dashed remnants remain — so the true reliance is likely larger)
- pjb with a synthetic frame *added*: 0.863 → **0.676** top-1 (−18.7 pt)

The encoder is fitting frame context, and `crop_ink()` letterboxes the whole
framed cell rather than the glyph, so glyph scale at input differs
systematically between framed training data and frameless queries.

### F3 — The app's real domain costs ~19 pt over the quoted number (HIGH)

Re-rendering val queries as canvas-style drawings (frame-stripped, skeleton,
uniform pen width — what the drawing screen produces) gives (P3, n=800):

| pen radius | top-1 | top-5 |
|---|---|---|
| 2 px | 0.765 | 0.850 |
| 4 px | 0.776 | 0.850 |
| 8 px (thick marker) | **0.654** | 0.774 |

This is consistent with pjb (real frameless canvas drawings: 0.863/0.988 but
only 6 easy, common classes). Product takeaway: **the app will miss the right
sign in its 5 suggestions roughly 1 time in 7** across the full inventory, and
thick strokes measurably hurt. (Caveat: canvas-sim images retain some
frame-remnant artifacts, see `examples/p3_canvas_sim0.png`, so 0.78 has noise
in both directions.)

**On stroke thickness specifically** (2026-07-11 follow-up): fixing the app's
pen to a single width does *not* by itself remove this variable, because the
query is crop-letterboxed to 160 px — the encoder sees stroke width *relative
to drawing extent*, which swings ~3× with how large the user draws even at a
fixed pen. The app therefore (a) ships exactly one on-screen pen — **do not add
a thickness slider**; it would reintroduce a nuisance variable the model is
measurably weak to (0.78 → 0.65 thin → thick) for zero lookup benefit — and
(b) normalizes at match time by re-rendering its vector strokes at ~2% of the
glyph extent (≈3 px post-letterbox, the best point in the sweep above), which
pins the variable entirely for app queries. Two caveats keep this from
"circumventing the need for stroke robustness": normalization only covers this
app's canvas (photos, scans and other frontends still present arbitrary
widths), and the dominant stroke failures are completeness problems, not width
— severed strokes 0.78, wobble 0.94, half-finished drawings 0.09 (F4) — which
no rendering normalization can remove. The training-side recommendations
stand.

### F4 — Corruption robustness is aug-specific; realistic novel corruptions fail (MEDIUM)

`--p-dropstroke` training raised the *identically-coded* dropstroke corruption
(the train aug and `stress_test.c_dropstroke` are the same algorithm — the
stress test partially measures "did you train on the test transform"). Novel
corruptions the training never mirrors (P4, n=800):

| corruption | top-1 | note |
|---|---|---|
| mirrored (hflip) | **0.699** | hieroglyphs legitimately face either direction on monuments |
| rot90 | 0.260 | orientation is essentially memorized ±12–20° |
| polarity inverted | 0.768 | dark-mode canvas / white-on-black photos |
| **half-finished drawing (top half)** | **0.091** | a user pausing mid-draw sees garbage |
| half (left half) | 0.586 | |
| thick strokes ×2, jpeg q18, gridlines | 0.95–0.97 | genuinely robust |

The half-drawing failure matters because the app matches on every stroke-end;
the mirror failure matters for transcription from photos.

### F5 — No rejection mechanism, and scores are displayed misleadingly (MEDIUM)

`Matcher.match` always returns top-k; blank/garbage input gets confident-looking
output, and `demo_server.py` renders raw cosine as a percentage ("93.1%").
Measured separability is actually good (P5): genuine top-1 scores (median 0.957)
vs impostors (true class removed from index; median 0.381, AUROC 0.999),
scribbles 0.996, latin letters 0.996 — but **closed geometric shapes score like
real signs** (AUROC 0.93; a drawn circle *is* close to signs like N5/Aa1, so
this is partly legitimate). A single threshold works well:

- **score ≥ 0.60**: keeps 99.0% of genuine top-1s, rejects ~99% of impostors,
  ~91% of scribbles, ~87% of letters (P5 threshold table)
- calibration (P7): below score 0.75 precision decays fast on unseen-writer
  data (0.6–0.75 band ≈ 60–74% precision on pjb); margin ≥ 0.1 → 96% precision

The dictionary app built in this work ships with score 0.60 + margin 0.03 as
its "low confidence" gate (`app/data/config.json`).

### F6 — Centroid prototypes dominate rankings but add ~nothing (MEDIUM)

On val, when the matcher is right, the winning prototype is a handwriting
centroid **767/773 times** — yet deleting all centroids from the index barely
moves accuracy (P6): val 0.966 → 0.961, and **pjb improves** 0.863 → 0.871.
The README's "recommended: big accuracy boost" for centroids is not supported
for this model; centroids mostly re-rank within already-correct results while
tying the index to the (leaky, framed) training distribution. The open-set
promise itself holds under its own assumptions: canonical-only classes (204 of
769 have zero handwriting) probed with procedural pen-sim queries score 0.992
top-1 — but note this is the training augmentation distribution, an upper
bound, and those 204 classes have never been evaluated on a single real human
drawing (val covers only the 565 classes with data).

### F7 — Selection/metric hygiene (LOW-MEDIUM)

- `best.pt` is chosen by *classifier-head* val accuracy, not the deployed
  nearest-prototype retrieval metric (correlated, but not the same objective).
- The `--p-dropstroke` sweep chose 0.10 over 0 on a −3.2 pt pjb delta that is
  inside the n=256 noise band (±4.2 pt) — the README says so itself; decisions
  at this probe size can't distinguish the candidates. The 6-class pjb probe is
  doing far too much load-bearing work across the whole project.
- Reported robustness/latency claims mix configs: "15 ms CPU" is resnet18@112;
  production resnet34@160 measures 52 ms 1-thread / 22 ms 4-thread (P9) —
  still fine for the app, but the README number is stale.

### F8 — Engineering nits (LOW)

- `load_encoder` uses `torch.load(..., weights_only=False)` — arbitrary code
  execution if a checkpoint is ever swapped; the checkpoint dict is plain
  tensors + metadata, so `weights_only=True` should work.
- `demo_server.py` reads `Content-Length` bytes with no cap (memory DoS) and
  has no auth — fine for its documented dev-only role, but don't let it become
  the app backend.
- `D36` exists in `utf-pngs` but not in `gardiner_hieroglyphs.csv` — it can be
  matched but shows an empty description (visible in the app).
- Classes with <8 handwriting files contribute no val items (`min_val_n=8`),
  so per-class coverage of the eval is silently partial.
- P8 measured "fixes" so they don't get cargo-culted: TTA (5-view) ≈ +0.5 pt
  val / +1 pt canvas-sim but −1.6 pt pjb — not worth 5× latency; mean-instead-
  of-max aggregation *hurts* pjb (−4.3 pt); unsharp+Otsu makes blur *worse*
  (0.455 → 0.304); blur+Otsu re-binarization genuinely helps lowres
  (0.391 → 0.574) if photo input is ever supported.

### What held up under attack

Credit where due: `--exclude-val` was correctly used for the production index
(verified in `slurm/matching.sbatch`); splits are deterministic and recorded;
the synth-filter design (frozen gate, real-only val, caps) already anticipated
feedback-loop collapse and its 2026-07-10 REJECT verdict was the right call;
garbage separability is strong; thick-stroke/jpeg/gridline robustness is real;
and 22 ms CPU latency leaves comfortable headroom for on-device deployment.

---

## Recommendations

Ordered by product impact per unit effort:

1. **Make the group-disjoint split the default** (adopt
   `review/train_grouped.py::build_items_grouped` into `hieromatch/data.py`,
   keyed on the `<src>_<drawing>` filename prefix) and restate README numbers
   as: held-out (group-disjoint) 0.935 / unseen-writer 0.86 / canvas-domain
   ~0.78 top-1, 0.85–0.99 top-5. Never quote the file-level split again.
2. **Train for the app domain**: add (a) random synthetic frames to *all*
   training sources + frame-stripping in preprocessing, (b) a canvas-style
   rendering branch (skeleton + uniform stroke, pen-width jitter) as an
   augmentation of both handwriting and canonical images, (c) horizontal-flip
   augmentation or flip-TTA (max over both orientations) — after checking the
   few mirror-distinct sign pairs, (d) progressive-stroke crops (train on
   partial drawings) so match-as-you-draw degrades gracefully instead of
   0.09 top-1 on half drawings.
3. **Ship a rejection threshold** (score 0.60 + margin 0.03 — already wired
   into the app) and stop displaying raw cosine as a percentage in
   `demo_server.py`. Keep the app at one fixed pen with match-time
   stroke-width normalization (done; see F3) — no thickness slider.
4. **Grow the unseen-writer probe** past 6 classes — the app's confirm-tap
   ("user selects the glyph they intended") is a labeled-data pump; store
   confirmed drawings and promote them into an expanded probe set before using
   them as training data.
5. **Simplify the index**: canonical-only prototypes are equal-or-better
   (pjb +0.8 pt) — either drop centroids (smaller index, no val-exclusion
   bookkeeping, better writer generalization) or keep them behind an A/B flag.
6. Evaluate the 204 canonical-only classes on at least a handful of real
   drawings each (one afternoon of data collection) — 0.99 on procedural
   queries is self-graded homework.
7. Report binomial CIs next to every probe metric; treat pjb deltas ≲ 6 pt as
   ties at n=256 (two-sample 95% band). Select `best.pt` by retrieval-against-index on val, not
   classifier-head accuracy.
8. Hygiene: `weights_only=True` in `load_encoder`; cap `demo_server` request
   size; add D36 to the CSV; note in stress_test that dropstroke/wobble share
   code with training augs (use held-out corruption *implementations* when
   claiming robustness).

### Addendum — deployment quantization (from the app build, jobs 8933864/8934061/8934101)

Two measured facts worth keeping with the pipeline: (1) **int8 static
quantization of this encoder silently trades away unseen-writer accuracy** —
val stays flat (−0.2…+0.5 pt) while pjb drops 2.3–2.7 pt even with per-channel
weights and canvas-style calibration data; any future quantized deployment
must gate on pjb, not val. (2) **fp16 ONNX passes numerical checks on the CPU
runtime but produces wrong rankings under onnxruntime-web's wasm backend**
(1.19.2) — an end-to-end test through the actual deployment runtime caught
what the Python-side check could not. The app therefore ships the fp32 model,
selected automatically by `slurm/app_export.sbatch`'s two-stage gate.

## Reproduction record

| Artifact | Path |
|---|---|
| Probe suite + results | `review/adversarial_probe.py`, `review/probe_results.json` (job 8927902, ~35 min on 1 GPU) |
| Grouped retrain | `review/train_grouped.py`, `runs/review_grouped/{best.pt,eval.json,stress.json,log.txt}` (job 8927903, ~2 h on 1 GPU) |
| Example images | `review/examples/` (frame stripping, canvas-sim, framed pjb, pen-sim) |
| Slurm scripts | `slurm/review_probe.sbatch`, `slurm/review_grouped.sbatch` |

Grouped-model stress (same 800-query protocol, group-disjoint val): clean
0.938, rotate20 0.907, wobble 0.866, occlude28 0.811, dropstroke 0.718, blur
0.399, lowres32 0.334 — the corruption *ordering* is unchanged; only the clean
baseline moves down, consistent with F1 being measurement inflation rather
than a training defect.
