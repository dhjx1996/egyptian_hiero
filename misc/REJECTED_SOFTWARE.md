# Software considered and rejected

Why the stacks that were set up (or evaluated) in earlier phases are **not** part
of the two pipelines, and when each would become relevant again. All deleted
code lives in the pre-cleanup copy of this directory; each had a self-contained
`setup_env.sh`, so restoring = copy the folder back + re-run it.

| Software | Was going to be | Rejected because | Would return for |
|---|---|---|---|
| PaliGemma 2 (3B VLM) | the matching/dictionary model | closed-set, no ranked top-k, 6 GB gated GPU model | language layer (meanings, Q&A) |
| InkSight | raster→stroke-sequence converter | neither pipeline needs vector/stroke output | stroke-order animation, vector fonts |
| InkFM | ink foundation model | paper only — no public weights | n/a (InkSight is the open stand-in) |
| SwiftSketch + diffvg | glyph→SVG sketch front-end | sketch-style, not glyph-faithful; heavy CUDA build | free-form vector sketching |
| informative-drawings | photo→line-drawing front-end | inputs are already clean line art | photos of real inscriptions |
| Hackathon prototypes | matching baseline | superseded by Pipeline 2 | reference only |

## PaliGemma 2 — vs. the embedding matcher (Pipeline 2)

Considered as the dictionary backend (LoRA fine-tune → emit Gardiner codes).
Rejected on Pipeline 2's actual requirements:

- **Open-set growth was the top requirement.** A fine-tuned VLM only knows the
  classes it was trained on — adding a symbol or a new script means another
  training run. The embedding + canonical-prototype design makes that a
  re-index (embed one more PNG), no retraining — which is what makes the
  other-languages story cheap.
- **A dictionary app wants ranked top-k with scores.** Cosine similarity gives
  that natively; a text-generating model needs constrained decoding or scoring
  hundreds of prompts per query.
- **Deployability.** 3B params, ~6 GB, gated weights, GPU-bound — vs. a ~45 MB
  encoder + few-MB index at ~15 ms/query on CPU.
- **The data favors supervised metric learning.** 44k labeled drawings over 565
  classes: the small encoder hit 0.905 top-1 retrieval after 3 epochs on the
  T4. Zero-shot PaliGemma answered "unanswerable" on glyphs, so it offered no
  head start — it needed fine-tuning either way.

Where it genuinely wins is **language**: "what does this sign mean, used how?"
— an open-vocabulary Q&A layer on top of the matcher's top-k. That is a future
feature, not the matching engine.

## InkSight — vs. raster end-to-end

Offline→online ink derenderer (image → stroke sequences with order). Rejected
because neither pipeline needs vector output:

- Pipeline 1's deliverable is **raster** handwriting: One-DM conditions on
  raster style refs, and the procedural engine gets centerlines from cheap,
  deterministic skeletonization (`skeleton_simplify`) — no 5.6 GB TF model
  needed to recover strokes.
- Pipeline 2 takes **raster** queries because that's what a canvas exports (and
  it covers photos/scans too); an online stroke-based recognizer would restrict
  the app to trajectory capture.
- It is trained on Latin text — flagged out-of-distribution on glyphs — and
  derendering was slow per image, so preprocessing 44k drawings through it was
  cost without demonstrated benefit.

Becomes the right tool the moment stroke order matters: writing-order
animations for teaching, vector fonts, stroke-level style features, or an
online recognizer.

## InkFM

A paper (`InkFM.pdf`, also deleted) with no public weights — never runnable.
InkSight was adopted as its open stand-in, then rejected as above.

## SwiftSketch (+ diffvg) and informative-drawings

The old "route A front-end": image → clean line drawing (informative-drawings)
or vector sketch (SwiftSketch, which needs the finicky diffvg CUDA build and
1024² inputs). Rejected because the pipeline's inputs are **canonical glyphs
that are already clean line art** — skeletonization covers the front-end for
free, on CPU, glyph-faithfully, while SwiftSketch's output is deliberately
sketch-style. informative-drawings would matter only if inputs were photographs
of real inscriptions (out of scope for now). diffvg existed solely as
SwiftSketch's dependency. Their orphaned conda env can be removed with
`rm -rf ~/envs/swiftsketch ~/.local/share/jupyter/kernels/swiftsketch`.

## Hackathon prototypes (`translator_app_presentations/`)

The five team demos were the "primitive implementations" of matching;
Pipeline 2 replaces them. Their GitHub links are preserved in
`../pipelines/matching/README.md` (the videos, ~0.9 GB, were deleted).

## Rejected component inside kept software

One-DM's own `train_finetune.py` is **not** used for hieroglyphs: its frozen
OCR/CTC auxiliary loss is Latin-specific (and its `vae_HTR138.pth` /
`RN18_class_10400.pth` weights were deleted accordingly — re-fetch via
`One-DM/download_weights.py`). Fine-tuning uses `train.py --one_dm` instead
(diffusion-MSE + writer-NCE, charset-agnostic).
