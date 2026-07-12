# Hieroglyph Dictionary — draw a sign, look it up

Minimalist dictionary app on top of the matching pipeline: **draw a glyph →
top-5 canonical Gardiner signs → tap the one you meant → description, details,
Unicode → back**. One codebase for Android phones/tablets and iPhone/iPad.

## How it runs everywhere (the platform decision)

This is an **installable PWA, with all inference on-device**: the production
encoder (`pipelines/matching/runs/default`) exported to ONNX (int8-quantized,
accuracy-gated) and executed in the browser via `onnxruntime-web` (WASM), with
the prototype index + cosine matching done in plain JS. Chosen over native
APK/Xcode builds because it is the only single artifact that covers all four
required device families with no app-store accounts, works **offline after the
first visit** (service worker caches the model), keeps drawings on-device
(privacy), and needs **zero backend** — hosting is any static file server.
If store distribution is ever wanted, this app wraps directly with
Capacitor/TWA without code changes.

## Layout

| Path | What |
|---|---|
| `index.html`, `css/`, `js/app.js` | UI: draw screen (canvas, undo/clear, auto-match on stroke end) + detail screen (info from `hiero_data`'s Gardiner CSV) |
| `js/preprocess.mjs` | faithful JS port of `hieromatch/data.py` preprocessing (crop-ink → letterbox → tensor); shared by browser and build test |
| `js/matcher.mjs` | ONNX session + per-label max cosine ranking (mirrors `match.py::Matcher`) |
| `data/` | generated: `model.int8.onnx`, `index.bin`, `index_meta.json`, `glyphs.json`, `config.json` (incl. the review-derived rejection threshold), `selftest.json`, `export_report.json` |
| `glyphs/` | generated: 769 canonical thumbnails |
| `vendor/ort/` | generated: vendored onnxruntime-web (self-contained, no CDN) |
| `build/` | `export_app_assets.py` (ONNX export + quantization gate + assets), `test_app_pipeline.mjs` (node end-to-end check of the real app JS vs torch reference) |
| `sw.js`, `manifest.webmanifest`, `icons/` | PWA shell (offline cache, install banners) |

## Build / rebuild assets

```bash
sbatch slurm/app_export.sbatch     # on Ginsburg (CPU job), or locally:
cd pipelines/matching && ./.venv/bin/python ../../app/build/export_app_assets.py
cd ../../app/build && npm install && node test_app_pipeline.mjs   # must PASS
```

The export job builds int8 and fp16 variants and gates them twice: a Python
accuracy gate (reject if top-1 drops >1.5 pt on val **or** the unseen-writer
probe; see `data/export_report.json`) and then the node test, which runs the
*actual* app JS (preprocess + matcher) under onnxruntime-web against
torch-computed references and picks the smallest model that passes. Both gates
have caught real failures: int8 costs ~2.3 pt of unseen-writer accuracy however
calibrated, and fp16 misranks under the wasm runtime despite passing on CPU —
so the app currently ships the **fp32 model (~85 MB, one-time download, cached
offline by the service worker)**.

## Serve / deploy

Any static host (GitHub Pages, Netlify, `python -m http.server` on a LAN).
HTTPS (or localhost) is required for the service worker/PWA install. Quick
local test:

```bash
cd app && python3 -m http.server 8080     # then http://localhost:8080
```

- **Install**: Android Chrome → "Add to Home screen"; iOS Safari → Share →
  "Add to Home Screen". Both give a standalone fullscreen app.
- **Self-test on any device**: open `…/?selftest=1` — runs bundled fixtures
  through the on-device pipeline and reports PASS/MISMATCH.

## Behavior notes (tied to the adversarial review)

- Matching triggers ~350 ms after each stroke ends; results show the best
  match (gold ring) + 4 alternatives.
- A "low confidence" banner appears when top-1 cosine < 0.60 or the margin to
  rank-2 < 0.03 — thresholds measured in `pipelines/matching/review/REVIEW.md`
  (P5/P7): ~99% of genuine matches kept; ~99% of impostor signs and ~90% of
  scribble/letter garbage flagged.
- Expected accuracy in this drawing domain is **~0.78 top-1 / ~0.85 top-5**
  across all 769 signs (review P3) — the 5-candidate tray plus manual pick is
  the product answer to that.
- There is exactly **one pen, and no thickness slider by design** (review F3):
  thick strokes cost ~12 pt, so a slider would only add a way to hurt
  accuracy. Matching re-renders the vector strokes at ~2% of the glyph extent
  (≈3 px after letterboxing), so how large you draw doesn't change the stroke
  width the model sees either; the on-screen pen is purely cosmetic.
- The canvas keeps a light background in dark mode on purpose: the encoder
  expects dark ink on a light ground (review P4: inverted polarity costs
  ~20 pt).
- The drawing and results survive the detail-screen round trip (back button);
  "✎ Next sign" on the detail screen instead returns to a cleared canvas for
  the next lookup.

## Planned features

- **History tab** — a locally-stored log of confirmed lookups (drawing +
  chosen sign), so users can revisit past identifications; doubles as the
  labeled-drawing collection point the review recommends for growing the
  unseen-writer probe. Not implemented yet.
