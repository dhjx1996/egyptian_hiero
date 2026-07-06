# egyptian_hiero — Setup, Workflow & Notes

Goal: two product pipelines around Gardiner signs (Egyptian hieroglyphs),
both generalizable to other symbol inventories / ancient scripts:
1. **Generation** — produce human-looking handwriting for any canonical symbol.
2. **Matching** — recognize a drawn symbol (dictionary-app backend).

This file is the map. Companion docs:
- **`../pipelines/README.md` — THE TWO PIPELINES (start here)**, with
  `../pipelines/generation/README.md` and `../pipelines/matching/README.md`
  as the full runbooks (T4 smoke + HPC/A100 training commands).
- **`../pipelines/smoke_results/`** — predictions + generated samples proving
  each pipeline works (with its own README).
- `REJECTED_SOFTWARE.md` — why InkSight/PaliGemma/SwiftSketch/… were considered
  and not used, and when each would become relevant again.
- `One-DM/HIERO_WORKFLOW.md` — One-DM internals + adaptation notes.

> **2026-07-06 cleanup:** stacks not used by the two pipelines were deleted —
> `InkSight/` (ink derenderer), `misc/PaliGemma/` (VLM dictionary route),
> `SwiftSketch/` + `diffvg/` + `informative-drawings/` (image→sketch/SVG route),
> `translator_app_presentations/` (hackathon videos; repo links preserved in
> `pipelines/matching/README.md`), the photo/SVG scripts
> (`capped_simplify.py`, `vector_handwrite.py`), old outputs, unused One-DM
> weights (`vae_HTR138.pth`, `RN18_class_10400.pth` — re-fetch via
> `One-DM/download_weights.py`), and the ~10 GB uv wheel cache. Everything is
> recoverable from the pre-cleanup copy of this directory. The orphaned conda
> env `/home/jovyan/envs/swiftsketch` (outside this tree) and its `swiftsketch`
> Jupyter kernel can be deleted too: `rm -rf ~/envs/swiftsketch
> ~/.local/share/jupyter/kernels/swiftsketch`.
> Rationale for each rejection: `REJECTED_SOFTWARE.md`.

## 0. Pipeline status (2026-07-06, verified on this box's T4)

- **Generation** (`pipelines/generation/`): procedural engine works today
  (CPU); One-DM route fully wired — `prep_dataset.py` builds a complete
  fine-tune dataset from ANY `<CLASS>/*.png` + `<CLASS>.png` corpus (Unicode
  optional → PUA fallback), `train.py --one_dm` fine-tunes (2-epoch smoke ran
  on the T4; real run belongs on A100s), `generate_hiero.py` samples per sign
  with aspect-aware widths, external glyphs (`--glyph`) and arbitrary style
  folders (`--style-dir`). Fine-tuning is REQUIRED for visual quality (the
  IAM-Latin checkpoint zero-shots to near-blank on glyphs). The full 565-class
  dataset is already prepped in `One-DM/data/hiero*`.
- **Matching** (`pipelines/matching/`): encoder + canonical-prototype index
  (open-set: new symbols/scripts = re-index, no retrain). T4 smoke (3 epochs):
  held-out handwriting top-1 0.905; unseen-writer pjb probe top-1 0.859 /
  top-5 0.973. Trained model + index live in `pipelines/matching/runs/smoke/`;
  visual evidence in `pipelines/smoke_results/matching/`.
- Both consume/emit the same labeled-tree layout, so generated handwriting can
  be scored for recognizability by the matcher (`evaluate.py --probe-dir`).

---

## 1. Environment (portable & reset-proof) — **set up & verified**

JupyterHub here wipes the container (`/srv/conda`, the overlay `/`) on shutdown,
but **`/home/jovyan` is a persistent NFS mount**. So the whole environment is
built **inside the project tree** on that mount and survives resets.

- `env/` — main env for `scripts/` + the procedural generation engine.
  - **Activate:** `source misc/env/activate.sh`
  - **Add a package:** `uv pip install <pkg>` then add it to `env/requirements.txt`
  - **Recover after a reset / repair:** `bash misc/env/bootstrap.sh` (idempotent)
- Built with **uv** (`misc/.tools/bin/uv`), a **project-local managed CPython
  3.11** (`misc/.tools/python/`) and `misc/.venv/`. The repo-root symlink
  `.tools -> misc/.tools` lets the root-level setup scripts find uv. Nothing
  points at the ephemeral `/srv/conda`. (The wheel cache was emptied in the
  cleanup; rebuilds need internet — see `RESETUP.md`.)
- Jupyter kernels (persistent in `~/.local/share/jupyter`): `egyptian_hiero`
  (main), `onedm`, `hieromatch`.
- **One-DM** and **`pipelines/matching`** each have their **own** env
  (`*/setup_env.sh` → `*/.venv`) so their torch stacks stay isolated from the
  main scripts env.
- **GPU:** heavy compute targets GPUs (A100 on the HPC; this box has a Tesla
  T4 for smoke tests). Both ML `setup_env.sh` scripts **auto-detect CUDA**;
  override with `DEVICE=cuda|cpu` (and `CUDA=cu121|cu124|cu118`). The `env/`
  scripts env stays CPU (pure image processing).

## 2. Data layout (`hiero_data/`)

| Path | What | Format |
|------|------|--------|
| `archaeohack-starterpack/data/utf-pngs/` | **769 canonical Gardiner glyphs** (font-rendered "ground truth"), one per sign | grayscale, ~360×662 |
| `archaeohack-starterpack/data/me-sign-examples-pjb/` | 6 signs drawn by one author ("pjb") | RGBA 200×200 — **single-writer probe** |
| `Hand-drawn Hieroglyph Dataset/` | **565 Gardiner classes, ~44k instances (~78/class)** of real hand-drawn signs | RGBA, variable — **main handwriting data** |
| `archaeohack-starterpack/data/gardiner_hieroglyphs*.{csv,json}` | Gardiner ↔ Unicode (U+13000+) ↔ description | — |
| `archaeohack-starterpack/lib/font/NotoSansEgyptianHieroglyphs-Regular.ttf` | font used to render canonical glyphs | — |

The Hand-drawn dataset is **one source per class** (e.g. all A1 = source `498`),
so a Gardiner class is a coherent "style group" (see One-DM mapping).
Beware `gardiner_hieroglyphs_with_unicode_hex.json`: its `unicode_hex` column is
corrupted (e.g. A10–A13 all "1300") — derive codepoints from the `hieroglyph`
char field instead (prep_dataset.py does).

## 3. How the pieces fit

(Operational front-ends live in `../pipelines/` — this is the tooling view.)

**Generation:**
- *Procedural route* (CPU, zero training): `scripts/skeleton_simplify.py`
  (canonical glyph → clean centerline strokes) → `scripts/handwriting_augment.py`
  (elastic wobble, pen width/pressure, pen lifts). Driven by
  `pipelines/generation/procedural_engine.py`.
- *Learned route*: **One-DM** (one-shot diffusion handwriting mimicker,
  `One-DM/`) fine-tuned on the Hand-drawn dataset via
  `pipelines/generation/prep_dataset.py` + `One-DM/train.py`; sampling via
  `One-DM/generate_hiero.py`. Mapping: writer id = Gardiner class, content =
  canonical glyph (32×32), style/target = hand-drawn instances.

**Matching:** `pipelines/matching/` — metric-learning encoder over handwriting
+ heavily-augmented canonical glyphs; recognition = cosine nearest-prototype
against an index of canonical embeddings (+ optional handwriting centroids).

## 4. Open suggestions (updated after the pipeline build)

1. **Run the HPC jobs** — One-DM fine-tune (~200 epochs, the quality gate) and
   the bigger matcher (`resnet34@160`, 100 epochs). Commands in the READMEs.
2. **Style-conditioning study**: after fine-tuning, probe how well One-DM
   transfers an unseen writer's style (`--style-dir` with pjb samples held out).
3. **FID/style metric** to complement the matcher-recognizability score when
   picking generation checkpoints.
4. **App feedback loop**: collect user drawings from the demo/app as new
   `<CLASS>/*.png` trees → matcher `--resume` fine-tune; rare-class synthesis
   from Pipeline 1.
5. If stroke order / vector output ever becomes a requirement, the deleted
   InkSight stack (offline→online ink) is the tool to restore from the original
   copy.

## 5. Quick start (today, on this box)
```bash
# generate handwriting for three signs (CPU, ~seconds)
python3 pipelines/generation/generate.py --signs A1,D21,N35 --n 4
# match a drawing against all 769 signs (uses the T4-smoke model)
cd pipelines/matching && ./.venv/bin/python match.py --ckpt runs/smoke/best.pt \
    --index runs/smoke/index.npz --image <drawing.png>
```

## 6. Software inventory (what's in this directory)

Goal split: **dictionary / matching** → `pipelines/matching/`; **handwriting
generation** → `pipelines/generation/` (procedural + One-DM).

- **`pipelines/`** — the two product pipelines + `smoke_results/` evidence.
  *Docs:* `pipelines/README.md` and per-pipeline READMEs.
- **`One-DM/`** — one-shot **diffusion handwriting generator** (the learned
  generation engine; upstream code + our backwards-compatible extensions:
  charset/index/content-size overrides, `generate_hiero.py` harness).
  - *Pros:* one-shot style mimicry, SOTA handwriting. *Cons:* needs fine-tune
    (GPU); heavy. *Docs:* https://github.com/dailenson/One-DM ·
    https://arxiv.org/abs/2409.04004 · `One-DM/HIERO_WORKFLOW.md`
  - Weights kept: `model_zoo/One-DM-ckpt.pt`, `model_zoo/sd-v1-5-vae`.
- **`misc/scripts/`** — procedural tools used by both pipelines:
  `skeleton_simplify.py` (glyph → clean strokes), `handwriting_augment.py`
  (strokes → human-looking raster; also the matcher's canonical-domain
  augmentation).
- **`misc/env/` + `misc/.tools/` (uv)** — portable, reset-proof envs +
  `misc/resetup.sh`. *Docs:* `RESETUP.md` + §1 above.
- **`misc/OneDM.pdf`** — the One-DM paper (reference).
