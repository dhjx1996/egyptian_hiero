# Pipeline 1 — Handwriting generation

Generate human-looking handwriting samples for **any symbol**: every Gardiner
sign in `utf-pngs`, any other Egyptian hieroglyph, or any other script's glyphs
(Mayan, cuneiform, ...). Two engines behind one front-end:

| Engine | What it is | Needs | Quality |
|---|---|---|---|
| `procedural` | skeletonize the canonical glyph → re-stroke with elastic wobble, pen-width/pressure variation, pen lifts (`misc/scripts`) | CPU only, zero training | good "clean hand" look, unlimited variants, works today |
| `onedm` | One-DM latent-diffusion handwriting mimicker, conditioned on (canonical glyph, style reference image) | GPU + **fine-tune** | learned realism + style transfer from real writers; near-blank before fine-tuning |

All commands run from the repo root unless noted. `generate.py` itself needs
only stock `python3` (it dispatches into the right venvs).

## 1. Quick start (works now, CPU)

```bash
python3 pipelines/generation/generate.py --signs A1,D21,N35 --n 6
# -> misc/outputs/generated/procedural/<SIGN>/<SIGN>_pNN.png
python3 pipelines/generation/generate.py --glyph /path/to/any_glyph.png --n 6
python3 pipelines/generation/generate.py --signs all --n 4       # all 769 signs
```
Useful knobs: `--stroke-budget` (max strokes kept), `--out-size 200` (resize
outputs), `--no-simplify` (input is already line art), `--seed`.

## 2. One-DM route

### 2.1 Prepare the training dataset (once per corpus)

```bash
One-DM/.venv/bin/python pipelines/generation/prep_dataset.py \
    --handwriting "hiero_data/Hand-drawn Hieroglyph Dataset" \
    --canonical   hiero_data/archaeohack-starterpack/data/utf-pngs \
    --unicode-json hiero_data/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --name hiero
```

This maps the corpus into One-DM's world — writer id = symbol class, content =
canonical glyph (32×32, 4× the detail of One-DM's stock 16×16 with identical
model shapes), style/target = hand-drawn instances — and writes everything
training needs under `One-DM/`:
`data/hiero/IAM64-new/{train,test}/<wid>/*.png` (+ `IAM64_laplace` mirrors),
`data/hiero_{train,test}.txt`, `data/hiero_content.pickle`,
`data/hiero_letters.txt`, `data/hiero_corpus.txt`, `data/hiero_wid_map.json`
(class ↔ wid/char/aspect), and `configs/hiero{,_train}.yml`.

Notes baked into the prep (don't fight them):
- style/target widths are padded to **multiples of 16** — the VAE (/8) plus the
  UNet's own downsample need even latent widths or skip connections crash;
- wid folders are **numeric** (training does `int(wid)`); classes live in the wid map;
- codepoints come from the JSON's `hieroglyph` char (its `unicode_hex` column is
  corrupted); classes without real codepoints get Private-Use-Area chars;
- classes with <4 usable samples are dropped (style sampling needs pairs).

### 2.2 Fine-tune (HPC, A100)

Zero-shot output from the IAM-Latin checkpoint is faint/blank on hieroglyphs —
fine-tuning is **required** for real quality. Entry point is `train.py`
initialized from the pretrained checkpoint (diffusion-MSE + writer-NCE losses;
do **not** use `train_finetune.py` — its frozen OCR/CTC auxiliary loss is
Latin-specific):

```bash
cd One-DM
# single A100
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/torchrun --nproc_per_node=1 --master_port=29500 \
  train.py --cfg configs/hiero_train.yml --one_dm model_zoo/One-DM-ckpt.pt \
  --stable_dif_path model_zoo/sd-v1-5-vae \
  --content_type hiero_content --letters data/hiero_letters.txt \
  --train_txt data/hiero_train.txt --test_txt data/hiero_test.txt
# N A100s: --nproc_per_node=N (per-GPU batch = TRAIN.IMS_PER_BATCH)
```

- Config (`configs/hiero_train.yml`, generated; regenerate with different
  `--lr/--epochs/--batch-size/--snapshot-every` or edit in place): defaults
  LR 5e-5, 200 epochs, batch 96/GPU (~24 GB; A100-80G takes 192+).
  ~40k train images → ~420 steps/epoch at batch 96 → very roughly 6–12 h for
  200 epochs on one A100; checkpoints + hieroglyph sample previews land in
  `Saved/hiero/.../model/<epoch>-ckpt.pt` and `.../sample/` every
  `SNAPSHOT_ITERS` epochs — judge visually from ~epoch 20 and stop when happy.
- Avoid `--log` (collides with torchrun's `--log-dir`); logs default to "debug".
- Resume/continue: pass the latest `Saved/.../<N>-ckpt.pt` as `--one_dm`.
- T4 smoke (verified on this box): prep a tiny set with
  `--name hiero_smoke --limit-classes 12 --limit-per-class 10 --epochs 2 --batch-size 8 --snapshot-every 1`
  and run the same command with the `hiero_smoke_*` files (~2 min).

### 2.3 Generate

```bash
# via the front-end (from repo root); use your fine-tuned ckpt
python3 pipelines/generation/generate.py --engine onedm \
    --signs A1,D21 --n 8 --ckpt One-DM/Saved/hiero/**/model/190-ckpt.pt
# in the style of a specific writer's samples (any folder of images):
python3 pipelines/generation/generate.py --engine onedm --signs A1 \
    --style-dir hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb/I10 \
    --ckpt <ckpt>
# a symbol the model never saw (content comes straight from the image):
python3 pipelines/generation/generate.py --engine onedm --glyph /path/new_glyph.png --ckpt <ckpt>
```
`--style-class same|random|<CLASS>` picks style refs from the prepped tree.
Widths are aspect-aware per sign (from the wid map / glyph bbox, /16-aligned);
override with the harness's `--width`. Direct harness (more knobs: `--chars`,
`--style-split`, `--batch`, `--eta`): `cd One-DM && ./.venv/bin/python
generate_hiero.py --help`.

Draft mode without fine-tuning: pass `--ckpt One-DM/model_zoo/One-DM-ckpt.pt`
(pretrained IAM) — mechanically identical, visually poor on glyphs.

## 3. Other scripts / languages (e.g. Mayan)

Nothing above is hieroglyph-specific. Recipe:
1. Lay out data: `my_maya/handwriting/<GLYPH>/*.png` (≥4 per glyph; more = better)
   and `my_maya/canonical/<GLYPH>.png`. No Unicode? Skip `--unicode-json` — PUA
   chars are auto-assigned.
2. `prep_dataset.py --handwriting my_maya/handwriting --canonical my_maya/canonical --name maya`
3. Fine-tune as §2.2 with the `maya_*` files/config (start from `One-DM-ckpt.pt`,
   or from your hieroglyph ckpt — closer domain, likely faster).
4. Generate: `generate.py --engine onedm --dataset-name maya --signs ... --ckpt <maya ckpt>`.
   The procedural engine needs no training at all:
   `generate.py --signs GLYPH1 --canonical-dir my_maya/canonical`.

More handwriting later? Drop the new samples into the tree, re-run prep (fresh
split dirs are rebuilt), continue training from your last checkpoint.

## 4. Evaluating generated handwriting

Outputs are labeled trees, so Pipeline 2's matcher scores recognizability directly:

```bash
cd pipelines/matching && ./.venv/bin/python evaluate.py \
  --ckpt runs/<run>/best.pt --index runs/<run>/index.npz \
  --probe-dir ../../misc/outputs/generated/onedm
```
High top-k = generated signs still read as their class. Complement with a
style/FID-type measure against real handwriting when picking checkpoints.

## 5. File inventory

| File | Role |
|---|---|
| `generate.py` | engine dispatcher (stdlib python3) |
| `procedural_engine.py` | skeleton+wobble engine (runs in `misc/.venv`) |
| `prep_dataset.py` | corpus → One-DM training dataset (runs in `One-DM/.venv`) |
| `One-DM/generate_hiero.py` | single-GPU sampling harness (no torchrun) |
| `One-DM/train.py` + `configs/<name>_train.yml` | fine-tuning entry |
| `One-DM/data/<name>_wid_map.json` | class ↔ wid/char/aspect + prep metadata |

Upstream One-DM edits (all backwards-compatible, Latin defaults unchanged):
`data_loader/loader.py` (charset/index-file/content-size overrides), `train.py`
(pass-through CLI args), `trainer/trainer.py` (validation previews follow the
dataset charset).
