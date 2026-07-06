# Pipeline 2 — Handwritten-symbol matching (dictionary app backend)

Match a drawn/handwritten symbol to its canonical glyph (Gardiner sign in
`utf-pngs`, or any other inventory). Architecture chosen for **open-set
generalization**: an encoder embeds images into a metric space; recognition is
cosine nearest-prototype against an **index** of canonical-glyph embeddings.

- add new symbols or whole new scripts → re-run `build_index.py` (**no retraining**);
- more handwriting samples → fine-tune the encoder (`--resume`) and re-index;
- classes with zero handwriting data are still matchable: during training,
  canonical glyphs are augmented with procedural "fake handwriting"
  (elastic/pen-sim from `misc/scripts`), bridging the canonical↔drawn domain gap.

This replaces/upgrades the hackathon prototypes in `translator_app_presentations/`
(team repos: `akhare2007/denominators`, `yuyanghu06/GreenTeamArchaeoHack`,
`hansel7121/Archaeohack-3A1W`, `KhalonManswell/ArchaeoHack-Group-Still-Loading`,
`Cici-Z1ang/ArchaeoHack-Team-404-found`).

Setup (once): `cd pipelines/matching && bash setup_env.sh` (CUDA auto-detected;
also run by `misc/resetup.sh`). All commands below run from `pipelines/matching/`.

**Verified smoke results (this box's T4, 2026-07-06)** — resnet18@112, only
3 epochs (~4 min/epoch), `runs/smoke/`: held-out handwriting retrieval
top-1 **0.905** / top-5 0.964; unseen-writer pjb probe top-1 **0.859** /
top-5 0.973 (n=256). Treat as the floor — the HPC recipe below should push
well past it.

## 1. Train the encoder

```bash
./.venv/bin/python train_encoder.py --name run1 --epochs 30        # T4-friendly
```
Data defaults: handwriting = `hiero_data/Hand-drawn Hieroglyph Dataset`
(`<CLASS>/*.png`), canonical = `utf-pngs` (`<CLASS>.png`); classes = union
(769), so canonical-only classes train via augmentation. Outputs in
`runs/<name>/`: `best.pt` (top val accuracy), `last.pt` (resumable), `classes.json`,
`val_split.json` (deterministic held-out handwriting), `log.txt`.

Key knobs: `--arch resnet18|resnet34|resnet50`, `--size` (input px),
`--embed-dim`, `--canon-repeat` (canonical oversampling), `--p-handwrite`
(pen-sim probability on canonical; costs CPU, helps canonical-only classes),
`--resume runs/<name>/last.pt`.

## 2. Build the index

```bash
./.venv/bin/python build_index.py --ckpt runs/run1/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/run1/val_split.json --out runs/run1/index.npz
```
Prototypes = every canonical glyph (`--canonical` repeatable, default utf-pngs)
+ optional per-class handwriting centroids (recommended: big accuracy boost for
classes with real data). A query is scored against all prototypes, aggregated
per label by max.

## 3. Match / evaluate / demo

```bash
# CLI
./.venv/bin/python match.py --ckpt runs/run1/best.pt --index runs/run1/index.npz \
    --image /path/drawn.png --top 5
# accuracy report (held-out handwriting + the single-writer pjb probe)
./.venv/bin/python evaluate.py --ckpt runs/run1/best.pt --index runs/run1/index.npz \
    --val-split runs/run1/val_split.json \
    --probe-dir ../../hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
# browser demo: draw on a canvas -> live top-k with glyphs + meanings
./.venv/bin/python demo_server.py --ckpt runs/run1/best.pt --index runs/run1/index.npz --port 8787
# (JupyterHub: https://<hub>/user/<you>/proxy/8787/ ; plain box: localhost:8787)
```
Python API for app integration: `from match import Matcher;
Matcher(ckpt, index).match(image, top=5)` → `[{label, score, char, description}]`
(≈15 ms/query on CPU with resnet18@112 — fine for on-device/server dictionary use).
`POST /match {"image": <dataURL>}` on the demo server returns the same JSON.

## 4. Training on the HPC (A100)

```bash
bash setup_env.sh            # auto-installs CUDA torch
./.venv/bin/python train_encoder.py --name a100 --arch resnet34 --size 160 \
    --epochs 100 --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12
./.venv/bin/python build_index.py --ckpt runs/a100/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/a100/val_split.json
./.venv/bin/python evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json --probe-dir <pjb>
```
Single-GPU is plenty (resnet34@160 ≈ minutes/epoch at bs 512; dataloading is
CPU-bound — raise `--workers`). Expect val top-1 well above the T4 smoke run;
the pjb probe measures writer-generalization (its 6 signs: I10 M17 N35 O1 V28 X1).
Deploy = copy `runs/<name>/best.pt` + `index.npz` (a few MB) anywhere.

## 5. Other scripts / languages + growth

- **New script** (e.g. Mayan): train with `--handwriting my_maya/handwriting
  --canonical my_maya/canonical --name maya` (same folder conventions as
  Pipeline 1 §3). No handwriting at all? Canonical-only training still gives a
  usable matcher thanks to the procedural augmentation.
- **Extend an existing model**: append another `--canonical <dir>` at index time
  — new symbols become matchable without retraining (accuracy is best after a
  fine-tune that has seen them, but nearest-prototype degrades gracefully).
- **User-submitted drawings** (the app's feedback loop): save them as
  `<CLASS>/*.png`, add as another handwriting tree, `--resume` fine-tune, re-index.
- **Synthetic boost**: Pipeline 1 outputs are labeled trees — generate variants
  for rare classes and add them as extra handwriting data.

## 6. File inventory

| File | Role |
|---|---|
| `hieromatch/model.py` | encoder (resnet→embedding) + cosine head |
| `hieromatch/data.py` | preprocessing, handwriting-style augmentation, datasets |
| `train_encoder.py` | training loop (AMP, cosine LR, val split, runs/) |
| `build_index.py` | canonical (+centroid) prototype index → `index.npz` |
| `match.py` | CLI + `Matcher` API |
| `evaluate.py` | top-k accuracy on labeled trees / val split |
| `demo_server.py` | stdlib HTTP canvas demo (draw → top-k) |
| `setup_env.sh`, `requirements.txt` | persistent uv venv (CUDA auto) |
