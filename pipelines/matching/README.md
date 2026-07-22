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

**Production model: `runs/default`** (symlink → `runs/a100_drop10`, resnet34@160,
100 epochs on the full A100 recipe below + `--p-dropstroke 0.10`). Verified
2026-07-11: held-out top-1 **0.971** / top-5 0.985; unseen-writer pjb probe
top-1 **0.863** / top-5 **0.988** (n=256); corruption robustness (n=800/case) —
clean 0.966, wobble 0.938, occlude28 0.868, dropstroke 0.779, blur 0.455,
lowres32 0.391. Selected over the plain `runs/a100` baseline (pjb 0.895 but
dropstroke only 0.680) as the better robustness/generalization tradeoff — see
`--p-dropstroke` below. To repoint production at a new run: `ln -sfn <name>
runs/default`.

> **2026-07-11 adversarial review** ([`review/REVIEW.md`](review/REVIEW.md)):
> the held-out 0.971 is inflated by dataset leakage (~13 near-duplicate
> variants per source drawing straddle the file-level split; a group-disjoint
> retrain of the same recipe measures **0.935**), the model partly keys on
> scan-frame borders absent from real queries, and app-domain (canvas-style)
> accuracy is ~0.78 top-1 / 0.85 top-5. See the review for evidence,
> reproduction jobs and prioritized recommendations before quoting numbers
> from this README.
>
> **2026-07-22 corrective actions** ([`review/CORRECTIVE_ACTIONS.md`](review/CORRECTIVE_ACTIONS.md)):
> the group-disjoint split is now the **default** (`--file-level-split` to
> reproduce old runs), and three isolated A/B retrains were run against the
> honest baseline (val 0.935 / pjb 0.840). Headline: stick-figure abstraction
> augmentation (`make_abstractions.py` + `--abstract`) lifts unseen-writer pjb
> to **0.973** at −1.5 pt val — candidate next production model. Frame aug: tie
> (baseline already frame-robust, 0.911). Partial-stroke aug (`--p-partial`)
> triples half-drawing robustness (0.11 → 0.35) but regresses ~2 pt elsewhere;
> off by default.

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
`--p-dropstroke` (probability of missing-stroke augmentation — pen-skip
slashes + dropped connected components; production uses 0.10, see §4),
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
# CLI (--ckpt/--index default to runs/default, i.e. the production model)
./.venv/bin/python match.py --image /path/drawn.png --top 5
# accuracy report (held-out handwriting + the single-writer pjb probe)
./.venv/bin/python evaluate.py --ckpt runs/default/best.pt --index runs/default/index.npz \
    --val-split runs/default/val_split.json \
    --probe-dir ../../hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
# corruption robustness (rotation/blur/noise/occlusion/low-res/dropped-strokes/wobble)
./.venv/bin/python stress_test.py --ckpt runs/default/best.pt --index runs/default/index.npz \
    --val-split runs/default/val_split.json
# browser demo: draw on a canvas -> live top-k with glyphs + meanings
./.venv/bin/python demo_server.py --port 8787
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
    --epochs 100 --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12 \
    --p-dropstroke 0.10       # production recipe; see robustness note below
./.venv/bin/python build_index.py --ckpt runs/a100/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/a100/val_split.json
./.venv/bin/python evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json --probe-dir <pjb>
./.venv/bin/python stress_test.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json
ln -sfn a100 runs/default     # promote once eval + stress numbers look good
```
Single-GPU is plenty (resnet34@160 ≈ minutes/epoch at bs 512; dataloading is
CPU-bound — raise `--workers`). Expect val top-1 in the high-0.9s (see the
production numbers above); the pjb probe measures writer-generalization
(its 6 signs: I10 M17 N35 O1 V28 X1).
Deploy = copy `runs/<name>/best.pt` + `index.npz` (a few MB) anywhere, or point
`runs/default` at it.

**`--p-dropstroke` tradeoff** (2026-07-11 sweep, `slurm/matcher_drop*.sbatch`):
missing/severed strokes were the sharpest measured weakness (dropstroke corruption
top-1 0.680 at `--p-dropstroke 0` vs clean 0.969). Sweeping the augmentation
probability trades that off against unseen-writer generalization (pjb):

| `--p-dropstroke` | dropstroke top1 | pjb top1 | pjb top5 | wobble top1 |
|---|---|---|---|---|
| 0 (`a100`) | 0.680 | 0.895 | 0.984 | 0.922 |
| **0.10 (`a100_drop10`, production)** | **0.779** | **0.863** | **0.988** | **0.938** |
| 0.25 (`a100_drop`) | 0.824 | 0.816 | 0.973 | 0.919 |

0.10 was chosen as the best tradeoff (+10pt robustness for -3pt writer
generalization, within noise at n=256); 0.25 over-corrects. Re-sweep if the
deployment's error profile shifts (e.g. mostly novice users → weight dropstroke
higher; mostly archival/expert transcription → weight pjb-style generalization
higher).

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
- **Synthetic boost** (`train_encoder.py --synthetic`, gated by `synth_filter.py`,
  orchestrated by `slurm/synth_feedback.sbatch`): Pipeline 1 outputs are labeled
  trees — quality-gate them (failure gate: bad ink, confident-impostor,
  off-class, canonical-clone, near-duplicate — all scored by a FROZEN baseline
  matcher so the loop can't self-amplify) and add as capped extra training data.
  **2026-07-10 run verdict: REJECT** — held-out/corruption metrics held, but
  unseen-writer pjb top-1 dropped 0.895→0.797 (the encoder keyed on One-DM's
  synthetic stroke texture). Kept as a re-runnable pipeline for when the
  generator's fidelity improves, not as a standing recommendation; always check
  the acceptance report (real-data regression check) before promoting a
  synthetic-boosted run.

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
