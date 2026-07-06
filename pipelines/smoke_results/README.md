# Smoke-test results (Tesla T4, this box, 2026-07-06)

Evidence that both pipelines work end-to-end. Everything here was produced by
the commands in the pipeline READMEs; "smoke" = mechanics + a first quality
signal, NOT final quality (that needs the HPC/A100 runs).

## generation/ — Pipeline 1

- **`procedural/{A1,D21,N35}/*.png`** — 4 variants each from the zero-training
  procedural engine (canonical glyph → skeleton → pen wobble/pressure/lifts).
  This is current *production-usable* output.
  `python3 pipelines/generation/generate.py --signs A1,D21,N35 --n 4`
- **`onedm_smoke_finetune/{A1,A15,G17}/*.png`** — One-DM diffusion samples from
  a checkpoint fine-tuned for only 2 epochs on a 12-class subset. Faint/rough
  **by design of the test**: it proves the full loop (prep → fine-tune →
  checkpoint → aspect-aware sampling) runs, not visual quality. G17 was NOT in
  the fine-tune set — its content came straight from the canonical PNG
  (`--glyph`), demonstrating the unseen-symbol path.
- **`onedm_training_previews/*.png`** — the validation previews One-DM's trainer
  wrote after epochs 0/1 (hieroglyph charset wired into training). Early-epoch
  quality, as expected.

Expected after the real A100 fine-tune (~200 epochs on the full 565-class /
39k-image dataset already prepped in `One-DM/data/hiero*`): samples that look
like the Hand-drawn dataset, per sign, in controllable writer styles.

## matching/ — Pipeline 2

Model: `pipelines/matching/runs/smoke/best.pt` — resnet18@112, **3 epochs**
(~4 min/epoch on the T4). Index: 769 canonical + 565 handwriting-centroid
prototypes (`runs/smoke/index.npz`).

- **`predictions_pjb.png`** — one drawing per class from the *unseen writer*
  probe (`me-sign-examples-pjb`): query vs. top-5 predicted canonical glyphs,
  correct hit outlined green. 5/6 top-1; the miss (O1) is still in the top-5.
- **`predictions_val.png`** — 8 held-out Hand-drawn queries: 8/8 top-1.
- **`predictions.json`** — the same predictions with scores, machine-readable.
- **`eval.json`** — full accuracy sweep (`evaluate.py`):
  - held-out handwriting (n=800): **top-1 0.905**, top-5 0.964, top-10 0.976
  - pjb unseen writer (n=256): **top-1 0.859**, top-5 0.973, top-10 0.980

Also verified live (not capturable as files): `demo_server.py` canvas page —
POSTing a pjb N35 drawing returned N35 "Water ripple" 𓈖 at rank 1 (cos 0.71).

## Reproduce

```bash
# generation evidence
python3 pipelines/generation/generate.py --signs A1,D21,N35 --n 4 --out <dir>
# one-dm smoke fine-tune + sampling: pipelines/generation/README.md §2.2 (T4 smoke note)
# matching evidence (after a training run)
cd pipelines/matching
./.venv/bin/python evaluate.py --ckpt runs/smoke/best.pt --index runs/smoke/index.npz \
    --val-split runs/smoke/val_split.json \
    --probe-dir ../../hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
```
The trained smoke model + index live in `pipelines/matching/runs/smoke/`
(best.pt, index.npz, eval.json, log.txt, val_split.json) and power the demo:
`./.venv/bin/python demo_server.py --ckpt runs/smoke/best.pt --index runs/smoke/index.npz`.
