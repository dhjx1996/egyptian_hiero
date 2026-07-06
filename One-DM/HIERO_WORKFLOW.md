# One-DM → Handwritten Hieroglyphs: Workflow & Adaptation Notes

> **Superseded as the runbook** (2026-07-06): the operational pipeline now lives
> in `../pipelines/generation/README.md` — `prep_dataset.py` (full training
> dataset from any corpus), `train.py --one_dm` fine-tuning (T4-smoke-verified),
> and `generate_hiero.py` (single-GPU sampling harness). This file remains for
> One-DM internals / background. The box now has a T4; prep scripts below were
> the first-generation (inference-only) versions.

---

## 1. What One-DM is (and how it conditions)

One-Shot Diffusion Mimicker (ECCV 2024, `dailenson/One-DM`). A **latent diffusion**
model that generates a **64-px-tall handwritten word image** from:

| Input | Branch | In this repo |
|-------|--------|--------------|
| **Content** = "what to write" | per-character **16×16 Unifont bitmap** stack, `data/<content_type>.pickle` (keyed by `ord(char)`); charset = module-level `letters` in `data_loader/loader.py` | `ContentData.get_content()` |
| **Style** = "whose hand" | **one** grayscale reference image, 64-px tall, dark-on-white | `Random_StyleIAMDataset` |
| **Style high-freq** | **Laplacian** of the style image (kernel `[[0,1,0],[1,-4,1],[0,1,0]]`) — the paper's key idea | same loader, `LAPLACE_PATH` |
| Noise | random latent `(B,4,H/8,W/8)` | `torch.randn` |

Decoder: a UNet (`models/unet.py`, `EMB_DIM=512`, ~ the 512-ch model) denoises in
the **Stable-Diffusion-v1.5 VAE** latent space; the VAE decodes latents → image.

## 2. Environment (done)

```bash
cd One-DM && bash setup_env.sh        # -> ./One-DM/.venv (CPU torch + diffusers)
```
Persistent (lives on NFS, survives JupyterHub resets), separate from the main
scripts env because One-DM's stack conflicts with the modern one. Verified:
`torch 2.x+cpu`, `models.unet.UNetModel` builds, `models.diffusion.Diffusion`
constructs, loaders import. Training extras: `requirements-train.txt` (GPU only).

## 3. Weights (download on the GPU machine)

```bash
./.venv/bin/python download_weights.py            # prints sources + gdown commands
./.venv/bin/python download_weights.py --vae      # pre-fetch the SD-1.5 VAE
```
**VAE fix:** the default `runwayml/stable-diffusion-v1-5` was deleted from HF.
Use `--stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5` (verified
live). Put `One-DM-ckpt.pt`, `vae_HTR138.pth`, `RN18_class_10400.pth` in `model_zoo/`.

## 4. Hieroglyph data-prep (tooling set up + smoke-tested)

```bash
# (a) content: render Gardiner glyphs into Unifont-format bitmaps
./.venv/bin/python prepare_hiero_content.py \
    --json ../hiero_data/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --font "../hiero_data/archaeohack-starterpack/lib/font/NotoSansEgyptianHieroglyphs-Regular.ttf" \
    --size 16 --out data/hiero_content.pickle --letters-out data/hiero_letters.txt

# (b) style+laplace: convert the Hand-drawn dataset (wid = Gardiner class)
./.venv/bin/python prepare_hiero_style.py \
    --src "../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --style-out data/hiero/IAM64-new/test \
    --laplace-out data/hiero/IAM64_laplace/test --height 64
```
Then point `data_loader/loader.py` `letters` at `data/hiero_letters.txt`, set
`content_type='hiero_content'`, and the config `DATA_LOADER` paths at `data/hiero/...`.

## 5. The real gap — why this needs fine-tuning, not just inference

The pretrained checkpoint was trained on **IAM Latin words**. Feeding hieroglyph
content + hand-drawn style **zero-shot will not be faithful** — the UNet has never
seen these glyph shapes. Expect to **fine-tune / retrain**. Two known issues to
solve first:

1. **Content resolution.** 16×16 is far too coarse for hieroglyphs — the A1
   "seated man" reduces to **~4 ink pixels** (vs. legible Latin letters). Either
   raise `--size` *and* adapt the content encoder (it assumes 16×16 in
   `loader.get_symbols`/`collate_fn_` and the content path in `models/`), or
   accept the content image as a weak class-hint and lean on a class embedding.
2. **Aspect ratio.** One-DM expects wide word strips (`style_len=352`, prefers
   width > 128). Single signs are ~64–130 px wide. Options: horizontally
   concatenate several instances of a sign into a style strip; or relax the width
   logic in `Random_StyleIAMDataset.get_style_ref`.

### Suggested training mapping (for the GPU phase — not yet built)
- **wid (writer)** = Gardiner class. **content** = canonical glyph (utf-png or the
  rendered bitmap). **style refs / target** = hand-drawn instances of that class.
- Pair construction per step: content = canonical(X); style = one hand-drawn X;
  target = another hand-drawn X. This teaches "render canonical X in hand-drawn
  texture," pooling intra-class variation as style.
- Hold out `me-sign-examples-pjb` (6 signs, single author "pjb") as a genuine
  **single-writer** style-transfer probe.
- Start from the pretrained checkpoint (`train_finetune.py`) rather than scratch.

## 6. Running inference later (GPU)

`test.py` is multi-GPU + CUDA only. For a single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 test.py \
    --cfg configs/IAM64.yml --one_dm model_zoo/One-DM-ckpt.pt \
    --generate_type oov_u --dir Generated/hiero \
    --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5
```
For a CPU smoke test you must additionally patch out `dist.init_process_group` /
`torch.cuda.set_device` and set `--device cpu` (the 512-ch model is heavy — watch
RAM). Treat CPU as debugging only.
```

(See `../PROJECT_NOTES.md` for how this fits the whole pipeline.)
