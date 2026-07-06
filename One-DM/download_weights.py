"""
Helper for fetching One-DM model weights.

One-DM needs three checkpoints in ./model_zoo and a Stable-Diffusion v1.5 VAE:
  - One-DM-ckpt.pt        (the diffusion UNet)        -> --one_dm
  - vae_HTR138.pth        (OCR model, finetune/eval)  -> --ocr_model
  - RN18_class_10400.pth  (ResNet18 style/feat model) -> --feat_model
These are hosted on Google Drive / Baidu / ShiZhi-AI (wisemodel). Google Drive
folder links cannot be fetched without per-file IDs, so the robust path is gdown
(for Drive) or the wisemodel raw URLs. See README.md for the canonical links.

IMPORTANT FIX — the VAE:
  test.py defaults to --stable_dif_path runwayml/stable-diffusion-v1-5, which was
  DELETED from Hugging Face. Use the live community mirror instead:
      stable-diffusion-v1-5/stable-diffusion-v1-5
  (verified reachable; vae/diffusion_pytorch_model.safetensors ~334 MB).
  Pass it directly (diffusers auto-downloads), or pre-snapshot it offline with
  `python download_weights.py --vae`.

Usage:
    ./.venv/bin/python download_weights.py --vae      # pre-fetch VAE locally
    ./.venv/bin/python download_weights.py --help
"""
import argparse, os, sys

VAE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_ZOO = os.path.join(os.path.dirname(__file__), "model_zoo")

# Canonical sources from README.md (Google Drive folder + ShiZhi-AI direct files)
WEIGHTS = {
    "One-DM-ckpt.pt":       "https://wisemodel.cn/models/SCUT-MMPR/One-DM/blob/main/One-DM-ckpt.pt",
    "vae_HTR138.pth":       "https://wisemodel.cn/models/SCUT-MMPR/One-DM/blob/main/vae_HTR138.pth",
    "RN18_class_10400.pth": "https://wisemodel.cn/models/SCUT-MMPR/One-DM/blob/main/RN18_class_10400.pth",
}
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/10KOQ05HeN2kaR2_OCZNl9D_Kh1p8BDaa"


def fetch_vae():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed (it is in requirements-infer.txt).")
    dst = os.path.join(MODEL_ZOO, "sd-v1-5-vae")
    os.makedirs(dst, exist_ok=True)
    print(f"Downloading VAE subfolder from {VAE_REPO} -> {dst}")
    snapshot_download(repo_id=VAE_REPO, allow_patterns=["vae/*"], local_dir=dst)
    print(f"Done. Use:  --stable_dif_path {dst}")


def instructions():
    print("Place these in ./model_zoo (gitignored):")
    for name, url in WEIGHTS.items():
        print(f"  - {name}\n      ShiZhi/wisemodel: {url}")
    print(f"\n  Google Drive folder (all three): {GDRIVE_FOLDER}")
    print("  From Drive with gdown:")
    print("      uv pip install --python ./.venv/bin/python gdown")
    print(f"      ./.venv/bin/python -m gdown --folder {GDRIVE_FOLDER} -O model_zoo")
    print("  (wisemodel 'blob' URLs are HTML pages; use the repo's raw/download link,")
    print("   or clone via its git endpoint, to get the actual binaries.)")
    print(f"\nVAE (SD-1.5) is auto-downloaded by diffusers if you pass:")
    print(f"      --stable_dif_path {VAE_REPO}")
    print("  or run `python download_weights.py --vae` to cache it locally.")


def main():
    ap = argparse.ArgumentParser(description="Fetch One-DM weights.")
    ap.add_argument("--vae", action="store_true", help="pre-download the SD-1.5 VAE locally")
    args = ap.parse_args()
    os.makedirs(MODEL_ZOO, exist_ok=True)
    if args.vae:
        fetch_vae()
    else:
        instructions()


if __name__ == "__main__":
    main()
