"""
Single-GPU (or CPU) One-DM sampling harness for symbol datasets built by
pipelines/generation/prep_dataset.py -- no torchrun/dist, loads the model once,
and gives explicit control over which signs render in which handwriting style.

Unlike test.py (which crosses a whole corpus with every style author), this
generates N variants per requested sign, each from a fresh noise seed and a
randomly chosen style reference image. Style references can come from the
prepped dataset tree or from ANY folder of sample images (--style-dir), and
content can come from the prepped content pickle (--signs/--chars) or from ANY
glyph image on disk (--glyph), so a fine-tuned checkpoint extends to unseen
symbols and unseen writers without re-prepping.

Run from One-DM/ with its venv:
    ./.venv/bin/python generate_hiero.py --cfg configs/hiero.yml \
        --one_dm model_zoo/One-DM-ckpt.pt --stable_dif_path model_zoo/sd-v1-5-vae \
        --content_type hiero_content --letters data/hiero_letters.txt \
        --wid-map data/hiero_wid_map.json --signs A1,D21,N35 --n 4 \
        --out Generated/hiero_harness
"""
import argparse
import json
import os
import random

import numpy as np
import cv2
import torch
import torchvision
from PIL import Image

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import ContentData
from models.unet import UNetModel
from models.diffusion import Diffusion
from diffusers import AutoencoderKL

LAP_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


# ---------------------------------------------------------------- style refs
def to_gray_on_white(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    return np.array(im.convert("L"))


def prep_style(path, height=64, min_w=16, max_w=352):
    g = to_gray_on_white(path)
    ys, xs = np.where(g < 200)
    if len(xs):
        g = g[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5]
    h, w = g.shape
    nw = max(1, round(w * height / h))
    g = cv2.resize(g, (min(nw, max_w), height), interpolation=cv2.INTER_AREA)
    if g.shape[1] < min_w:
        c = np.full((height, min_w), 255, np.uint8)
        x0 = (min_w - g.shape[1]) // 2
        c[:, x0:x0 + g.shape[1]] = g
        g = c
    return g


def style_batch(paths, device):
    """List of image paths -> (B,1,64,W) style + laplace tensors (laplace computed
    on the fly with One-DM's kernel, so any folder of samples works as a style)."""
    imgs = [prep_style(p) for p in paths]
    W = max(im.shape[1] for im in imgs)
    style = np.ones((len(imgs), 1, 64, W), np.float32)
    lap = np.zeros((len(imgs), 1, 64, W), np.float32)
    for i, im in enumerate(imgs):
        style[i, 0, :, :im.shape[1]] = im / 255.0
        l = np.abs(cv2.filter2D(im.astype(np.float32), -1, LAP_KERNEL))
        if l.max() > 0:
            l = l / l.max()
        lap[i, 0, :, :im.shape[1]] = l
    return torch.from_numpy(style).to(device), torch.from_numpy(lap).to(device)


# ---------------------------------------------------------------- content
def glyph_content(path, size, ink_thresh=200):
    """Any glyph image -> (1,1,size,size) content tensor + aspect (external symbols)."""
    g = to_gray_on_white(path)
    ys, xs = np.where(g < ink_thresh)
    if not len(xs):
        raise SystemExit(f"{path}: blank glyph")
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    gh, gw = g.shape
    scale = (size - 2) / max(gh, gw)
    nw, nh = max(1, round(gw * scale)), max(1, round(gh * scale))
    small = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, np.uint8)
    canvas[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = small
    mat = (canvas < ink_thresh).astype(np.float32)
    return torch.from_numpy(1.0 - mat).view(1, 1, size, size), gw / gh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default="configs/hiero.yml")
    ap.add_argument("--one_dm", required=True, help="checkpoint (pretrained or fine-tuned)")
    ap.add_argument("--stable_dif_path", default="model_zoo/sd-v1-5-vae")
    ap.add_argument("--content_type", default="hiero_content", help="content pickle stem in data/")
    ap.add_argument("--letters", default="data/hiero_letters.txt")
    ap.add_argument("--wid-map", default="data/hiero_wid_map.json", help="prep_dataset map (class names, aspects)")
    ap.add_argument("--signs", default="", help="comma-separated class names (e.g. A1,D21) or 'all'")
    ap.add_argument("--chars", default="", help="literal unicode glyph chars instead of class names")
    ap.add_argument("--glyph", action="append", default=[],
                    help="path to ANY glyph image (repeatable); bypasses the content pickle")
    ap.add_argument("--style-root", default="", help="prepped style tree (defaults to the cfg STYLE_PATH)")
    ap.add_argument("--style-split", default="train", choices=["train", "test"])
    ap.add_argument("--style-class", default="same",
                    help="'same' = the sign's own class (self-styled), 'random', or a class/wid name")
    ap.add_argument("--style-dir", default="", help="ANY folder of sample images to use as the style")
    ap.add_argument("--n", type=int, default=4, help="variants per sign")
    ap.add_argument("--width", default="auto", help="'auto' (from glyph aspect) or pixels")
    ap.add_argument("--sampling_timesteps", type=int, default=25)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8, help="max variants per forward pass")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="Generated/hiero_harness")
    args = ap.parse_args()

    cfg_from_file(args.cfg)
    assert_and_infer_cfg()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    wid_map = {}
    if args.wid_map and os.path.isfile(args.wid_map):
        wid_map = json.load(open(args.wid_map, encoding="utf-8"))["classes"]
    char_of = {c: m["char"] for c, m in wid_map.items()}
    wid_of = {c: str(m["wid"]) for c, m in wid_map.items()}
    aspect_of = {c: m.get("aspect") for c, m in wid_map.items()}

    # ---- resolve the work list: (name, char or None, glyph path or None)
    jobs = []
    if args.signs:
        names = sorted(wid_map) if args.signs == "all" else [s.strip() for s in args.signs.split(",") if s.strip()]
        for nme in names:
            if nme not in char_of:
                raise SystemExit(f"sign '{nme}' not in {args.wid_map}")
            jobs.append((nme, char_of[nme], None))
    for ch in args.chars:
        jobs.append((f"U+{ord(ch):X}", ch, None))
    for gp in args.glyph:
        jobs.append((os.path.splitext(os.path.basename(gp))[0], None, gp))
    if not jobs:
        raise SystemExit("nothing to generate: pass --signs, --chars or --glyph")

    # ---- style source
    style_root = args.style_root or os.path.join(cfg.DATA_LOADER.STYLE_PATH, args.style_split)
    def style_paths_for(name):
        if args.style_dir:
            d = args.style_dir
        else:
            cls = args.style_class
            if cls == "same":
                cls = name
            if cls == "random":
                cls = random.choice(os.listdir(style_root))
            d = os.path.join(style_root, wid_of.get(cls, cls))
            if not os.path.isdir(d):
                raise SystemExit(f"no style folder for '{cls}' under {style_root} "
                                 f"(pass --style-dir or --style-class)")
        pool = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".png")]
        if not pool:
            raise SystemExit(f"no .png style refs in {d}")
        return [random.choice(pool) for _ in range(args.n)]

    # ---- model
    print(f"[gen] device={device}  ckpt={args.one_dm}")
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM,
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
                     attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS,
                     context_dim=cfg.MODEL.EMB_DIM).to(device)
    miss, unexp = unet.load_state_dict(torch.load(args.one_dm, map_location="cpu"), strict=False)
    print(f"[gen] loaded unet (missing={len(miss)} unexpected={len(unexp)})")
    unet.eval()
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    diffusion = Diffusion(device=device)

    loader = None
    if any(ch is not None for _, ch, _ in jobs):
        loader = ContentData(content_type=args.content_type, letters_path=args.letters)

    # ---- generate
    content_size = None
    for name, ch, gpath in jobs:
        if gpath:
            csize = content_size or 32
            content1, aspect = glyph_content(gpath, csize)
        else:
            content1 = loader.get_content(ch)          # (1, 1, H, W)
            content_size = content1.shape[-1]
            aspect = aspect_of.get(name) or 1.0
        # width must be a multiple of 16: the VAE /8 plus the UNet's own /2 need
        # an even latent width or the decoder skip-connection cat fails
        if args.width == "auto":
            w = int(np.clip(round(64 * float(aspect) / 16) * 16, 32, 256))
        else:
            w = max(32, (int(args.width) // 16) * 16)

        spaths = style_paths_for(name)
        outd = os.path.join(args.out, name)
        os.makedirs(outd, exist_ok=True)
        done = 0
        with torch.no_grad():
            while done < args.n:
                bpaths = spaths[done:done + args.batch]
                style, lap = style_batch(bpaths, device)
                content = content1.to(device).repeat(len(bpaths), 1, 1, 1)
                x = torch.randn((len(bpaths), 4, 64 // 8, w // 8), device=device)
                imgs = diffusion.ddim_sample(unet, vae, len(bpaths), x, style, lap, content,
                                             args.sampling_timesteps, args.eta)
                for i in range(len(bpaths)):
                    im = torchvision.transforms.ToPILImage()(imgs[i]).convert("L")
                    im.save(os.path.join(outd, f"{name}_v{done + i:02d}.png"))
                done += len(bpaths)
        print(f"[gen] {name}: {args.n} variants ({w}px wide) -> {outd}")

    print(f"[gen] done -> {args.out}")


if __name__ == "__main__":
    main()
