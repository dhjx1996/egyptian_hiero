"""
Build One-DM *style* + *laplace* reference folders from the Hand-drawn
Hieroglyph Dataset.

One-DM's inference (Random_StyleIAMDataset) reads style references from
    <STYLE_PATH>/<wid>/<sample>.png
and matching high-frequency maps from
    <LAPLACE_PATH>/<wid>/<sample>.png
where <wid> is a "writer id". Style images are grayscale, dark ink on light
background, 64px tall, variable width (the loader prefers width > 128).

Mapping chosen for this dataset (see HIERO_WORKFLOW.md for rationale):
  wid = Gardiner class (e.g. "A1"). Each class folder in the Hand-drawn dataset
  comes from a single source and holds ~80 instances of that one sign, so the
  class doubles as a coherent "style/writer" group: content = canonical glyph,
  style refs = hand-drawn instances of (typically) the same glyph.

This converts RGBA drawings -> grayscale-on-white, crops to the ink, resizes to
64px tall, and writes both the style image and a Laplacian high-frequency map
(kernel [[0,1,0],[1,-4,1],[0,1,0]], matching One-DM's style-enhancement input).

Usage (One-DM dir, its venv):
    ./.venv/bin/python prepare_hiero_style.py \
        --src "../hiero_data/Hand-drawn Hieroglyph Dataset" \
        --style-out data/hiero/IAM64-new/test \
        --laplace-out data/hiero/IAM64_laplace/test \
        --height 64 [--limit-classes N]
"""
import argparse, os, glob
import numpy as np
import cv2
from PIL import Image

LAP_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def to_gray_on_white(path):
    """RGBA/RGB/L drawing -> uint8 grayscale, dark ink on white background."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    g = np.array(im.convert("L"))
    return g


def crop_to_ink(g, pad=4, ink_thresh=200):
    ys, xs = np.where(g < ink_thresh)            # ink = dark
    if len(xs) == 0:
        return g
    y0, y1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + 1 + pad)
    return g[y0:y1, x0:x1]


def resize_h(g, height):
    h, w = g.shape
    nw = max(1, round(w * height / h))
    return cv2.resize(g, (nw, height), interpolation=cv2.INTER_AREA)


def laplace_map(g):
    """High-frequency map a la One-DM: |Laplacian|, normalized to 0..255 uint8."""
    lap = cv2.filter2D(g.astype(np.float32), -1, LAP_KERNEL)
    lap = np.abs(lap)
    m = lap.max()
    if m > 0:
        lap = lap / m * 255.0
    return lap.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Hand-drawn Hieroglyph Dataset root")
    ap.add_argument("--style-out", default="data/hiero/IAM64-new/test")
    ap.add_argument("--laplace-out", default="data/hiero/IAM64_laplace/test")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--limit-classes", type=int, default=0, help="process only first N classes (smoke test)")
    ap.add_argument("--limit-per-class", type=int, default=0, help="cap samples per class")
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)))
    if args.limit_classes:
        classes = classes[:args.limit_classes]

    n_imgs = 0
    for wid in classes:
        files = sorted(glob.glob(os.path.join(args.src, wid, "*.png")))
        if args.limit_per_class:
            files = files[:args.limit_per_class]
        s_dir = os.path.join(args.style_out, wid)
        l_dir = os.path.join(args.laplace_out, wid)
        os.makedirs(s_dir, exist_ok=True); os.makedirs(l_dir, exist_ok=True)
        for fp in files:
            g = resize_h(crop_to_ink(to_gray_on_white(fp)), args.height)
            name = os.path.splitext(os.path.basename(fp))[0] + ".png"
            cv2.imwrite(os.path.join(s_dir, name), g)
            cv2.imwrite(os.path.join(l_dir, name), laplace_map(g))
            n_imgs += 1
    print(f"wrote {n_imgs} style+laplace images for {len(classes)} classes")
    print(f"  style   -> {args.style_out}")
    print(f"  laplace -> {args.laplace_out}")


if __name__ == "__main__":
    main()
