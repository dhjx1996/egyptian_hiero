"""
Build a complete One-DM fine-tuning dataset from ANY symbol-handwriting corpus.

Generalization contract (works for Egyptian hieroglyphs, Mayan glyphs, any script):
  --handwriting  <root>/<CLASS>/*.png   real handwritten samples, one folder per symbol
  --canonical    <dir>/<CLASS>.png      canonical glyph per symbol (font render, scan, ...)
  --unicode-json (optional)             [{"gardiner_num": CLASS, "unicode_hex": "13000"}, ...]
                                        classes without a real codepoint are auto-assigned
                                        Private-Use-Area codepoints (U+E000+), so scripts
                                        with no Unicode block still work.

One-DM models a symbol dataset as: writer id (wid) = symbol class, content = the
canonical glyph bitmap, style/target = hand-drawn instances of that class. Training
teaches "render canonical X with hand-drawn texture" (see One-DM/HIERO_WORKFLOW.md).

Outputs (all under --onedm-root, default <repo>/One-DM):
  data/<name>/IAM64-new/{train,test}/<wid>/*.png     style/target images (64px tall)
  data/<name>/IAM64_laplace/{train,test}/<wid>/*.png Laplacian high-freq maps
  data/<name>_train.txt, data/<name>_test.txt        index files: "<wid>,<stem> <char>"
  data/<name>_content.pickle                         content bitmaps (default 32x32)
  data/<name>_letters.txt                            charset (one char per class)
  data/<name>_corpus.txt                             all class chars (for test.py)
  data/<name>_wid_map.json                           class <-> wid/char/aspect map
  configs/<name>.yml, configs/<name>_train.yml       generation + training configs

NOTE wids are numeric (0..N-1) because IAMDataset training does int(wid); the map
back to class names lives in <name>_wid_map.json.

Run with One-DM's venv (needs numpy/cv2/PIL), from anywhere:
    One-DM/.venv/bin/python pipelines/generation/prep_dataset.py \
        --handwriting "hiero_data/Hand-drawn Hieroglyph Dataset" \
        --canonical hiero_data/archaeohack-starterpack/data/utf-pngs \
        --unicode-json hiero_data/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
        --name hiero
"""
import argparse
import datetime
import json
import os
import pickle
import random
import re
import shutil
import sys

import numpy as np
import cv2
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAP_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
STYLE_LEN = 352            # One-DM's max style width (data_loader.loader.style_len)
PUA_START = 0xE000         # BMP Private Use Area for classes without a codepoint


# ---------------------------------------------------------------- image ops
def to_gray_on_white(path):
    """RGBA/RGB/L image -> uint8 grayscale, dark ink on white background."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    return np.array(im.convert("L"))


def crop_to_ink(g, pad=4, ink_thresh=200):
    ys, xs = np.where(g < ink_thresh)
    if len(xs) == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + 1 + pad)
    return g[y0:y1, x0:x1]


def style_image(g, height, min_w=16, max_w=STYLE_LEN):
    """Crop -> resize to `height` keeping aspect -> pad width up to a multiple of 16.
    Width % 16 == 0 is REQUIRED for training: the VAE downsamples /8 and the UNet
    halves once more, so odd latent widths break the decoder skip connections."""
    c = crop_to_ink(g)
    if c is None:
        return None
    h, w = c.shape
    nw = max(1, round(w * height / h))
    out = cv2.resize(c, (min(nw, max_w), height), interpolation=cv2.INTER_AREA)
    target_w = min(max_w, max(min_w, ((out.shape[1] + 15) // 16) * 16))
    if out.shape[1] != target_w:
        canvas = np.full((height, target_w), 255, np.uint8)
        x0 = max(0, (target_w - out.shape[1]) // 2)
        canvas[:, x0:x0 + out.shape[1]] = out[:, :target_w]
        out = canvas
    return out


def laplace_map(g):
    lap = np.abs(cv2.filter2D(g.astype(np.float32), -1, LAP_KERNEL))
    m = lap.max()
    if m > 0:
        lap = lap / m * 255.0
    return lap.astype(np.uint8)


def content_bitmap(g, size, ink_thresh=200):
    """Canonical glyph image -> size x size float32 {0,1} bitmap (1 = ink), centered,
    aspect-preserving (same convention as One-DM's unifont pickle)."""
    c = crop_to_ink(g, pad=0, ink_thresh=ink_thresh)
    if c is None:
        return None, None
    gh, gw = c.shape
    aspect = gw / gh
    scale = (size - 2) / max(gh, gw)
    nw, nh = max(1, round(gw * scale)), max(1, round(gh * scale))
    small = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    mat = (canvas < ink_thresh).astype(np.float32)
    return mat, aspect


def sanitize_stem(stem):
    s = re.sub(r"[\s,]+", "_", stem)
    return s or "img"


# ---------------------------------------------------------------- codepoints
def assign_codepoints(classes, unicode_json):
    """Map each class to a unique unicode codepoint. Prefers the literal glyph char
    ('hieroglyph' field) over 'unicode_hex' (the hex column is corrupted in the
    Gardiner JSON, e.g. A10-A13 all say '1300'); classes without a resolvable real
    codepoint get sequential Private-Use-Area assignments. Collisions -> PUA, so
    the charset is always one unique char per class."""
    known = {}
    if unicode_json:
        for e in json.load(open(unicode_json, encoding="utf-8")):
            key = e.get("gardiner_num") or e.get("class") or e.get("name")
            if not key:
                continue
            cp, ch = None, e.get("hieroglyph") or e.get("char") or ""
            if isinstance(ch, str) and len(ch) == 1:
                cp = ord(ch)
            else:
                hexcode = e.get("unicode_hex") or e.get("codepoint")
                if hexcode:
                    try:
                        cp = int(str(hexcode), 16)
                    except ValueError:
                        pass
            if cp:
                known[str(key)] = cp
    cps, used, pua = {}, set(), []
    next_pua = PUA_START
    for c in classes:
        cp = known.get(c)
        if cp is None or cp in used:
            while next_pua in used:
                next_pua += 1
            cp = next_pua
            pua.append(c)
        cps[c] = cp
        used.add(cp)
    return cps, pua


# ---------------------------------------------------------------- configs
GEN_CFG = """MODEL:
  STYLE_ENCODER_LAYERS: 3
  NUM_IMGS: 15
  IN_CHANNELS: 4
  OUT_CHANNELS: 4
  NUM_RES_BLOCKS: 1
  NUM_HEADS: 4
  EMB_DIM: 512
SOLVER:
  BASE_LR: 0.0001
  EPOCHS: 1
  WARMUP_ITERS: 20000
  TYPE: AdamW
  GRAD_L2_CLIP: 5.0
TRAIN:
  TYPE: train
  IMS_PER_BATCH: 16
  SNAPSHOT_BEGIN: 100000
  SNAPSHOT_ITERS: 100000
  VALIDATE_BEGIN: 100000
  VALIDATE_ITERS: 100000
  SEED: 1001
  IMG_H: 64
  IMG_W: 64
TEST:
  TYPE: test
  IMS_PER_BATCH: 16
  IMG_H: 64
  IMG_W: 64
DATA_LOADER:
  NUM_THREADS: 4
  IAMGE_PATH: ./data/{name}/IAM64-new
  STYLE_PATH: ./data/{name}/IAM64-new
  LAPLACE_PATH: ./data/{name}/IAM64_laplace
"""

TRAIN_CFG = """# Fine-tune One-DM on the '{name}' symbol dataset (built by prep_dataset.py).
# Entry point: train.py --one_dm model_zoo/One-DM-ckpt.pt (recon + writer-NCE losses;
# train_finetune.py is NOT used -- its frozen OCR/CTC loss is Latin-specific).
OUTPUT_DIR: Saved/{name}
MODEL:
  STYLE_ENCODER_LAYERS: 3
  NUM_IMGS: 15
  IN_CHANNELS: 4
  OUT_CHANNELS: 4
  NUM_RES_BLOCKS: 1
  NUM_HEADS: 4
  EMB_DIM: 512
SOLVER:
  BASE_LR: {lr}
  EPOCHS: {epochs}
  WARMUP_ITERS: 20000
  TYPE: AdamW
  GRAD_L2_CLIP: 5.0
TRAIN:
  TYPE: train
  IMS_PER_BATCH: {batch}
  SNAPSHOT_BEGIN: 0            # epochs (train.py path snapshots per-epoch)
  SNAPSHOT_ITERS: {snap}
  VALIDATE_BEGIN: 0
  VALIDATE_ITERS: {snap}
  SEED: 1001
  IMG_H: 64
  IMG_W: 64
TEST:
  TYPE: test
  IMS_PER_BATCH: {batch}
  IMG_H: 64
  IMG_W: 64
DATA_LOADER:
  NUM_THREADS: {threads}
  IAMGE_PATH: ./data/{name}/IAM64-new
  STYLE_PATH: ./data/{name}/IAM64-new
  LAPLACE_PATH: ./data/{name}/IAM64_laplace
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handwriting", required=True, help="root of <CLASS>/*.png handwriting tree")
    ap.add_argument("--canonical", default="", help="dir of <CLASS>.png canonical glyphs")
    ap.add_argument("--unicode-json", default="", help="optional class->codepoint json")
    ap.add_argument("--name", default="hiero", help="dataset name (output prefix)")
    ap.add_argument("--onedm-root", default=os.path.join(REPO, "One-DM"))
    ap.add_argument("--content-size", type=int, default=32,
                    help="content bitmap size; 16 and 32 are drop-in for the pretrained model")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-per-class", type=int, default=4)
    ap.add_argument("--limit-classes", type=int, default=0, help="first N classes only (smoke test)")
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--lr", type=float, default=5e-5, help="generated train config LR")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--snapshot-every", type=int, default=10, help="epochs between checkpoints")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hw_root, name = args.handwriting, args.name
    data_dir = os.path.join(args.onedm_root, "data")
    cfg_dir = os.path.join(args.onedm_root, "configs")
    style_root = os.path.join(data_dir, name, "IAM64-new")
    lap_root = os.path.join(data_dir, name, "IAM64_laplace")

    classes = sorted(d for d in os.listdir(hw_root) if os.path.isdir(os.path.join(hw_root, d)))
    if args.limit_classes:
        classes = classes[:args.limit_classes]

    # ---- pass 1: gather usable files per class
    files_of, dropped = {}, []
    for c in classes:
        fs = sorted(f for f in os.listdir(os.path.join(hw_root, c)) if f.lower().endswith(".png"))
        if args.limit_per_class:
            fs = fs[:args.limit_per_class]
        if len(fs) >= args.min_per_class:
            files_of[c] = fs
        else:
            dropped.append(c)
    classes = sorted(files_of)
    if not classes:
        sys.exit("no classes with enough samples found")
    print(f"[prep] {len(classes)} classes ({len(dropped)} dropped: <{args.min_per_class} samples)")

    codepoints, pua = assign_codepoints(classes, args.unicode_json)
    if pua:
        print(f"[prep] {len(pua)} classes got Private-Use-Area codepoints (no real Unicode): {pua[:8]}{'...' if len(pua) > 8 else ''}")

    # ---- fresh split dirs (regenerable derived data)
    for root in (style_root, lap_root):
        for split in ("train", "test"):
            p = os.path.join(root, split)
            if os.path.isdir(p):
                shutil.rmtree(p)
            os.makedirs(p)

    # ---- pass 2: convert images, build splits + index lines + content
    train_lines, test_lines, symbols, wid_map = [], [], [], {}
    n_bad = 0
    for wid, c in enumerate(classes):
        char = chr(codepoints[c])
        cdir = os.path.join(hw_root, c)
        fs = list(files_of[c])
        rng.shuffle(fs)
        n = len(fs)
        n_test = max(2, round(n * args.val_frac)) if (n >= 8 and args.val_frac > 0) else 0
        split_of = {f: ("test" if i < n_test else "train") for i, f in enumerate(fs)}

        seen, n_written = set(), {"train": 0, "test": 0}
        for f in fs:
            g = to_gray_on_white(os.path.join(cdir, f))
            img = style_image(g, args.height)
            if img is None:
                n_bad += 1
                continue
            stem = sanitize_stem(os.path.splitext(f)[0])
            while stem in seen:
                stem += "_x"
            seen.add(stem)
            split = split_of[f]
            for root, arr in ((style_root, img), (lap_root, laplace_map(img))):
                d = os.path.join(root, split, str(wid))
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, stem + ".png"), arr)
            (train_lines if split == "train" else test_lines).append(f"{wid},{stem} {char}")
            n_written[split] += 1

        # content bitmap: canonical glyph preferred, else the first handwriting sample
        mat, aspect, src = None, None, "canonical"
        if args.canonical:
            cpth = os.path.join(args.canonical, c + ".png")
            if os.path.isfile(cpth):
                mat, aspect = content_bitmap(to_gray_on_white(cpth), args.content_size)
        if mat is None:
            src = "handwriting-fallback"
            mat, aspect = content_bitmap(to_gray_on_white(os.path.join(cdir, files_of[c][0])), args.content_size)
        if mat is None:
            sys.exit(f"class {c}: no usable content source")
        symbols.append({"idx": [codepoints[c]], "mat": mat, "class": c})
        wid_map[c] = {"wid": wid, "char": char, "codepoint": f"{codepoints[c]:X}",
                      "aspect": round(float(aspect), 4), "content_src": src,
                      "n_train": n_written["train"], "n_test": n_written["test"]}

    # classes whose test split ended up too thin for style-pair sampling: keep their
    # lines in train only (loader samples 2 style refs per item)
    test_ok = {c for c in classes if wid_map[c]["n_test"] >= 2}
    test_lines = [ln for ln in test_lines if classes[int(ln.split(",")[0])] in test_ok]

    # ---- write outputs
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    def wtxt(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    wtxt(os.path.join(data_dir, f"{name}_train.txt"), train_lines)
    wtxt(os.path.join(data_dir, f"{name}_test.txt"), test_lines)
    with open(os.path.join(data_dir, f"{name}_content.pickle"), "wb") as f:
        pickle.dump(symbols, f)
    wtxt(os.path.join(data_dir, f"{name}_letters.txt"),
         ["".join(chr(codepoints[c]) for c in classes)])
    wtxt(os.path.join(data_dir, f"{name}_corpus.txt"), [wid_map[c]["char"] for c in classes])
    meta = {"name": name, "created": datetime.date.today().isoformat(),
            "handwriting": os.path.abspath(hw_root),
            "canonical": os.path.abspath(args.canonical) if args.canonical else "",
            "content_size": args.content_size, "height": args.height,
            "n_classes": len(classes), "n_train": len(train_lines), "n_test": len(test_lines),
            "pua_classes": pua, "dropped_classes": dropped}
    with open(os.path.join(data_dir, f"{name}_wid_map.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "classes": wid_map}, f, ensure_ascii=False, indent=1)

    with open(os.path.join(cfg_dir, f"{name}.yml"), "w") as f:
        f.write(GEN_CFG.format(name=name))
    with open(os.path.join(cfg_dir, f"{name}_train.yml"), "w") as f:
        f.write(TRAIN_CFG.format(name=name, lr=args.lr, epochs=args.epochs,
                                 batch=args.batch_size, snap=args.snapshot_every,
                                 threads=args.threads))

    print(f"[prep] wrote {len(train_lines)} train / {len(test_lines)} test images "
          f"({n_bad} unreadable/blank skipped) for {len(classes)} classes")
    print(f"[prep] content: {args.content_size}x{args.content_size} "
          f"({sum(1 for c in classes if wid_map[c]['content_src'] != 'canonical')} handwriting-fallback)")
    print(f"[prep] outputs under {data_dir}/{name}* and {cfg_dir}/{name}*.yml")
    print(f"[prep] next: see pipelines/generation/README.md (fine-tune + generate)")


if __name__ == "__main__":
    main()
