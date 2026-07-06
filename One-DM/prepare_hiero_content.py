"""
Build a One-DM *content* dictionary for Egyptian hieroglyphs.

One-DM represents the "what to write" (content) as one small binary glyph bitmap
per character, loaded from data/<content_type>.pickle by
data_loader.loader.IAMDataset.get_symbols(). For Latin it ships `unifont.pickle`
(16x16 Unifont bitmaps keyed by ord(char)).

This script produces the hieroglyph equivalent: it renders each Gardiner sign's
Unicode codepoint (U+13000+ block) with the Noto Egyptian Hieroglyphs font into a
size x size binary bitmap, and saves a pickle in the SAME format One-DM expects:
    [ {'idx': [codepoint], 'mat': np.ndarray(size,size) float32 in {0,1}}, ... ]

It also writes a `letters` string (all rendered hieroglyph chars) — One-DM keys its
charset on the module-level `letters` in data_loader/loader.py, which must be
swapped to this string when adapting the model to hieroglyphs.

NOTE on resolution: One-DM's content path is 16x16 (very coarse). Detailed
hieroglyphs are barely legible at 16px; the content image is only a class hint
that the content encoder upsamples. Keep --size 16 for drop-in compatibility, or
raise it AND adjust the content encoder (see HIERO_WORKFLOW.md). An alternative
content source is downsampling the canonical utf-pngs instead of font rendering.

Usage (from the One-DM dir, using its venv):
    ./.venv/bin/python prepare_hiero_content.py \
        --json ../hiero_data/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
        --font ../hiero_data/archaeohack-starterpack/lib/font/NotoSansEgyptianHieroglyphs-Regular.ttf \
        --size 16 --out data/hiero_content.pickle --letters-out data/hiero_letters.txt
"""
import argparse, json, os, pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_glyph(char, font_path, size, supersample=8, ink_thresh=128):
    """Render one unicode char to a size x size binary {0,1} bitmap (1 = ink).
    Renders large then center-fits to the box so glyphs are normalized."""
    big = size * supersample
    # pick a font size that mostly fills the supersampled canvas
    font = ImageFont.truetype(font_path, int(big * 0.8))
    img = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(img)
    # measure and center
    try:
        l, t, r, b = d.textbbox((0, 0), char, font=font)
    except Exception:
        return None
    w, h = r - l, b - t
    if w <= 0 or h <= 0:
        return None
    d.text(((big - w) / 2 - l, (big - h) / 2 - t), char, fill=255, font=font)
    arr = np.array(img)
    if arr.max() == 0:
        return None  # font has no glyph for this codepoint -> skip
    # crop to ink bbox, then fit into a centered square preserving aspect
    ys, xs = np.where(arr > ink_thresh)
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    gh, gw = arr.shape
    scale = (size - 2) / max(gh, gw)          # 1px margin each side
    nw, nh = max(1, round(gw * scale)), max(1, round(gh * scale))
    g = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(g, ((size - nw) // 2, (size - nh) // 2))
    mat = (np.array(canvas) > ink_thresh).astype(np.float32)
    return mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="gardiner_hieroglyphs_with_unicode_hex.json")
    ap.add_argument("--font", required=True, help="NotoSansEgyptianHieroglyphs-Regular.ttf")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", default="data/hiero_content.pickle")
    ap.add_argument("--letters-out", default="data/hiero_letters.txt")
    ap.add_argument("--limit", type=int, default=0, help="render only first N (smoke test)")
    args = ap.parse_args()

    entries = json.load(open(args.json))
    if args.limit:
        entries = entries[:args.limit]

    symbols, chars, skipped = [], [], []
    for e in entries:
        hexcode = e.get("unicode_hex")
        if not hexcode:
            skipped.append(e.get("gardiner_num", "?")); continue
        cp = int(hexcode, 16)
        mat = render_glyph(chr(cp), args.font, args.size)
        if mat is None:
            skipped.append(e.get("gardiner_num", "?")); continue
        symbols.append({"idx": [cp], "mat": mat, "gardiner": e.get("gardiner_num")})
        chars.append(chr(cp))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(symbols, f)
    with open(args.letters_out, "w") as f:
        f.write("".join(chars))
    print(f"rendered {len(symbols)} glyphs @ {args.size}x{args.size} -> {args.out}")
    print(f"letters ({len(chars)} chars) -> {args.letters_out}")
    if skipped:
        print(f"skipped {len(skipped)} (no codepoint/glyph): {skipped[:10]}{'...' if len(skipped)>10 else ''}")


if __name__ == "__main__":
    main()
