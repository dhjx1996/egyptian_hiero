"""
Procedural handwriting engine: canonical glyph image -> N human-looking variants.

Zero-training route (runs on CPU today): skeletonize the glyph into clean
centerline strokes (misc/scripts/skeleton_simplify), then re-render with elastic
wobble, varying pen width/pressure and pen lifts (misc/scripts/handwriting_augment).
Works for ANY glyph image of any script -- no dataset, no model, no GPU.

Run with the main scripts env (needs cv2/scipy/skimage):
    misc/.venv/bin/python pipelines/generation/procedural_engine.py \
        --signs A1,D21,N35 --n 6 --out misc/outputs/generated/procedural

Importable API: variants(glyph_png, n, ...) -> list of HxW uint8 arrays.
"""
import argparse
import os
import sys

import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "misc", "scripts"))
import skeleton_simplify                                    # noqa: E402
from handwriting_augment import handwrite                   # noqa: E402

DEFAULT_CANONICAL = os.path.join(
    REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")


def variants(glyph_png, n, size=512, seed=0, simplify=True, stroke_budget=99,
             thickness=4, out_size=0):
    """Generate n handwriting-like uint8 (dark-on-white) variants of one glyph."""
    if simplify:
        base = skeleton_simplify.simplify(glyph_png, n=stroke_budget, size=size,
                                          thickness=thickness)
    else:
        base = cv2.imread(glyph_png, cv2.IMREAD_GRAYSCALE)
        if base is None:
            raise SystemExit(f"unreadable image: {glyph_png}")
        h, w = base.shape
        s = size / max(h, w)
        base = cv2.resize(base, (max(1, round(w * s)), max(1, round(h * s))),
                          interpolation=cv2.INTER_AREA)
    outs = []
    for i in range(n):
        rng = np.random.default_rng(seed * 100003 + i)
        v = handwrite(base, rng)
        if out_size:
            v = cv2.resize(v, (out_size, out_size), interpolation=cv2.INTER_AREA)
        outs.append(v)
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signs", default="", help="comma-separated sign names resolved in --canonical-dir, or 'all'")
    ap.add_argument("--glyph", action="append", default=[], help="path to ANY glyph image (repeatable)")
    ap.add_argument("--canonical-dir", default=DEFAULT_CANONICAL)
    ap.add_argument("--n", type=int, default=6, help="variants per glyph")
    ap.add_argument("--out", default=os.path.join(REPO, "misc", "outputs", "generated", "procedural"))
    ap.add_argument("--size", type=int, default=512, help="working canvas size")
    ap.add_argument("--out-size", type=int, default=0, help="resize outputs to this square size (0 = keep)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stroke-budget", type=int, default=99, help="max strokes kept by the simplifier")
    ap.add_argument("--no-simplify", action="store_true",
                    help="skip skeleton_simplify (input is already clean line art)")
    args = ap.parse_args()

    jobs = []
    if args.signs == "all":
        jobs = [(os.path.splitext(f)[0], os.path.join(args.canonical_dir, f))
                for f in sorted(os.listdir(args.canonical_dir)) if f.endswith(".png")]
    elif args.signs:
        for s in args.signs.split(","):
            s = s.strip()
            p = os.path.join(args.canonical_dir, s + ".png")
            if not os.path.isfile(p):
                raise SystemExit(f"no canonical glyph {p}")
            jobs.append((s, p))
    for gp in args.glyph:
        jobs.append((os.path.splitext(os.path.basename(gp))[0], gp))
    if not jobs:
        raise SystemExit("nothing to do: pass --signs or --glyph")

    for name, path in jobs:
        outd = os.path.join(args.out, name)
        os.makedirs(outd, exist_ok=True)
        for i, v in enumerate(variants(path, args.n, size=args.size, seed=args.seed,
                                       simplify=not args.no_simplify,
                                       stroke_budget=args.stroke_budget,
                                       out_size=args.out_size)):
            cv2.imwrite(os.path.join(outd, f"{name}_p{i:02d}.png"), v)
        print(f"[procedural] {name}: {args.n} variants -> {outd}")


if __name__ == "__main__":
    main()
