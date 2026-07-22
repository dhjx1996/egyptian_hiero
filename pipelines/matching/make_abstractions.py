"""Generate a stick-figure "abstraction" bank from canonical glyphs.

Reuses misc/scripts/skeleton_simplify: skeletonize each canonical OUTLINE to its
centerline, trace strokes, keep a FRACTION of the longest strokes (a stroke
budget) -> a sparse skeletal rendering -- the way people redraw figurative signs
as stick figures (e.g. A39 human-between-giraffes -> loop-head figure + single-
line legs; see A39_abstracted.jpg). This is the training/index bridge for the
"abstraction gap": abstract queries land near an abstract prototype instead of a
detailed one they no longer resemble.

Output: <out>/lvl<frac>/<CLASS>.png flat dirs -- consumable directly as
  build_index.py --canonical <dir>       (one abstract prototype per class/level)
  train_encoder.py --abstract <root>     (extra canonical-style training source)

    ./.venv/bin/python make_abstractions.py --fracs 0.4,0.7 --workers 24
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "misc", "scripts"))
import skeleton_simplify as ss  # noqa: E402

D_CANON = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")


def abstract_one(args):
    src, outs, fracs, size, thickness = args
    try:
        ink, size = ss.to_ink(src, size)
        strokes = ss.strokes_of(ink, size)
    except Exception as e:
        return src, f"ERR {e}"
    if not strokes:
        return src, "SKIP no-strokes"
    for frac, out in zip(fracs, outs):
        n = max(3, round(frac * len(strokes)))
        cv2.imwrite(out, ss.render(strokes, size, n, thickness))
    return src, f"ok {len(strokes)}str"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", default=D_CANON)
    ap.add_argument("--out", default=os.path.join(REPO, "hiero_data", "abstractions"))
    ap.add_argument("--fracs", default="1.0",
                    help="stroke-budget fractions, one level dir each. 1.0 keeps ALL traced "
                         "strokes (still skeletonized+simplified: outline->centerline, the core "
                         "'draw it as single lines' transform). Lower fracs drop the SHORTEST "
                         "strokes first, which deletes small salient parts of compound figures "
                         "(e.g. the human in A39) -> semantically wrong renders; avoid <~0.8.")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(",")]
    lvl_dirs = [os.path.join(args.out, f"lvl{f:g}") for f in fracs]
    for d in lvl_dirs:
        os.makedirs(d, exist_ok=True)

    files = sorted(f for f in os.listdir(args.canonical) if f.lower().endswith(".png"))
    jobs = [(os.path.join(args.canonical, f),
             [os.path.join(d, f) for d in lvl_dirs], fracs, args.size, args.thickness)
            for f in files]

    n_ok = n_skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for src, status in ex.map(abstract_one, jobs):
            if status.startswith("ok"):
                n_ok += 1
            else:
                n_skip += 1
                print(f"[abstract] {os.path.basename(src)}: {status}")
    print(f"[abstract] {n_ok} glyphs x {len(fracs)} levels -> {args.out} ({n_skip} skipped)")


if __name__ == "__main__":
    main()
