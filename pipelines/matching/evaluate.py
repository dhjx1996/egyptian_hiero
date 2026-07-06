"""
Evaluate matcher retrieval accuracy (top-1/5/10) against the canonical index.

Two query sources:
  --val-split runs/<name>/val_split.json   held-out handwriting from training
  --probe-dir <root>/<CLASS>/*.png         any labeled tree (e.g. me-sign-examples-pjb)

    ./.venv/bin/python evaluate.py --ckpt runs/run1/best.pt --index runs/run1/index.npz \
        --val-split runs/run1/val_split.json \
        --probe-dir ../../hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
"""
import argparse
import json
import os
from collections import defaultdict

from match import Matcher


def rank_of(hits, truth):
    for i, h in enumerate(hits):
        if h["label"] == truth:
            return i
    return None


def evaluate(matcher, pairs, tag, top=10):
    known = set(matcher.labels)
    hits_at = defaultdict(int)
    n = skipped = 0
    for path, truth in pairs:
        if truth not in known:
            skipped += 1
            continue
        r = rank_of(matcher.match(path, top=top), truth)
        n += 1
        if r is not None:
            for k in (1, 5, 10):
                if r < k:
                    hits_at[k] += 1
    if n:
        print(f"[eval:{tag}] n={n}" + (f" (skipped {skipped} not-in-index)" if skipped else "") +
              f" | top1 {hits_at[1]/n:.3f} | top5 {hits_at[5]/n:.3f} | top10 {hits_at[10]/n:.3f}")
    else:
        print(f"[eval:{tag}] no usable queries")
    return {"n": n, **{f"top{k}": (hits_at[k] / n if n else 0) for k in (1, 5, 10)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--val-split", default="")
    ap.add_argument("--probe-dir", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0, help="cap #val queries (quick check)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    m = Matcher(args.ckpt, args.index, device=args.device)
    report = {}

    if args.val_split:
        v = json.load(open(args.val_split))
        pairs = [(e["path"], e["class"]) for e in v["val"]]
        if args.limit:
            pairs = pairs[:args.limit]
        report["val"] = evaluate(m, pairs, "val-handwriting")

    for pd in args.probe_dir:
        pairs = []
        for c in sorted(os.listdir(pd)):
            d = os.path.join(pd, c)
            if os.path.isdir(d):
                pairs += [(os.path.join(d, f), c) for f in sorted(os.listdir(d))
                          if f.lower().endswith(".png")]
        report[os.path.basename(pd.rstrip("/"))] = evaluate(m, pairs, os.path.basename(pd.rstrip("/")))

    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=1)
        print(f"[eval] json -> {args.json_out}")


if __name__ == "__main__":
    main()
