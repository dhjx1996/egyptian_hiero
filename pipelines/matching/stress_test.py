"""Reviewer-grade stress tests for the matching encoder (beyond evaluate.py).

Measures how retrieval holds up under input corruption a real drawn/scanned query
would suffer, plus latency and the worst confusions. Runs against the trained
encoder + prototype index; queries come from the held-out handwriting split.

    ./.venv/bin/python stress_test.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
        --val-split runs/a100/val_split.json --device cuda --limit 1200 \
        --json-out runs/a100/stress.json
"""
import argparse, json, os, random, time
from collections import Counter, defaultdict

import numpy as np
import cv2

from match import Matcher
from hieromatch.data import load_gray


# ---------------------------------------------------------------- corruptions
def c_clean(g, rng):
    return g

def c_rotate(g, rng, deg=20):
    a = rng.choice([-deg, deg])
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
    return cv2.warpAffine(g, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)

def c_blur(g, rng):
    s = max(g.shape) / 40.0                 # scale blur to image size
    return cv2.GaussianBlur(g, (0, 0), s)

def c_noise(g, rng, sigma=35):
    return np.clip(g.astype(np.float32) + rng.normal(0, sigma, g.shape), 0, 255).astype(np.uint8)

def c_occlude(g, rng, frac=0.28):
    h, w = g.shape
    ph, pw = int(h * frac), int(w * frac)
    y = int(rng.integers(0, max(1, h - ph))); x = int(rng.integers(0, max(1, w - pw)))
    out = g.copy(); out[y:y + ph, x:x + pw] = 255       # erase a chunk of ink
    return out

def c_lowres(g, rng, px=32):
    h, w = g.shape
    small = cv2.resize(g, (max(1, px * w // max(h, w)), max(1, px * h // max(h, w))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def c_thin(g, rng):                          # dilate = thin ink (dark-on-white)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(g, k)

def c_dropstroke(g, rng, n_cuts=3):
    """Missing strokes: sever the drawing with white slashes (pen skips) and, when
    the glyph has separable parts, delete one connected component outright."""
    out = g.copy()
    ink = (out < 200).astype(np.uint8)
    n_cc, cc = cv2.connectedComponents(ink)
    if n_cc > 2:                                          # background + >=2 parts
        sizes = np.bincount(cc.ravel())[1:]               # per-component ink pixels
        minor = [i + 1 for i, s in enumerate(sizes) if s <= 0.5 * sizes.sum()]
        if minor:                                         # drop a stroke, not the glyph
            out[cc == int(rng.choice(minor))] = 255
    ys, xs = np.where(out < 200)
    if len(xs):
        w_line = max(2, max(g.shape) // 24)
        for _ in range(n_cuts):
            i, j = rng.integers(0, len(xs), 2)
            cv2.line(out, (int(xs[i]), int(ys[i])), (int(xs[j]), int(ys[j])), 255, w_line)
    return out

def c_wobble(g, rng, alpha=None, sigma=4.0):
    """Shaky-hand strokes: strong short-wavelength elastic displacement."""
    h, w = g.shape
    alpha = alpha or max(h, w) / 12.0
    dx = cv2.GaussianBlur((rng.random((h, w)).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(g, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=255)

CORRUPTIONS = {
    "clean": c_clean, "rotate20": c_rotate, "blur": c_blur, "noise": c_noise,
    "occlude28": c_occlude, "lowres32": c_lowres, "thin": c_thin,
    "dropstroke": c_dropstroke, "wobble": c_wobble,
}


def topk_hits(matcher, pairs, corrupt, rng, top=10):
    hits = {1: 0, 5: 0, 10: 0}; n = 0
    conf = Counter()
    for path, truth in pairs:
        if truth not in matcher._known:
            continue
        g = corrupt(load_gray(path), rng)
        r = matcher.match(g, top=top)
        n += 1
        rank = next((i for i, h in enumerate(r) if h["label"] == truth), None)
        if rank is not None:
            for k in (1, 5, 10):
                if rank < k: hits[k] += 1
        if not r or r[0]["label"] != truth:
            conf[(truth, r[0]["label"] if r else "NONE")] += 1
    return n, hits, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=1200, help="sampled queries per corruption")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    m = Matcher(args.ckpt, args.index, device=args.device)
    m._known = set(m.labels)

    v = json.load(open(args.val_split))
    allpairs = [(e["path"], e["class"]) for e in v["val"] if e["class"] in m._known]
    rng = random.Random(args.seed)
    rng.shuffle(allpairs)
    pairs = allpairs[:args.limit] if args.limit else allpairs
    print(f"[stress] device={args.device} prototypes={len(m.labels)} "
          f"val-usable={len(allpairs)} sampled={len(pairs)}")

    report = {"device": args.device, "n": len(pairs), "corruptions": {}}
    nrng = np.random.default_rng(args.seed)
    print(f"\n{'corruption':12s} {'top1':>7s} {'top5':>7s} {'top10':>7s}")
    for name, fn in CORRUPTIONS.items():
        n, hits, conf = topk_hits(m, pairs, fn, nrng)
        row = {k: (hits[k] / n if n else 0.0) for k in (1, 5, 10)}
        report["corruptions"][name] = {"n": n, **{f"top{k}": row[k] for k in (1, 5, 10)}}
        print(f"{name:12s} {row[1]:7.3f} {row[5]:7.3f} {row[10]:7.3f}")
        if name == "clean":
            report["top_confusions"] = [{"truth": t, "pred": p, "count": c}
                                        for (t, p), c in conf.most_common(12)]

    # latency (clean queries, warm)
    lat_pairs = pairs[:300]
    for p, _ in lat_pairs[:10]:
        m.match(load_gray(p))                              # warmup
    t0 = time.perf_counter()
    for p, _ in lat_pairs:
        m.match(load_gray(p))
    ms = 1000 * (time.perf_counter() - t0) / max(1, len(lat_pairs))
    report["latency_ms_per_query"] = round(ms, 2)
    print(f"\n[latency] {ms:.2f} ms/query on {args.device} (n={len(lat_pairs)}, incl. file load+preprocess)")

    print("\n[top confusions on clean queries]  truth -> pred (count)")
    for c in report.get("top_confusions", [])[:12]:
        print(f"  {c['truth']:8s} -> {c['pred']:8s}  ({c['count']})")

    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=1)
        print(f"\n[stress] json -> {args.json_out}")


if __name__ == "__main__":
    main()
