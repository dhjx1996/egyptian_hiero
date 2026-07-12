"""
Adversarial probe suite for the matching pipeline (2026-07 review).

Runs AGAINST the production model (runs/default) and quantifies the failure
hypotheses raised by code/data inspection:

  P1  train/val leakage      dataset ships ~13 augmented near-duplicates per
                             source drawing; the file-level random split puts
                             siblings on both sides -> held-out 0.971 suspect
  P2  scan-frame reliance    every dataset image carries a cell/frame border;
                             canonical + real app queries are frameless
  P3  canvas-domain gap      accuracy on val images re-rendered as app-canvas
                             style strokes (skeleton + uniform pen width)
  P4  novel corruptions      corruptions NOT mirrored in training augmentation
                             (mirror/rot90/invert/half-glyph/jpeg/grid...) --
                             contrast with the trained-on ones in stress_test
  P5  open-set / garbage     score distributions for scribbles, shapes, latin
                             letters, blanks, and true-class-removed impostors;
                             threshold table for the app's "no match" state
  P6  prototype-kind bias    centroid-vs-canonical winner mix; index without
                             centroids; canonical-only classes probed with
                             procedural handwriting; acc vs train-count
  P7  score calibration      reliability of cosine scores (val + pjb) for the
                             app's confidence display
  P8  test-time fixes        TTA, mean-vs-max aggregation, blur/lowres
                             remediation -- measured, not speculative
  P9  latency reality check  production config on CPU (the README quotes
                             resnet18@112 numbers)

Each probe is independent and crash-isolated; results are flushed to
--json-out after every probe. Examples are saved under review/examples/.

    ./.venv/bin/python review/adversarial_probe.py --smoke     # quick sanity
    ./.venv/bin/python review/adversarial_probe.py             # full (GPU)
"""
import argparse
import base64
import glob
import io
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(MATCH))
sys.path.insert(0, MATCH)

from match import Matcher                                    # noqa: E402
from hieromatch.data import (load_gray, crop_ink, letterbox,  # noqa: E402
                             preprocess_array)

D_HAND = os.path.join(REPO, "hiero_data", "Hand-drawn Hieroglyph Dataset")
D_CANON = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")
D_PJB = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "me-sign-examples-pjb")


# ================================================================ batch matcher
class ProbeMatcher:
    """Vectorized wrapper around Matcher with the exact same per-label max
    aggregation, plus masking (exclude labels / exclude prototype kinds)."""

    def __init__(self, ckpt, index, device):
        self.m = Matcher(ckpt, index, device=device)
        self.device = self.m.device
        self.size = self.m.meta["size"]
        self.emb = self.m.emb                                 # (P, D)
        self.labels = self.m.labels
        self.kinds = np.array(self.m.kinds)
        self.uniq = sorted(set(self.labels))
        self.lab2i = {l: i for i, l in enumerate(self.uniq)}
        self.proto_lab = torch.tensor([self.lab2i[l] for l in self.labels],
                                      device=self.device)
        self.is_centroid = torch.tensor(self.kinds == "centroid", device=self.device)

    @torch.no_grad()
    def embed_arrays(self, arrays, bs=256):
        out = []
        for i in range(0, len(arrays), bs):
            x = torch.stack([preprocess_array(g, self.size) for g in arrays[i:i + bs]])
            out.append(self.m.encoder.embed(x.to(self.device)))
        return torch.cat(out) if out else torch.zeros(0, self.emb.shape[1], device=self.device)

    @torch.no_grad()
    def label_scores(self, Q, drop_centroids=False, agg="max"):
        """Q (N,D) normalized -> (N,L) per-label aggregated cosine."""
        sims = Q @ self.emb.T                                 # (N, P)
        if drop_centroids:
            sims = sims.masked_fill(self.is_centroid[None, :], -2.0)
        N, L = sims.shape[0], len(self.uniq)
        idx = self.proto_lab[None, :].expand(N, -1)
        if agg == "max":
            out = torch.full((N, L), -2.0, device=self.device)
            out.scatter_reduce_(1, idx, sims, reduce="amax", include_self=True)
        else:                                                 # mean over label's prototypes
            valid = (sims > -1.5).float()
            ssum = torch.zeros(N, L, device=self.device).scatter_add_(1, idx, sims * valid)
            cnt = torch.zeros(N, L, device=self.device).scatter_add_(1, idx, valid)
            out = torch.where(cnt > 0, ssum / cnt.clamp(min=1), torch.full_like(ssum, -2.0))
        return out

    def topk(self, arrays, k=10, **kw):
        """-> (top_labels [N][k], top_scores (N,k) np, winner_kind [N])"""
        Q = self.embed_arrays(arrays)
        ls = self.label_scores(Q, **kw)
        sc, li = ls.topk(k, dim=1)
        top_labels = [[self.uniq[j] for j in row] for row in li.cpu().numpy()]
        # winner kind: best prototype within the winning label
        sims = (Q @ self.emb.T).cpu().numpy()
        kinds = []
        for n, labs in enumerate(top_labels):
            pidx = [i for i, l in enumerate(self.labels) if l == labs[0]]
            kinds.append(self.kinds[pidx[int(np.argmax(sims[n, pidx]))]])
        return top_labels, sc.cpu().numpy(), kinds


def hitrates(top_labels, truths, ks=(1, 5, 10)):
    n = len(truths)
    hits = {k: 0 for k in ks}
    for labs, t in zip(top_labels, truths):
        for k in ks:
            if t in labs[:k]:
                hits[k] += 1
    return {f"top{k}": round(hits[k] / max(n, 1), 4) for k in ks} | {"n": n}


# ================================================================ image helpers
def group_id(path):
    """<src>_<drawing>_<variant>_<CLASS>-x-y.png -> '<src>_<drawing>' (the source
    drawing all ~13 augmented variants derive from); fall back to full stem."""
    stem = os.path.splitext(os.path.basename(path))[0]
    toks = stem.split("_")
    return "_".join(toks[:2]) if len(toks) >= 4 else stem


def strip_frame(g):
    """Remove scan-cell border: connected ink components that touch the image
    border AND span most of a dimension. Reverts if it would erase the glyph."""
    ink = (g < 200).astype(np.uint8)
    tot = int(ink.sum())
    if tot == 0:
        return g, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    h, w = g.shape
    out = g.copy()
    removed = 0
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        touches = x <= 2 or y <= 2 or (x + bw) >= w - 2 or (y + bh) >= h - 2
        longish = bw >= 0.85 * w or bh >= 0.85 * h
        if touches and longish:
            out[lab == k] = 255
            removed += 1
    if (out < 200).sum() < 0.15 * tot:                        # nuked the glyph -> revert
        return g, 0
    return out, removed


def add_frame(g):
    out = g.copy()
    h, w = out.shape
    t = max(2, min(h, w) // 80)
    cv2.rectangle(out, (1, 1), (w - 2, h - 2), 20, t)
    return out


def canvas_sim(g, pen_r=4):
    """Re-render as an app-canvas drawing: frame stripped, skeletonized,
    uniform pen width, clean white background."""
    from skimage.morphology import skeletonize
    s, _ = strip_frame(g)
    c = letterbox(s, 320)
    sk = skeletonize(c < 200)
    out = np.full((320, 320), 255, np.uint8)
    out[sk] = 0
    if pen_r > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pen_r - 1, 2 * pen_r - 1))
        out = cv2.erode(out, k)                               # erode = thicken dark ink
    return out


# --------------------------------------------------------- novel corruptions
def c_hflip(g, rng): return g[:, ::-1].copy()
def c_rot45(g, rng):
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.choice([-45, 45])), 0.75)
    return cv2.warpAffine(g, M, (w, h), borderValue=255)
def c_rot90(g, rng): return np.rot90(g, int(rng.choice([1, 3]))).copy()
def c_invert(g, rng): return (255 - g).copy()
def c_thick2(g, rng):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.erode(g, k, iterations=2)
def c_jpeg(g, rng):
    ok, buf = cv2.imencode(".jpg", g, [int(cv2.IMWRITE_JPEG_QUALITY), 18])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else g
def c_grid(g, rng):
    out = g.copy()
    for y in range(0, out.shape[0], 28):
        out[y:y + 1, :] = np.minimum(out[y:y + 1, :], 205)
    for x in range(0, out.shape[1], 28):
        out[:, x:x + 1] = np.minimum(out[:, x:x + 1], 205)
    return out
def _half(g, which):
    ys, xs = np.where(g < 200)
    if len(xs) == 0:
        return g
    out = g.copy()
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if which == "left":
        out[:, (x0 + x1) // 2:] = 255
    elif which == "top":
        out[(y0 + y1) // 2:, :] = 255
    return out
def c_half_left(g, rng): return _half(g, "left")
def c_half_top(g, rng): return _half(g, "top")

NOVEL = {"hflip": c_hflip, "rot45": c_rot45, "rot90": c_rot90, "invert": c_invert,
         "thick2": c_thick2, "jpeg18": c_jpeg, "gridlines": c_grid,
         "half_left": c_half_left, "half_top": c_half_top}


# --------------------------------------------------------- garbage generators
def gen_scribble(rng, size=360):
    g = np.full((size, size), 255, np.uint8)
    for _ in range(rng.integers(1, 5)):
        pt = rng.uniform(60, size - 60, 2)
        vel = rng.uniform(-8, 8, 2)
        pts = []
        for _ in range(rng.integers(20, 80)):
            vel = 0.9 * vel + rng.normal(0, 3, 2)
            pt = np.clip(pt + vel, 10, size - 10)
            pts.append(pt.copy())
        cv2.polylines(g, [np.array(pts, np.int32)], False, 0, int(rng.integers(3, 9)))
    return g

def gen_shape(rng, size=360):
    g = np.full((size, size), 255, np.uint8)
    t = int(rng.integers(3, 8))
    kind = rng.integers(0, 4)
    c, r = (size // 2, size // 2), int(rng.integers(60, 140))
    if kind == 0:
        cv2.circle(g, c, r, 0, t)
    elif kind == 1:
        cv2.rectangle(g, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), 0, t)
    elif kind == 2:
        pts = np.array([[c[0], c[1] - r], [c[0] - r, c[1] + r], [c[0] + r, c[1] + r]])
        cv2.polylines(g, [pts], True, 0, t)
    else:
        ang = np.linspace(0, 4 * np.pi, 200)
        pts = np.stack([c[0] + ang * r / 13 * np.cos(ang),
                        c[1] + ang * r / 13 * np.sin(ang)], 1).astype(np.int32)
        cv2.polylines(g, [pts], False, 0, t)
    return g

def gen_letters(rng, n):
    from PIL import Image, ImageDraw, ImageFont
    cands = (glob.glob("/usr/share/fonts/**/DejaVuSans*.ttf", recursive=True)
             + glob.glob("/usr/share/fonts/**/LiberationSans*.ttf", recursive=True))
    if not cands:
        return []
    font = ImageFont.truetype(cands[0], 220)
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZabdeghkqrt2345789&?@#"
    out = []
    for _ in range(n):
        im = Image.new("L", (360, 360), 255)
        ImageDraw.Draw(im).text((80, 40), chars[int(rng.integers(0, len(chars)))],
                                fill=0, font=font)
        a = np.array(im.rotate(float(rng.uniform(-15, 15)), fillcolor=255))
        out.append(a)
    return out


# ================================================================ probes
class Report:
    def __init__(self, path):
        self.path, self.d = path, {}

    def put(self, key, value):
        self.d[key] = value
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f, indent=1, default=float)
        os.replace(tmp, self.path)
        print(f"[probe] wrote section '{key}' -> {self.path}", flush=True)


def load_val(val_split):
    v = json.load(open(val_split))
    return [(e["path"], e["class"]) for e in v["val"]]


def train_files_by_class(val_paths):
    """All dataset files minus val -> the training pool (per class)."""
    val = set(os.path.abspath(p) for p in val_paths)
    out = {}
    for c in sorted(os.listdir(D_HAND)):
        d = os.path.join(D_HAND, c)
        if not os.path.isdir(d):
            continue
        fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".png")]
        out[c] = [p for p in fs if os.path.abspath(p) not in val]
    return out


def p1_leakage(pm, pairs, train_by_class, lim_emb):
    groups_per_class, variants_per_group = [], []
    for c, fs in train_by_class.items():
        gs = Counter(group_id(p) for p in fs)
        if gs:
            groups_per_class.append(len(gs))
            variants_per_group += list(gs.values())
    train_groups = {c: set(group_id(p) for p in fs) for c, fs in train_by_class.items()}
    leaked = [(p, t) for p, t in pairs if group_id(p) in train_groups.get(t, set())]
    clean = [(p, t) for p, t in pairs if group_id(p) not in train_groups.get(t, set())]

    # pixel-space nearest same-class train neighbour (64px letterbox, GPU L2)
    dev = pm.device
    cache = {}
    def small(p):
        return torch.from_numpy(letterbox(load_gray(p), 64).astype(np.float32) / 255.0)
    rng = random.Random(0)
    sample = pairs if len(pairs) <= lim_emb else rng.sample(pairs, lim_emb)
    dists, same_group_nn = [], 0
    for p, t in sample:
        tr = train_by_class.get(t, [])
        if not tr:
            continue
        if t not in cache:
            cache[t] = (torch.stack([small(q) for q in tr]).to(dev).reshape(len(tr), -1), tr)
        T, tr_paths = cache[t]
        q = small(p).to(dev).reshape(1, -1)
        d = ((T - q) ** 2).mean(1).sqrt()
        j = int(d.argmin())
        dists.append(float(d[j]))
        if group_id(tr_paths[j]) == group_id(p):
            same_group_nn += 1
    dists = np.array(dists)
    qs = {f"q{q}": round(float(np.quantile(dists, q / 100)), 4)
          for q in (5, 10, 25, 50, 75, 90)}

    # accuracy on leaked vs group-clean val subsets
    def acc(pp):
        if not pp:
            return {"n": 0}
        arrs = [load_gray(p) for p, _ in pp]
        tl, _, _ = pm.topk(arrs, k=10)
        return hitrates(tl, [t for _, t in pp])
    rng.shuffle(leaked)
    res = {
        "groups_per_class_median": float(np.median(groups_per_class)),
        "variants_per_group_median": float(np.median(variants_per_group)),
        "variants_per_group_mean": round(float(np.mean(variants_per_group)), 2),
        "val_n": len(pairs), "val_leaked_n": len(leaked), "val_group_clean_n": len(clean),
        "val_leaked_frac": round(len(leaked) / max(len(pairs), 1), 4),
        "pixel_nn_dist_quantiles": qs,
        "pixel_nn_same_group_frac": round(same_group_nn / max(len(dists), 1), 4),
        "pixel_nn_dist_lt_0.05_frac": round(float((dists < 0.05).mean()), 4),
        "pixel_nn_dist_lt_0.08_frac": round(float((dists < 0.08).mean()), 4),
        "acc_leaked_subset": acc(leaked[:1500]),
        "acc_group_clean_subset": acc(clean),
    }
    return res


def p2_frames(pm, sample, pjb_pairs, exdir):
    orig = [load_gray(p) for p, _ in sample]
    truths = [t for _, t in sample]
    stripped, n_removed = [], 0
    for g in orig:
        s, r = strip_frame(g)
        stripped.append(s)
        n_removed += (r > 0)
    tl0, _, _ = pm.topk(orig, k=10)
    tl1, _, _ = pm.topk(stripped, k=10)
    pjb_g = [load_gray(p) for p, _ in pjb_pairs]
    pjb_t = [t for _, t in pjb_pairs]
    tl2, _, _ = pm.topk(pjb_g, k=10)
    tl3, _, _ = pm.topk([add_frame(g) for g in pjb_g], k=10)
    for i in (0, 1, 2):
        cv2.imwrite(os.path.join(exdir, f"p2_val{i}_orig.png"), orig[i])
        cv2.imwrite(os.path.join(exdir, f"p2_val{i}_stripped.png"), stripped[i])
    cv2.imwrite(os.path.join(exdir, "p2_pjb_framed.png"), add_frame(pjb_g[0]))
    return {"val_frame_detected_frac": round(n_removed / len(orig), 4),
            "val_clean": hitrates(tl0, truths),
            "val_frame_stripped": hitrates(tl1, truths),
            "pjb_clean": hitrates(tl2, pjb_t),
            "pjb_with_frame_added": hitrates(tl3, pjb_t)}


def p3_canvas(pm, sample, exdir):
    truths = [t for _, t in sample]
    out = {}
    for pen in (2, 4, 8):
        arrs = [canvas_sim(load_gray(p), pen_r=pen) for p, _ in sample]
        tl, _, _ = pm.topk(arrs, k=10)
        out[f"pen_r{pen}"] = hitrates(tl, truths)
        if pen == 4:
            for i in (0, 1, 2):
                cv2.imwrite(os.path.join(exdir, f"p3_canvas_sim{i}.png"), arrs[i])
    return out


def p4_novel(pm, sample):
    truths = [t for _, t in sample]
    rng = np.random.default_rng(0)
    out = {}
    for name, fn in NOVEL.items():
        arrs = [fn(load_gray(p), rng) for p, _ in sample]
        tl, _, _ = pm.topk(arrs, k=10)
        out[name] = hitrates(tl, truths)
    return out


def p5_openset(pm, sample, n_garbage):
    rng = np.random.default_rng(0)
    garbage = {"scribble": [gen_scribble(rng) for _ in range(n_garbage)],
               "shape": [gen_shape(rng) for _ in range(n_garbage)],
               "blank": [np.full((360, 360), 255, np.uint8) for _ in range(8)]}
    letters = gen_letters(rng, n_garbage)
    if letters:
        garbage["latin"] = letters

    arrs = [load_gray(p) for p, _ in sample]
    truths = [t for _, t in sample]
    Q = pm.embed_arrays(arrs)
    ls = pm.label_scores(Q)
    sc, li = ls.topk(2, dim=1)
    top1 = [pm.uniq[j] for j in li[:, 0].cpu().numpy()]
    s1 = sc[:, 0].cpu().numpy()
    margins = (sc[:, 0] - sc[:, 1]).cpu().numpy()
    genuine = s1[[a == b for a, b in zip(top1, truths)]]
    gmarg = margins[[a == b for a, b in zip(top1, truths)]]

    # impostor: true label's prototypes masked out
    ti = torch.tensor([pm.lab2i.get(t, -1) for t in truths], device=pm.device)
    sims = Q @ pm.emb.T
    mask = pm.proto_lab[None, :] == ti[:, None]
    N, L = sims.shape[0], len(pm.uniq)
    outm = torch.full((N, L), -2.0, device=pm.device)
    outm.scatter_reduce_(1, pm.proto_lab[None, :].expand(N, -1),
                         sims.masked_fill(mask, -2.0), reduce="amax", include_self=True)
    impostor = outm.max(1).values.cpu().numpy()

    neg_scores = {}
    for name, gs in garbage.items():
        Qg = pm.embed_arrays(gs)
        neg_scores[name] = pm.label_scores(Qg).max(1).values.cpu().numpy()
    neg_scores["impostor"] = impostor

    def dist(a):
        return {"n": len(a), "mean": round(float(np.mean(a)), 4),
                "q10": round(float(np.quantile(a, .1)), 4),
                "q50": round(float(np.quantile(a, .5)), 4),
                "q90": round(float(np.quantile(a, .9)), 4)}

    def auroc(pos, neg):
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.r_[pos, neg]
        order = np.argsort(s)
        ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        rp = ranks[y == 1].sum()
        npos, nneg = len(pos), len(neg)
        return round(float((rp - npos * (npos + 1) / 2) / (npos * nneg)), 4)

    table = []
    for t in np.arange(0.30, 0.92, 0.04):
        row = {"threshold": round(float(t), 2),
               "genuine_kept": round(float((genuine >= t).mean()), 3)}
        for name, ns in neg_scores.items():
            row[f"{name}_passed"] = round(float((ns >= t).mean()), 3)
        table.append(row)
    return {"genuine_top1_score": dist(genuine), "genuine_margin": dist(gmarg),
            "negatives": {k: dist(v) for k, v in neg_scores.items()},
            "auroc_vs": {k: auroc(genuine, v) for k, v in neg_scores.items()},
            "threshold_table": table}


def p6_prototype_bias(pm, sample, pjb_pairs, train_by_class, n_canon_only, exdir):
    arrs = [load_gray(p) for p, _ in sample]
    truths = [t for _, t in sample]
    tl, sc, kinds = pm.topk(arrs, k=10)
    correct = [l[0] == t for l, t in zip(tl, truths)]
    kw = Counter(k for k, c in zip(kinds, correct) if c)
    kwrong = Counter(k for k, c in zip(kinds, correct) if not c)

    tlnc, _, _ = pm.topk(arrs, k=10, drop_centroids=True)
    pjb_g = [load_gray(p) for p, _ in pjb_pairs]
    pjb_t = [t for _, t in pjb_pairs]
    tlp, _, _ = pm.topk(pjb_g, k=10)
    tlpnc, _, _ = pm.topk(pjb_g, k=10, drop_centroids=True)

    # canonical-only classes probed with the SAME procedural pen-sim used in training
    sys.path.insert(0, os.path.join(REPO, "misc", "scripts"))
    import handwriting_augment as hw
    hand_classes = set(train_by_class)
    canon_only = sorted(set(pm.uniq) - hand_classes)
    rng = np.random.default_rng(3)
    picks = list(rng.choice(canon_only, min(n_canon_only, len(canon_only)), replace=False))
    co_arrs, co_truth = [], []
    for c in picks:
        f = os.path.join(D_CANON, c + ".png")
        if not os.path.isfile(f):
            continue
        base = letterbox(load_gray(f), 224)
        for v in range(3):
            try:
                co_arrs.append(hw.handwrite(base, np.random.default_rng(1000 + v)))
                co_truth.append(c)
            except Exception:
                pass
    co = {"n_canon_only_classes_total": len(canon_only)}
    if co_arrs:
        tco, _, kco = pm.topk(co_arrs, k=10)
        co |= hitrates(tco, co_truth)
        wrong_kind = Counter(k for l, t, k in zip(tco, co_truth, kco) if l[0] != t)
        co["wrong_winner_kind"] = dict(wrong_kind)
        cv2.imwrite(os.path.join(exdir, "p6_canon_only_pensim.png"), co_arrs[0])

    # per-class accuracy vs #train samples (full sample)
    per = defaultdict(lambda: [0, 0])
    for l, t in zip(tl, truths):
        per[t][0] += (l[0] == t)
        per[t][1] += 1
    buckets = {"<20": [0, 0], "20-50": [0, 0], ">50": [0, 0]}
    for c, (h, n) in per.items():
        ntr = len(train_by_class.get(c, []))
        b = "<20" if ntr < 20 else ("20-50" if ntr <= 50 else ">50")
        buckets[b][0] += h
        buckets[b][1] += n
    return {"winner_kind_when_correct": dict(kw),
            "winner_kind_when_wrong": dict(kwrong),
            "val_full_index": hitrates(tl, truths),
            "val_canonical_only_index": hitrates(tlnc, truths),
            "pjb_full_index": hitrates(tlp, pjb_t),
            "pjb_canonical_only_index": hitrates(tlpnc, pjb_t),
            "canon_only_classes_pensim": co,
            "acc_by_train_count": {k: {"n": n, "top1": round(h / n, 4) if n else None}
                                   for k, (h, n) in buckets.items()}}


def p7_calibration(pm, pairs, pjb_pairs, bs=800):
    def run(pp):
        scores, correct, margins = [], [], []
        for i in range(0, len(pp), bs):
            chunk = pp[i:i + bs]
            Q = pm.embed_arrays([load_gray(p) for p, _ in chunk])
            sc, li = pm.label_scores(Q).topk(2, dim=1)
            for (p, t), s, l in zip(chunk, sc.cpu().numpy(), li.cpu().numpy()):
                scores.append(float(s[0]))
                margins.append(float(s[0] - s[1]))
                correct.append(pm.uniq[l[0]] == t)
        scores, correct, margins = np.array(scores), np.array(correct), np.array(margins)
        bins = []
        for lo in np.arange(0.5, 1.0, 0.05):
            m = (scores >= lo) & (scores < lo + 0.05)
            if m.sum():
                bins.append({"score_bin": f"{lo:.2f}-{lo+0.05:.2f}", "n": int(m.sum()),
                             "precision": round(float(correct[m].mean()), 3)})
        mb = []
        for lo in (0, .02, .05, .1, .2):
            m = margins >= lo
            mb.append({"margin_ge": lo, "n": int(m.sum()),
                       "precision": round(float(correct[m].mean()), 3) if m.sum() else None})
        return {"n": len(pp), "top1": round(float(correct.mean()), 4),
                "score_bins": bins, "margin_bins": mb}
    return {"val": run(pairs), "pjb": run(pjb_pairs)}


def p8_fixes(pm, sample, pjb_pairs):
    truths = [t for _, t in sample]
    pjb_t = [t for _, t in pjb_pairs]
    out = {}

    # --- mean vs max aggregation
    for name, pp, tt in (("val", sample, truths), ("pjb", pjb_pairs, pjb_t)):
        arrs = [load_gray(p) for p, _ in pp]
        Q = pm.embed_arrays(arrs)
        for agg in ("max", "mean"):
            sc, li = pm.label_scores(Q, agg=agg).topk(10, dim=1)
            tl = [[pm.uniq[j] for j in row] for row in li.cpu().numpy()]
            out[f"agg_{agg}_{name}"] = hitrates(tl, tt)

    # --- TTA (mean embedding over small rotations/scales)
    def tta_embed(arrs):
        views = [arrs]
        for ang, s in ((-8, 1.0), (8, 1.0), (0, 0.92), (0, 1.08)):
            v = []
            for g in arrs:
                h, w = g.shape
                M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
                v.append(cv2.warpAffine(g, M, (w, h), borderValue=255))
            views.append(v)
        Qs = [pm.embed_arrays(v) for v in views]
        return torch.nn.functional.normalize(torch.stack(Qs).mean(0), dim=-1)
    for name, pp, tt in (("val", sample, truths), ("pjb", pjb_pairs, pjb_t)):
        arrs = [load_gray(p) for p, _ in pp]
        Q = tta_embed(arrs)
        sc, li = pm.label_scores(Q).topk(10, dim=1)
        tl = [[pm.uniq[j] for j in row] for row in li.cpu().numpy()]
        out[f"tta_{name}"] = hitrates(tl, tt)
    # TTA on canvas-sim (the app domain)
    arrs = [canvas_sim(load_gray(p), 4) for p, _ in sample]
    Q = tta_embed(arrs)
    sc, li = pm.label_scores(Q).topk(10, dim=1)
    tl = [[pm.uniq[j] for j in row] for row in li.cpu().numpy()]
    out["tta_canvas_sim_val"] = hitrates(tl, truths)

    # --- blur / lowres remediation (mirror stress_test corruptions, then fix)
    def blur(g):
        return cv2.GaussianBlur(g, (0, 0), max(g.shape) / 40.0)
    def unsharp(g):
        b = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 60.0)
        sh = np.clip(1.8 * g.astype(np.float32) - 0.8 * b, 0, 255).astype(np.uint8)
        _, o = cv2.threshold(sh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return o
    def lowres(g):
        h, w = g.shape
        px = 32
        small = cv2.resize(g, (max(1, px * w // max(h, w)), max(1, px * h // max(h, w))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    def lowres_fix(g):
        b = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 100.0)
        _, o = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return o
    for name, corrupt, fix in (("blur", blur, unsharp), ("lowres32", lowres, lowres_fix)):
        broken = [corrupt(load_gray(p)) for p, _ in sample]
        tl, _, _ = pm.topk(broken, k=10)
        out[f"{name}_corrupted"] = hitrates(tl, truths)
        tl, _, _ = pm.topk([fix(g) for g in broken], k=10)
        out[f"{name}_fixed"] = hitrates(tl, truths)
    return out


def p9_latency(ckpt, index, sample):
    res = {}
    for threads in (1, 4):
        torch.set_num_threads(threads)
        mc = Matcher(ckpt, index, device="cpu")
        qs = [load_gray(p) for p, _ in sample[:40]]
        for g in qs[:5]:
            mc.match(g)
        t0 = time.perf_counter()
        for g in qs:
            mc.match(g)
        res[f"cpu_{threads}thread_ms"] = round(1000 * (time.perf_counter() - t0) / len(qs), 1)
    return res


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.path.join(MATCH, "runs", "default"))
    ap.add_argument("--limit", type=int, default=800, help="probe sample size")
    ap.add_argument("--garbage-n", type=int, default=300)
    ap.add_argument("--canon-only-classes", type=int, default=120)
    ap.add_argument("--leak-emb-limit", type=int, default=4600, help="P1 pixel-NN sample")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out", default=os.path.join(HERE, "probe_results.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.limit, args.garbage_n = 60, 25
        args.canon_only_classes, args.leak_emb_limit = 8, 200
        args.json_out = args.json_out.replace(".json", "_smoke.json")

    ckpt = os.path.join(args.run, "best.pt")
    index = os.path.join(args.run, "index.npz")
    val_split = os.path.join(args.run, "val_split.json")
    exdir = os.path.join(HERE, "examples")
    os.makedirs(exdir, exist_ok=True)

    rep = Report(args.json_out)
    pm = ProbeMatcher(ckpt, index, args.device)
    pairs = [pt for pt in load_val(val_split) if pt[1] in pm.lab2i]
    rng = random.Random(0)
    rng.shuffle(pairs)
    sample = pairs[:args.limit]
    pjb_pairs = []
    for c in sorted(os.listdir(D_PJB)):
        d = os.path.join(D_PJB, c)
        if os.path.isdir(d):
            pjb_pairs += [(os.path.join(d, f), c) for f in sorted(os.listdir(d))
                          if f.lower().endswith(".png")]
    if args.smoke:
        pjb_pairs = pjb_pairs[::5]
    train_by_class = train_files_by_class([p for p, _ in pairs])

    rep.put("meta", {
        "run": os.path.realpath(args.run), "device": args.device,
        "arch": pm.m.meta["arch"], "size": pm.size,
        "n_prototypes": len(pm.labels), "n_labels": len(pm.uniq),
        "prototype_kinds": dict(Counter(pm.kinds.tolist())),
        "val_n": len(pairs), "sample_n": len(sample), "pjb_n": len(pjb_pairs),
        "val_classes": len(set(t for _, t in pairs)),
        "hand_classes": len(train_by_class),
        "canon_no_csv": sorted(set(pm.uniq) - set(pm.m.desc))[:5],
        "smoke": args.smoke})

    probes = [
        ("P1_leakage", lambda: p1_leakage(pm, pairs, train_by_class, args.leak_emb_limit)),
        ("P2_frames", lambda: p2_frames(pm, sample, pjb_pairs, exdir)),
        ("P3_canvas_sim", lambda: p3_canvas(pm, sample, exdir)),
        ("P4_novel_corruptions", lambda: p4_novel(pm, sample)),
        ("P5_openset", lambda: p5_openset(pm, sample, args.garbage_n)),
        ("P6_prototype_bias", lambda: p6_prototype_bias(pm, sample, pjb_pairs,
                                                        train_by_class,
                                                        args.canon_only_classes, exdir)),
        ("P7_calibration", lambda: p7_calibration(pm, pairs, pjb_pairs)),
        ("P8_fixes", lambda: p8_fixes(pm, sample, pjb_pairs)),
        ("P9_latency", lambda: p9_latency(ckpt, index, sample)),
    ]
    for name, fn in probes:
        t0 = time.time()
        try:
            rep.put(name, fn())
            print(f"[probe] {name} done in {time.time()-t0:.0f}s", flush=True)
        except Exception:
            rep.put(name, {"error": traceback.format_exc()})
            print(f"[probe] {name} FAILED", flush=True)
    print("[probe] all done")


if __name__ == "__main__":
    main()
