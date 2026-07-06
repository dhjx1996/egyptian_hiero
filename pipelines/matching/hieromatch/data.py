"""Data loading, preprocessing and handwriting-style augmentation.

The domain-bridging idea: canonical glyphs are trained WITH heavy "fake
handwriting" augmentation (including the procedural handwrite() engine from
misc/scripts) so classes that have no real handwriting samples still land near
their canonical prototype in embedding space -- that is what makes the matcher
open-set over the full canonical inventory (769 signs) and over new scripts.
"""
import json
import os
import random
import sys

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_HW = None  # lazy handwriting_augment module


# ---------------------------------------------------------------- basics
def load_gray(path):
    """Any image file -> uint8 grayscale, dark ink on white."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    return np.array(im.convert("L"))


def crop_ink(g, thresh=200, pad=6):
    ys, xs = np.where(g < thresh)
    if len(xs) == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + 1 + pad)
    return g[y0:y1, x0:x1]


def letterbox(g, size, jitter=0.0, rng=None):
    """Fit ink crop into a size x size white square, centered (optionally jittered)."""
    c = crop_ink(g)
    if c is None:
        return np.full((size, size), 255, np.uint8)
    h, w = c.shape
    margin = 0.08 + (rng.uniform(0, jitter) if (rng and jitter) else 0)
    s = size * (1 - margin) / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    if rng and jitter:
        y0 += rng.integers(-int(size * jitter / 2), int(size * jitter / 2) + 1)
        x0 += rng.integers(-int(size * jitter / 2), int(size * jitter / 2) + 1)
        y0, x0 = int(np.clip(y0, 0, size - nh)), int(np.clip(x0, 0, size - nw))
    canvas[y0:y0 + nh, x0:x0 + nw] = c
    return canvas


def to_tensor(g):
    return torch.from_numpy((255.0 - g.astype(np.float32)) / 127.5 - 1.0).unsqueeze(0)
    # ink -> +1, background -> -1 (drawing polarity explicit and background-invariant)


def preprocess_array(g, size):
    return to_tensor(letterbox(g, size))


def preprocess_path(path, size):
    return preprocess_array(load_gray(path), size)


# ---------------------------------------------------------------- augmentation
def _affine(g, rng):
    h, w = g.shape
    ang = rng.uniform(-12, 12)
    sc = rng.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 1] += rng.uniform(-0.08, 0.08)                       # shear
    M[:, 2] += rng.uniform(-0.04, 0.04, 2) * (w, h)           # translate
    return cv2.warpAffine(g, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)


def _elastic(g, rng, alpha=8.0, sigma=7.0):
    h, w = g.shape
    dx = cv2.GaussianBlur((rng.random((h, w), dtype=np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w), dtype=np.float32) * 2 - 1), (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(g, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=255)


def _morph(g, rng):
    k = int(rng.integers(1, 4))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 1, k + 1))
    return cv2.erode(g, kern) if rng.random() < 0.5 else cv2.dilate(g, kern)
    # NB erode darkens (thickens ink), dilate lightens (thins ink)


def _photo(g, rng):
    if rng.random() < 0.5:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.4, 1.1))
    if rng.random() < 0.5:                                    # ink strength / gamma
        gamma = rng.uniform(0.6, 1.6)
        g = (np.power(g / 255.0, gamma) * 255).astype(np.uint8)
    if rng.random() < 0.3:
        g = np.clip(g.astype(np.float32) + rng.normal(0, 8, g.shape), 0, 255).astype(np.uint8)
    return g


def _handwrite(g, rng):
    """Procedural pen simulation (misc/scripts/handwriting_augment) -- the strongest
    canonical->handwriting bridge. Falls back to elastic if scikit-image is absent."""
    global _HW
    if _HW is None:
        sys.path.insert(0, os.path.join(REPO, "misc", "scripts"))
        import handwriting_augment as _hwmod
        _HW = _hwmod
    seed_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
    return _HW.handwrite(g, seed_rng)


def augment(g, rng, source, size, p_handwrite=0.35):
    """g: uint8 gray dark-on-white (any shape). Returns size x size uint8."""
    work = letterbox(g, 224 if source == "canon" else size, jitter=0.06, rng=rng)
    if source == "canon":
        if rng.random() < p_handwrite:
            try:
                work = _handwrite(work, rng)
            except Exception:
                work = _elastic(work, rng, alpha=12, sigma=6)
        else:
            work = _elastic(work, rng, alpha=10, sigma=6)
        work = letterbox(work, size)
    if rng.random() < 0.9:
        work = _affine(work, rng)
    if rng.random() < 0.5:
        work = _elastic(work, rng, alpha=6, sigma=7)
    if rng.random() < 0.6:
        work = _morph(work, rng)
    work = _photo(work, rng)
    return letterbox(work, size, jitter=0.05, rng=rng)


# ---------------------------------------------------------------- datasets
def build_items(handwriting_root, canonical_dir, val_frac=0.1, seed=17, canon_repeat=8,
                min_val_n=8):
    """Returns (classes, train_items, val_items). Item = (path, class_idx, source).
    Classes = canonical stems UNION handwriting dirs, so canonical-only classes are
    trainable (via augmentation) and stay matchable. Deterministic split."""
    canon = {}
    if canonical_dir:
        canon = {os.path.splitext(f)[0]: os.path.join(canonical_dir, f)
                 for f in os.listdir(canonical_dir) if f.lower().endswith(".png")}
    hand = {}
    if handwriting_root:
        for c in sorted(os.listdir(handwriting_root)):
            d = os.path.join(handwriting_root, c)
            if os.path.isdir(d):
                fs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png"))
                if fs:
                    hand[c] = fs
    classes = sorted(set(canon) | set(hand))
    idx = {c: i for i, c in enumerate(classes)}

    rng = random.Random(seed)
    train_items, val_items = [], []
    for c, fs in hand.items():
        fs = list(fs)
        rng.shuffle(fs)
        n_val = max(1, round(len(fs) * val_frac)) if len(fs) >= min_val_n else 0
        for p in fs[:n_val]:
            val_items.append((p, idx[c], "hand"))
        for p in fs[n_val:]:
            train_items.append((p, idx[c], "hand"))
    for c, p in canon.items():
        train_items.extend([(p, idx[c], "canon")] * canon_repeat)
    return classes, train_items, val_items


class TrainDataset(Dataset):
    def __init__(self, items, size=112, p_handwrite=0.35, seed=0):
        self.items, self.size, self.p_handwrite, self.seed = items, size, p_handwrite, seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, source = self.items[i]
        rng = np.random.default_rng((self.seed * 1000003 + self.epoch * 7919 + i) % 2**31)
        g = augment(load_gray(path), rng, source, self.size, self.p_handwrite)
        return to_tensor(g), label


class EvalDataset(Dataset):
    def __init__(self, items, size=112):
        self.items, self.size = items, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, _ = self.items[i]
        return preprocess_path(path, self.size), label


def save_split(path, classes, val_items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes,
                   "val": [{"path": os.path.abspath(p), "class": classes[li]}
                           for p, li, _ in val_items]}, f, indent=0)
