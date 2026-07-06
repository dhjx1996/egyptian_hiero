"""
Build the matching index: embed canonical glyphs (+ optional per-class
handwriting centroids) with a trained encoder.

Open-set growth: run again with MORE --canonical dirs (e.g. add a Mayan glyph
folder) and the new symbols become matchable with no retraining. Prototypes are
tagged by kind ('canonical' / 'centroid'); match.py scores a query against all
prototypes and aggregates per label.

    ./.venv/bin/python build_index.py --ckpt runs/run1/best.pt --out runs/run1/index.npz
"""
import argparse
import json
import os

import numpy as np
import torch

from hieromatch.data import preprocess_path
from hieromatch.model import load_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
D_HAND = os.path.join(REPO, "hiero_data", "Hand-drawn Hieroglyph Dataset")
D_CANON = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")


@torch.no_grad()
def embed_paths(encoder, paths, size, device, bs=256):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([preprocess_path(p, size) for p in paths[i:i + bs]]).to(device)
        out.append(encoder.embed(x).cpu())
    return torch.cat(out) if out else torch.zeros(0, encoder.embed_dim)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--canonical", action="append", default=[],
                    help="dir of <CLASS>.png (repeatable; default: the Gardiner utf-pngs)")
    ap.add_argument("--handwriting", default="",
                    help=f"optional <root>/<CLASS>/*.png for centroid prototypes (e.g. {D_HAND!r})")
    ap.add_argument("--exclude-val", default="", help="val_split.json to keep centroids train-only")
    ap.add_argument("--max-per-class", type=int, default=40, help="centroid sample cap")
    ap.add_argument("--out", default="", help="default: <ckpt dir>/index.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    encoder, meta = load_encoder(args.ckpt, device)
    size = meta["size"]
    canon_dirs = args.canonical or [D_CANON]

    labels, kinds, embs = [], [], []

    for cd in canon_dirs:
        files = sorted(f for f in os.listdir(cd) if f.lower().endswith(".png"))
        e = embed_paths(encoder, [os.path.join(cd, f) for f in files], size, device)
        embs.append(e)
        labels += [os.path.splitext(f)[0] for f in files]
        kinds += ["canonical"] * len(files)
        print(f"[index] {len(files)} canonical prototypes from {cd}")

    if args.handwriting:
        excl = set()
        if args.exclude_val and os.path.isfile(args.exclude_val):
            excl = {v["path"] for v in json.load(open(args.exclude_val))["val"]}
        n = 0
        for c in sorted(os.listdir(args.handwriting)):
            d = os.path.join(args.handwriting, c)
            if not os.path.isdir(d):
                continue
            fs = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".png")]
            fs = [p for p in fs if os.path.abspath(p) not in excl][:args.max_per_class]
            if not fs:
                continue
            e = embed_paths(encoder, fs, size, device)
            centroid = torch.nn.functional.normalize(e.mean(0, keepdim=True), dim=-1)
            embs.append(centroid)
            labels.append(c)
            kinds.append("centroid")
            n += 1
        print(f"[index] {n} handwriting-centroid prototypes from {args.handwriting}")

    emb = torch.cat(embs).numpy().astype(np.float32)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "index.npz")
    np.savez_compressed(out, emb=emb, label=np.array(labels), kind=np.array(kinds),
                        meta=json.dumps({"ckpt": os.path.abspath(args.ckpt), "size": size,
                                         "embed_dim": meta["embed_dim"],
                                         "canonical_dirs": [os.path.abspath(c) for c in canon_dirs]}))
    print(f"[index] {emb.shape[0]} prototypes ({len(set(labels))} labels, dim {emb.shape[1]}) -> {out}")


if __name__ == "__main__":
    main()
