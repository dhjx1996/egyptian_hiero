"""
Match handwritten symbol image(s) against the canonical index (Pipeline 2 CLI).

    ./.venv/bin/python match.py --ckpt runs/run1/best.pt --index runs/run1/index.npz \
        --image /path/drawn.png --top 5

Python API:  from match import Matcher; Matcher(ckpt, index).match(pil_or_path)
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from hieromatch.data import preprocess_path, preprocess_array, load_gray
from hieromatch.model import load_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
D_CSV = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "gardiner_hieroglyphs.csv")


def load_descriptions(csv_path):
    desc = {}
    if csv_path and os.path.isfile(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row.get("gardiner_num") or ""
                if key:
                    desc[key] = {"char": row.get("hieroglyph", ""),
                                 "description": row.get("description", "")}
    return desc


class Matcher:
    def __init__(self, ckpt, index, device="cpu", csv_path=D_CSV):
        self.device = torch.device(device)
        self.encoder, self.meta = load_encoder(ckpt, self.device)
        ix = np.load(index, allow_pickle=False)
        self.emb = torch.from_numpy(ix["emb"]).to(self.device)     # (P, D), L2-normed
        self.labels = [str(x) for x in ix["label"]]
        self.kinds = [str(x) for x in ix["kind"]]
        self.desc = load_descriptions(csv_path)

    @torch.no_grad()
    def match(self, image, top=5):
        """image: path or uint8 gray array. Returns [{label, score, kind, char, description}]."""
        if isinstance(image, str):
            x = preprocess_path(image, self.meta["size"])
        else:
            x = preprocess_array(image, self.meta["size"])
        q = self.encoder.embed(x.unsqueeze(0).to(self.device))[0]  # (D,)
        sims = (self.emb @ q).cpu().numpy()                        # cosine per prototype
        best = {}
        for i, (lab, s) in enumerate(zip(self.labels, sims)):
            if lab not in best or s > sims[best[lab]]:
                best[lab] = i
        ranked = sorted(best.items(), key=lambda kv: -sims[kv[1]])[:top]
        out = []
        for lab, i in ranked:
            d = self.desc.get(lab, {})
            out.append({"label": lab, "score": round(float(sims[i]), 4), "kind": self.kinds[i],
                        "char": d.get("char", ""), "description": d.get("description", "")})
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--image", action="append", default=[], help="query image (repeatable)")
    ap.add_argument("--dir", default="", help="match every .png in a directory")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--csv", default=D_CSV, help="Gardiner metadata csv ('' to skip)")
    ap.add_argument("--json-out", default="", help="write results as JSON")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    queries = list(args.image)
    if args.dir:
        queries += [os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir))
                    if f.lower().endswith(".png")]
    if not queries:
        ap.error("pass --image and/or --dir")

    m = Matcher(args.ckpt, args.index, device=args.device, csv_path=args.csv)
    results = {}
    for q in queries:
        r = m.match(load_gray(q), top=args.top)
        results[q] = r
        print(f"\n{q}")
        for j, hit in enumerate(r):
            print(f"  {j+1}. {hit['label']:8s} {hit['char']:2s} cos={hit['score']:.3f} "
                  f"[{hit['kind']}] {hit['description'][:60]}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"\n[match] json -> {args.json_out}")


if __name__ == "__main__":
    main()
