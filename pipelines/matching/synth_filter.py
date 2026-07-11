"""
Failure + fidelity gate for GENERATED handwriting samples before they feed back
into matcher training (train_encoder.py --synthetic).

Design note: labels are correct BY CONSTRUCTION (each sample was generated
conditioned on its class), so this gate rejects FAILED or DEGENERATE
generations -- it does NOT select for polish. Low-confidence, sloppy,
ambiguous-but-on-class samples are deliberately KEPT (rejecting them would push
the classifier toward the mean and erode outlier handling).

Rejection reasons:
  bad_ink            ink fraction outside [--ink-min, --ink-max] (blank canvases,
                     flooded/grey failure cells; model-free)
  confident_impostor the FROZEN baseline matcher assigns ANOTHER class with
                     score >= --reject-other-score (a mislabeled trainer)
  off_class          the intended class is not in the matcher's top-5
  canonical_clone    embedding cosine to the class's CANONICAL prototype
                     >= --max-canon-sim: the sample is a near-copy of the
                     canonical glyph (a real risk with guidance/SDEdit-style
                     sampling). Such samples would re-weight training toward
                     clean canonical forms through the lightly-augmented synth
                     path and damage canonical->handwriting domain bridging.
  near_duplicate     embedding cosine >= --max-dup-sim to an already-accepted
                     sample of the same class: collapsed/low-diversity output
                     adds no information, only distribution shrinkage.

The gate matcher is the frozen real-data baseline (never the model being
retrained), so the generate -> filter -> retrain loop cannot self-amplify.

Input tree:  <gen-root>/<CLASS>/*.png   (generate.py / generate_hiero.py layout)
Output tree: <out>/<CLASS>/*.png        (accepted copies) + <out>/filter_stats.json
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

from match import Matcher
from hieromatch.data import load_gray, preprocess_array


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", required=True, help="<root>/<CLASS>/*.png generated samples")
    ap.add_argument("--ckpt", default="runs/default/best.pt", help="FROZEN baseline encoder")
    ap.add_argument("--index", default="runs/default/index.npz", help="baseline prototype index")
    ap.add_argument("--out", required=True, help="accepted-samples tree (created)")
    ap.add_argument("--ink-min", type=float, default=0.005, help="min ink fraction (rejects blanks)")
    ap.add_argument("--ink-max", type=float, default=0.70, help="max ink fraction (rejects flooded cells)")
    ap.add_argument("--reject-other-score", type=float, default=0.45,
                    help="reject when the top1 is a DIFFERENT class at/above this score")
    ap.add_argument("--max-canon-sim", type=float, default=0.92,
                    help="reject near-copies of the canonical glyph at/above this cosine")
    ap.add_argument("--max-dup-sim", type=float, default=0.97,
                    help="reject same-class samples this similar to an already-accepted one")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = Matcher(args.ckpt, args.index, device)

    # per-class canonical prototype row (index stores one canonical emb per class)
    canon_idx = {lab: i for i, (lab, kind) in enumerate(zip(m.labels, m.kinds)) if kind == "canonical"}

    @torch.no_grad()
    def embed(g):
        x = preprocess_array(g, m.meta["size"]).unsqueeze(0).to(m.device)
        return m.encoder.embed(x)[0]                      # (D,), L2-normed

    stats, n_keep, n_all = {}, 0, 0
    reasons = ("bad_ink", "confident_impostor", "off_class", "canonical_clone", "near_duplicate")
    for cls in sorted(os.listdir(args.gen_root)):
        d = os.path.join(args.gen_root, cls)
        if not os.path.isdir(d):
            continue
        kept, kept_embs = [], []
        why = dict.fromkeys(reasons, 0)
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(".png"):
                continue
            n_all += 1
            g = load_gray(os.path.join(d, f))
            ink = float((g < 200).mean())
            if not (args.ink_min <= ink <= args.ink_max):
                why["bad_ink"] += 1
                continue
            q = embed(g)
            sims = (m.emb @ q).cpu().numpy()
            # nearest prototype per class (same ranking rule as Matcher.match)
            best = {}
            for i, lab in enumerate(m.labels):
                if lab not in best or sims[i] > sims[best[lab]]:
                    best[lab] = i
            ranked = sorted(best.items(), key=lambda kv: -sims[kv[1]])[:5]
            top_labels = [lab for lab, _ in ranked]
            if top_labels[0] != cls and sims[ranked[0][1]] >= args.reject_other_score:
                why["confident_impostor"] += 1
            elif cls not in top_labels:
                why["off_class"] += 1
            elif cls in canon_idx and sims[canon_idx[cls]] >= args.max_canon_sim:
                why["canonical_clone"] += 1
            elif kept_embs and max(float(e @ q) for e in kept_embs) >= args.max_dup_sim:
                why["near_duplicate"] += 1
            else:
                kept.append(f)
                kept_embs.append(q)
        if kept:
            od = os.path.join(args.out, cls)
            os.makedirs(od, exist_ok=True)
            for f in kept:
                shutil.copy2(os.path.join(d, f), os.path.join(od, f))
        n_keep += len(kept)
        stats[cls] = {"kept": len(kept), **why}

    os.makedirs(args.out, exist_ok=True)
    summary = {"total": n_all, "kept": n_keep,
               "accept_rate": round(n_keep / max(n_all, 1), 4),
               "ink_min": args.ink_min, "ink_max": args.ink_max,
               "reject_other_score": args.reject_other_score,
               "max_canon_sim": args.max_canon_sim, "max_dup_sim": args.max_dup_sim,
               "gate_ckpt": os.path.abspath(args.ckpt), "classes": stats}
    with open(os.path.join(args.out, "filter_stats.json"), "w") as f:
        json.dump(summary, f, indent=1)
    rej = {k: sum(s[k] for s in stats.values()) for k in reasons}
    print(f"[filter] kept {n_keep}/{n_all} ({summary['accept_rate']:.1%}) -> {args.out}")
    print(f"[filter] rejected: {rej}")
    print(f"[filter] classes with >=1 accepted: {sum(1 for s in stats.values() if s['kept'])}/{len(stats)}")


if __name__ == "__main__":
    main()
