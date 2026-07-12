"""
Leakage-corrected retrain for the 2026-07 adversarial review.

Identical to train_encoder.py (production recipe) EXCEPT the val split is
GROUP-disjoint: dataset filenames are `<src>_<drawing>_<variant>_<CLASS>-x-y.png`
where the ~13 `<variant>`s are augmented near-duplicates of one source drawing.
train_encoder.py splits at file level, so nearly every val image has ~12
near-identical siblings in train; this script holds out whole source drawings.

The delta between this run's held-out top-1 and the production 0.971 bounds the
leakage inflation. pjb (unseen writer) is evaluated identically for both, as the
model-quality control.

    ./.venv/bin/python review/train_grouped.py --name review_grouped --arch resnet34 \
        --size 160 --epochs 100 --batch-size 512 --workers 16 --p-handwrite 0.5 \
        --canon-repeat 12 --p-dropstroke 0.10
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH = os.path.dirname(HERE)
sys.path.insert(0, MATCH)

from hieromatch.data import TrainDataset, EvalDataset, save_split      # noqa: E402
from hieromatch.model import HieroEncoder, CosineHead, save_encoder    # noqa: E402
from train_encoder import validate, D_HAND, D_CANON                    # noqa: E402


def group_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    toks = stem.split("_")
    return "_".join(toks[:2]) if len(toks) >= 4 else stem


def build_items_grouped(handwriting_root, canonical_dir, val_frac=0.1, seed=17,
                        canon_repeat=8, min_groups=3):
    """Same item tuples as hieromatch.data.build_items, but the val split holds
    out whole SOURCE-DRAWING groups (all augmented variants together)."""
    canon = {os.path.splitext(f)[0]: os.path.join(canonical_dir, f)
             for f in os.listdir(canonical_dir) if f.lower().endswith(".png")}
    hand = {}
    for c in sorted(os.listdir(handwriting_root)):
        d = os.path.join(handwriting_root, c)
        if os.path.isdir(d):
            fs = sorted(os.path.join(d, f) for f in os.listdir(d)
                        if f.lower().endswith(".png"))
            if fs:
                hand[c] = fs
    classes = sorted(set(canon) | set(hand))
    idx = {c: i for i, c in enumerate(classes)}

    rng = random.Random(seed)
    train_items, val_items, n_val_groups = [], [], 0
    for c, fs in hand.items():
        groups = defaultdict(list)
        for p in fs:
            groups[group_id(p)].append(p)
        gids = sorted(groups)
        rng.shuffle(gids)
        n_val_g = max(1, round(len(gids) * val_frac)) if len(gids) >= min_groups else 0
        n_val_groups += n_val_g
        for g in gids[:n_val_g]:
            val_items += [(p, idx[c], "hand") for p in groups[g]]
        for g in gids[n_val_g:]:
            train_items += [(p, idx[c], "hand") for p in groups[g]]
    for c, p in canon.items():
        train_items.extend([(p, idx[c], "canon")] * canon_repeat)
    print(f"[grouped] held-out groups: {n_val_groups} "
          f"({len(val_items)} files) | train hand files "
          f"{sum(1 for it in train_items if it[2] == 'hand')}")
    return classes, train_items, val_items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handwriting", default=D_HAND)
    ap.add_argument("--canonical", default=D_CANON)
    ap.add_argument("--name", default="review_grouped")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--canon-repeat", type=int, default=12)
    ap.add_argument("--p-handwrite", type=float, default=0.5)
    ap.add_argument("--p-dropstroke", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    outd = os.path.join(MATCH, "runs", args.name)
    os.makedirs(outd, exist_ok=True)

    classes, train_items, val_items = build_items_grouped(
        args.handwriting, args.canonical, val_frac=args.val_frac, seed=args.seed,
        canon_repeat=args.canon_repeat)
    print(f"[grouped] {len(classes)} classes | train {len(train_items)} | val {len(val_items)}")
    save_split(os.path.join(outd, "val_split.json"), classes, val_items)
    with open(os.path.join(outd, "classes.json"), "w") as f:
        json.dump(classes, f)

    train_ds = TrainDataset(train_items, size=args.size, p_handwrite=args.p_handwrite,
                            seed=args.seed, p_dropstroke=args.p_dropstroke)
    val_ds = EvalDataset(val_items, size=args.size)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=256, num_workers=max(2, args.workers // 2))

    encoder = HieroEncoder(args.arch, args.embed_dim).to(device)
    head = CosineHead(args.embed_dim, len(classes)).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(train_ld)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best = 0.0

    logf = open(os.path.join(outd, "log.txt"), "a")
    for ep in range(args.epochs):
        encoder.train()
        train_ds.set_epoch(ep)
        t0, seen, lsum, csum = time.time(), 0, 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = head(encoder(x))
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            seen += len(y)
            lsum += loss.item() * len(y)
            csum += (logits.argmax(1) == y).sum().item()
        v1, v5 = validate(encoder, head, val_ld, device) if val_items else (0.0, 0.0)
        line = (f"epoch {ep:3d} | loss {lsum/max(seen,1):.4f} | train-acc {csum/max(seen,1):.3f} "
                f"| val top1 {v1:.3f} top5 {v5:.3f} | {time.time()-t0:.0f}s")
        print("[grouped] " + line, flush=True)
        logf.write(line + "\n")
        logf.flush()
        torch.save({"state_dict": encoder.state_dict(), "head": head.state_dict(),
                    "opt": opt.state_dict(), "epoch": ep, "best": best,
                    "arch": args.arch, "embed_dim": args.embed_dim, "size": args.size,
                    "classes": classes}, os.path.join(outd, "last.pt"))
        if v1 >= best:
            best = v1
            save_encoder(os.path.join(outd, "best.pt"), encoder, classes, args.size,
                         extra={"val_top1": v1, "epoch": ep})
    print(f"[grouped] done. best val top1 {best:.3f} -> {outd}/best.pt")


if __name__ == "__main__":
    main()
