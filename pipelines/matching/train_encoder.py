"""
Train the symbol-matching encoder (Pipeline 2).

Metric-learning setup: encoder embedding + cosine classifier over ALL classes
(canonical stems UNION handwriting folders). Real handwriting trains directly;
canonical-only classes train through handwriting-style augmentation, so the
whole inventory is matchable. Generalizes to any script: point --handwriting at
a <CLASS>/*.png tree and --canonical at a <CLASS>.png dir.

Smoke (T4):  ./.venv/bin/python train_encoder.py --epochs 3 --name smoke
HPC (A100):  see README.md (bigger arch/size/epochs).
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hieromatch.data import build_items, TrainDataset, EvalDataset, save_split
from hieromatch.model import HieroEncoder, CosineHead, save_encoder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
D_HAND = os.path.join(REPO, "hiero_data", "Hand-drawn Hieroglyph Dataset")
D_CANON = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")


@torch.no_grad()
def validate(encoder, head, loader, device):
    encoder.eval()
    top1 = top5 = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = head(encoder(x))
        rank = logits.topk(5, dim=1).indices
        top1 += (rank[:, 0] == y).sum().item()
        top5 += (rank == y[:, None]).any(1).sum().item()
        n += len(y)
    return top1 / max(n, 1), top5 / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handwriting", default=D_HAND, help="<root>/<CLASS>/*.png (set '' for canonical-only)")
    ap.add_argument("--canonical", default=D_CANON, help="<dir>/<CLASS>.png (set '' to skip)")
    ap.add_argument("--name", default="run1")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--canon-repeat", type=int, default=8, help="canonical oversampling per epoch")
    ap.add_argument("--p-handwrite", type=float, default=0.35,
                    help="prob. of procedural pen simulation on canonical samples")
    ap.add_argument("--p-dropstroke", type=float, default=0.0,
                    help="prob. of missing-stroke augmentation (pen skips / dropped parts)")
    ap.add_argument("--p-frame", type=float, default=0.0,
                    help="prob. of synthetic scan-cell border (frame-invariance, review F2)")
    ap.add_argument("--p-partial", type=float, default=0.0,
                    help="prob. of half-finished-drawing augmentation (review F4)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--file-level-split", action="store_true",
                    help="legacy leaky split (default is group-disjoint: whole source "
                         "drawings held out, review F1). Only for reproducing old runs.")
    ap.add_argument("--synthetic", default="",
                    help="<root>/<CLASS>/*.png of quality-gated GENERATED samples "
                         "(see synth_filter.py); adds to training only, never to val")
    ap.add_argument("--synth-cap-frac", type=float, default=0.25,
                    help="max synthetic items as a fraction of real handwriting train items")
    ap.add_argument("--synth-per-class", type=int, default=20)
    ap.add_argument("--abstract", default="",
                    help="<root>/lvl*/<CLASS>.png stick-figure abstraction bank "
                         "(make_abstractions.py); adds canonical-style train items")
    ap.add_argument("--abstract-repeat", type=int, default=4,
                    help="per-level oversampling of abstraction renders")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--resume", default="", help="last.pt to continue from")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    outd = os.path.join(HERE, "runs", args.name)
    os.makedirs(outd, exist_ok=True)

    classes, train_items, val_items = build_items(
        args.handwriting, args.canonical, val_frac=args.val_frac, seed=args.seed,
        canon_repeat=args.canon_repeat, synthetic_root=args.synthetic or None,
        synth_cap_frac=args.synth_cap_frac, synth_per_class=args.synth_per_class,
        abstract_root=args.abstract or None, abstract_repeat=args.abstract_repeat,
        group_val=not args.file_level_split)
    n_hand = sum(1 for it in train_items if it[2] == "hand")
    n_synth = sum(1 for it in train_items if it[2] == "synth")
    split = "file-level(leaky)" if args.file_level_split else "group-disjoint"
    print(f"[train] {len(classes)} classes | {split} split | train {len(train_items)} "
          f"(hand {n_hand}, synth {n_synth}, canon x{args.canon_repeat}"
          f"{f', abstract x{args.abstract_repeat}' if args.abstract else ''}) | val {len(val_items)}")
    save_split(os.path.join(outd, "val_split.json"), classes, val_items)
    with open(os.path.join(outd, "classes.json"), "w") as f:
        json.dump(classes, f)

    train_ds = TrainDataset(train_items, size=args.size, p_handwrite=args.p_handwrite, seed=args.seed,
                            p_dropstroke=args.p_dropstroke, p_frame=args.p_frame,
                            p_partial=args.p_partial)
    val_ds = EvalDataset(val_items, size=args.size)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=256, num_workers=max(2, args.workers // 2))

    encoder = HieroEncoder(args.arch, args.embed_dim).to(device)
    head = CosineHead(args.embed_dim, len(classes)).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(train_ld)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    start_ep, best = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        encoder.load_state_dict(ck["state_dict"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start_ep, best = ck["epoch"] + 1, ck.get("best", 0.0)
        print(f"[train] resumed from {args.resume} at epoch {start_ep}")

    logf = open(os.path.join(outd, "log.txt"), "a")
    for ep in range(start_ep, args.epochs):
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
        print("[train] " + line)
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
    print(f"[train] done. best val top1 {best:.3f} -> {outd}/best.pt")


if __name__ == "__main__":
    main()
