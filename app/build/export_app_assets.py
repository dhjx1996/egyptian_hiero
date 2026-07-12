"""
Export everything the dictionary app needs, from the production matcher run.

Produces (under app/):
  data/model.onnx           fp32 encoder (image -> L2-normalized embedding)
  data/model.int8.onnx      static-quantized (QDQ) version, ~4x smaller
  data/index.bin            prototype matrix, float32 row-major (P x D)
  data/index_meta.json      {dim, labels[], kinds[]}
  data/glyphs.json          {label: {char, desc, details, priority}}
  data/selftest.json        PNG fixtures + expected top-5 (torch reference)
  data/config.json          {model, size, thresh, pad, margin, score_threshold}
  data/export_report.json   torch-vs-onnx accuracy check (int8 gate)
  glyphs/<LABEL>.png        320px canonical thumbnails
  icons/icon-{192,512}.png  PWA icons

    ./.venv/bin/python ../../app/build/export_app_assets.py   (from pipelines/matching)
"""
import argparse
import base64
import csv
import io
import json
import os
import random
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
REPO = os.path.dirname(APP)
MATCH = os.path.join(REPO, "pipelines", "matching")
sys.path.insert(0, MATCH)

from hieromatch.model import load_encoder                     # noqa: E402
from hieromatch.data import (load_gray, preprocess_path,      # noqa: E402
                             preprocess_array)

D_HAND = os.path.join(REPO, "hiero_data", "Hand-drawn Hieroglyph Dataset")
D_CANON = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")
D_CSV = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data",
                     "gardiner_hieroglyphs.csv")
D_PJB = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data",
                     "me-sign-examples-pjb")


class EmbedWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        return F.normalize(self.enc(x), dim=-1)


def labeled_tree(root):
    pairs = []
    for c in sorted(os.listdir(root)):
        d = os.path.join(root, c)
        if os.path.isdir(d):
            pairs += [(os.path.join(d, f), c) for f in sorted(os.listdir(d))
                      if f.lower().endswith(".png")]
    return pairs


def retrieval_top1(emb_q, truths, index_emb, labels, ks=(1, 5)):
    """Same per-label max aggregation as match.Matcher."""
    uniq = sorted(set(labels))
    lab2i = {l: i for i, l in enumerate(uniq)}
    lab_idx = np.array([lab2i[l] for l in labels])
    sims = emb_q @ index_emb.T                                # (N, P)
    L = len(uniq)
    agg = np.full((sims.shape[0], L), -2.0, np.float32)
    np.maximum.at(agg.T, lab_idx, sims.T)                     # per-label max
    order = np.argsort(-agg, axis=1)
    out = {}
    for k in ks:
        hit = sum(t in [uniq[j] for j in row[:k]] for row, t in zip(order, truths))
        out[f"top{k}"] = round(hit / max(len(truths), 1), 4)
    out["n"] = len(truths)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=os.path.join(MATCH, "runs", "default"))
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--val-check-n", type=int, default=400)
    ap.add_argument("--int8-max-drop", type=float, default=0.015,
                    help="max allowed pjb top1 drop for shipping int8")
    args = ap.parse_args()

    run = os.path.realpath(args.run)
    dd = os.path.join(APP, "data")
    os.makedirs(dd, exist_ok=True)
    report = {"run": run}

    # ---------------------------------------------------------------- encoder
    enc, meta = load_encoder(os.path.join(run, "best.pt"), "cpu")
    size = meta["size"]
    wrapper = EmbedWrapper(enc).eval()
    fp32_path = os.path.join(dd, "model.onnx")
    dummy = torch.zeros(1, 1, size, size)
    torch.onnx.export(wrapper, dummy, fp32_path, opset_version=17,
                      input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}})
    print(f"[export] fp32 onnx -> {fp32_path} ({os.path.getsize(fp32_path)/1e6:.1f} MB)")

    # ---------------------------------------------------------------- int8
    # Calibration must cover the APP's input domain (clean frameless canvas
    # strokes), not just the framed scan domain — the first calibration attempt
    # (raw handwriting only) passed val but dropped pjb 2.7pt (see review F2/F3).
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    def canvas_style(path, pen_r=4):
        from skimage.morphology import skeletonize
        from hieromatch.data import letterbox
        g = letterbox(load_gray(path), 320)
        out = np.full((320, 320), 255, np.uint8)
        out[skeletonize(g < 200)] = 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pen_r - 1, 2 * pen_r - 1))
        return cv2.erode(out, k)

    rng = random.Random(7)
    hand_pairs = labeled_tree(D_HAND)
    raw_paths = [p for p, _ in rng.sample(hand_pairs, args.calib_n)]
    canvas_paths = [p for p, _ in rng.sample(hand_pairs, args.calib_n)]
    canon_paths = [os.path.join(D_CANON, f) for f in
                   rng.sample(sorted(os.listdir(D_CANON)), 40)]

    class Reader(CalibrationDataReader):
        """Yields raw scans, canvas-style renderings (thin + thick pen) and
        canonical glyphs, matching the deployed query mix."""
        def __init__(self):
            items = ([("raw", p) for p in raw_paths]
                     + [("canvas", p) for p in canvas_paths]
                     + [("canon", p) for p in canon_paths])
            rng.shuffle(items)
            self.it = iter(items)

        def get_next(self):
            nxt = next(self.it, None)
            if nxt is None:
                return None
            kind, p = nxt
            if kind == "canvas":
                g = canvas_style(p, pen_r=rng.choice([2, 4, 8]))
                x = preprocess_array(g, size).unsqueeze(0).numpy()
            else:
                x = preprocess_path(p, size).unsqueeze(0).numpy()
            return {"image": x}

    pre_path = os.path.join(dd, "model.pre.onnx")
    quant_pre_process(fp32_path, pre_path, skip_symbolic_shape=True)
    int8_path = os.path.join(dd, "model.int8.onnx")
    quantize_static(pre_path, int8_path, Reader(),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    os.remove(pre_path)
    print(f"[export] int8 onnx -> {int8_path} ({os.path.getsize(int8_path)/1e6:.1f} MB)")

    # ---------------------------------------------------------------- index
    ix = np.load(os.path.join(run, "index.npz"), allow_pickle=False)
    emb = ix["emb"].astype(np.float32)
    labels = [str(x) for x in ix["label"]]
    kinds = [str(x) for x in ix["kind"]]
    emb.tofile(os.path.join(dd, "index.bin"))
    json.dump({"dim": int(emb.shape[1]), "count": int(emb.shape[0]),
               "labels": labels, "kinds": kinds},
              open(os.path.join(dd, "index_meta.json"), "w"))
    print(f"[export] index {emb.shape} -> index.bin "
          f"({os.path.getsize(os.path.join(dd,'index.bin'))/1e6:.1f} MB)")

    # ------------------------------------------------- accuracy gate for int8
    val = json.load(open(os.path.join(run, "val_split.json")))["val"]
    rng.shuffle(val)
    checks = {"val": [(e["path"], e["class"]) for e in val[:args.val_check_n]],
              "pjb": labeled_tree(D_PJB)}
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    def embed_all(pairs, runner):
        out = []
        for i in range(0, len(pairs), 64):
            x = torch.stack([preprocess_path(p, size) for p, _ in pairs[i:i + 64]]).numpy()
            out.append(runner(x))
        return np.concatenate(out)

    # fp16 fallback candidate (half the size, no calibration risk; ORT-web
    # runs it with inserted casts)
    fp16_path = os.path.join(dd, "model.fp16.onnx")
    try:
        import onnx
        from onnxconverter_common import float16
        m16 = float16.convert_float_to_float16(onnx.load(fp32_path), keep_io_types=True)
        onnx.save(m16, fp16_path)
        sess_fp16 = ort.InferenceSession(fp16_path, providers=["CPUExecutionProvider"])
        print(f"[export] fp16 onnx -> {fp16_path} ({os.path.getsize(fp16_path)/1e6:.1f} MB)")
    except Exception as e:
        sess_fp16 = None
        print(f"[export] fp16 conversion unavailable: {e}")

    for tag, pairs in checks.items():
        truths = [t for _, t in pairs]
        with torch.no_grad():
            e_t = embed_all(pairs, lambda x: wrapper(torch.from_numpy(x)).numpy())
        e_f = embed_all(pairs, lambda x: sess_fp32.run(None, {"image": x})[0])
        e_q = embed_all(pairs, lambda x: sess_int8.run(None, {"image": x})[0])
        report[tag] = {"torch": retrieval_top1(e_t, truths, emb, labels),
                       "onnx_fp32": retrieval_top1(e_f, truths, emb, labels),
                       "onnx_int8": retrieval_top1(e_q, truths, emb, labels),
                       "fp32_int8_cos": round(float(np.mean(np.sum(e_f * e_q, 1))), 4)}
        if sess_fp16 is not None:
            e_h = embed_all(pairs, lambda x: sess_fp16.run(None, {"image": x})[0])
            report[tag]["onnx_fp16"] = retrieval_top1(e_h, truths, emb, labels)
        print(f"[export] {tag}: {json.dumps(report[tag])}")

    def gate(kind):
        d1 = report["pjb"]["onnx_fp32"]["top1"] - report["pjb"][kind]["top1"]
        d2 = report["val"]["onnx_fp32"]["top1"] - report["val"][kind]["top1"]
        ok = d1 <= args.int8_max_drop and d2 <= args.int8_max_drop
        print(f"[export] gate {kind}: pjb drop {d1:.3f}, val drop {d2:.3f} -> "
              f"{'PASS' if ok else 'FAIL'}")
        return ok

    if gate("onnx_int8"):
        ship = "model.int8.onnx"
    elif sess_fp16 is not None and gate("onnx_fp16"):
        ship = "model.fp16.onnx"
    else:
        ship = "model.onnx"
    print(f"[export] shipping {ship}")
    report["ship"] = ship

    # ---------------------------------------------------------------- glyph data
    glyphs = {}
    with open(D_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = row.get("gardiner_num") or ""
            if k:
                glyphs[k] = {"char": row.get("hieroglyph", ""),
                             "desc": row.get("description", ""),
                             "details": row.get("details", ""),
                             "priority": (row.get("is_priority", "") or "").upper() == "TRUE"}
    json.dump(glyphs, open(os.path.join(dd, "glyphs.json"), "w"), ensure_ascii=False)
    print(f"[export] glyphs.json ({len(glyphs)} entries)")

    gd = os.path.join(APP, "glyphs")
    os.makedirs(gd, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(D_CANON)):
        if not f.lower().endswith(".png"):
            continue
        g = load_gray(os.path.join(D_CANON, f))
        h, w = g.shape
        s = 320.0 / max(h, w)
        g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(gd, f), g, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        n += 1
    print(f"[export] {n} thumbnails -> app/glyphs/")

    # ---------------------------------------------------------------- selftest
    fix_pairs = checks["pjb"][::97][:3] + checks["val"][:1]
    fixtures = []
    for p, t in fix_pairs:
        with torch.no_grad():
            e = wrapper(preprocess_path(p, size).unsqueeze(0)).numpy()
        r = retrieval_topk_list(e[0], emb, labels)
        fixtures.append({"name": os.path.basename(p), "truth": t,
                         "png_b64": base64.b64encode(open(p, "rb").read()).decode(),
                         "expected_top5": r})
    json.dump(fixtures, open(os.path.join(dd, "selftest.json"), "w"))
    print(f"[export] selftest.json ({len(fixtures)} fixtures)")

    # ---------------------------------------------------------------- config
    cfg_path = os.path.join(dd, "config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        cfg = json.load(open(cfg_path))
    cfg |= {"model": ship, "size": size, "thresh": 200, "pad": 6, "margin": 0.08,
            "top_k": 5}
    cfg.setdefault("score_threshold", 0.55)   # refined from probe P5/P7 results
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    json.dump(report, open(os.path.join(dd, "export_report.json"), "w"), indent=1)

    # ---------------------------------------------------------------- icons
    make_icons(os.path.join(APP, "icons"))
    print("[export] done.")


def retrieval_topk_list(q, index_emb, labels, k=5):
    best = {}
    sims = index_emb @ q
    for i, l in enumerate(labels):
        if l not in best or sims[i] > sims[best[l]]:
            best[l] = i
    ranked = sorted(best.items(), key=lambda kv: -sims[kv[1]])[:k]
    return [[l, round(float(sims[i]), 4)] for l, i in ranked]


def make_icons(outdir):
    os.makedirs(outdir, exist_ok=True)
    src = load_gray(os.path.join(D_CANON, "A1.png"))
    for px in (192, 512):
        bg = np.zeros((px, px, 3), np.uint8)
        bg[:] = (167, 94, 43)                                  # BGR lapis #2b5ea7
        h, w = src.shape
        s = px * 0.62 / max(h, w)
        g = cv2.resize(src, (max(1, round(w * s)), max(1, round(h * s))),
                       interpolation=cv2.INTER_AREA)
        mask = g < 160
        y0 = (px - g.shape[0]) // 2
        x0 = (px - g.shape[1]) // 2
        roi = bg[y0:y0 + g.shape[0], x0:x0 + g.shape[1]]
        roi[mask] = (240, 244, 248)
        cv2.imwrite(os.path.join(outdir, f"icon-{px}.png"), bg)


if __name__ == "__main__":
    main()
