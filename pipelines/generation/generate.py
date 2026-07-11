#!/usr/bin/env python3
"""
Pipeline 1 front-end: generate handwriting samples for any symbol.

Dispatches to two engines (each runs in its own venv via subprocess, so this
script itself needs only stock python3):

  procedural  skeletonize + wobble (misc/scripts). CPU, zero training, works for
              any glyph image today. Good baseline + augmentation source.
  onedm       One-DM latent-diffusion mimicker. GPU; faithful after fine-tuning
              on a handwriting corpus (see README.md / prep_dataset.py).

Examples (from the repo root):
  python3 pipelines/generation/generate.py --signs A1,D21,N35 --n 6
  python3 pipelines/generation/generate.py --engine onedm --signs A1 --n 4 \
      --ckpt One-DM/Saved/hiero/.../200-ckpt.pt
  python3 pipelines/generation/generate.py --engine both --glyph /path/mayan_glyph.png
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ONEDM = os.path.join(REPO, "One-DM")
SCRIPTS_PY = os.path.join(REPO, "misc", ".venv", "bin", "python")
ONEDM_PY = os.path.join(ONEDM, ".venv", "bin", "python")
DEFAULT_CANONICAL = os.path.join(REPO, "hiero_data", "archaeohack-starterpack", "data", "utf-pngs")


def run(cmd, cwd=None):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd)
    if p.returncode != 0:
        sys.exit(p.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="procedural", choices=["procedural", "onedm", "both"])
    ap.add_argument("--signs", default="", help="comma-separated sign/class names, or 'all'")
    ap.add_argument("--glyph", action="append", default=[], help="ANY glyph image path (repeatable)")
    ap.add_argument("--n", type=int, default=6, help="variants per symbol")
    ap.add_argument("--out", default=os.path.join(REPO, "misc", "outputs", "generated"))
    ap.add_argument("--canonical-dir", default=DEFAULT_CANONICAL)
    # procedural knobs
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out-size", type=int, default=0)
    ap.add_argument("--stroke-budget", type=int, default=99)
    ap.add_argument("--no-simplify", action="store_true")
    # one-dm knobs
    ap.add_argument("--dataset-name", default="hiero", help="prep_dataset --name (drives cfg/content/charset paths)")
    ap.add_argument("--ckpt", default=os.path.join(ONEDM, "model_zoo", "One-DM-ckpt.pt"),
                    help="One-DM checkpoint (use your fine-tuned Saved/... ckpt for real quality)")
    ap.add_argument("--style-class", default="same", help="'same', 'random', or a class name")
    ap.add_argument("--style-dir", default="", help="ANY folder of handwriting samples as the style")
    ap.add_argument("--style-split", default="train", choices=["train", "test"])
    ap.add_argument("--timesteps", type=int, default=25)
    ap.add_argument("--cfg-scale", type=float, default=1.0,
                    help="classifier-free guidance weight (ckpt must be trained with COND_DROP_PROB > 0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    if not args.signs and not args.glyph:
        ap.error("pass --signs and/or --glyph")

    if args.engine in ("procedural", "both"):
        if not os.path.isfile(SCRIPTS_PY):
            sys.exit(f"main scripts env missing ({SCRIPTS_PY}) - run: bash misc/resetup.sh")
        cmd = [SCRIPTS_PY, os.path.join(HERE, "procedural_engine.py"),
               "--n", str(args.n), "--out", os.path.join(args.out, "procedural"),
               "--canonical-dir", args.canonical_dir, "--size", str(args.size),
               "--seed", str(args.seed), "--stroke-budget", str(args.stroke_budget)]
        if args.out_size:
            cmd += ["--out-size", str(args.out_size)]
        if args.no_simplify:
            cmd += ["--no-simplify"]
        if args.signs:
            cmd += ["--signs", args.signs]
        for g in args.glyph:
            cmd += ["--glyph", os.path.abspath(g)]
        run(cmd)

    if args.engine in ("onedm", "both"):
        if not os.path.isfile(ONEDM_PY):
            sys.exit(f"One-DM env missing ({ONEDM_PY}) - run: bash misc/resetup.sh")
        name = args.dataset_name
        cmd = [ONEDM_PY, "generate_hiero.py",
               "--cfg", f"configs/{name}.yml",
               "--one_dm", os.path.abspath(args.ckpt),
               "--stable_dif_path", "model_zoo/sd-v1-5-vae",
               "--content_type", f"{name}_content",
               "--letters", f"data/{name}_letters.txt",
               "--wid-map", f"data/{name}_wid_map.json",
               "--n", str(args.n), "--out", os.path.join(os.path.abspath(args.out), "onedm"),
               "--style-class", args.style_class, "--style-split", args.style_split,
               "--sampling_timesteps", str(args.timesteps), "--seed", str(args.seed),
               "--cfg-scale", str(args.cfg_scale)]
        if args.signs:
            cmd += ["--signs", args.signs]
        if args.style_dir:
            cmd += ["--style-dir", os.path.abspath(args.style_dir)]
        if args.device:
            cmd += ["--device", args.device]
        for g in args.glyph:
            cmd += ["--glyph", os.path.abspath(g)]
        run(cmd, cwd=ONEDM)

    print(f"[generate] outputs under {args.out}/<engine>/<symbol>/")


if __name__ == "__main__":
    main()
