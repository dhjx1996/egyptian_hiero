# Moving this project to another machine — Re-setup guide

**TL;DR:** copy/unzip the folder, then run **`bash misc/resetup.sh`**. That
rebuilds all environments (main scripts env + One-DM + `pipelines/matching`)
for the new location and platform. Done. Then follow `pipelines/README.md`
for the two product pipelines.

(2026-07-06 cleanup: the InkSight / PaliGemma / SwiftSketch / diffvg /
informative-drawings stacks were removed — not used by the two pipelines.
Recover them from the pre-cleanup copy of this directory if ever needed.)

## Can I just unzip and use it as-is?

**Only in one narrow case.** The virtual environments hardcode machine- and
path-specific details:

- `.venv/bin/python` is an **absolute** symlink, console-script shebangs are
  absolute (`#!/.../.venv/bin/python`), and `pyvenv.cfg` records an absolute
  base-interpreter path.
- The interpreter, `uv`, and all wheels are **Linux x86_64 / glibc** binaries.

So a raw unzip runs **only if** the target is **Linux x86_64** (compatible glibc)
**and** the folder lands at the **exact same absolute path**. Change the path,
the OS, or the CPU arch → broken symlinks/shebangs. (This is normal for any
Python venv.) Don't rely on it — re-run the bootstrap instead. It's quick.

## The reliable way: re-run the bootstrap

```bash
cd /wherever/egyptian_hiero
bash resetup.sh                 # main env + One-DM env
# or just the main scripts env:
SKIP_ONEDM=1 bash resetup.sh
```

What it does (all idempotent, all relative to wherever the folder now lives):
1. Ensures a **working `uv`** — re-downloads the correct build if the copied one
   won't run here (now **arch/OS-aware**: Linux/macOS, x86_64/arm64).
2. **Rebuilds the venvs** if the copied ones can't execute (stale path/machine),
   re-installing from `env/requirements.txt` / `One-DM/requirements-infer.txt`.
3. Re-registers the Jupyter kernels with the new absolute paths.
4. Verifies imports (including `cairosvg`, which needs system libcairo).

Works on any **Linux x86_64/arm64 or macOS** machine. Needs **internet** unless
you shipped the uv cache (see below).

## What to put in the zip

The `.venv`s and `.tools/` are **regenerable** — excluding them makes a much
smaller archive and avoids stale-symlink confusion:

```bash
# Lean archive (re-setup will re-download on the target; needs internet):
tar --exclude='.venv' --exclude='One-DM/.venv' --exclude='pipelines/matching/.venv' \
    --exclude='.tools' --exclude='**/__pycache__' \
    -czf ../egyptian_hiero.tgz -C .. egyptian_hiero
# (keep pipelines/matching/runs/ and One-DM/Saved/ if you want trained ckpts along)
```

Keep everything else (code, `scripts/`, `env/`, the cloned repos, `hiero_data/`,
docs). This drops ~1.8 GB of regenerable env files.

**Offline target?** The wheel cache (`misc/.tools/uv-cache`) was emptied in the
2026-07-06 cleanup (freed ~10 GB), so re-setup now needs internet. To prep an
offline move: on a networked machine of the same platform, run
`bash misc/resetup.sh` once (repopulates the cache), then ship the archive
including `misc/.tools/uv-cache/`.

## System dependencies on the target

- None beyond a working glibc: everything (opencv-headless, torch, …) is
  self-contained in the wheels.

## If the target has a GPU (A100, for the heavy ML)

The ML env scripts (`One-DM`, `pipelines/matching`) **auto-detect CUDA** via
`nvidia-smi`, so on a GPU box just run `bash misc/resetup.sh` — they install
CUDA torch automatically. Force/override with `DEVICE=cuda CUDA=cu121`
(A100/sm_80 works with cu118/cu121/cu124). Then follow `pipelines/*/README.md`
(the two product pipelines, incl. HPC training commands).
- One-DM training extras: `cd One-DM && bash setup_env.sh --train`.
- One-DM weights: `One-DM-ckpt.pt` + `model_zoo/sd-v1-5-vae` are kept in the
  tree. `vae_HTR138.pth` / `RN18_class_10400.pth` were removed (only the unused
  Latin `train_finetune.py` / scratch paths need them) — re-fetch via
  `One-DM/download_weights.py` if ever wanted.
