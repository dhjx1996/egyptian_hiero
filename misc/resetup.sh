#!/usr/bin/env bash
# ============================================================================
# One-shot re-setup after unzipping / moving this project to a new location or
# machine. Rebuilds ALL environments (main scripts env + One-DM +
# pipelines/matching) for THIS path and platform. Safe to re-run (idempotent).
#
#   bash misc/resetup.sh                                         # rebuild all envs
#   SKIP_ONEDM=1 SKIP_MATCHING=1 bash misc/resetup.sh            # main env only
#
# Layout note: env/ and resetup.sh live under misc/, while One-DM/, pipelines/
# and hiero_data/ are at the REPO ROOT. This script accounts for that
# (MISC = this dir, REPO = its parent). The main bootstrap creates the
# <repo-root>/.tools -> misc/.tools bridge so the repo-root setup scripts find uv.
#
# GPU-first: this project targets GPUs. The ML setup_env.sh scripts auto-detect
# CUDA (GPU box -> CUDA torch / tensorflow[and-cuda]; CPU box -> CPU builds, only
# useful for dry import-checks). Force with DEVICE=cuda|cpu and CUDA=cu121|cu124|cu118.
# ============================================================================
set -euo pipefail
MISC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../egyptian_hiero/misc
REPO="$(cd "$MISC/.." && pwd)"                          # .../egyptian_hiero
echo "==> Re-setting up egyptian_hiero"
echo "    repo: $REPO"
echo "    misc: $MISC"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "    GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -) (CUDA builds)"
else
  echo "    GPU:  none detected -> CPU builds (import-check only for the ML envs)"
fi
echo

# Main scripts env (also creates the <repo-root>/.tools -> misc/.tools bridge).
bash "$MISC/env/bootstrap.sh"

# One-DM env (repo root). CUDA torch auto-detected.
if [ "${SKIP_ONEDM:-0}" = "1" ]; then
  echo; echo "==> One-DM env skipped (SKIP_ONEDM=1)"
elif [ -f "$REPO/One-DM/setup_env.sh" ]; then
  echo; echo "==> One-DM env (auto-detects CUDA)"
  bash "$REPO/One-DM/setup_env.sh"
else
  echo; echo "==> One-DM env: setup_env.sh not found at $REPO/One-DM (skipped)"
fi

# Matching pipeline env (repo root pipelines/matching). CUDA torch auto-detected.
if [ "${SKIP_MATCHING:-0}" = "1" ]; then
  echo; echo "==> matching env skipped (SKIP_MATCHING=1)"
elif [ -f "$REPO/pipelines/matching/setup_env.sh" ]; then
  echo; echo "==> matching pipeline env (auto-detects CUDA)"
  bash "$REPO/pipelines/matching/setup_env.sh"
else
  echo; echo "==> matching env: setup_env.sh not found at $REPO/pipelines/matching (skipped)"
fi

echo; echo "==> Verifying main env imports ..."
"$MISC/.venv/bin/python" - <<'PY'
import importlib
mods = ["numpy", "PIL", "cv2", "skimage", "scipy"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m} ({e.__class__.__name__}: {e})")
print("  main env:", "OK" if not bad else "ISSUES -> " + "; ".join(bad))
PY

echo; echo "Done. Activate the main env with:  source misc/env/activate.sh"
echo "ML envs: One-DM/.venv, pipelines/matching/.venv (Jupyter kernels: onedm, hieromatch)."
echo "Next: pipelines/README.md (the two pipelines) + pipelines/smoke_results/ (evidence)."
