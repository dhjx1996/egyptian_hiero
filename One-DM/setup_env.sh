#!/usr/bin/env bash
# ============================================================================
# Isolated, persistent env for One-DM (separate from the main scripts env
# because One-DM pins an old / conflicting dependency stack).
#
# Creates ./One-DM/.venv on the persistent NFS mount, reusing the project's
# cached uv binary + managed CPython, and installs a CPU-only inference stack.
# (Training needs GPUs -> see requirements-train.txt; cannot run on this box.)
#
# Usage:   cd One-DM && bash setup_env.sh
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGY="$(cd "$HERE/.." && pwd)"
export UV_CACHE_DIR="$EGY/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$EGY/.tools/python"
UV="$EGY/.tools/bin/uv"
VENV="$HERE/.venv"
PYVER="3.11"

[ -x "$UV" ] || { echo "uv not found at $UV - run ../env/bootstrap.sh first"; exit 1; }

if ! "$VENV/bin/python" -c "pass" >/dev/null 2>&1; then
  [ -e "$VENV" ] && { echo "[onedm] existing venv is stale -> rebuilding ..."; rm -rf "$VENV"; }
  echo "[onedm] creating venv (CPython $PYVER) ..."
  "$UV" venv --python "$PYVER" --python-preference only-managed "$VENV"
fi

# Device: CUDA on a GPU box (e.g. A100), else CPU. Override with DEVICE=cuda|cpu
# and CUDA=cu121|cu124|cu118 (default cu121; A100/sm_80 works with all of these).
if [ -z "${DEVICE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then DEVICE=cuda; else DEVICE=cpu; fi
fi
if [ "$DEVICE" = "cuda" ]; then
  CUDA="${CUDA:-cu121}"
  echo "[onedm] installing CUDA torch + torchvision ($CUDA) ..."
  "$UV" pip install --python "$VENV/bin/python" \
    torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
else
  echo "[onedm] installing CPU torch + torchvision ..."
  "$UV" pip install --python "$VENV/bin/python" \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[onedm] installing inference requirements ..."
"$UV" pip install --python "$VENV/bin/python" -r "$HERE/requirements-infer.txt"

# Training extras (multi-GPU runs): bash setup_env.sh --train  (or TRAIN=1)
if [ "${1:-}" = "--train" ] || [ "${TRAIN:-0}" = "1" ]; then
  echo "[onedm] installing training extras (requirements-train.txt) ..."
  "$UV" pip install --python "$VENV/bin/python" -r "$HERE/requirements-train.txt"
fi

echo "[onedm] registering Jupyter kernel 'onedm' ..."
"$UV" pip install --python "$VENV/bin/python" ipykernel >/dev/null 2>&1 || true
"$VENV/bin/python" -m ipykernel install --user \
  --name onedm --display-name "Python (One-DM)" >/dev/null 2>&1 \
  && echo "[onedm] kernel registered." || echo "[onedm] (kernel registration skipped)"

echo
echo "[onedm] DONE ($DEVICE). Python: $VENV/bin/python"
echo "  GPU box: re-run on the A100 (auto-detects CUDA); add --train for training deps."
