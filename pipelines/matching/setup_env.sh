#!/usr/bin/env bash
# ============================================================================
# Isolated, persistent env for the matching pipeline (Pipeline 2).
# Mirrors One-DM/setup_env.sh: uv + project-managed CPython on the NFS mount,
# CUDA torch auto-detected (override: DEVICE=cuda|cpu, CUDA=cu121|cu124|cu118).
#
# Usage:   cd pipelines/matching && bash setup_env.sh
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGY="$(cd "$HERE/../.." && pwd)"
export UV_CACHE_DIR="$EGY/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$EGY/.tools/python"
UV="$EGY/.tools/bin/uv"
VENV="$HERE/.venv"
PYVER="3.11"

[ -x "$UV" ] || { echo "uv not found at $UV - run misc/env/bootstrap.sh first"; exit 1; }

if ! "$VENV/bin/python" -c "pass" >/dev/null 2>&1; then
  [ -e "$VENV" ] && { echo "[matching] existing venv is stale -> rebuilding ..."; rm -rf "$VENV"; }
  echo "[matching] creating venv (CPython $PYVER) ..."
  "$UV" venv --python "$PYVER" --python-preference only-managed "$VENV"
fi

if [ -z "${DEVICE:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then DEVICE=cuda; else DEVICE=cpu; fi
fi
if [ "$DEVICE" = "cuda" ]; then
  CUDA="${CUDA:-cu121}"
  echo "[matching] installing CUDA torch + torchvision ($CUDA) ..."
  "$UV" pip install --python "$VENV/bin/python" \
    torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
else
  echo "[matching] installing CPU torch + torchvision ..."
  "$UV" pip install --python "$VENV/bin/python" \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[matching] installing requirements ..."
"$UV" pip install --python "$VENV/bin/python" -r "$HERE/requirements.txt"

echo "[matching] registering Jupyter kernel 'hieromatch' ..."
"$UV" pip install --python "$VENV/bin/python" ipykernel >/dev/null 2>&1 || true
"$VENV/bin/python" -m ipykernel install --user \
  --name hieromatch --display-name "Python (hiero-matching)" >/dev/null 2>&1 \
  && echo "[matching] kernel registered." || echo "[matching] (kernel registration skipped)"

echo
echo "[matching] DONE ($DEVICE). Python: $VENV/bin/python"
echo "  next: cd pipelines/matching && ./.venv/bin/python train_encoder.py --help"
