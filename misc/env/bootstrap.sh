#!/usr/bin/env bash
# ============================================================================
# Portable-environment bootstrap for the egyptian_hiero project.
#
# Re-creates / repairs the project-local uv virtual environment, which lives on
# the PERSISTENT NFS mount (/home/jovyan) so it survives JupyterHub container /
# overlay resets. Everything (uv binary, the managed CPython, the wheel cache,
# and the .venv) lives under this project tree -> nothing depends on the
# ephemeral /srv/conda environment that is wiped on shutdown.
#
# Idempotent: safe to re-run. After a NORMAL reset you usually do NOT need this
# (the .venv persists and just works); run it only on first setup or to repair.
#
# Usage:   bash env/bootstrap.sh
# ============================================================================
set -euo pipefail

EGY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$EGY/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$EGY/.tools/python"
UV="$EGY/.tools/bin/uv"
VENV="$EGY/.venv"
PYVER="3.11"

# 1) Ensure a WORKING uv. Re-download if missing OR if a copied binary won't run
#    on this platform (e.g. moved from x86_64 to arm64/macOS). Arch-aware; no curl
#    needed (fetch with Python urllib).
if ! "$UV" --version >/dev/null 2>&1; then
  echo "[bootstrap] fetching uv for this platform ..."
  mkdir -p "$EGY/.tools/bin"
  python3 - "$EGY/.tools" <<'PY'
import urllib.request, tarfile, shutil, os, sys, platform
root = sys.argv[1]
m = platform.machine().lower()
s = platform.system().lower()
arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(m)
if arch is None:
    sys.exit(f"[bootstrap] unsupported arch {m!r}; install uv manually: https://astral.sh/uv")
if s == "linux":
    triple = f"{arch}-unknown-linux-gnu"
elif s == "darwin":
    triple = f"{arch}-apple-darwin"
else:
    sys.exit(f"[bootstrap] unsupported OS {s!r} (Windows: install uv manually from https://astral.sh/uv)")
url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{triple}.tar.gz"
print("[bootstrap] downloading", url)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
tgz = os.path.join(root, "uv.tar.gz")
with urllib.request.urlopen(req, timeout=180) as r, open(tgz, "wb") as f:
    shutil.copyfileobj(r, f)
with tarfile.open(tgz) as t:
    t.extractall(root)
d = os.path.join(root, f"uv-{triple}")
for b in ("uv", "uvx"):
    src = os.path.join(d, b)
    if os.path.exists(src):
        dst = os.path.join(root, "bin", b)
        shutil.copy(src, dst); os.chmod(dst, 0o755)
shutil.rmtree(d, ignore_errors=True); os.remove(tgz)
print("[bootstrap] uv installed.")
PY
fi
echo "[bootstrap] uv: $("$UV" --version)"

# 1b) Bridge for the reorganized layout: env/ + resetup.sh live under misc/, so
#     uv + the managed CPython + wheel cache land in misc/.tools. But One-DM/ and
#     pipelines/matching/ are at the REPO ROOT and their setup_env.sh look for
#     <repo-root>/.tools/bin/uv. Symlink <repo-root>/.tools -> misc/.tools (relative,
#     so it survives moving the folder) whenever those repo-root scripts exist.
REPO="$(cd "$EGY/.." && pwd)"
if [ -f "$REPO/One-DM/setup_env.sh" ] || [ -f "$REPO/pipelines/matching/setup_env.sh" ]; then
  if [ ! -e "$REPO/.tools" ]; then
    ln -s "$(basename "$EGY")/.tools" "$REPO/.tools" \
      && echo "[bootstrap] linked $REPO/.tools -> $(basename "$EGY")/.tools (repo-root ML scripts can find uv)" \
      || echo "[bootstrap] (could not create .tools bridge; repo-root ML setup_env.sh may not find uv)"
  fi
fi

# 2) Ensure the venv (project-local managed CPython -> persistent). If a copied
#    venv can't actually execute (moved path/machine -> broken symlinks/shebangs),
#    rebuild it from scratch.
if ! "$VENV/bin/python" -c "pass" >/dev/null 2>&1; then
  [ -e "$VENV" ] && { echo "[bootstrap] existing venv is stale -> rebuilding ..."; rm -rf "$VENV"; }
  echo "[bootstrap] creating venv (CPython $PYVER) at $VENV ..."
  "$UV" venv --python "$PYVER" --python-preference only-managed "$VENV"
else
  echo "[bootstrap] venv already present and runnable at $VENV"
fi

# 3) Install / update requirements (this is where you 'add packages to it').
echo "[bootstrap] syncing env/requirements.txt ..."
"$UV" pip install --python "$VENV/bin/python" -r "$EGY/env/requirements.txt"

# 4) Register a Jupyter kernel (kernelspec -> ~/.local/share/jupyter, persistent).
echo "[bootstrap] registering Jupyter kernel 'egyptian_hiero' ..."
"$VENV/bin/python" -m ipykernel install --user \
  --name egyptian_hiero --display-name "Python (egyptian_hiero)" >/dev/null 2>&1 \
  && echo "[bootstrap] kernel registered." \
  || echo "[bootstrap] (kernel registration skipped/failed - non-fatal)"

echo
echo "[bootstrap] DONE."
echo "  Activate (shell):  source $EGY/env/activate.sh"
echo "  Python:            $VENV/bin/python"
echo "  Jupyter:           pick kernel 'Python (egyptian_hiero)'"
