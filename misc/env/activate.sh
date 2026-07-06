# ============================================================================
# Activate the portable egyptian_hiero environment in your current shell.
# Usage:   source env/activate.sh
# ============================================================================
_EGY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="$_EGY/.tools/uv-cache"
export UV_PYTHON_INSTALL_DIR="$_EGY/.tools/python"
export PATH="$_EGY/.tools/bin:$PATH"          # puts 'uv' on PATH
# shellcheck disable=SC1091
source "$_EGY/.venv/bin/activate"
echo "egyptian_hiero env active: $(python --version) @ $(command -v python)"
echo "  add a package:  uv pip install <pkg>   (then add it to env/requirements.txt)"
