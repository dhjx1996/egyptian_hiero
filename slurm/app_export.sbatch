#!/bin/bash
#SBATCH --job-name=hiero-appexport
#SBATCH --account=crew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/appexport_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/appexport_%j.err
# Dictionary-app asset build (CPU): export the production encoder to ONNX
# (fp32 + int8-QDQ with accuracy gate), pack the prototype index, glyph
# thumbnails/data, PWA icons and selftest fixtures; then verify the ACTUAL app
# JS pipeline end-to-end under node (onnxruntime-web) against torch references.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
ROOT=/burg-archive/home/dh3065/egyptian_hiero
UV=$ROOT/.tools/bin/uv
export UV_CACHE_DIR=$ROOT/.tools/uv-cache
PYBIN=$ROOT/pipelines/matching/.venv/bin/python
export PATH=/burg-archive/home/dh3065/.nvm/versions/node/v24.18.0/bin:$PATH

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="

echo "=== [1/4] deps (onnx + onnxruntime into the matching venv) ==="
"$UV" pip install --python "$PYBIN" onnx onnxruntime onnxconverter-common

echo "=== [2/4] export app assets ==="
cd "$ROOT/pipelines/matching"
"$PYBIN" "$ROOT/app/build/export_app_assets.py"

echo "=== [3/4] vendor onnxruntime-web + node end-to-end verification ==="
cd "$ROOT/app/build"
if npm install --no-audit --no-fund; then
  mkdir -p "$ROOT/app/vendor/ort"
  cp node_modules/onnxruntime-web/dist/ort.min.js \
     node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm \
     "$ROOT/app/vendor/ort/"

  echo "=== [4/4] pick the smallest model that passes the REAL JS pipeline ==="
  # The Python-side accuracy gate can pass a model the wasm runtime then
  # mishandles (seen with fp16), so the node test has the final say.
  CANDIDATE=$("$PYBIN" -c "import json;print(json.load(open('$ROOT/app/data/config.json'))['model'])")
  SHIPPED=""
  for M in $(printf "%s\nmodel.onnx\n" "$CANDIDATE" | uniq); do
    if node test_app_pipeline.mjs --model "$M"; then SHIPPED=$M; break
    else echo "WARN: $M failed the JS pipeline test"; fi
  done
  [ -n "$SHIPPED" ] || { echo "ERROR: no model passes the JS pipeline"; exit 1; }
  "$PYBIN" - <<PY
import json
p = "$ROOT/app/data/config.json"
c = json.load(open(p)); c["model"] = "$SHIPPED"
json.dump(c, open(p, "w"), indent=1)
print("config.json model ->", "$SHIPPED")
PY
  # keep only the shipped model in the deployable tree (report keeps the numbers)
  find "$ROOT/app/data" -name 'model*.onnx' ! -name "$SHIPPED" -delete
else
  echo "WARN: npm install failed (no network on this node?) - run steps 3/4 manually"
fi

echo "--- asset sizes ---"
du -sh "$ROOT"/app/data/* "$ROOT"/app/glyphs 2>/dev/null | sort -k2
echo "=== app export DONE $(date) ==="
