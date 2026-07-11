#!/bin/bash
#SBATCH --job-name=hiero-prep128
#SBATCH --account=crew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/prep128_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/prep128_%j.err
# 128px TARGET image tree for the One-DM high-res fine-tune. Same seed as the
# 64px 'hiero' prep => identical class wids, file stems and train/test split, so
# the training config can mix 128px targets (IAMGE_PATH) with the existing 64px
# style/laplace trees (STYLE_PATH/LAPLACE_PATH) -- the style encoder hardcodes
# 64px-input feature shapes. --max-width 512 keeps aspect<=4 undistorted.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero; cd "$ROOT"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="
One-DM/.venv/bin/python pipelines/generation/prep_dataset.py \
    --handwriting "hiero_data/Hand-drawn Hieroglyph Dataset" \
    --canonical hiero_data/archaeohack-starterpack/data/utf-pngs \
    --unicode-json hiero_data/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --name hiero128 --height 128 --max-width 512 \
    --epochs 100 --batch-size 48 --snapshot-every 5 --threads 12

echo "--- consistency check vs 64px tree (same split expected) ---"
One-DM/.venv/bin/python - <<'PY'
import json
a = json.load(open("One-DM/data/hiero_wid_map.json"))["classes"]
b = json.load(open("One-DM/data/hiero128_wid_map.json"))["classes"]
assert list(a) == list(b), "class order differs!"
bad = [c for c in a if (a[c]["wid"], a[c]["n_train"], a[c]["n_test"]) != (b[c]["wid"], b[c]["n_train"], b[c]["n_test"])]
assert not bad, f"split mismatch for {bad[:5]}"
ta = open("One-DM/data/hiero_train.txt").read(); tb = open("One-DM/data/hiero128_train.txt").read()
assert ta == tb, "train index differs"
print("OK: hiero128 split identical to hiero (64px)")
PY
echo "=== prep128 DONE $(date) ==="
