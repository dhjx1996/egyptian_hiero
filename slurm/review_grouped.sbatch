#!/bin/bash
#SBATCH --job-name=hiero-revgrouped
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/revgrouped_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/revgrouped_%j.err
# 2026-07 adversarial review, leakage-corrected retrain: production recipe
# (resnet34@160, 100 ep, bs512, p-handwrite 0.5, canon-repeat 12, p-dropstroke
# 0.10) but with a GROUP-disjoint val split (whole source drawings held out,
# not individual augmented variants). Then index/evaluate/stress identically
# to the production run. Output: runs/review_grouped/.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
PJB=$ROOT/hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
cd "$ROOT/pipelines/matching"
PY=./.venv/bin/python

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== [1/4] grouped-split train (production recipe) ==="
$PY review/train_grouped.py --name review_grouped --arch resnet34 --size 160 \
    --epochs 100 --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12 \
    --p-dropstroke 0.10

echo "=== [2/4] build index (val groups excluded from centroids) ==="
$PY build_index.py --ckpt runs/review_grouped/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/review_grouped/val_split.json --out runs/review_grouped/index.npz

echo "=== [3/4] evaluate (group-disjoint val + pjb) ==="
$PY evaluate.py --ckpt runs/review_grouped/best.pt --index runs/review_grouped/index.npz \
    --val-split runs/review_grouped/val_split.json --probe-dir "$PJB" \
    --json-out runs/review_grouped/eval.json

echo "=== [4/4] stress test ==="
$PY stress_test.py --ckpt runs/review_grouped/best.pt --index runs/review_grouped/index.npz \
    --val-split runs/review_grouped/val_split.json --limit 800 \
    --json-out runs/review_grouped/stress.json

echo "=== review grouped retrain DONE $(date) ==="
