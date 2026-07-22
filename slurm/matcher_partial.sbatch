#!/bin/bash
#SBATCH --job-name=hiero-partial
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/seshat/slurm/logs/partial_%j.out
#SBATCH --error=/burg-archive/home/dh3065/seshat/slurm/logs/partial_%j.err
# Run 3 (review F4). review_grouped recipe with ONE new variable: --p-partial 0.5
# (keep only a contiguous PART of the drawing -> the encoder degrades
# gracefully on half-finished draws instead of 0.09 top1). Uses train_encoder.py, whose
# split default is now group-disjoint (byte-identical to review_grouped's 7344),
# so this is a clean A/B vs the review_grouped baseline. Half-draw robustness is
# read from stress_test's new `half_top` corruption; regression gate is pjb/val.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/seshat
PJB=$ROOT/hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
ABSTRACT_PROBE=$ROOT/hiero_data/abstraction-probe
cd "$ROOT/pipelines/matching"
PY=./.venv/bin/python
NAME=review_grouped_partial

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== [1/4] train: review_grouped recipe + partial-stroke aug (isolated variable) ==="
$PY train_encoder.py --name "$NAME" --arch resnet34 --size 160 \
    --epochs 100 --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12 \
    --p-dropstroke 0.10 --p-partial 0.5

echo "=== [2/4] build index (val excluded) ==="
$PY build_index.py --ckpt runs/$NAME/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/$NAME/val_split.json --out runs/$NAME/index.npz

echo "=== [3/4] evaluate (grouped val + pjb + A39 abstraction probe) ==="
$PY evaluate.py --ckpt runs/$NAME/best.pt --index runs/$NAME/index.npz \
    --val-split runs/$NAME/val_split.json --probe-dir "$PJB" --probe-dir "$ABSTRACT_PROBE" \
    --json-out runs/$NAME/eval.json

echo "=== [4/4] stress (incl. new frame/half_top corruptions) ==="
$PY stress_test.py --ckpt runs/$NAME/best.pt --index runs/$NAME/index.npz \
    --val-split runs/$NAME/val_split.json --limit 800 --json-out runs/$NAME/stress.json

echo "=== partial run DONE $(date) ==="
