#!/bin/bash
#SBATCH --job-name=hiero-frames
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/seshat/slurm/logs/frames_%j.out
#SBATCH --error=/burg-archive/home/dh3065/seshat/slurm/logs/frames_%j.err
# Run 2 (review F2). review_grouped recipe with ONE new variable: --p-frame 0.5
# (random synthetic scan-cell borders on all sources -> frame becomes an
# uninformative nuisance the encoder must ignore). Uses train_encoder.py, whose
# split default is now group-disjoint (byte-identical to review_grouped's 7344),
# so this is a clean A/B vs the review_grouped baseline. Frame robustness is read
# from stress_test's new `frame` corruption; regression gate is pjb/grouped-val.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/seshat
PJB=$ROOT/hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
ABSTRACT_PROBE=$ROOT/hiero_data/abstraction-probe
cd "$ROOT/pipelines/matching"
PY=./.venv/bin/python
NAME=review_grouped_frames

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# baseline re-stress with the NEW corruption set (frame/half_top) so F2/F4 have an
# apples-to-apples baseline; run first so a train crash can't cost it.
echo "=== [0/5] baseline review_grouped re-stress (new corruptions) ==="
$PY stress_test.py --ckpt runs/review_grouped/best.pt --index runs/review_grouped/index.npz \
    --val-split runs/review_grouped/val_split.json --limit 800 \
    --json-out runs/review_grouped/stress_new.json || echo "WARN baseline stress failed"

echo "=== [1/5] train: review_grouped recipe + frame aug (isolated variable) ==="
# workers 16->12: worker count doesn't change the seeded shuffle/aug (verified), but
# fewer worker shm sockets avoids the transient 'pin memory thread exited' crash
# that killed job 9114535 at epoch 44.
$PY train_encoder.py --name "$NAME" --arch resnet34 --size 160 \
    --epochs 100 --batch-size 512 --workers 12 --p-handwrite 0.5 --canon-repeat 12 \
    --p-dropstroke 0.10 --p-frame 0.5

echo "=== [2/5] build index (val excluded) ==="
$PY build_index.py --ckpt runs/$NAME/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/$NAME/val_split.json --out runs/$NAME/index.npz

echo "=== [3/5] evaluate (grouped val + pjb + A39 abstraction probe) ==="
$PY evaluate.py --ckpt runs/$NAME/best.pt --index runs/$NAME/index.npz \
    --val-split runs/$NAME/val_split.json --probe-dir "$PJB" --probe-dir "$ABSTRACT_PROBE" \
    --json-out runs/$NAME/eval.json

echo "=== [4/5] stress (incl. new frame/half_top corruptions) ==="
$PY stress_test.py --ckpt runs/$NAME/best.pt --index runs/$NAME/index.npz \
    --val-split runs/$NAME/val_split.json --limit 800 --json-out runs/$NAME/stress.json

echo "=== frames run DONE $(date) ==="
