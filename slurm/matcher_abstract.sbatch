#!/bin/bash
#SBATCH --job-name=hiero-abstract
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/seshat/slurm/logs/abstract_%j.out
#SBATCH --error=/burg-archive/home/dh3065/seshat/slurm/logs/abstract_%j.err
# Abstraction-gap corrective run. IDENTICAL to review_grouped (group-disjoint
# split, resnet34@160, 100ep, bs512, p-handwrite 0.5, canon-repeat 12,
# p-dropstroke 0.10) with ONE new variable: --abstract, a stick-figure
# abstraction bank (make_abstractions.py) mixed in as canonical-style training
# items. review_grouped is the A/B baseline; the ONLY delta is abstraction, so
# any pjb/val move is attributable to it. Index recipe is unchanged (canonical +
# centroids, no abstract prototypes) so the ENCODER is the isolated variable.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/seshat
PJB=$ROOT/hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
ABSTRACT_PROBE=$ROOT/hiero_data/abstraction-probe
cd "$ROOT/pipelines/matching"
PY=./.venv/bin/python
NAME=review_grouped_abstract

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# regenerate the abstraction bank on-node (cheap, ~2s; clean dir first so no
# stale stroke-budget levels leak in). frac=1.0 = salience-preserving skeletal
# line-drawing (keeps the A39 stick figure); lower fracs drop the human -> noise.
echo "=== [0/5] (re)build abstraction bank ==="
rm -rf ../../hiero_data/abstractions
$PY make_abstractions.py --fracs 1.0 --workers 24

echo "=== [1/5] grouped-split train + abstraction (isolated variable) ==="
$PY review/train_grouped.py --name "$NAME" --arch resnet34 --size 160 \
    --epochs 100 --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12 \
    --p-dropstroke 0.10 --abstract ../../hiero_data/abstractions --abstract-repeat 6

echo "=== [2/5] build index (val groups excluded; standard recipe, no abstract prototypes) ==="
$PY build_index.py --ckpt runs/$NAME/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/$NAME/val_split.json --out runs/$NAME/index.npz

echo "=== [3/5] evaluate (group-disjoint val + pjb unseen-writer + real abstraction probe) ==="
$PY evaluate.py --ckpt runs/$NAME/best.pt --index runs/$NAME/index.npz \
    --val-split runs/$NAME/val_split.json --probe-dir "$PJB" --probe-dir "$ABSTRACT_PROBE" \
    --json-out runs/$NAME/eval.json

echo "=== [4/5] stress test (regression check vs review_grouped) ==="
$PY stress_test.py --ckpt runs/$NAME/best.pt --index runs/$NAME/index.npz \
    --val-split runs/$NAME/val_split.json --limit 800 \
    --json-out runs/$NAME/stress.json

echo "=== [5/5] also index WITH abstract prototypes (the shipped-app variant) ==="
$PY build_index.py --ckpt runs/$NAME/best.pt \
    --canonical ../../hiero_data/archaeohack-starterpack/data/utf-pngs \
    --canonical ../../hiero_data/abstractions/lvl1 \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/$NAME/val_split.json --out runs/$NAME/index_abstract.npz
$PY evaluate.py --ckpt runs/$NAME/best.pt --index runs/$NAME/index_abstract.npz \
    --val-split runs/$NAME/val_split.json --probe-dir "$PJB" --probe-dir "$ABSTRACT_PROBE" \
    --json-out runs/$NAME/eval_abstract_index.json

echo "=== abstraction corrective run DONE $(date) ==="
