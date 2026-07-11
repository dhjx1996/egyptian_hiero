#!/bin/bash
#SBATCH --job-name=hiero-match
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/match_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/match_%j.err
# Pipeline 2 (matching) — full training on ALL hiero_data.
# train_encoder (resnet34@160) -> build_index -> evaluate (held-out + unseen-writer pjb).
# 1x A100, 24 cores. Threads pinned to 1/process because the DataLoader spawns
# --workers processes that each do CPU pen-sim augmentation (numpy/PIL); letting
# BLAS/OMP fan out per worker would oversubscribe the shared node.
set -euo pipefail

# --- hard thread pin (whole node is visible; only cpus-per-task cores are ours) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd /burg-archive/home/dh3065/egyptian_hiero/pipelines/matching
PY=./.venv/bin/python

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="
echo "cpus-per-task=${SLURM_CPUS_PER_TASK:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "=== [1/3] train encoder ==="
$PY train_encoder.py --name a100 --arch resnet34 --size 160 \
    --epochs 100 --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12

echo "=== [2/3] build index ==="
$PY build_index.py --ckpt runs/a100/best.pt \
    --handwriting "../../hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/a100/val_split.json --out runs/a100/index.npz

echo "=== [3/3] evaluate (held-out handwriting + pjb unseen-writer) ==="
$PY evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json \
    --probe-dir ../../hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb

echo "=== matching pipeline DONE $(date) ==="
