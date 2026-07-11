#!/bin/bash
#SBATCH --job-name=hiero-prep
#SBATCH --account=crew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/prep_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/prep_%j.err
# One-time CPU prep: map ALL hiero_data into One-DM's training layout.
# --threads 12 also becomes the generated config's DATA_LOADER.NUM_THREADS, which
# the 2-GPU DDP training reuses per-rank (2x12=24 <= 28 cores). Keep them equal.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /burg-archive/home/dh3065/egyptian_hiero
PY=One-DM/.venv/bin/python

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} ==="
$PY pipelines/generation/prep_dataset.py \
    --handwriting "hiero_data/Hand-drawn Hieroglyph Dataset" \
    --canonical   hiero_data/archaeohack-starterpack/data/utf-pngs \
    --unicode-json hiero_data/archaeohack-starterpack/data/gardiner_hieroglyphs_with_unicode_hex.json \
    --name hiero --threads 12 --epochs 200 --batch-size 96 --snapshot-every 10

echo "=== prep summary ==="
wc -l One-DM/data/hiero_train.txt One-DM/data/hiero_test.txt One-DM/data/hiero_letters.txt
echo "train wids:"; ls One-DM/data/hiero/IAM64-new/train | wc -l
echo "=== prep DONE $(date) ==="
