#!/bin/bash
#SBATCH --job-name=hiero-strokes
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/strokes_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/strokes_%j.err
# Targeted robustness question: missing strokes (dropstroke: white pen-skip
# slashes + minor-component deletion) and shaky-hand strokes (wobble: strong
# short-wavelength elastic). Full suite rerun for a single consistent table.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
cd "$ROOT/pipelines/matching"
echo "=== $(date) host=$(hostname) ==="
./.venv/bin/python stress_test.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json --device cuda --limit 800 \
    --json-out runs/a100/stress_strokes.json
echo "=== STROKES DONE $(date) ==="
