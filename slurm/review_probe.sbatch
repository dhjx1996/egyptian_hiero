#!/bin/bash
#SBATCH --job-name=hiero-revprobe
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/revprobe_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/revprobe_%j.err
# 2026-07 adversarial review, probe suite: leakage / scan-frame reliance /
# canvas-domain gap / novel corruptions / open-set thresholds / prototype bias /
# calibration / test-time fixes / CPU latency. Smoke pass first (fail fast),
# then the full run. Results -> pipelines/matching/review/probe_results.json.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
cd "$ROOT/pipelines/matching"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== [1/2] smoke ==="
./.venv/bin/python review/adversarial_probe.py --smoke

echo "=== [2/2] full probe ==="
./.venv/bin/python review/adversarial_probe.py

echo "=== review probe DONE $(date) ==="
