#!/bin/bash
#SBATCH --job-name=hiero-gensmoke
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/gensmoke_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/gensmoke_%j.err
# De-risk the generate.py --engine onedm integration path (never run before) using
# whatever fine-tuned checkpoint already exists. Small: 3 signs x 2. Then score
# recognizability with the matcher so we know the full stress run will work.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero; cd "$ROOT"
MPY=pipelines/matching/.venv/bin/python

echo "=== $(date) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="
CKPT=$(ls -1 One-DM/Saved/hiero/*/*/model/*-ckpt.pt 2>/dev/null \
  | awk -F'/model/' '{n=$2; sub(/-ckpt.pt/,"",n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "using ckpt: $CKPT"
SIGNS=$(One-DM/.venv/bin/python - <<'PY'
import json
d=json.load(open("One-DM/data/hiero_wid_map.json"))
print(",".join(list((d.get("classes") or d).keys())[:3]))
PY
)
echo "signs: $SIGNS"

python3 pipelines/generation/generate.py --engine onedm --signs "$SIGNS" --n 2 \
    --ckpt "$CKPT" --out misc/outputs/gensmoke
echo "--- generated files ---"; find misc/outputs/gensmoke/onedm -name '*.png' | head

echo "--- recognizability (matcher on generated) ---"
( cd pipelines/matching && ../../$MPY evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --probe-dir ../../misc/outputs/gensmoke/onedm )
echo "=== gen smoke DONE $(date) ==="
