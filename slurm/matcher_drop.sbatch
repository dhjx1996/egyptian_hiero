#!/bin/bash
#SBATCH --job-name=hiero-matchdrop
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/matchdrop_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/matchdrop_%j.err
# Retrain the matcher with missing-stroke augmentation (--p-dropstroke 0.25).
# Motivation: dropstroke is the measured weak spot (top1 0.680 vs 0.969 clean).
# Recipe otherwise identical to runs/a100. Acceptance: dropstroke top1 up
# materially, no clean/held-out/pjb regression.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
PJB=$ROOT/hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb
cd "$ROOT/pipelines/matching"
echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="

./.venv/bin/python train_encoder.py --name a100_drop --arch resnet34 --size 160 --epochs 100 \
    --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12 --p-dropstroke 0.25

./.venv/bin/python build_index.py --ckpt runs/a100_drop/best.pt \
    --handwriting "$ROOT/hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/a100_drop/val_split.json --out runs/a100_drop/index.npz

echo "--- clean eval: held-out + unseen writer ---"
./.venv/bin/python evaluate.py --ckpt runs/a100_drop/best.pt --index runs/a100_drop/index.npz \
    --val-split runs/a100_drop/val_split.json --probe-dir "$PJB" --json-out runs/a100_drop/eval.json

echo "--- corruption suite ---"
./.venv/bin/python stress_test.py --ckpt runs/a100_drop/best.pt --index runs/a100_drop/index.npz \
    --val-split runs/a100_drop/val_split.json --device cuda --limit 800 \
    --json-out runs/a100_drop/stress.json

echo "--- comparison vs baseline runs/a100 ---"
./.venv/bin/python - <<'PY'
import json
a = json.load(open("runs/a100/stress_strokes.json"))["corruptions"]
b = json.load(open("runs/a100_drop/stress.json"))["corruptions"]
print(f"{'corruption':<12}{'baseline':>10}{'drop-aug':>10}{'delta':>9}")
for c in a:
    if c in b:
        print(f"{c:<12}{a[c]['top1']:>10.3f}{b[c]['top1']:>10.3f}{b[c]['top1']-a[c]['top1']:>+9.3f}")
PY
echo "=== MATCHDROP DONE $(date) ==="
