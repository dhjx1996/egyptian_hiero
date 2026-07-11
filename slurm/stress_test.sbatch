#!/bin/bash
#SBATCH --job-name=hiero-stress
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/stress_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/stress_%j.err
# Reviewer-grade stress tests, run AFTER both trainings.
#  - matcher: corruption robustness, latency, confusions, held-out + unseen-writer pjb
#  - generation: procedural + One-DM recognizability (matcher scores generated glyphs)
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ROOT=/burg-archive/home/dh3065/egyptian_hiero
cd "$ROOT"
MPY=pipelines/matching/.venv/bin/python
RUN=pipelines/matching/runs/a100
PJB=hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="

# --- best fine-tuned One-DM checkpoint + guidance scale (from gen_select);
#     falls back to the newest Saved/hiero ckpt at scale 1.0 ---
GEN_SCALE=1.0
GEN_DATASET=hiero
if [ -f slurm/best_gen.env ]; then
  source slurm/best_gen.env
  ONEDM_CKPT=$BEST_CKPT
  GEN_SCALE=$BEST_SCALE
  GEN_DATASET=$BEST_DATASET
else
  ONEDM_CKPT=$(ls -1 One-DM/Saved/hiero/*/*/model/*-ckpt.pt 2>/dev/null \
    | awk -F'/model/' '{n=$2; sub(/-ckpt.pt/,"",n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2- || true)
fi
echo "One-DM ckpt: ${ONEDM_CKPT:-<none found>}  scale=$GEN_SCALE dataset=$GEN_DATASET"

# --- 20 in-vocab classes from the One-DM wid map (guaranteed generatable) ---
SIGNS=$(One-DM/.venv/bin/python - <<'PY'
import json, os
p="One-DM/data/hiero_wid_map.json"
d=json.load(open(p))
# wid map may be {class: {...}} or {"classes": {...}}; grab class keys
keys=list((d.get("classes") or d).keys())
print(",".join(keys[:20]))
PY
)
echo "signs: $SIGNS"

echo; echo "########## [1] MATCHER — held-out + unseen-writer ##########"
( cd pipelines/matching && ../../$MPY evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json --probe-dir "../../$PJB" \
    --json-out runs/a100/eval.json )

echo; echo "########## [2] MATCHER — corruption robustness + latency + confusions ##########"
( cd pipelines/matching && ../../$MPY stress_test.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json --device cuda --limit 1200 --json-out runs/a100/stress.json )

echo; echo "########## [3] PROCEDURAL generation -> recognizability ##########"
python3 pipelines/generation/generate.py --engine procedural --signs "$SIGNS" --n 8 \
    --out misc/outputs/stress
( cd pipelines/matching && ../../$MPY evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --probe-dir ../../misc/outputs/stress/procedural )

if [ -n "${ONEDM_CKPT:-}" ]; then
  echo; echo "########## [4] ONE-DM generation -> recognizability ##########"
  python3 pipelines/generation/generate.py --engine onedm --signs "$SIGNS" --n 8 \
      --ckpt "$ONEDM_CKPT" --timesteps 50 --cfg-scale "$GEN_SCALE" \
      --dataset-name "$GEN_DATASET" --out misc/outputs/stress
  ( cd pipelines/matching && ../../$MPY evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
      --probe-dir ../../misc/outputs/stress/onedm )

  echo; echo "########## [5] ONE-DM style transfer (pjb writer) — visual sample ##########"
  SREF=$(ls -d "$PJB"/*/ 2>/dev/null | head -1)
  FIRST=$(echo "$SIGNS" | cut -d, -f1)
  python3 pipelines/generation/generate.py --engine onedm --signs "$FIRST" --n 4 \
      --style-dir "$SREF" --ckpt "$ONEDM_CKPT" --cfg-scale "$GEN_SCALE" \
      --dataset-name "$GEN_DATASET" --out misc/outputs/stress_style || true
  echo "style refs from: $SREF  -> misc/outputs/stress_style/onedm/"
else
  echo "!! skipping One-DM generation tests — no fine-tuned checkpoint found"
fi

echo; echo "=== STRESS TESTS DONE $(date) ==="
