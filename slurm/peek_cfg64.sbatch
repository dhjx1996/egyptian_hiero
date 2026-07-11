#!/bin/bash
#SBATCH --job-name=hiero-peek64
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/peek64_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/peek64_%j.err
# Early quality read on the finished cfg64 model (~6h before gen_select runs):
# cfg_scale sweep on the final ckpt, matcher-scored. Baselines to beat: the
# non-CFG 64px model scored top1 0.400 on these same 10 signs (valcfg job).
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
MPY=$ROOT/pipelines/matching/.venv/bin/python
cd "$ROOT/One-DM"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="
CKPT=$(ls -1 Saved/hiero_cfg64/*/*/model/*-ckpt.pt | awk -F'/model/' '{n=$2; sub(/-ckpt.pt/,"",n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "ckpt: $CKPT"
SIGNS=A1,A17,D21,D36,G1,M17,N35,O1,Q3,S29
for S in 1.0 1.75 2.5 4.0; do
  ./.venv/bin/python generate_hiero.py --cfg configs/hiero.yml --one_dm "$CKPT" \
      --stable_dif_path model_zoo/sd-v1-5-vae --content_type hiero_content \
      --letters data/hiero_letters.txt --wid-map data/hiero_wid_map.json \
      --signs $SIGNS --n 3 --cfg-scale $S --sampling_timesteps 25 --seed 42 \
      --out "$ROOT/misc/outputs/peek64/s$S"
  echo "--- recognizability at cfg_scale=$S (broken-model baseline: 0.400) ---"
  ( cd "$ROOT/pipelines/matching" && "$MPY" evaluate.py --ckpt runs/a100/best.pt \
      --index runs/a100/index.npz --probe-dir "$ROOT/misc/outputs/peek64/s$S" )
done
echo "=== PEEK64 DONE $(date) ==="
