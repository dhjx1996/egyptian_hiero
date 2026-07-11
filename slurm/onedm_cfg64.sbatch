#!/bin/bash
#SBATCH --job-name=hiero-cfg64
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/cfg64_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/cfg64_%j.err
# Path A of the One-DM fix: continue the finished 64px hiero fine-tune with
# conditioning dropout (TRAIN.COND_DROP_PROB=0.1) so classifier-free guidance
# works at sampling time. Cheap adaptation: model already fits hiero; it only
# needs to learn the unconditional mode. 1 GPU, 60 epochs, half LR.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
cd "$ROOT/One-DM"
PORT=$((22000 + SLURM_JOB_ID % 10000))

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="

CKPT=$(ls -1 Saved/hiero/*/*/model/*-ckpt.pt 2>/dev/null \
  | awk -F'/model/' '{n=$2; sub(/-ckpt.pt/,"",n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$CKPT" ] || { echo "no Saved/hiero checkpoint found"; exit 1; }
echo "continuing from: $CKPT"

./.venv/bin/torchrun --standalone --nproc_per_node=1 --master_port=$PORT train.py \
    --cfg configs/hiero_cfg64_train.yml --one_dm "$CKPT" \
    --stable_dif_path model_zoo/sd-v1-5-vae --content_type hiero_content \
    --letters data/hiero_letters.txt --train_txt data/hiero_train.txt --test_txt data/hiero_test.txt

echo "=== CFG64 TRAIN DONE $(date) ==="
