#!/bin/bash
#SBATCH --job-name=hiero-onedm
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/onedm_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/onedm_%j.err
# Pipeline 1 (One-DM) — full fine-tune on ALL hiero_data from the pretrained
# IAM-Latin checkpoint. 2x A100 DDP via torchrun (per-GPU batch = IMS_PER_BATCH=96,
# effective 192). Config DATA_LOADER.NUM_THREADS=12 => 2 ranks x 12 = 24 workers <= 28.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

cd /burg-archive/home/dh3065/egyptian_hiero/One-DM

# prep (dependency) must have produced these:
for f in configs/hiero_train.yml data/hiero_train.txt data/hiero_test.txt \
         data/hiero_letters.txt data/hiero_content.pickle model_zoo/One-DM-ckpt.pt; do
  [ -e "$f" ] || { echo "MISSING required input: $f" >&2; exit 2; }
done

PORT=$(( 20000 + (${SLURM_JOB_ID:-0} % 10000) ))

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="
echo "cpus-per-task=${SLURM_CPUS_PER_TASK:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} master_port=$PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

./.venv/bin/torchrun --standalone --nproc_per_node=2 --master_port=$PORT \
  train.py --cfg configs/hiero_train.yml --one_dm model_zoo/One-DM-ckpt.pt \
  --stable_dif_path model_zoo/sd-v1-5-vae \
  --content_type hiero_content --letters data/hiero_letters.txt \
  --train_txt data/hiero_train.txt --test_txt data/hiero_test.txt

echo "=== One-DM fine-tune DONE $(date) ==="
ls -1 Saved/hiero/*/model/*.pt 2>/dev/null | tail -5 || true
