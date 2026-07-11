#!/bin/bash
#SBATCH --job-name=hiero-128
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/onedm128_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/onedm128_%j.err
# Path B of the One-DM fix: fine-tune from the IAM-pretrained checkpoint at
# 128px target resolution (16x16+ latents instead of 8x8 -- enough for glyph
# detail) WITH conditioning dropout for classifier-free guidance. Style/laplace
# refs stay 64px (the style encoder hardcodes 64px feature shapes; targets and
# styles are independent inputs). 2xA100 DDP, walltime-capped: snapshots every
# 5 epochs mean a usable model even if 100 epochs don't fit in 12h.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
cd "$ROOT/One-DM"
PORT=$((23000 + SLURM_JOB_ID % 10000))

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="

for f in data/hiero128_train.txt data/hiero128_content.pickle data/hiero128/IAM64-new/train \
         data/hiero/IAM64-new/train data/hiero/IAM64_laplace/train model_zoo/One-DM-ckpt.pt; do
  [ -e "$f" ] || { echo "missing required input: $f"; exit 1; }
done

# prep_dataset.py regenerates configs/hiero128_train.yml with same-tree paths;
# re-point styles at the 64px trees + enable cond dropout (idempotent)
./.venv/bin/python - <<'PY'
p = "configs/hiero128_train.yml"
s = open(p).read()
s = s.replace("STYLE_PATH: ./data/hiero128/IAM64-new", "STYLE_PATH: ./data/hiero/IAM64-new")
s = s.replace("LAPLACE_PATH: ./data/hiero128/IAM64_laplace", "LAPLACE_PATH: ./data/hiero/IAM64_laplace")
if "COND_DROP_PROB" not in s:
    s = s.replace("  SEED: 1001", "  SEED: 1001\n  COND_DROP_PROB: 0.1")
open(p, "w").write(s)
print(s)
PY

./.venv/bin/torchrun --standalone --nproc_per_node=2 --master_port=$PORT train.py \
    --cfg configs/hiero128_train.yml --one_dm model_zoo/One-DM-ckpt.pt \
    --stable_dif_path model_zoo/sd-v1-5-vae --content_type hiero128_content \
    --letters data/hiero128_letters.txt --train_txt data/hiero128_train.txt --test_txt data/hiero128_test.txt

echo "=== ONE-DM 128 TRAIN DONE $(date) ==="
