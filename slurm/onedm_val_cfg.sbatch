#!/bin/bash
#SBATCH --job-name=hiero-valcfg
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/valcfg_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/valcfg_%j.err
# De-risk the CFG + 128px changes before the 12h fine-tunes:
#  [1] 2-epoch smoke train @128px targets / 64px styles, COND_DROP_PROB=0.1
#  [2] 2-epoch smoke train @64px, COND_DROP_PROB=0.1
#  [3] generate from the smoke-128 ckpt with --cfg-scale (plumbing only)
#  [4] cfg-scale sweep on the EXISTING full 64px ckpt, matcher-scored
#      (bonus signal: does zero-context guidance help a non-CFG-trained model?)
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
MPY=$ROOT/pipelines/matching/.venv/bin/python
cd "$ROOT/One-DM"
PORT=$((21000 + SLURM_JOB_ID % 10000))

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset} ==="

echo "########## [1] smoke train 128px targets + 64px styles + cond-drop ##########"
./.venv/bin/torchrun --standalone --nproc_per_node=1 --master_port=$PORT train.py \
    --cfg configs/hs128_train.yml --one_dm model_zoo/One-DM-ckpt.pt \
    --stable_dif_path model_zoo/sd-v1-5-vae --content_type hs128_content \
    --letters data/hs128_letters.txt --train_txt data/hs128_train.txt --test_txt data/hs128_test.txt

echo "########## [2] smoke train 64px + cond-drop ##########"
./.venv/bin/torchrun --standalone --nproc_per_node=1 --master_port=$((PORT+1)) train.py \
    --cfg configs/hs64_train.yml --one_dm model_zoo/One-DM-ckpt.pt \
    --stable_dif_path model_zoo/sd-v1-5-vae --content_type hs64_content \
    --letters data/hs64_letters.txt --train_txt data/hs64_train.txt --test_txt data/hs64_test.txt

echo "########## [3] generate @128 with cfg-scale from smoke ckpt (plumbing) ##########"
SMOKE_CKPT=$(ls -1 Saved/hs128/*/*/model/*-ckpt.pt | awk -F'/model/' '{n=$2; sub(/-ckpt.pt/,"",n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "smoke ckpt: $SMOKE_CKPT"
./.venv/bin/python generate_hiero.py --cfg configs/hs128.yml --one_dm "$SMOKE_CKPT" \
    --stable_dif_path model_zoo/sd-v1-5-vae --content_type hs128_content \
    --letters data/hs128_letters.txt --wid-map data/hs128_wid_map.json \
    --signs A1,A10 --n 2 --cfg-scale 2.5 --sampling_timesteps 25 \
    --out "$ROOT/misc/outputs/valcfg/smoke128"
./.venv/bin/python - <<'PY'
import glob
import numpy as np
from PIL import Image
for f in sorted(glob.glob("../misc/outputs/valcfg/smoke128/*/*.png")):
    a = np.array(Image.open(f).convert("L"))
    print(f"{f}: {a.shape[0]}x{a.shape[1]}  ink={(a<200).mean()*100:.1f}%")
    assert a.shape[0] == 128, "expected 128px output height"
print("[3] OK: 128px outputs produced")
PY

echo "########## [4] cfg sweep on existing full 64px ckpt (matcher-scored) ##########"
FULL_CKPT=$(ls -1 Saved/hiero/*/*/model/*-ckpt.pt | awk -F'/model/' '{n=$2; sub(/-ckpt.pt/,"",n); print n, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
echo "full 64px ckpt: $FULL_CKPT"
SIGNS=A1,A17,D21,D36,G1,M17,N35,O1,Q3,S29
for S in 1.0 1.5 2.5 4.0; do
  ./.venv/bin/python generate_hiero.py --cfg configs/hiero.yml --one_dm "$FULL_CKPT" \
      --stable_dif_path model_zoo/sd-v1-5-vae --content_type hiero_content \
      --letters data/hiero_letters.txt --wid-map data/hiero_wid_map.json \
      --signs $SIGNS --n 3 --cfg-scale $S --sampling_timesteps 25 --seed 42 \
      --out "$ROOT/misc/outputs/valcfg/sweep64/s$S"
  echo "--- recognizability at cfg_scale=$S ---"
  ( cd "$ROOT/pipelines/matching" && "$MPY" evaluate.py --ckpt runs/a100/best.pt \
      --index runs/a100/index.npz --probe-dir "$ROOT/misc/outputs/valcfg/sweep64/s$S" )
done

echo "=== VALCFG DONE $(date) ==="
