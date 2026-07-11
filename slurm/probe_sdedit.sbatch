#!/bin/bash
#SBATCH --job-name=hiero-sdedit
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/sdedit_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/sdedit_%j.err
# SDEdit probe: initialize sampling from the noised CANONICAL glyph so content
# structure is inherited by construction and diffusion only re-textures.
# Sweep init strength; same signs/seed as gen_select (pure-noise reference:
# top1 0.567 at cfg_scale 4.0). Strength 1.0-equivalent = pure noise; lower
# strength = more canonical structure survives.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
MPY=$ROOT/pipelines/matching/.venv/bin/python
CANON=$ROOT/hiero_data/archaeohack-starterpack/data/utf-pngs
cd "$ROOT/One-DM"
source "$ROOT/slurm/best_gen.env"
echo "=== $(date) host=$(hostname) ckpt=$BEST_CKPT scale=$BEST_SCALE ==="

SIGNS=$(./.venv/bin/python - <<'PY'
import json
m = json.load(open("data/hiero_wid_map.json"))["classes"]
ok = sorted((c for c in m if m[c]["n_train"] >= 60), key=lambda c: m[c]["aspect"])
step = max(1, len(ok) // 20)
print(",".join(ok[::step][:20]))
PY
)
for STR in 0.4 0.6 0.8; do
  ./.venv/bin/python generate_hiero.py --cfg configs/hiero128.yml --one_dm "$BEST_CKPT" \
      --stable_dif_path model_zoo/sd-v1-5-vae --content_type hiero128_content \
      --letters data/hiero128_letters.txt --wid-map data/hiero128_wid_map.json \
      --signs "$SIGNS" --n 3 --cfg-scale "$BEST_SCALE" --sampling_timesteps 50 --seed 42 \
      --init-strength $STR --canonical-dir "$CANON" \
      --out "$ROOT/misc/outputs/sdedit/str$STR"
  echo "--- recognizability at init-strength=$STR (pure-noise ref: $BEST_TOP1) ---"
  ( cd "$ROOT/pipelines/matching" && "$MPY" evaluate.py --ckpt runs/a100/best.pt \
      --index runs/a100/index.npz --probe-dir "$ROOT/misc/outputs/sdedit/str$STR" )
done
echo "=== SDEDIT PROBE DONE $(date) ==="
