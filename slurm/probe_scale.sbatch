#!/bin/bash
#SBATCH --job-name=hiero-probescale
#SBATCH --account=crew
#SBATCH --constraint=a100|v100s|a40|rtx8000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/probescale_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/probescale_%j.err
# gen_select's cfg_scale sweep peaked at its 4.0 cap while still rising; probe
# higher scales on the winning ckpt (hiero128 e99). Same signs/seed as gen_select
# so rows are directly comparable. If a higher scale wins by >2pt, update
# slurm/best_gen.env (stress/synth jobs source it at their own runtime).
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
MPY=$ROOT/pipelines/matching/.venv/bin/python
cd "$ROOT/One-DM"
source "$ROOT/slurm/best_gen.env"
echo "=== $(date) host=$(hostname) probing above scale=$BEST_SCALE on $BEST_CKPT ==="

SIGNS=$(./.venv/bin/python - <<'PY'
import json
m = json.load(open("data/hiero_wid_map.json"))["classes"]
ok = sorted((c for c in m if m[c]["n_train"] >= 60), key=lambda c: m[c]["aspect"])
step = max(1, len(ok) // 20)
print(",".join(ok[::step][:20]))
PY
)
BEST_T1=$BEST_TOP1; BEST_S=$BEST_SCALE
for S in 5.0 6.0 8.0; do
  ./.venv/bin/python generate_hiero.py --cfg configs/hiero128.yml --one_dm "$BEST_CKPT" \
      --stable_dif_path model_zoo/sd-v1-5-vae --content_type hiero128_content \
      --letters data/hiero128_letters.txt --wid-map data/hiero128_wid_map.json \
      --signs "$SIGNS" --n 3 --cfg-scale $S --sampling_timesteps 25 --seed 42 \
      --out "$ROOT/misc/outputs/probescale/s$S"
  LINE=$(cd "$ROOT/pipelines/matching" && "$MPY" evaluate.py --ckpt runs/a100/best.pt \
      --index runs/a100/index.npz --probe-dir "$ROOT/misc/outputs/probescale/s$S" | tail -1)
  echo "scale $S :: $LINE"
  T1=$(echo "$LINE" | sed -n 's/.*top1 \([0-9.]*\).*/\1/p')
  if awk "BEGIN{exit !($T1 > $BEST_T1 + 0.02)}"; then BEST_T1=$T1; BEST_S=$S; fi
done

if [ "$BEST_S" != "$BEST_SCALE" ]; then
  echo "UPDATING best_gen.env: scale $BEST_SCALE -> $BEST_S (top1 $BEST_TOP1 -> $BEST_T1)"
  sed -i "s/^BEST_SCALE=.*/BEST_SCALE=$BEST_S/; s/^BEST_TOP1=.*/BEST_TOP1=$BEST_T1/" "$ROOT/slurm/best_gen.env"
  cat "$ROOT/slurm/best_gen.env"
else
  echo "scale 4.0 stands (no higher scale beat it by >2pt)"
fi
echo "=== PROBESCALE DONE $(date) ==="
