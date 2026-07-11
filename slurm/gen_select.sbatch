#!/bin/bash
#SBATCH --job-name=hiero-genselect
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/genselect_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/genselect_%j.err
# Model/scale selection after the CFG retrains: sweep (family x ckpt x cfg_scale),
# score recognizability with the frozen matcher, write slurm/best_gen.env for the
# stress test + synthetic-feedback jobs. Runs under afterany (the 128px job may
# TIMEOUT at the walltime with usable snapshots), so families are skipped gracefully.
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
MPY=$ROOT/pipelines/matching/.venv/bin/python
OUT=$ROOT/misc/outputs/genselect
cd "$ROOT/One-DM"

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="

# gen config for the 128 family (prep regenerates it with same-tree style paths)
./.venv/bin/python - <<'PY'
import os
p = "configs/hiero128.yml"
if os.path.isfile(p):
    s = open(p).read()
    s = s.replace("STYLE_PATH: ./data/hiero128/IAM64-new", "STYLE_PATH: ./data/hiero/IAM64-new")
    s = s.replace("LAPLACE_PATH: ./data/hiero128/IAM64_laplace", "LAPLACE_PATH: ./data/hiero/IAM64_laplace")
    open(p, "w").write(s)
PY

# 20 eval signs: well-populated classes, evenly spaced across the aspect range
SIGNS=$(./.venv/bin/python - <<'PY'
import json
m = json.load(open("data/hiero_wid_map.json"))["classes"]
ok = sorted((c for c in m if m[c]["n_train"] >= 60), key=lambda c: m[c]["aspect"])
step = max(1, len(ok) // 20)
print(",".join(ok[::step][:20]))
PY
)
echo "signs: $SIGNS"

last2() {  # family dir -> last two ckpts, newest last
  ls -1 "$1"/*/*/model/*-ckpt.pt 2>/dev/null \
    | awk -F'/model/' '{n=$2; sub(/-ckpt.pt/,"",n); print n, $0}' | sort -n | tail -2 | cut -d' ' -f2-
}

rm -rf "$OUT"; mkdir -p "$OUT"
RESULTS="$OUT/results.tsv"
echo -e "family\tckpt\tscale\ttop1\ttop5\tn" > "$RESULTS"

for FAMILY in hiero_cfg64:hiero hiero128:hiero128; do
  SAVED_DIR=Saved/${FAMILY%%:*}
  NAME=${FAMILY##*:}
  CKPTS=$(last2 "$SAVED_DIR")
  [ -n "$CKPTS" ] || { echo "!! no ckpts for $SAVED_DIR -- skipping family"; continue; }
  while read -r CKPT; do
    EP=$(basename "$CKPT" | sed 's/-ckpt.pt//')
    for S in 1.0 1.75 2.5 4.0; do
      TAG="${SAVED_DIR##*/}_e${EP}_s${S}"
      ./.venv/bin/python generate_hiero.py --cfg configs/$NAME.yml --one_dm "$CKPT" \
          --stable_dif_path model_zoo/sd-v1-5-vae --content_type ${NAME}_content \
          --letters data/${NAME}_letters.txt --wid-map data/${NAME}_wid_map.json \
          --signs "$SIGNS" --n 3 --cfg-scale $S --sampling_timesteps 25 --seed 42 \
          --out "$OUT/$TAG" || { echo "!! generation failed for $TAG"; continue; }
      LINE=$(cd "$ROOT/pipelines/matching" && "$MPY" evaluate.py --ckpt runs/a100/best.pt \
          --index runs/a100/index.npz --probe-dir "$OUT/$TAG" | tail -1)
      echo "$TAG :: $LINE"
      T1=$(echo "$LINE" | sed -n 's/.*top1 \([0-9.]*\).*/\1/p')
      T5=$(echo "$LINE" | sed -n 's/.*top5 \([0-9.]*\).*/\1/p')
      N=$(echo "$LINE" | sed -n 's/.*n=\([0-9]*\).*/\1/p')
      echo -e "${SAVED_DIR##*/}\t$CKPT\t$S\t${T1:-0}\t${T5:-0}\t${N:-0}" >> "$RESULTS"
    done
  done <<< "$CKPTS"
done

echo "########## results ##########"
column -t "$RESULTS"

./.venv/bin/python - <<PY
import csv
rows = list(csv.DictReader(open("$RESULTS"), delimiter="\t"))
rows = [r for r in rows if int(r["n"]) > 0]
assert rows, "no successful generation runs"
best = max(rows, key=lambda r: (float(r["top1"]), float(r["top5"])))
name = "hiero128" if best["family"] == "hiero128" else "hiero"
import os
best["ckpt"] = os.path.abspath(best["ckpt"])  # consumers run from other cwds
with open("$ROOT/slurm/best_gen.env", "w") as f:
    f.write(f"BEST_FAMILY={best['family']}\nBEST_CKPT={best['ckpt']}\n"
            f"BEST_SCALE={best['scale']}\nBEST_DATASET={name}\nBEST_TOP1={best['top1']}\n")
print("best:", best)
PY
cat "$ROOT/slurm/best_gen.env"
echo "=== GENSELECT DONE $(date) ==="
