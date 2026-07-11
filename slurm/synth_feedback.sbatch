#!/bin/bash
#SBATCH --job-name=hiero-synthfeed
#SBATCH --account=crew
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/synthfeed_%j.out
#SBATCH --error=/burg-archive/home/dh3065/egyptian_hiero/slurm/logs/synthfeed_%j.err
# Secondary pipeline: One-DM outputs -> matcher training data, with collapse guards.
#  [1] generate 12 variants/class for ALL in-vocab classes with the best ckpt/scale
#      (slurm/best_gen.env, written by gen_select)
#  [2] quality-gate with the FROZEN baseline matcher (synth_filter.py); ABORT if
#      the accept rate is too low to be useful
#  [3] retrain the matcher with capped synthetic augmentation (never in val)
#  [4] rebuild index + evaluate BOTH matchers on real held-out + unseen-writer pjb;
#      the synth model is a candidate -- acceptance requires no regression on real data
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT=/burg-archive/home/dh3065/egyptian_hiero
MPY=$ROOT/pipelines/matching/.venv/bin/python
SYNTH=$ROOT/misc/outputs/synth
PJB=$ROOT/hiero_data/archaeohack-starterpack/data/me-sign-examples-pjb

echo "=== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="
[ -f "$ROOT/slurm/best_gen.env" ] || { echo "missing slurm/best_gen.env (run gen_select first)"; exit 1; }
source "$ROOT/slurm/best_gen.env"
echo "using: $BEST_CKPT  scale=$BEST_SCALE  dataset=$BEST_DATASET  (select top1=$BEST_TOP1)"

echo "########## [1] generate synthetic corpus (all classes x 12) ##########"
cd "$ROOT/One-DM"
if [ -d "$SYNTH/onedm" ] && [ "$(ls "$SYNTH/onedm" | wc -l)" -ge 500 ]; then
  echo "reusing existing synthetic corpus ($(ls "$SYNTH/onedm" | wc -l) classes)"
else
  rm -rf "$SYNTH"; mkdir -p "$SYNTH"
  ./.venv/bin/python generate_hiero.py --cfg configs/$BEST_DATASET.yml --one_dm "$BEST_CKPT" \
      --stable_dif_path model_zoo/sd-v1-5-vae --content_type ${BEST_DATASET}_content \
      --letters data/${BEST_DATASET}_letters.txt --wid-map data/${BEST_DATASET}_wid_map.json \
      --signs all --n 12 --batch 12 --cfg-scale "$BEST_SCALE" --sampling_timesteps 25 \
      --seed 20260710 --out "$SYNTH/onedm"
fi
rm -rf "$SYNTH/accepted"

echo "########## [2] failure gate (frozen baseline matcher; keeps messy on-class samples) ##########"
cd "$ROOT/pipelines/matching"
"$MPY" synth_filter.py --gen-root "$SYNTH/onedm" --ckpt runs/a100/best.pt \
    --index runs/a100/index.npz --out "$SYNTH/accepted" \
    --ink-min 0.005 --ink-max 0.70 --reject-other-score 0.45 \
    --max-canon-sim 0.92 --max-dup-sim 0.97
"$MPY" - <<'PY'
import json, sys
s = json.load(open("../../misc/outputs/synth/accepted/filter_stats.json"))
n_cls = sum(1 for c in s["classes"].values() if c["kept"])
print(f"[gate] accept_rate={s['accept_rate']:.1%} classes_with_accepts={n_cls}")
if s["accept_rate"] < 0.15 or n_cls < 50:
    sys.exit("ABORT: synthetic quality too low to be useful -- not retraining the matcher")
PY

echo "########## [3] retrain matcher with capped synthetic data ##########"
"$MPY" train_encoder.py --name a100_synth --arch resnet34 --size 160 --epochs 100 \
    --batch-size 512 --workers 16 --p-handwrite 0.5 --canon-repeat 12 \
    --synthetic "$SYNTH/accepted" --synth-cap-frac 0.25 --synth-per-class 20

echo "########## [4] index + side-by-side evaluation (real data only) ##########"
"$MPY" build_index.py --ckpt runs/a100_synth/best.pt \
    --handwriting "$ROOT/hiero_data/Hand-drawn Hieroglyph Dataset" \
    --exclude-val runs/a100_synth/val_split.json --out runs/a100_synth/index.npz

echo "--- baseline (real-only) ---"
"$MPY" evaluate.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json --probe-dir "$PJB" --json-out runs/a100/eval_final.json
echo "--- candidate (real + gated synthetic) ---"
"$MPY" evaluate.py --ckpt runs/a100_synth/best.pt --index runs/a100_synth/index.npz \
    --val-split runs/a100_synth/val_split.json --probe-dir "$PJB" --json-out runs/a100_synth/eval.json

echo "--- corruption robustness (outlier-handling check: synthetic data must not"
echo "    push the model toward the mean at the expense of degraded inputs) ---"
"$MPY" stress_test.py --ckpt runs/a100/best.pt --index runs/a100/index.npz \
    --val-split runs/a100/val_split.json --device cuda --limit 800 --json-out runs/a100/stress_base.json
"$MPY" stress_test.py --ckpt runs/a100_synth/best.pt --index runs/a100_synth/index.npz \
    --val-split runs/a100_synth/val_split.json --device cuda --limit 800 --json-out runs/a100_synth/stress.json

"$MPY" - <<'PY'
import json
a = json.load(open("runs/a100/eval_final.json"))
b = json.load(open("runs/a100_synth/eval.json"))
sa = json.load(open("runs/a100/stress_base.json"))["corruptions"]
sb = json.load(open("runs/a100_synth/stress.json"))["corruptions"]
print("\n========== ACCEPTANCE REPORT ==========")
print(f"{'metric':<26}{'baseline':>10}{'synth':>10}{'delta':>9}")
verdict, reasons = True, []
for block in sorted(set(a) & set(b)):
    if not isinstance(a[block], dict):
        continue
    for k in ("top1", "top5"):
        if k in a[block]:
            d = b[block][k] - a[block][k]
            print(f"{block+'.'+k:<26}{a[block][k]:>10.3f}{b[block][k]:>10.3f}{d:>+9.3f}")
            if k == "top1" and d < -0.005:
                verdict, _ = False, reasons.append(f"{block}.top1 {d:+.3f}")
for c in sorted(set(sa) & set(sb)):
    d = sb[c]["top1"] - sa[c]["top1"]
    print(f"{'corrupt.'+c+'.top1':<26}{sa[c]['top1']:>10.3f}{sb[c]['top1']:>10.3f}{d:>+9.3f}")
    # corruption metrics are noisier (sampled queries): allow 2pt slack per cell
    if d < -0.02:
        verdict, _ = False, reasons.append(f"corrupt.{c} {d:+.3f}")
print("VERDICT:", "ACCEPT (no clean or corruption regression)" if verdict
      else f"REJECT ({', '.join(reasons)}) -- keep runs/a100 as production")
PY
echo "=== SYNTH FEEDBACK DONE $(date) ==="
