#!/bin/bash
# Stage app/ for static hosting: drop build tooling + demo video, split the
# model under Cloudflare Pages' 25 MiB per-file cap (app.js::fetchModel
# reassembles), stamp the service-worker VERSION so clients pick up deploys.
#   bash app/build/stage.sh <outdir> [version]
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: stage.sh <outdir> [version]}
VER=${2:-$(date +%Y%m%d%H%M%S)}

rm -rf "$OUT"
mkdir -p "$OUT"
rsync -a --exclude build --exclude app_demo.mp4 --exclude README.md "$SRC/" "$OUT/"

python3 - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
cfg_p = os.path.join(out, "data", "config.json")
cfg = json.load(open(cfg_p))
mp = os.path.join(out, "data", cfg["model"])
data = open(mp, "rb").read()
CH = 20 * 1024 * 1024
parts = range(0, len(data), CH)
for i, off in enumerate(parts):
    with open(f"{mp}.part{i}", "wb") as f:
        f.write(data[off:off + CH])
os.remove(mp)
cfg["model_parts"] = len(parts)
with open(cfg_p, "w") as f:
    json.dump(cfg, f, indent=1)
print(f"[stage] split {cfg['model']} ({len(data)} bytes) into {len(parts)} parts")
EOF

sed -i "s/^const VERSION = .*/const VERSION = \"$VER\";/" "$OUT/sw.js"
echo "[stage] staged -> $OUT (sw version $VER)"
