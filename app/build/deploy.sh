#!/bin/bash
# One-command deploy of app/ to Cloudflare Pages (seshat-690.pages.dev).
#   bash app/build/deploy.sh
# Must run from a machine with the generated assets (model/index/vendor are
# git-ignored, which is why CI can't deploy). Needs a Cloudflare API token
# with "Cloudflare Pages: Edit" in .cloudflare_token at repo root (git-ignored).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

[ -s "$ROOT/.cloudflare_token" ] || { echo "ERROR: $ROOT/.cloudflare_token missing — create a Cloudflare API token with 'Cloudflare Pages: Edit' and save it there." >&2; exit 1; }
VER=$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet HEAD -- app || echo "-dirty")
bash "$ROOT/app/build/stage.sh" "$STAGE/app" "$VER"

CLOUDFLARE_API_TOKEN=$(cat "$ROOT/.cloudflare_token") \
CLOUDFLARE_ACCOUNT_ID=7b766e08ce99089e9558176f243c7cef \
npx --yes wrangler@latest pages deploy "$STAGE/app" \
    --project-name=seshat --branch=main --commit-dirty=true
