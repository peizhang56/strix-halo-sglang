#!/usr/bin/env bash
# Apply the in-repo SGLang patches to the container's SGLang checkout.
# The checkout lives in the image (/sgl-workspace/sglang), so these are lost
# whenever the container is recreated -- re-run this after `launch_docker.sh`.
set -euo pipefail

SGLANG_SRC="${SGLANG_SRC:-/sgl-workspace/sglang}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for patch in "$PATCH_DIR"/*.patch; do
    if git -C "$SGLANG_SRC" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied: $(basename "$patch")"
        continue
    fi
    git -C "$SGLANG_SRC" apply "$patch"
    echo "applied: $(basename "$patch")"
done
