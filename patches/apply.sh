#!/usr/bin/env bash
# Prepare the gfx1151_optim SGLang tree for this container.
#
# The image already ships SGLang + an AOT sgl_kernel wheel under
# /sgl-workspace/sglang. Do not patch or overwrite that tree. Clone
# hubertlu-tw/sglang@gfx1151_optim next to it, rebuild kernels in the clone,
# and leave PYTHONPATH wiring to sglang_server.sh.
#
# Re-run after every launch_docker.sh (the clone lives in the image FS unless
# you bind-mount it). First run clones + compiles; later runs are no-ops unless
# SGLANG_REBUILD_KERNELS=1 or the .so is missing.
#
# The in-repo *.patch files (quark-int4-w4a16, gfx1151 GEMM configs) are
# superseded by gfx1151_optim and are not applied here.
set -euo pipefail

SGL_GFX1151="${SGL_GFX1151:-/sgl-workspace/sglang-gfx1151}"
IMAGE_TREE="${SGLANG_IMAGE_TREE:-/sgl-workspace/sglang}"
REPO_URL="${SGLANG_GFX1151_REPO:-https://github.com/hubertlu-tw/sglang.git}"
BRANCH="${SGLANG_GFX1151_BRANCH:-gfx1151_optim}"
AOT="${SGL_GFX1151}/python/sglang/kernels/aot"

if [ -e "$IMAGE_TREE" ] && [ "$(readlink -f "$SGL_GFX1151" 2>/dev/null || true)" = "$(readlink -f "$IMAGE_TREE")" ]; then
    echo "refusing to clone over the image tree ${IMAGE_TREE}" >&2
    exit 1
fi

if [ ! -d "${SGL_GFX1151}/.git" ]; then
    echo "cloning ${REPO_URL} (${BRANCH}) -> ${SGL_GFX1151}"
    git clone -b "${BRANCH}" --single-branch "${REPO_URL}" "${SGL_GFX1151}"
else
    echo "updating ${SGL_GFX1151} (${BRANCH})"
    git -C "${SGL_GFX1151}" fetch origin "${BRANCH}"
    git -C "${SGL_GFX1151}" checkout "${BRANCH}"
    git -C "${SGL_GFX1151}" merge --ff-only "origin/${BRANCH}"
fi

so="$(find "${AOT}/python/sgl_kernel" -maxdepth 1 -name 'common_ops*.so' 2>/dev/null | head -n 1 || true)"
if [ "${SGLANG_REBUILD_KERNELS:-0}" = "1" ] || [ -z "${so}" ]; then
    echo "rebuilding AOT kernels (AMDGPU_TARGET=gfx1151) in ${AOT}"
    (
        cd "${AOT}"
        rm -rf build
        AMDGPU_TARGET=gfx1151 python3 setup_rocm.py build_ext --inplace
    )
else
    echo "kernels already built: ${so}  (SGLANG_REBUILD_KERNELS=1 to rebuild)"
fi

export PYTHONPATH="${SGL_GFX1151}/python/sglang/kernels/aot/python:${SGL_GFX1151}/python${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<PY
import os, sglang, sgl_kernel, torch
root = os.path.realpath("${SGL_GFX1151}")
print("sglang    ", sglang.__file__)
print("sgl_kernel", sgl_kernel.__file__)
print("wvSplitK  ", hasattr(torch.ops.sgl_kernel, "wvSplitK"))
print("wvSplitK_int4_g", hasattr(torch.ops.sgl_kernel, "wvSplitK_int4_g"))
assert os.path.realpath(sglang.__file__).startswith(root), sglang.__file__
assert os.path.realpath(sgl_kernel.__file__).startswith(root), sgl_kernel.__file__
assert hasattr(torch.ops.sgl_kernel, "wvSplitK")
assert hasattr(torch.ops.sgl_kernel, "wvSplitK_int4_g")
print("ok: clone is on PYTHONPATH and wvSplitK ops exist")
PY
