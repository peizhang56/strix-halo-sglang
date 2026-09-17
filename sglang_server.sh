#!/usr/bin/env bash
set -euo pipefail

# Default: Quark AWQ W4A16 + DFlash2, the gfx1151_optim step-7 stack
# (row-streaming GDN decode + grouped GDN fwd_o + wvSplitK). Measured
# GSM8K 10q --parallel 10: 15.991 output tok/s, accuracy 1.000.
#
#   ./sglang_server.sh                     # int4 + DFlash2  (default, step 7)
#   ./sglang_server.sh Qwen/Qwen3.8-27B    # same flags, bf16 weights
#   SGLANG_SPEC=eagle ./sglang_server.sh   # MTP instead of DFlash2
#   SGLANG_SPEC=none ./sglang_server.sh    # no draft, no graphs
#
# Requires ./patches/apply.sh first: that clones gfx1151_optim to
# /sgl-workspace/sglang-gfx1151 and rebuilds AOT kernels there. The image
# tree /sgl-workspace/sglang is left alone.

SGL_GFX1151="${SGL_GFX1151:-/sgl-workspace/sglang-gfx1151}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SGL_GFX1151}/python/sglang/kernels/aot/python:${SGL_GFX1151}/python${PYTHONPATH:+:$PYTHONPATH}"

# Accept a real directory, otherwise keep the Hugging Face repo id so
# huggingface_hub can resolve snapshots under the mounted HF cache.
_resolve_local_model() {
    local id="$1"
    if [ -f "${id}/config.json" ]; then
        printf '%s\n' "$id"
        return
    fi
    printf '%s\n' "$id"
}

python3 - <<PY
import os, sglang, sgl_kernel, torch, sys
root = os.path.realpath("${SGL_GFX1151}")
ok = (
    os.path.realpath(sglang.__file__ or "").startswith(root)
    and os.path.realpath(sgl_kernel.__file__ or "").startswith(root)
    and hasattr(torch.ops.sgl_kernel, "wvSplitK")
    and hasattr(torch.ops.sgl_kernel, "wvSplitK_int4_g")
)
if not ok:
    print(
        "sglang_server.sh: not using ${SGL_GFX1151}.\n"
        f"  sglang     {sglang.__file__}\n"
        f"  sgl_kernel {sgl_kernel.__file__}\n"
        f"  wvSplitK   {hasattr(torch.ops.sgl_kernel, 'wvSplitK')}\n"
        f"  wvSplitK_int4_g {hasattr(torch.ops.sgl_kernel, 'wvSplitK_int4_g')}\n"
        "Run ./patches/apply.sh first (clone gfx1151_optim beside the image tree,\n"
        "rebuild AOT kernels in that clone, then relaunch).",
        file=sys.stderr,
    )
    sys.exit(1)
print("using sglang   ", sglang.__file__)
print("using sgl_kernel", sgl_kernel.__file__)
PY

# Weights are already in /root/.cache/huggingface; without this SGLang makes ~11
# blocking metadata round-trips to huggingface.co at startup and can stall on retries.
export HF_HUB_OFFLINE=1

# The first forward pass JIT-compiles Triton/inductor kernels. The default cache
# dirs live in the container's /tmp, so every launch recompiled from scratch;
# /workspace is one of the two persistent mounts, so keep them there instead.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/workspace/.cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/workspace/.cache/triton}"

# SGLang warms up with one generate request and kills the whole server if it does
# not answer within 600 s. A cold-compile launch here can exceed that, so raise it.
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-1800}"

# config.json declares mamba_ssm_dtype=float32. Forcing bf16 is part of the
# combination under which bounded graph capture runs clean here.
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"

MODEL="$(_resolve_local_model "${1:-amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}")"
echo "model-path: $MODEL"

# 8192 was enough for benchmarking but CANNOT run Claude Code: its system prompt plus tool
# schemas measured 19,334 tokens before any conversation starts. Costs nothing to raise --
# the KV pool is sized from --mem-fraction-static, not from this, and measured the same
# 660,706 tokens at 64k with 28.3 GB still free. Model max is 262144 if you need more.
# Set SGLANG_CONTEXT_LEN=8192 to reproduce the old benchmark baseline.
CONTEXT_LEN="${SGLANG_CONTEXT_LEN:-65536}"

# Step 7 used 0.93. That setting has hung this WSL2 host before (GTT is carved
# out of the same physical RAM), so keep SGLANG_MEM_FRACTION=0.85 as an escape
# hatch if the box wedges. 0.93 is the default only because it is the measured
# fastest stack.
MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.93}"

# Qwen3.5's stock chat template calls raise_exception() on any reasoning effort outside
# xhigh/medium/low. Claude Code sends one of six levels in output_config and defaults to
# "high", so the template throws and SGLang answers HTTP 500 before any inference runs --
# which the client reports as a transient server fault. The patched copy collapses all six
# levels onto the three the model knows instead of raising. Use SGLANG_CHAT_TEMPLATE= (set
# but empty) to fall back to the stock template, or point it at another file.
# launch_docker.sh mounts this repo at /workspace. When the scripts live
# elsewhere (e.g. /sgl-workspace/strix-halo-sglang), fall back to SCRIPT_DIR.
if [ -n "${SGLANG_CHAT_TEMPLATE+x}" ]; then
    CHAT_TEMPLATE="${SGLANG_CHAT_TEMPLATE}"
elif [ -f /workspace/chat_template_qwen3_agentic.jinja ]; then
    CHAT_TEMPLATE="/workspace/chat_template_qwen3_agentic.jinja"
else
    CHAT_TEMPLATE="${SCRIPT_DIR}/chat_template_qwen3_agentic.jinja"
fi
template_flags=()
if [ -n "$CHAT_TEMPLATE" ]; then
    if [ -f "$CHAT_TEMPLATE" ]; then
        template_flags=(--chat-template "$CHAT_TEMPLATE")
    else
        echo "warning: chat template $CHAT_TEMPLATE not found; using the model's own" >&2
    fi
fi

# Which speculative decoder to run:
#   dflash (default) -- DFlash2, gfx1151_optim step 7. Fastest measured stack.
#   eagle            -- the model's own MTP head.
#   none             -- no draft model, no graphs (SGLANG_NO_SPEC=1 is a synonym).
SPEC="${SGLANG_SPEC:-dflash}"
[ "${SGLANG_NO_SPEC:-0}" = "1" ] && SPEC=none

if [ "$SPEC" = "none" ]; then
    perf_flags=(
        --max-running-requests 4
        --disable-cuda-graph
    )
else
    MAX_RUNNING="${SGLANG_MAX_RUNNING:-4}"
    # Step 7 used 4096. The older 1024 optimum was for the pre-wvSplitK
    # dequant-to-bf16 GEMM path; do not drop back to 1024 unless you are
    # reproducing that baseline.
    CHUNKED_PREFILL="${SGLANG_CHUNKED_PREFILL:-4096}"

    perf_flags=(
        --max-running-requests "$MAX_RUNNING"
        --chunked-prefill-size "$CHUNKED_PREFILL"
    )

    case "$SPEC" in
    eagle)
        perf_flags+=(
            --speculative-algorithm EAGLE
            --speculative-num-steps 3
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens 4
            --speculative-draft-attention-backend triton
        )
        ;;
    dflash)
        DRAFT_MODEL="$(_resolve_local_model "${SGLANG_DFLASH_DRAFT:-incoai/Qwen3.8-27B-DFlash2}")"
        echo "draft-model: $DRAFT_MODEL"
        perf_flags+=(
            --speculative-algorithm DFLASH
            --speculative-draft-model-path "$DRAFT_MODEL"
            --speculative-draft-attention-backend triton
            --speculative-num-draft-tokens 8
        )
        ;;
    *)
        echo "unknown SGLANG_SPEC='$SPEC' (want: eagle, dflash, none)" >&2
        exit 2
        ;;
    esac
fi

exec python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LEN" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    ${template_flags[@]+"${template_flags[@]}"} \
    "${perf_flags[@]}"
