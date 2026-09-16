#!/usr/bin/env bash
set -euo pipefail

# Default: the int4 (Quark AWQ W4A16) checkpoint with CUDA graphs and EAGLE/MTP
# speculative decoding, a 1024-token prefill chunk and 4 running requests.
# Measured batch 1, 8192 in / 512 out: prefill 163 tok/s, total 156 s (was 44 tok/s
# and 294 s before the chunk change); GSM8K 10 q 1.000 in 111 s at 13.5 tok/s.
# Run-to-run spread is ±14% on identical settings, so quote ranges.
# See kb/qwen38-int4-perf.md and kb/qwen38-int4-benchmark.md.
#
#   ./sglang_server.sh                     # int4 + graphs + MTP  (default)
#   ./sglang_server.sh Qwen/Qwen3.8-27B    # same flags, bf16 weights (see caveat)
#   SGLANG_NO_SPEC=1 ./sglang_server.sh    # fall back to the old no-graph config
#
# The int4 model needs the `quark-int4-w4a16` branch checked out in
# /sgl-workspace/sglang -- redo that after every launch_docker.sh.

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

MODEL="${1:-amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16}"

# 8192 was enough for benchmarking but CANNOT run Claude Code: its system prompt plus tool
# schemas measured 19,334 tokens before any conversation starts. Costs nothing to raise --
# the KV pool is sized from --mem-fraction-static, not from this, and measured the same
# 660,706 tokens at 64k with 28.3 GB still free. Model max is 262144 if you need more.
# Set SGLANG_CONTEXT_LEN=8192 to reproduce the old benchmark baseline.
CONTEXT_LEN="${SGLANG_CONTEXT_LEN:-65536}"

# Fraction of GPU memory reserved for weights + KV pool.
#
# DO NOT RAISE THIS TO 0.93. It was tried and it HUNG THE WHOLE MACHINE -- not an OOM
# kill, not a failed launch, a hard hang of the host requiring a restart. The ~103 GB of
# GTT is carved out of the same 46 GB of physical RAM the host is using, so 0.93 asks for
# ~96 GB of a pool that is mostly not really there; the kernel thrashes and WSL2 wedges.
#
# The trap is that it does not fail every time. One 0.93 server launched fine, captured
# graphs, and scored GSM8K 1.000 over six runs with a KV pool of 829,842 tokens and
# 7.2-8.7 GB reported free -- so a single clean run is NOT evidence that this setting is
# safe. It is marginal, and marginal here means it takes the host down with it.
#
# 0.75 gives a KV pool of 660,706 tokens with 28.3 GB free, which was never the
# constraint at 64k context: every GSM8K throughput number was identical at 0.75 and
# 0.93. The larger pool buys headroom nobody was using, at the risk of a hard hang.
#
# A large bf16 checkpoint needs its own thought: Qwen3.6-35B-A3B is 71.9 GB of weights
# and 0.75 of ~103 GB is only a ~77 GB budget, leaving almost nothing for KV. Raise it
# for those cautiously and read "Memory pool end. avail mem=" to see what you got.
MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.85}"

# Qwen3.5's stock chat template calls raise_exception() on any reasoning effort outside
# xhigh/medium/low. Claude Code sends one of six levels in output_config and defaults to
# "high", so the template throws and SGLang answers HTTP 500 before any inference runs --
# which the client reports as a transient server fault. The patched copy collapses all six
# levels onto the three the model knows instead of raising. Use SGLANG_CHAT_TEMPLATE= (set
# but empty) to fall back to the stock template, or point it at another file.
CHAT_TEMPLATE="${SGLANG_CHAT_TEMPLATE-/workspace/chat_template_qwen3_agentic.jinja}"
template_flags=()
if [ -n "$CHAT_TEMPLATE" ]; then
    if [ -f "$CHAT_TEMPLATE" ]; then
        template_flags=(--chat-template "$CHAT_TEMPLATE")
    else
        echo "warning: chat template $CHAT_TEMPLATE not found; using the model's own" >&2
    fi
fi

# Which speculative decoder to run:
#   eagle  (default) -- the model's own MTP head. Validated here; see CLAUDE.md.
#   dflash            -- DFlash2, a separate block-diffusion drafter. See below.
#   none              -- no draft model, no graphs (SGLANG_NO_SPEC=1 is a synonym).
SPEC="${SGLANG_SPEC:-eagle}"
[ "${SGLANG_NO_SPEC:-0}" = "1" ] && SPEC=none

if [ "$SPEC" = "none" ]; then
    # The previously validated conservative path: no graph capture, no draft model.
    # Slower (2.5-2.6 tok/s single-stream) but it is what the bf16 baseline in
    # CLAUDE.md was measured with. Use it when bisecting a graph/spec problem.
    perf_flags=(
        --max-running-requests 4
        --disable-cuda-graph
    )
else
    # config.json declares mamba_ssm_dtype=float32. Forcing bf16 is part of the
    # combination under which bounded graph capture runs clean here; on its own,
    # `--cuda-graph-max-bs-decode` produced corrupted output. Which of the two
    # actually fixes it was not isolated, so keep them together.
    export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"

    # 4 is validated here: GSM8K 10q at --parallel 4 scored 1.000 with correct output
    # and no stall, and batch-4 aggregate decode is 23.1 tok/s vs 4.8 at batch 1.
    #
    # --cuda-graph-max-bs-decode used to be passed alongside this, and CLAUDE.md said
    # the two "must always move together". That is not true: SGLang already clamps graph
    # capture to --max-running-requests. A server launched WITHOUT the flag logged
    # "Capture target verify CUDA graph begin ... bs=[1, 2, 3, 4]" -- exactly the bounded
    # capture we wanted -- so the flag was redundant and is no longer passed. Raising
    # SGLANG_MAX_RUNNING therefore moves capture on its own.
    MAX_RUNNING="${SGLANG_MAX_RUNNING:-4}"

    # THE one big lever on this box. The int4 W4A16 path has no native AWQ GEMM on
    # ROCm, so above _FUSED_GEMM_MAX_ROWS=256 rows it dequantizes the whole weight to
    # bf16 and calls hipBLAS (quark/schemes/quark_w4a16_int4.py:144). That temporary
    # is ~51 GB across the model, and at an 8192-row chunk it does not fit in cache,
    # so prefill collapses to ~44 tok/s. Smaller chunks keep the dequantized tile
    # resident and prefill runs ~3.7x faster. Measured, batch 1, 8192 in / 512 out:
    #
    #   chunk 8192 (default): prefill  43.89 tok/s, total 294.49 s
    #   chunk 2048          : prefill  63.58 tok/s, total 236.23 s
    #   chunk 1024          : prefill 163.14 tok/s, total 155.97 s   <-- best
    #   chunk  512          : prefill 146.06 tok/s, total 179.18 s
    #
    # Decode is unaffected (ITL ~207 ms either way). See kb/qwen38-int4-perf.md.
    CHUNKED_PREFILL="${SGLANG_CHUNKED_PREFILL:-1024}"

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
        )
        ;;
    dflash)
        # DFlash2 (incoai/Qwen3.8-27B-DFlash2, 3.85 GB bf16, 5 sliding-attention
        # layers) is a *block-diffusion* drafter: it proposes a whole block of 8
        # tokens in one pass and a learned selector traces a path through the
        # per-position candidates, where EAGLE walks 3 sequential steps for 4
        # tokens. Decoding is lossless -- greedy output matches the target exactly.
        #
        # It works against the int4 target even though the card lists bf16
        # Qwen/Qwen3.8-27B as the base: the drafter taps target layers
        # [5,19,33,47,61] and the target lm_head, and the Quark checkpoint leaves
        # `lm_head.weight` dense BF16 [248320, 5120] (only the decoder layers are
        # AWQ-packed), which is exactly what the DFlash2 selector requires
        # (models/dflash.py:1190 raises otherwise).
        #
        # Two gotchas specific to this box:
        #   - `--speculative-num-draft-tokens` MUST equal the drafter's
        #     dflash_config.block_size (8). SGLang errors if they disagree and
        #     defaults to 16 -- not 8 -- if it cannot read the config.
        #   - there is no flashinfer on ROCm, so the selector's top-k falls back to
        #     torch.topk over a 248,320-entry vocabulary. dflash.py:1039 warns this
        #     "roughly halves end-to-end throughput". Expect the published DFlash2
        #     speedups NOT to transfer in full here.
        # The ROCm paths are real, not accidental: the draft-backend resolver has an
        # explicit is_hip branch (arg_groups/speculative_hook.py:731).
        DRAFT_MODEL="${SGLANG_DFLASH_DRAFT:-incoai/Qwen3.8-27B-DFlash2}"
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

python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static "$MEM_FRACTION" \
    --context-length "$CONTEXT_LEN" \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    ${template_flags[@]+"${template_flags[@]}"} \
    "${perf_flags[@]}"
