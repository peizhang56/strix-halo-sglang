# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton W4A16 kernels for GPTQ checkpoints on ROCm.

The GEMM is adapted from vLLM's Triton W4A16 kernel. GPTQ checkpoints pack
weights along K, so weights are repacked once at load time to the N-packed
layout consumed by the GEMM.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

GPTQ_TRITON_SUPPORTED_GROUP_SIZES = {-1, 32, 64, 128, 256}

# Offline-tuned on Radeon 8060S (gfx1151), ROCm 7.2.4, Triton 3.5.1.
# Key: (M bucket, K, N), value: (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages).
#
# Entries marked "Qwen3.8" were added/corrected by tune_w4a16.py and
# confirmed by verify_w4a16.py, which interleaves baseline and
# candidate within one pass and takes the median ratio over 7 repeats.
# Separate-pass timing on this box drifts ~8%, which is enough to invent
# speedups that do not exist, so nothing below ~1.08x was adopted.
#
# The shapes were read off a LIVE server (instrumented gptq_w4a16_skinny_gemm)
# rather than derived from config.json -- a derivation of the qkv shape gave
# (5120, 8192), which this model never executes. m=8 is DFLASH's decode block
# (8 draft tokens x batch 1); m=16 is batch 2. Per decode step the call mix is
# o_proj/gate_up/down x4, gdn_qkvz/gdn_ba x3, attn_qkv x1.
#
# (5120, 96) is deliberately absent: every candidate tried was SLOWER than the
# generic heuristic there (0.86x at m=8), and at 0.04 ms it is not worth more.
_GFX1151_W4A16_CONFIGS = {
    (8, 5120, 16384): (16, 32, 128, 2, 2),  # Qwen3.8 gdn_qkvz   1.09x
    (8, 17408, 5120): (8, 32, 128, 2, 1),  # Qwen3.8 down_proj  1.08x
    (12, 5120, 16384): (16, 64, 128, 4, 1),
    (12, 5120, 14336): (32, 16, 128, 2, 1),
    (12, 5120, 34816): (16, 32, 128, 2, 1),
    (12, 17408, 5120): (16, 32, 128, 2, 1),
    (16, 5120, 16384): (16, 64, 128, 4, 2),  # Qwen3.8 gdn_qkvz   1.16x
    (16, 5120, 34816): (16, 32, 128, 2, 1),
    (16, 17408, 5120): (16, 32, 128, 2, 2),  # Qwen3.8 down_proj  1.11x
    (64, 5120, 16384): (64, 32, 128, 4, 1),
    (64, 5120, 96): (128, 64, 64, 8, 2),
    (64, 6144, 5120): (128, 32, 128, 4, 1),
    (64, 5120, 14336): (64, 64, 128, 4, 1),
    (64, 5120, 34816): (64, 64, 128, 4, 1),
    (64, 17408, 5120): (64, 64, 128, 4, 1),
    (128, 5120, 16384): (128, 64, 64, 8, 2),
    (128, 5120, 96): (64, 32, 128, 4, 1),
    (128, 6144, 5120): (128, 32, 128, 4, 1),
    (128, 5120, 14336): (128, 64, 64, 8, 2),
    (128, 5120, 34816): (128, 64, 64, 8, 2),
    (128, 17408, 5120): (128, 64, 64, 8, 2),
    (512, 5120, 16384): (128, 128, 64, 8, 2),
    (512, 6144, 5120): (128, 64, 64, 8, 2),
    (512, 5120, 14336): (128, 128, 64, 8, 2),
    (512, 17408, 5120): (128, 256, 32, 8, 2),
    (1024, 5120, 16384): (128, 128, 64, 8, 2),
    (1024, 6144, 5120): (128, 128, 64, 8, 2),
    (1024, 5120, 14336): (128, 256, 32, 8, 2),
    (1024, 5120, 34816): (128, 512, 32, 16, 2),
    (1024, 17408, 5120): (128, 128, 64, 8, 2),
    (4096, 5120, 16384): (128, 256, 32, 8, 2),
    (4096, 5120, 96): (64, 128, 32, 4, 3),
    (4096, 5120, 14336): (128, 256, 32, 8, 2),
    (4096, 5120, 34816): (128, 256, 32, 8, 2),
    (8, 5120, 14336): (16, 64, 128, 4, 2),  # Qwen3.8 attn_qkv   1.35x new
    (8, 6144, 5120): (16, 64, 128, 4, 2),  # Qwen3.8 o_proj     1.20x new
    (8, 5120, 34816): (16, 32, 128, 2, 2),  # Qwen3.8 gate_up    1.35x new
    (16, 5120, 14336): (16, 64, 128, 4, 2),  # Qwen3.8 attn_qkv   1.36x new
    (16, 6144, 5120): (16, 32, 128, 2, 2),  # Qwen3.8 o_proj     1.34x new
}


@functools.cache
def _is_gfx1151() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    return arch.startswith("gfx1151")


def _gfx1151_w4a16_config(
    m: int, k: int, n: int, group_size: int
) -> tuple[int, int, int, int, int] | None:
    if (
        os.getenv("SGLANG_GFX1151_W4A16_TUNING", "1").lower()
        not in ("1", "true")
        or group_size != 128
        or not _is_gfx1151()
    ):
        return None
    if m in (8, 12, 16):
        bucket = m
    elif m <= 32:
        return None
    elif m <= 64:
        bucket = 64
    elif m <= 128:
        bucket = 128
    elif m <= 512:
        bucket = 512
    elif m <= 1024:
        bucket = 1024
    elif m <= 4096:
        bucket = 4096
    else:
        return None
    return _GFX1151_W4A16_CONFIGS.get((bucket, k, n))


@triton.jit
def _repack_gptq_w4_kernel(
    source,
    destination,
    K,
    N,
    BLOCK_K: tl.constexpr,
    BLOCK_N_PACKED: tl.constexpr,
):
    """Change [K/8, N] K-packed weights into [K, N/8] N-packed weights."""
    offs_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_np = (
        tl.program_id(1) * BLOCK_N_PACKED + tl.arange(0, BLOCK_N_PACKED)
    )
    nibble = tl.arange(0, 8)
    offs_n = offs_np[:, None] * 8 + nibble[None, :]

    source_offsets = (offs_k // 8)[:, None, None] * N + offs_n[None, :, :]
    source_mask = (offs_k[:, None, None] < K) & (offs_n[None, :, :] < N)
    packed_k = tl.load(source + source_offsets, mask=source_mask, other=0)

    values = (packed_k >> ((offs_k % 8)[:, None, None] * 4)) & 0xF
    packed_n = tl.sum(values << (nibble[None, None, :] * 4), axis=2)

    destination_offsets = offs_k[:, None] * (N // 8) + offs_np[None, :]
    destination_mask = (offs_k[:, None] < K) & (offs_np[None, :] < N // 8)
    tl.store(destination + destination_offsets, packed_n, mask=destination_mask)


def repack_gptq_w4(qweight: torch.Tensor, k: int) -> torch.Tensor:
    """Repack GPTQ int4 weights from [K/8, N] to [K, N/8]."""
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError("GPTQ Triton requires a two-dimensional int32 qweight")
    if k % 8 != 0 or qweight.shape[0] != k // 8:
        raise ValueError(
            f"Unexpected GPTQ qweight shape {tuple(qweight.shape)} for K={k}"
        )

    n = qweight.shape[1]
    if n % 8 != 0:
        raise ValueError(f"GPTQ Triton requires N divisible by 8, but got N={n}")

    output = torch.empty((k, n // 8), dtype=qweight.dtype, device=qweight.device)
    block_k = 8
    block_n_packed = 32
    grid = (triton.cdiv(k, block_k), triton.cdiv(n // 8, block_n_packed))
    _repack_gptq_w4_kernel[grid](
        qweight,
        output,
        k,
        n,
        BLOCK_K=block_k,
        BLOCK_N_PACKED=block_n_packed,
    )
    return output


@triton.jit
def _repack_gptq_w4_to_skinny_kernel(
    source,
    destination,
    K8,
    N,
    BLOCK_K8: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Transpose GPTQ words and apply the ExLlama nibble shuffle."""
    offs_k8 = tl.program_id(0) * BLOCK_K8 + tl.arange(0, BLOCK_K8)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    source_offsets = offs_k8[:, None] * N + offs_n[None, :]
    word = tl.load(
        source + source_offsets,
        mask=(offs_k8[:, None] < K8) & (offs_n[None, :] < N),
        other=0,
    )

    shuffled = word & 0x0000000F
    shuffled |= (word & 0x00000F00) >> 4
    shuffled |= (word & 0x000F0000) >> 8
    shuffled |= (word & 0x0F000000) >> 12
    shuffled |= (word & 0x000000F0) << 12
    shuffled |= (word & 0x0000F000) << 8
    shuffled |= (word & 0x00F00000) << 4
    shuffled |= word & -0x10000000

    destination_offsets = offs_n[None, :] * K8 + offs_k8[:, None]
    tl.store(
        destination + destination_offsets,
        shuffled,
        mask=(offs_k8[:, None] < K8) & (offs_n[None, :] < N),
    )


def repack_gptq_w4_to_skinny(qweight: torch.Tensor, k: int) -> torch.Tensor:
    """Convert [K/8, N] sequential GPTQ words to [N, K/8] ExLlama words."""
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError("GPTQ skinny kernels require a 2D int32 qweight")
    if k % 8 != 0 or qweight.shape[0] != k // 8:
        raise ValueError(
            f"Unexpected GPTQ qweight shape {tuple(qweight.shape)} for K={k}"
        )

    k8, n = qweight.shape
    output = torch.empty((n, k8), dtype=qweight.dtype, device=qweight.device)
    block_k8 = 32
    block_n = 32
    grid = (triton.cdiv(k8, block_k8), triton.cdiv(n, block_n))
    _repack_gptq_w4_to_skinny_kernel[grid](
        qweight,
        output,
        k8,
        n,
        BLOCK_K8=block_k8,
        BLOCK_N=block_n,
    )
    return output


@triton.jit
def _repack_awq_w4_to_skinny_kernel(
    source,
    destination,
    K,
    N,
    SIGNED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K8: tl.constexpr,
):
    """Transpose AWQ words and move their packing from N to K."""
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k8 = tl.program_id(1) * BLOCK_K8 + tl.arange(0, BLOCK_K8)
    lanes = tl.arange(0, 8)
    offs_k = offs_k8[:, None] * 8 + lanes[None, :]

    source_offsets = (
        offs_k[None, :, :] * (N // 8) + (offs_n // 8)[:, None, None]
    )
    source_mask = (offs_n[:, None, None] < N) & (offs_k[None, :, :] < K)
    words = tl.load(source + source_offsets, mask=source_mask, other=0)

    # AWQ maps logical lanes to physical nibbles as [0, 4, 1, 5, 2, 6, 3, 7].
    source_nibble = (offs_n % 2) * 4 + offs_n // 2 % 4
    values = (words >> (source_nibble[:, None, None] * 4)) & 0xF
    if SIGNED:
        values ^= 0x8

    destination_nibble = (lanes % 2) * 4 + lanes // 2
    packed = tl.sum(values << (destination_nibble[None, None, :] * 4), axis=2)
    destination_offsets = offs_n[:, None] * (K // 8) + offs_k8[None, :]
    destination_mask = (offs_n[:, None] < N) & (offs_k8[None, :] < K // 8)
    tl.store(destination + destination_offsets, packed, mask=destination_mask)


def repack_awq_w4_to_skinny(
    qweight: torch.Tensor, *, signed: bool = False
) -> torch.Tensor:
    """Convert AWQ-packed [K, N/8] words to ExLlama-packed [N, K/8].

    ``signed=True`` also converts each signed int4 code to the unsigned
    zero-point-8 convention consumed by the skinny kernels.
    """
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError("AWQ skinny kernels require a 2D int32 qweight")

    k, n8 = qweight.shape
    n = n8 * 8
    if k % 8 != 0:
        raise ValueError(f"AWQ skinny kernels require K divisible by 8, got K={k}")

    output = torch.empty((n, k // 8), dtype=qweight.dtype, device=qweight.device)
    block_n = 32
    block_k8 = 32
    grid = (triton.cdiv(n, block_n), triton.cdiv(k // 8, block_k8))
    _repack_awq_w4_to_skinny_kernel[grid](
        qweight,
        output,
        k,
        n,
        SIGNED=signed,
        BLOCK_N=block_n,
        BLOCK_K8=block_k8,
    )
    return output


@triton.jit
def _repack_awq_qzeros_to_skinny_kernel(
    source,
    destination,
    NUM_GROUPS,
    N8,
    SIGNED: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_N8: tl.constexpr,
):
    """Undo AWQ's nibble interleave and transpose packed zero points."""
    offs_group = (
        tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    )
    offs_n8 = tl.program_id(1) * BLOCK_N8 + tl.arange(0, BLOCK_N8)
    source_offsets = offs_group[:, None] * N8 + offs_n8[None, :]
    words = tl.load(
        source + source_offsets,
        mask=(offs_group[:, None] < NUM_GROUPS) & (offs_n8[None, :] < N8),
        other=0,
    )

    lanes = tl.arange(0, 8)
    source_nibble = (lanes % 2) * 4 + lanes // 2
    values = (words[:, :, None] >> (source_nibble[None, None, :] * 4)) & 0xF
    if SIGNED:
        values ^= 0x8
    packed = tl.sum(values << (lanes[None, None, :] * 4), axis=2)

    destination_offsets = offs_n8[None, :] * NUM_GROUPS + offs_group[:, None]
    mask = (offs_group[:, None] < NUM_GROUPS) & (offs_n8[None, :] < N8)
    tl.store(destination + destination_offsets, packed, mask=mask)


def repack_awq_qzeros_to_skinny(
    qzeros: torch.Tensor, *, signed: bool = False
) -> torch.Tensor:
    """Convert AWQ-packed [groups, N/8] zero points to [N/8, groups]."""
    if qzeros.dtype != torch.int32 or qzeros.ndim != 2:
        raise ValueError("AWQ skinny kernels require 2D int32 zero points")

    num_groups, n8 = qzeros.shape
    output = torch.empty(
        (n8, num_groups), dtype=qzeros.dtype, device=qzeros.device
    )
    block_groups = 32
    block_n8 = 32
    grid = (
        triton.cdiv(num_groups, block_groups),
        triton.cdiv(n8, block_n8),
    )
    _repack_awq_qzeros_to_skinny_kernel[grid](
        qzeros,
        output,
        num_groups,
        n8,
        SIGNED=signed,
        BLOCK_GROUPS=block_groups,
        BLOCK_N8=block_n8,
    )
    return output


@triton.jit
def _gptq_w4a16_skinny_gemm_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    zeros_ptr,
    c_ptr,
    M,
    N,
    K,
    K8,
    num_groups,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """W4A16 GEMM reading ExLlama-shuffled [N, K/8] weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    shifts = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts = tl.reshape(
        tl.broadcast_to(shifts[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_N, BLOCK_K))
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )

        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_packed = tl.load(
            b_ptr + offs_n[:, None] * K8 + offs_k8[None, :],
            mask=(offs_n[:, None] < N) & (offs_k8[None, :] < K8),
            other=0,
        )
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF

        group_idx = (k_start * BLOCK_K) // group_size
        scales = tl.load(
            scales_ptr + offs_n * num_groups + group_idx,
            mask=offs_n < N,
            other=1.0,
        )
        if HAS_ZP:
            zero_word = tl.load(
                zeros_ptr + (offs_n // 8) * num_groups + group_idx,
                mask=offs_n < N,
                other=0,
            )
            zero = (zero_word >> (4 * (offs_n % 8))) & 0xF
            b = (b - zero[:, None]).to(scales.dtype) * scales[:, None]
        else:
            b = (b - ZP_BIAS).to(scales.dtype) * scales[:, None]

        accumulator += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    output = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        c_ptrs,
        output,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def gptq_w4a16_skinny_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    qzeros: torch.Tensor | None = None,
    zp_bias: int = 8,
) -> torch.Tensor:
    """Run W4A16 GEMM from the shared HIP-skinny weight layout."""
    if input.ndim != 2 or input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("GPTQ skinny Triton expects a 2D fp16/bf16 input")
    input = input.contiguous()
    if not qweight.is_contiguous() or not scales.is_contiguous():
        raise ValueError("GPTQ skinny weights and scales must be contiguous")

    m, k = input.shape
    n, k8 = qweight.shape
    effective_group_size = k if group_size == -1 else group_size
    if k8 != k // 8:
        raise ValueError(
            f"Unexpected skinny qweight shape {tuple(qweight.shape)} for K={k}"
        )
    if k % effective_group_size != 0:
        raise ValueError(
            f"GPTQ input size {k} is not divisible by group size "
            f"{effective_group_size}"
        )
    num_groups = k // effective_group_size
    if scales.shape != (n, num_groups):
        raise ValueError(
            f"Unexpected skinny scales shape {tuple(scales.shape)}; "
            f"expected {(n, num_groups)}"
        )
    if qzeros is not None and qzeros.shape != (n // 8, num_groups):
        raise ValueError(
            f"Unexpected skinny qzeros shape {tuple(qzeros.shape)}; "
            f"expected {(n // 8, num_groups)}"
        )

    tuned_config = _gfx1151_w4a16_config(
        m, k, n, effective_group_size
    )
    num_stages = None
    if tuned_config is not None:
        block_m, block_n, block_k, num_warps, num_stages = tuned_config
    else:
        if m <= 32:
            block_m, block_n, block_k, num_warps = 32, 32, 128, 4
        elif m <= 64:
            block_m, block_n, block_k, num_warps = 64, 64, 32, 4
        elif m <= 128:
            if k >= 2 * n:
                block_m, block_n, block_k, num_warps = 64, 16, 64, 1
            elif n > k:
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            else:
                block_m, block_n, block_k, num_warps = 64, 32, 64, 4
        elif m <= 1024:
            if k >= 2 * n:
                block_m, block_n, block_k, num_warps = 64, 64, 64, 4
            elif n >= 4 * k:
                block_m, block_n, block_k, num_warps = 128, 64, 64, 8
            else:
                block_m, block_n, block_k, num_warps = 64, 128, 32, 4
        elif k >= 2 * n:
            block_m, block_n, block_k, num_warps = 128, 512, 32, 16
        else:
            block_m, block_n, block_k, num_warps = 128, 64, 64, 8
    block_k = min(block_k, effective_group_size)

    output = torch.empty((m, n), dtype=input.dtype, device=input.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    launch_kwargs = {} if num_stages is None else {"num_stages": num_stages}
    _gptq_w4a16_skinny_gemm_kernel[grid](
        input,
        qweight,
        scales,
        qzeros if qzeros is not None else scales,
        output,
        m,
        n,
        k,
        k8,
        num_groups,
        effective_group_size,
        HAS_ZP=qzeros is not None,
        ZP_BIAS=zp_bias,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        **launch_kwargs,
    )
    return output


@triton.jit
def _gptq_w4a16_gemm_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    zeros_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    group_size,
    ZERO_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute A[M,K] @ dequant(B)[K,N] for GPTQ-v1/v2 int4 weights."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    shifts_row = tl.arange(0, 8) * 4
    shifts_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_2d, (BLOCK_N,))
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & mask_k[None, :],
            other=0.0,
        )

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_packed = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & (offs_bn[None, :] < N // 8),
            other=0,
        )
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF

        group_idx = (k_start * BLOCK_K) // group_size
        scale_offsets = group_idx * N + offs_n
        scales = tl.load(
            scales_ptr + scale_offsets, mask=offs_n < N, other=1.0
        )
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        zero_offsets = group_idx * (N // 8) + offs_bn
        z_packed = tl.load(
            zeros_ptr + zero_offsets, mask=offs_bn < N // 8, other=0
        )
        zero = tl.interleave(z_packed, z_packed)
        zero = tl.interleave(zero, zero)
        zero = tl.interleave(zero, zero)
        zero = (zero >> shifts_1d) & 0xF
        zero = (zero + ZERO_OFFSET) & 0xF
        zero = tl.broadcast_to(zero[None, :], (BLOCK_K, BLOCK_N))

        b = (b - zero).to(a.dtype) * scales
        accumulator += tl.dot(a, b, out_dtype=tl.float32)

    output = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        output,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def gptq_w4a16_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
    use_v2_format: bool = False,
) -> torch.Tensor:
    """Run a fused GPTQ int4 GEMM on fp16 or bf16 activations."""
    if input.ndim != 2:
        raise ValueError("GPTQ Triton GEMM expects a two-dimensional input")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported GPTQ activation dtype: {input.dtype}")
    if not input.is_contiguous():
        input = input.contiguous()
    if not qweight.is_contiguous() or not scales.is_contiguous():
        raise ValueError("GPTQ Triton weights and scales must be contiguous")

    m, k = input.shape
    n = qweight.shape[1] * 8
    effective_group_size = k if group_size == -1 else group_size
    if group_size not in GPTQ_TRITON_SUPPORTED_GROUP_SIZES and group_size != k:
        raise ValueError(f"Unsupported GPTQ Triton group size: {group_size}")
    if k % effective_group_size != 0:
        raise ValueError(
            f"GPTQ input size {k} is not divisible by group size "
            f"{effective_group_size}"
        )
    if qweight.shape != (k, n // 8):
        raise ValueError(
            f"Unexpected repacked qweight shape {tuple(qweight.shape)} for "
            f"K={k}, N={n}"
        )
    expected_scales = (k // effective_group_size, n)
    expected_zeros = (k // effective_group_size, n // 8)
    if scales.shape != expected_scales:
        raise ValueError(
            f"Unexpected GPTQ scales shape {tuple(scales.shape)}; "
            f"expected {expected_scales}"
        )
    if qzeros.shape != expected_zeros:
        raise ValueError(
            f"Unexpected GPTQ qzeros shape {tuple(qzeros.shape)}; "
            f"expected {expected_zeros}"
        )

    if m <= 32:
        block_m, block_n, block_k = 32, 32, 64
    elif m <= 64:
        block_m, block_n, block_k = 64, 64, 32
    else:
        block_m, block_n, block_k = 128, 32, 64
    if effective_group_size < block_k:
        block_k = effective_group_size

    output = torch.empty((m, n), dtype=input.dtype, device=input.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _gptq_w4a16_gemm_kernel[grid](
        input,
        qweight,
        scales,
        qzeros,
        output,
        m,
        n,
        k,
        input.stride(0),
        input.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        output.stride(0),
        output.stride(1),
        effective_group_size,
        ZERO_OFFSET=0 if use_v2_format else 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return output


__all__ = [
    "GPTQ_TRITON_SUPPORTED_GROUP_SIZES",
    "gptq_w4a16_gemm",
    "gptq_w4a16_skinny_gemm",
    "repack_awq_qzeros_to_skinny",
    "repack_awq_w4_to_skinny",
    "repack_gptq_w4",
    "repack_gptq_w4_to_skinny",
]
