# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused add_rms_norm + per-token-group FP8 quantization kernel.

Combines three operations into a single Triton kernel:
  1. Residual add:    x = x + residual; residual = x
  2. RMSNorm:         x = x * rsqrt(mean(x^2) + eps) * weight
  3. Group quant:     x_q = clamp(x / scale, fp8_min, fp8_max)

This eliminates:
  - The intermediate bf16 tensor write from fused_add_rms_norm
  - The intermediate bf16 tensor read by per_token_group_quant
  - One kernel launch overhead
"""

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)


@triton.jit
def _fused_add_rmsnorm_group_quant_kernel(
    # Input pointers
    x_ptr,            # [M, N] bf16/fp16, will be ignored (only residual used)
    residual_ptr,     # [M, N] bf16/fp16, updated in-place
    weight_ptr,       # [N] bf16/fp16
    # Output pointers
    output_q_ptr,     # [M, N] fp8
    output_s_ptr,     # scale output (layout depends on COLUMN_MAJOR)
    # Scalar args
    eps,              # float, for RMSNorm numerical stability
    quant_eps,        # float, for group quant absmax floor
    # Shape
    M,                # num tokens
    N,                # hidden size
    # Strides
    x_stride_m,       # stride of x along token dim
    res_stride_m,     # stride of residual along token dim
    out_q_stride_m,   # stride of output_q along token dim
    out_s_stride_m,   # stride of scale along token dim
    out_s_stride_g,   # stride of scale along group dim
    # Constexpr
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
):
    """
    One program per token. Each program:
      Pass 1: load x + residual, write residual back, accumulate x^2
      Reduce: compute variance, rsqrt
      Pass 2: reload residual, normalize, group-quantize, write fp8 + scale
    """
    token_id = tl.program_id(0)
    if token_id >= M:
        return

    token_id_64 = token_id.to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # ---- Pass 1: residual add + accumulate x^2 ----
    x_offsets = token_id_64 * x_stride_m + cols
    res_offsets = token_id_64 * res_stride_m + cols

    x_val = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    res_val = tl.load(residual_ptr + res_offsets, mask=mask, other=0.0)

    # Residual add in INPUT DTYPE to match CUDA kernel semantics.
    # The CUDA fused_add_rms_norm_kernel does:
    #   scalar_t z = input[...]; z += residual[...];
    # where the add happens at bf16/fp16 precision, not float32.
    added_dtype = (x_val + res_val).to(
        residual_ptr.dtype.element_ty
    )

    # Write updated residual back (in original dtype)
    tl.store(residual_ptr + res_offsets, added_dtype, mask=mask)

    # Convert to float32 for variance computation (matches CUDA kernel)
    added = added_dtype.to(tl.float32)

    # Variance reduction: sum(x^2) / N
    x_sq = added * added
    x_sq = tl.where(mask, x_sq, 0.0)
    variance = tl.sum(x_sq, axis=0) * (1.0 / N)
    rms_inv = tl.math.rsqrt(variance + eps)

    # ---- Pass 2: normalize + per-group quantize ----
    # Use float32 `added` from registers (no reload from global mem).
    # The CUDA kernel reloads bf16 from residual, but since we did
    # the add in bf16 above, our float32 `added` has the same values
    # (bf16 → float32 is exact). Keeping float32 avoids an extra
    # global memory read.

    # RMSNorm: matches CUDA kernel precision:
    #   input = (scalar_t)(x * s_variance) * weight
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    # Step 1: x * rms_inv → float32
    # Step 2: cast to input dtype → bf16/fp16
    # Step 3: multiply by weight (bf16 * bf16 → bf16)
    x_scaled = (added * rms_inv).to(
        residual_ptr.dtype.element_ty
    )
    x_normed = (x_scaled * w).to(tl.float32)

    # ---- Per-group quantization ----
    # Reshape to [GROUPS_PER_ROW, GROUP_SIZE] for group-wise ops
    x_grouped = tl.reshape(x_normed, (GROUPS_PER_ROW, GROUP_SIZE))

    # Per-group absmax
    group_absmax = tl.max(tl.abs(x_grouped), axis=1)
    group_absmax = tl.maximum(group_absmax, quant_eps)

    # Per-group scale: absmax / fp8_max (matches CUDA kernel division)
    group_scale = group_absmax / fp8_max

    # Quantize: clamp(x / scale, fp8_min, fp8_max)
    x_quant = tl.clamp(
        x_grouped / group_scale[:, None], fp8_min, fp8_max
    )
    x_quant = tl.reshape(x_quant, (BLOCK_SIZE,))

    # Write fp8 output
    out_q_offsets = token_id_64 * out_q_stride_m + cols
    tl.store(
        output_q_ptr + out_q_offsets,
        x_quant.to(output_q_ptr.dtype.element_ty),
        mask=mask,
    )

    # Write float32 scale
    group_ids = tl.arange(0, GROUPS_PER_ROW)
    if COLUMN_MAJOR:
        # Column-major layout: shape [GROUPS_PER_ROW, M]
        # strides [1, GROUPS_PER_ROW], i.e. group is fast axis
        out_s_offsets = (
            group_ids * out_s_stride_g
            + token_id_64 * out_s_stride_m
        )
    else:
        # Row-major layout: shape [M, GROUPS_PER_ROW]
        out_s_offsets = (
            token_id_64 * out_s_stride_m + group_ids
        )
    tl.store(output_s_ptr + out_s_offsets, group_scale)


def fused_add_rmsnorm_group_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    group_size: int = 128,
    quant_eps: float = 1e-10,
    column_major_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused add + RMSNorm + per-token-group FP8 quantization.

    Replaces the two-step pipeline:
        residual += x
        x = RMSNorm(x, weight, epsilon)
        x_q, x_s = per_token_group_quant_fp8(x, group_size)

    with a single kernel that does all three.

    Args:
        x: Input tensor [M, N], bf16/fp16.
        residual: Residual tensor [M, N], bf16/fp16.
                  Modified in-place to (x + residual).
        weight: RMSNorm weight [N], bf16/fp16.
        epsilon: RMSNorm epsilon for numerical stability.
        group_size: Quantization group size (default 128).
        quant_eps: Minimum absmax floor to avoid zero-scale.
        column_major_scales: If True, output scales in column-major
            layout [GROUPS_PER_ROW, M] with stride [1, M].

    Returns:
        (x_q, x_s, residual):
            x_q: FP8 quantized output [M, N].
            x_s: Per-group scales, float32. Layout depends on
                 column_major_scales.
            residual: Updated residual (same tensor, modified
                      in-place).
    """
    assert x.dim() == 2
    assert x.is_contiguous()
    assert residual.is_contiguous()
    assert x.shape == residual.shape
    M, N = x.shape
    assert x.shape[-1] % group_size == 0
    assert weight.dim() == 1 and weight.shape[0] == N
    assert weight.is_contiguous()
    groups_per_row = N // group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_min_val, fp8_max_val = get_fp8_min_max()

    # Allocate fp8 output
    output_q = torch.empty(M, N, device=x.device, dtype=fp8_dtype)

    # Allocate scale output with appropriate layout
    if column_major_scales:
        # Column-major layout: logical [M, groups_per_row], strides (1, M)
        # CUDA detects column-major via stride(0) < stride(1).
        # Triton offset: token_id * stride(0) + group_id * stride(1)
        #   = token_id * 1 + group_id * M
        output_s = torch.empty(
            groups_per_row, M,
            device=x.device,
            dtype=torch.float32,
        ).permute(1, 0)
        out_s_stride_g = output_s.stride(1)  # = M
        out_s_stride_m = output_s.stride(0)  # = 1
        assert out_s_stride_g == M and out_s_stride_m == 1
    else:
        # Row-major: [M, groups_per_row]
        output_s = torch.empty(
            M, groups_per_row,
            device=x.device,
            dtype=torch.float32,
        )
        out_s_stride_g = output_s.stride(1)
        out_s_stride_m = output_s.stride(0)

    # BLOCK_SIZE must be next power of 2 >= N
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Tune num_warps based on hidden size
    # N=2048 → BLOCK=2048 → 8 warps (256 threads, 8 elem/thread)
    # N=4096 → BLOCK=4096 → 16 warps (512 threads, 8 elem/thread)
    num_warps = max(4, min(BLOCK_SIZE // 256, 16))
    num_stages = 1

    grid = (M,)
    _fused_add_rmsnorm_group_quant_kernel[grid](
        x,
        residual,
        weight,
        output_q,
        output_s,
        epsilon,
        quant_eps,
        M,
        N,
        x.stride(0),
        residual.stride(0),
        output_q.stride(0),
        output_s.stride(0),
        out_s_stride_g,
        fp8_min=fp8_min_val,
        fp8_max=fp8_max_val,
        GROUP_SIZE=group_size,
        GROUPS_PER_ROW=groups_per_row,
        BLOCK_SIZE=BLOCK_SIZE,
        COLUMN_MAJOR=column_major_scales,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output_q, output_s, residual
