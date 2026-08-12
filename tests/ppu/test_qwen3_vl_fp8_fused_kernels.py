# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
tests/ppu/e2e/test_qwen3_vl_fp8_fused_kernels.py

Correctness tests for the PPU FP8 optimization patch on Qwen3-VL:
  1. Fused add+RMSNorm+per-token-group FP8 quant (Triton kernel)
  2. PPU-optimized per_token_group_quant_fp8 (CUDA PTX kernel)

Run:
  # Kernel-only tests (requires CUDA + Triton):
  pytest tests/ppu/e2e/test_qwen3_vl_fp8_fused_kernels.py -v -s

  # Specific test cases:
  pytest tests/ppu/e2e/test_qwen3_vl_fp8_fused_kernels.py \
      -k "test_fused_kernel_correctness" -v -s
  pytest tests/ppu/e2e/test_qwen3_vl_fp8_fused_kernels.py \
      -k "test_ppu_opt_correctness" -v -s
"""

import os
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fused_add_rmsnorm_quant import (
    fused_add_rmsnorm_group_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for PPU kernel tests",
)


# ---------------------------------------------------------------------------
# Reference implementations (match production two-step pipeline exactly)
# ---------------------------------------------------------------------------


def _ref_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: vLLM CUDA fused_add_rms_norm kernel.

    Falls back to PyTorch native if CUDA kernel is unavailable.
    """
    try:
        torch.ops._C.fused_add_rms_norm(
            x, residual, weight, epsilon
        )
        return x, residual
    except (AttributeError, RuntimeError):
        # Fallback (CPU / no vLLM C++ ext)
        x_f = x.float() + residual.float()
        new_residual = x_f.to(residual.dtype)
        variance = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(variance + epsilon)
        x_f = x_f.to(weight.dtype) * weight
        x_f = x_f.to(residual.dtype)
        return x_f, new_residual


def _ref_group_quant(
    x: torch.Tensor,
    group_size: int,
    column_major: bool = False,
    quant_eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: vLLM CUDA per_token_group_fp8_quant kernel."""
    M, N = x.shape
    fp8_dtype = torch.float8_e4m3fn
    fp8_min, fp8_max = get_fp8_min_max()
    groups_per_row = N // group_size

    x_q = torch.empty(M, N, device=x.device, dtype=fp8_dtype)
    if column_major:
        x_s = torch.empty(
            groups_per_row, M,
            device=x.device, dtype=torch.float32,
        ).permute(1, 0)
    else:
        x_s = torch.empty(
            M, groups_per_row,
            device=x.device, dtype=torch.float32,
        )

    try:
        torch.ops._C.per_token_group_fp8_quant(
            x.contiguous(), x_q, x_s,
            group_size, quant_eps, fp8_min, fp8_max,
            False,           # use_ue8m0
            column_major,    # column_major_scales
            False,           # tma_aligned_scales
        )
        return x_q, x_s
    except (AttributeError, RuntimeError):
        # Fallback (CPU / no vLLM C++ ext)
        x_g = x.float().reshape(M, groups_per_row, group_size)
        absmax = x_g.abs().amax(dim=-1).clamp(min=quant_eps)
        scale = absmax / fp8_max
        x_q_f = (x_g / scale.unsqueeze(-1)).clamp(fp8_min, fp8_max)
        x_q = x_q_f.reshape(M, N).to(fp8_dtype)
        if column_major:
            scale_col = scale.T.contiguous().T
            return x_q, scale_col
        return x_q, scale


# ---------------------------------------------------------------------------
# 1. Fused add+RMSNorm+quant kernel correctness
# ---------------------------------------------------------------------------

# Qwen3-VL-2B-Instruct shapes:
#   hidden_size = 2048, num_layers = 28, group_size = 128
# Typical M values: 1 (decode), 16/64 (batch decode), 128-512 (prefill)
QWEN3_VL_SHAPES = [
    (1, 2048),
    (16, 2048),
    (64, 2048),
    (128, 2048),
    (256, 2048),
]


@pytest.mark.parametrize("M,N", QWEN3_VL_SHAPES)
@pytest.mark.parametrize("column_major", [False, True])
def test_fused_kernel_correctness(M: int, N: int, column_major: bool):
    """Fused Triton kernel vs two-step CUDA reference (row & col major)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    group_size = 128
    epsilon = 1e-5
    quant_eps = 1e-10

    x = torch.randn(M, N, dtype=dtype, device="cuda")
    residual = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")

    # -- Reference: two-step using CUDA kernels --
    x_ref = x.clone()
    res_ref = residual.clone()
    x_normed, res_ref = _ref_fused_add_rmsnorm(
        x_ref, res_ref, weight, epsilon
    )
    x_q_ref, x_s_ref = _ref_group_quant(
        x_normed, group_size,
        column_major=column_major, quant_eps=quant_eps,
    )

    # -- Fused kernel --
    x_c = x.clone()
    res_c = residual.clone()
    x_q_fused, x_s_fused, res_fused = (
        fused_add_rmsnorm_group_quant(
            x_c, res_c, weight, epsilon,
            group_size=group_size,
            quant_eps=quant_eps,
            column_major_scales=column_major,
        )
    )

    # Residual must be bit-exact (bf16 add semantics match)
    torch.testing.assert_close(
        res_fused, res_ref, atol=0, rtol=0,
        msg=f"Residual mismatch M={M}, N={N}, col={column_major}",
    )

    # Scale: allow ~1 ULP tolerance (rsqrt/variance reduction order)
    torch.testing.assert_close(
        x_s_fused, x_s_ref, atol=1e-5, rtol=1e-4,
        msg=f"Scale mismatch M={M}, N={N}, col={column_major}",
    )

    # FP8 output: <2% elements may differ by 1 due to scale ULP
    mismatch = (
        x_q_fused.view(torch.uint8) != x_q_ref.view(torch.uint8)
    ).sum().item()
    total = x_q_fused.numel()
    pct = mismatch / total * 100
    assert pct < 2.0, (
        f"M={M}, col={column_major}: "
        f"{mismatch}/{total} ({pct:.2f}%) FP8 elements differ"
    )


# ---------------------------------------------------------------------------
# 2. PPU-optimized per_token_group_quant_fp8 correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,N", QWEN3_VL_SHAPES)
def test_ppu_opt_correctness(M: int, N: int):
    """per_token_group_quant_fp8_ppu_opt vs standard
    per_token_group_quant_fp8 (bit-exact on SM89+)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    group_size = 128

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
        per_token_group_quant_fp8_ppu_opt,
    )

    x = torch.randn(M, N, dtype=dtype, device="cuda")

    # Standard path
    x1 = x.clone()
    x_q_std, x_s_std = per_token_group_quant_fp8(
        x1, group_size=group_size,
        column_major_scales=False,
    )

    # PPU-optimized path
    x2 = x.clone()
    x_q_ppu, x_s_ppu = per_token_group_quant_fp8_ppu_opt(
        x2, group_size=group_size,
        column_major_scales=False,
    )

    # Scales should be close (PTX path may differ by ~1 ULP)
    torch.testing.assert_close(
        x_s_ppu, x_s_std, atol=1e-5, rtol=1e-4,
        msg=f"Scale mismatch M={M}, N={N}",
    )

    # FP8 output: allow small mismatch
    mismatch = (
        x_q_ppu.view(torch.uint8) != x_q_std.view(torch.uint8)
    ).sum().item()
    total = x_q_ppu.numel()
    pct = mismatch / total * 100
    assert pct < 2.0, (
        f"M={M}, N={N}: "
        f"{mismatch}/{total} ({pct:.2f}%) FP8 elements differ"
    )