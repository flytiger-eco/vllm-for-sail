# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops import (
    fused_inv_rope_fp8_quant,
    fused_inv_rope_float32,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, wo_a.weight_scale_inv),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))


# FIXME: use deepgemm int8 einsum later
def deep_gemm_int8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
):
    # Fused inv RoPE → float32 (skip INT8 roundtrip)
    o_f = fused_inv_rope_float32(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
    )
    # Weight dequant (per-channel INT8 → float32) + reshape for einsum
    wo_a_f = _dequant_channel(wo_a.weight, wo_a.weight_scale).reshape(
        n_groups, o_lora_rank, -1
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    torch.ops.vllm.deepseek_v4_unquantized_einsum(
        o_f, wo_a_f, z, "bhr,hdr->bhd",
    )
    return wo_b(z.flatten(1))


# FIXME: use deepgemm fp8 einsum later
def deep_gemm_fp8_channel_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
):
    wo_a_view = wo_a.weight.reshape(
        n_groups, o_lora_rank, -1
    )
    o_f = fused_inv_rope_float32(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    torch.ops.vllm.deepseek_v4_unquantized_einsum(
        o_f, wo_a_view, z, "bhr,hdr->bhd",
    )
    return wo_b(z.flatten(1))



def _dequant_channel(
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize per-channel INT8 quantized weight back to float32.

    For a weight tensor of shape [K, N]:
      - b:        INT8 quantized values, shape [K, N]
      - b_scale:  per-output-channel scales, shape [K] or [K, 1]

    Returns:
      Dequantized float32 tensor of shape [K, N].
    """
    if b_scale is None:
        return b.to(torch.float32)

    b_scale_f = b_scale.to(torch.float32)

    if b_scale_f.dim() == 1:
        b_scale_f = b_scale_f.unsqueeze(-1)  # [K] -> [K, 1] for broadcasting
    # b.to(float32) handles the int8 -> float conversion,
    # then multiply by per-channel scale.
    return (b.to(torch.float32) * b_scale_f)


def _deepseek_v4_unquantized_einsum_torch(
    equation: str,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> None:
    result = torch.einsum(equation, a, b)
    out.copy_(result.to(out.dtype))


def deepseek_v4_unquantized_einsum(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    _deepseek_v4_unquantized_einsum_torch(equation, a, b, out)


def deepseek_v4_unquantized_einsum_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    equation: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_unquantized_einsum",
    op_func=deepseek_v4_unquantized_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_unquantized_einsum_fake,
)