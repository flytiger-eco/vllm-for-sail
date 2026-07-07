# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops import (
    fused_inv_rope_fp8_quant,
    fused_inv_rope_fp8_quant_channelwise,
    fused_inv_rope_int8_quant_channelwise,
)
from vllm.platforms import current_platform
from vllm.utils.ppu_deep_gemm import fp8_einsum
from vllm.utils.ppu_deep_gemm import int8_einsum
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
    # Channelwise INT8 GEMM via PPU DeepGEMM int8_einsum:
    #   - activation: per-(token, group) symmetric INT8 quant.
    #     o_int8 has shape (b=T, h=G, r=R) so o_scale is (b, h, 1).
    #   - weight:     per-output-channel INT8. compressed-tensors
    #     W8A8 channel scheme stores wo_a as 2D [G*d, R] / scale as
    #     [G*d, 1]; reshape to 3D so int8_bmm sees (h=G, d=lora_rank,
    #     r=R) for `hdr` and a (h, d, 1) scale broadcasting along r.
    o_int8, o_scale = fused_inv_rope_int8_quant_channelwise(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
    )
    wo_a_view = wo_a.weight.reshape(
        n_groups, o_lora_rank, -1
    )
    wo_a_scale_view = wo_a.weight_scale.reshape(
        n_groups, o_lora_rank, 1
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    torch.ops.vllm.deepseek_v4_int8_einsum(
        o_int8,
        o_scale,
        wo_a_view,
        wo_a_scale_view,
        z,
        "bhr,hdr->bhd",
        [1, 1, o_int8.shape[-1]],
    )
    return wo_b(z.flatten(1))


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
    # Channelwise FP8 GEMM via PPU DeepGEMM fp8_einsum:
    #   - activation: per-(token, group) symmetric FP8 quant.
    #     o_fp8 has shape (b=T, h=G, r=R) so o_scale is (b, h, 1).
    #   - weight:     per-output-channel FP8. Channelwise scheme stores
    #     wo_a as 2D [G*d, R] / scale as [G*d, 1]; reshape to 3D so
    #     fp8_bmm sees (h=G, d=lora_rank, r=R) for `hdr` and a
    #     (h, d, 1) scale broadcasting along r.
    o_fp8, o_scale = fused_inv_rope_fp8_quant_channelwise(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
    )
    wo_a_view = wo_a.weight.reshape(
        n_groups, o_lora_rank, -1
    )
    wo_a_scale_view = wo_a.weight_scale.reshape(
        n_groups, o_lora_rank, 1
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    torch.ops.vllm.deepseek_v4_fp8_channel_einsum(
        o_fp8,
        o_scale,
        wo_a_view,
        wo_a_scale_view,
        z,
        "bhr,hdr->bhd",
        [1, 1, o_fp8.shape[-1]],
    )
    return wo_b(z.flatten(1))


def deepseek_v4_int8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    int8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def deepseek_v4_int8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_int8_einsum",
    op_func=deepseek_v4_int8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_int8_einsum_fake,
)


def deepseek_v4_fp8_channel_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def deepseek_v4_fp8_channel_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_channel_einsum",
    op_func=deepseek_v4_fp8_channel_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_channel_einsum_fake,
)