# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused inverse RoPE + block-scaled FP8 quantization kernel for DeepseekV4 attention.

Output scale format is pre-transformed (MN-major TMA-aligned; FP32 on SM90,
INT32-packed UE8M0 on SM100) so fp8_einsum skips transform_sf_into_required_layout.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fused_inv_rope_int8_quant_channelwise_per_group(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    int8_ptr,
    scale_ptr,
    num_tokens,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    int8_stride_token,
    int8_stride_group,
    scale_stride_token,
    scale_stride_group,
    heads_per_group: tl.constexpr,
    int8_max: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Fused inverse RoPE + per-(token, group) symmetric INT8 quantisation.

    One program per (token, group) pair. Each program loads the full
    ``[heads_per_group, HEAD_DIM]`` tile, applies inverse RoPE to the
    trailing rope_dim slice of every head, computes a single absmax over
    the whole tile (one scale per (token, group)), then stores the
    symmetric INT8 quantised values into ``[T, G, heads_per_group * HEAD_DIM]``
    and the fp32 scale into ``[T, G, 1]`` so it broadcasts against the
    activation tensor of einsum equation ``"bhr,hdr->bhd"`` along the
    reduction axis.
    """
    pid_token = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1).to(tl.int64)

    if pid_token >= num_tokens:
        return

    head_offsets = tl.arange(0, heads_per_group)
    dim_offsets = tl.arange(0, HEAD_DIM)

    base = (
        o_ptr
        + pid_token * o_stride_token
        + (pid_g * heads_per_group) * o_stride_head
    )
    addrs = base + head_offsets[:, None] * o_stride_head + dim_offsets[None, :]
    x = tl.load(addrs).to(tl.float32)

    # -- inverse RoPE on trailing rope_dim per head -------------------------
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = dim_offsets >= ROPE_START
    rope_local = dim_offsets - ROPE_START
    partner_addrs = (
        base
        + head_offsets[:, None] * o_stride_head
        + (dim_offsets ^ 1)[None, :]
    )
    x_partner = tl.load(partner_addrs, mask=is_rope[None, :], other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v[None, :] + x_partner * sin_v[None, :]
    x_sub = x * cos_v[None, :] - x_partner * sin_v[None, :]
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even[None, :], x_add, x_sub)
    x = tl.where(is_rope[None, :], rotated, x)

    # -- per-(token, group) absmax → channelwise scale ----------------------
    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / int8_max

    # -- symmetric INT8 quantise --------------------------------------------
    x_q = tl.clamp(x / scale, -int8_max, int8_max)
    x_int8 = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    # -- store INT8 to [T, G, heads_per_group * HEAD_DIM] -------------------
    out_base = (
        int8_ptr
        + pid_token * int8_stride_token
        + pid_g * int8_stride_group
    )
    full_offsets = head_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
    tl.store(out_base + full_offsets, x_int8)

    # -- store scalar scale to [T, G, 1] ------------------------------------
    scale_addr = (
        scale_ptr
        + pid_token * scale_stride_token
        + pid_g * scale_stride_group
    )
    tl.store(scale_addr, scale)


def fused_inv_rope_int8_quant_channelwise(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + channelwise INT8 quant matching ``"bhr,hdr->bhd"``.

    Designed to feed PPU DeepGEMM ``int8_einsum``. The activation tensor is
    laid out as ``[T, G, R]`` (b=T, h=G, r=R = heads_per_group * head_dim),
    so the corresponding per-(token, group) scale has shape ``[T, G, 1]``
    — broadcasting against the reduction axis ``r`` as ``int8_einsum``
    expects (`(b, h, 1)` for `(b, h, r)`).

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16/fp16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] cos||sin (fp32).
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).

    Returns:
        o_int8: [T, G, heads_per_group * head_dim] int8.
        o_scale: [T, G, 1] float32 (one scale per (token, group)).
    """
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    int8_max_val = 127
    return torch.ops.vllm.fused_inv_rope_int8_quant_channelwise_kernel(
        o,
        positions,
        cos_sin_cache,
        heads_per_group,
        nope_dim,
        rope_dim // 2,
        int8_max_val,
        num_tokens,
        n_groups,
        d,
        head_dim,
    )


def _fused_inv_rope_int8_quant_channelwise_kernel_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    int8_max_val: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    int8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.int8,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    grid = (num_tokens, n_groups)
    _fused_inv_rope_int8_quant_channelwise_per_group[grid](
        o,
        positions,
        cos_sin_cache,
        int8_buf,
        scale_buf,
        num_tokens,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        int8_stride_token=int8_buf.stride(0),
        int8_stride_group=int8_buf.stride(1),
        scale_stride_token=scale_buf.stride(0),
        scale_stride_group=scale_buf.stride(1),
        heads_per_group=heads_per_group,
        int8_max=int8_max_val,
        eps=1e-10,
        HEAD_DIM=head_dim,
        ROPE_START=rope_start,
        HALF_ROPE=half_rope,
        num_warps=1,
        num_stages=1,
    )
    return int8_buf, scale_buf


def _fused_inv_rope_int8_quant_channelwise_kernel_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    int8_max_val: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    int8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.int8,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    return int8_buf, scale_buf


direct_register_custom_op(
    op_name="fused_inv_rope_int8_quant_channelwise_kernel",
    op_func=_fused_inv_rope_int8_quant_channelwise_kernel_impl,
    fake_impl=_fused_inv_rope_int8_quant_channelwise_kernel_fake,
)


@triton.jit
def _fused_inv_rope_fp8_quant_channelwise_per_group(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_token,
    fp8_stride_group,
    scale_stride_token,
    scale_stride_group,
    heads_per_group: tl.constexpr,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Fused inverse RoPE + per-(token, group) symmetric FP8 quantisation.

    One program per (token, group) pair. Identical structure to the INT8
    channelwise kernel but casts to ``float8_e4m3fn`` and uses ``fp8_max``
    (448.0) as the symmetric range. The output scale is raw fp32 — no
    TMA / UE8M0 pre-transform — matching how ``int8_einsum`` receives raw
    per-channel scales.
    """
    pid_token = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1).to(tl.int64)

    if pid_token >= num_tokens:
        return

    head_offsets = tl.arange(0, heads_per_group)
    dim_offsets = tl.arange(0, HEAD_DIM)

    base = (
        o_ptr
        + pid_token * o_stride_token
        + (pid_g * heads_per_group) * o_stride_head
    )
    addrs = base + head_offsets[:, None] * o_stride_head + dim_offsets[None, :]
    x = tl.load(addrs).to(tl.float32)

    # -- inverse RoPE on trailing rope_dim per head -------------------------
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = dim_offsets >= ROPE_START
    rope_local = dim_offsets - ROPE_START
    partner_addrs = (
        base
        + head_offsets[:, None] * o_stride_head
        + (dim_offsets ^ 1)[None, :]
    )
    x_partner = tl.load(partner_addrs, mask=is_rope[None, :], other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v[None, :] + x_partner * sin_v[None, :]
    x_sub = x * cos_v[None, :] - x_partner * sin_v[None, :]
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even[None, :], x_add, x_sub)
    x = tl.where(is_rope[None, :], rotated, x)

    # -- per-(token, group) absmax -> channelwise scale ----------------------
    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / fp8_max

    # -- symmetric FP8 quantise ---------------------------------------------
    x_fp8 = (x / scale).to(tl.float8e4nv)

    # -- store FP8 to [T, G, heads_per_group * HEAD_DIM] --------------------
    out_base = (
        fp8_ptr
        + pid_token * fp8_stride_token
        + pid_g * fp8_stride_group
    )
    full_offsets = head_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
    tl.store(out_base + full_offsets, x_fp8)

    # -- store scalar scale to [T, G, 1] ------------------------------------
    scale_addr = (
        scale_ptr
        + pid_token * scale_stride_token
        + pid_g * scale_stride_group
    )
    tl.store(scale_addr, scale)


def fused_inv_rope_fp8_quant_channelwise(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + channelwise FP8 quant matching ``"bhr,hdr->bhd"``.

    Designed to feed PPU DeepGEMM ``fp8_einsum``.  Same output layout as the
    INT8 channelwise variant: ``o_fp8 [T, G, R]`` + ``o_scale [T, G, 1]``.
    The scale is raw fp32 (no TMA / UE8M0 pre-transform), mirroring how
    ``int8_einsum`` receives raw per-channel scales.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16/fp16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] cos||sin (fp32).
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).

    Returns:
        o_fp8: [T, G, heads_per_group * head_dim] float8_e4m3fn.
        o_scale: [T, G, 1] float32 (one scale per (token, group)).
    """
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return torch.ops.vllm.fused_inv_rope_fp8_quant_channelwise_kernel(
        o,
        positions,
        cos_sin_cache,
        heads_per_group,
        nope_dim,
        rope_dim // 2,
        fp8_max,
        num_tokens,
        n_groups,
        d,
        head_dim,
    )


def _fused_inv_rope_fp8_quant_channelwise_kernel_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    fp8_max: float,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    grid = (num_tokens, n_groups)
    _fused_inv_rope_fp8_quant_channelwise_per_group[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_token=fp8_buf.stride(0),
        fp8_stride_group=fp8_buf.stride(1),
        scale_stride_token=scale_buf.stride(0),
        scale_stride_group=scale_buf.stride(1),
        heads_per_group=heads_per_group,
        fp8_max=fp8_max,
        eps=1e-10,
        HEAD_DIM=head_dim,
        ROPE_START=rope_start,
        HALF_ROPE=half_rope,
        num_warps=1,
        num_stages=1,
    )
    return fp8_buf, scale_buf


def _fused_inv_rope_fp8_quant_channelwise_kernel_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    fp8_max: float,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    return fp8_buf, scale_buf


direct_register_custom_op(
    op_name="fused_inv_rope_fp8_quant_channelwise_kernel",
    op_func=_fused_inv_rope_fp8_quant_channelwise_kernel_impl,
    fake_impl=_fused_inv_rope_fp8_quant_channelwise_kernel_fake,
)


@triton.jit(do_not_specialize=["num_tokens"])
def _fused_inv_rope_fp8_quant_per_head(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_group,
    fp8_stride_token,
    scale_stride_group,
    scale_stride_k,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    CHUNKS_PER_HEAD: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TMA_ALIGNED_SCALES: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,  # triton metadata
):
    # Cast every stride to int64 — without this, Python-int strides are
    # inferred as int32 and `pid_token(int64) × stride(int32)` can lower to
    # int32 arithmetic, wrapping past 2³¹ for large prefill batches → IMA.
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)
    o_stride_token = o_stride_token.to(tl.int64)
    o_stride_head = o_stride_head.to(tl.int64)
    cache_stride_pos = cache_stride_pos.to(tl.int64)
    fp8_stride_group = fp8_stride_group.to(tl.int64)
    fp8_stride_token = fp8_stride_token.to(tl.int64)
    scale_stride_group = scale_stride_group.to(tl.int64)
    scale_stride_k = scale_stride_k.to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh
    qb_start = head_in_group * CHUNKS_PER_HEAD
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    # Padding rows in the TMA-aligned scale buffer: fill with zero and skip quant.
    if pid_token >= num_tokens:
        if TMA_ALIGNED_SCALES:
            scale_addr = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + head_in_group * scale_stride_k
            )
            tl.store(scale_addr, tl.zeros((), dtype=tl.int32))
        else:
            block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
            qb_indices = qb_start + block_offsets
            scale_addrs = (
                scale_ptr
                + g * scale_stride_group
                + pid_token
                + qb_indices * scale_stride_k
            )
            tl.store(scale_addrs, tl.zeros((CHUNKS_PER_HEAD,), dtype=tl.float32))
        return

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(input_base + offsets).to(tl.float32)

    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(
        tl.float32
    )
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    x = tl.where(is_rope, rotated, x)

    x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
    block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
    scale_raw = block_absmax * (1.0 / fp8_max)
    scales = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    scales_exp = tl.reshape(
        tl.broadcast_to(
            tl.reshape(scales, (CHUNKS_PER_HEAD, 1)),
            (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE),
        ),
        (HEAD_DIM,),
    )
    x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)

    fp8_base = (
        fp8_ptr
        + g * fp8_stride_group
        + pid_token * fp8_stride_token
        + qb_start * QUANT_GROUP_SIZE
    )
    tl.store(fp8_base + offsets, x_quant)

    block_offsets = tl.arange(0, CHUNKS_PER_HEAD)
    qb_indices = qb_start + block_offsets
    if TMA_ALIGNED_SCALES:
        scale_bits = scales.to(tl.int32, bitcast=True)
        ue8m0_bytes = (scale_bits >> 23) & 0xFF
        packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))
        scale_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token
            + head_in_group * scale_stride_k
        )
        tl.store(scale_addr, packed_val)
    else:
        scale_addrs = (
            scale_ptr + g * scale_stride_group + pid_token + qb_indices * scale_stride_k
        )
        tl.store(scale_addrs, scales)


def fused_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
    tma_aligned_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + block-scaled FP8 quantization.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] with cos||sin.
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).
        quant_group_size: FP8 quantization block size (default 128).
        tma_aligned_scales: Output INT32 packed UE8M0 for SM100 (True)
                            or FP32 for SM90 (False).

    Returns:
        o_fp8: [T, G, D] float8_e4m3fn, strides (D, T*D, 1).
        o_scale: Pre-transformed scale tensor for fp8_einsum.
    """
    from vllm.utils.deep_gemm import get_tma_aligned_size

    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    num_scale_blocks = d // quant_group_size
    chunks_per_head = head_dim // quant_group_size

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    tma_aligned_T = get_tma_aligned_size(num_tokens, 4)
    if tma_aligned_scales:
        packed_sf_k = (num_scale_blocks + 3) // 4
        scale_inner = packed_sf_k
    else:
        scale_inner = num_scale_blocks

    # Run kernel through a custom op so inductor sees an opaque boundary.
    # It's a pytorch bug, see https://github.com/vllm-project/vllm/issues/41106
    fp8_buf, scale_buf = torch.ops.vllm.fused_inv_rope_fp8_quant_kernel(
        o,
        positions,
        cos_sin_cache,
        heads_per_group,
        quant_group_size,
        chunks_per_head,
        nope_dim % quant_group_size,
        rope_dim // 2,
        tma_aligned_scales,
        fp8_max,
        tma_aligned_T,
        num_tokens,
        n_groups,
        d,
        scale_inner,
    )
    return fp8_buf.transpose(0, 1).contiguous(), scale_buf.transpose(0, 1).contiguous()


def _fused_inv_rope_fp8_quant_kernel_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    rope_start: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_T,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_T, 1, tma_aligned_T),
    )
    grid = (tma_aligned_T, n_groups * heads_per_group)
    use_gdc = current_platform.is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": use_gdc}
    _fused_inv_rope_fp8_quant_per_head[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_buf.stride(0),
        fp8_stride_token=fp8_buf.stride(1),
        scale_stride_group=scale_buf.stride(0),
        scale_stride_k=scale_buf.stride(2),
        fp8_max=fp8_max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=rope_start,
        HALF_ROPE=half_rope,
        TMA_ALIGNED_SCALES=tma_aligned_scales,
        USE_GDC=use_gdc,
        num_stages=1,
        **pdl_kwargs,
        num_warps=1,
    )
    return fp8_buf, scale_buf


def _fused_inv_rope_fp8_quant_kernel_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    rope_start: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_dtype = torch.int32 if tma_aligned_scales else torch.float32
    scale_buf = torch.empty(
        n_groups * scale_inner * tma_aligned_T,
        dtype=scale_dtype,
        device=o.device,
    ).as_strided(
        (n_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_T, 1, tma_aligned_T),
    )
    return fp8_buf, scale_buf


direct_register_custom_op(
    op_name="fused_inv_rope_fp8_quant_kernel",
    op_func=_fused_inv_rope_fp8_quant_kernel_impl,
    fake_impl=_fused_inv_rope_fp8_quant_kernel_fake,
)
