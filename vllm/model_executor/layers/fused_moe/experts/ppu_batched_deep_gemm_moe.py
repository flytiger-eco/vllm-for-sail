# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.ppu_mxfp4_utils import downcast_to_mxfp4
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.ppu_deep_gemm import (
    DeepGemmQuantScaleFMT,
    fp8_m_grouped_gemm_nt_masked,
    int8_m_grouped_gemm_nt_masked,
    bf16_m_grouped_gemm_nt_masked,
    fp4_m_grouped_gemm_nt_masked,
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)
from vllm.utils.math_utils import cdiv, round_up

logger = init_logger(__name__)


def scales_shape_stride_dtype(
    E: int, T: int, G: int, quant_scale_fmt: DeepGemmQuantScaleFMT
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]:
    shape = (E, T, G)
    strides = (T * G, 1, T)
    if quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    ]:
        return shape, strides, torch.float32

    assert quant_scale_fmt == DeepGemmQuantScaleFMT.UE8M0
    shape = (E, T, cdiv(G, 4))
    strides = (T * cdiv(G, 4), 1, T)
    return shape, strides, torch.int32


@triton.jit
def _silu_mul_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    quant_min: tl.constexpr,
    quant_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        if SWIGLU_LIMIT > 0.0:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
            up = tl.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        gate = gate * (1.0 / (1.0 + tl.exp(-ALPHA * gate)))
        y = gate * (up + BETA)

        y_s = tl.maximum(tl.max(tl.abs(y)), eps) / quant_max

        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, quant_min, quant_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def _persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
    use_int8: bool = False,
    use_fp8: bool = False,
    swiglu_limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    We launch a fixed grid of threads to accommodate CUDA graphs. Let `P2`
    be a parallelization factor for persistent_masked_m_silu_mul_quant over the
    hidden dimension.

    Let `expert_offsets = [0] + [num_tokens.cumsum()]` and
    `total_tokens = expert_offsets[-1]`.
    persistent_masked_m_silu_mul_quant launches `total_tokens x P2` number of
    thread blocks. Each thread block contains `NUM_WARPS` warps.

    Every thread block needs to find it's corresponding expert by warp-parallel scanning
    over the `expert_offsets` array.

    The i-th warp in the first thread block processes
    `[i * warp_chunk_size, (i + 1) * warp_chunk_size]` groups
    sequentially, where `warp_chunk_size = ((H / GROUP_SIZE) / P2) / NUM_WARPS`,
    pipelining loads and computes.

    The shared memory layout for 4 warps with a 2-stage pipeline for SiLU V2
    can is visualized like so:

                         stage0                              stage1
    ┌─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┐
    │gate0│up0│gate1│up1│gate2│up2│gate3│up3│gate0│up0│gate1│up1│gate2│up2│gate3│up3│
    └─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┘

    with the main difference between V1 and V2 being the global load
    stride between warps, and between half-warps. Regarding the latter stride,
    we assign the first half warp of every warp for `gate` loads and the second
    half-warp to `up` loads.

    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s` depends on quant_scale_fmt,
      - quant_scale_fmt == FLOAT32,
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0,
         `y_s`: Int32 tensor, shape (E, T, H // group_size // 4), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0_FLOAT32_SPARSE
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    Let NUM_WARPS be the number of warps in a single thread block and
    `GROUP_SIZE = 128` be the size of the quantization group.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    assert use_fp8 or use_int8
    # allocate outputs
    if use_fp8:
        quant_dtype = torch.float8_e4m3fn
    elif use_int8:
        quant_dtype = torch.int8
    y_q = torch.empty((E, T, H), dtype=quant_dtype, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G, quant_scale_fmt)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    ceil_ue8m0 = quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

    cuda_arch = current_platform.get_device_capability(
        device_id=y.device.index
    ).to_int()

    swiglu_limit_val = swiglu_limit if swiglu_limit is not None else 0.0
    is_plain_silu = swiglu_limit_val == 0.0 and alpha == 1.0 and beta == 0.0

    if cuda_arch >= 80 and use_fp8 and group_size == 128 and is_plain_silu:
        # C++ fast path only supports plain silu(gate) * up.
        torch.ops._C.persistent_masked_m_silu_mul_quant(
            y, tokens_per_expert, y_q, y_s, ceil_ue8m0
        )
    else:
        stride_cnt_e = tokens_per_expert.stride()[0]

        # Static grid over experts and H-groups.
        # A loop inside the kernel handles the token dim
        grid = (E * G,)
        # strides (elements)
        stride_i_e, stride_i_t, stride_i_h = y.stride()
        stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

        if quant_dtype == torch.float8_e4m3fn:
            f_info = torch.finfo(quant_dtype)
        elif quant_dtype == torch.int8:
            f_info = torch.iinfo(quant_dtype)
        quant_max = f_info.max
        quant_min = f_info.min

        eps: float = 1e-10
        assert y_s.dtype == torch.float32, (
            f"_silu_mul_fp8_quant_deep_gemm does"
            f"not support {y_s.dtype} scales. Only torch.float32 supported."
        )
        _silu_mul_quant_deep_gemm[grid](
            y,
            y_q,
            y_s,
            tokens_per_expert,
            H,
            group_size,
            stride_i_e,
            stride_i_t,
            stride_i_h,
            stride_yq_e,
            stride_yq_t,
            stride_yq_h,
            ys_strides[0],
            ys_strides[1],
            ys_strides[2],
            stride_cnt_e,
            eps,
            quant_min,
            quant_max,
            ceil_ue8m0,
            swiglu_limit_val,
            alpha,
            beta,
            BLOCK=triton.next_power_of_2(group_size),
            NUM_STAGES=4,
            num_warps=1,
        )

    return y_q, y_s


def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H) float32
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
    swiglu_limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
):
    return _persistent_masked_m_silu_mul_quant(
        y,
        tokens_per_expert,
        num_parallel_tokens=16,
        group_size=group_size,
        quant_scale_fmt=quant_scale_fmt,
        use_int8=False,
        use_fp8=True,
        swiglu_limit=swiglu_limit,
        alpha=alpha,
        beta=beta,
    )


def int8_persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H) float32
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 1,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
    swiglu_limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
):
    return _persistent_masked_m_silu_mul_quant(
        y,
        tokens_per_expert,
        num_parallel_tokens=16,
        group_size=group_size,
        quant_scale_fmt=quant_scale_fmt,
        use_int8=True,
        use_fp8=False,
        swiglu_limit=swiglu_limit,
        alpha=alpha,
        beta=beta,
    )


@triton.jit
def _silu_mul_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # unquantized activations (E, T, H)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask_h = cols < BLOCK

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        base_i_offset = e * stride_i_e + t * stride_i_t + g * GROUP_SIZE * stride_i_h
        base_yq_offset = (
            e * stride_yq_e + t * stride_yq_t + g * GROUP_SIZE * stride_yq_h
        )

        mask = mask_h
        x = tl.load(
            input_ptr + base_i_offset + cols * stride_i_h, mask=mask, other=0.0
        ).to(tl.float32)
        y2 = tl.load(
            input_ptr + base_i_offset + H * stride_i_h + cols * stride_i_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        if SWIGLU_LIMIT > 0.0:
            x = tl.minimum(x, SWIGLU_LIMIT)
            y2 = tl.clamp(y2, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        x = x * (1.0 / (1.0 + tl.exp(-ALPHA * x)))
        y = x * (y2 + BETA)
        y_q = y

        tl.store(y_q_ptr + base_yq_offset + cols * stride_yq_h, y_q, mask=mask)


def silu_mul_deep_gemm(
    y: torch.Tensor,  # (E, T, 2*H) float32
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    group_size: int = 128,
    eps: float = 1e-10,
    swiglu_limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
):
    """Compute gate * sigmoid(alpha * gate) * (up + beta) with optional clamping.

    y has shape (E, T, 2*H). The first half of the last dimension is the gate,
    the second half is the up projection.

    Returns `y_q` where
    * `y_q` is the BF16 tensor of shape `(E, T, H)`, same layout as `y[..., :H]`.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = H // group_size
    assert H % group_size == 0, "H must be divisible by group_size"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E, (
        "tokens_per_expert must be shape (E,)"
    )
    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    # allocate outputs
    y_q = torch.empty((E, T, H), dtype=torch.bfloat16, device=y.device)

    # strides (elements)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    stride_cnt_e = tokens_per_expert.stride()[0]

    # static grid over experts and H-groups.
    # A loop inside the kernel handles the token dim
    grid = (E * G,)

    swiglu_limit_val = swiglu_limit if swiglu_limit is not None else 0.0

    _silu_mul_deep_gemm[grid](
        y,
        y_q,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        stride_cnt_e,
        eps,
        swiglu_limit_val,
        alpha,
        beta,
        BLOCK=group_size,
        NUM_STAGES=8,
        num_warps=1,
    )

    return y_q


_MXFP4_BLOCK_SIZE = 32  # Number of bf16 inputs per mxfp4 scale
_E2M1_MAX = 6.0  # Max representable value of fp4 e2m1


@triton.jit
def _silu_mul_mxfp4_quant_deep_gemm(
    # Pointers ---------------------------------------------------------------
    input_ptr,   # bf16 activations (E, T, 2*H)
    y_q_ptr,     # packed fp4 output (E, T, H//2), uint8
    y_s_ptr,     # mxfp4 scales (E, H//32//2, T), uint16 (packed E8M0 pairs, col-major)
    counts_ptr,  # int32 valid tokens per expert (E,)
    # Sizes ------------------------------------------------------------------
    H: tl.constexpr,   # hidden dimension (output half, i.e. H = N//2)
    # Strides for input (elements) -------------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) ---------------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements): layout [E, H//32//2, T] ------------------
    stride_ys_e,
    stride_ys_h,   # stride along H//32//2 dim (hidden dim, second dim)
    stride_ys_t,   # stride along T dim (fastest, third dim)
    # Meta -------------------------------------------------------------------
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
):
    """Fused SiLU+mul + MXFP4 quantisation for batched-expert (EP) layout.

    Grid: (hidden_dim_split, token_blocks_per_expert, expert_num)

    Scale output layout: [E, H//32//2, T] uint16 (col-major: H//32//2 is contiguous over T).
    Packing: uint16 = lo_e8m0 | (hi_e8m0 << 8), where lo/hi are adjacent groups.
    """
    GROUP_SIZE: tl.constexpr = 32  # mxfp4 block size
    QUANT_MAX: tl.constexpr = 6.0  # max abs value of e2m1
    NUM_GROUPS: tl.constexpr = BLOCK_N // GROUP_SIZE
    NUM_SCALE_PAIRS: tl.constexpr = NUM_GROUPS // 2

    BLOCK_NUM_PER_EXPERT = tl.num_programs(1)

    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    token_num_cur_expert = tl.load(counts_ptr + expert_id).to(tl.int64)

    stride_i_e = tl.cast(stride_i_e, tl.int64)
    stride_i_t = tl.cast(stride_i_t, tl.int64)
    stride_yq_e = tl.cast(stride_yq_e, tl.int64)
    stride_yq_t = tl.cast(stride_yq_t, tl.int64)
    stride_ys_e = tl.cast(stride_ys_e, tl.int64)
    stride_ys_h = tl.cast(stride_ys_h, tl.int64)

    # Input offsets for this hidden-dim block
    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_i_e + offs_in_d * stride_i_h

    # Output offsets: packed fp4 (2 e2m1 per uint8)
    offs_out_d = hidden_dim_block_index * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    y_q_ptr_offs = y_q_ptr + expert_id * stride_yq_e + offs_out_d * stride_yq_h

    # Scale offsets: one uint16 pair per 2*GROUP_SIZE elements
    # Layout [E, H//32//2, T]: stride_ys_h along H//32//2 dim, stride_ys_t along T dim
    offs_scale_pairs = hidden_dim_block_index * NUM_SCALE_PAIRS + tl.arange(0, NUM_SCALE_PAIRS)

    mask_d = offs_in_d < H

    for token_index in tl.range(
        token_id, token_num_cur_expert, BLOCK_NUM_PER_EXPERT, num_stages=NUM_STAGES
    ):
        # -- Load gate and up --
        gate = tl.load(
            input_ptr_offs + token_index * stride_i_t,
            mask=mask_d, other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_i_t + H * stride_i_h,
            mask=mask_d, other=0.0,
        ).to(tl.float32)

        # -- Clamp: gate at max only, up both ways (matches C++ reference) --
        if SWIGLU_LIMIT > 0.0:
            gate = tl.minimum(gate, SWIGLU_LIMIT)
            up = tl.clamp(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)

        # -- gate * sigmoid(alpha * gate) * (up + beta) --
        gate = gate / (1.0 + tl.exp(-ALPHA * gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = (up + BETA) * gate

        # -- MXFP4 e2m1 quantisation with E8M0 scale, RTNE rounding --
        gate_up_grouped = tl.reshape(gate_up, [NUM_GROUPS, GROUP_SIZE])

        # Per-group absmax
        _absmax = tl.max(tl.abs(gate_up_grouped), axis=1)
        _absmax = tl.maximum(_absmax, 1e-10)

        # E8M0 scale: round-up to power-of-2, per group
        dequant_scale = _absmax / QUANT_MAX
        ds_uint32 = dequant_scale.to(tl.uint32, bitcast=True)
        ds_e8m0_uint32 = (ds_uint32 + 0x007FFFFF) & 0x7F800000
        dequant_scale_rounded = ds_e8m0_uint32.to(tl.float32, bitcast=True)
        quant_scale = 1.0 / dequant_scale_rounded

        # Pack two adjacent uint8 E8M0 exponents into one uint16
        # scale_e8m0: [NUM_GROUPS] uint32 (exponent byte in bits [7:0] after >> 23)
        # Reshape to [NUM_SCALE_PAIRS, 2], split lo/hi, pack into uint16
        scale_pairs = tl.reshape(ds_e8m0_uint32, [NUM_SCALE_PAIRS, 2])
        lo_u32, hi_u32 = tl.split(scale_pairs)
        scale_u16 = ((lo_u32 >> 23) | ((hi_u32 >> 23) << 8)).to(tl.uint16)

        # Broadcast quant_scale and apply
        quant_scale_broadcast = tl.reshape(quant_scale, [NUM_GROUPS, 1])
        quant_vals_grouped = gate_up_grouped * quant_scale_broadcast
        quant_vals = tl.reshape(quant_vals_grouped, [BLOCK_N])

        # FP32 -> e2m1 via PTX hardware instruction
        # cvt.rn.satfinite.e2m1x2.f32: two f32 -> one packed e2m1x2 (uint8)
        pairs = tl.reshape(quant_vals, [BLOCK_N // 2, 2])
        lo_f, hi_f = tl.split(pairs)
        lo_f32 = lo_f.to(tl.float32)
        hi_f32 = hi_f.to(tl.float32)

        packed = tl.inline_asm_elementwise(
            """
            {
                .reg .b8 r;
                cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;
                mov.b32 $0, {r, r, r, r};
            }
            """,
            constraints="=r,f,f",
            args=[hi_f32, lo_f32],
            dtype=tl.uint8,
            is_pure=True,
            pack=1,
        )

        tl.store(
            y_q_ptr_offs + token_index * stride_yq_t,
            packed,
            mask=offs_out_d < (H // 2),
        )
        # Store uint16 scale pairs into [E, H//32//2, T] layout:
        # base = expert_id * stride_ys_e + offs_scale_pairs * stride_ys_h
        # token offset = token_index * stride_ys_t
        tl.store(
            y_s_ptr
            + expert_id * stride_ys_e
            + offs_scale_pairs * stride_ys_h
            + token_index * stride_ys_t,
            scale_u16,
            mask=offs_scale_pairs < (H // GROUP_SIZE // 2),
        )


def silu_mul_mxfp4_quant_masked(
    y: torch.Tensor,               # (E, T, 2*H) bf16
    tokens_per_expert: torch.Tensor,  # (E,) int32
    swiglu_limit: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU+mul + MXFP4 quantisation (masked / batched-expert layout).

    Args:
        y: Activations tensor of shape ``(E, T, 2*H)`` in bf16.  The first
           ``H`` elements of the last dimension are the gate projections and
           the last ``H`` elements are the up projections.
        tokens_per_expert: Per-expert valid-token counts, shape ``(E,)``.
        swiglu_limit: Optional clamp limit for the gate before silu (DeepSeek
           V4).  When ``None`` or ``0.0``, no clamping is applied.

    Returns:
        Tuple ``(y_q, y_s)`` where:
        * ``y_q``: Packed FP4 tensor, shape ``(E, T, H//2)``, dtype uint8.
          Two consecutive E2M1 values are packed into one uint8.
        * ``y_s``: E8M0 scale tensor, shape ``(E, H//32//2, T)``, dtype uint16.
          Two adjacent per-group uint8 E8M0 exponents are packed into one
          uint16 (lo | (hi << 8)), stored in col-major [E, H//32//2, T] layout.
          Call ``.permute(0, 2, 1)`` before passing to
          fp4_m_grouped_gemm_nt_masked which expects ``[E, T, H//32//2]``.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0
    H = H2 // 2
    assert H % (_MXFP4_BLOCK_SIZE * 2) == 0, f"H must be divisible by {_MXFP4_BLOCK_SIZE * 2} for uint16 packing, got {H}"

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    y_q = torch.empty((E, T, H // 2), dtype=torch.uint8, device=y.device)
    # Scale output: [E, H//32//2, T] uint16 (col-major: H//32//2 dim is contiguous over T)
    y_s = torch.empty((E, H // _MXFP4_BLOCK_SIZE // 2, T), dtype=torch.uint16, device=y.device)

    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()
    stride_ys_e, stride_ys_h, stride_ys_t = y_s.stride()

    # Dynamic BLOCK_N: larger blocks for better memory throughput
    if H > 256:
        BLOCK_N = 256
    elif H > 128:
        BLOCK_N = 128
    else:
        BLOCK_N = 64

    hidden_dim_split_block_num = triton.cdiv(H, BLOCK_N)

    # Token parallelism: more blocks per expert for small expert counts
    if E < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    grid = (hidden_dim_split_block_num, BLOCK_NUM_PER_EXPERT, E)

    swiglu_limit_val = swiglu_limit if swiglu_limit is not None else 0.0

    _silu_mul_mxfp4_quant_deep_gemm[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        stride_ys_e,
        stride_ys_h,
        stride_ys_t,
        BLOCK_N=BLOCK_N,
        NUM_STAGES=3,
        SWIGLU_LIMIT=swiglu_limit_val,
        ALPHA=alpha,
        BETA=beta,
        num_warps=1,
    )

    return y_q, y_s


class PPUBatchedDeepGemmExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        """
        max_num_tokens: Maximum number of tokens from a DP Rank
        num_dispatchers: The number of DP dispatchers.
        quant_config: Quantization configuration
        """
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.block_wise = quant_config.block_shape is not None
        if self.block_wise:
            assert (
                self.block_shape[1] == get_mk_alignment_for_contiguous_layout(is_blockwise=self.block_wise)[1]
            )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        # Gated-activation params: silu == swigluoai with alpha=1, beta=0.
        # FP8 (silu) configs leave these None, reproducing plain silu.
        self.gemm1_alpha = (
            quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def supports_packed_ue8m0_act_scales(self) -> bool:
        """
        DeepGemm supports packed ue8m0 activation scales format in devices == sm100
        """
        return (
            is_deep_gemm_e8m0_used()
            and current_platform.is_device_capability_family(100)
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # FIXME (varun): We should be able to dispatch only from the leader
        # DP ranks in the case of TP > 1. At the moment, all the Ranks
        # end up sending their tokens. This needs to be fixed.
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = M if self.max_num_tokens is None else self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dispatchers, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

    def estimate_expected_m(
        self, global_num_experts: int, max_tokens_per_expert: int, topk: int
    ) -> int:
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            logger.warning_once(
                "DPMetadata unavailable. Defaulting expected_m to "
                f"{max_tokens_per_expert}.",
                scope="local",
            )
            return max_tokens_per_expert

        total_num_tokens = dp_meta.num_tokens_across_dp_cpu.sum().item()
        total_num_tokens_replicated = total_num_tokens * topk

        # Assume even load balancing
        assert global_num_experts != 0
        estimate = round_up(int(total_num_tokens_replicated // global_num_experts), 16)
        # clamp estimate
        estimate = max(estimate, 16)
        estimate = min(max_tokens_per_expert, estimate)
        return estimate

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3

        a1q = hidden_states

        assert w2.size(1) == w1.size(2)

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        expected_m = self.estimate_expected_m(
            global_num_experts=global_num_experts,
            max_tokens_per_expert=max_num_tokens,
            topk=topk_ids.size(-1),
        )

        if self.quant_config.use_fp8_w8a8:
            # fp8
            fp8_m_grouped_gemm_nt_masked(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                workspace1,
                expert_num_tokens,
                expected_m,
            )

            quant_scale_fmt = DeepGemmQuantScaleFMT.from_oracle()
            group_size = (
                self.block_shape[1]
                if self.block_shape
                else workspace1.shape[-1] // 2
            )
            a2q, a2q_scale = persistent_masked_m_silu_mul_quant(
                workspace1,
                expert_num_tokens,
                group_size=group_size,
                quant_scale_fmt=quant_scale_fmt,
                swiglu_limit=self.gemm1_clamp_limit,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )

            fp8_m_grouped_gemm_nt_masked(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                output,
                expert_num_tokens,
                expected_m,
            )
        elif self.quant_config.quant_dtype == torch.int8:
            # int8
            int8_m_grouped_gemm_nt_masked(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                workspace1,
                expert_num_tokens,
                expected_m,
            )

            a2q, a2q_scale = int8_persistent_masked_m_silu_mul_quant(
                workspace1,
                expert_num_tokens,
                group_size=workspace1.shape[-1] // 2,
                swiglu_limit=self.gemm1_clamp_limit,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )

            int8_m_grouped_gemm_nt_masked(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                output,
                expert_num_tokens,
                expected_m,
            )
        else:
            # bf16
            bf16_m_grouped_gemm_nt_masked(
                a1q, w1, workspace1, expert_num_tokens, expected_m
            )

            a2q = silu_mul_deep_gemm(
                workspace1,
                expert_num_tokens,
                swiglu_limit=self.gemm1_clamp_limit,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )

            bf16_m_grouped_gemm_nt_masked(
                a2q, w2, output, expert_num_tokens, expected_m
            )


class PPUBatchedDeepGemmExpertsMXFP4(mk.FusedMoEExpertsModular):
    """
    PPU DeepGEMM-based batched fused MoE expert implementation for MXFP4.

    Operates in BatchedExperts activation format (E, T, N/K) used for EP
    (Expert Parallelism) via DeepEP. Both weights and activations are MXFP4
    (w4a4), using deepgemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        # Gated-activation params: silu == swigluoai with alpha=1, beta=0.
        # FP8 (silu) configs leave these None, reproducing plain silu.
        self.gemm1_alpha = (
            quant_config.gemm1_alpha if quant_config.gemm1_alpha is not None else 1.0
        )
        self.gemm1_beta = (
            quant_config.gemm1_beta if quant_config.gemm1_beta is not None else 0.0
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def _supports_current_device() -> bool:
        # MXFP4 (w4a4) requires sm90+ on PPU; disable on sm80
        if current_platform.is_device_capability((8, 0)):
            return False
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def supports_packed_ue8m0_act_scales(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dispatchers = self.num_dispatchers
        num_experts = local_num_experts
        max_num_tokens = M if self.max_num_tokens is None else self.max_num_tokens
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (num_experts, max_num_tokens * num_dispatchers, max(K, N))
        workspace2 = (num_experts, max_num_tokens * num_dispatchers, activation_out_dim)
        output = (num_experts, max_num_tokens * num_dispatchers, K)
        return (workspace13, workspace2, output)

    def estimate_expected_m(
        self, global_num_experts: int, max_tokens_per_expert: int, topk: int
    ) -> int:
        dp_meta = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_meta is None:
            logger.warning_once(
                "DPMetadata unavailable. Defaulting expected_m to "
                f"{max_tokens_per_expert}.",
                scope="local",
            )
            return max_tokens_per_expert

        total_num_tokens = dp_meta.num_tokens_across_dp_cpu.sum().item()
        total_num_tokens_replicated = total_num_tokens * topk

        assert global_num_experts != 0
        estimate = round_up(int(total_num_tokens_replicated // global_num_experts), 16)
        estimate = max(estimate, 16)
        estimate = min(max_tokens_per_expert, estimate)
        return estimate

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert expert_tokens_meta is not None
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        assert hidden_states.ndim == 3
        assert self.quant_config.quant_dtype == "mxfp4"

        a1q = hidden_states

        # For mxfp4: w1.size(2) is hidden_size // 2 (two fp4 packed per uint8),
        # and w2.size(1) is hidden_size (the full uncompressed row count).
        assert w2.size(1) == w1.size(2) * 2

        E, max_num_tokens, N, K, _ = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        expected_m = self.estimate_expected_m(
            global_num_experts=global_num_experts,
            max_tokens_per_expert=max_num_tokens,
            topk=topk_ids.size(-1),
        )

        # workspace1: (E, T, N) — gemm1 bf16 output
        workspace1 = _resize_cache(workspace13, (E, max_num_tokens, N))

        fp4_m_grouped_gemm_nt_masked(
            (a1q, a1q_scale),
            (w1, self.w1_scale),
            self.quant_config._w1.bias,
            workspace1,
            expert_num_tokens,
            expected_m,
        )

        activation_out_dim = self.adjust_N_for_activation(N, activation)

        # SiLU+mul on (E, T, 2*H) -> (E, T, H//2) uint8 + scales, fused with
        # mxfp4 quantisation.  The fused Triton kernel supports SILU and
        # SWIGLUOAI_UNINTERLEAVE (with clamp/alpha/beta); other activation
        # variants fall back to a two-step approach.
        if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI_UNINTERLEAVE):
            a2q, a2q_scale = silu_mul_mxfp4_quant_masked(
                workspace1, expert_num_tokens,
                swiglu_limit=self.gemm1_clamp_limit,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )
            # silu_mul_mxfp4_quant_masked outputs scale in [E, H//32//2, T] layout.
            # fp4_m_grouped_gemm_nt_masked expects [E, T, H//32//2], so permute.
            a2q_scale = a2q_scale.permute(0, 2, 1)
        else:
            # Fallback for SWIGLUSTEP / SWIGLUOAI: reshape (E,T,2H) -> (E*T, 2H),
            # apply 2D-only activation, reshape back, then quantise separately.
            act_out_2d = torch.empty(
                (E * max_num_tokens, activation_out_dim),
                dtype=workspace1.dtype,
                device=workspace1.device,
            )

            self.activation(
                activation,
                act_out_2d,
                workspace1.view(E * max_num_tokens, N),
                clamp_limit=self.gemm1_clamp_limit,
                alpha=self.gemm1_alpha,
                beta=self.gemm1_beta,
            )
            act_out = act_out_2d.view(E, max_num_tokens, activation_out_dim)
            a2q, a2q_scale = downcast_to_mxfp4(act_out, axis=-1)

        # for mxfp4, output hidden_size = K * 2 (unpacked)
        fp4_m_grouped_gemm_nt_masked(
            (a2q, a2q_scale),
            (w2, self.w2_scale),
            self.quant_config._w2.bias,
            output,
            expert_num_tokens,
            expected_m,
        )
