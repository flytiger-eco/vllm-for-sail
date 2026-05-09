from enum import Enum

import torch

from vllm.logger import init_logger
from vllm.triton_utils import triton
import triton.language as tl

logger = init_logger(__name__)


# for fp4 quantization of hidden_states on PPU
# copy from https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/numerics_details/mxfp.py
class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1


class ScaleFormat(Enum):
    """Output format of the E8M0 activation scale from downcast_to_mxfp4.

    UINT8_ROW_MAJOR:
        dtype  : torch.uint8
        shape  : (..., S_groups)  where S_groups = K // 32
        layout : row-major contiguous
        usage  : legacy path; caller must call preprocess_mxfp4_scales() before
                 passing to DeepGEMM.

    UINT16_COL_MAJOR:
        dtype  : torch.uint16
        shape  : (..., S_groups // 2)  (two adjacent uint8 E8M0 scales packed)
        layout : col-major — physical allocation is [S_groups//2, N] contiguous,
                 logical view is [N, S_groups//2] via .t().
                 stride(quantization-axis) == N  (non-unit)
        packing: uint16 = lo_e8m0 | (hi_e8m0 << 8)
        usage  : optimised path; ready for direct use with DeepGEMM fp4 grouped
                 GEMM kernels (no preprocess_mxfp4_scales() call needed).
                 Requires K to be a multiple of 64 (S_groups even).
    """
    UINT8_ROW_MAJOR = 0
    UINT16_COL_MAJOR = 1


MXFP_BLOCK_SIZE: tl.constexpr = tl.constexpr(32)


# ---------------------------------------------------------------------------
# Kernel: uint8 row-major scale output (legacy path)
# ---------------------------------------------------------------------------

@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")


@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr,
                             DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0):
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // MXFP_BLOCK_SIZE

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    else:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = (quant_tensor & 0x7FFFFF)

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas = tl.where(exponents < E8_BIAS, (0x400000 | (mantissas >> 1)) >> adjusted_exponents, mantissas)

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def _downcast_to_mxfp(mx_tensor_ptr, stride_mxt_outer, stride_mxt_quant: tl.constexpr,
                      mx_scale_ptr, stride_mx_scale_outer, stride_mx_scale_quant,
                      src_ptr, stride_src_outer, stride_src_quant,
                      outer_dim, quant_dim,
                      BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr,
                      DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr):
    """Legacy kernel: uint8 row-major scale output. Supports fp4 and fp8."""

    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
                     f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.")

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert((src_dtype == tl.bfloat16) or (src_dtype == tl.float16) or (src_dtype == tl.float32), f"{src_dtype=} must be bfloat16 or float16 or float32")
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(src_tensor, full_mask_src, mx_tensor_dtype,
                                                        DEQUANT_SCALE_ROUNDING_MODE)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


# ---------------------------------------------------------------------------
# Kernel: uint16 col-major scale output (optimised path)
# ---------------------------------------------------------------------------

@triton.jit
def _downcast_to_mxfp4_u16(
    # Output: packed e2m1 tensor (uint8, two fp4 per byte)
    mx_tensor_ptr, stride_mxt_outer, stride_mxt_quant: tl.constexpr,
    # Output: preprocessed E8M0 scales (uint16, packed col-major layout [S_pairs, N])
    mx_scale_ptr, stride_mx_scale_pair, stride_mx_scale_outer,
    # Input: source tensor (bf16/fp16/fp32)
    src_ptr, stride_src_outer, stride_src_quant,
    outer_dim, quant_dim, orig_quant_dim,
    BLOCK_SIZE_OUT_DIM: tl.constexpr,
    BLOCK_SIZE_QUANT_DIM: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr,
):
    """Fused MXFP4 downcast kernel: E8M0 scale + e2m1 quantization in one pass.

    Outputs:
    - Packed e2m1 values (uint8, 2 values per byte).
    - E8M0 scales packed as uint16 in col-major physical layout [S_pairs, N].
      Logical shape after .t() is [N, S_pairs].
      Packing: uint16 = lo_e8m0 | (hi_e8m0 << 8)

    Requires SM 8.9+ for PTX cvt.rn.satfinite.e2m1x2.f32 instruction.
    """
    tl.static_assert(stride_mxt_quant == 1, f"Output stride {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0,
                     f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")
    tl.static_assert((BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE) % 2 == 0,
                     "Number of scale groups per block must be even for uint16 packing")

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_tensor_ptr.dtype.element_ty == tl.uint8,
                     "Output tensor must be uint8 (packed e2m1).")
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint16,
                     "Output scale must be uint16 (packed E8M0).")
    tl.static_assert(
        (src_dtype == tl.bfloat16) or (src_dtype == tl.float16) or (src_dtype == tl.float32),
        f"{src_dtype=} must be bfloat16, float16, or float32",
    )

    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_SCALE_PAIRS: tl.constexpr = BLOCK_SIZE_QUANT_MX_SCALE // 2
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // 2  # 2 e2m1 per uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_pair = quant_block * BLOCK_SIZE_QUANT_MX_SCALE_PAIRS
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_pair * stride_mx_scale_pair + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_pairs = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE_PAIRS)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < orig_quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < orig_quant_dim // 2
    full_mask_mxt = mask_mxt_quant & mask_n

    S_GROUPS_PAIRS = quant_dim // MXFP_BLOCK_SIZE // 2
    scale_mask_k = start_mx_scale_pair + offs_scale_pairs < S_GROUPS_PAIRS
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_pairs * stride_mx_scale_pair + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer

    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src, other=0.0)

    # ======== E8M0 scale computation ========
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(full_mask_src, abs_tensor, -1.0)
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    # Clamp to avoid log2(0) in scale computation
    max_val = tl.maximum(max_val, 1e-10)

    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # ROUND_UP: 2 ** ceil(log2(max / 6.0))
        dequant_scale = max_val / 6.0
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    else:
        # ROUND_DOWN: 2 ** floor(log2(max / 6.0))
        dequant_scale = max_val / 6.0
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000

    # dequant_scale_rounded is non-zero (max_val >= 1e-10)
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = 1.0 / dequant_scale_rounded

    # Apply per-group scale
    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * quant_scale
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])

    # Pack scale exponents into uint16 pairs.
    # dequant_scale_exponent: [OUT, SCALE, 1] uint32 with exponent in bits [30:23]
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])
    # Zero out exponents for padding groups (beyond original quant_dim)
    ORIG_S_GROUPS = orig_quant_dim // MXFP_BLOCK_SIZE
    scale_group_idx = quant_block * BLOCK_SIZE_QUANT_MX_SCALE + tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :]
    dequant_scale_exponent = tl.where(scale_group_idx < ORIG_S_GROUPS, dequant_scale_exponent, 0)
    # Reshape to pairs: [OUT, S//2, 2], extract exponent bytes, pack into uint16
    de_pairs = tl.reshape(dequant_scale_exponent, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE_PAIRS, 2])
    lo_u32, hi_u32 = tl.split(de_pairs)
    scale_tensor = ((lo_u32 >> 23) | ((hi_u32 >> 23) << 8)).to(tl.uint16)

    # ======== FP32 -> e2m1 conversion via PTX hardware instruction ========
    # cvt.rn.satfinite.e2m1x2.f32: two f32 values -> one packed e2m1x2 (uint8)
    # Requires SM 8.9+
    pairs = tl.reshape(quant_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
    lo_f, hi_f = tl.split(pairs)
    lo_f32 = lo_f.to(tl.float32)
    hi_f32 = hi_f.to(tl.float32)

    out_tensor = tl.inline_asm_elementwise(
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

    # Store scale (uint16 packed, col-major [S_pairs, N] layout) and quantized tensor
    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def downcast_to_mxfp4(
    src_tensor: torch.Tensor,
    axis: int,
    DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP,
    scale_format: ScaleFormat = ScaleFormat.UINT16_COL_MAJOR,
):
    """Convert src tensor to MXFP4 (packed e2m1 uint8) with E8M0 scales.

    Args:
        src_tensor: Input tensor (bf16/fp16/fp32).
        axis:       Quantization axis.
        DEQUANT_SCALE_ROUNDING_MODE: ROUND_UP (default) or ROUND_DOWN.
        scale_format: Controls the dtype and memory layout of the returned scale.

            ScaleFormat.UINT16_COL_MAJOR (default):
                dtype  : torch.uint16
                shape  : src_tensor.shape with quant axis replaced by S_groups // 2
                layout : col-major — physical [S_groups//2, N] contiguous,
                         logical view [N, S_groups//2] via non-unit stride.
                packing: uint16 = lo_e8m0 | (hi_e8m0 << 8)
                note   : K must be a multiple of 64. Ready for direct use with
                         DeepGEMM fp4 GEMM (no preprocess_mxfp4_scales() needed).

            ScaleFormat.UINT8_ROW_MAJOR:
                dtype  : torch.uint8
                shape  : src_tensor.shape with quant axis replaced by S_groups
                layout : row-major contiguous
                note   : Caller must call preprocess_mxfp4_scales() before
                         passing to DeepGEMM.

    Returns:
        (out_quant_tensor, out_scale):
          - out_quant_tensor: uint8, two e2m1 values packed per byte.
            Shape is same as src_tensor except the quantization axis is halved.
          - out_scale: see scale_format above.
    """
    out_quant_type = torch.uint8
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim

    # Move quantization axis to last dim
    src_tensor = src_tensor.transpose(axis, ndim - 1)
    L = src_tensor.shape[-1]
    assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"

    out_shape = src_tensor.shape[:-1] + (L // 2,)
    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)

    if scale_format == ScaleFormat.UINT16_COL_MAJOR:
        out_quant_tensor, out_scale = _downcast_to_mxfp4_u16_wrapper(
            src_tensor, out_quant_tensor, L, DEQUANT_SCALE_ROUNDING_MODE
        )
    else:
        out_quant_tensor, out_scale = _downcast_to_mxfp_u8_wrapper(
            src_tensor, out_quant_tensor, L, DEQUANT_SCALE_ROUNDING_MODE
        )

    out_quant_tensor = out_quant_tensor.transpose(axis, ndim - 1)
    out_scale = out_scale.transpose(axis, ndim - 1)
    return out_quant_tensor, out_scale


def _downcast_to_mxfp4_u16_wrapper(
    src_tensor: torch.Tensor,
    out_quant_tensor: torch.Tensor,
    L: int,
    mode: DequantScaleRoundingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner helper for uint16 col-major scale path."""
    # Pad quant_dim to multiple of 64 for even scale groups (uint16 packing requires
    # an even number of groups). Kernel uses mask for OOB elements — no actual data pad.
    padded_L = ((L + 63) // 64) * 64
    S_groups = padded_L // MXFP_BLOCK_SIZE.value
    S_groups_pairs = S_groups // 2

    if src_tensor.numel() > 0:
        kernel_src = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        N = kernel_src.shape[0]  # flattened outer dim

        # Scale output: [S_groups_pairs, N] uint16 contiguous (col-major physical layout)
        kernel_scale = torch.empty(
            (S_groups_pairs, N), dtype=torch.uint16, device=src_tensor.device
        )

        BLOCK_OUT = 32
        BLOCK_QUANT = 128
        NUM_WARPS = 4

        grid = (
            triton.cdiv(N, BLOCK_OUT),
            triton.cdiv(padded_L, BLOCK_QUANT),
        )

        _downcast_to_mxfp4_u16[grid](
            kernel_quant, *kernel_quant.stride(),
            kernel_scale, *kernel_scale.stride(),
            kernel_src, *kernel_src.stride(),
            N, padded_L, L,
            BLOCK_SIZE_OUT_DIM=BLOCK_OUT,
            BLOCK_SIZE_QUANT_DIM=BLOCK_QUANT,
            DEQUANT_SCALE_ROUNDING_MODE=mode.value,
            num_warps=NUM_WARPS,
        )

        # Reshape scale from physical [S_pairs, N] to logical [..., S_pairs]
        batch_shape = src_tensor.shape[:-1]
        out_scale = kernel_scale.t().reshape(*batch_shape, S_groups_pairs)
    else:
        batch_shape = src_tensor.shape[:-1]
        out_scale = torch.empty(
            *batch_shape, S_groups_pairs, dtype=torch.uint16, device=src_tensor.device
        )

    return out_quant_tensor, out_scale


def _downcast_to_mxfp_u8_wrapper(
    src_tensor: torch.Tensor,
    out_quant_tensor: torch.Tensor,
    L: int,
    mode: DequantScaleRoundingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner helper for uint8 row-major scale path (legacy)."""
    S_groups = triton.cdiv(L, MXFP_BLOCK_SIZE.value)
    batch_shape = src_tensor.shape[:-1]
    out_scale = src_tensor.new_empty((*batch_shape, S_groups), dtype=torch.uint8)

    if src_tensor.numel() > 0:
        kernel_src = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        kernel_scale = out_scale.view(-1, out_scale.shape[-1])

        BLOCK_OUT_DIM = 128
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value
        grid_out = triton.cdiv(kernel_src.shape[0], BLOCK_OUT_DIM)
        grid_quant = triton.cdiv(kernel_src.shape[1], BLOCK_QUANT_DIM)

        _downcast_to_mxfp[(grid_out, grid_quant)](
            kernel_quant, *kernel_quant.stride(),
            kernel_scale, *kernel_scale.stride(),
            kernel_src, *kernel_src.stride(),
            *kernel_src.shape,
            BLOCK_OUT_DIM, BLOCK_QUANT_DIM,
            mode.value,
            num_warps=8,
        )

    return out_quant_tensor, out_scale
