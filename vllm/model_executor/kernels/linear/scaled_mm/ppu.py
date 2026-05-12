# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm.utils.ppu_deep_gemm import (
    get_deep_gemm_config,
    int8_gemm_nt,
    is_deep_gemm_supported,
    fp8_gemm_nt,
    should_use_deepgemm_for_fp8_linear,
)

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)

logger = init_logger(__name__)

if current_platform.is_ppu():
    try:
        from acext import int8_gemm as acext_int8_gemm_kernel
    except ImportError:
        logger.debug("Fail to import acext int8_gemm")
        acext_int8_gemm_kernel = None


def w8a8_int8_matmul_acext(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert (bias is None) or (bias.shape[0] == b.shape[0] and bias.dtype == out_dtype)
    return acext_int8_gemm_kernel(a, b, scale_b, scale_a, bias, out_dtype)


def w8a8_int8_matmul_acext_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty((m, n), dtype=out_dtype, device=a.device)


if current_platform.is_ppu():
    direct_register_custom_op(
        op_name="w8a8_int8_matmul_acext",
        op_func=w8a8_int8_matmul_acext,
        mutates_args=[],
        fake_impl=w8a8_int8_matmul_acext_fake,
    )


def w8a8_int8_matmul_deepgemm(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.empty((x_q.shape[0], w_q.shape[0]), dtype=out_dtype, device=x_q.device)
    M, N, K = x_q.shape[0], w_q.shape[0], x_q.shape[1]
    best_config = get_deep_gemm_config(M, N, K, num_groups=1)
    int8_gemm_nt((x_q, scale_x), (w_q, scale_w), out, configs=best_config)
    return out


def w8a8_int8_matmul_deepgemm_fake(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    m = x_q.shape[0]
    n = w_q.shape[0]
    return torch.empty((m, n), dtype=out_dtype, device=x_q.device)

if current_platform.is_ppu():
    direct_register_custom_op(
        op_name="w8a8_int8_matmul_deepgemm",
        op_func=w8a8_int8_matmul_deepgemm,
        mutates_args=[],
        fake_impl=w8a8_int8_matmul_deepgemm_fake,
    )


def _fp8_gemm_nt_op(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    fp8_gemm_nt(
        (q_input, input_scale),
        (weight, weight_scale),
        output,
    )


def _fp8_gemm_nt_op_fake(
    q_input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    return None


if current_platform.is_ppu():
    direct_register_custom_op(
        "w8a8_fp8_matmul_deepgemm",
        _fp8_gemm_nt_op,
        mutates_args=["output"],
        fake_impl=_fp8_gemm_nt_op_fake,
    )


class PPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_ppu():
            return False, "requires PPU."
        return True, None

    @classmethod
    def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.use_acext_int8_gemm = False
        self.use_deepgemm_int8_gemm = False

        # check acext
        if (
            (acext_int8_gemm_kernel is not None)
            and (not envs.VLLM_SAIL_DENSE_BACKEND
                 or envs.VLLM_SAIL_DENSE_BACKEND == "acext")
            and current_platform.is_device_capability((8, 0))
        ):
            self.use_acext_int8_gemm = True

        # check deepgemm
        if (
            is_deep_gemm_supported()
            and (not envs.VLLM_SAIL_DENSE_BACKEND
                 or envs.VLLM_SAIL_DENSE_BACKEND == "deep_gemm")
        ):
            self.use_deepgemm_int8_gemm = True

        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        config = self.config
        # WEIGHT PROCESS
        weight = getattr(layer, w_q_name)
        if self.use_acext_int8_gemm or self.use_deepgemm_int8_gemm:
            # acext and DG kernels need row major weight.
            replace_parameter(
                layer,
                w_q_name,
                torch.nn.Parameter(weight.data, requires_grad=False),
            )
            self.weight_RowMajor = True
        else:
            # Cutlass kernels need col major weight.
            replace_parameter(
                layer,
                w_q_name,
                torch.nn.Parameter(weight.t().data, requires_grad=False),
            )
            self.weight_RowMajor = False

        # WEIGHT SCALE
        # Cutlass kernels support only per-tensor and per-channel.
        # If we have a fused module (QKV, MLP) with per tensor scales (thus N
        # scales being passed to the kernel), convert to the per-channel case.
        is_fused_module = len(layer.logical_widths) > 1
        weight_scale = getattr(layer, w_s_name)
        if is_fused_module and not config.is_channelwise:
            weight_scale = convert_to_channelwise(weight_scale, layer.logical_widths)
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(weight_scale.data, requires_grad=False),
        )

        # INPUT SCALE
        if config.is_static_input_scheme:
            input_scale = getattr(layer, i_s_name)

            if config.input_symmetric:
                replace_parameter(
                    layer,
                    i_s_name,
                    torch.nn.Parameter(input_scale.max(), requires_grad=False),
                )
                setattr(layer, i_zp_name, None)
            else:
                input_zero_point = getattr(layer, i_zp_name)

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
                replace_parameter(
                    layer, i_s_name, torch.nn.Parameter(scale, requires_grad=False)
                )

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
                replace_parameter(
                    layer, i_zp_name, torch.nn.Parameter(azp, requires_grad=False)
                )

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/w8a8/cutlass/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/main/csrc/quantization/w8a8/cutlass/Epilogues.md
        if not config.input_symmetric:
            weight = getattr(layer, w_q_name)
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if config.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = getattr(layer, i_zp_name) * azp_adj
            setattr(
                layer,
                azp_adj_name,
                torch.nn.Parameter(azp_adj, requires_grad=False),
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, azp_adj = self._get_layer_params(layer)

        # Flatten leading dimensions to avoid issues with deepgemm/acext
        # kernels that only accept 2D inputs. The GPU cutlass_scaled_mm
        # wrapper applies the same flatten/reshape pattern internally
        # (see vllm/_custom_ops.py).
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        symmetric = azp_adj is None
        if symmetric and (i_s is None) and envs.VLLM_SAIL_USE_TRITON_INT8_QUANT:
            x_q, x_s = per_token_group_quant_int8(
                x.contiguous(),
                x.shape[-1],
                dtype=torch.int8,
                use_triton=True,
                use_rounding=True,
            )
            x_zp = None
        else:
            x_q, x_s, x_zp = ops.scaled_int8_quant(
                x.contiguous(), i_s, i_zp, symmetric=symmetric
            )

        if x_zp is not None:
            # Currently, static is always per-tensor and dynamic is per-token
            static = i_zp is not None
            azp = None if static else x_zp
            out = ops.cutlass_scaled_mm_azp(
                x_q,
                w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
            return out.view(*original_shape[:-1], out.shape[-1])
        if (bias is None) and self.use_deepgemm_int8_gemm:
            out = torch.ops.vllm.w8a8_int8_matmul_deepgemm(
                x_q, w_q, scale_x=x_s, scale_w=w_s, out_dtype=x.dtype
            )
            return out.view(*original_shape[:-1], out.shape[-1])
        elif self.use_acext_int8_gemm:
            out = torch.ops.vllm.w8a8_int8_matmul_acext(
                x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=bias
            )
            return out.view(*original_shape[:-1], out.shape[-1])
        else:
            out = ops.cutlass_scaled_mm(
                x_q,
                w_q.t() if self.weight_RowMajor else w_q,
                scale_a=x_s,
                scale_b=w_s,
                out_dtype=x.dtype,
                bias=bias,
            )
            return out.view(*original_shape[:-1], out.shape[-1])


class PPUDeepGemmFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """PPU DeepGEMM kernel for FP8 channelwise (per-channel weight scale)."""

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_ppu():
            return False, "DeepGEMM is only supported on ppu platform"
        if current_platform.is_device_capability((8, 0)):
            return False, "FP8 DeepGEMM is not supported on SM80 devices."
        if not is_deep_gemm_supported():
            return False, "DeepGEMM library is not available."
        return True, None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        if config.out_dtype != torch.bfloat16:
            return False, "Supports only output dtype of bfloat16"
        if not should_use_deepgemm_for_fp8_linear(
            config.out_dtype, config.weight_shape
        ):
            return False, "The provided metadata is not supported."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Base class stores weight as [K, N] (col-major).
        # fp8_gemm_nt channelwise requires row-major [N, K], transpose here.
        w = getattr(layer, "weight")
        replace_parameter(
            layer,
            "weight",
            torch.nn.Parameter(w.data.t().contiguous(), requires_grad=False),
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # B is [N, K] (row-major after process_weights_after_loading).
        # output_shape[-1] from base class is w.shape[1] which was K before
        # transpose, now correct N = B.shape[0].
        M = A.shape[0]
        N = B.shape[0]
        output = torch.empty((M, N), dtype=out_dtype, device=A.device)
        torch.ops.vllm.w8a8_fp8_matmul_deepgemm(A, As, B, Bs, output)
        if bias is not None:
            output = output + bias
        correct_output_shape = output_shape[:-1] + [N]
        return output.view(*correct_output_shape)


class PPUCutlassFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_ppu():
            return False, "requires PPU."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # Fused GEMM_DQ
        output = ops.cutlass_scaled_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )
        return output.view(*output_shape)


class PPUCutlassFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        act_scale_descriptor = config.activation_quant_key.scale
        self.weight_group_shape = config.weight_quant_key.scale.group_shape
        self.quant_fp8 = QuantFP8(
            static=act_scale_descriptor.static,
            group_shape=act_scale_descriptor.group_shape,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
            column_major_scales=True,
        )
        self.is_hopper = current_platform.is_device_capability(90)

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_ppu():
            return False, "requires PPU."
        return True, None

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype

        return ops.cutlass_scaled_mm(
            A,
            B.T,
            out_dtype=out_dtype,
            scale_a=As,
            scale_b=Bs.T,
        )


class PPUDeepGemmFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)

        act_scale_descriptor = config.activation_quant_key.scale
        self.is_deep_gemm_supported = is_deep_gemm_supported()
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=act_scale_descriptor.group_shape,
            use_ue8m0=False,
            tma_aligned_scales=False,
            column_major_scales=True,
        )

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_ppu():
            return False, "DeepGEMM is only supported on ppu platform"
        if current_platform.is_device_capability((8, 0)):
            return False, "FP8 DeepGEMM is not supported on SM80 devices."
        if not is_deep_gemm_supported():
            return False, "Currently, only sm89 PPU are supported."
        return True, None

    @classmethod
    def can_implement(cls, config):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason
        if config.out_dtype != torch.bfloat16:
            return (False, "Supports only output dtype of bfloat16")

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )
        model_config = get_current_vllm_config().model_config

        if model_config is None:
            return False, "Model configuration is required."

        if not should_use_deepgemm_for_fp8_linear(
            config.out_dtype, config.weight_shape
        ):
            return False, "The provided metadata is not supported."
        return True, None

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        params = self._get_layer_params(layer)
        assert layer.weight_block_size is not None

        if self.is_deep_gemm_supported:
            weight_scale_invs = params.weight_scale_inv
            scale_attr = (
                params.WEIGHT_SCALE_INV
                if weight_scale_invs is not None
                else params.WEIGHT_SCALE
            )
            dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
                wq=params.weight,
                ws=weight_scale_invs
                if weight_scale_invs is not None
                else params.weight_scale,
                quant_block_shape=tuple(layer.weight_block_size),
                use_e8m0=False,
                is_bmm=getattr(layer, "is_bmm", False),
                bmm_batch_size=getattr(layer, "bmm_batch_size", 0),
            )
            replace_parameter(layer, params.WEIGHT, dg_weight)
            replace_parameter(layer, scale_attr, dg_weight_scale)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        output = torch.empty(
            (A.shape[0], B.shape[0]),
            dtype=out_dtype,
            device=A.device,
        )
        torch.ops.vllm.w8a8_fp8_matmul_deepgemm(A, As, B, Bs, output)
        return output
