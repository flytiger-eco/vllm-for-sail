# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acext-based fused MoE expert implementation."""

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


# Acext availability check
if current_platform.is_ppu() and current_platform.is_device_capability((8,0)):
    try:
        from acext import fusedmoe_wrapper, get_enum_from_booleans
        _ACEXT_AVAILABLE = True
    except ImportError:
        _ACEXT_AVAILABLE = False
else:
    _ACEXT_AVAILABLE = False


def _pad_to_multiple_of_16(value: int) -> int:
    """Calculate paddings to ensure value can be divide by 16."""
    padding = (16 - value % 16) % 16
    return value + padding


class AcextExperts(mk.FusedMoEExpertsModular):
    """Acext-based fused MoE expert implementation."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.ep_rank = moe_config.ep_rank
        self.ep_size = moe_config.ep_size

    @property
    def expects_unquantized_inputs(self) -> bool:
        """
        Whether or not the PrepareFinalize should defer input quantization
        in the prepare step. If True, then the Experts kernel will
        execute the input quantization itself.

        Sample subclasses that override are AITER and FlashInfer CUTLASS.
        """
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard
    
    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_ppu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        """Whether Acext supports non-gated MoE models."""
        # TODO: Verify if Acext supports act_and_mul=False
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """Whether Acext supports the given quantization scheme."""
        # PPU NOTE: Acext supports bf16 and int8 quantization
        SUPPORTED_W_A = [
            (None, None),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """Whether Acext supports the given activation function."""
        # Acext supports SILU activation
        return activation in [MoEActivation.SILU]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        """Whether Acext supports the given parallel configuration."""
        # TODO: Verify EP support in Acext
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self) -> bool:
        """Whether Acext supports expert mapping."""
        # TODO: Verify expert map support in Acext
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        """Return the weight reduction implementation."""
        return TopKWeightAndReduceNoOP()

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
        """
        Compute workspace shapes for Acext experts.

        Acext handles memory internally, so we don't need large workspace buffers.
        Returns minimal workspace shapes.
        """
        # PPU Note (kai): Acext manages memory internally, minimal workspace needed
        workspace1 = (0, ) 
        workspace2 = (0, )
        output = (M, K)
        return (workspace1, workspace2, output)

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
        """Apply Acext fused MoE computation."""
        # Check constraints
        assert hidden_states.size(-1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
        )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"

        # Get problem dimensions
        num_tokens = hidden_states.size(0)
        top_k_num = topk_ids.size(1)

        # Calculate padded size for Acext
        expanded_source_row_to_dest_size = _pad_to_multiple_of_16(
            num_tokens * top_k_num
        )

        # Determine quantization type
        use_fp8_w8a8 = self.quant_config.use_fp8_w8a8
        use_int8_w8a8 = self.quant_config.use_int8_w8a8
        use_int8_w8a16 = self.quant_config.use_int8_w8a16
        use_int4_w4a16 = self.quant_config.use_int4_w4a16
        use_fp8_w8a16 = self.quant_config.use_fp8_w8a16

        Q_type = get_enum_from_booleans(
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            use_fp8_w8a16,
        )

        # Get status from Acext to check if it can handle this case
        w1_scale = getattr(self.quant_config, 'w1_scale', None)
        w2_scale = getattr(self.quant_config, 'w2_scale', None)
        w1_zp = getattr(self.quant_config, 'w1_zp', None)
        w2_zp = getattr(self.quant_config, 'w2_zp', None)

        # Call Acext fused MoE wrapper
        fusedmoe_wrapper(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            output,
            expanded_source_row_to_dest_size,
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1q_scale,
            a2_scale,
            self.ep_rank,
            self.ep_size,
            Q_type,
            None,
        )


def is_acext_supported() -> bool:
    """Check if Acext is supported on the current platform."""
    return _ACEXT_AVAILABLE
