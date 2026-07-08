# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific DeepSeek V4 model.

Inherits from the NVIDIA implementation and configures weight mapping
to support PPU-specific quantization schemes (INT8/INT4 channelwise,
FP8 channelwise dense + FP4 MoE mixed-precision).
"""

from vllm.config import VllmConfig

from ..nvidia.model import (
    DeepseekV4ForCausalLM as NvidiaDeepseekV4ForCausalLM,
    _make_deepseek_v4_weights_mapper,
)


class DeepseekV4ForCausalLM(NvidiaDeepseekV4ForCausalLM):
    """PPU-specific DeepSeek V4 model.

    Differences from NVIDIA version:
    1. Adds packed_modules_mapping for fused weight loading
    2. Detects fp8_channelwise_layers and int8/int4 expert_dtype for mapper
    """

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        expert_dtype = getattr(config, "expert_dtype", "fp4")

        # Detect fp8 channelwise layers from quantization config
        fp8_channelwise_layers: list[str] = []
        quant_cfg = getattr(config, "quantization_config", {}) or {}
        fp8_channelwise_layers = (
            quant_cfg.get("fp8_channelwise_layers", []) or []
        )

        # Detect full channelwise model → use int8 mapper
        is_full_channelwise = False
        quant_config = getattr(vllm_config, "quant_config", None)
        target_scheme_map = getattr(quant_config, "target_scheme_map", None)
        if target_scheme_map:
            for scheme_dict in target_scheme_map.values():
                weights_args = scheme_dict.get("weights")
                strategy = getattr(weights_args, "strategy", None)
                if strategy is not None and "channel" in str(strategy).lower():
                    is_full_channelwise = True
                    break

        # Set appropriate mapper before super().__init__ builds the model
        if is_full_channelwise:
            self.__class__.hf_to_vllm_mapper = (
                _make_deepseek_v4_weights_mapper("int8")
            )
        elif expert_dtype != "fp4" or fp8_channelwise_layers:
            self.__class__.hf_to_vllm_mapper = (
                _make_deepseek_v4_weights_mapper(
                    expert_dtype,
                    fp8_channelwise_layers=(
                        fp8_channelwise_layers
                        if expert_dtype == "fp4"
                        else None
                    ),
                )
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)
