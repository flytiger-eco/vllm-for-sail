# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific quantization config for DeepSeek V4.

Extends the base DeepseekV4FP8Config with PPU-specific overrides:
- Mixed-precision checkpoint detection (mxfp4 MoE + fp8 channelwise dense)
- GptOssMxfp4MoEMethod for PPU MoE layers
"""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

from ..quant_config import DeepseekV4FP8Config


class DeepseekV4FP8ConfigPPU(DeepseekV4FP8Config):
    """PPU-specific DeepSeek V4 FP8 config.

    Adds:
    - mxfp4 + fp8_channelwise override_quantization_method
    """

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        # First try the base class logic
        result = super().override_quantization_method(
            hf_quant_cfg, user_quant, hf_config
        )
        if result is not None:
            return result
        # PPU: mixed-precision checkpoint (mxfp4 MoE + fp8 channelwise dense)
        if not isinstance(hf_quant_cfg, dict):
            return None
        quant_method = hf_quant_cfg.get("quant_method")
        model_type = getattr(hf_config, "model_type", None)
        if (
            quant_method == "mxfp4"
            and model_type == "deepseek_v4"
            and hf_quant_cfg.get("fp8_channelwise_layers")
        ):
            return "deepseek_v4_fp8"
        return None