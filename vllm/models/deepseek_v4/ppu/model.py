# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific DeepSeek V4 model.

Inherits from the NVIDIA implementation and overrides weight loading
to support PPU-specific quantization schemes (INT8/INT4 channelwise,
FP8 channelwise dense + FP4 MoE mixed-precision).
"""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

from ..nvidia.model import (
    DeepseekV4ForCausalLM as NvidiaDeepseekV4ForCausalLM,
)
from ..nvidia.model import (
    _make_deepseek_v4_weights_mapper,
)


def _pre_dequant_wo_a_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Pre-dequant FP8 channelwise wo_a before TP sharding.

    Buffers only wo_a entries; all others pass through immediately.
    Block-FP8 scales (2D, both dims > 1) are passed through unchanged.
    """
    wo_a_buf: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}

    for name, tensor in weights:
        if name.endswith(".wo_a.weight"):
            key = name[: -len(".weight")]
            wo_a_buf.setdefault(key, {})["weight"] = (name, tensor)
        elif ".wo_a." in name and "scale" in name:
            key = name.split(".wo_a.")[0] + ".wo_a"
            wo_a_buf.setdefault(key, {})["scale"] = (name, tensor)
        else:
            yield name, tensor

    for key, buf in wo_a_buf.items():
        if "weight" not in buf:
            if "scale" in buf:
                yield buf["scale"]
            continue

        w_name, w = buf["weight"]

        if "scale" not in buf or w.dtype != torch.float8_e4m3fn:
            yield w_name, w
            if "scale" in buf:
                yield buf["scale"]
            continue

        s_name, s = buf["scale"]

        is_channelwise = s.dim() == 1 or (
            s.dim() == 2 and (s.shape[0] == 1 or s.shape[1] == 1)
        )
        if not is_channelwise:
            yield w_name, w
            yield s_name, s
            continue

        scale_flat = s.float().reshape(-1)
        if scale_flat.numel() == w.shape[0]:
            # Per-output-channel
            dequanted = w.float() * scale_flat.unsqueeze(1)
        elif scale_flat.numel() == w.shape[1]:
            # Per-input-channel
            dequanted = w.float() * scale_flat.unsqueeze(0)
        else:
            raise ValueError(
                f"wo_a scale shape {tuple(s.shape)} does not match weight "
                f"shape {tuple(w.shape)} (expected per-output or per-input "
                f"channel)"
            )
        yield w_name, dequanted.contiguous()


class DeepseekV4ForCausalLM(NvidiaDeepseekV4ForCausalLM):
    """PPU-specific DeepSeek V4 model.

    Differences from NVIDIA version:
    1. Adds packed_modules_mapping for fused weight loading
    2. Detects fp8_channelwise_layers and int8/int4 expert_dtype for mapper
    3. Pre-dequants wo_a weights at load time for channelwise path
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

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Load weights with PPU-specific wo_a pre-dequantization."""
        weights = _pre_dequant_wo_a_weights(weights)
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )
        self.model.finalize_mega_moe_weights()
        return loaded_params
