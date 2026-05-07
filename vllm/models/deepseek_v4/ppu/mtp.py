# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific DeepSeek V4 MTP (Multi-Token Prediction) model.

Inherits from the NVIDIA MTP implementation and overrides weight loading
to support INT8/INT4 expert_dtype scale suffix conventions.
"""

from collections.abc import Iterable
from typing_extensions import override

import regex as re
import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

from ..nvidia.model import make_deepseek_v4_expert_params_mapping
from ..nvidia.mtp import (
    DeepSeekV4MTP as NvidiaDeepSeekV4MTP,
)
from ..nvidia.mtp import (
    get_spec_layer_idx_from_weight_name,
)
from .model import _pre_dequant_wo_a_weights

_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DeepSeekV4MTP(NvidiaDeepSeekV4MTP):
    """PPU-specific MTP draft model.

    Adds INT8/INT4 expert_dtype support with appropriate scale suffix
    conventions:
    - INT8/INT4 experts: .weight_scale (not .weight_scale_inv)
    - INT8/INT4 non-expert: .weight_scale (not .weight_scale_inv)
    """

    @override
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        # PPU: pre-dequant fp8 channelwise wo_a (wo_a is built with
        # quant_config=None, so the raw scale has no parameter to load into).
        if current_platform.is_ppu():
            weights = _pre_dequant_wo_a_weights(weights)
        # Reuse the parent's internal helpers
        def _remap_weight_name(name: str) -> str:
            remap_table = {
                ".attn.compressor.": ".attn.mla_attn.compressor.",
                ".shared_experts.w2": ".shared_experts.down_proj",
            }
            for old_pattern, new_pattern in remap_table.items():
                if old_pattern in name:
                    name = name.replace(old_pattern, new_pattern)
            return name

        def _find_mtp_layer_idx(name: str) -> int:
            subnames = name.split(".")
            for subname in subnames:
                try:
                    return int(subname)
                except ValueError:
                    continue
            return 0

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        first_layer = next(iter(self.model.layers.values()))
        if first_layer.mtp_block.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = RoutedExperts.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        # INT8/INT4 experts and non-expert weights use .weight_scale
        expert_dtype = getattr(self.config, "expert_dtype", "fp4")
        # PPU mxfp4 MoE + fp8 channelwise dense uses ``.weight_scale`` (no
        # ``_inv``) for both experts and channelwise dense layers.
        has_fp8_channelwise = False
        if current_platform.is_ppu() and expert_dtype == "fp4":
            quant_cfg = getattr(self.config, "quantization_config", {}) or {}
            has_fp8_channelwise = bool(quant_cfg.get("fp8_channelwise_layers"))
        expert_scale_suffix = (
            ".weight_scale"
            if expert_dtype in ("fp4", "int8", "int4")
            else ".weight_scale_inv"
        )
        # INT8/INT4 non-expert weights also use .weight_scale
        # Mixed-precision FP4+FP8-channelwise also uses ``.weight_scale`` uniformly.
        non_expert_scale_suffix = (
            ".weight_scale"
            if expert_dtype in ("int8", "int4") or has_fp8_channelwise
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            mtp_layer_idx = _find_mtp_layer_idx(name)
            name = name.replace(
                f"mtp.{mtp_layer_idx}.",
                f"model.layers.{self.config.num_hidden_layers + mtp_layer_idx}.",
            )

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue

            name = _remap_weight_name(name)
            name = self._rewrite_spec_layer_name(spec_layer, name)

            if (
                spec_layer != self.model.mtp_start_layer_idx
                and ".layers" not in name
            ):
                continue
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else non_expert_scale_suffix
                )
                name = name.removesuffix(".scale") + suffix
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # Expert weight handling
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(name)
                    break
                else:
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    # Attention sink TP slicing
                    if name.endswith(".attn_sink"):
                        loaded_weight = loaded_weight[
                            head_rank_start:head_rank_end
                        ]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params
