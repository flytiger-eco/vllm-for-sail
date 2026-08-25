# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific DSpark draft model for DeepSeek V4.

Inherits from the NVIDIA DSpark implementation and exposes the weight
mapping attributes that the PPU mixed-precision recipe (mxfp4 MoE + fp8
channel-wise dense) needs on the draft model's quantization config.
"""

from vllm.model_executor.models.interfaces import SupportsQuant

from ..nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM as NvidiaDSparkDeepseekV4ForCausalLM,
)
from ..nvidia.model import _make_deepseek_v4_weights_mapper


class DSparkDeepseekV4ForCausalLM(
    NvidiaDSparkDeepseekV4ForCausalLM, SupportsQuant
):
    """PPU-specific DSpark draft model.

    Differences from the NVIDIA version (mirrors ``ppu/model.py``):
    1. Adds ``packed_modules_mapping`` so ``is_layer_skipped`` can expand
       fused dense modules (``fused_wqa_wkv``/``gate_up_proj``) back to the
       per-shard names listed in ``fp8_channelwise_layers``.
    2. Adds ``hf_to_vllm_mapper`` so ``apply_vllm_mapper`` maps and
       index-strips the channelwise layer list of the draft's quant config.

    Why this is needed here and not on the target model: the DSpark loader
    builds a *fresh* quant config for the draft
    (``load_dspark_model`` → ``get_draft_quant_config``), unlike the MTP
    draft which shares the target's already-configured config. The NVIDIA
    DSpark class is a plain ``nn.Module`` that declares neither attribute,
    so on PPU the fresh config was left unconfigured: fused dense layers
    failed the ``fp8_channelwise_layers`` match, silently fell back to
    ``Fp8LinearMethod`` (per-tensor scales), and loading the channel-wise
    checkpoint scales (shape ``[N, 1]``, ``N > 1``) tripped
    ``PerTensorScaleParameter._load_into_shard_id``'s
    ``assert loaded_weight.shape[0] == 1``.

    ``SupportsQuant`` injects both the fused mapping and the weights
    mapper into the draft's quant config at instance creation (``__new__``),
    before any decoder layer is built — independent of the loader path
    (same mechanism as the MiniMax M3 fix).
    """

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # Any DSV4 mapper variant works here: DeepseekV4FP8Config's
    # apply_vllm_mapper only uses the name-level maps (prefix/substr/suffix)
    # on the fp8_channelwise_layers entries, and those maps are identical
    # across expert_dtype variants (the variants differ only in the
    # ``.scale`` regexes, which never apply to layer-path entries). The fp4
    # default matches the class attribute on the NVIDIA target model.
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")
