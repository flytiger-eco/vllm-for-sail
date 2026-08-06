# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify MiniMax M3 packed_modules_mapping is injected into the quant config
so PPU FP8 channel-wise dense routing sees fused layers correctly.

Regression tests for the silent-fallback bug where
``MiniMaxM3SparseForCausalLM`` declared neither ``packed_modules_mapping``
nor ``SupportsQuant``, leaving ``Mxfp4Config.packed_modules_mapping`` empty:
``is_layer_skipped`` could not expand ``qkv_proj``/``gate_up_proj`` back to
the shard names listed in ``fp8_channelwise_layers``, and fused dense layers
silently fell back to ``UnquantizedLinearMethod``.

Checks the mapping injection through the real ``SupportsQuant.__new__``
path (no GPU, no weights).
"""

import pytest

from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.models.interfaces import SupportsQuant

FP8_CHANNELWISE_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

EXPECTED_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _model_classes():
    from vllm.models.minimax_m3 import (
        MiniMaxM3SparseForCausalLM,
        MiniMaxM3SparseForConditionalGeneration,
    )

    return MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration


def _make_quant_config() -> Mxfp4Config:
    return Mxfp4Config(fp8_channelwise_layers=list(FP8_CHANNELWISE_LAYERS))


class TestPackedModulesMappingInjection:
    """The fix contract: model classes expose the fused mapping and inject it
    into the quant config at instance creation, before any layer is built."""

    def test_model_classes_declare_mapping_and_supports_quant(self):
        for cls in _model_classes():
            assert issubclass(cls, SupportsQuant), (
                f"{cls.__name__} must inherit SupportsQuant so the mapping is "
                "injected in __new__ regardless of loader path"
            )
            assert cls.packed_modules_mapping == EXPECTED_FUSED_MAPPING

    def test_new_injects_mapping_into_quant_config(self):
        """Exercise the real SupportsQuant.__new__ injection path without
        building the full model (no __init__, no weights, no distributed)."""
        causal_lm, _ = _model_classes()
        quant_config = _make_quant_config()
        assert quant_config.packed_modules_mapping == {}

        # SupportsQuant._find_quant_config picks up a QuantizationConfig
        # positional arg, same as it picks vllm_config.quant_config at real
        # model bring-up.
        instance = causal_lm.__new__(causal_lm, quant_config)

        assert instance.quant_config is quant_config
        assert quant_config.packed_modules_mapping == EXPECTED_FUSED_MAPPING