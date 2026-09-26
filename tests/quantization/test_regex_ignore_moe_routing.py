# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for regex ("re:") ignore entries in MoE expert routing.

Regression test: the w4a8 checkpoint Qwen3.8-2.4T-A95B-FP8-MoE-Quant-W-INT4-
PerChannel-A-INT8-PerChannel-Dense-Quant-W-INT8-PerChannel-A-INT8-PerChannel
declares its unquantized MTP experts only through "re:.*mtp.*". All 12 entries
of its quantization_config.ignore are regex patterns and not one of them
contains "experts". is_layer_skipped() compares ignore entries as exact strings
and, for a prefix containing "experts", narrows the candidates down to entries
that themselves contain "experts", so nothing matched: the BF16 MTP experts were
routed to the INT4-packed method and weight loading died with "The size of
tensor a (4096) must match the size of tensor b (8192)".

Only the routing decision is exercised here. The MoE method classes are stubbed
because instantiating them selects a platform backend.
"""

import pytest

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import fp8 as fp8_mod
from vllm.model_executor.layers.quantization import mixed_precision_w4 as w4_mod
from vllm.model_executor.layers.quantization import mxfp4 as mxfp4_mod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mixed_precision_w4 import (
    MixedPrecisionW4Config,
)
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

# quantization_config.ignore of the w4a8 checkpoint above: every entry is a
# regex and the MTP experts are covered by the last one only.
REGEX_ONLY_IGNORE = [
    "re:.*lm_head",
    "re:visual.*",
    "re:model.visual.*",
    "re:.*mlp.shared_expert_gate$",
    "re:.*mlp.gate$",
    "re:.*conv1d$",
    "re:.*in_proj_a$",
    "re:.*in_proj_b$",
    "re:.*fc$",
    "re:.*pre_fc_norm_embedding$",
    "re:.*pre_fc_norm_hidden$",
    "re:.*mtp.*",
]

# int8 checkpoints spell every unquantized expert out explicitly instead.
EXPLICIT_EXPERT_IGNORE = [
    "model.mtp.layers.0.mlp.experts.0.gate_proj",
    "model.mtp.layers.0.mlp.experts.0.up_proj",
]

MTP_EXPERTS = "model.mtp.layers.0.mlp.experts"
MTP_EXPERTS_NO_MODEL = "mtp.layers.0.mlp.experts"
MAIN_EXPERTS = "model.layers.0.mlp.experts"

UNQUANTIZED = "unquantized"
QUANTIZED = "quantized"


def _routed_experts() -> RoutedExperts:
    """Bare RoutedExperts instance; routing only reads ``moe_config`` off it."""
    layer = object.__new__(RoutedExperts)
    layer.moe_config = "moe_config"
    return layer


def _stub_unquantized(monkeypatch, mod) -> None:
    monkeypatch.setattr(
        mod, "UnquantizedFusedMoEMethod", lambda *args, **kwargs: UNQUANTIZED
    )


@pytest.mark.parametrize(
    "ignore,prefix,expected",
    [
        # regex-only ignore list: MTP experts must fall back to unquantized,
        # main model experts must stay quantized
        (REGEX_ONLY_IGNORE, MTP_EXPERTS, UNQUANTIZED),
        (REGEX_ONLY_IGNORE, MTP_EXPERTS_NO_MODEL, UNQUANTIZED),
        (REGEX_ONLY_IGNORE, MAIN_EXPERTS, QUANTIZED),
        # explicit per-expert entries keep working through is_layer_skipped()
        (EXPLICIT_EXPERT_IGNORE, MTP_EXPERTS, UNQUANTIZED),
        (EXPLICIT_EXPERT_IGNORE, MAIN_EXPERTS, QUANTIZED),
    ],
)
def test_mixed_precision_w4_expert_routing(monkeypatch, ignore, prefix, expected):
    _stub_unquantized(monkeypatch, w4_mod)
    monkeypatch.setattr(w4_mod, "W4AInt8MoEMethod", lambda *a, **k: QUANTIZED)
    monkeypatch.setattr(
        MixedPrecisionW4Config,
        "get_int8_channelwise_quant_method",
        lambda *a, **k: QUANTIZED,
    )
    config = MixedPrecisionW4Config(ignored_layers=list(ignore))
    assert config.get_quant_method(_routed_experts(), prefix) == expected


@pytest.mark.parametrize(
    "ignore,prefix,expected",
    [
        (REGEX_ONLY_IGNORE, MTP_EXPERTS, UNQUANTIZED),
        (REGEX_ONLY_IGNORE, MAIN_EXPERTS, QUANTIZED),
        (EXPLICIT_EXPERT_IGNORE, MTP_EXPERTS, UNQUANTIZED),
        (EXPLICIT_EXPERT_IGNORE, MAIN_EXPERTS, QUANTIZED),
    ],
)
def test_fp8_expert_routing(monkeypatch, ignore, prefix, expected):
    _stub_unquantized(monkeypatch, fp8_mod)
    monkeypatch.setattr(fp8_mod, "Fp8MoEMethod", lambda *a, **k: QUANTIZED)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=list(ignore),
    )
    assert config.get_quant_method(_routed_experts(), prefix) == expected


@pytest.mark.parametrize(
    "ignore,prefix,expected",
    [
        (REGEX_ONLY_IGNORE, MTP_EXPERTS, UNQUANTIZED),
        (REGEX_ONLY_IGNORE, MAIN_EXPERTS, QUANTIZED),
        (EXPLICIT_EXPERT_IGNORE, MTP_EXPERTS, UNQUANTIZED),
        (EXPLICIT_EXPERT_IGNORE, MAIN_EXPERTS, QUANTIZED),
    ],
)
def test_mxfp4_expert_routing(monkeypatch, ignore, prefix, expected):
    _stub_unquantized(monkeypatch, mxfp4_mod)
    monkeypatch.setattr(
        Mxfp4Config, "_make_moe_method", lambda *a, **k: QUANTIZED
    )
    config = Mxfp4Config(ignored_layers=list(ignore))
    assert config.get_quant_method(_routed_experts(), prefix) == expected
