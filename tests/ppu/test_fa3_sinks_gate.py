# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the attention-sink capability gate for FlashAttention on PPU.

Upstream rejects attention sinks below SM90 because its FA3 sink path is
Hopper-only. The PPU FA3 kernel takes ``s_aux`` (sink biases) through
``flash_attn_3.fwd`` (``vllm/vllm_flash_attn/ppu``), so sink models such as
gpt-oss should resolve to FA3 on PPU instead of silently falling back to
TRITON_ATTN. FA2 has no ``s_aux`` support (the PPU wrapper raises), so a
forced FA2 must keep failing closed.

The platform and ``get_flash_attn_version`` are stubbed, so this pins the
branching of ``FlashAttentionBackend.supports_combination`` without a device.
"""

import pytest
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends import flash_attn as fa_mod

SINK_REJECTION = "sink not supported on compute capability < 9.0"


class FakePlatform:
    def __init__(self, *, is_ppu: bool):
        self._is_ppu = is_ppu

    def is_ppu(self) -> bool:
        return self._is_ppu

    def is_xpu(self) -> bool:
        return False


@pytest.fixture
def pin(monkeypatch):
    def _pin(*, is_ppu: bool, fa_version: int):
        monkeypatch.setattr(fa_mod, "current_platform", FakePlatform(is_ppu=is_ppu))
        monkeypatch.setattr(
            fa_mod, "get_flash_attn_version", lambda **_: fa_version
        )

    return _pin


def _sink_reason(capability: tuple[int, int]) -> str | None:
    return fa_mod.FlashAttentionBackend.supports_combination(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=True,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=DeviceCapability(*capability),
    )


def test_ppu_fa3_allows_sinks_below_sm90(pin):
    pin(is_ppu=True, fa_version=3)
    assert _sink_reason((8, 9)) is None
    assert _sink_reason((8, 0)) is None


def test_ppu_forced_fa2_keeps_sinks_rejected(pin):
    pin(is_ppu=True, fa_version=2)
    assert _sink_reason((8, 9)) == SINK_REJECTION


def test_cuda_sink_gate_unchanged(pin):
    pin(is_ppu=False, fa_version=3)
    assert _sink_reason((8, 9)) == SINK_REJECTION
    assert _sink_reason((9, 0)) is None
