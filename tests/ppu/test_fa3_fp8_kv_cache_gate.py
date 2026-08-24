# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the FlashAttention FP8 KV-cache gate on PPU.

Upstream restricts FP8 KV cache in the FlashAttention backend to FA3 on the
SM90 family and FA4 on the SM100 family
(``fa_utils.flash_attn_supports_kv_cache_dtype``). The PPU FA3 kernel
(``vllm/vllm_flash_attn/ppu``, ``flash_attn_3.fwd``) takes q/k/v descales,
so the PPU sm_89 SKU can serve FP8 KV cache with FA3; the sm_80 SKU has no
FP8 tensor support and must keep the dtype rejected.

This test pins the gate contract without needing a device: the platform is
stubbed and ``get_flash_attn_version`` is monkeypatched, so it exercises the
branching of ``flash_attn_supports_kv_cache_dtype`` itself.
"""

import pytest

from vllm.v1.attention.backends import fa_utils


class FakePlatform:
    def __init__(self, *, is_ppu: bool, supports_fp8: bool = False, families=()):
        self._is_ppu = is_ppu
        self._supports_fp8 = supports_fp8
        self._families = set(families)

    def is_ppu(self) -> bool:
        return self._is_ppu

    def is_xpu(self) -> bool:
        return False

    def supports_fp8(self) -> bool:
        return self._supports_fp8

    def is_device_capability_family(self, family: int) -> bool:
        return family in self._families


@pytest.fixture
def pin_fa_version(monkeypatch):
    def _pin(version: int):
        monkeypatch.setattr(
            fa_utils, "get_flash_attn_version", lambda **_: version
        )

    return _pin


def _gate(monkeypatch, platform: FakePlatform, dtype: str = "fp8_e4m3") -> bool:
    monkeypatch.setattr(fa_utils, "current_platform", platform)
    return fa_utils.flash_attn_supports_kv_cache_dtype(dtype)


def test_ppu_sm89_fa3_allows_fp8_kv_cache(monkeypatch, pin_fa_version):
    pin_fa_version(3)
    platform = FakePlatform(is_ppu=True, supports_fp8=True)
    assert _gate(monkeypatch, platform, "fp8_e4m3")
    assert _gate(monkeypatch, platform, "fp8")


def test_ppu_sm80_fa3_keeps_fp8_kv_cache_rejected(monkeypatch, pin_fa_version):
    pin_fa_version(3)
    platform = FakePlatform(is_ppu=True, supports_fp8=False)
    assert not _gate(monkeypatch, platform, "fp8_e4m3")


def test_ppu_fa2_keeps_fp8_kv_cache_rejected(monkeypatch, pin_fa_version):
    pin_fa_version(2)
    platform = FakePlatform(is_ppu=True, supports_fp8=True)
    assert not _gate(monkeypatch, platform, "fp8_e4m3")


def test_fp8_e5m2_rejected_everywhere(monkeypatch, pin_fa_version):
    pin_fa_version(3)
    platform = FakePlatform(is_ppu=True, supports_fp8=True)
    assert not _gate(monkeypatch, platform, "fp8_e5m2")


def test_cuda_gate_unchanged(monkeypatch, pin_fa_version):
    pin_fa_version(3)
    hopper = FakePlatform(is_ppu=False, families=(90,))
    assert _gate(monkeypatch, hopper)
    ampere = FakePlatform(is_ppu=False, families=(80,))
    assert not _gate(monkeypatch, ampere)

    pin_fa_version(4)
    blackwell = FakePlatform(is_ppu=False, families=(100,))
    assert _gate(monkeypatch, blackwell)
