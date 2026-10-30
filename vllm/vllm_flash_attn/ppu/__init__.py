# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.vllm_flash_attn.flash_attn_interface import (
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    fa3_qsa_unsupported_reason,
    fa3_qsa_uses_allow_aiu_flag,
    fa_version_unsupported_reason,
    flash_attn_qsa_varlen_func,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa3_qsa_supported,
    is_fa_version_supported,
)

if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "PPU requires PPU's fa2 or fa3 whl"
    )

__all__ = [
    "fa3_qsa_unsupported_reason",
    "fa3_qsa_uses_allow_aiu_flag",
    "fa_version_unsupported_reason",
    "flash_attn_qsa_varlen_func",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa3_qsa_supported",
    "is_fa_version_supported",
]
