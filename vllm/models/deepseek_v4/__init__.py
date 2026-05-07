# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/``, ``amd/``, and ``ppu/``;
this module picks the right one for the current platform and re-exports the
public classes used by the model registry and quantization config lookup.
"""

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

# Pick the per-platform implementation. The NVIDIA branch is the static
# default that mypy sees; the ROCm and PPU branches override at runtime.
if TYPE_CHECKING or (
    not current_platform.is_rocm() and not current_platform.is_ppu()
):
    from .nvidia.model import DeepseekV4ForCausalLM
    from .nvidia.mtp import DeepSeekV4MTP
    from .quant_config import DeepseekV4FP8Config
elif current_platform.is_rocm():
    from .amd.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .amd.mtp import DeepSeekV4MTP  # type: ignore[assignment]
    from .quant_config import DeepseekV4FP8Config
else:  # PPU
    from .ppu.model import DeepseekV4ForCausalLM  # type: ignore[assignment]
    from .ppu.mtp import DeepSeekV4MTP  # type: ignore[assignment]
    from .ppu.quant_config import (
        DeepseekV4FP8ConfigPPU as DeepseekV4FP8Config,  # type: ignore[assignment]
    )

__all__ = [
    "DeepSeekV4MTP",
    "DeepseekV4FP8Config",
    "DeepseekV4ForCausalLM",
]
