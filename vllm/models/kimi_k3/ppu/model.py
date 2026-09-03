# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific Kimi-K3 model.

PPU is CUDA-compatible, so the K3 text/vision stack is reused verbatim from the
NVIDIA implementation. This module is the platform-isolation seam (mirrors
``vllm.models.deepseek_v4.ppu``) and the place for any future PPU-only
weight-mapping overrides.

The CUDA-only fast paths are already inert on PPU:

- MegaMoE is gated by ``kernel_config.moe_backend == "deep_gemm_mega_moe"``,
  which ``PPUPlatform.apply_config_platform_defaults`` rejects; the default
  ``FusedMoE`` + ``LatentMoERunner`` path runs instead.
- ``enable_kimi_k3_low_latency_gemm`` (CuTe-DSL skinny GEMM) returns early off
  SM103, so it is a no-op on PPU.
- FlashKDA is arch-gated off; ``resolve_kda_prefill_backend`` falls back to the
  Triton KDA prefill path.
"""

from ..nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
)

__all__ = [
    "KimiK3ForConditionalGeneration",
    "KimiLinearForCausalLM",
]
