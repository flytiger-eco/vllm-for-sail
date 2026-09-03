# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU-specific Kimi-K3 MTP (Multi-Token Prediction) draft model.

Reused verbatim from the NVIDIA implementation. The CUDA-only fast paths
(low-latency CuTe-DSL GEMM, MegaMoE) are already inert on PPU; see
``vllm.models.kimi_k3.ppu.model`` for details.
"""

from ..nvidia.mtp import KimiK3MTP

__all__ = ["KimiK3MTP"]
