# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU SAIL CUDA PLA kernel integration for GDN decode.

Mirrors SGLang's ``SGLANG_SAIL_PLA_CUDA`` mechanism (sglang commit
b7d85d39bd9e3a4165c068a15314629647e8b393, "feat(pla): support ppu cuda pla
kernel"): when enabled, the GDN decode PLA computations are routed from the
community Triton kernels to the PPU SAIL CUDA kernels provided by the
external ``pla`` package (Fused Sigmoid Gating Delta Rule CUDA kernels):

* ``fused_sigmoid_gating_delta_rule_forward_k_last``
* ``fused_sigmoid_gating_delta_rule_forward_k_last_packed``

The routing is controlled by the ``VLLM_PPU_USE_PLA`` environment variable
(default: on) and is only effective when all of the
following hold:

* running on the PPU platform (``current_platform.is_ppu()``);
* the ``pla`` package is importable;
* the per-call tensor constraints of the CUDA kernels are satisfied
  (checked at the call sites): float32 ssm state pool, int32 state
  indices, head dims K == V == 128, and -- because vLLM reserves slot 0
  as NULL_BLOCK_ID (unlike sglang's PAD_SLOT_ID = -1) -- every call passes
  ``is_sglang=False``.

If ``VLLM_PPU_USE_PLA`` is enabled (the default) but the ``pla`` package
cannot be imported on the PPU platform, an ``ImportError`` is raised so
that the caller can decide how to handle the missing dependency (e.g.
fall back to Triton, or surface the error to the user).
"""

import threading
from collections.abc import Callable
from typing import Any

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_lock = threading.Lock()
_resolved: bool = False
_k_last_fn: Callable[..., Any] | None = None
_k_last_packed_fn: Callable[..., Any] | None = None


def _resolve_sail_cuda_pla() -> None:
    """Resolve the PPU SAIL CUDA PLA kernels from the ``pla`` package once.

    Thread-safe: guarded by a module-level lock so concurrent calls from
    multiple threads resolve exactly once.

    Raises ``ImportError`` when ``VLLM_PPU_USE_PLA`` is enabled on the PPU
    platform but the ``pla`` package is not installed.
    """
    global _resolved, _k_last_fn, _k_last_packed_fn
    # Fast path: after the first resolution, skip the lock entirely.
    if _resolved:
        return
    with _lock:
        if _resolved:
            return
        _resolved = True

        if not envs.VLLM_PPU_USE_PLA:
            logger.warning_once(
                "VLLM_PPU_USE_PLA is disabled; falling back to the community "
                "Triton PLA kernels. WARNING: using the Triton implementation "
                "will result in SIGNIFICANT performance degradation. "
                "Install the PLA library and remove the VLLM_PPU_USE_PLA=0 "
                "setting to restore full performance."
            )
            return
        if not current_platform.is_ppu():
            return

        try:
            from pla.decode import (
                fused_sigmoid_gating_delta_rule_forward_k_last,
                fused_sigmoid_gating_delta_rule_forward_k_last_packed,
            )
        except ImportError:
            raise ImportError(
                "VLLM_PPU_USE_PLA is enabled but the PPU `pla` package "
                "is not installed. Install the PLA library or set "
                "VLLM_PPU_USE_PLA=0 to fall back to Triton."
            )

        _k_last_fn = fused_sigmoid_gating_delta_rule_forward_k_last
        _k_last_packed_fn = fused_sigmoid_gating_delta_rule_forward_k_last_packed
        logger.info_once(
            "VLLM_PPU_USE_PLA is enabled: routing GDN decode to the PPU SAIL "
            "CUDA PLA kernels (k_last / k_last_packed)."
        )


def get_sail_cuda_pla_k_last() -> Callable[..., Any] | None:
    """Return ``fused_sigmoid_gating_delta_rule_forward_k_last`` if usable."""
    _resolve_sail_cuda_pla()
    return _k_last_fn


def get_sail_cuda_pla_k_last_packed() -> Callable[..., Any] | None:
    """Return ``fused_sigmoid_gating_delta_rule_forward_k_last_packed``."""
    _resolve_sail_cuda_pla()
    return _k_last_packed_fn
