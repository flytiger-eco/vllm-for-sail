# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU SAIL CUDA PLA kernel integration for GDN prefill.

Prefill counterpart of :mod:`sail_cuda_pla` (which routes GDN *decode* to the
PPU SAIL CUDA ``k_last`` / ``k_last_packed`` kernels). When enabled, the GDN
prefill computation is routed from the community Triton/FLA chunk kernels to
the FlashQLA CUDA kernel provided by the external ``pla`` package:

* ``pla.prefill.flashqla.chunk_gated_delta_rule_fwd``

The routing is controlled by the ``VLLM_PPU_USE_PLA`` environment variable
(default: on) -- the same switch that gates the decode PLA kernels -- and is
only effective when all of the following hold:

* running on the PPU platform (``current_platform.is_ppu()``);
* the ``pla`` package is importable;
* the per-model constraints of the FlashQLA kernel are satisfied (checked by
  the caller): head dims K == V == 128, and the TP-sharded
  ``(num_v_heads, num_k_heads)`` pair present in ``SUPPORTED_HEAD_CONFIGS``.

Unlike the decode resolver, a missing ``pla`` package is *not* an error here:
prefill has a fully functional Triton/FLA fallback, so the resolver logs a
warning and leaves the kernel handle unset rather than breaking deployments
that do not ship ``pla``.

Note the state-layout difference the caller must bridge: vLLM/FLA store the
ssm state as ``[N, H, DV, DK]`` while FlashQLA takes ``initial_state`` as
``[batch, H, DK, DV]`` fp32 and returns the final state in the same layout.
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
_chunk_fwd_fn: Callable[..., Any] | None = None
_supported_head_configs: frozenset[tuple[int, int]] = frozenset()


def _resolve_sail_cuda_pla_prefill() -> None:
    """Resolve the PPU SAIL CUDA PLA prefill kernel from ``pla`` once.

    Thread-safe: guarded by a module-level lock so concurrent calls from
    multiple threads resolve exactly once.
    """
    global _resolved, _chunk_fwd_fn, _supported_head_configs
    # Fast path: after the first resolution, skip the lock entirely.
    if _resolved:
        return
    with _lock:
        if _resolved:
            return
        _resolved = True

        if not envs.VLLM_PPU_USE_PLA:
            logger.warning_once(
                "VLLM_PPU_USE_PLA is disabled; GDN prefill falls back to the "
                "community Triton/FLA chunk kernels. WARNING: using the "
                "Triton implementation will result in SIGNIFICANT performance "
                "degradation on PPU. Install the PLA library and remove the "
                "VLLM_PPU_USE_PLA=0 setting to restore full performance."
            )
            return
        if not current_platform.is_ppu():
            return

        try:
            from pla.prefill.flashqla import chunk_gated_delta_rule_fwd
            from pla.prefill.flashqla.ops import SUPPORTED_HEAD_CONFIGS
        except ImportError:
            logger.warning_once(
                "VLLM_PPU_USE_PLA is enabled but the PPU `pla` prefill package "
                "is not importable; GDN prefill falls back to the community "
                "Triton/FLA chunk kernels. Build it with "
                "`PLA_BUILD_NAMESPACE=pla.prefill pip install .` to restore "
                "full performance."
            )
            return

        _chunk_fwd_fn = chunk_gated_delta_rule_fwd
        _supported_head_configs = SUPPORTED_HEAD_CONFIGS
        logger.info_once(
            "VLLM_PPU_USE_PLA is enabled: routing GDN prefill to the PPU SAIL "
            "CUDA PLA FlashQLA kernel (chunk_gated_delta_rule_fwd)."
        )


def get_sail_cuda_pla_prefill_fwd() -> Callable[..., Any] | None:
    """Return ``chunk_gated_delta_rule_fwd`` if usable, else ``None``."""
    _resolve_sail_cuda_pla_prefill()
    return _chunk_fwd_fn


def get_sail_cuda_pla_prefill_head_configs() -> frozenset[tuple[int, int]]:
    """Return FlashQLA's supported ``(num_v_heads, num_k_heads)`` whitelist.

    Empty when the PLA prefill kernel is unavailable, so membership tests
    against it naturally reject every configuration.
    """
    _resolve_sail_cuda_pla_prefill()
    return _supported_head_configs
