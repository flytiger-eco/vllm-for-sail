# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU platform support. Code inside this file can safely assume the PPU
platform. Since PPU hardware is CUDA-compatible, this platform inherits 
from CudaPlatformBase and reuses all CUDA attention backends, 
distributed communication, compilation, and kernel infrastructure.
"""

from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .cuda import NvmlCudaPlatform
from .interface import DeviceCapability, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)

# pytorch 2.5 uses cudnn sdpa by default, which will cause crash on some models
# see https://github.com/huggingface/diffusers/issues/9704 for details
torch.backends.cuda.enable_cudnn_sdp(False)


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    if use_mla:
        return [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.TRITON_MLA,
            AttentionBackendEnum.FLASHMLA_SPARSE,
        ]
    else:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.FLEX_ATTENTION,
        ]


class PPUPlatform(NvmlCudaPlatform):
    _enum = PlatformEnum.PPU
    device_name: str = "ppu"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    ]

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", tuple[int, list[str]]],
    ]:
        valid_backends_priorities = []
        invalid_reasons: dict[AttentionBackendEnum, tuple[int, list[str]]] = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = (priority, invalid_reasons_i)
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        if cls.has_device_capability(80):
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.TRITON_ATTN,
            ]

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        from vllm._ppu_ops import ppu_ops  # noqa: F401 (side-effect: register ops)

        compilation_config = vllm_config.compilation_config
        attention_config = vllm_config.attention_config

        # PPU NOTE: set 0 to get better performance for fa3 on PPU 
        attention_config.flash_attn_max_num_splits_for_cuda_graph = 0

        # Default dispatch to ppu's sparse_attn_indexer implementation
        compilation_config.custom_ops.append("+sparse_attn_indexer")

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return False

    # -----------------------------------------------------------------
    # Everything below is inherited from NvmlCudaPlatform:
    #   - set_device, get_current_memory_usage
    #   - get_vit_attn_backend, get_supported_vit_attn_backends
    #   - get_punica_wrapper
    #   - get_device_communicator_cls (CudaCommunicator / NCCL)
    #   - supports_fp8, opaque_attention_op
    #   - get_static_graph_wrapper_cls (CUDAGraphWrapper)
    #   - stateless_init_device_torch_dist_pg (ProcessGroupNCCL)
    #   - device_count, check_if_supports_dtype
    #   - insert_blocks_to_device, swap_out_blocks_to_host
    #   - support_hybrid_kv_cache, support_static_graph_mode
    #   - num_compute_units, use_custom_op_collectives
    # -----------------------------------------------------------------