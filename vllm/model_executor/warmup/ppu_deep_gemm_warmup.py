# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup deep_gemm kernels.
DeepGEMM JIT's the kernels. The warmup aims to JIT all the kernels that would
be used during model execution beforehand.
"""

import time

import torch
from tqdm import tqdm

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_node_count,
    get_world_group,
    is_global_first_rank,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.ppu_deep_gemm_moe import PPUDeepGemmExperts
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEModularMethod
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.tracing import instrument
from vllm.utils.ppu_deep_gemm import (
    fp8_gemm_nt,
    get_compile_mode,
    get_deep_gemm_config,
    get_mk_alignment_for_contiguous_layout,
    int8_gemm_nt,
    m_grouped_bf16_gemm_nt_nopad,
    m_grouped_fp8_gemm_nt_nopad,
    m_grouped_int8_gemm_nt_nopad,
    set_compile_mode,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import num_compute_units

logger = init_logger(__name__)


_LOCAL_RANK_INFO: tuple[int, int] | None = None


def _get_local_rank_and_size() -> tuple[int, int]:
    global _LOCAL_RANK_INFO
    if _LOCAL_RANK_INFO is not None:
        return _LOCAL_RANK_INFO
    try:
        world = get_world_group()
        local_rank = world.local_rank
        # NOTE: assume even node layout
        node_count = get_node_count()
        if node_count > 0 and world.world_size % node_count == 0:
            _LOCAL_RANK_INFO = (local_rank, world.world_size // node_count)
    except Exception:
        _LOCAL_RANK_INFO = (0, 1)
    return _LOCAL_RANK_INFO


def _shard_m_values_chunked(m_values: list[int]) -> list[int]:
    rank, world_size = _get_local_rank_and_size()
    if world_size <= 1 or not m_values:
        return m_values
    chunk = int(len(m_values) // world_size)
    start = rank * chunk
    end = min(len(m_values), start + chunk)
    return m_values[start:end]


# PPU NOTE: need align with ppu deep gemm if on PPU
def _generate_optimal_warmup_m_values(
    max_tokens: int, n: int, device: torch.device
) -> list[int]:
    """
    Generate M values that cover all possible DeepGEMM kernel configurations.
    Reference: https://github.com/deepseek-ai/DeepGEMM/blob/79f48ee15a82dd5fad5cd9beaa393c1f755e6b55/csrc/jit_kernels/heuristics/common.hpp

    Args:
        max_tokens: Maximum number of tokens to warmup for
        n: The actual N dimension from the weight tensor
        device: The torch device to get properties from.
    """

    # DeepGEMM's possible block sizes
    block_ms = [64, 128, 256]
    block_ns = list(range(16, min(257, n + 1), 16))
    num_sms = num_compute_units(device.index)

    m_values = set()

    # Always include small cases
    m_values.update([1, 2, 4] + [i for i in range(8, 65, 8)])

    # Collect M values where different wave patterns occur
    for block_m in block_ms:
        for block_n in block_ns:
            if block_n > n:
                continue

            # Add key M boundaries for this block combination
            for wave in range(1, 11):  # Up to 10 waves
                # M where this block config transitions to next wave
                target_blocks = wave * num_sms
                m = target_blocks * block_m // cdiv(n, block_n)
                if 1 <= m <= max_tokens:
                    m_values.add(m)

            # Add block_m boundaries
            for multiple in range(1, max_tokens // block_m + 1):
                m = multiple * block_m
                if m <= max_tokens:
                    m_values.add(m)

    return sorted(m_values)


def _extract_data_from_linear_base_module(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Extract weights, weight scales and quantization block sizes from the given
    LinearBase module.
    """
    assert isinstance(m, LinearBase)
    if isinstance(m.quant_method, Fp8LinearMethod):
        assert m.quant_method.block_quant
        assert m.quant_method.quant_config is not None

        w = m.weight
        ws = m.weight_scale_inv if hasattr(m, "weight_scale_inv") else m.weight_scale
        quant_block_size = m.quant_method.quant_config.weight_block_size

        assert isinstance(w, torch.Tensor)
        assert isinstance(ws, torch.Tensor)
        assert quant_block_size is not None
        return (w, ws, quant_block_size)
    elif isinstance(m.quant_method, CompressedTensorsLinearMethod):
        w = m.weight
        ws = m.weight_scale_inv if hasattr(m, "weight_scale_inv") else m.weight_scale
        assert isinstance(w, torch.Tensor)
        assert isinstance(ws, torch.Tensor)
        quant_block_size = 1
        return (w, ws, quant_block_size)
    else:
        raise RuntimeError("Unsupported quant_method for current platform")


def _extract_data_from_fused_moe_module(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Extract weights, weight scales and num_topk from FusedMoE module.
    """
    assert isinstance(m, FusedMoE)
    w13 = m.w13_weight
    w2 = m.w2_weight
    if w13.dtype in [torch.bfloat16, torch.float16]:
        assert isinstance(w13, torch.Tensor)
        assert isinstance(w2, torch.Tensor)
        w13_s = None
        w2_s = None
    elif w13.dtype in [torch.int8, torch.uint8, torch.float8_e4m3fn]:
        w13_s = (
            m.w13_weight_scale_inv
            if hasattr(m, "w13_weight_scale_inv")
            else m.w13_weight_scale
        )
        w2_s = (
            m.w2_weight_scale_inv
            if hasattr(m, "w2_weight_scale_inv")
            else m.w2_weight_scale
        )

        assert isinstance(w13, torch.Tensor)
        assert isinstance(w13_s, torch.Tensor)
        assert isinstance(w2, torch.Tensor)
        assert isinstance(w2_s, torch.Tensor)
    else:
        raise RuntimeError

    num_topk = m.top_k
    return w13, w13_s, w2, w2_s, num_topk


def _fp8_linear_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    """
    Return True if the input module/layer could be processed with DeepGEMM.
    """
    if envs.VLLM_PPU_DENSE_BACKEND and envs.VLLM_PPU_DENSE_BACKEND != "deepgemm":
        return False

    # FIXME: this logic is brittle and incorrect - since we
    # could use DeepGEMM with for than just Fp8LinearMethod
    block_size = get_mk_alignment_for_contiguous_layout()[0]
    if not (
        isinstance(module, LinearBase)
        and isinstance(module.quant_method, Fp8LinearMethod)
        and module.quant_method.block_quant
        and not module.quant_method.use_marlin
    ):
        return False

    w, _, block_sizes = _extract_data_from_linear_base_module(module)

    block_sizes_valid = (
        block_sizes[1] == get_mk_alignment_for_contiguous_layout()[1]
    )

    return (
        block_sizes_valid
        and w.ndim == 2
        and w.shape[0] % block_size == 0
        and w.shape[1] % block_size == 0
    )


def _fused_moe_grouped_gemm_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    if envs.VLLM_PPU_MOE_BACKEND and envs.VLLM_PPU_MOE_BACKEND != "deepgemm":
        return False

    if not isinstance(module, FusedMoE):
        return False

    moe_quant_config = module.quant_method.get_fused_moe_quant_config(module)

    if (
        moe_quant_config is None
        or moe_quant_config.quant_dtype != torch.float8_e4m3fn
        or (
            moe_quant_config.block_shape
            and moe_quant_config.block_shape[1]
            != get_mk_alignment_for_contiguous_layout()[1]
        )
    ):
        return False

    if not isinstance(module.quant_method, FusedMoEModularMethod):
        # modular kernels could invoke deep_gemm_moe_fp8
        return True

    # Further check if the ModularKernel implementation uses the DeepGemmExperts
    return isinstance(
        module.quant_method.moe_kernel, (PPUDeepGemmExperts)
    )


FP8_GEMM_NT_WARMUP_CACHE: set[torch.Size] = set()


def _get_gemm_nt_m_values(w: torch.Tensor, max_tokens: int) -> list[int]:
    """Get the M values to warmup for a given weight tensor."""
    n, _ = w.size()
    device = w.device

    # Use optimal M values only if VLLM_DEEP_GEMM_WARMUP is set to "relax".
    # Otherwise warmup all token sizes to avoid JIT compilation in hotpath
    if envs.VLLM_DEEP_GEMM_WARMUP == "relax":
        return _generate_optimal_warmup_m_values(max_tokens, n, device)
    else:
        assert envs.VLLM_DEEP_GEMM_WARMUP == "full", (
            "Expected "
            'VLLM_DEEP_GEMM_WARMUP env to be set to "full" but got '
            f"{envs.VLLM_DEEP_GEMM_WARMUP}"
        )
        return list(range(1, max_tokens + 1))


def _deepgemm_fp8_gemm_nt_warmup(
    w: torch.Tensor,
    ws: torch.Tensor,
    max_tokens: int,
    pbar: tqdm | None = None,
):
    if w.size() in FP8_GEMM_NT_WARMUP_CACHE:
        return

    n, k = w.size()
    _, block_k = get_mk_alignment_for_contiguous_layout()

    device = w.device
    a1q = torch.empty((max_tokens, k), device=device, dtype=torch.float8_e4m3fn)
    a1q_scales = torch.empty(
        (max_tokens, k // block_k), device=device, dtype=torch.float32
    )
    out = torch.empty((max_tokens, n), device=device, dtype=torch.bfloat16)

    m_values = _get_gemm_nt_m_values(w, max_tokens)
    m_values = _shard_m_values_chunked(m_values)

    for num_tokens in m_values:
        fp8_gemm_nt(
            (a1q[:num_tokens], a1q_scales[:num_tokens]), (w, ws), out[:num_tokens]
        )
        if pbar is not None:
            pbar.update(1)

    FP8_GEMM_NT_WARMUP_CACHE.add(w.size())


GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE: set[torch.Size] = set()


def _get_grouped_gemm_params(
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_topk: int,
    max_tokens: int,
) -> tuple[int, int, torch.Tensor]:
    assert w1.size(0) == w2.size(0), "w1 and w2 must have the same number of experts"

    block_m, block_k = get_mk_alignment_for_contiguous_layout()
    num_experts = w1.size(0)
    device = w1.device

    # Assumes all ranks have the same max_num_batched_tokens
    max_tokens = get_dp_group().world_size * max_tokens

    # This is the maximum GroupedGemm M size that we expect to run
    # the grouped_gemm with.
    MAX_M = compute_aligned_M(
        max_tokens, num_topk, num_experts, block_m, expert_tokens_meta=None
    )
    # Distribute expert-ids evenly.
    MAX_BLOCKS = MAX_M // block_m
    expert_ids_block = torch.randint(
        low=0, high=num_experts, size=(MAX_BLOCKS,), device=device, dtype=torch.int32
    )
    expert_ids = torch.repeat_interleave(expert_ids_block, block_m, dim=0)

    return MAX_M, block_m, expert_ids, block_k


def _deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_topk: int,
    max_tokens: int,
    pbar: tqdm | None = None,
):
    if (
        w1.size() in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
        and w2.size() in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    ):
        return

    MAX_M, block_m, expert_ids, block_k = _get_grouped_gemm_params(
        w1, w2, num_topk, max_tokens
    )
    device = w1.device

    def _warmup(w: torch.Tensor, w_scale: torch.Tensor):
        _, n, k = w.size()
        a1q = torch.empty((MAX_M, k), device=device, dtype=torch.float8_e4m3fn)
        if block_k == 1:
            # channel-wise
            a1q_scales = torch.empty((MAX_M, 1), device=device, dtype=torch.float32)
        else:
            # block-wise
            a1q_scales = torch.empty(
                (MAX_M, k // block_k), device=device, dtype=torch.float32
            )
        out = torch.empty((MAX_M, n), device=device, dtype=torch.bfloat16)

        m_values = list(range(block_m, MAX_M + 1, block_m))
        m_values = _shard_m_values_chunked(m_values)

        for num_tokens in m_values:
            m_grouped_fp8_gemm_nt_nopad(
                (a1q[:num_tokens], a1q_scales[:num_tokens]),
                (w, w_scale),
                out[:num_tokens],
                expert_ids[:num_tokens],
            )

            if pbar is not None:
                pbar.update(1)

    for w, ws in [(w1, w1_scale), (w2, w2_scale)]:
        if w.size() not in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE:
            _warmup(w, ws)
            GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE.add(w.size())


def deepgemm_fp8_gemm_nt_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
):
    dg_modules = [m for m in model.modules() if _fp8_linear_may_use_deep_gemm(m)]

    for dgm in dg_modules:
        w, ws, _ = _extract_data_from_linear_base_module(dgm)
        _deepgemm_fp8_gemm_nt_warmup(w=w, ws=ws, max_tokens=max_tokens, pbar=pbar)


def deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
):
    dg_modules = [
        m for m in model.modules() if _fused_moe_grouped_gemm_may_use_deep_gemm(m)
    ]

    for dgm in dg_modules:
        w13, w13_scale, w2, w2_scale, num_topk = _extract_data_from_fused_moe_module(
            dgm
        )
        _deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
            w13, w2, w13_scale, w2_scale, num_topk, max_tokens, pbar=pbar
        )


def _int8_linear_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    """
    Return True if the input module/layer could be processed with DeepGEMM.
    """
    if envs.VLLM_PPU_DENSE_BACKEND and envs.VLLM_PPU_DENSE_BACKEND != "deepgemm":
        return False

    if not (
        isinstance(module, LinearBase)
        and isinstance(module.quant_method, CompressedTensorsLinearMethod)
        and getattr(module.quant_method.quantization_config, "quant_format", None) == "int-quantized"
    ):
        return False

    w, _, _ = _extract_data_from_linear_base_module(module)

    return (
        has_deep_gemm()
        and w.ndim == 2
    )


def _fused_moe_grouped_gemm_may_use_deep_gemm_int8(module: torch.nn.Module) -> bool:
    if envs.VLLM_PPU_MOE_BACKEND and envs.VLLM_PPU_MOE_BACKEND != "deepgemm":
        return False

    if not isinstance(module, FusedMoE):
        return False

    moe_quant_config = module.quant_method.get_fused_moe_quant_config(module)

    if moe_quant_config is None or moe_quant_config.quant_dtype != torch.int8:
        return False

    if not isinstance(module.quant_method, FusedMoEModularMethod):
        # modular kernels could invoke deep_gemm_moe_int8
        return True

    mk: FusedMoEKernel = module.quant_method.moe_kernel
    # Further check if the ModularKernel implementation uses the DeepGemmExperts
    return isinstance(mk.fused_experts, (PPUDeepGemmExperts))


INT8_GEMM_NT_WARMUP_CACHE: set[torch.Size] = set()


def _deepgemm_int8_gemm_nt_warmup(
    w: torch.Tensor,
    ws: torch.Tensor,
    max_tokens: int,
    pbar: tqdm | None = None,
):
    if w.size() in INT8_GEMM_NT_WARMUP_CACHE:
        return

    n, k = w.size()
    block_m = get_mk_alignment_for_contiguous_layout()[0]

    device = w.device
    a1q = torch.empty((max_tokens, k), device=device, dtype=torch.int8)
    a1q_scales = torch.empty(
        (max_tokens, k // block_m), device=device, dtype=torch.float32
    )
    out = torch.empty((max_tokens, n), device=device, dtype=torch.bfloat16)

    m_values = _get_gemm_nt_m_values(w, max_tokens)
    m_values = _shard_m_values_chunked(m_values)

    for num_tokens in m_values:
        int8_gemm_nt(
            (a1q[:num_tokens], a1q_scales[:num_tokens]), (w, ws), out[:num_tokens]
        )
        if pbar is not None:
            pbar.update(1)

    INT8_GEMM_NT_WARMUP_CACHE.add(w.size())


GROUPED_INT8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE: set[torch.Size] = set()


def _deepgemm_grouped_int8_gemm_nt_contiguous_warmup(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_topk: int,
    max_tokens: int,
    pbar: tqdm | None = None,
):
    if (
        w1.size() in GROUPED_INT8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
        and w2.size() in GROUPED_INT8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    ):
        return

    MAX_M, block_m, expert_ids, _ = _get_grouped_gemm_params(
        w1, w2, num_topk, max_tokens
    )
    device = w1.device

    def _warmup(w: torch.Tensor, w_scale: torch.Tensor):
        e, n, k = w.size()
        # int8 w8a8 deepgemm use int8 input with per-token quant sacle
        a1q = torch.empty((MAX_M, k), device=device, dtype=torch.int8)
        a1q_scales = torch.empty((MAX_M, 1), device=device, dtype=torch.float32)
        out = torch.empty((MAX_M, n), device=device, dtype=torch.bfloat16)

        m_values = list(range(block_m, MAX_M + 1, block_m))
        m_values = _shard_m_values_chunked(m_values)

        for num_tokens in m_values:
            best_config = get_deep_gemm_config(num_tokens, n, k, num_groups=e)
            m_grouped_int8_gemm_nt_nopad(
                (a1q[:num_tokens], a1q_scales[:num_tokens]),
                (w, w_scale),
                out[:num_tokens],
                expert_ids[:num_tokens],
                None,
                best_config,
            )
            if pbar is not None:
                pbar.update(1)

    for w, ws in [(w1, w1_scale), (w2, w2_scale)]:
        if w.size() not in GROUPED_INT8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE:
            _warmup(w, ws)
            GROUPED_INT8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE.add(w.size())


def deepgemm_int8_gemm_nt_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
):
    dg_modules = [m for m in model.modules() if _int8_linear_may_use_deep_gemm(m)]

    for dgm in dg_modules:
        w, ws, _ = _extract_data_from_linear_base_module(dgm)
        _deepgemm_int8_gemm_nt_warmup(w=w, ws=ws, max_tokens=max_tokens, pbar=pbar)


def deepgemm_grouped_int8_gemm_nt_contiguous_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
):
    dg_modules = [
        m for m in model.modules() if _fused_moe_grouped_gemm_may_use_deep_gemm_int8(m)
    ]
    for dgm in dg_modules:
        w13, w13_scale, w2, w2_scale, num_topk = _extract_data_from_fused_moe_module(
            dgm
        )
        _deepgemm_grouped_int8_gemm_nt_contiguous_warmup(
            w13, w2, w13_scale, w2_scale, num_topk, max_tokens, pbar=pbar
        )


def _fused_moe_grouped_gemm_may_use_deep_gemm_bf16(module: torch.nn.Module) -> bool:
    if envs.VLLM_PPU_MOE_BACKEND and envs.VLLM_PPU_MOE_BACKEND != "deepgemm":
        return False

    if not isinstance(module, FusedMoE):
        return False

    moe_quant_config = module.quant_method.get_fused_moe_quant_config(module)

    if moe_quant_config is None or moe_quant_config.quant_dtype not in [
        torch.bfloat16,
        torch.float16,
        None,
    ]:
        return False

    # Note: ref vllm/model_executor/layers/fused_moe/config.py
    # moe_quant_config.quant_dtype means unquantized or is already quantized.
    # quant dtype return _a1.dtype. So we need more check here for wna16.
    if (
        moe_quant_config is None
        or moe_quant_config.use_int8_w8a16
        or moe_quant_config.use_int4_w4a16
        or moe_quant_config.use_fp8_w8a16
        or moe_quant_config.use_nvfp4_w4a16
        or moe_quant_config.use_mxfp4_w4a16
    ):
        return False

    if not isinstance(module.quant_method, FusedMoEModularMethod):
        # modular kernels could invoke deep_gemm_moe_bf16
        return True

    mk: FusedMoEKernel = module.quant_method.moe_kernel
    # Further check if the ModularKernel implementation uses the DeepGemmExperts
    return isinstance(mk.fused_experts, (PPUDeepGemmExperts))


GROUPED_BF16_GEMM_NT_CONTIGUOUS_WARMUP_CACHE: set[torch.Size] = set()


def _deepgemm_grouped_bf16_gemm_nt_contiguous_warmup(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_topk: int,
    max_tokens: int,
    pbar: tqdm | None = None,
):
    if (
        w1.size() in GROUPED_BF16_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
        and w2.size() in GROUPED_BF16_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    ):
        return

    MAX_M, block_m, expert_ids, _ = _get_grouped_gemm_params(
        w1, w2, num_topk, max_tokens
    )
    device = w1.device

    def _warmup(w: torch.Tensor, w_scale: torch.Tensor):
        _, n, k = w.size()
        a1q = torch.empty((MAX_M, k), device=device, dtype=torch.bfloat16)
        out = torch.empty((MAX_M, n), device=device, dtype=torch.bfloat16)

        # Generate M values in block_m increments (already optimized for MoE)
        m_values = list(range(block_m, MAX_M + 1, block_m))
        m_values = _shard_m_values_chunked(m_values)

        for num_tokens in m_values:
            m_grouped_bf16_gemm_nt_nopad(
                a1q[:num_tokens], w, out[:num_tokens], expert_ids[:num_tokens]
            )
            if pbar is not None:
                pbar.update(1)

    for w, ws in [(w1, w1_scale), (w2, w2_scale)]:
        if w.size() not in GROUPED_BF16_GEMM_NT_CONTIGUOUS_WARMUP_CACHE:
            _warmup(w, ws)
            GROUPED_BF16_GEMM_NT_CONTIGUOUS_WARMUP_CACHE.add(w.size())


def deepgemm_grouped_bf16_gemm_nt_contiguous_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
):
    dg_modules = [
        m for m in model.modules() if _fused_moe_grouped_gemm_may_use_deep_gemm_bf16(m)
    ]
    for dgm in dg_modules:
        w13, w13_scale, w2, w2_scale, num_topk = _extract_data_from_fused_moe_module(
            dgm
        )
        _deepgemm_grouped_bf16_gemm_nt_contiguous_warmup(
            w13, w2, w13_scale, w2_scale, num_topk, max_tokens, pbar=pbar
        )


def _count_warmup_iterations(model: torch.nn.Module, max_tokens: int) -> int:
    seen_fp8_sizes: set[torch.Size] = set(FP8_GEMM_NT_WARMUP_CACHE)
    seen_grouped_sizes: set[torch.Size] = set(
        GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    )

    total = 0
    for m in model.modules():
        if _fp8_linear_may_use_deep_gemm(m):
            w, _, _ = _extract_data_from_linear_base_module(m)
            if w.size() not in seen_fp8_sizes:
                total += len(_get_gemm_nt_m_values(w, max_tokens))
                seen_fp8_sizes.add(w.size())
        elif _fused_moe_grouped_gemm_may_use_deep_gemm(m):
            w13, _, w2, _, num_topk = _extract_data_from_fused_moe_module(m)
            if w13.size() in seen_grouped_sizes and w2.size() in seen_grouped_sizes:
                continue
            MAX_M, block_m, _, _ = _get_grouped_gemm_params(
                w13, w2, num_topk, max_tokens
            )
            n_values = (MAX_M - block_m) // block_m + 1
            if w13.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w13.size())
            if w2.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w2.size())
    return total


def _count_warmup_iterations_int8(model: torch.nn.Module, max_tokens: int) -> int:
    seen_int8_sizes: set[torch.Size] = set(INT8_GEMM_NT_WARMUP_CACHE)
    seen_grouped_sizes: set[torch.Size] = set(
        GROUPED_INT8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    )

    total = 0
    for m in model.modules():
        if _int8_linear_may_use_deep_gemm(m):
            w, _, _ = _extract_data_from_linear_base_module(m)
            if w.size() not in seen_int8_sizes:
                total += len(_get_gemm_nt_m_values(w, max_tokens))
                seen_int8_sizes.add(w.size())
        elif _fused_moe_grouped_gemm_may_use_deep_gemm_int8(m):
            w13, _, w2, _, num_topk = _extract_data_from_fused_moe_module(m)
            if w13.size() in seen_grouped_sizes and w2.size() in seen_grouped_sizes:
                continue
            MAX_M, block_m, _, _ = _get_grouped_gemm_params(
                w13, w2, num_topk, max_tokens
            )
            n_values = (MAX_M - block_m) // block_m + 1
            if w13.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w13.size())
            if w2.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w2.size())
    return total


def _count_warmup_iterations_bf16(model: torch.nn.Module, max_tokens: int) -> int:
    seen_grouped_sizes: set[torch.Size] = set(
        GROUPED_BF16_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    )

    total = 0
    for m in model.modules():
        if _fused_moe_grouped_gemm_may_use_deep_gemm_bf16(m):
            w13, _, w2, _, num_topk = _extract_data_from_fused_moe_module(m)
            if w13.size() in seen_grouped_sizes and w2.size() in seen_grouped_sizes:
                continue
            MAX_M, block_m, _, _ = _get_grouped_gemm_params(
                w13, w2, num_topk, max_tokens
            )
            n_values = (MAX_M - block_m) // block_m + 1
            if w13.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w13.size())
            if w2.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w2.size())
    return total


@instrument(span_name="DeepGemm warmup")
def deep_gemm_warmup(model: torch.nn.Module, max_tokens: int):
    total = _count_warmup_iterations(model, max_tokens)
    total_int8 = _count_warmup_iterations_int8(model, max_tokens)
    total_bf16 = _count_warmup_iterations_bf16(model, max_tokens)
    if total == 0 and total_int8 == 0 and total_bf16 == 0:
        return

    start = time.time()
    old_compile_mode = get_compile_mode()
    set_compile_mode(1)

    # Only show progress bar on rank 0 to avoid cluttered output
    if is_global_first_rank():
        with tqdm(total=total, desc="DeepGEMM warmup") as pbar:
            if total:
                deepgemm_fp8_gemm_nt_warmup(model, max_tokens, pbar)
                deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(model, max_tokens, pbar)
            if total_int8:
                deepgemm_int8_gemm_nt_warmup(model, max_tokens, pbar)
                deepgemm_grouped_int8_gemm_nt_contiguous_warmup(model, max_tokens, pbar)
            if total_bf16:
                deepgemm_grouped_bf16_gemm_nt_contiguous_warmup(model, max_tokens, pbar)

    else:
        if total:
            deepgemm_fp8_gemm_nt_warmup(model, max_tokens, None)
            deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(model, max_tokens, None)
        if total_int8:
            deepgemm_int8_gemm_nt_warmup(model, max_tokens, None)
            deepgemm_grouped_int8_gemm_nt_contiguous_warmup(model, max_tokens, None)
        if total_bf16:
            deepgemm_grouped_bf16_gemm_nt_contiguous_warmup(model, max_tokens, None)
    set_compile_mode(old_compile_mode)
    elapsed = time.time() - start
    logger.info(f"DeepGemm warmup elapsed time: {elapsed:.6f}s")  # noqa
