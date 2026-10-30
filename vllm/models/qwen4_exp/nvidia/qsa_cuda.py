# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp QSA backed by PPU's FA3 CUDA kernel.

This is a drop-in alternative to the Triton path in :mod:`.qsa`: same metadata builder, same KV
cache layout, same ``forward_qsa`` contract, so the two can be switched per layer at construction
time and compared logit-for-logit.

Two things FA3's QSA path needs that the Triton path did not:

* **Absolute indices.** FA3 is handed a ``(total_q, 1, topk)`` table of KV-pool token indices,
  while the indexer emits request-relative logical positions. The translation runs once per step
  in :func:`~.ops.qsa_fa3.qsa_fa3_prepare_page_table`.
* **Padding semantics.** FA3 masks only the columns past the *table width*, never a ``-1`` inside
  a row, so every padded column is redirected to an anchor token and removed from the result in
  closed form. See :mod:`.ops.qsa_fa3` for the derivation.
* **Build numerics.** FA3's AIU paged-KV load path fetches 16 contiguous tokens per
  16-column group of the table, so it silently attends to the wrong K/V for the
  scattered lists the indexer emits. The CUDA path therefore calls FA3 with
  ``qsa_allow_aiu=False`` on wheels that carry the argument (the kernel routes QSA
  onto the per-column load itself); older wheels get the same routing via a
  one-token-per-page view of the pool, applied inside
  :func:`~vllm.vllm_flash_attn.flash_attn_qsa_varlen_func`. That view needs one KV
  head per rank, which :func:`qsa_cuda_unsupported_reason` enforces on such wheels.
  :func:`vllm.vllm_flash_attn.is_fa3_qsa_supported` self-tests the calling convention
  before the CUDA path is selected.

The Triton path in :mod:`.qsa` stays as the fallback for builds and head shapes this kernel cannot
take; :func:`select_qsa_backend` decides between them once per layer.
"""

from __future__ import annotations

from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from .qsa import (
    Qwen4ExpQSAFlashAttentionBackend,
    Qwen4ExpQSAFlashAttentionImpl,
    Qwen4ExpQSAMetadataBuilder,
)

logger = init_logger(__name__)

# FA3 instantiates QSA for head dim 256 only, and rejects more than 32 query heads per KV head.
QSA_HEAD_DIM = 256
QSA_MAX_QHEADS_PER_KVHEAD = 32


def qsa_cuda_shape_reason(
    head_dim: int, num_heads: int, num_kv_heads: int
) -> str | None:
    """Return why FA3's QSA kernel rejects these head shapes, or ``None`` if it accepts them."""
    if head_dim != QSA_HEAD_DIM:
        return f"FA3 QSA is instantiated for head dim {QSA_HEAD_DIM} only, got {head_dim}"
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        return f"invalid GQA ratio: {num_heads} query heads over {num_kv_heads} KV heads"
    group_size = num_heads // num_kv_heads
    if group_size > QSA_MAX_QHEADS_PER_KVHEAD:
        return (
            f"FA3 QSA supports at most {QSA_MAX_QHEADS_PER_KVHEAD} query heads per KV head, "
            f"got {group_size}"
        )
    return None


def qsa_cuda_unsupported_reason(
    *,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    device: int | None = None,
) -> str | None:
    """Return why FA3's CUDA QSA cannot run this config, or ``None`` if it can.

    Ordered cheapest check first; the build probe is last because it issues a real kernel launch.

    Args:
        head_dim: main attention head dim.
        num_heads: / num_kv_heads: per-rank query / KV head counts.
    """
    reason = qsa_cuda_shape_reason(head_dim, num_heads, num_kv_heads)
    if reason is not None:
        return reason
    if not current_platform.is_ppu():
        return "FA3 QSA ships only in PPU's flash_attn_3 whl"
    # Past the is_ppu() gate the build is a PPU one, where setup.py has replaced
    # `vllm.vllm_flash_attn` with `vllm/vllm_flash_attn/ppu/`, so the QSA trio exists. It is
    # imported here rather than at module scope because :mod:`.qsa` imports this module
    # unconditionally, and a top-level import would make that an ImportError on every other build.
    from vllm.vllm_flash_attn import (
        fa3_qsa_unsupported_reason,
        fa3_qsa_uses_allow_aiu_flag,
        is_fa3_qsa_supported,
    )

    # Wheels predating qsa_allow_aiu are steered off the AIU bulk load through a
    # one-token-per-page view of the K/V split. That view merges the (block, token)
    # dims, which requires one KV head per rank: K and V interleave per token, so
    # with more the split view has no valid strides for the merge.
    if not fa3_qsa_uses_allow_aiu_flag() and num_kv_heads != 1:
        return (
            "the installed FA3 build lacks qsa_allow_aiu, and with "
            f"{num_kv_heads} KV heads per rank the one-token-per-page view that "
            "steers it off the AIU bulk load does not exist"
        )

    if not is_fa3_qsa_supported(device):
        return f"the installed FA3 build cannot run QSA ({fa3_qsa_unsupported_reason(device)})"
    return None


def select_qsa_backend(
    *,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[bool, str]:
    """Decide between the CUDA and Triton QSA paths for one layer config.

    Returns ``(use_cuda, explanation)``. The CUDA kernel is preferred whenever it can run
    this config, and Triton takes over when it cannot. Both outcomes are logged once per
    distinct message, since a model with dozens of QSA layers would otherwise repeat it
    dozens of times.
    """
    if not envs.VLLM_USE_CUDA_QSA:
        logger.info_once(
            "Qwen4Exp QSA is on the Triton kernel (VLLM_USE_CUDA_QSA=false)."
        )
        return False, "VLLM_USE_CUDA_QSA=false"
    reason = qsa_cuda_unsupported_reason(
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    if reason is None:
        logger.info_once("Qwen4Exp QSA is on PPU FA3's CUDA kernel.")
        return True, "FA3 CUDA QSA"
    logger.warning_once(
        "Qwen4Exp QSA falls back to the Triton kernel because %s. Set "
        "VLLM_USE_CUDA_QSA=false to silence this.",
        reason,
    )
    return False, f"Triton QSA ({reason})"


# Scratch buffers the CUDA path needs on top of the indexer's own top-k output. Registered on the
# layer as non-persistent buffers so they follow `.to(device)` like `topk_indices_buffer` does, and
# sized for max_num_batched_tokens so a forward pass allocates nothing and stays capturable in a
# CUDA graph.
QSA_FA3_PAGE_TABLE = "qsa_fa3_page_table"
QSA_FA3_ANCHOR = "qsa_fa3_anchor"
QSA_FA3_PADDING = "qsa_fa3_padding"
QSA_FA3_RECOMPUTE_FLAGS = "qsa_fa3_recompute_flags"


def register_qsa_cuda_buffers(
    layer: torch.nn.Module, max_tokens: int, selection_width: int, num_heads: int
) -> None:
    """Attach the CUDA path's scratch buffers to a QSA owner layer.

    ``qsa_fa3_page_table`` keeps the singleton dimension FA3 needs to recognise a QSA table; the
    correction kernels take a squeezed view of the same storage.
    """
    layer.register_buffer(
        QSA_FA3_PAGE_TABLE,
        torch.empty(max_tokens, 1, selection_width, dtype=torch.int32),
        persistent=False,
    )
    layer.register_buffer(
        QSA_FA3_ANCHOR,
        torch.empty(max_tokens, dtype=torch.int32),
        persistent=False,
    )
    layer.register_buffer(
        QSA_FA3_PADDING,
        torch.empty(max_tokens, dtype=torch.int32),
        persistent=False,
    )
    layer.register_buffer(
        QSA_FA3_RECOMPUTE_FLAGS,
        torch.empty(max_tokens, num_heads, dtype=torch.uint8),
        persistent=False,
    )


class Qwen4ExpQSACudaMetadataBuilder(Qwen4ExpQSAMetadataBuilder):
    """Identical metadata; the CUDA path only consumes it differently."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSACudaFlashAttentionBackend(Qwen4ExpQSAFlashAttentionBackend):
    """FullAttentionSpec backend running QSA on PPU's FA3 CUDA kernel."""

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_CUDA"

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSACudaFlashAttentionImpl]:
        return Qwen4ExpQSACudaFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSACudaMetadataBuilder]:
        return Qwen4ExpQSACudaMetadataBuilder


class Qwen4ExpQSACudaFlashAttentionImpl(Qwen4ExpQSAFlashAttentionImpl):
    """Run paged sparse GQA through FA3's CUDA QSA kernel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The layer gates the full config (platform, build probe) before it picks this impl;
        # re-check the half that belongs to the kernel itself so a hand-constructed impl
        # still fails at construction rather than mid-forward.
        reason = qsa_cuda_shape_reason(
            self.head_size, self.num_heads, self.num_kv_heads
        )
        if reason is not None:
            raise NotImplementedError(f"Qwen4Exp QSA CUDA path unavailable: {reason}")

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        page_table = getattr(layer, QSA_FA3_PAGE_TABLE, None)
        if page_table is None:
            raise RuntimeError(
                "QSA owner did not register the CUDA scratch buffers; "
                "register_qsa_cuda_buffers must run when the CUDA path is selected"
            )
        anchor = getattr(layer, QSA_FA3_ANCHOR)[:num_tokens]
        padding = getattr(layer, QSA_FA3_PADDING)[:num_tokens]
        recompute_flags = getattr(layer, QSA_FA3_RECOMPUTE_FLAGS)[:num_tokens]
        page_table = page_table[:num_tokens]

        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA requires BF16 Q/K/V")

        # Only reachable when select_qsa_backend cleared the is_ppu() gate, so the QSA entry
        # point exists; see the note in :func:`qsa_cuda_unsupported_reason`.
        from vllm.vllm_flash_attn import flash_attn_qsa_varlen_func

        from .ops.qsa_fa3 import qsa_fa3_prepare_page_table, qsa_fa3_remove_padding

        qsa_fa3_prepare_page_table(
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            page_table.squeeze(1),
            anchor,
            padding,
            key_cache.shape[1],
            key_cache.shape[0],
        )
        # qsa_allow_aiu=False routes QSA off the AIU paged-KV bulk load inside the
        # kernel -- it fetches 16 contiguous tokens per 16-column group of the table,
        # wrong for the scattered lists the indexer emits. On wheels predating the
        # argument the wrapper applies the one-token-per-page view instead; either
        # way each table entry is honored individually.
        # FA3 writes every row of `out`, so the padded tail of the buffer stays zeroed above.
        out, softmax_lse = flash_attn_qsa_varlen_func(
            query[:num_tokens],
            key_cache,
            value_cache,
            page_table,
            attn_metadata.query_start_loc,
            attn_metadata.max_query_len,
            softmax_scale=self.scale,
            out=output[:num_tokens],
            qsa_allow_aiu=False,
            return_softmax_lse=True,
        )
        # The correction kernels decode the table's absolute indices against the
        # PAGE_SIZE of the cache they are given, so the real (block, 16) pool works
        # here regardless of the layout the wrapper handed FA3.
        qsa_fa3_remove_padding(
            out,
            softmax_lse,
            query[:num_tokens],
            key_cache,
            value_cache,
            page_table.squeeze(1),
            anchor,
            padding,
            recompute_flags,
            self.scale,
        )
        return output


__all__ = [
    "QSA_FA3_ANCHOR",
    "QSA_FA3_PADDING",
    "QSA_FA3_PAGE_TABLE",
    "QSA_FA3_RECOMPUTE_FLAGS",
    "Qwen4ExpQSACudaFlashAttentionBackend",
    "Qwen4ExpQSACudaFlashAttentionImpl",
    "Qwen4ExpQSACudaMetadataBuilder",
    "qsa_cuda_shape_reason",
    "qsa_cuda_unsupported_reason",
    "register_qsa_cuda_buffers",
    "select_qsa_backend",
]
