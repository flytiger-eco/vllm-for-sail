# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue between the QSA indexer and FA3's CUDA query-sparse attention.

FA3's QSA path consumes a ``(total_q, 1, topk)`` table of *absolute* KV-pool token indices,
whereas the indexer emits request-relative logical positions padded with ``-1``. The
translation

    absolute = block_table[req, logical // page_size] * page_size + logical % page_size

is the same one the Triton path performs per column, lifted into one pass so FA3 can be
handed the table directly.

Padding is left to FA3 itself. The expansion kernel emits a row's usable selections as a
contiguous prefix, so the same pass counts them into ``seqused_k``, and the caller hands
every query row over as its own one-token batch (``cu_seqlens_q = arange(total_q + 1)``,
``max_seqlen_q = 1``). FA3 bounds the columns it walks by ``seqused_k`` and the ``-1`` tail
never enters the softmax -- no anchor redirection and no output correction pass.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def _qsa_fa3_prepare_table_kernel(
    logical_indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    page_table_ptr,
    seqused_k_ptr,
    stride_logical_row,
    stride_block_table_req,
    stride_page_table_row,
    num_requests,
    num_cache_blocks,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    """Translate one row of logical selections into absolute KV-pool indices.

    Columns whose logical entry is ``-1`` (or resolves to a stale block-table slot) stay
    negative: ``seqused_k`` already keeps them out of FA3's walk, and the per-column load
    skips a negative entry should one ever be read.
    """
    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row)
    request_live = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    offsets = tl.arange(0, COLUMN_BLOCK)

    valid = 0
    for start in tl.static_range(0, SELECTION_WIDTH, COLUMN_BLOCK):
        columns = start + offsets
        in_range = columns < SELECTION_WIDTH
        logical = tl.load(
            logical_indices_ptr + row * stride_logical_row + columns,
            mask=in_range & request_live,
            other=-1,
        )
        safe_logical = tl.maximum(logical, 0)
        page_rank = safe_logical // PAGE_SIZE
        page = tl.load(
            block_table_ptr
            + safe_request * stride_block_table_req
            + tl.minimum(page_rank, PAGE_TABLE_WIDTH - 1),
            mask=in_range & request_live & (logical >= 0),
            other=-1,
        )
        usable = (
            in_range
            & request_live
            & (logical >= 0)
            & (page_rank < PAGE_TABLE_WIDTH)
            & (page >= 0)
            & (page < num_cache_blocks)
        )
        valid += tl.sum(tl.where(usable, 1, 0))
        absolute = page * PAGE_SIZE + safe_logical % PAGE_SIZE
        tl.store(
            page_table_ptr + row * stride_page_table_row + columns,
            tl.where(usable, absolute, -1),
            mask=in_range,
        )

    tl.store(seqused_k_ptr + row, valid)


def qsa_fa3_prepare_page_table(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    page_size: int,
    num_cache_blocks: int,
    column_block: int = 512,
) -> None:
    """Fill ``page_table`` with the absolute indices FA3's QSA path expects.

    Args:
        logical_indices: ``(num_tokens, selection_width)`` int32, request-relative, ``-1``
            padded. The usable entries of each row must form a contiguous prefix, which the
            indexer's expansion kernel guarantees.
        block_table: ``(num_requests, page_table_width)`` int32.
        token_to_req: ``(num_tokens,)`` int32.
        page_table: ``(num_tokens, selection_width)`` int32, output. Callers add the
            singleton dimension FA3 needs to recognise a QSA table.
        seqused_k: ``(num_tokens,)`` int32, output. Usable-column count per row, handed to
            FA3 so the padded tail stays out of the softmax.
        page_size: tokens per KV-cache block.
        num_cache_blocks: blocks in the KV pool, used to reject stale block-table entries.
    """
    if not HAS_TRITON:
        raise RuntimeError("FA3 QSA page-table preparation requires Triton")
    num_tokens = logical_indices.shape[0]
    if num_tokens == 0:
        return
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert page_table.dtype == seqused_k.dtype == torch.int32
    assert page_table.shape == logical_indices.shape
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert page_table.stride(1) == 1
    assert seqused_k.shape == (num_tokens,)

    _qsa_fa3_prepare_table_kernel[(num_tokens,)](
        logical_indices,
        block_table,
        token_to_req,
        page_table,
        seqused_k,
        logical_indices.stride(0),
        block_table.stride(0),
        page_table.stride(0),
        block_table.shape[0],
        num_cache_blocks,
        PAGE_SIZE=page_size,
        PAGE_TABLE_WIDTH=block_table.shape[1],
        SELECTION_WIDTH=logical_indices.shape[1],
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )


__all__ = [
    "qsa_fa3_prepare_page_table",
]
