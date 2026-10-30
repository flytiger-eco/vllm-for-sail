# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glue kernels between the QSA indexer and FA3's CUDA query-sparse attention.

FA3's QSA path consumes a ``(total_q, 1, topk)`` table of *absolute* KV-pool token indices,
whereas the indexer emits request-relative logical positions padded with ``-1``. Two gaps are
bridged here:

1. **Index space.** ``absolute = block_table[req, logical // page_size] * page_size
   + logical % page_size`` — the same translation the Triton path performs per column, lifted
   into one pass so FA3 can be handed the table directly.

2. **Padding.** FA3 derives ``seqlen_k`` from the table width and masks only the columns past
   it; a negative entry merely skips the load (``paged_kv.h``), leaving a stale K/V row in smem
   that still produces a real score. Every padded column is therefore redirected to a per-row
   *anchor* token, and the anchor's contribution is removed in closed form from the LSE that FA3
   returns. Writing ``D`` for the denominator FA3 normalised by, ``n`` for a row's padded-column
   count and ``s_a`` / ``v_a`` for the anchor's score and value:

       D       = D_valid + n * exp(s_a)
       out * D = O_valid + n * exp(s_a) * v_a
       out     = (1 - w) * out_valid + w * v_a        with w = n * exp(s_a - LSE)
       => out_valid = (out - w * v_a) / (1 - w)

   That subtraction is exact, but it is catastrophic in BF16 once ``w`` approaches 1: ``out``
   then carries ``out_valid`` only in its low bits, and dividing by ``1 - w`` amplifies the
   rounding error by ``1 / (1 - w)``. Rows whose anchor weight exceeds ``ANCHOR_WEIGHT_LIMIT``
   are recomputed exactly from the KV cache instead. That is cheap precisely because such rows
   are the short ones (small usable prefix) produced at the very start of a prefill.

The derivation holds for *any* token the padded columns are pointed at, so the choice is a
numerical one, not a semantic one: ``w`` scales with ``exp(s_a)``, so a low-scoring anchor keeps
the correction well conditioned. The row's *last* usable column is used because pass 1 of
:func:`_qsa_fa3_prepare_table_kernel` already locates it while counting padded columns, so it
costs no extra work; it is not guaranteed to be the lowest-scoring selection.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

# Anchor weight above which subtracting the padding back out costs more accuracy than recomputing
# the row. The divisor alone would bound the damage to 1/(1-1/4) = 4/3, but that understates it:
# FA3 hands back an output that is *already* rounded to the q dtype, and the numerator
# ``out - w * v_a`` cancels as well, so the measured error of a corrected row runs about twice the
# divisor's prediction -- up to ~4.3x the BF16 rounding of the same exact result at w = 1/4,
# against the ~2x that FA3 tolerates in its own QSA kernel. Past the limit the factor grows
# without bound (35x by w = 0.8). Raising the limit is therefore a precision-for-recompute trade,
# not a free win, and 1/4 keeps the corrected rows within a small constant factor of a bare BF16
# store. Exceeding it means few usable columns -- ``valid_count ~ selection_width / (1 + w)``, and
# never more than the token's own position -- so the rows that fall to
# :func:`_qsa_fa3_recompute_kernel` are short-context ones whose walk shrinks as ``w`` grows,
# while the long rows that dominate decode stay on the closed-form correction.
ANCHOR_WEIGHT_LIMIT = 0.25

_LOG2E = 1.4426950408889634


@triton.jit
def _qsa_fa3_prepare_table_kernel(
    logical_indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    page_table_ptr,
    anchor_ptr,
    padding_ptr,
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

    Padded columns are pointed at the row's last usable token and counted, so the caller can
    remove their contribution afterwards. ``anchor`` and ``padding`` are written per row.
    """
    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row)
    request_live = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    offsets = tl.arange(0, COLUMN_BLOCK)

    # Pass 1: find the last usable column. The expansion kernel emits -1 as a contiguous suffix,
    # but deriving the bound here keeps this correct even if that ever changes.
    valid_end = 0
    padding = 0
    for start in tl.static_range(0, SELECTION_WIDTH, COLUMN_BLOCK):
        columns = start + offsets
        in_range = columns < SELECTION_WIDTH
        logical = tl.load(
            logical_indices_ptr + row * stride_logical_row + columns,
            mask=in_range & request_live,
            other=-1,
        )
        page_rank = tl.maximum(logical, 0) // PAGE_SIZE
        usable = in_range & request_live & (logical >= 0) & (page_rank < PAGE_TABLE_WIDTH)
        page = tl.load(
            block_table_ptr
            + safe_request * stride_block_table_req
            + tl.minimum(page_rank, PAGE_TABLE_WIDTH - 1),
            mask=usable,
            other=-1,
        )
        usable &= (page >= 0) & (page < num_cache_blocks)
        valid_end = tl.maximum(valid_end, tl.max(tl.where(usable, columns + 1, 0)))
        padding += tl.sum(tl.where(in_range & ~usable, 1, 0))

    # The anchor is the last usable column. valid_end == 0 only if the row selected nothing at
    # all, which the indexer cannot produce (its causal tail always keeps token 0); fall back to
    # logical 0 so the table still holds an addressable index.
    anchor_logical = tl.load(
        logical_indices_ptr + row * stride_logical_row + (valid_end - 1),
        mask=request_live & (valid_end > 0),
        other=0,
    )
    anchor_logical = tl.minimum(tl.maximum(anchor_logical, 0), num_cache_blocks * PAGE_SIZE - 1)
    anchor_rank = tl.minimum(anchor_logical // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
    anchor_page = tl.load(
        block_table_ptr + safe_request * stride_block_table_req + anchor_rank,
        mask=request_live,
        other=0,
    )
    anchor_page = tl.minimum(tl.maximum(anchor_page, 0), num_cache_blocks - 1)
    anchor = anchor_page * PAGE_SIZE + anchor_logical % PAGE_SIZE

    # Pass 2: emit the table. Every column gets a legal address, so FA3 never reads a negative
    # index and never leaves a stale K/V row in shared memory.
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
            mask=in_range & request_live,
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
        absolute = page * PAGE_SIZE + safe_logical % PAGE_SIZE
        tl.store(
            page_table_ptr + row * stride_page_table_row + columns,
            tl.where(usable, absolute, anchor),
            mask=in_range,
        )

    tl.store(anchor_ptr + row, anchor)
    tl.store(padding_ptr + row, padding)


@triton.jit
def _qsa_fa3_apply_padding_kernel(
    output_ptr,
    lse_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    anchor_ptr,
    padding_ptr,
    recompute_ptr,
    stride_output_row,
    stride_output_head,
    stride_query_row,
    stride_query_head,
    stride_cache_block,
    stride_cache_token,
    stride_cache_head,
    stride_lse_head,
    stride_lse_row,
    stride_recompute_row,
    softmax_scale,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WEIGHT_LIMIT: tl.constexpr,
) -> None:
    """Subtract the anchor's padding contribution out of FA3's output, one (row, head) at a time.

    Rows whose anchor weight is too large for a stable subtraction are left untouched and flagged
    for exact recomputation instead.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    padded = tl.load(padding_ptr + row)
    flagged = 0
    if padded != 0:
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < HEAD_DIM
        kv_head = head // GROUP_SIZE
        anchor = tl.load(anchor_ptr + row).to(tl.int64)
        cache_offset = (
            (anchor // PAGE_SIZE) * stride_cache_block
            + (anchor % PAGE_SIZE) * stride_cache_token
            + kv_head * stride_cache_head
        )
        query = tl.load(
            query_ptr + row * stride_query_row + head * stride_query_head + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            k_cache_ptr + cache_offset + dims,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(query * key) * softmax_scale
        lse = tl.load(lse_ptr + head * stride_lse_head + row * stride_lse_row)
        weight = padded.to(tl.float32) * tl.exp(score - lse)
        if weight <= WEIGHT_LIMIT:
            value = tl.load(
                v_cache_ptr + cache_offset + dims,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            base = output_ptr + row * stride_output_row + head * stride_output_head + dims
            current = tl.load(base, mask=dim_mask, other=0.0).to(tl.float32)
            corrected = (current - weight * value) / (1.0 - weight)
            tl.store(base, corrected.to(output_ptr.dtype.element_ty), mask=dim_mask)
        else:
            flagged = 1
    tl.store(
        recompute_ptr + row * stride_recompute_row + head,
        flagged.to(recompute_ptr.dtype.element_ty),
    )


@triton.jit
def _qsa_fa3_recompute_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    padding_ptr,
    recompute_ptr,
    stride_output_row,
    stride_output_head,
    stride_query_row,
    stride_query_head,
    stride_cache_block,
    stride_cache_token,
    stride_cache_head,
    stride_page_table_row,
    stride_recompute_row,
    softmax_scale_log2,
    PAGE_SIZE: tl.constexpr,
    SELECTION_WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Recompute flagged rows exactly, masking every padded column the way the Triton path does.

    Shares one K/V tile across the ``GROUP_SIZE`` query heads of a KV head, and only walks the
    usable prefix, so a flagged row costs less here than the full-width pass FA3 already made.
    """
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    first_head = kv_head * GROUP_SIZE
    head_offsets = tl.arange(0, BLOCK_M)
    head_mask = head_offsets < GROUP_SIZE
    flags = tl.load(
        recompute_ptr + row * stride_recompute_row + first_head + head_offsets,
        mask=head_mask,
        other=0,
    )
    if tl.sum(flags.to(tl.int32)) == 0:
        return

    valid_count = SELECTION_WIDTH - tl.load(padding_ptr + row)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    query = tl.load(
        query_ptr
        + row * stride_query_row
        + (first_head + head_offsets[:, None]) * stride_query_head
        + dim_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    for start in range(0, valid_count, BLOCK_N):
        columns = start + column_offsets
        live = columns < valid_count
        absolute = tl.load(
            page_table_ptr + row * stride_page_table_row + columns,
            mask=live,
            other=0,
        ).to(tl.int64)
        page = absolute // PAGE_SIZE
        slot = absolute % PAGE_SIZE
        keys = tl.load(
            k_cache_ptr
            + page[None, :] * stride_cache_block
            + slot[None, :] * stride_cache_token
            + kv_head * stride_cache_head
            + dim_offsets[:, None],
            mask=live[None, :],
            other=0.0,
        )
        scores = tl.dot(query, keys) * softmax_scale_log2
        scores = tl.where(live[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        values = tl.load(
            v_cache_ptr
            + page[:, None] * stride_cache_block
            + slot[:, None] * stride_cache_token
            + kv_head * stride_cache_head
            + dim_offsets[None, :],
            mask=live[:, None],
            other=0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    tl.store(
        output_ptr
        + row * stride_output_row
        + (first_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :],
        normalized,
        mask=head_mask[:, None],
    )


def qsa_fa3_prepare_page_table(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    page_table: torch.Tensor,
    anchor: torch.Tensor,
    padding: torch.Tensor,
    page_size: int,
    num_cache_blocks: int,
    column_block: int = 512,
) -> None:
    """Fill ``page_table`` with the absolute indices FA3's QSA path expects.

    Args:
        logical_indices: ``(num_tokens, selection_width)`` int32, request-relative, ``-1`` padded.
        block_table: ``(num_requests, page_table_width)`` int32.
        token_to_req: ``(num_tokens,)`` int32.
        page_table: ``(num_tokens, selection_width)`` int32, output. Callers add the singleton
            dimension FA3 needs to recognise a QSA table.
        anchor: ``(num_tokens,)`` int32, output. Absolute index of each row's padding anchor.
        padding: ``(num_tokens,)`` int32, output. Padded-column count per row.
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
    assert page_table.dtype == anchor.dtype == padding.dtype == torch.int32
    assert page_table.shape == logical_indices.shape
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert page_table.stride(1) == 1
    assert anchor.shape == padding.shape == (num_tokens,)

    _qsa_fa3_prepare_table_kernel[(num_tokens,)](
        logical_indices,
        block_table,
        token_to_req,
        page_table,
        anchor,
        padding,
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


def qsa_fa3_remove_padding(
    output: torch.Tensor,
    softmax_lse: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    anchor: torch.Tensor,
    padding: torch.Tensor,
    recompute_flags: torch.Tensor,
    softmax_scale: float,
    weight_limit: float = ANCHOR_WEIGHT_LIMIT,
    recompute_block_n: int = 16,
) -> None:
    """Turn FA3's padded-row output back into attention over the usable columns only.

    Two passes, both branch-free on the host and therefore safe to capture in a CUDA graph:
    a per-(row, head) pass that subtracts the anchor contribution using the returned LSE, and a
    per-(row, kv-head) pass that recomputes the rows where that subtraction would be ill-
    conditioned. Rows with no padding at all exit the first pass immediately.

    Args:
        output: ``(num_tokens, num_heads, head_dim)``, FA3's result, corrected in place.
        softmax_lse: FA3's LSE, ``(num_heads, num_tokens)`` float32 in the varlen layout.
        query: ``(num_tokens, num_heads, head_dim)``, same dtype as ``output``.
        key_cache: ``(num_blocks, page_size, num_kv_heads, head_dim)``.
        value_cache: same as ``key_cache``.
        page_table: the absolute table handed to FA3, ``(num_tokens, selection_width)``.
        anchor: / padding: per-row outputs of :func:`qsa_fa3_prepare_page_table`.
        recompute_flags: ``(num_tokens, num_heads)`` scratch, written then read here.
        softmax_scale: the scale FA3 was called with, needed to reproduce the anchor score.
    """
    if not HAS_TRITON:
        raise RuntimeError("FA3 QSA padding correction requires Triton")
    num_tokens, num_heads, head_dim = output.shape
    if num_tokens == 0:
        return
    num_kv_heads = key_cache.shape[2]
    group_size = num_heads // num_kv_heads
    assert output.dtype == query.dtype
    assert key_cache.dtype == value_cache.dtype == output.dtype
    assert softmax_lse.dtype == torch.float32
    assert anchor.dtype == padding.dtype == page_table.dtype == torch.int32
    assert recompute_flags.shape == (num_tokens, num_heads)
    assert anchor.shape == padding.shape == (num_tokens,)
    assert page_table.shape[0] == num_tokens
    assert output.stride(2) == query.stride(2) == 1
    assert key_cache.stride(3) == value_cache.stride(3) == 1

    _qsa_fa3_apply_padding_kernel[(num_tokens, num_heads)](
        output,
        softmax_lse,
        query,
        key_cache,
        value_cache,
        anchor,
        padding,
        recompute_flags,
        output.stride(0),
        output.stride(1),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        recompute_flags.stride(0),
        softmax_scale,
        PAGE_SIZE=key_cache.shape[1],
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        BLOCK_D=triton.next_power_of_2(head_dim),
        WEIGHT_LIMIT=weight_limit,
        num_warps=1,
    )

    _qsa_fa3_recompute_kernel[(num_tokens, num_kv_heads)](
        output,
        query,
        key_cache,
        value_cache,
        page_table,
        padding,
        recompute_flags,
        output.stride(0),
        output.stride(1),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        page_table.stride(0),
        recompute_flags.stride(0),
        softmax_scale * _LOG2E,
        PAGE_SIZE=key_cache.shape[1],
        SELECTION_WIDTH=page_table.shape[1],
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        BLOCK_M=max(16, triton.next_power_of_2(group_size)),
        BLOCK_N=recompute_block_n,
        num_warps=4,
    )


__all__ = [
    "ANCHOR_WEIGHT_LIMIT",
    "qsa_fa3_prepare_page_table",
    "qsa_fa3_remove_padding",
]
