# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UT for the Qwen4Exp QSA DeepGEMM indexer backend (VLLM_QSA_INDEXER_BACKEND).

Covers the FP8 shadow-cache contract and the decode scoring path:

1. ``qsa_store_cache_rows_fp8`` scatters per-row e4m3 values and fp32 scales
   into DeepGEMM's fused layout (per block: value plane, then scale plane),
   byte-identical to a torch reference.

2. ``qsa_store_cache_rows_fp8_readback`` mirrors rows already committed to
   the BF16 cache (the fused pre-indexer path) with the same result.

3. ``QSAMetadataBuilder._build_fp8_decode_metadata`` only engages for
   uniform decode batches (every request contributes ``next_n`` tokens
   covering the whole batch) and fills per-token visible counts with the
   Triton kernel's ``(pos + 1) // compress_ratio`` rule.

4. ``qsa_select_paged_tokens_deepgemm`` matches the Triton BF16 production
   scorer's top-k selection on production-shaped inputs.

5. ``QSAMetadataBuilder._build_fp8_packed_capacity`` upper-bounds the packed
   key count from host-side batch sizes; ``gather_packed_qsa_fp8_keys``
   repacks the planar shadow byte-identically.

6. ``qsa_select_packed_tokens_deepgemm`` matches the Triton scorer on
   non-uniform prefill/mixed batches (fresh prefill, chunked prefill with
   history, prompts shorter than one compression group), with and without
   row chunking; multi-request batches of short prompts match it exactly,
   row for row (top-k paddings must not swallow real picks).

Needs a PPU/CUDA device and a deep_gemm build with the avg-logits kernels.
"""

from types import SimpleNamespace

import pytest
import torch

PAGE = 64
HEAD_DIM = 128
NUM_HEADS = 4
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
E4M3_MAX = 448.0


def _require_deepgemm_avg():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA/PPU device")
    from vllm.utils.ppu_deep_gemm import has_fp8_mqa_avg_logits

    if not has_fp8_mqa_avg_logits():
        pytest.skip("installed deep_gemm lacks the QSA avg-logits kernels")


def _reference_fp8_shadow(
    rows: torch.Tensor, slots: torch.Tensor, num_blocks: int
) -> torch.Tensor:
    """Torch mirror of DeepGEMM's kv_cache_cast_to_fp8 scatter layout."""
    rows = rows.float()
    amax = rows.abs().amax(dim=1).clamp(min=1e-4)
    scales = amax / E4M3_MAX
    values = (rows * (E4M3_MAX / amax)[:, None]).to(torch.float8_e4m3fn)
    shadow = torch.zeros(
        num_blocks, PAGE * (HEAD_DIM + 4), dtype=torch.uint8, device=rows.device
    )
    valid = slots >= 0
    blocks = slots[valid] // PAGE
    tokens = slots[valid] % PAGE
    base = blocks * (PAGE * (HEAD_DIM + 4))
    value_offsets = (
        base[:, None] + tokens[:, None] * HEAD_DIM + torch.arange(HEAD_DIM, device=rows.device)
    )
    shadow.view(-1).index_put_(
        (value_offsets.reshape(-1),),
        values.view(torch.uint8)[valid].reshape(-1),
    )
    scale_offsets = base[:, None] + PAGE * HEAD_DIM + tokens[:, None] * 4 + torch.arange(
        4, device=rows.device
    )
    shadow.view(-1).index_put_(
        (scale_offsets.reshape(-1),),
        scales[valid].view(torch.uint8).reshape(-1),
    )
    return shadow.view(num_blocks, PAGE, 1, HEAD_DIM + 4)


def test_fp8_store_layout():
    _require_deepgemm_avg()
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_store_cache_rows_fp8

    torch.manual_seed(0)
    num_blocks, num_rows = 8, 37
    cache_fp8 = torch.zeros(
        num_blocks, PAGE, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    rows = torch.randn(num_rows, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slots = torch.randperm(num_blocks * PAGE, device="cuda")[:num_rows].to(torch.int64)
    slots[::5] = -1  # padding rows must be skipped
    qsa_store_cache_rows_fp8(cache_fp8, slots, rows, PAGE)
    torch.cuda.synchronize()
    reference = _reference_fp8_shadow(rows[:, 0], slots, num_blocks)
    assert torch.equal(cache_fp8, reference)


def test_fp8_store_readback():
    _require_deepgemm_avg()
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        qsa_store_cache_rows,
        qsa_store_cache_rows_fp8_readback,
    )

    torch.manual_seed(1)
    num_blocks, num_rows = 8, 29
    cache_bf16 = torch.zeros(
        num_blocks, PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    cache_fp8 = torch.zeros(
        num_blocks, PAGE, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    rows = torch.randn(num_rows, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slots = torch.randperm(num_blocks * PAGE, device="cuda")[:num_rows].to(torch.int64)
    slots[::4] = -1
    qsa_store_cache_rows(cache_bf16, slots, rows)
    qsa_store_cache_rows_fp8_readback(cache_fp8, slots, cache_bf16)
    torch.cuda.synchronize()
    reference = _reference_fp8_shadow(rows[:, 0], slots, num_blocks)
    assert torch.equal(cache_fp8, reference)


def test_fp8_block_table_expand_k2():
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _expand_qsa_fp8_block_table

    block_table = torch.tensor([[5, 9, 2], [7, 1, 4]], dtype=torch.int32, device="cuda")
    expanded = _expand_qsa_fp8_block_table(block_table, 2)
    expected = torch.tensor(
        [[10, 11, 18, 19, 4, 5], [14, 15, 2, 3, 8, 9]],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(expanded, expected)
    assert _expand_qsa_fp8_block_table(block_table, 1) is block_table


def test_fp8_store_layout_k2():
    _require_deepgemm_avg()
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_store_cache_rows_fp8

    torch.manual_seed(2)
    bf16_page = 2 * PAGE  # 128 compressed rows per BF16 block -> two FP8 pages
    num_bf16_blocks, num_rows = 6, 53
    cache_fp8 = torch.zeros(
        2 * num_bf16_blocks, PAGE, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    rows = torch.randn(num_rows, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slots = torch.randperm(num_bf16_blocks * bf16_page, device="cuda")[:num_rows].to(
        torch.int64
    )
    slots[::7] = -1  # padding rows must be skipped
    qsa_store_cache_rows_fp8(cache_fp8, slots, rows, bf16_page)
    torch.cuda.synchronize()
    # Flat slot values are layout-invariant: the FP8 shadow holds the same rows
    # addressed by 64-row pages, so the reference just doubles the page count.
    reference = _reference_fp8_shadow(rows[:, 0], slots, 2 * num_bf16_blocks)
    assert torch.equal(cache_fp8, reference)


def _fake_builder(next_n_max: int, max_requests: int = 16):
    from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder
    from vllm.utils.ppu_deep_gemm import (
        get_num_sms,
        get_paged_mqa_logits_metadata,
    )

    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.fp8_decode = True
    builder.compress_ratio = COMPRESS_RATIO
    builder.fp8_next_n_max = next_n_max
    builder.fp8_index_heads = NUM_HEADS
    builder.fp8_index_head_dim = HEAD_DIM
    builder.fp8_num_sms = get_num_sms()
    builder.decode_context_lens_buffer = torch.zeros(
        max_requests * next_n_max, dtype=torch.int32, device="cuda"
    )
    schedule_rows = 1
    for next_n in range(1, next_n_max + 1):
        probe = get_paged_mqa_logits_metadata(
            torch.ones(1, next_n, dtype=torch.int32, device="cuda"),
            PAGE,
            builder.fp8_num_sms,
            metadata_extra=(next_n, NUM_HEADS, HEAD_DIM, 1),
        )
        schedule_rows = max(schedule_rows, probe.shape[0])
    builder.decode_schedule_buffer = torch.zeros(
        (schedule_rows, 2), dtype=torch.int32, device="cuda"
    )
    return builder


def test_build_fp8_decode_metadata_uniform():
    _require_deepgemm_avg()
    builder = _fake_builder(next_n_max=2)
    num_reqs, next_n = 5, 2
    num_tokens = num_reqs * next_n
    seq_lens = torch.tensor(
        [4096, 37, 1027, 3, 8192], dtype=torch.int32, device="cuda"
    )
    query_start_loc_cpu = torch.arange(
        0, (num_reqs + 1) * next_n, next_n, dtype=torch.int32
    )
    common = SimpleNamespace(query_start_loc_cpu=query_start_loc_cpu)
    # Decode positions: seq_len - next_n + j for token j of each request.
    logical_positions = (
        seq_lens.to(torch.int64).repeat_interleave(next_n)
        - next_n
        + torch.arange(num_tokens, device="cuda") % next_n
    )
    uniform_next_n, context_lens, schedule = builder._build_fp8_decode_metadata(
        common, logical_positions, num_tokens
    )
    assert uniform_next_n == next_n
    assert context_lens.shape == (num_reqs, next_n)
    assert context_lens.dtype == torch.int32
    expected = (
        torch.div(logical_positions + 1, COMPRESS_RATIO, rounding_mode="floor")
        .clamp(min=0)
        .to(torch.int32)
        .view(num_reqs, next_n)
    )
    assert torch.equal(context_lens, expected)
    assert schedule is not None
    assert schedule.dtype == torch.int32
    assert schedule.shape[1] == 2
    assert (schedule.shape[0] - 1) % builder.fp8_num_sms == 0


def test_build_fp8_decode_metadata_rejects_nonuniform():
    _require_deepgemm_avg()
    builder = _fake_builder(next_n_max=2)

    # Mixed query lengths: a prefill chunk sharing the batch.
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 2, 6], dtype=torch.int32)
    )
    positions = torch.zeros(6, dtype=torch.int64, device="cuda")
    assert builder._build_fp8_decode_metadata(common, positions, 6) == (0, None, None)

    # Uniform but above next_n_max (e.g. a small uniform prefill chunk).
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 3, 6], dtype=torch.int32)
    )
    assert builder._build_fp8_decode_metadata(common, positions, 6) == (0, None, None)

    # Uniform decode shape but not covering padded rows.
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32)
    )
    assert builder._build_fp8_decode_metadata(common, positions, 6) == (0, None, None)


def _make_paged_case(batch_size: int, next_n: int, seed: int):
    torch.manual_seed(seed)
    seq_lens = torch.randint(
        1024, 8193, (batch_size,), dtype=torch.int32, device="cuda"
    )
    seq_lens[0] = 2  # visible == 0 for this request's rows
    max_visible = int(seq_lens.max()) // COMPRESS_RATIO
    max_blocks = (max_visible + PAGE - 1) // PAGE + 2
    num_blocks = batch_size * max_blocks + 4
    permutation = torch.randperm(num_blocks, device="cuda")[: batch_size * max_blocks]
    block_table = permutation.view(batch_size, max_blocks).to(torch.int32)
    cache_bf16 = torch.randn(
        num_blocks, PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    rows = batch_size * next_n
    q = torch.randn(rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    token_to_req = torch.arange(batch_size, device="cuda", dtype=torch.int32)
    token_to_req = token_to_req.repeat_interleave(next_n)
    positions = (
        seq_lens.to(torch.int64).repeat_interleave(next_n)
        - next_n
        + torch.arange(rows, device="cuda") % next_n
    )
    context_lens = (
        torch.div(positions + 1, COMPRESS_RATIO, rounding_mode="floor")
        .clamp(min=0)
        .to(torch.int32)
        .view(batch_size, next_n)
    )
    return (
        seq_lens,
        block_table,
        cache_bf16,
        q,
        token_to_req,
        positions,
        context_lens,
        num_blocks,
    )


def _assert_topk_recall(
    reference: torch.Tensor,
    selected: torch.Tensor,
    mean_min: float = 0.98,
    row_min: float = 0.90,
) -> None:
    assert selected.shape == reference.shape
    assert selected.dtype == reference.dtype == torch.int32
    recalls = []
    for row in range(reference.shape[0]):
        ref_valid = reference[row][reference[row] >= 0]
        sel_valid = selected[row][selected[row] >= 0]
        if not ref_valid.numel():
            # Nothing visible: both paths must return all-padding rows.
            assert not sel_valid.numel()
            continue
        overlap = torch.isin(sel_valid, ref_valid).float().mean()
        recalls.append(float(overlap))
    recalls = torch.tensor(recalls)
    assert recalls.mean() >= mean_min, f"mean recall {recalls.mean():.4f}"
    assert recalls.min() >= row_min, f"min recall {recalls.min():.4f}"


@pytest.mark.parametrize("batch_size,next_n", [(8, 1), (6, 2)])
def test_deepgemm_decode_matches_triton(batch_size, next_n):
    _require_deepgemm_avg()
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        qsa_select_paged_tokens,
        qsa_select_paged_tokens_deepgemm,
        qsa_store_cache_rows_fp8,
    )
    from vllm.utils.ppu_deep_gemm import (
        get_num_sms,
        get_paged_mqa_logits_metadata,
    )

    (
        seq_lens,
        block_table,
        cache_bf16,
        q,
        token_to_req,
        positions,
        context_lens,
        num_blocks,
    ) = _make_paged_case(batch_size, next_n, seed=batch_size * 10 + next_n)

    # Fill the FP8 shadow through the production store path.
    cache_fp8 = torch.empty(
        num_blocks, PAGE, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    all_slots = torch.arange(num_blocks * PAGE, dtype=torch.int64, device="cuda")
    qsa_store_cache_rows_fp8(cache_fp8, all_slots, cache_bf16.view(-1, HEAD_DIM), PAGE)

    schedule = get_paged_mqa_logits_metadata(
        context_lens.clamp(min=1),
        PAGE,
        get_num_sms(),
        metadata_extra=(next_n, NUM_HEADS, HEAD_DIM, 1),
    )
    reference = qsa_select_paged_tokens(
        q,
        cache_bf16,
        block_table,
        token_to_req,
        positions,
        seq_lens,
        TOKEN_TOPK,
        COMPRESS_RATIO,
    )
    selected = qsa_select_paged_tokens_deepgemm(
        q,
        cache_fp8,
        block_table,
        context_lens,
        schedule,
        next_n,
        token_to_req,
        positions,
        seq_lens,
        TOKEN_TOPK,
        COMPRESS_RATIO,
        1,
    )
    torch.cuda.synchronize()
    _assert_topk_recall(reference, selected)


def test_build_fp8_packed_capacity():
    _require_deepgemm_avg()
    from vllm.models.qwen4_exp.common.qsa_cache import QSAMetadataBuilder

    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.fp8_decode = True
    builder.compress_ratio = COMPRESS_RATIO
    common = SimpleNamespace(num_reqs=3, max_seq_len=100)
    assert builder._build_fp8_packed_capacity(common) == 3 * 25

    builder.fp8_decode = False
    assert builder._build_fp8_packed_capacity(common) == 0


def test_packed_gather_matches_store():
    _require_deepgemm_avg()
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        gather_packed_qsa_fp8_keys,
        qsa_store_cache_rows_fp8,
    )

    (
        seq_lens,
        block_table,
        cache_bf16,
        _q,
        _token_to_req,
        _positions,
        packed_capacity,
        num_blocks,
    ) = _make_prefill_case(seed=11)
    cache_fp8 = torch.empty(
        num_blocks, PAGE, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    all_slots = torch.arange(num_blocks * PAGE, dtype=torch.int64, device="cuda")
    qsa_store_cache_rows_fp8(cache_fp8, all_slots, cache_bf16.view(-1, HEAD_DIM), PAGE)

    packed_keys, packed_scales, cu_seqlens = gather_packed_qsa_fp8_keys(
        cache_fp8, block_table, seq_lens, COMPRESS_RATIO, packed_capacity, 1
    )
    torch.cuda.synchronize()

    expected_cu = [0]
    for seq_len in seq_lens.tolist():
        expected_cu.append(expected_cu[-1] + seq_len // COMPRESS_RATIO)
    assert cu_seqlens.tolist() == expected_cu
    total = expected_cu[-1]

    # Byte-exact against an independent re-quantization of the BF16 rows at
    # the same logical positions (the store formula is deterministic).
    flat_bf16 = cache_bf16.view(-1, HEAD_DIM)
    ref_values = torch.empty(total, HEAD_DIM, dtype=torch.uint8, device="cuda")
    ref_scales = torch.empty(total, dtype=torch.float32, device="cuda")
    cursor = 0
    for req, seq_len in enumerate(seq_lens.tolist()):
        count = seq_len // COMPRESS_RATIO
        if not count:
            continue
        rows = torch.arange(count, device="cuda")
        blocks = block_table[req, rows // PAGE].long()
        source = flat_bf16[blocks * PAGE + rows % PAGE].float()
        amax = source.abs().amax(dim=1).clamp(min=1e-4)
        ref_scales[cursor : cursor + count] = amax / E4M3_MAX
        ref_values[cursor : cursor + count] = (
            (source * (E4M3_MAX / amax)[:, None])
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        cursor += count
    assert torch.equal(packed_keys[:total].view(torch.uint8), ref_values)
    assert torch.equal(packed_scales[:total], ref_scales)


def _make_prefill_case(seed: int):
    """Non-uniform batch over fresh/chunked/tiny/ratio-aligned prompts."""
    torch.manual_seed(seed)
    # (seq_len, query_len): fresh prefill, chunked prefill with history,
    # prompt shorter than one compression group, ratio-divisible length.
    shapes = [(700, 700), (3000, 512), (3, 3), (2048, 2048)]
    num_reqs = len(shapes)
    seq_lens = torch.tensor(
        [seq_len for seq_len, _ in shapes], dtype=torch.int32, device="cuda"
    )
    max_visible = int(seq_lens.max()) // COMPRESS_RATIO
    max_blocks = (max_visible + PAGE - 1) // PAGE + 2
    num_blocks = num_reqs * max_blocks + 4
    permutation = torch.randperm(num_blocks, device="cuda")[: num_reqs * max_blocks]
    block_table = permutation.view(num_reqs, max_blocks).to(torch.int32)
    cache_bf16 = torch.randn(
        num_blocks, PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    rows = sum(query_len for _, query_len in shapes)
    q = torch.randn(rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    token_to_req = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device="cuda"),
        torch.tensor(
            [query_len for _, query_len in shapes],
            dtype=torch.int64,
            device="cuda",
        ),
    )
    positions = torch.cat(
        [
            torch.arange(seq_len - query_len, seq_len, dtype=torch.int64, device="cuda")
            for seq_len, query_len in shapes
        ]
    )
    packed_capacity = num_reqs * (int(seq_lens.max()) // COMPRESS_RATIO)
    return (
        seq_lens,
        block_table,
        cache_bf16,
        q,
        token_to_req,
        positions,
        packed_capacity,
        num_blocks,
    )


def _run_prefill_deepgemm(case, logits_workspace_bytes: int | None = None):
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    (
        seq_lens,
        block_table,
        cache_bf16,
        q,
        token_to_req,
        positions,
        packed_capacity,
        num_blocks,
    ) = case

    # Fill the FP8 shadow through the production store path.
    cache_fp8 = torch.empty(
        num_blocks, PAGE, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    all_slots = torch.arange(num_blocks * PAGE, dtype=torch.int64, device="cuda")
    qsa_ops.qsa_store_cache_rows_fp8(
        cache_fp8, all_slots, cache_bf16.view(-1, HEAD_DIM), PAGE
    )

    reference = qsa_ops.qsa_select_paged_tokens(
        q,
        cache_bf16,
        block_table,
        token_to_req,
        positions,
        seq_lens,
        TOKEN_TOPK,
        COMPRESS_RATIO,
    )
    if logits_workspace_bytes is None:
        selected = qsa_ops.qsa_select_packed_tokens_deepgemm(
            q,
            cache_fp8,
            block_table,
            seq_lens,
            token_to_req,
            positions,
            TOKEN_TOPK,
            COMPRESS_RATIO,
            packed_capacity,
            1,
        )
    else:
        original = qsa_ops._LOGITS_WORKSPACE_BYTES
        qsa_ops._LOGITS_WORKSPACE_BYTES = logits_workspace_bytes
        try:
            selected = qsa_ops.qsa_select_packed_tokens_deepgemm(
                q,
                cache_fp8,
                block_table,
                seq_lens,
                token_to_req,
                positions,
                TOKEN_TOPK,
                COMPRESS_RATIO,
                packed_capacity,
                1,
            )
        finally:
            qsa_ops._LOGITS_WORKSPACE_BYTES = original
    torch.cuda.synchronize()
    return reference, selected


def test_deepgemm_prefill_matches_triton():
    _require_deepgemm_avg()
    reference, selected = _run_prefill_deepgemm(_make_prefill_case(seed=5))
    _assert_topk_recall(reference, selected)


def test_deepgemm_prefill_row_chunking():
    _require_deepgemm_avg()
    case = _make_prefill_case(seed=7)
    packed_capacity = case[6]
    # NUM_HEADS == 4 -> the operator's block_kv is 128; budget 173 rows per
    # chunk so the 3263-row batch splits into many chunks.
    aligned_kv = -(-(packed_capacity + 128) // 256) * 256
    reference, selected = _run_prefill_deepgemm(
        case, logits_workspace_bytes=173 * aligned_kv * 12
    )
    _assert_topk_recall(reference, selected)


def _make_multi_small_case(seq_lens_list: list[int], seed: int):
    """Multi-request batch of short prompts, every token a query row.

    Every row's visible count is far below block_topk, so top-k paddings
    dominate the picks; for requests past the first those paddings rebase to
    -1 and must not swallow the real picks (the packed multi-request
    regression).
    """
    torch.manual_seed(seed)
    num_reqs = len(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
    max_visible = int(seq_lens.max()) // COMPRESS_RATIO
    max_blocks = (max_visible + PAGE - 1) // PAGE + 2
    num_blocks = num_reqs * max_blocks + 4
    permutation = torch.randperm(num_blocks, device="cuda")[: num_reqs * max_blocks]
    block_table = permutation.view(num_reqs, max_blocks).to(torch.int32)
    cache_bf16 = torch.randn(
        num_blocks, PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    q = torch.randn(
        int(seq_lens.sum()), NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    token_to_req = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device="cuda"),
        seq_lens.to(torch.int64),
    )
    positions = torch.cat(
        [torch.arange(n, dtype=torch.int64, device="cuda") for n in seq_lens_list]
    )
    packed_capacity = num_reqs * max_visible
    return (
        seq_lens,
        block_table,
        cache_bf16,
        q,
        token_to_req,
        positions,
        packed_capacity,
        num_blocks,
    )


@pytest.mark.parametrize(
    "seq_lens_list", [[60, 60, 64], [64, 60, 60], [60, 60, 60], [64, 64, 64]]
)
def test_deepgemm_packed_multi_request_exact(seq_lens_list):
    _require_deepgemm_avg()
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    # visible <= 16 << block_topk = 512: every visible block must be picked,
    # so FP8 tie-breaks cannot occur and exact equality is deterministic;
    # top-k paddings rebase to -1 for requests past the first and must not
    # swallow the real picks (the packed multi-request regression).
    token_topk = TOKEN_TOPK
    (
        seq_lens,
        block_table,
        cache_bf16,
        q,
        token_to_req,
        positions,
        packed_capacity,
        num_blocks,
    ) = _make_multi_small_case(seq_lens_list, seed=13)
    cache_fp8 = torch.empty(
        num_blocks, PAGE, 1, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    all_slots = torch.arange(num_blocks * PAGE, dtype=torch.int64, device="cuda")
    qsa_ops.qsa_store_cache_rows_fp8(
        cache_fp8, all_slots, cache_bf16.view(-1, HEAD_DIM), PAGE
    )

    reference = qsa_ops.qsa_select_paged_tokens(
        q,
        cache_bf16,
        block_table,
        token_to_req,
        positions,
        seq_lens,
        token_topk,
        COMPRESS_RATIO,
    )
    selected = qsa_ops.qsa_select_packed_tokens_deepgemm(
        q,
        cache_fp8,
        block_table,
        seq_lens,
        token_to_req,
        positions,
        token_topk,
        COMPRESS_RATIO,
        packed_capacity,
        1,
    )
    torch.cuda.synchronize()
    mismatch = (reference != selected).any(dim=1)
    assert not mismatch.any(), (
        f"{int(mismatch.sum())}/{reference.shape[0]} rows differ; "
        f"first row {int(mismatch.nonzero()[0])}: "
        f"ref {reference[mismatch.nonzero()[0]].tolist()} "
        f"vs sel {selected[mismatch.nonzero()[0]].tolist()}"
    )


if __name__ == "__main__":
    test_fp8_store_layout()
    print("PASS test_fp8_store_layout")
    test_fp8_store_readback()
    print("PASS test_fp8_store_readback")
    test_build_fp8_decode_metadata_uniform()
    print("PASS test_build_fp8_decode_metadata_uniform")
    test_build_fp8_decode_metadata_rejects_nonuniform()
    print("PASS test_build_fp8_decode_metadata_rejects_nonuniform")
    for batch_size, next_n in [(8, 1), (6, 2)]:
        test_deepgemm_decode_matches_triton(batch_size, next_n)
        print(f"PASS test_deepgemm_decode_matches_triton B={batch_size} nn={next_n}")
    test_build_fp8_packed_capacity()
    print("PASS test_build_fp8_packed_capacity")
    test_packed_gather_matches_store()
    print("PASS test_packed_gather_matches_store")
    test_deepgemm_prefill_matches_triton()
    print("PASS test_deepgemm_prefill_matches_triton")
    test_deepgemm_prefill_row_chunking()
    print("PASS test_deepgemm_prefill_row_chunking")
    for seq_lens_list in [[60, 60, 64], [64, 60, 60], [60, 60, 60], [64, 64, 64]]:
        test_deepgemm_packed_multi_request_exact(seq_lens_list)
        print(f"PASS test_deepgemm_packed_multi_request_exact {seq_lens_list}")
