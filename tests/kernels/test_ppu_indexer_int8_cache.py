# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_group_quant_int8,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

pytestmark = pytest.mark.skipif(
    not current_platform.is_ppu(), reason="PPU INT8 indexer cache"
)


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 129])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 1.0, 1000.0])
def test_int8_cache_bytes(num_tokens, block_size, unified, magnitude):
    from vllm.v1.attention.ops.ppu_mla_sparse import indexer_k_quant_and_cache_int8

    torch.manual_seed(0)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    backing = torch.full(
        (num_blocks, 3 if unified else 1, block_size, 132),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )
    original = backing.clone()
    cache = backing[:, 1 if unified else 0]
    expected = torch.full_like(cache, 0xA5).contiguous()
    k = (torch.randn(num_tokens, 256, device="cuda") * magnitude).to(
        torch.bfloat16
    )[:, ::2]
    slots = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device="cuda"
    )[:num_tokens].contiguous()
    if num_tokens > 1:
        slots[-1] = -1

    indexer_k_quant_and_cache_int8(k, cache, slots, 128)

    if num_tokens:
        values, scales = per_token_group_quant_int8(k.contiguous(), 128)
        reference_scales = k.float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        torch.testing.assert_close(scales, reference_scales, rtol=1e-6, atol=0)
        assert torch.isfinite(scales).all() and (scales > 0).all()
        error = (values.float() * scales - k.float()).abs()
        assert torch.all(error <= scales * 1.01)
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot < 0:
                continue
            page, token = divmod(slot, block_size)
            flat = expected[page].view(-1)
            flat[token * 128 : (token + 1) * 128] = values[row].view(torch.uint8)
            start = block_size * 128 + token * 4
            flat[start : start + 4] = scales[row].view(torch.uint8)

    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    if unified:
        torch.testing.assert_close(backing[:, 0], original[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(backing[:, 2], original[:, 2], rtol=0, atol=0)


@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("unified", [False, True])
def test_int8_cache_gather(block_size, unified):
    from vllm.v1.attention.ops.ppu_mla_sparse import indexer_k_quant_and_cache_int8

    torch.manual_seed(1)
    num_tokens = block_size + 3
    backing = torch.zeros(
        (3, 3 if unified else 1, block_size, 132),
        dtype=torch.uint8,
        device="cuda",
    )
    cache = backing[:, 1 if unified else 0]
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device="cuda")
    logical = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
    slots = block_table[0, logical // block_size].to(torch.int64) * block_size
    slots = torch.cat((slots + logical % block_size, logical.new_full((2,), -1)))
    k = torch.randn(num_tokens + 2, 128, dtype=torch.bfloat16, device="cuda")
    indexer_k_quant_and_cache_int8(k, cache, slots, 128)

    gathered = torch.empty(num_tokens, 128, dtype=torch.int8, device="cuda")
    scale_bytes = torch.empty(num_tokens, 4, dtype=torch.uint8, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(
        cache,
        gathered,
        scale_bytes,
        block_table,
        torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda"),
    )
    values, scales = per_token_group_quant_int8(k[:num_tokens].contiguous(), 128)
    torch.testing.assert_close(gathered, values, rtol=0, atol=0)
    torch.testing.assert_close(scale_bytes.view(torch.float32), scales, rtol=0, atol=0)


@pytest.mark.parametrize("q_dtype", [torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("skip_insert", [False, True])
def test_indexer_writer_dispatch(monkeypatch, q_dtype, skip_insert):
    from vllm.v1.attention.ops import ppu_mla_sparse as indexer

    layer = "model.layers.0.self_attn.indexer.k_cache"
    slots = torch.tensor([15, 16, -1], dtype=torch.int64)
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=torch.empty(0, dtype=torch.int32),
        max_seq_len=32,
        slot_mapping=slots,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=0,
        num_prefill_tokens=0,
    )
    monkeypatch.setattr(
        indexer,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer: metadata}),
    )
    calls = []
    monkeypatch.setattr(
        indexer,
        "indexer_k_quant_and_cache_int8",
        lambda *args: calls.append(("int8", args)),
    )
    monkeypatch.setattr(
        indexer.ops,
        "indexer_k_quant_and_cache",
        lambda *args: calls.append(("fp8", args)),
    )
    k = torch.zeros(5, 128, dtype=torch.bfloat16)
    topk = torch.full((5, 2), 99, dtype=torch.int32)
    result = indexer.ppu_sparse_attn_indexer(
        torch.empty(5, 1),
        layer,
        torch.empty(1),
        torch.empty(5, 1, 128, dtype=q_dtype),
        None,
        k,
        torch.empty(5, 1),
        128,
        "ue8m0",
        2,
        128,
        32,
        32,
        topk,
        skip_insert,
    )
    assert result is topk and torch.all(topk == -1)
    if skip_insert:
        assert calls == []
    else:
        assert len(calls) == 1
        name, args = calls[0]
        assert name == ("int8" if q_dtype == torch.int8 else "fp8")
        torch.testing.assert_close(args[0], k[: slots.numel()])
        assert args[2] is slots
        assert args[3] == 128
        if name == "fp8":
            assert args[4] == "ue8m0"


def test_int8_cache_graph_replay():
    from vllm.v1.attention.ops.ppu_mla_sparse import indexer_k_quant_and_cache_int8

    k = torch.ones(3, 128, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([63, 64, -1], dtype=torch.int64, device="cuda")
    cache = torch.zeros(2, 64, 132, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        indexer_k_quant_and_cache_int8(k, cache, slots, 128)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indexer_k_quant_and_cache_int8(k, cache, slots, 128)
    k.fill_(-2)
    graph.replay()
    reference = torch.zeros_like(cache)
    indexer_k_quant_and_cache_int8(k, reference, slots, 128)
    torch.testing.assert_close(cache, reference, rtol=0, atol=0)
