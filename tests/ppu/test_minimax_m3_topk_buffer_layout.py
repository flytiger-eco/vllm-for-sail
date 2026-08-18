# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify MiniMax M3's shared ``topk_indices_buffer`` is allocated token-major
``[padded_num_tokens, num_index_heads, topk]`` on every platform.

Regression test for a layout divergence: an earlier PPU workaround allocated
the buffer head-major ``[H, T, MK]`` on PPU, which was self-consistent while
the Triton indexer/attend also accessed the buffer head-major. Upstream #49149
unified all consumers on the token-major layout (indexer writes via
``buf.transpose(0, 1)``, attend reads via ``topk_buffer[:n].transpose(0, 1)``,
MSA reads token-major natively). Keeping the head-major allocation turned the
top-k block indices into garbage, and the sparse-attention (MSA) kernel
crashed with invalid KV page accesses.

The test drives the real ``MiniMaxM3Model.__init__`` allocation path with the
heavy components stubbed (no GPU, no distributed), then checks both the shape
and the stride contract the transpose-view consumers rely on.
"""

import types

import pytest
import torch

MAX_NUM_BATCHED_TOKENS = 130  # not a multiple of 4: exercises the padding
NUM_INDEX_HEADS = 4
TOPK_BLOCKS = 16


@pytest.fixture
def m3_model_cls(monkeypatch):
    try:
        from vllm.models.minimax_m3.nvidia import model as m3_model
    except ModuleNotFoundError as e:
        pytest.skip(f"MiniMax M3 model import requires optional deps: {e}")

    pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(m3_model, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(m3_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(m3_model, "VocabParallelEmbedding", lambda *a, **k: None)
    monkeypatch.setattr(m3_model, "MiniMAXGemmaRMSNorm", lambda *a, **k: None)
    monkeypatch.setattr(m3_model, "make_layers", lambda n, fn, prefix="": (0, 0, []))
    return m3_model.MiniMaxM3Model


@pytest.fixture
def vllm_config():
    hf_config = types.SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        sparse_attention_config={
            "sparse_num_index_heads": NUM_INDEX_HEADS,
            "sparse_topk_blocks": TOPK_BLOCKS,
        },
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_text_config=hf_config),
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS
        ),
        quant_config=None,
    )


def test_topk_indices_buffer_is_token_major(m3_model_cls, vllm_config):
    model = m3_model_cls(vllm_config=vllm_config)
    buf = model.topk_indices_buffer

    padded = (MAX_NUM_BATCHED_TOKENS + 3) // 4 * 4
    assert buf.dtype == torch.int32
    # Token-major [T, H, MK]; the pad keeps the head stride int4-aligned.
    assert buf.shape == (padded, NUM_INDEX_HEADS, TOPK_BLOCKS)
    assert buf.stride() == (NUM_INDEX_HEADS * TOPK_BLOCKS, TOPK_BLOCKS, 1)


def test_transpose_views_match_kernel_contracts(m3_model_cls, vllm_config):
    """The indexer writes head-major through ``buf.transpose(0, 1)`` and the
    attend reads ``buf[:num_tokens].transpose(0, 1)``; both views must land on
    the same underlying storage so writes are what reads see."""
    model = m3_model_cls(vllm_config=vllm_config)
    buf = model.topk_indices_buffer

    padded = buf.shape[0]
    num_tokens = padded - 4
    writer_view = buf.transpose(0, 1)
    reader_view = buf[:num_tokens].transpose(0, 1)
    assert writer_view.shape == (NUM_INDEX_HEADS, padded, TOPK_BLOCKS)
    assert reader_view.shape == (NUM_INDEX_HEADS, num_tokens, TOPK_BLOCKS)

    writer_view[:, :num_tokens, :] = torch.arange(
        NUM_INDEX_HEADS * num_tokens * TOPK_BLOCKS, dtype=torch.int32
    ).view(NUM_INDEX_HEADS, num_tokens, TOPK_BLOCKS)
    assert torch.equal(reader_view, writer_view[:, :num_tokens, :])
