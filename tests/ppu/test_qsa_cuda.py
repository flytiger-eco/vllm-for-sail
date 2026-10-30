# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UT for the Qwen4Exp QSA CUDA path (PPU FA3 kernel integration).

Covers the patch's three contracts:

1. The build probe's top-k list is scattered and unsorted, so a load path
   that fetches contiguous runs cannot match the numerical reference by
   accident -- a contiguous list (the original probe) passes exactly on the
   builds whose AIU bulk load corrupts scattered lists.

2. The one-token-per-page view ``forward_qsa`` hands FA3 preserves physical
   addressing: token ``abs`` at ``[abs // block_size, abs % block_size]``
   before the view sits at ``[abs, 0]`` after it. That is what makes the
   table's absolute indices valid page ids and keeps page_size off the
   multiples of 16 that route paged loads onto the AIU bulk path at runtime.

3. On PPU (FA3 wheel with the QSA op registered), the build probe passes and
   a production-shaped call -- block-16 pool viewed one-token-per-page,
   scattered table -- matches an fp32 gather reference.

CPU-only tests run anywhere; the rest need PPU's flash-attention wheel.
"""

import importlib

import pytest
import torch

BLOCK_SIZE = 16
POOL_TOKENS = 256
HEAD_DIM = 256
SOFTMAX_SCALE = HEAD_DIM ** -0.5


def _qsa_iface():
    """Load the PPU FA interface; skip where the wheel is absent.

    The module lives at ``vllm/vllm_flash_attn/ppu/`` in the source tree and
    is copied over ``vllm/vllm_flash_attn/`` at build time (setup.py
    ``copy_ppu_flash_attn_overrides``), so both layouts are tried.
    """
    for module_path in (
        "vllm.vllm_flash_attn.ppu.flash_attn_interface",
        "vllm.vllm_flash_attn.flash_attn_interface",
    ):
        try:
            iface = importlib.import_module(module_path)
        except ImportError:
            continue
        if hasattr(iface, "_QSA_PROBE_TOPK"):
            return iface
    pytest.skip("PPU flash-attention interface unavailable (needs PPU's FA whl)")


def _require_fa3_qsa_op():
    """Skip unless this build runs on a device with flash_attn_3.fwd."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA/PPU device")
    try:
        registered = hasattr(torch.ops, "flash_attn_3") and hasattr(
            torch.ops.flash_attn_3, "fwd"
        )
    except (RuntimeError, AttributeError):
        registered = False
    if not registered:
        pytest.skip("flash_attn_3.fwd is not registered in this build")


def test_probe_topk_list_is_scattered_and_unsorted():
    """A contiguous or sorted list would pass on AIU-corrupted builds."""
    iface = _qsa_iface()
    entries = list(iface._QSA_PROBE_TOPK)
    assert len(entries) >= 16
    assert len(set(entries)) == len(entries), "duplicate entries waste width"
    assert entries != sorted(entries), "the list must be unsorted"
    ordered = sorted(entries)
    assert all(
        later - earlier >= 2 for earlier, later in zip(ordered, ordered[1:])
    ), "adjacent entries let a contiguous-run load match by accident"
    assert all(0 <= entry < POOL_TOKENS for entry in entries)


def test_one_token_per_page_view_preserves_addressing():
    """forward_qsa's view must map absolute token indices to the same memory."""
    # PPU processes may default to a non-CPU device; pin every allocation.
    gen = torch.Generator(device="cpu").manual_seed(0x5A1)
    num_blocks = POOL_TOKENS // BLOCK_SIZE
    # vLLM hands the layer (num_blocks, num_kv_heads, block_size, 2*head_dim);
    # mirror the production unpack: transpose(1, 2).split(head_dim, dim=-1).
    kv_cache = torch.randn(
        num_blocks, 1, BLOCK_SIZE, 2 * HEAD_DIM, generator=gen, device="cpu"
    )
    key_cache, _ = kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
    assert key_cache.stride(-1) == 1, "the FA3 wrapper requires a unit last stride"

    viewed = key_cache.view(-1, 1, *key_cache.shape[2:])
    assert viewed.shape == (POOL_TOKENS, 1, 1, HEAD_DIM)
    assert viewed.stride(-1) == 1
    # Scattered and boundary absolute indices, the addressing a QSA table holds.
    for absolute in (0, 1, BLOCK_SIZE - 1, BLOCK_SIZE, BLOCK_SIZE + 1, 230, 255):
        block, offset = divmod(absolute, BLOCK_SIZE)
        assert torch.equal(viewed[absolute, 0], key_cache[block, offset])


def test_scattered_list_exposes_contiguous_run_load():
    """Model the AIU bulk load on the probe's inputs: it must miss by O(1).

    The AIU paged-KV path reads one table entry per 16-column group and
    fetches the 16 tokens contiguous with it. On the probe's scattered list
    that loads mostly wrong K/V, so the result lands far past the probe's 0.1
    hard cap -- the property the numerical self-test relies on to catch
    corrupted builds.
    """
    iface = _qsa_iface()
    gen = torch.Generator(device="cpu").manual_seed(0x5A1)
    q = torch.randn(2, HEAD_DIM, generator=gen, device="cpu")
    pool_k = torch.randn(POOL_TOKENS, HEAD_DIM, generator=gen, device="cpu")
    pool_v = torch.randn(POOL_TOKENS, HEAD_DIM, generator=gen, device="cpu")
    table = torch.tensor(iface._QSA_PROBE_TOPK)

    # Gather reference: what the correct per-column load computes.
    scores = (q @ pool_k[table].t()) * SOFTMAX_SCALE
    ref = torch.softmax(scores, dim=-1) @ pool_v[table]

    # AIU-style load: per 16-column group, read the first entry and fetch the
    # 16 tokens contiguous with it, ignoring the other 15 entries.
    groups = table.view(-1, BLOCK_SIZE)
    aiu_k = torch.cat([pool_k[head : head + BLOCK_SIZE] for head in groups[:, 0]])
    aiu_v = torch.cat([pool_v[head : head + BLOCK_SIZE] for head in groups[:, 0]])
    scores = (q @ aiu_k.t()) * SOFTMAX_SCALE
    aiu_out = torch.softmax(scores, dim=-1) @ aiu_v

    diff = (aiu_out - ref).abs().max().item()
    assert diff > 0.1, (
        f"the probe's scattered list survives a contiguous-run load "
        f"(max diff {diff:.3g}); the self-test can no longer catch "
        "AIU-corrupted builds"
    )


def test_wrapper_rejects_two_dimensional_table():
    """A 2-D table would be read as an ordinary page table: dense attention."""
    iface = _qsa_iface()
    gen = torch.Generator(device="cpu").manual_seed(0x5A1)
    q = torch.randn(1, 2, HEAD_DIM, generator=gen, device="cpu")
    cache = torch.randn(
        POOL_TOKENS // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM,
        generator=gen, device="cpu",
    )
    flat_table = torch.zeros(1, BLOCK_SIZE, dtype=torch.int32, device="cpu")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
    with pytest.raises(ValueError, match="2-D table"):
        iface.flash_attn_qsa_varlen_func(
            q, cache, cache, flat_table, cu_seqlens_q, 1
        )


def test_probe_fa3_qsa_passes_on_one_token_per_page():
    """The build probe must let the CUDA path through on a healthy build."""
    iface = _qsa_iface()
    _require_fa3_qsa_op()
    ok, reason = iface._probe_fa3_qsa()
    assert ok, f"QSA build probe failed: {reason}"


def test_one_token_per_page_call_matches_reference():
    """Production call pattern: block-16 pool, viewed, scattered table."""
    iface = _qsa_iface()
    _require_fa3_qsa_op()
    dev = torch.cuda.current_device()
    gen = torch.Generator(device="cpu").manual_seed(0x5A1)
    q = torch.randn(1, 2, HEAD_DIM, generator=gen, device="cpu").to(
        device=dev, dtype=torch.bfloat16
    )
    pool_k = torch.randn(
        POOL_TOKENS // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM,
        generator=gen, device="cpu",
    ).to(device=dev, dtype=torch.bfloat16)
    pool_v = torch.randn(
        POOL_TOKENS // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM,
        generator=gen, device="cpu",
    ).to(device=dev, dtype=torch.bfloat16)
    topk_table = torch.tensor(
        iface._QSA_PROBE_TOPK, dtype=torch.int32, device=dev
    ).view(1, 1, -1)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)

    out, _lse = iface.flash_attn_qsa_varlen_func(
        q,
        pool_k.view(-1, 1, *pool_k.shape[2:]),
        pool_v.view(-1, 1, *pool_v.shape[2:]),
        topk_table,
        cu_seqlens_q,
        1,
        softmax_scale=SOFTMAX_SCALE,
        return_softmax_lse=True,
    )
    torch.cuda.synchronize()

    # fp32 gather reference against the ORIGINAL block-16 layout: the absolute
    # indices the table holds address tokens, which the view maps to pages.
    idx = topk_table.view(-1).long()
    k_sel = pool_k.reshape(-1, HEAD_DIM)[idx]
    v_sel = pool_v.reshape(-1, HEAD_DIM)[idx]
    scores = (q[0].float() @ k_sel.float().t()) * SOFTMAX_SCALE
    ref = torch.softmax(scores, dim=-1) @ v_sel.float()

    diff = (out[0].float() - ref).abs().max().item()
    assert diff <= 0.1, (
        f"one-token-per-page call missed the gather reference: {diff:.3g}"
    )
