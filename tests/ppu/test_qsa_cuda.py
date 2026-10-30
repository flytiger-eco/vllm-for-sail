# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UT for the Qwen4Exp QSA CUDA path (PPU FA3 kernel integration).

Covers the patch's contracts:

1. The build probe's top-k list is scattered and unsorted, so a load path
   that fetches contiguous runs cannot match the numerical reference by
   accident -- a contiguous list (the original probe) passes exactly on the
   builds whose AIU bulk load corrupts scattered lists.

2. ``qsa_allow_aiu`` routes QSA between the two paged-KV load semantics:
   False (the production default) honors each table entry individually;
   True rides the AIU bulk load that reads one entry per 16-column group and
   fetches the 16 tokens contiguous with it. Wheels predating the argument
   reach the False routing through a one-token-per-page view of the pool
   applied inside ``flash_attn_qsa_varlen_func`` -- a view that merges the
   (block, token) dims, so it exists only for one KV head per rank, which
   ``qsa_cuda_unsupported_reason`` gates on.

3. On PPU (FA3 wheel with the QSA op registered), the build probe passes and
   a production-shaped call -- block-16 pool as given, scattered table,
   ``qsa_allow_aiu=False`` -- matches an fp32 gather reference on either
   wheel generation.

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
    """The legacy-wheel view must map absolute indices to the same memory.

    Wheels predating ``qsa_allow_aiu`` are routed off the AIU bulk load by
    viewing the block-16 pool one-token-per-page inside the wrapper. The
    view keeps token ``abs`` at ``[abs // block_size, abs % block_size]``
    addressable as ``[abs, 0]``, which is what makes the table's absolute
    indices valid page ids under it.
    """
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
    corrupted builds, and the reason ``qsa_allow_aiu`` defaults to False.
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


def test_probe_fa3_qsa_passes():
    """The build probe must let the CUDA path through on a healthy build.

    The probe follows the wheel: ``qsa_allow_aiu=False`` on the real paged
    cache where the argument exists, the one-token-per-page view otherwise.
    """
    iface = _qsa_iface()
    _require_fa3_qsa_op()
    ok, reason = iface._probe_fa3_qsa()
    assert ok, f"QSA build probe failed: {reason}"


def test_production_call_matches_reference():
    """Production call pattern: block-16 pool as given, scattered table.

    ``forward_qsa`` hands the wrapper the real (block, 16) pool with
    ``qsa_allow_aiu=False``; the wrapper routes by wheel (the flag to the
    kernel, a one-token-per-page view on legacy builds), so the same call
    must match the gather reference on either generation.
    """
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
        pool_k,
        pool_v,
        topk_table,
        cu_seqlens_q,
        1,
        softmax_scale=SOFTMAX_SCALE,
        qsa_allow_aiu=False,
        return_softmax_lse=True,
    )
    torch.cuda.synchronize()

    # fp32 gather reference against the block-16 pool: the absolute indices
    # the table holds address tokens in it, whichever layout the wrapper
    # handed the kernel.
    idx = topk_table.view(-1).long()
    k_sel = pool_k.reshape(-1, HEAD_DIM)[idx]
    v_sel = pool_v.reshape(-1, HEAD_DIM)[idx]
    scores = (q[0].float() @ k_sel.float().t()) * SOFTMAX_SCALE
    ref = torch.softmax(scores, dim=-1) @ v_sel.float()

    diff = (out[0].float() - ref).abs().max().item()
    assert diff <= 0.1, (
        f"production call missed the gather reference: {diff:.3g}"
    )


def test_flag_detection_matches_op_schema():
    """The flag probe must mirror what the op's schema actually carries."""
    iface = _qsa_iface()
    _require_fa3_qsa_op()
    schema = torch.ops.flash_attn_3.fwd.default._schema
    has_arg = any(arg.name == "qsa_allow_aiu" for arg in schema.arguments)
    assert iface.fa3_qsa_uses_allow_aiu_flag() == has_arg


def test_legacy_wheel_rejects_allow_aiu_true(monkeypatch):
    """A wheel without the argument must refuse the certification, not drop it."""
    iface = _qsa_iface()
    monkeypatch.setattr(iface, "fa3_qsa_uses_allow_aiu_flag", lambda: False)
    gen = torch.Generator(device="cpu").manual_seed(0x5A1)
    q = torch.randn(1, 2, HEAD_DIM, generator=gen, device="cpu")
    cache = torch.randn(
        POOL_TOKENS // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM,
        generator=gen, device="cpu",
    )
    topk_table = torch.zeros(1, 1, BLOCK_SIZE, dtype=torch.int32, device="cpu")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
    with pytest.raises(ValueError, match="qsa_allow_aiu=True requires"):
        iface.flash_attn_qsa_varlen_func(
            q, cache, cache, topk_table, cu_seqlens_q, 1, qsa_allow_aiu=True
        )


def test_allow_aiu_flag_flips_load_semantics():
    """On a flag wheel the same scattered list must change with the flag.

    False honors every entry (the gather reference); True rides the AIU bulk
    load, which reads one entry per 16-column group and fetches the 16 tokens
    contiguous with it -- the wrong K/V for this list, so the output must
    diverge. The flag is the router between the two load semantics, not
    advice.
    """
    iface = _qsa_iface()
    _require_fa3_qsa_op()
    if not iface.fa3_qsa_uses_allow_aiu_flag():
        pytest.skip("needs an FA3 build with the qsa_allow_aiu argument")
    dev = torch.cuda.current_device()
    gen = torch.Generator(device="cpu").manual_seed(0x5A1)
    q = torch.randn(1, 2, HEAD_DIM, generator=gen, device="cpu").to(
        device=dev, dtype=torch.bfloat16
    )
    pool_k = torch.randn(
        POOL_TOKENS, 1, 1, HEAD_DIM, generator=gen, device="cpu",
    ).to(device=dev, dtype=torch.bfloat16)
    pool_v = torch.randn(
        POOL_TOKENS, 1, 1, HEAD_DIM, generator=gen, device="cpu",
    ).to(device=dev, dtype=torch.bfloat16)
    paged_k = pool_k.view(POOL_TOKENS // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM)
    paged_v = pool_v.view(POOL_TOKENS // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM)
    topk_table = torch.tensor(
        iface._QSA_PROBE_TOPK, dtype=torch.int32, device=dev
    ).view(1, 1, -1)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)

    out_safe, _ = iface.flash_attn_qsa_varlen_func(
        q, paged_k, paged_v, topk_table, cu_seqlens_q, 1,
        softmax_scale=SOFTMAX_SCALE, qsa_allow_aiu=False,
        return_softmax_lse=True,
    )
    out_aiu, _ = iface.flash_attn_qsa_varlen_func(
        q, paged_k, paged_v, topk_table, cu_seqlens_q, 1,
        softmax_scale=SOFTMAX_SCALE, qsa_allow_aiu=True,
        return_softmax_lse=True,
    )
    torch.cuda.synchronize()

    idx = topk_table.view(-1).long()
    k_sel = pool_k.reshape(-1, HEAD_DIM)[idx]
    v_sel = pool_v.reshape(-1, HEAD_DIM)[idx]
    scores = (q[0].float() @ k_sel.float().t()) * SOFTMAX_SCALE
    ref = torch.softmax(scores, dim=-1) @ v_sel.float()

    diff_safe = (out_safe[0].float() - ref).abs().max().item()
    diff_aiu = (out_aiu[0].float() - ref).abs().max().item()
    assert diff_safe <= 0.1, (
        f"qsa_allow_aiu=False missed the gather reference: {diff_safe:.3g}"
    )
    assert diff_aiu > 0.1, (
        f"qsa_allow_aiu=True matched the reference (max diff {diff_aiu:.3g}); "
        "the flag no longer routes QSA onto the AIU bulk load"
    )


def _qsa_cuda_module():
    """Load the CUDA QSA module; skip outside a full vLLM environment."""
    try:
        return importlib.import_module("vllm.models.qwen4_exp.nvidia.qsa_cuda")
    except Exception as e:  # noqa: BLE001 - the import chain needs all of vllm
        pytest.skip(f"cannot import the CUDA QSA module here: {e}")


def _stub_qsa_entry_points(monkeypatch, *, allow_aiu_flag: bool) -> None:
    """Point qsa_cuda_unsupported_reason's late import at stub entry points.

    The check imports the QSA trio from ``vllm.vllm_flash_attn`` at call time,
    so the stubs land in that module; ``raising=False`` covers non-PPU builds
    where the trio does not exist there.
    """
    import vllm.vllm_flash_attn as vfa

    monkeypatch.setattr(
        vfa, "fa3_qsa_uses_allow_aiu_flag", lambda: allow_aiu_flag, raising=False
    )
    monkeypatch.setattr(
        vfa, "is_fa3_qsa_supported", lambda device=None: True, raising=False
    )
    monkeypatch.setattr(
        vfa, "fa3_qsa_unsupported_reason", lambda device=None: None, raising=False
    )


def test_legacy_wheel_gate_rejects_extra_kv_heads(monkeypatch):
    """Without the flag argument, more than one KV head per rank must not run.

    The legacy fallback views the (block, token) dims into one page dim, a
    merge whose strides only exist for one KV head per rank.
    """
    qsa_cuda = _qsa_cuda_module()
    monkeypatch.setattr(qsa_cuda.current_platform, "is_ppu", lambda: True)
    _stub_qsa_entry_points(monkeypatch, allow_aiu_flag=False)
    reason = qsa_cuda.qsa_cuda_unsupported_reason(
        head_dim=HEAD_DIM, num_heads=16, num_kv_heads=2
    )
    assert reason is not None
    assert "lacks qsa_allow_aiu" in reason


def test_flag_wheel_accepts_extra_kv_heads(monkeypatch):
    """With the flag argument the view constraint is gone: the kernel routes."""
    qsa_cuda = _qsa_cuda_module()
    monkeypatch.setattr(qsa_cuda.current_platform, "is_ppu", lambda: True)
    _stub_qsa_entry_points(monkeypatch, allow_aiu_flag=True)
    assert qsa_cuda.qsa_cuda_unsupported_reason(
        head_dim=HEAD_DIM, num_heads=16, num_kv_heads=2
    ) is None
