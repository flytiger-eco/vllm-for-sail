# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity, fallback and benchmark tests for the PPU SAIL CUDA PLA decode path.

With ``VLLM_PPU_USE_PLA`` enabled, the three GDN decode PLA wrappers
route from the community Triton kernels to the PPU SAIL CUDA kernels
(``k_last`` / ``k_last_packed``) provided by the external ``pla`` package:

* ``fused_recurrent_gated_delta_rule_packed_decode`` -> ``k_last_packed``
* ``fused_sigmoid_gating_delta_rule_update``         -> ``k_last``

These tests monkeypatch ``sail_cuda_pla``'s resolved kernel handles to force
each implementation, then compare:

1. numerical parity of the attention output and the updated fp32 ssm state
   pool between the CUDA and Triton implementations;
2. the per-call gating: calls that violate the CUDA kernel constraints
   (bf16 state pool, vLLM-style speculative decoding, head dims != 128)
   must fall back to Triton, i.e. produce bitwise-identical results to the
   forced-Triton path;
3. the NULL_BLOCK_ID=0 isolation: a padded request (CUDA-graph padding) must
   not disturb the outputs/states of real requests;
4. kernel-level wall-clock time and peak memory of both implementations
   (recorded via stdout, no hard perf assertions).

The kernel tests require the PPU platform with the ``pla`` package installed;
the env-var resolution tests are platform-independent.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm import envs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
    sail_cuda_pla,
)

DEVICE = torch.device("cuda")

# Small GDN dims; K == V == 128 matches the CUDA kernel's hard requirement.
H = 4  # num key heads
HV = 8  # num value heads
K = 128  # head_k_dim
V = 128  # head_v_dim


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _requires_ppu_pla():
    """Skip unless on PPU with the SAIL ``pla`` package; return the package."""
    if not current_platform.is_ppu():
        pytest.skip("PPU SAIL CUDA PLA tests require the PPU platform")
    return pytest.importorskip("pla.decode", reason="PPU SAIL `pla` package not found")


def _force_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the wrappers onto the community Triton kernels."""
    monkeypatch.setattr(sail_cuda_pla, "_resolved", True)
    monkeypatch.setattr(sail_cuda_pla, "_k_last_fn", None)
    monkeypatch.setattr(sail_cuda_pla, "_k_last_packed_fn", None)


def _force_cuda(monkeypatch: pytest.MonkeyPatch, pla_module) -> None:
    """Force the wrappers onto the PPU SAIL CUDA kernels."""
    monkeypatch.setattr(sail_cuda_pla, "_resolved", True)
    monkeypatch.setattr(
        sail_cuda_pla,
        "_k_last_fn",
        pla_module.fused_sigmoid_gating_delta_rule_forward_k_last,
    )
    monkeypatch.setattr(
        sail_cuda_pla,
        "_k_last_packed_fn",
        pla_module.fused_sigmoid_gating_delta_rule_forward_k_last_packed,
    )


def _make_decode_inputs(
    batch: int,
    *,
    state_dtype: torch.dtype = torch.float32,
    num_slots: int | None = None,
    query_lens: list[int] | None = None,
    head_dim: int = K,
    seed: int = 0,
) -> SimpleNamespace:
    """Build GDN decode inputs mirroring the production call sites.

    q/k are [1, total, H, D], v is [1, total, HV, D] (varlen-flattened), a/b
    are [total, HV], the ssm state pool is [num_slots, HV, V, K] (k-last), and
    state slot 0 is left out of the indices (vLLM's reserved NULL_BLOCK_ID).
    """
    torch.manual_seed(seed)
    num_slots = num_slots or (batch + 1)
    query_lens = query_lens or [1] * batch
    assert len(query_lens) == batch
    total = sum(query_lens)

    q = torch.randn(1, total, H, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    k = torch.randn(1, total, H, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    v = torch.randn(1, total, HV, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    a = torch.randn(total, HV, dtype=torch.bfloat16, device=DEVICE) * 0.1
    b = torch.randn(total, HV, dtype=torch.bfloat16, device=DEVICE) * 0.1
    A_log = torch.randn(HV, dtype=torch.float32, device=DEVICE) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=DEVICE) * 0.1
    ssm_state = (
        torch.randn(num_slots, HV, head_dim, head_dim, dtype=state_dtype, device=DEVICE)
        * 0.05
    )

    # Real requests occupy slots >= 1; slot 0 is the reserved null block.
    perm = torch.randperm(num_slots - 1)[:batch] + 1
    ssm_state_indices = perm.to(torch.int32).to(DEVICE)
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=DEVICE,
    )
    # packed layout: [q (H*D) | k (H*D) | v (HV*D)] along the last dim
    mixed_qkv = torch.cat(
        [
            q.reshape(total, H * head_dim),
            k.reshape(total, H * head_dim),
            v.reshape(total, HV * head_dim),
        ],
        dim=-1,
    ).contiguous()

    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        ssm_state=ssm_state,
        ssm_state_indices=ssm_state_indices,
        cu_seqlens=cu_seqlens,
        mixed_qkv=mixed_qkv,
        scale=head_dim**-0.5,
    )


def _run_packed(inp: SimpleNamespace, ssm_state: torch.Tensor) -> torch.Tensor:
    batch = inp.mixed_qkv.shape[0]
    out = torch.empty(batch, 1, HV, inp.v.shape[-1], dtype=inp.v.dtype, device=DEVICE)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=inp.mixed_qkv,
        a=inp.a,
        b=inp.b,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        scale=inp.scale,
        initial_state=ssm_state,
        out=out,
        ssm_state_indices=inp.ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return out


def _run_update(
    inp: SimpleNamespace, ssm_state: torch.Tensor, **overrides
) -> torch.Tensor:
    call_kwargs = dict(
        A_log=inp.A_log,
        a=inp.a,
        b=inp.b,
        dt_bias=inp.dt_bias,
        q=inp.q,
        k=inp.k,
        v=inp.v,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=inp.cu_seqlens,
        ssm_state_indices=inp.ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    call_kwargs.update(overrides)
    o, _ = fused_sigmoid_gating_delta_rule_update(**call_kwargs)
    return o


def _assert_parity(
    cuda_out: torch.Tensor,
    triton_out: torch.Tensor,
    cuda_state: torch.Tensor,
    triton_state: torch.Tensor,
) -> None:
    # bf16 outputs: vLLM default tolerances. fp32 state pool: both kernels
    # accumulate in fp32 and differ only in softplus/exp implementation
    # details, so a non-bitwise tolerance is used.
    torch.testing.assert_close(
        cuda_out,
        triton_out,
        atol=get_default_atol(triton_out),
        rtol=get_default_rtol(triton_out),
    )
    torch.testing.assert_close(cuda_state, triton_state, atol=1e-3, rtol=1e-2)


def _bench(fn, *, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    """Return (avg milliseconds per call, peak CUDA memory in MiB)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1 << 20)
    return start.elapsed_time(end) / iters, peak_mib


# ---------------------------------------------------------------------------
# env var resolution (platform-independent)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "new_val,expected",
    [
        (None, True),  # default: on
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
    ],
)
def test_ppu_pla_cuda_env_resolution(
    monkeypatch: pytest.MonkeyPatch,
    new_val: str | None,
    expected: bool,
) -> None:
    if new_val is None:
        monkeypatch.delenv("VLLM_PPU_USE_PLA", raising=False)
    else:
        monkeypatch.setenv("VLLM_PPU_USE_PLA", new_val)

    # Call the resolver lambda directly: envs.__getattr__ may be wrapped in
    # functools.cache after service init, which would bypass monkeypatched
    # env vars.
    resolve = envs.environment_variables["VLLM_PPU_USE_PLA"]
    assert resolve() is expected



# ---------------------------------------------------------------------------
# CUDA vs Triton numerical parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [1, 4, 32])
def test_packed_decode_cuda_matches_triton(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    pla_module = _requires_ppu_pla()
    inp = _make_decode_inputs(batch, state_dtype=torch.float32, seed=batch)

    _force_triton(monkeypatch)
    state_t = inp.ssm_state.clone()
    out_t = _run_packed(inp, state_t)

    _force_cuda(monkeypatch, pla_module)
    state_c = inp.ssm_state.clone()
    out_c = _run_packed(inp, state_c)

    _assert_parity(out_c, out_t, state_c, state_t)


@pytest.mark.parametrize(
    "query_lens",
    [
        [1, 1, 1, 1],  # pure decode batch
        [2, 1, 3],  # varlen (mixed token counts per request)
    ],
)
def test_update_cuda_matches_triton(
    monkeypatch: pytest.MonkeyPatch, query_lens: list[int]
) -> None:
    pla_module = _requires_ppu_pla()
    batch = len(query_lens)
    inp = _make_decode_inputs(
        batch, state_dtype=torch.float32, query_lens=query_lens, seed=sum(query_lens)
    )

    _force_triton(monkeypatch)
    state_t = inp.ssm_state.clone()
    out_t = _run_update(inp, state_t)

    _force_cuda(monkeypatch, pla_module)
    state_c = inp.ssm_state.clone()
    out_c = _run_update(inp, state_c)

    _assert_parity(out_c, out_t, state_c, state_t)


# ---------------------------------------------------------------------------
# fallback gating: CUDA-eligible handles present, but the call must fall
# back to Triton -> results must be bitwise-identical to the Triton path
# ---------------------------------------------------------------------------
def test_update_falls_back_for_bf16_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    pla_module = _requires_ppu_pla()
    # The CUDA k_last kernel requires a float32 ssm state pool; a bf16 pool
    # must route back to Triton.
    inp = _make_decode_inputs(4, state_dtype=torch.bfloat16, seed=11)

    _force_triton(monkeypatch)
    state_t = inp.ssm_state.clone()
    out_t = _run_update(inp, state_t)

    _force_cuda(monkeypatch, pla_module)
    state_c = inp.ssm_state.clone()
    out_c = _run_update(inp, state_c)

    torch.testing.assert_close(out_c, out_t, atol=0, rtol=0)
    torch.testing.assert_close(state_c, state_t, atol=0, rtol=0)


def test_update_falls_back_for_spec_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pla_module = _requires_ppu_pla()
    batch, num_spec = 3, 2
    inp = _make_decode_inputs(
        batch,
        state_dtype=torch.float32,
        num_slots=batch * num_spec + 1,
        query_lens=[num_spec] * batch,
        seed=12,
    )
    # vLLM-style speculative decoding: per-token state slots (2D indices)
    # plus num_accepted_tokens. The CUDA kernel cannot express this, so the
    # wrapper must fall back to Triton.
    idx2d = (
        (torch.randperm(batch * num_spec).reshape(batch, num_spec) + 1)
        .to(torch.int32)
        .to(DEVICE)
    )
    num_accepted = torch.randint(1, num_spec + 1, (batch,), dtype=torch.int32).to(
        DEVICE
    )

    _force_triton(monkeypatch)
    state_t = inp.ssm_state.clone()
    out_t = _run_update(
        inp, state_t, ssm_state_indices=idx2d, num_accepted_tokens=num_accepted
    )

    _force_cuda(monkeypatch, pla_module)
    state_c = inp.ssm_state.clone()
    out_c = _run_update(
        inp, state_c, ssm_state_indices=idx2d, num_accepted_tokens=num_accepted
    )

    torch.testing.assert_close(out_c, out_t, atol=0, rtol=0)
    torch.testing.assert_close(state_c, state_t, atol=0, rtol=0)


def test_update_falls_back_for_non_128_head_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pla_module = _requires_ppu_pla()
    # The CUDA kernel hard-codes K == V == 128; other head dims must fall back.
    inp = _make_decode_inputs(4, state_dtype=torch.float32, head_dim=64, seed=13)

    _force_triton(monkeypatch)
    state_t = inp.ssm_state.clone()
    out_t = _run_update(inp, state_t)

    _force_cuda(monkeypatch, pla_module)
    state_c = inp.ssm_state.clone()
    out_c = _run_update(inp, state_c)

    torch.testing.assert_close(out_c, out_t, atol=0, rtol=0)
    torch.testing.assert_close(state_c, state_t, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------
def test_packed_null_state_idx_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NULL_BLOCK_ID=0 (CUDA-graph padding) request must not disturb the
    real requests in the same batch on the CUDA path.

    Note the semantic difference vs Triton: the CUDA kernel treats slot 0 as
    a valid slot (reads/writes it), while the Triton kernel skips idx <= 0.
    Slot 0 is vLLM's reserved null block that real requests never use, so
    this is safe; this test pins the isolation property.
    """
    pla_module = _requires_ppu_pla()
    batch = 4
    inp = _make_decode_inputs(batch, state_dtype=torch.float32, num_slots=8, seed=14)
    padded_idx = inp.ssm_state_indices.clone()
    padded_idx[1] = 0  # NULL_BLOCK_ID: padded request under CUDA graphs

    _force_cuda(monkeypatch, pla_module)
    state_pad = inp.ssm_state.clone()
    inp_padded = SimpleNamespace(**{**vars(inp), "ssm_state_indices": padded_idx})
    out_pad = _run_packed(inp_padded, state_pad)

    # Reference: the same batch without the padded request.
    keep = [0, 2, 3]
    inp_ref = SimpleNamespace(
        **{
            **vars(inp),
            "mixed_qkv": inp.mixed_qkv[keep].contiguous(),
            "a": inp.a[keep].contiguous(),
            "b": inp.b[keep].contiguous(),
            "ssm_state_indices": inp.ssm_state_indices[keep].clone(),
            "cu_seqlens": torch.arange(len(keep) + 1, dtype=torch.int32, device=DEVICE),
        }
    )
    state_ref = inp.ssm_state.clone()
    out_ref = _run_packed(inp_ref, state_ref)

    # Real requests: bitwise-identical outputs and state updates (same CUDA
    # kernel, deterministic per (request, head) program).
    torch.testing.assert_close(out_pad[keep], out_ref, atol=0, rtol=0)
    used_slots = inp.ssm_state_indices[keep].long()
    torch.testing.assert_close(
        state_pad[used_slots], state_ref[used_slots], atol=0, rtol=0
    )
    # ... while slot 0 IS touched by the CUDA kernel (documented difference
    # from the Triton kernel, which skips idx <= 0 entirely).
    assert not torch.equal(state_pad[0], inp.ssm_state[0])


# ---------------------------------------------------------------------------
# performance benchmarks (recorded, not asserted)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch", [32, 128])
def test_packed_decode_perf(monkeypatch: pytest.MonkeyPatch, batch: int) -> None:
    pla_module = _requires_ppu_pla()
    inp = _make_decode_inputs(batch, state_dtype=torch.float32, seed=21)

    _force_triton(monkeypatch)
    state = inp.ssm_state.clone()
    t_ms, t_mem = _bench(lambda: _run_packed(inp, state))

    _force_cuda(monkeypatch, pla_module)
    state = inp.ssm_state.clone()
    c_ms, c_mem = _bench(lambda: _run_packed(inp, state))

    print(
        f"\n[packed_decode perf] batch={batch}: "
        f"triton {t_ms:.3f} ms / {t_mem:.1f} MiB  vs  "
        f"cuda {c_ms:.3f} ms / {c_mem:.1f} MiB  "
        f"(speedup {t_ms / c_ms:.2f}x)"
    )


@pytest.mark.parametrize("batch", [32, 128])
def test_update_perf(monkeypatch: pytest.MonkeyPatch, batch: int) -> None:
    pla_module = _requires_ppu_pla()
    inp = _make_decode_inputs(batch, state_dtype=torch.float32, seed=22)

    _force_triton(monkeypatch)
    state = inp.ssm_state.clone()
    t_ms, t_mem = _bench(lambda: _run_update(inp, state))

    _force_cuda(monkeypatch, pla_module)
    state = inp.ssm_state.clone()
    c_ms, c_mem = _bench(lambda: _run_update(inp, state))

    print(
        f"\n[update perf] batch={batch}: "
        f"triton {t_ms:.3f} ms / {t_mem:.1f} MiB  vs  "
        f"cuda {c_ms:.3f} ms / {c_mem:.1f} MiB  "
        f"(speedup {t_ms / c_ms:.2f}x)"
    )
