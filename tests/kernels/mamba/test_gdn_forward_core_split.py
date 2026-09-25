# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration test for the non-spec decode split in
``GatedDeltaNet._forward_core``.

On a pure non-spec batch that mixes prefills with 1-token decodes, the layer
peels the decodes (the contiguous decode-first front slice) off to
``fused_sigmoid_gating_delta_rule_update`` -- the same recurrent update kernel
the spec-decode path uses -- and runs only the prefill tail through
``chunk_gated_delta_rule``. This must produce the same core-attention output and
the same ssm-state pool update as running *everything* through
``chunk_gated_delta_rule`` (the previous behavior).

Both paths are exercised through the REAL ``_forward_core``:

* ``meta_split`` is built by the real ``GDNAttentionMetadataBuilder`` for a
  mixed batch, so ``num_decodes > 0`` triggers the peel (and the builder rebases
  ``chunk_indices``/``chunk_offsets`` to the prefill-only tail).
* ``meta_unified`` is the same metadata with the decodes reclassified as
  prefills and full-batch chunk metadata, which forces ``_forward_core`` through
  the existing chunk-only path on identical inputs (the conv is unified over all
  non-spec tokens in both paths, so it cancels out and only the recurrent split
  is compared).

The Triton/FLA chunk backend is forced so the prefill-only ``chunk_indices``
must stay consistent with the rebased ``cu_seqlens`` (a stringent, backend
portable check of the split wiring).

CPU routing tests also cover mixed spec/prefill batches with lightweight
kernel stand-ins, checking that token partitioning keeps QKV and gates aligned.
"""

from __future__ import annotations

import dataclasses
import types
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.utils import (
    FLA_CHUNK_SIZE,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

# Small GDN dims; head_k_dim/head_v_dim=128 keeps the chunk/update kernels happy.
H = 4  # num key heads
HV = 8  # num value heads
K = 128  # head_k_dim
V = 128  # head_v_dim
CONV_KERNEL = 4
KEY_DIM = H * K
VALUE_DIM = HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"


def _make_vllm_config():
    # A small, ungated GDN model whose config is cached locally; only the config
    # (scheduler/cache/compilation/hf) is used here, never the weights. Inject
    # linear_key_head_dim=128 and request the CuteDSL prefill backend -- the
    # supported GDN chunk kernel on Blackwell (the Triton/FLA chunk kernel is
    # unsupported on SM10x). CuteDSL consumes chunk_indices/chunk_offsets, so
    # this also exercises the prefill-only chunk-metadata wiring.
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.additional_config = {"gdn_prefill_backend": "cutedsl"}
    return cfg


def _build_layer(
    vllm_config, conv_state, ssm_state, A_log, dt_bias, conv_weight, conv_bias
):
    """A minimal object that runs the real ``_forward_core`` bound to it."""
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = False
    layer.tp_size = 1
    layer.num_k_heads = H
    layer.num_v_heads = HV
    layer.head_k_dim = K
    layer.head_v_dim = V
    layer.key_dim = KEY_DIM
    layer.value_dim = VALUE_DIM
    layer.activation = "silu"
    layer.A_log = A_log
    layer.dt_bias = dt_bias
    layer.conv1d = types.SimpleNamespace(weight=conv_weight, bias=conv_bias)
    layer.kv_cache = (conv_state, ssm_state)
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in (
        "rearrange_mixed_qkv",
        "_forward_core",
    ):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def _run_forward_core(layer, meta, mixed_qkv, b, a, num_tokens):
    core_attn_out = torch.zeros(
        num_tokens,
        layer.num_v_heads // layer.tp_size,
        layer.head_v_dim,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b.clone(),
            a=a.clone(),
            core_attn_out=core_attn_out,
        )
    return core_attn_out


@pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ),
    reason="CuteDSL prefill requires CUDA SM10x",
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_decodes,prefill_lens", [(3, [512, 300]), (4, [64, 5])])
@pytest.mark.parametrize("fresh_prefill", [False, True])
def test_forward_core_split_matches_unified(
    state_dtype: torch.dtype,
    num_decodes: int,
    prefill_lens: list[int],
    fresh_prefill: bool,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config()

    # Decode-first batch: D 1-token decodes (with context), then the prefills.
    decode_seq_lens = [64] * num_decodes
    prefill_seq_lens = [
        pl if (fresh_prefill and i == 0) else pl + 37
        for i, pl in enumerate(prefill_lens)
    ]
    seq_lens = decode_seq_lens + prefill_seq_lens
    query_lens = [1] * num_decodes + list(prefill_lens)
    batch = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)

    builder = GDNAttentionMetadataBuilder(
        # GDNAttentionMetadataBuilder declares kv_cache_spec as AttentionSpec but
        # asserts isinstance(kv_cache_spec, MambaSpec). MambaSpec is correct here.
        kv_cache_spec=MambaSpec(  # type: ignore[arg-type]
            block_size=BLOCK_SIZE, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    with set_current_vllm_config(vllm_config):
        meta_split = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert meta_split.spec_sequence_masks is None
    assert meta_split.num_decodes == num_decodes
    assert meta_split.num_prefills == len(prefill_lens)
    assert meta_split.num_decode_tokens == num_decodes
    assert builder.gdn_prefill_backend == "cutedsl"

    num_tokens = sum(query_lens)

    # Full-batch chunk metadata for the unified reference path, built the same
    # way the builder would for a non-split batch (backend-matched).
    cu_full = meta_split.non_spec_query_start_loc
    assert cu_full is not None
    if builder.gdn_prefill_backend == "cutedsl":
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            prepare_metadata_cutedsl,
        )

        full_ci, full_co = prepare_metadata_cutedsl(
            cu_full, int(cu_full[-1].item()), FLA_CHUNK_SIZE
        )
    else:
        cu_full_cpu = cu_full.cpu()
        full_ci = prepare_chunk_indices(cu_full_cpu, FLA_CHUNK_SIZE).to(device)
        full_co = prepare_chunk_offsets(cu_full_cpu, FLA_CHUNK_SIZE).to(device)
    meta_unified = dataclasses.replace(
        meta_split,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=meta_split.num_decodes + meta_split.num_prefills,
        num_prefill_tokens=(
            meta_split.num_decode_tokens + meta_split.num_prefill_tokens
        ),
        chunk_indices=full_ci,
        chunk_offsets=full_co,
        # Unified path: the chunk kernel processes the full non-spec batch.
        prefill_query_start_loc=meta_split.non_spec_query_start_loc,
        prefill_state_indices=meta_split.non_spec_state_indices_tensor,
        prefill_has_initial_state=meta_split.has_initial_state,
    )

    # Size the state pools from the indices the builder actually produced.
    assert meta_split.non_spec_state_indices_tensor is not None
    pool_size = int(meta_split.non_spec_state_indices_tensor.max().item()) + 1
    conv_state_shape, temporal_state_shape = (
        MambaStateShapeCalculator.gated_delta_net_state_shape(
            1, H, HV, K, V, CONV_KERNEL, num_spec=0
        )
    )
    conv_state0 = (
        torch.randn(pool_size, *conv_state_shape, dtype=torch.bfloat16, device=device)
        * 0.05
    )
    ssm_state0 = (
        torch.randn(pool_size, *temporal_state_shape, dtype=state_dtype, device=device)
        * 0.05
    )

    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    conv_weight = (
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device) * 0.1
    )
    conv_bias = torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1

    mixed_qkv = (
        torch.randn(num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1
    )
    a = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1

    # ---- Split path (real _forward_core, meta_split) ----
    conv_state_split = conv_state0.clone()
    ssm_state_split = ssm_state0.clone()
    layer_split = _build_layer(
        vllm_config,
        conv_state_split,
        ssm_state_split,
        A_log,
        dt_bias,
        conv_weight,
        conv_bias,
    )
    out_split = _run_forward_core(layer_split, meta_split, mixed_qkv, b, a, num_tokens)

    # ---- Unified path (real _forward_core, meta_unified) ----
    conv_state_unified = conv_state0.clone()
    ssm_state_unified = ssm_state0.clone()
    layer_unified = _build_layer(
        vllm_config,
        conv_state_unified,
        ssm_state_unified,
        A_log,
        dt_bias,
        conv_weight,
        conv_bias,
    )
    out_unified = _run_forward_core(
        layer_unified, meta_unified, mixed_qkv, b, a, num_tokens
    )

    # Conv is unified in both paths, so the conv-state update must be identical.
    torch.testing.assert_close(conv_state_split, conv_state_unified, atol=0, rtol=0)

    # Chunk vs. recurrent update accumulate in different orders; mirror the
    # tolerances used by the kernel-level parity test.
    if state_dtype == torch.float32:
        atol = rtol = 2e-2
    else:
        atol = rtol = 6e-2
    torch.testing.assert_close(out_split, out_unified, atol=atol, rtol=rtol)
    torch.testing.assert_close(ssm_state_split, ssm_state_unified, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "spec_mask", [[True, False, True], [False, True, True], [True, True]]
)
def test_forward_core_spec_gates_follow_token_order(monkeypatch, spec_mask):
    """Partitioning must preserve each token's QKV/gates association.

    Use CPU stand-ins for the kernels: the routing is the behavior under test,
    and distinct per-token gates make incorrect prefix reads observable.
    """
    mask = torch.tensor(spec_mask)
    spec_tokens = mask.repeat_interleave(3).nonzero().flatten()
    non_spec_tokens = (~mask).repeat_interleave(3).nonzero().flatten()
    num_spec, num_prefill = sum(spec_mask), len(spec_mask) - sum(spec_mask)
    num_tokens = len(spec_mask) * 3
    prefill_indices = (~mask).nonzero().flatten() * 3 + 1
    prefill_cu = torch.arange(num_prefill + 1, dtype=torch.int32) * 3
    meta = GDNAttentionMetadata(
        num_prefills=num_prefill,
        num_prefill_tokens=3 * num_prefill,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_spec,
        num_spec_decode_tokens=3 * num_spec,
        num_actual_tokens=num_tokens,
        has_initial_state=torch.ones(num_prefill, dtype=torch.bool),
        spec_query_start_loc=torch.arange(num_spec + 1, dtype=torch.int32) * 3,
        non_spec_query_start_loc=prefill_cu,
        spec_state_indices_tensor=torch.arange(
            1, num_tokens + 1, dtype=torch.int32
        ).reshape(-1, 3)[mask],
        non_spec_state_indices_tensor=prefill_indices,
        spec_sequence_masks=mask,
        spec_token_indx=spec_tokens,
        non_spec_token_indx=non_spec_tokens,
        num_accepted_tokens=torch.ones(num_spec, dtype=torch.int32),
        prefill_query_start_loc=prefill_cu,
        prefill_state_indices=prefill_indices,
        prefill_has_initial_state=torch.ones(num_prefill, dtype=torch.bool),
    )

    def recurrent(*, q, k, v, a, b, **kwargs):
        # Device kernels consume the first T gate rows; extra rows may not
        # trigger a shape error, so compare values rather than just shapes.
        t = q.shape[1]
        return q + a[:t][None, :, :, None] * k + b[:t][None, :, :, None] * v, None

    def post_conv(*, conv_output, a, b, **kwargs):
        q, k, v = conv_output.reshape(-1, 3, 1, 1).unbind(1)
        return q, k, v, a, b

    def chunk(*, q, k, v, g, beta, initial_state, **kwargs):
        return q + g.unsqueeze(-1) * k + beta.unsqueeze(-1) * v, initial_state

    for name in ("causal_conv1d_update", "causal_conv1d_fn"):
        monkeypatch.setattr(qwen_gdn_linear_attn, name, lambda x, *args, **kw: x)
    monkeypatch.setattr(qwen_gdn_linear_attn, "fused_post_conv_prep", post_conv)
    monkeypatch.setattr(
        qwen_gdn_linear_attn, "fused_sigmoid_gating_delta_rule_update", recurrent
    )
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        tp_size=1,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=1,
        head_v_dim=1,
        key_dim=1,
        value_dim=1,
        activation="silu",
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        conv1d=types.SimpleNamespace(weight=torch.ones(3, 1, 1), bias=None),
        kv_cache=(
            torch.zeros(num_tokens + 1, 3, 3),
            torch.zeros(num_tokens + 1, 1, 1, 1),
        ),
        chunk_gated_delta_rule=chunk,
    )
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    mixed_qkv = torch.arange(1, num_tokens * 3 + 1, dtype=torch.float32).reshape(-1, 3)
    a = torch.arange(1, num_tokens + 1, dtype=torch.float32).unsqueeze(-1)
    b = 10 * a - 3
    actual = _run_forward_core(layer, meta, mixed_qkv, b, a, num_tokens)
    expected = mixed_qkv[:, :1] + a * mixed_qkv[:, 1:2] + b * mixed_qkv[:, 2:3]
    torch.testing.assert_close(actual, expected.unsqueeze(-1), atol=0, rtol=0)
