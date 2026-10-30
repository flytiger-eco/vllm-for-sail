# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
# Use relative import to support build-from-source installation in vLLM

try:
    # ppu use fa2 whl, do not compile fa2_C
    import flash_attn  # noqa: F401
    FA2_UNAVAILABLE_REASON = None
    FA2_AVAILABLE = True
except ImportError as e:
    FA2_UNAVAILABLE_REASON = str(e)
    FA2_AVAILABLE = False

try:
    # ppu use fa3 whl, do not compile fa3_C
    import flash_attn_3._C   # noqa: F401
    FA3_UNAVAILABLE_REASON = None
    FA3_AVAILABLE = True
except ImportError as e:
    FA3_UNAVAILABLE_REASON = str(e)
    FA3_AVAILABLE = False

#Add for nvtx profiling
import os
_nvtx_env = os.getenv("VLLM_PPU_NVTX_PROFILE", "").strip().lower()
_nvtx_sail_env = os.getenv("VLLM_SAIL_NVTX_PROFILE", "").strip().lower()
NVTX_PROFILE = _nvtx_env in ("true", "1") or _nvtx_sail_env in ("true", "1")
_dump_env = os.getenv("VLLM_PPU_NVTX_VFA_DUMP_SEQLEN", "").strip().lower()
_dump_sail_env = os.getenv(
    "VLLM_SAIL_NVTX_VFA_DUMP_SEQLEN", ""
).strip().lower()
NVTX_PROFILE_DUMP_SEQLEN = (
    _dump_env in ("true", "1") or _dump_sail_env in ("true", "1")
)
if NVTX_PROFILE:
    try:
        from torch.cuda.nvtx import range_push as  th_nvtx_range_push
        from torch.cuda.nvtx import range_pop  as  th_nvtx_range_pop
    except ImportError:
        NVTX_PROFILE = False
        NVTX_PROFILE_DUMP_SEQLEN = False
if not NVTX_PROFILE:
    def th_nvtx_range_push(label):
        pass
    def th_nvtx_range_pop():
        pass

# isort: on

DEFAULT_FA_VERSION = 2

# Cached result of the QSA build probe; None means "not probed yet".
_FA3_QSA_SUPPORTED: Optional[bool] = None
_FA3_QSA_UNSUPPORTED_REASON: Optional[str] = None

# Top-k list for the QSA build probe's numerical self-test: scattered (no two
# entries adjacent when sorted) and unsorted, so a load path that fetches contiguous
# runs cannot match the reference by accident. Every entry is an absolute token index
# into the probe's 256-token pool; the qsa_allow_aiu layout reads them against
# 16-token pages, the legacy view layout against one-token pages. The probe appends
# _QSA_PROBE_PAD padding slots to the table and masks them through seqused_k, so a
# build that still force-nulls seqused_k for QSA walks the tail, picks up stale
# scores, and fails the comparison.
_QSA_PROBE_TOPK = (3, 40, 77, 114, 151, 188, 225, 8, 45, 82, 119, 156, 193, 230, 31, 68)
_QSA_PROBE_PAD = (-1, -1, -1, -1)

# Cached result of the qsa_allow_aiu schema check; None means "not checked yet".
_FA3_QSA_FLAG_SUPPORTED: bool | None = None


def fa3_qsa_uses_allow_aiu_flag() -> bool:
    """Whether the installed ``flash_attn_3.fwd`` carries ``qsa_allow_aiu``.

    Wheels rebuilt after that argument landed route QSA off the AIU paged-KV bulk
    load through the kernel's own runtime check. On older wheels
    :func:`flash_attn_qsa_varlen_func` reaches the same per-column load by handing
    FA3 a one-token-per-page view of the pool instead -- a workaround that only
    exists for one KV head per rank, which callers must gate on.
    """
    global _FA3_QSA_FLAG_SUPPORTED
    if _FA3_QSA_FLAG_SUPPORTED is None:
        try:
            schema = torch.ops.flash_attn_3.fwd.default._schema
            _FA3_QSA_FLAG_SUPPORTED = any(
                arg.name == "qsa_allow_aiu" for arg in schema.arguments
            )
        except (AttributeError, RuntimeError):
            _FA3_QSA_FLAG_SUPPORTED = False
    return _FA3_QSA_FLAG_SUPPORTED

def _is_fa2_supported(device = None) -> Tuple[bool, Optional[str]]:
    if not FA2_AVAILABLE:
        return False, f"FA2 is unavaible due to: {FA2_UNAVAILABLE_REASON}"
    if torch.cuda.get_device_capability(device)[0] < 8:
        return False, \
            "FA2 is only supported on devices with compute capability >= 8"
    return True, None

def _is_fa3_supported(device = None) -> Tuple[bool, Optional[str]]:
    if not FA3_AVAILABLE:
        return False, f"FA3 is unavaible due to: {FA3_UNAVAILABLE_REASON}"
    if torch.cuda.get_device_capability(device)[0] < 8 \
        or torch.cuda.get_device_capability(device)[0] >= 10 \
        or torch.cuda.get_device_capability(device) == (8, 6):
        return False, \
            "FA3 is only supported on devices with compute capability >= 8" \
            " excluding 8.6 and Blackwell archs (>=10)"
    return True, None

def is_fa_version_supported(fa_version: int, device = None) -> bool:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[0]
    elif fa_version == 3:
        return _is_fa3_supported(device)[0]


def _probe_fa3_qsa(device=None) -> Tuple[bool, Optional[str]]:
    """Detect whether the installed FA3 whl computes QSA correctly.

    QSA has no dedicated op: `fwd` picks the path from the rank of `page_table`, so a
    build without it rejects a 3-D table with a TORCH_CHECK. Passing that check is not
    enough: the probe issues one call in the exact convention the CUDA path uses and
    compares the result against a PyTorch reference on a scattered, unsorted list. On
    wheels that carry `qsa_allow_aiu` that means the real paged cache with
    `qsa_allow_aiu=False` (the kernel routes QSA off the AIU bulk load itself); on
    older wheels it means the one-token-per-page view `flash_attn_qsa_varlen_func`
    applies for them. Either way a build whose routing still lands on the bulk load
    -- one table entry read per 16-column group, 16 contiguous tokens fetched, the
    other 15 entries ignored -- cannot match the reference. The table also carries a
    padded tail masked through seqused_k, so a build whose QSA path force-nulls
    seqused_k walks the padding, picks up stale scores, and fails the same way. The
    result is cached because it is a property of the whl, not of the arguments.
    """
    global _FA3_QSA_SUPPORTED, _FA3_QSA_UNSUPPORTED_REASON
    if _FA3_QSA_SUPPORTED is not None or _FA3_QSA_UNSUPPORTED_REASON is not None:
        return _FA3_QSA_SUPPORTED, _FA3_QSA_UNSUPPORTED_REASON

    ok, reason = _is_fa3_supported(device)
    if not ok:
        _FA3_QSA_UNSUPPORTED_REASON = reason
        return False, reason
    if not hasattr(torch.ops, "flash_attn_3") or not hasattr(torch.ops.flash_attn_3, "fwd"):
        _FA3_QSA_UNSUPPORTED_REASON = "torch.ops.flash_attn_3.fwd is not registered"
        return False, _FA3_QSA_UNSUPPORTED_REASON

    softmax_scale = 256 ** -0.5
    try:
        dev = device if device is not None else torch.cuda.current_device()
        # Deterministic inputs, generated off-device so the probe never disturbs the
        # global RNG state. device="cpu" is explicit: this process may run with a
        # non-CPU default device, where randn would target that device and reject
        # the CPU generator. The pool holds 256 tokens; the qsa_allow_aiu layout
        # groups them into 16 pages of 16 tokens, the legacy layout into 256
        # single-token pages. Same storage, so the same absolute-index list and
        # reference cover both. 2 query heads over 1 KV head is inside the kernel's
        # GQA limit.
        gen = torch.Generator(device="cpu").manual_seed(0x5A1)
        q = torch.randn(1, 2, 256, generator=gen, device="cpu",
                        dtype=torch.float32).to(device=dev, dtype=torch.bfloat16)
        pool_k = torch.randn(256, 1, 1, 256, generator=gen, device="cpu",
                             dtype=torch.float32).to(device=dev, dtype=torch.bfloat16)
        pool_v = torch.randn(256, 1, 1, 256, generator=gen, device="cpu",
                             dtype=torch.float32).to(device=dev, dtype=torch.bfloat16)
        topk_table = torch.tensor(
            _QSA_PROBE_TOPK + _QSA_PROBE_PAD, dtype=torch.int32, device=dev).view(1, 1, -1)
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        seqused_k = torch.tensor([len(_QSA_PROBE_TOPK)], dtype=torch.int32, device=dev)
        if fa3_qsa_uses_allow_aiu_flag():
            k_cache, v_cache = pool_k.view(16, 16, 1, 256), pool_v.view(16, 16, 1, 256)
        else:
            k_cache, v_cache = pool_k, pool_v

        out, _lse = flash_attn_qsa_varlen_func(
            q, k_cache, v_cache, topk_table, cu_seqlens_q, 1,
            softmax_scale=softmax_scale, qsa_allow_aiu=False,
            seqused_k=seqused_k,
            return_softmax_lse=True,
        )
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - any failure means "no QSA in this build"
        _FA3_QSA_UNSUPPORTED_REASON = f"{type(e).__name__}: {e}"
        return False, _FA3_QSA_UNSUPPORTED_REASON

    # Reference: per-query-token gather attention in fp32 over the usable prefix (the
    # padded tail is masked through seqused_k), plus an all-bf16 PyTorch pass whose
    # rounding error sizes the tolerance (the criterion of test_qsa_cases.py, plus a
    # hard cap: a load path that fetches the wrong K/V misses by O(1), far past both
    # bounds).
    idx = topk_table.view(-1)[: len(_QSA_PROBE_TOPK)].long()
    k_sel = pool_k.reshape(-1, 256)[idx]
    v_sel = pool_v.reshape(-1, 256)[idx]
    scores = (q[0].float() @ k_sel.float().t()) * softmax_scale
    ref = torch.softmax(scores, dim=-1) @ v_sel.float()
    pt = torch.softmax((q[0] @ k_sel.t()) * softmax_scale, dim=-1) @ v_sel

    diff = (out[0].float() - ref).abs().max().item()
    pt_diff = (pt.float() - ref).abs().max().item()
    if diff > 2 * pt_diff + 1e-4 or diff > 0.1:
        convention = (
            "qsa_allow_aiu=False on the paged cache"
            if fa3_qsa_uses_allow_aiu_flag()
            else "a one-token-per-page view of the pool"
        )
        _FA3_QSA_UNSUPPORTED_REASON = (
            f"QSA self-test mismatch on a scattered top-k list (max diff {diff:.3g} "
            f"vs tolerance {2 * pt_diff + 1e-4:.3g}): the kernel returned wrong values "
            f"for {convention}, the calling convention the CUDA path uses"
        )
        return False, _FA3_QSA_UNSUPPORTED_REASON

    _FA3_QSA_SUPPORTED = True
    return True, None


def is_fa3_qsa_supported(device=None) -> bool:
    return _probe_fa3_qsa(device)[0]


def fa3_qsa_unsupported_reason(device=None) -> Optional[str]:
    return _probe_fa3_qsa(device)[1]


def flash_attn_qsa_varlen_func(
    q,
    k_cache,
    v_cache,
    topk_table,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale=None,
    out=None,
    num_splits=1,
    scheduler_metadata=None,
    return_softmax_lse=False,
    qsa_allow_aiu: bool = False,
    seqused_k=None,
):
    """Query-sparse attention (QSA) forward over a paged BF16/FP16 KV cache.

    This is FA3's QSA path, selected purely by the rank of `topk_table`. Unlike a normal page
    table, which maps (batch, page_rank) -> page id, a QSA table holds one row per *unpadded
    query token* and lists the absolute KV-pool token indices that token attends to:

        absolute_index = page_id * page_size + offset_in_page

    Because `seqlen_k` is then redefined to `topk` (the table width), the table is batch-wide
    and every row must already be causal, so `is_causal` degenerates inside the kernel.

    Arguments:
        q: (total_q, nheads, headdim). headdim must be 256.
        k_cache: (num_pages, page_size, nheads_k, headdim), stride(-1) == 1.
        v_cache: same shape/strides as k_cache.
        topk_table: (total_q, 1, topk), int32, stride(-1) == 1. Absolute token indices; a
            negative entry makes the kernel skip the load. Columns past `seqused_k` are never
            walked, so padded rows need their usable entries as a contiguous prefix.
        cu_seqlens_q: (batch_size + 1,), int32. Required by QSA.
        max_seqlen_q: int. Required together with cu_seqlens_q.
        softmax_scale: float. Defaults to 1 / sqrt(headdim).
        out: (total_q, nheads, headdim). Written in place when given.
        num_splits: 1 disables split-KV; 0 lets FA3 pick.
        scheduler_metadata: pass None so FA3 derives it with QSA semantics. The dense
            precomputation from `get_scheduler_metadata` assumes seqlen_k = pages * page_size.
        qsa_allow_aiu: bool. Certify every 16-column-aligned group of `topk_table` as
            16 pool-contiguous tokens, the layout the AIU paged-KV bulk load assumes
            (one entry read per group, 16 contiguous tokens fetched, the other 15
            ignored). False (the default) routes QSA through the per-column load that
            honors each entry individually. On wheels predating the argument the
            same routing is reached by handing FA3 a one-token-per-page view of the
            pool -- a view that exists only when the K/V split keeps a block's tokens
            stride-contiguous, i.e. one KV head per rank; callers with more must
            gate on :func:`fa3_qsa_uses_allow_aiu_flag` first.
        seqused_k: (batch_size,) int32, optional. Per-batch count of usable top-k columns;
            FA3 walks only that prefix of each row, keeping padded -1 slots out of the
            softmax. To express a per-row count, give every query row its own batch
            (cu_seqlens_q = arange(total_q + 1), max_seqlen_q = 1). Honored for QSA only
            under causal, which this wrapper always passes; builds whose QSA path still
            force-nulls it fail the `_probe_fa3_qsa` self-test, so callers gated on
            `is_fa3_qsa_supported` never reach this with an incompatible wheel.

    Return:
        out: (total_q, nheads, headdim).
        softmax_lse [if return_softmax_lse]: (nheads, total_q), float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if q.shape[-1] != 256:
        raise ValueError(f"QSA only supports head dim 256, got {q.shape[-1]}")
    if topk_table.dim() != 3 or topk_table.size(1) != 1:
        raise ValueError(
            "QSA topk_table must have shape (total_q, 1, topk); a 2-D table would be read as "
            "an ordinary page table and silently produce dense attention"
        )
    if topk_table.size(0) != q.size(0):
        raise ValueError("QSA topk_table must have exactly one row per query token")
    if topk_table.dtype != torch.int32:
        raise ValueError("QSA topk_table must be int32")
    if q.size(1) % k_cache.size(2):
        raise ValueError("QSA requires nheads to be a multiple of nheads_k")
    if q.size(1) // k_cache.size(2) > 32:
        raise ValueError("QSA supports at most 32 query heads per KV head")

    q = maybe_contiguous(q)
    topk_table = maybe_contiguous(topk_table)
    cu_seqlens_q = maybe_contiguous(cu_seqlens_q)
    seqused_k = maybe_contiguous(seqused_k)
    # k_cache/v_cache are transposed views of vLLM's packed cache: stride(-1) == 1 is required,
    # the leading strides may be non-canonical, which FA3 reads from the tensor itself.
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("QSA k_cache/v_cache must have a contiguous last dimension")

    if fa3_qsa_uses_allow_aiu_flag():
        # Rebuilt wheel: the caller's certification routes QSA off the AIU bulk load
        # inside the kernel; the cache goes through exactly as given.
        fwd_extra = {"qsa_allow_aiu": qsa_allow_aiu}
    else:
        if qsa_allow_aiu:
            raise ValueError(
                "qsa_allow_aiu=True requires an FA3 build with the qsa_allow_aiu "
                "argument; this wheel predates it"
            )
        # Legacy wheel: reach the same per-column routing through the cache shape.
        # The AIU bulk load is picked at runtime whenever page_size % 16 == 0, so hand
        # FA3 a one-token-per-page view of the same pool: page_size becomes 1, every
        # table entry is honored individually, and the absolute indices the table
        # already holds are page ids under the view. Zero-copy, but the (block,
        # token) merge needs one KV head per rank -- K and V interleave per token, so
        # with more the split view has no valid strides.
        try:
            k_cache = k_cache.view(-1, 1, *k_cache.shape[2:])
            v_cache = v_cache.view(-1, 1, *v_cache.shape[2:])
        except RuntimeError as e:
            raise RuntimeError(
                f"this FA3 build lacks qsa_allow_aiu, and the {k_cache.shape[2]} "
                "KV-head-per-rank K/V split cannot take the one-token-per-page view "
                "that steers QSA off the AIU bulk load"
            ) from e
        fwd_extra = {}

    if NVTX_PROFILE:
        th_nvtx_range_push(
            f"[FW_FMHA] --format=flash_attn_3_qsa,Forward,type:QSA,"
            f"seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},"
            f"num_heads:{q.shape[-2]},num_heads_k:{k_cache.shape[-2]},"
            f"batch_size:{len(cu_seqlens_q) - 1},topk:{topk_table.shape[-1]},"
            f"page_size:{k_cache.shape[1]},data_type:{q.dtype}"
        )

    out, softmax_lse, _, _ = torch.ops.flash_attn_3.fwd(
        q, k_cache, v_cache,
        None, None,           # k_new, v_new: QSA rejects both
        None,                 # q_v
        out,
        cu_seqlens_q,
        None,                 # cu_seqlens_k: passing it would make seqlen_k = max_seqlen_k
        None,                 # cu_seqlens_k_new
        None, seqused_k,      # seqused_q, seqused_k (per-row usable topk columns)
        max_seqlen_q,
        None,                 # max_seqlen_k: seqlen_k is already topk
        topk_table,           # 3-D -> selects the QSA path
        None,                 # kv_batch_idx: rejected
        None,                 # leftpad_k: rejected, indices are already absolute
        None, None, None,     # rotary_cos, rotary_sin, seqlens_rotary
        None, None, None,     # q_descale, k_descale, v_descale
        softmax_scale,
        True,                 # is_causal: degenerates for QSA, the topk list is pre-filtered
        -1, -1,               # window_size
        0,                    # attention_chunk
        0.0,                  # softcap
        True,                 # is_rotary_interleaved
        scheduler_metadata,
        num_splits,
        True,                 # pack_gqa: QSA carries qhead_per_khead rows in one kBlockM
        0,                    # sm_margin
        None,                 # s_aux
        **fwd_extra,          # qsa_allow_aiu, only on wheels that carry it
    )
    if NVTX_PROFILE:
        th_nvtx_range_pop()
    return (out, softmax_lse) if return_softmax_lse else out


def fa_version_unsupported_reason(fa_version: int, device = None) \
    -> Optional[str]:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[1]
    elif fa_version == 3:
        return _is_fa3_supported(device)[1]


#
#  For vLLM we only care about `flash_attn_varlen_func` and 
#   `flash_attn_with_kvcache` so we only maintain wrappers for these two.
#


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# NOTE only used in FA3
def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    has_softcap=False,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = torch.ops.flash_attn_3.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0], window_size[1],
        0, # attention_chunk
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )

    return scheduler_metadata


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: list[int] | None = None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # FA4 Only
    output_scale=None,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    # FA4 only
    mask_mod=None,
    aux_tensors=None,
    dynamic_causal: "torch.Tensor | None" = None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "seqused_k must be provided if block_table is provided"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if NVTX_PROFILE:
        if torch.cuda.is_current_stream_capturing():
            nvtx_message = (f"[FW_FMHA] --format=flash_attn_{fa_version},Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{real_window_size[0]},window_size_right:{real_window_size[1]},softcap:{softcap}")
        else:
            if NVTX_PROFILE_DUMP_SEQLEN:
                cu_seqlens_q_list = cu_seqlens_q.flatten().cpu().tolist() if cu_seqlens_q is not None else "[]"
                cu_seqlens_k_list = cu_seqlens_k.flatten().cpu().tolist() if cu_seqlens_k is not None else "[]"
                nvtx_message = (f"[FW_FMHA] --format=flash_attn_{fa_version},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{real_window_size[0]},window_size_right:{real_window_size[1]},softcap:{softcap},cu_seqlens_q:{cu_seqlens_q_list},cu_seqlens_k:{cu_seqlens_k_list}")
            else:
                nvtx_message = (f"[FW_FMHA] --format=flash_attn_{fa_version},Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_value:{v.shape[-1]},num_heads_k:{k.shape[-2]},num_heads:{q.shape[-2]},batch_size:{len(cu_seqlens_q) - 1},seqlen_k:{max_seqlen_k},data_type:{q.dtype},window_size_left:{real_window_size[0]},window_size_right:{real_window_size[1]},softcap:{softcap}")
        th_nvtx_range_push(nvtx_message)

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    
    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
                raise NotImplementedError(
                    "FA2 does not support scheduler_metadata, q_descale, "
                    "k_descale, v_descale"
                )
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        if s_aux is not None:
            raise NotImplementedError("FA2 does not support s_aux")
        out_fa2, softmax_lse, _, _ = torch.ops.flash_attn._flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp 
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=real_window_size[0],
            window_size_right=real_window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=(return_softmax_lse and dropout_p > 0),
            block_table=block_table,
            leftpad_k=None,
            seqused_k=seqused_k,
            zero_tensors=False,
        )
        # PPU NOTE: ppu fa2 api do not support inplace,
        # so it needs to be done in framework.
        if out is not None:
            out.copy_(out_fa2)
        else:
            out = out_fa2
    elif fa_version == 3:
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        # [Note]: workaround for a known FA3 issue
        # PPU FA3 use max_seqlen_k to choose tile
        # max_seqlen_k is needed when cu_seqlens_k on prefill
        if cu_seqlens_k is None:
            max_seqlen_k = 1

        out, softmax_lse, _, _ = torch.ops.flash_attn_3.fwd(
            q, k, v,
            None, None,       # k_new, v_new
            q_v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,     # cu_seqlens_k
            None,             # cu_seqlens_k_new
            None, seqused_k,  # seqused_q, seqused_k
            max_seqlen_q, max_seqlen_k,
            block_table,
            None,             # kv_batch_idx
            None,             # leftpad_k
            None, None, None, # rotary_cos, rotary_sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal,
            real_window_size[0], real_window_size[1],
            0,                # attention chunk
            softcap,
            True,             # rotary_interleaved
            scheduler_metadata,
            num_splits,
            None,             # pack_gqa
            0,                # sm_margin
            s_aux,            # s_aux
        )
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    if NVTX_PROFILE:
        th_nvtx_range_pop()
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    raise RuntimeError("PPU FA do not support sparse_attn now")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = torch.ops.flash_attn.fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out


def sparse_attn_varlen_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    *,
    return_softmax_lse=False,
    out=None,
):
    """Compute attention with vertical and slash sparsity patterns.
    Most Arguments are the same with the flash_attn_varlen_func interface, except for 4 extra args:
    block_count and block_offset for slash sparsity patterns, and
    column_count and column_index for vertical sparsity patterns.
    For more details please refer to Appendix C.4.2 of paper https://arxiv.org/abs/2407.02490.
    
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        block_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        block_offset: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_S)
        column_count: (batch_size, nheads, cdiv(seqlen, BLOCK_M))
        column_index: (batch_size, nheads, cdiv(seqlen, BLOCK_M), NNZ_V)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    raise RuntimeError("PPU FA do not support sparse_attn now")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
        
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse = torch.ops.flash_attn.varlen_fwd_sparse(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        False,
        causal,
        softcap,
        return_attn_probs and dropout_p > 0,
        None,
    )
    return (out, softmax_lse) if return_softmax_lse else out