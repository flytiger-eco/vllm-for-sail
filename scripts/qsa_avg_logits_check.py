# SPDX-License-Identifier: Apache-2.0
"""Offline numerics check: DeepGEMM fp8_*_avg_logits vs BF16 references.

Compares, for QSA indexer scoring semantics
``logits = sum_h(relu(q @ k)) * k_scale / sqrt(head_dim)``:

* Part 1 (non-paged): ``fp8_mqa_avg_logits`` vs an exact BF16 einsum reference.
* Part 2 (paged): ``fp8_paged_mqa_avg_logits`` vs the same BF16 reference and
  vs the production Triton kernel ``qsa_mqa_paged`` (BF16 cache).

Metrics per query row: top-k recall against the BF16 reference, plus a
cosine-style diff on the masked (valid) region.

Run inside the build container, e.g.:
    python3 scripts/qsa_avg_logits_check.py
"""

from __future__ import annotations

import argparse
import math

import torch

import vllm.utils.ppu_deep_gemm as pdg
from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_mqa_paged

E4M3_MAX = 448.0


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double().flatten()
    y = y.double().flatten()
    denom = (x * x).sum() + (y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def bf16_ref_logits(
    q: torch.Tensor,  # [rows, H, D] bf16
    k: torch.Tensor,  # [N, D] bf16
    ks: torch.Tensor,  # [rows] int64 start (inclusive)
    ke: torch.Tensor,  # [rows] int64 end (exclusive)
) -> torch.Tensor:
    score = torch.einsum("mhd,nd->hmn", q.float(), k.float())
    logits = score.relu().sum(dim=0) * (q.shape[-1] ** -0.5)
    col = torch.arange(k.shape[0], device=q.device)
    mask = (col[None, :] >= ks[:, None]) & (col[None, :] < ke[:, None])
    return logits.masked_fill(~mask, float("-inf"))


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (dim=-1) symmetric fp8 quant. Returns (fp8, dequant_scale)."""
    amax = x.abs().float().amax(dim=-1).clamp(min=1e-4)
    scale = amax / E4M3_MAX
    return (x.float() / scale[:, None]).to(torch.float8_e4m3fn), scale


def topk_recall(
    logits: torch.Tensor,  # [rows, N] candidate (already -inf masked)
    ref: torch.Tensor,  # [rows, N] reference (already -inf masked)
    k: int,
) -> torch.Tensor:
    rows = logits.shape[0]
    recall = torch.zeros(rows, dtype=torch.float64)
    for r in range(rows):
        valid = int(torch.isfinite(ref[r]).sum().item())
        kr = min(k, valid)
        if kr == 0:
            continue
        top_ref = set(ref[r].topk(kr).indices.tolist())
        top_x = set(logits[r].topk(kr).indices.tolist())
        recall[r] = len(top_ref & top_x) / kr
    return recall


def report(name: str, recall: torch.Tensor, diff: float, fail_recall: float) -> bool:
    ok = recall.min().item() >= fail_recall and diff < 2e-3
    print(
        f"  {name}: diff={diff:.2e}  recall mean={recall.mean():.4f} "
        f"min={recall.min():.4f}  -> {'OK' if ok else 'SUSPECT'}"
    )
    return ok


def part_nonpaged(args) -> bool:
    print(f"[Part 1] fp8_mqa_avg_logits  M={args.rows} N={args.keys} "
          f"H={args.heads} D={args.head_dim}")
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    M, N, H, D = args.rows, args.keys, args.heads, args.head_dim
    q = torch.randn((M, H, D), device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn((N, D), device="cuda", dtype=torch.bfloat16, generator=g)
    ke = torch.randint(1, N + 1, (M,), device="cuda", generator=g).to(torch.int32)
    ks = torch.zeros(M, device="cuda", dtype=torch.int32)

    ref = bf16_ref_logits(q, k, ks.long(), ke.long())

    q_fp8, q_scale = quant_rows(q.reshape(M, H * D))
    q_fp8 = q_fp8.reshape(M, H, D).contiguous()
    k_fp8, k_scale = quant_rows(k)

    logits = pdg.fp8_mqa_avg_logits(
        q_fp8,
        (k_fp8, k_scale),
        ks,
        ke,
        clean_logits=True,
        q_scale=q_scale,
    )
    assert logits.shape == (M, N) and logits.dtype == torch.float32

    masked = torch.isfinite(ref)
    diff = calc_diff(logits.masked_fill(~masked, 0), ref.masked_fill(~masked, 0))
    recall = topk_recall(logits, ref, args.topk)
    return report("deepgemm-vs-bf16ref", recall, diff, args.recall_floor)


def paged_ref(
    q: torch.Tensor,  # [B, next_n, H, D] bf16
    cache: torch.Tensor,  # [nb, page, 1, D] bf16
    block_tables: torch.Tensor,  # [B, max_blocks]
    visible: torch.Tensor,  # [B, next_n] compressed rows visible per token
    max_cols: int,
) -> torch.Tensor:
    B, next_n, H, D = q.shape
    page = cache.shape[1]
    out = torch.full((B * next_n, max_cols), float("-inf"), device=q.device)
    for i in range(B):
        nblk = int((int(visible[i].max()) + page - 1) // page)
        blocks = block_tables[i, :nblk].long()
        k = cache[blocks].reshape(-1, D).float()  # [nblk*page, D]
        s = torch.einsum("nhd,cd->hnc", q[i].float(), k)
        s = s.relu().sum(dim=0) * (D**-0.5)
        cols = torch.arange(k.shape[0], device=q.device)
        for j in range(next_n):
            m = cols < int(visible[i, j])
            out[i * next_n + j, : k.shape[0]] = s[j].masked_fill(~m, float("-inf"))
    return out


def part_paged(args) -> bool:
    R = args.compress_ratio
    B, nn, H, D, page = args.batch, args.next_n, args.heads, args.head_dim, 64
    print(f"[Part 2] fp8_paged_mqa_avg_logits  B={B} next_n={nn} H={H} "
          f"D={D} page={page} R={R}")
    g = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    # Per-request compressed-row counts (sequence length in compressed units).
    comp_lens = torch.randint(20, args.max_comp_rows, (B,), device="cuda", generator=g)
    comp_lens = comp_lens.clamp(min=args.topk + 8) if args.long else comp_lens
    seq_lens = comp_lens * R  # token-level sequence lengths
    max_blocks = int((comp_lens.max().item() + page - 1) // page)
    num_blocks = int(((comp_lens + page - 1) // page).sum().item() * 1.3) + 1
    cache = torch.randn((num_blocks, page, 1, D), device="cuda",
                        dtype=torch.bfloat16, generator=g)
    block_tables = torch.zeros((B, max_blocks), device="cuda", dtype=torch.int32)
    pool = torch.randperm(num_blocks, device="cuda", generator=g).to(torch.int32)
    cnt = 0
    for i in range(B):
        nblk = int((int(comp_lens[i]) + page - 1) // page)
        block_tables[i, :nblk] = pool[cnt:cnt + nblk]
        cnt += nblk

    q = torch.randn((B, nn, H, D), device="cuda", dtype=torch.bfloat16, generator=g)
    # Token positions: token j of request i sits at seq_len-nn+j (last is the
    # current token). Visible compressed rows follow the Triton kernel's rule
    # visible = (pos+1)//R; DeepGEMM's 2D context_lens must match it exactly.
    offs = torch.arange(nn, device="cuda")
    positions = (seq_lens[:, None] - nn + offs).reshape(-1).long()
    visible = ((positions + 1) // R).clamp(min=1).reshape(B, nn).to(torch.int32)
    max_cols = max_blocks * page

    ref = paged_ref(q, cache, block_tables, visible, max_cols)

    # --- DeepGEMM fp8 path ---
    q_scale = q.abs().float().amax(dim=(2, 3)).clamp(min=1e-4) / E4M3_MAX  # [B, nn]
    q_fp8 = (q.float() / q_scale[:, :, None, None]).to(torch.float8_e4m3fn)
    cache_fp8 = torch.empty((num_blocks, page * (D + 4)), device="cuda",
                            dtype=torch.uint8)
    flat = cache.reshape(num_blocks * page, D)
    k_fp8, k_scale = quant_rows(flat)
    cache_fp8[:, : page * D] = k_fp8.view(torch.uint8).view(num_blocks, page * D)
    cache_fp8[:, page * D:] = k_scale.view(torch.uint8).view(num_blocks, page * 4)
    cache_fp8 = cache_fp8.view(num_blocks, page, 1, D + 4)

    schedule = pdg.get_paged_mqa_logits_metadata(
        visible, page, pdg.get_num_sms(), metadata_extra=(nn, H, D, 1)
    )
    dg_logits = pdg.fp8_paged_mqa_avg_logits(
        q_fp8, cache_fp8, visible, block_tables, schedule, max_cols
    )
    assert dg_logits.shape == (B * nn, max_cols)
    dg_logits = dg_logits * q_scale.reshape(-1)[:, None]  # descale
    col = torch.arange(max_cols, device="cuda")
    dg_mask = col[None, :] < visible.reshape(-1, 1)
    dg_logits = dg_logits.masked_fill(~dg_mask, float("-inf"))

    masked = torch.isfinite(ref)
    diff_dg = calc_diff(dg_logits.masked_fill(~masked, 0),
                        ref.masked_fill(~masked, 0))
    recall_dg = topk_recall(dg_logits, ref, args.topk)
    ok = report("deepgemm-vs-bf16ref", recall_dg, diff_dg, args.recall_floor)

    # --- Triton BF16 production path ---
    q_flat = q.reshape(B * nn, H, D)
    token_to_req = torch.arange(B, device="cuda",
                                dtype=torch.int32).repeat_interleave(nn)
    tri_logits, tri_visible = qsa_mqa_paged(
        q_flat, cache, block_tables, token_to_req, positions,
        seq_lens.long(), R, num_columns=max_cols,
    )
    assert torch.equal(tri_visible, visible.reshape(-1).to(tri_visible.dtype)), (
        f"visible mismatch: triton {tri_visible[:8]} vs harness "
        f"{visible.reshape(-1)[:8]}")
    tri_mask = col[None, :] < tri_visible[:, None]
    tri_logits = tri_logits.masked_fill(~tri_mask, float("-inf"))
    diff_tri = calc_diff(tri_logits.masked_fill(~masked, 0),
                         ref.masked_fill(~masked, 0))
    recall_tri = topk_recall(tri_logits, ref, args.topk)
    report("triton-vs-bf16ref", recall_tri, diff_tri, args.recall_floor)

    both = torch.isfinite(dg_logits) & torch.isfinite(tri_logits)
    diff_x = calc_diff(dg_logits.masked_fill(~both, 0),
                       tri_logits.masked_fill(~both, 0))
    recall_x = topk_recall(dg_logits, tri_logits, args.topk)
    ok &= report("deepgemm-vs-triton", recall_x, diff_x, args.recall_floor)
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--compress-ratio", type=int, default=4)
    p.add_argument("--topk", type=int, default=512)
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--keys", type=int, default=8192)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--next-n", type=int, default=1)
    p.add_argument("--max-comp-rows", type=int, default=600)
    p.add_argument("--long", action="store_true",
                   help="force every row to have > topk valid columns")
    p.add_argument("--recall-floor", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-nonpaged", action="store_true")
    p.add_argument("--skip-paged", action="store_true")
    args = p.parse_args()

    torch.cuda.init()
    assert pdg.has_fp8_mqa_avg_logits(), (
        "installed deep_gemm lacks fp8_*_avg_logits; rebuild it from main")
    print(f"deep_gemm avg_logits available; num_sms={pdg.get_num_sms()}")

    ok = True
    if not args.skip_nonpaged:
        ok &= part_nonpaged(args)
    if not args.skip_paged:
        ok &= part_paged(args)
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
