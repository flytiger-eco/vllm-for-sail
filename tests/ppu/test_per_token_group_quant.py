# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for per_token_group_quant_8bit_kernel.

Compares the original kernel vs PPU-optimized kernel:
  - per_token_group_quant_fp8: original vectorize_with_alignment
  - per_token_group_quant_fp8_ppu_opt: SM89+ bf16x2 + e4m3x2 PTX

Shapes match Qwen3-VL-2B-Instruct:
  hidden_size=2048, group_size=128, dtype=bf16

Run:
  pytest tests/kernels/core/test_per_token_group_quant_bench.py -v -s
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_ppu_opt,
)

# Qwen3-VL-2B-Instruct shapes
HIDDEN_SIZE = 2048
GROUP_SIZE = 128
EPS = 1e-10


def _bench(fn, warmup: int = 50, num_iters: int = 300) -> float:
    """Return average kernel time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / num_iters


@pytest.mark.parametrize(
    "M",
    [1, 4, 16, 64, 128, 256, 512],
)
def test_benchmark_original_vs_ppu_opt(M: int):
    """Benchmark original vs PPU-optimized kernel."""
    torch.manual_seed(42)
    x = torch.randn(
        M, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    ) * 0.01

    def run_orig():
        return per_token_group_quant_fp8(
            x, group_size=GROUP_SIZE, eps=EPS,
            column_major_scales=True,
        )

    def run_ppu():
        return per_token_group_quant_fp8_ppu_opt(
            x, group_size=GROUP_SIZE, eps=EPS,
            column_major_scales=True,
        )

    orig_us = _bench(run_orig)
    ppu_us = _bench(run_ppu)
    speedup = orig_us / ppu_us if ppu_us > 0 else float("inf")

    print(
        f"\n  M={M:4d}: "
        f"orig={orig_us:7.2f}us  "
        f"ppu_opt={ppu_us:7.2f}us  "
        f"speedup={speedup:.2f}x"
    )


@pytest.mark.parametrize("M", [64, 256])
def test_correctness_ppu_opt_vs_original(M: int):
    """Verify ppu_opt produces same results as original."""
    torch.manual_seed(42)
    x = torch.randn(
        M, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )

    q_orig, s_orig = per_token_group_quant_fp8(
        x, group_size=GROUP_SIZE, eps=EPS,
        column_major_scales=False,
    )
    q_ppu, s_ppu = per_token_group_quant_fp8_ppu_opt(
        x, group_size=GROUP_SIZE, eps=EPS,
        column_major_scales=False,
    )

    # Check scales match
    scale_diff = (
        s_orig.float() - s_ppu.float()
    ).abs().max().item()
    print(f"\n  M={M}: max scale diff = {scale_diff:.2e}")
    assert scale_diff < 1e-3, (
        f"Scale diff too large: {scale_diff}"
    )

    # Check quantized output matches
    q_orig_f32 = q_orig.to(torch.float32)
    q_ppu_f32 = q_ppu.to(torch.float32)
    q_diff = (q_orig_f32 - q_ppu_f32).abs().max().item()
    print(f"  M={M}: max quant diff = {q_diff:.2e}")
    assert q_diff <= 1.0, (
        f"Quant diff too large: {q_diff}"
    )
