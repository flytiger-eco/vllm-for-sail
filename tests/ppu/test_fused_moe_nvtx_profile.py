# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional tests for the NVTX profiling branch of ``fused_experts_op``.

When NVTX profiling is enabled (``VLLM_PPU_NVTX_PROFILE`` /
``VLLM_SAIL_NVTX_PROFILE`` / ``SAIL_NVTX_PROFILE``), ``fused_experts_op``
calls ``fused_experts_impl`` through its own positional-argument list.
A stale extra argument in that branch (leftover from the removed
``inplace`` parameter) previously crashed every MoE forward with:

    TypeError: fused_experts_impl() takes from 5 to 24 positional
    arguments but 25 were given

These tests run ``torch.ops.vllm.fused_experts`` with the NVTX branch
active and check that it executes and matches the default branch.

Run:
  pytest --import-mode=importlib --noconftest tests/ppu/test_fused_moe_nvtx_profile.py -v -s
"""

import os
import subprocess
import sys

import pytest
import torch

import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused_moe Triton kernels require CUDA",
)

NUM_EXPERTS = 8
HIDDEN = 64
INTERMEDIATE = 64
TOP_K = 2


def _make_inputs(M: int):
    """Build a small unquantized MoE problem on CUDA."""
    torch.manual_seed(42)
    hidden_states = torch.randn(
        M, HIDDEN, dtype=torch.bfloat16, device="cuda"
    )
    w1 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.1
    )
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (M, TOP_K), dtype=torch.int32, device="cuda"
    )
    topk_weights = torch.softmax(
        torch.randn(M, TOP_K, device="cuda"), dim=-1
    )
    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.mark.parametrize("M", [1, 16, 128])
def test_nvtx_branch_runs_and_matches_default(M: int, monkeypatch):
    """The NVTX branch must call fused_experts_impl with a valid
    argument list and produce the same output as the default branch."""
    inputs = _make_inputs(M)

    monkeypatch.setattr(fused_moe_module, "NVTX_PROFILE", False)
    ref_out = fused_experts(*inputs)

    nvtx_pushed = []
    monkeypatch.setattr(
        fused_moe_module,
        "th_nvtx_range_push",
        lambda label: nvtx_pushed.append(label),
    )
    monkeypatch.setattr(
        fused_moe_module, "th_nvtx_range_pop", lambda: None
    )
    monkeypatch.setattr(fused_moe_module, "NVTX_PROFILE", True)
    nvtx_out = fused_experts(*inputs)

    # The NVTX branch of fused_experts_op must actually have run.
    assert any(
        label.startswith(("P_MoE", "D_MoE")) for label in nvtx_pushed
    ), f"NVTX branch not executed, labels={nvtx_pushed}"

    torch.testing.assert_close(nvtx_out, ref_out)


_SUBPROC_SCRIPT = """
import sys

# `python -c` prepends the current working directory ('' entry) to
# sys.path; when pytest is invoked from the repo root the source tree
# would shadow the installed vllm (and lacks compiled vllm._C). Drop
# the cwd entries so the installed package is exercised.
sys.path[:] = [p for p in sys.path if p not in ("", ".")]

import torch

import vllm.model_executor.layers.fused_moe.fused_moe as fm

if not fm.NVTX_PROFILE:
    # torch.cuda.nvtx import failed; nothing to validate.
    print("NVTX_DISABLED")
    raise SystemExit(0)

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

torch.manual_seed(42)
hs = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
w1 = torch.randn(8, 128, 64, dtype=torch.bfloat16, device="cuda") * 0.1
w2 = torch.randn(8, 64, 64, dtype=torch.bfloat16, device="cuda") * 0.1
ids = torch.randint(0, 8, (16, 2), dtype=torch.int32, device="cuda")
w = torch.softmax(torch.randn(16, 2, device="cuda"), dim=-1)
out = fused_experts(hs, w1, w2, w, ids)
assert out.shape == (16, 64)
print("NVTX_OK")
"""


def test_nvtx_enabled_via_env_end_to_end():
    """Production enablement path: env var set at import time must not
    break torch.ops.vllm.fused_experts (regression: stale positional
    arg in the NVTX branch raised TypeError)."""
    env = dict(os.environ, VLLM_PPU_NVTX_PROFILE="1")
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROC_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "NVTX_DISABLED" in proc.stdout:
        pytest.skip("torch.cuda.nvtx not importable in this environment")
    assert "NVTX_OK" in proc.stdout, (
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert proc.returncode == 0, proc.stderr
