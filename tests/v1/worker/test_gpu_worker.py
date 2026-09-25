# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


@pytest.mark.parametrize(
    "scheduled,drafts,has_gdn,expected",
    [
        ({"a": 3, "b": 3}, {"b": [1, 2], "a": [3, 4]}, True, [2, 2]),
        ({"a": 3, "b": 3}, {"b": [1, 2]}, True, [-1, 2]),
        ({"a": 3, "b": 3}, {"a": [1], "b": [2, 3]}, True, [-1, 2]),
        ({"a": 1, "b": 1}, {"a": [], "b": []}, True, [0, 0]),
        ({"a": 3, "b": 3}, {}, True, None),
        ({"a": 3, "b": 3}, {"a": [1, 2], "b": [3, 4]}, False, None),
    ],
)
def test_pp_sp_prepass_uses_current_gdn_draft_markers(
    scheduled, drafts, has_gdn, expected
):
    """PP's padding prepass must use current markers before input preparation."""
    from vllm.v1.worker import gpu_worker

    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=sum(scheduled.values()),
        num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens=drafts,
    )
    determine = Mock(
        return_value=(
            None,
            SimpleNamespace(num_tokens=sum(scheduled.values())),
            None,
            None,
            None,
        )
    )
    runner = SimpleNamespace(
        _has_gdn_attention=has_gdn,
        num_spec_tokens=2,
        _determine_batch_execution_and_padding=determine,
        execute_model=Mock(return_value=None),
    )
    worker = SimpleNamespace(
        _pp_send_work=[],
        use_v2_model_runner=False,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=True)
            ),
            parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        ),
        model_runner=runner,
        annotate_profile=lambda _: nullcontext(),
    )
    with (
        patch.object(
            gpu_worker, "get_pp_group", return_value=SimpleNamespace(is_first_rank=True)
        ),
        patch.object(gpu_worker, "is_residual_scattered_for_sp", return_value=False),
    ):
        assert gpu_worker.Worker.execute_model(worker, scheduler_output) is None

    actual = determine.call_args.kwargs["num_decode_draft_tokens_cpu"]
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, np.array(expected, dtype=np.int32))
