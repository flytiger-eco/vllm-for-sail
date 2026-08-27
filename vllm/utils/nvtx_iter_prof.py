# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared iteration-level NVTX profiling helpers.

Gated by ``VLLM_PPU_NVTX_PROFILE``. When the flag is off or the required
packages (``model_prof``, ``nvtx``) are unavailable, every symbol degrades
to a no-op so call sites need no conditional logic.

Imported by the engine core, scheduler, v1 model runner, and v2 model
runner — keep it free of model-runner-specific state.
"""
import functools

import vllm.envs as envs

NVTX_PROFILE = envs.VLLM_PPU_NVTX_PROFILE
if NVTX_PROFILE:
    try:
        from model_prof import prof_iter
        from nvtx import annotate, mark
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push

        def sche_mark(sche_output):
            if len(sche_output.scheduled_new_reqs) > 0:
                reqs = [req.req_id for req in sche_output.scheduled_new_reqs]
                mark(f"new_reqs: {reqs}")
            mark(
                f"total_tokens={sche_output.total_num_scheduled_tokens},"
                f"req_id:num_tokens={sche_output.num_scheduled_tokens}"
            )
            if len(sche_output.finished_req_ids) > 0:
                mark(f"finish_req: {sche_output.finished_req_ids}")
    except ImportError:
        NVTX_PROFILE = False

if not NVTX_PROFILE:

    def th_nvtx_range_push(label):
        pass

    def th_nvtx_range_pop():
        pass

    def prof_iter(iteration):
        pass

    def sche_mark(sche_output):
        pass

    def mark(label):
        pass

    def annotate(name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator
