# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random

import torch

import vllm.envs as envs

# from sglang.srt.layers.moe.ep_moe.kernels import _fwd_kernel_ep_scatter_1
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deep_gemm_block_shape,
    deepgemm_moe_permute,
)


def torch_validate_result(
    # validate object
    output_tensor,
    output_tensor_scale,
    output_index,
    num_recv_tokens_per_expert,
    expert_ids,
    # source
    recv_x,
    recv_x_scale,
    recv_topk,
    num_experts,
):
    # breakpoint()
    (
        output_tensor,
        output_index,
        num_recv_tokens_per_expert,
        expert_ids,
        recv_x,
        recv_topk,
    ) = map(
        lambda x: x.cpu(),
        [
            output_tensor,
            output_index,
            num_recv_tokens_per_expert,
            expert_ids,
            recv_x,
            recv_topk,
        ],
    )

    assert (recv_topk >= 0).all()
    expert_token_num = torch.bincount(recv_topk.view(-1), minlength=num_experts)
    assert (num_recv_tokens_per_expert == expert_token_num).all()

    cumsum = torch.cumsum(expert_token_num, dim=0)
    expert_start_loc = torch.cat([torch.tensor([0]), cumsum[:-1]])
    expert_end_loc = expert_start_loc + expert_token_num

    recv_topk_flatten = recv_topk.view(-1)
    output_index_flatten = output_index.view(-1)

    check_bins = torch.bincount(
        output_index_flatten, minlength=len(output_index_flatten)
    )
    assert len(check_bins) == len(output_index_flatten), (
        f"{len(check_bins)} not equal {len(output_index_flatten)}"
    )
    torch.set_printoptions(threshold=float("inf"))
    print(check_bins)
    assert (check_bins == 1).all()

    check_status = (expert_start_loc[recv_topk_flatten] <= output_index_flatten) & (
        output_index_flatten < expert_end_loc[recv_topk_flatten]
    )
    if check_status.all():
        print("output_index check pass")
    else:
        print("output_index check fail")
        return

    num_tokens = recv_x.shape[0]
    for i in range(num_tokens):
        check_status = output_tensor[output_index[i]] == recv_x[i].unsqueeze(0)
        if not check_status.all():
            print(f"output_tensor[{i}] check fail!")
            break
    else:
        print("output_tensor check pass")

    expert_ids_flatten = expert_ids.view(-1)
    assert (expert_ids_flatten[output_index_flatten] == recv_topk_flatten).all()
    print("expert_ids check pass")

    if output_tensor_scale is not None:
        # breakpoint()
        output_tensor_scale, recv_x_scale = map(
            lambda x: x.cpu(), [output_tensor_scale, recv_x_scale]
        )
        for i in range(num_tokens):
            check_status = output_tensor_scale[output_index[i]] == recv_x_scale[
                i
            ].unsqueeze(0)
            if not check_status.all():
                print(f"output_tensor_scale[{i}] check fail!")
                break
        else:
            print("output_tensor_scale check pass")


def recover_topk_ids(
    expert_num_tokens: list[int],
    num_tokens,
    num_experts,
    num_experts_per_tok,
) -> torch.Tensor:
    assert len(expert_num_tokens) == num_experts
    assert max(expert_num_tokens) <= num_tokens
    assert sum(expert_num_tokens) == (num_tokens * num_experts_per_tok)

    indices = list(range(len(expert_num_tokens)))
    random.shuffle(indices)
    tensor_slices = [
        torch.ones(size=(expert_num_tokens[i],), dtype=torch.int32) * i for i in indices
    ]
    topk_ids = torch.cat(tensor_slices)
    topk_ids = topk_ids.view(num_experts_per_tok, num_tokens).t().contiguous()

    rand_idx = torch.argsort(torch.rand(num_tokens, num_experts_per_tok), dim=1)
    topk_ids = torch.gather(topk_ids, 1, rand_idx)

    indices = torch.randperm(topk_ids.size(0))
    topk_ids = topk_ids[indices]
    return topk_ids.contiguous()


def base_test(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    expert_num_tokens: list,
    use_fp8: bool = True,
    debug: bool = False,
):
    block_k = deep_gemm_block_shape()[1]
    if use_fp8:
        assert block_k == 128, "Please check....."

    hidden_size_scale = hidden_size // block_k
    hidden_states = torch.randn(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    hidden_states_scale = torch.randn(
        (num_tokens, hidden_size_scale), dtype=torch.float32, device="cuda"
    )
    topk_ids = recover_topk_ids(expert_num_tokens, num_tokens, num_experts, topk)

    # expert_map = torch.randperm(num_experts, dtype=torch.int32, device='cuda')
    expert_map = None
    topk_ids = topk_ids.cuda()

    with torch.cuda.nvtx.range("main-test"):
        a, a_scale, expert_ids, inv_perm, num_recv_tokens_per_expert = (
            deepgemm_moe_permute(
                aq=hidden_states,
                aq_scale=hidden_states_scale,
                topk_ids=topk_ids,
                local_num_experts=num_experts,
                expert_map=expert_map,
                expert_tokens_meta=None,
                use_fp8=True,
                use_int8=False,
            )
        )

        a, a_scale, expert_ids, inv_perm, num_recv_tokens_per_expert = (
            deepgemm_moe_permute(
                aq=hidden_states,
                aq_scale=hidden_states_scale,
                topk_ids=topk_ids,
                local_num_experts=num_experts,
                expert_map=expert_map,
                expert_tokens_meta=None,
                use_fp8=True,
                use_int8=False,
            )
        )

        torch.cuda.synchronize()
    # breakpoint()
    torch_validate_result(
        a,
        a_scale,
        inv_perm,
        num_recv_tokens_per_expert,
        expert_ids,
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_experts,
    )

    print("OK")


def set_deterministic_seeds(seed: int = 2025):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    # boost CE freq
    BBA = torch.randn(3 * 1000 * 1000 * 1000, dtype=torch.float32).to(
        device=torch.device("cuda")
    )
    set_deterministic_seeds()
    num_experts = 512
    num_tokens = 884
    hidden_size = 2048
    topk = 10
    expert_num_tokens = [
        0,
        106,
        18,
        0,
        19,
        0,
        0,
        0,
        33,
        36,
        1,
        0,
        0,
        0,
        12,
        0,
        5,
        1,
        7,
        8,
        0,
        0,
        0,
        1,
        0,
        0,
        28,
        1,
        0,
        2,
        0,
        15,
        457,
        22,
        0,
        0,
        6,
        5,
        24,
        9,
        4,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        211,
        0,
        4,
        0,
        0,
        0,
        0,
        8,
        0,
        0,
        0,
        4,
        3,
        0,
        6,
        0,
        23,
        59,
        3,
        36,
        0,
        1,
        0,
        1,
        0,
        3,
        0,
        0,
        0,
        0,
        65,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        8,
        0,
        0,
        0,
        47,
        8,
        1,
        1,
        0,
        0,
        43,
        2,
        0,
        1,
        50,
        4,
        0,
        3,
        0,
        0,
        249,
        0,
        30,
        0,
        1,
        89,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        29,
        0,
        1,
        35,
        0,
        3,
        3,
        0,
        1,
        155,
        0,
        4,
        16,
        9,
        0,
        0,
        0,
        35,
        0,
        12,
        34,
        17,
        2,
        37,
        0,
        77,
        11,
        7,
        4,
        0,
        0,
        0,
        0,
        0,
        85,
        0,
        0,
        0,
        14,
        34,
        0,
        0,
        137,
        118,
        23,
        0,
        0,
        17,
        0,
        4,
        58,
        0,
        75,
        0,
        56,
        17,
        0,
        0,
        0,
        3,
        6,
        0,
        0,
        0,
        0,
        4,
        2,
        0,
        1,
        3,
        29,
        0,
        2,
        0,
        5,
        9,
        0,
        0,
        18,
        0,
        6,
        173,
        13,
        0,
        190,
        0,
        2,
        56,
        0,
        12,
        0,
        0,
        67,
        4,
        9,
        175,
        0,
        14,
        0,
        1,
        0,
        14,
        3,
        0,
        0,
        0,
        2,
        3,
        0,
        0,
        121,
        0,
        4,
        0,
        16,
        72,
        0,
        6,
        0,
        0,
        32,
        2,
        56,
        0,
        1,
        0,
        1,
        0,
        0,
        15,
        0,
        29,
        0,
        0,
        31,
        0,
        0,
        2,
        3,
        0,
        3,
        4,
        0,
        3,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        3,
        0,
        9,
        26,
        0,
        0,
        0,
        0,
        2,
        0,
        0,
        35,
        0,
        0,
        0,
        5,
        20,
        0,
        0,
        0,
        4,
        0,
        0,
        0,
        10,
        7,
        0,
        0,
        1,
        16,
        0,
        0,
        0,
        7,
        0,
        0,
        1,
        17,
        21,
        0,
        0,
        3,
        0,
        5,
        12,
        17,
        0,
        0,
        0,
        0,
        14,
        0,
        41,
        0,
        1,
        0,
        0,
        3,
        0,
        158,
        1,
        28,
        5,
        5,
        0,
        0,
        0,
        0,
        8,
        2,
        0,
        0,
        0,
        2,
        0,
        11,
        0,
        66,
        8,
        0,
        0,
        4,
        4,
        1,
        122,
        5,
        450,
        2,
        10,
        109,
        0,
        10,
        202,
        302,
        48,
        212,
        0,
        0,
        10,
        3,
        6,
        1,
        17,
        0,
        28,
        0,
        1,
        0,
        260,
        9,
        142,
        0,
        1,
        0,
        1,
        0,
        4,
        0,
        1,
        23,
        20,
        44,
        3,
        0,
        368,
        0,
        0,
        17,
        60,
        0,
        8,
        130,
        0,
        0,
        0,
        70,
        14,
        0,
        0,
        16,
        242,
        0,
        8,
        27,
        0,
        2,
        0,
        22,
        65,
        1,
        0,
        35,
        0,
        0,
        289,
        0,
        0,
        0,
        8,
        34,
        11,
        0,
        0,
        0,
        33,
        114,
        0,
        0,
        7,
        0,
        0,
        4,
        1,
        0,
        0,
        0,
        0,
        22,
        0,
        0,
        0,
        0,
        39,
        0,
        3,
        0,
        128,
        0,
        23,
        81,
        5,
        0,
        6,
        0,
        0,
        50,
        10,
        0,
        0,
        1,
        0,
        0,
        4,
        8,
        0,
        1,
        4,
        22,
        0,
        1,
        3,
        0,
        1,
        0,
        0,
        0,
        0,
        6,
        1,
        0,
        0,
        10,
        0,
        0,
        1,
        0,
        38,
        0,
        0,
        2,
        0,
        5,
    ]
    # expert_num_tokens = [1132,857,657,570,506,475,408,367,344,332,289,278,260,267,250,256,234,234,238,216,210,198,201,195,203,197,206,185,177,177,161,171,171,173,164,156,155,139,141,152,162,143,154,122,139,152,132,145,123,140,131,116,116,116,110,114,126,134,108,125,106,120,104,122,120,112,130,101,117,101,128,107,102,107,89,110,98,105,69,90,96,90,98,103,87,95,85,82,77,100,110,95,90,98,77,81,81,80,87,107,100,91,77,78,97,79,71,76,72,77,80,93,89,78,74,83,73,75,77,84,64,70,80,78,58,83,71,76,72,66,70,76,91,84,79,60,79,89,83,80,71,76,73,67,86,65,73,71,74,66,74,58,58,76,53,71,65,70,65,70,57,58,62,48,58,54,55,54,68,75,61,61,62,63,56,65,66,51,66,72,59,69,54,58,58,60,63,70,52,63,51,56,53,66,59,74,62,59,63,57,42,47,50,55,59,60,58,38,42,48,53,48,53,64,64,57,42,55,51,59,61,45,52,54,52,54,47,64,65,57,43,68,39,47,43,59,58,60,52,55,48,57,50,64,62,47,65,45,58,57,52,54,51,49,51,43,45,46,51,47,44,39,55,50,51,57,55,42,46,34,43,46,37,54,42,50,44,44,54,42,40,49,48,42,51,41,49,28,44,47,43,51,37,65,46,41,46,52,48,38,45,50,46,37,46,35,41,42,41,43,41,52,38,44,38,35,45,46,49,47,44,46,40,37,42,48,40,45,43,46,36,44,38,46,47,44,39,43,36,42,43,43,49,49,35,42,44,46,35,58,39,53,48,44,53,32,38,42,31,43,31,44,47,40,41,40,41,36,43,43,43,38,35,34,42,44,40,33,43,34,45,42,50,46,32,36,40,45,40,34,36,39,39,28,34,35,28,35,42,40,39,35,36,48,38,35,41,34,36,36,44,37,35,44,39,34,45,33,32,47,36,55,32,43,39,34,36,38,36,30,24,37,25,31,25,38,36,48,40,35,32,38,33,36,34,42,41,51,37,35,45,44,36,34,36,34,33,29,28,28,29,39,23,31,53,37,34,41,24,36,27,28,34,37,46,42,36,30,28,45,38,33,34,31,36,38,27,37,39,44,25,35,29,34,33,32,32,39,29,32,35,31,40,40,38,28,34,29,27,43,40,37]

    num_tokens = sum(expert_num_tokens) // topk
    print(f"num_tokens={num_tokens}")

    base_test(num_tokens, hidden_size, num_experts, topk, expert_num_tokens)