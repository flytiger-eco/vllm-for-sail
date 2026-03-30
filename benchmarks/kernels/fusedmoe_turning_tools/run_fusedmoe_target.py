import torch
import triton
import triton.language as tl
import multiprocessing as mp
import numpy as np
import os, json, math, random, inspect, functools, argparse, itertools
from tqdm import tqdm
from triton.runtime import driver
from typing import Any, Set, List, Dict, Tuple, Callable
from triton.runtime.errors import OutOfResources
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size


@functools.lru_cache
def get_search_space(max_block_m: int=64, max_block_n: int=256, max_block_k: int=256) -> List[Dict]:
    '''
        tuning space
    '''
    arr = [16, 32, 64, 128, 256, 512]
    block_m_range = filter(lambda x: x <= max_block_m, arr)
    block_n_range = filter(lambda x: x <= max_block_n, arr)
    block_k_range = filter(lambda x: x <= max_block_k and x > 16, arr)
    stages_range = [1, 2, 3, 4]
    warps_range = [2, 4, 8, 16]
    group_m_range = [1] # [1, 16, 32, 64]

    configs = list()
    for block_m, block_n, block_k, group_m, num_stages, num_warps in itertools.product(block_m_range, block_n_range, block_k_range, group_m_range, stages_range, warps_range):
        config = {
            "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k, "GROUP_SIZE_M": group_m, "SPLIT_K": 1,
            "num_warps": num_warps, "num_stages": num_stages
        }
        configs.append(config)

    return configs


def get_candidate_configs(load_file: str='./candidate_configs.json') -> Dict:
    with open(load_file, 'r') as f:
        return json.load(f)


def package_args(func: Callable) -> Callable:
    '''
    pack the All arguments into a Python dictionary.
    '''
    def wrapper(*args, **kwargs) -> Dict:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)
    return wrapper


def moe_align_block_size_dummy(
    topk_ids: torch.Tensor, align_size: int, num_experts: int
) -> Tuple:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (align_size - 1)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(topk_ids.numel() * align_size, max_num_tokens_padded)
    sorted_token_ids = torch.zeros((max_num_tokens_padded, ), dtype=torch.int32)
    expert_ids = torch.zeros((num_experts, topk_ids.numel() // align_size, ), dtype=torch.int32)
    num_tokens_post_padded = torch.tensor([100], dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


result_cache = dict()
def moe_align_block_size_for_cache(
    topk_ids: torch.Tensor, align_size: int, num_experts: int
) -> Tuple:
    key = (topk_ids.numel(), num_experts, align_size)
    global result_cache
    if key in result_cache.keys():
        return result_cache[key]
    else:
        if topk_ids.device.type == "cpu":
            result = moe_align_block_size_dummy(topk_ids, align_size, num_experts)
        else:
            result = moe_align_block_size(topk_ids, align_size, num_experts)
        result_cache[key] = result
        return result


def get_input_metadata(input_metadata_path: str, model_name: str) -> Dict:
    with open(input_metadata_path, "r") as f:
        content = json.load(f)
    return content[model_name]


def recover_topk_ids(
    expert_num_tokens: List[int], num_tokens: int, model_config: Dict
) -> torch.Tensor:
    num_experts = model_config["num_experts"]
    num_experts_per_tok = model_config["num_experts_per_tok"]
    assert len(expert_num_tokens) == num_experts
    assert max(expert_num_tokens) <= num_tokens
    assert sum(expert_num_tokens) == (num_tokens * num_experts_per_tok)

    indices = list(range(len(expert_num_tokens)))
    random.shuffle(indices)
    tensor_slices = [torch.ones(size=(expert_num_tokens[i], ), dtype=torch.int32) * i for i in indices]
    topk_ids = torch.cat(tensor_slices)
    topk_ids = topk_ids.view(num_experts_per_tok, num_tokens).t().contiguous()

    rand_idx = torch.argsort(torch.rand(num_tokens, num_experts_per_tok), dim=1)
    topk_ids = torch.gather(topk_ids, 1, rand_idx)

    indices = torch.randperm(topk_ids.size(0))
    topk_ids = topk_ids[indices]
    return topk_ids.contiguous()


def prepare_fake_inputs(batch_size: int,
                model_config: Dict,
                tp_size: int,
                data_dtype: torch.dtype,
                down_gemm: bool=False,
                use_int8_w8a8: bool=False,
                use_fp8_w8a8: bool=False,
                quant_group_shape: List[int]=None) -> Tuple[torch.Tensor]:

    topk_num = model_config["num_experts_per_tok"]
    M = batch_size * topk_num if down_gemm else batch_size
    N = model_config["hidden_size"] if down_gemm else 2 * model_config["moe_intermediate_size"]
    K = model_config["moe_intermediate_size"] if down_gemm else model_config["hidden_size"]
    E = model_config["num_experts"]

    if down_gemm:
        assert K % tp_size == 0
        K = K // tp_size
    else:
        assert N % tp_size == 0
        N = N // tp_size

    assert not (use_int8_w8a8 and use_fp8_w8a8)
    if use_int8_w8a8:
        a = torch.empty((M, K), dtype=torch.float32, device="cpu").to(torch.int8)
        b = torch.empty((E, N, K), dtype=torch.float32, device="cpu").to(torch.int8)
        assert quant_group_shape[0] == 0 or quant_group_shape[1] == 0
        Ns, Ks = N, 1 # per-channel
        a_scale = torch.empty((M, Ks), dtype=torch.float32, device="cpu")
        b_scale = torch.empty((E, Ns, Ks), dtype=torch.float32, device="cpu")
    elif use_fp8_w8a8:
        a = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="cpu")
        b = torch.empty((E, N, K), dtype=torch.float8_e4m3fn, device="cpu")
        assert quant_group_shape[0] > 0 and quant_group_shape[1] > 0
        assert N % quant_group_shape[0] == 0
        assert K % quant_group_shape[1] == 0
        Ns = N // quant_group_shape[0]
        Ks = K // quant_group_shape[1]
        a_scale = torch.empty((M, Ks), dtype=torch.float32, device="cpu")
        b_scale = torch.empty((E, Ns, Ks), dtype=torch.float32, device="cpu")
    else:
        a = torch.empty((M, K), dtype=data_dtype, device="cpu")
        b = torch.empty((E, N, K), dtype=data_dtype, device="cpu")
        a_scale = None
        b_scale = None

    if down_gemm:
        topk_weight = torch.empty((M, ), dtype=data_dtype, device="cpu")
    else:
        topk_weight = None

    c = torch.empty((M, topk_num, N), dtype=data_dtype, device="cpu")
    return a, a_scale, b, b_scale, c, topk_weight


def prepare_inputs(batch_size: int,
                model_config: Dict,
                tp_size: int,
                data_dtype: torch.dtype,
                down_gemm: bool=False,
                use_int8_w8a8: bool=False,
                use_fp8_w8a8: bool=False,
                quant_group_shape: List[int]=None) -> Tuple[torch.Tensor]:

    topk_num = model_config["num_experts_per_tok"]
    M = batch_size * topk_num if down_gemm else batch_size
    N = model_config["hidden_size"] if down_gemm else 2 * model_config["moe_intermediate_size"]
    K = model_config["moe_intermediate_size"] if down_gemm else model_config["hidden_size"]
    E = model_config["num_experts"]

    if down_gemm:
        assert K % tp_size == 0
        K = K // tp_size
    else:
        assert N % tp_size == 0
        N = N // tp_size

    assert not (use_int8_w8a8 and use_fp8_w8a8)
    if use_int8_w8a8:
        a = (torch.randn((M, K), dtype=torch.float32, device="cuda") * 1000.0).to(torch.int8)
        b = (torch.randn((E, N, K), dtype=torch.float32, device="cuda") * 1000.0).to(torch.int8)
        assert quant_group_shape[0] == 0 or quant_group_shape[1] == 0
        Ns, Ks = N, 1 # per-channel
        a_scale = torch.randn((M, Ks), dtype=torch.float32, device="cuda")
        b_scale = torch.randn((E, Ns, Ks), dtype=torch.float32, device="cuda")
    elif use_fp8_w8a8:
        a = torch.randn((M, K), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn((E, N, K), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        assert quant_group_shape[0] > 0 and quant_group_shape[1] > 0
        assert N % quant_group_shape[0] == 0
        assert K % quant_group_shape[1] == 0
        Ns = N // quant_group_shape[0]
        Ks = K // quant_group_shape[1]
        a_scale = torch.randn((M, Ks), dtype=torch.float32, device="cuda")
        b_scale = torch.randn((E, Ns, Ks), dtype=torch.float32, device="cuda")
    else:
        a = torch.randn((M, K), dtype=data_dtype, device="cuda")
        b = torch.randn((E, N, K), dtype=data_dtype, device="cuda")
        a_scale = None
        b_scale = None

    if down_gemm:
        topk_weight = torch.randn((M, ), dtype=data_dtype, device="cuda")
    else:
        topk_weight = None

    c = torch.empty((M, topk_num, N), dtype=data_dtype, device="cuda")
    return a, a_scale, b, b_scale, c, topk_weight


def main_tune_loop(
    a, a_scale, b, b_scale, c, quant_group_shape, topk_weight, topk_ids, search_configs
) -> None:
    _, K = a.shape
    E, N, _ = b.shape
    topk_num = topk_ids.shape[1] if topk_weight is None else 1
    compute_type = tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float16

    use_fp8_w8a8 = (a.dtype == torch.float8_e4m3fn)
    use_int8_w8a8=(a.dtype == torch.int8 and b.dtype == torch.int8)

    up_dn = "up" if topk_weight is None else "dn"
    bs = topk_ids.shape[0]

    for config in tqdm(search_configs, desc=f"bs={bs} {up_dn} tuning"):
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_for_cache(topk_ids, config["BLOCK_SIZE_M"], E)
        grid = lambda META: (
                triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
                * triton.cdiv(b.shape[1], META["BLOCK_SIZE_N"]),
            )

        torch.cuda.nvtx.range_push(f"bs={bs}, up_dn={up_dn}, config={config}")
        try:
            # print(f"debug: sorted_token_ids.shape[0]={sorted_token_ids.shape[0]}, topk_ids.numel()={topk_ids.numel()}")
            fused_moe_kernel[grid](
                a, b, c, None, a_scale, b_scale,
                topk_weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
                N, K, sorted_token_ids.shape[0], topk_ids.numel(),
                a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
                c.stride(1), c.stride(2),
                *((0, 0) if a_scale is None else a_scale.stride()),
                *((0, 0, 0) if b_scale is None else b_scale.stride()),
                0, 0,
                *((0, 0) if quant_group_shape is None else quant_group_shape),
                MUL_ROUTED_WEIGHT=(topk_weight is not None),
                top_k=topk_num,
                compute_type=compute_type,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=False,
                per_channel_quant=use_int8_w8a8,
                HAS_BIAS=False,
                use_valu=False,
                even_Ks=((K % config["BLOCK_SIZE_K"] == 0) and current_platform.has_device_capability((8, 9))),
                naive_block_assignment=(sorted_token_ids is None),
                **config
            )
        except OutOfResources as e:
            tqdm.write(f"compile error: {e}")
            continue
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def get_quant_config(quant_config: str, model_config: Dict):
    if quant_config == "int8_w8a8":
        use_int8_w8a8, use_fp8_w8a8 = True, False
        # per-channel
        quant_group = [0, 0]
    elif quant_config == "fp8_w8a8":
        use_int8_w8a8, use_fp8_w8a8 = False, True
        # block-wise
        quant_group = model_config["quantization_config"]["weight_block_size"]
    else:
        use_int8_w8a8, use_fp8_w8a8 = False, False
        quant_group = None
    return use_int8_w8a8, use_fp8_w8a8, quant_group


def tuning_fusedmoe_target_kernel(opts: Any, candidate_configs: Dict, model_metadata: Dict) -> None:
    '''
        tuning kernel iterate total batch size
    '''
    model_config =  model_metadata["model_config"]
    expert_token_distribution = model_metadata["expert_token_distribution"]
    use_int8_w8a8, use_fp8_w8a8, quant_group = get_quant_config(opts.quant_config, model_config)

    batch_size = [int(k) for k in expert_token_distribution.keys()] if opts.batch_size is None else opts.batch_size
    for M in sorted(batch_size):
        dist = expert_token_distribution.get(str(M))
        topk_ids = recover_topk_ids(dist, M, model_config).cuda()

        a, a_scale, b, b_scale, c, topk_weight = prepare_inputs(M, model_config, opts.tp_size, opts.dtype, False, use_int8_w8a8, use_fp8_w8a8, quant_group)
        main_tune_loop(a, a_scale, b, b_scale, c, quant_group, topk_weight, topk_ids, candidate_configs["up_gemm"])

        a, a_scale, b, b_scale, c, topk_weight = prepare_inputs(M, model_config, opts.tp_size, opts.dtype, True, use_int8_w8a8, use_fp8_w8a8, quant_group)
        main_tune_loop(a, a_scale, b, b_scale, c, quant_group, topk_weight, topk_ids, candidate_configs["dn_gemm"])
    print("tuning Finish!")


@functools.lru_cache
def get_limit_blocks(n_regs: int, num_warps: int, smems: int, kernel_name: str) -> int:
    '''
    calculate the limit blocks per multiproc for triton kernel
    '''
    properties = driver.active.utils.get_device_properties(0)
    regs_per_multiproc = properties["max_num_regs"]     # 128*1024
    smem_per_multiproc = properties["max_shared_mem"]   # 256*1024
    thread_per_warp = properties["warpSize"]            # 32
    reg_alloc_unit_size = 64
    warp_alloc_granularity = 8
    smem_alloc_unit_size = 128
    max_threadblocks_per_multiproc = math.floor(64 / num_warps)

    regs_per_warp = math.ceil(n_regs * thread_per_warp / reg_alloc_unit_size) * reg_alloc_unit_size
    regs_per_warp = max(regs_per_warp, reg_alloc_unit_size)
    warps_per_sm = math.floor(regs_per_multiproc / regs_per_warp / warp_alloc_granularity) * warp_alloc_granularity
    limit_blocks_due_to_regs = math.floor(warps_per_sm / num_warps)

    smem_per_block = math.ceil(smems / smem_alloc_unit_size) * smem_alloc_unit_size
    smem_per_block = max(smem_per_block, smem_alloc_unit_size)
    limit_blocks_due_to_smem = math.floor(smem_per_multiproc / smem_per_block)
    return min(limit_blocks_due_to_regs, limit_blocks_due_to_smem, max_threadblocks_per_multiproc)


def precompile_and_calculate(kernel_func: triton.JITFunction, all_args: Dict) -> float:
    '''
    This function's purpose is to pre-compile the Triton kernel and calculate the grid parameters required for Persistence based on the compilation results.
    '''
    compiled_kernel = kernel_func.warmup(*all_args["args"], grid=(1, ), **all_args["kwargs"])
    compiled_kernel._init_handles()
    num_regs = compiled_kernel.n_regs
    num_warps = compiled_kernel.metadata.num_warps
    shared_mem_size = compiled_kernel.metadata.shared
    kernel_name = compiled_kernel.metadata.name
    limit_blocks = get_limit_blocks(num_regs, num_warps, shared_mem_size, kernel_name)
    occ = num_warps * limit_blocks / 64
    return occ


def compile_worker_function(
    opts: Any, model_config: Dict, down_gemm: bool, specialize_pattens: List, config: dict
) -> Dict:
    M = 140      # dummy value
    use_int8_w8a8, use_fp8_w8a8, quant_group_shape = get_quant_config(opts.quant_config, model_config)
    a, a_scale, b, b_scale, c, topk_weight = prepare_fake_inputs(M, model_config, opts.tp_size, opts.dtype, down_gemm, use_int8_w8a8, use_fp8_w8a8, quant_group_shape)
    _, K = a.shape
    E, N, _ = b.shape
    topk_ids = torch.empty((M, model_config["num_experts_per_tok"]), dtype=torch.int32, device="cpu")
    topk_num = topk_ids.shape[1] if topk_weight is None else 1
    compute_type = tl.bfloat16 if a.dtype == torch.bfloat16 else tl.float16

    use_fp8_w8a8 = (a.dtype == torch.float8_e4m3fn)
    use_int8_w8a8 = (a.dtype == torch.int8 and b.dtype == torch.int8)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_for_cache(topk_ids, config["BLOCK_SIZE_M"], E)
    # print(f"debug: sorted_token_ids.shape[0]={sorted_token_ids.shape[0]}, topk_ids.numel()={topk_ids.numel()}")
    args = package_args(fused_moe_kernel[1])(
        a, b, c, None, a_scale, b_scale,
        topk_weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, sorted_token_ids.shape[0], topk_ids.numel(),
        a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
        c.stride(1), c.stride(2),
        *((0, 0) if a_scale is None else a_scale.stride()),
        *((0, 0, 0) if b_scale is None else b_scale.stride()),
        0, 0,
        *((0, 0) if quant_group_shape is None else quant_group_shape),
        MUL_ROUTED_WEIGHT=(topk_weight is not None),
        top_k=topk_num,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=False,
        per_channel_quant=use_int8_w8a8,
        HAS_BIAS=False,
        use_valu=False,
        even_Ks=((K % config["BLOCK_SIZE_K"] == 0) and current_platform.has_device_capability((8, 9))),
        naive_block_assignment=(sorted_token_ids is None),
        **config
    )
    try:
        for fake_param_1, fake_param_2 in specialize_pattens:
            temp = list(args["args"])
            temp[12] = fake_param_1
            temp[13] = fake_param_2
            args["args"] = tuple(temp)
            occ = precompile_and_calculate(fused_moe_kernel, args)

    except OutOfResources as e:
        tqdm.write(f"compile error: {e}")
        return None
    return config


def get_specialize_patterns(batch_sizes: List, num_experts: int, topk_num: int) -> Set[Tuple]:

    specialize_patterns = set()
    topk_ids_numel = topk_num * np.array(batch_sizes)
    topk_ids_numel_is_align = (topk_ids_numel % 16 == 0)
    if np.any(topk_ids_numel_is_align):
        specialize_patterns.add((16, 16))
    
    topk_ids_numel_is_not_align = ~topk_ids_numel_is_align
    if np.any(topk_ids_numel_is_not_align):
        max_padded_not_align = (topk_ids_numel > num_experts) & topk_ids_numel_is_not_align
        if np.any(max_padded_not_align):
            specialize_patterns.add((17, 17))
        if np.any(~max_padded_not_align):
            specialize_patterns.add((16, 17))
    
    assert len(specialize_patterns) > 0
    return specialize_patterns


def compiling_fusedmoe_target_kernel(opts: Any, search_configs: List, model_metadata: Dict):
    '''
        compiling kernel iterate total search_configs
    '''
    model_config =  model_metadata["model_config"]
    expert_token_distribution = model_metadata["expert_token_distribution"]
    batch_size = [int(k) for k in expert_token_distribution.keys()] if opts.batch_size is None else opts.batch_size
    specialize_patterns = get_specialize_patterns(batch_size, model_config["num_experts"], model_config["num_experts_per_tok"])

    num_proc = int(os.environ.get("KERNEL_COMPILATION_PARALLELISM", mp.cpu_count() - 1))
    chunksize = max(1, len(search_configs) // (num_proc * 4))
    print(f"use {num_proc} processes to compile")

    worker_up_function = functools.partial(compile_worker_function, opts, model_config, False, specialize_patterns)
    worker_dn_function = functools.partial(compile_worker_function, opts, model_config, True, specialize_patterns)
    compiled_up_results = list()
    compiled_dn_results = list()
    with mp.Pool(processes=num_proc) as pool:
        for res in tqdm(pool.imap(worker_up_function, search_configs, chunksize=chunksize), total=len(search_configs), desc="Compiling up Gemm"):
            if res is not None:
                compiled_up_results.append(res)

        for res in tqdm(pool.imap(worker_dn_function, search_configs, chunksize=chunksize), total=len(search_configs), desc="Compiling down Gemm"):
            if res is not None:
                compiled_dn_results.append(res)

    # save to local disk
    compiled_results = {"up_gemm": compiled_up_results, "dn_gemm": compiled_dn_results}
    with open("candidate_configs.json", "w", encoding="utf-8") as f:
        json.dump(compiled_results, f, ensure_ascii=False, indent=4)


def _data_argtype(name):
    # dtype=[torch.bfloat16, torch.float16, torch.float32]
    return eval(f"torch.{name.lower()}")


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Run Fusedmoe Target kernel")
    parser.add_argument("--task", type=str, choices=["compile", "tune"], required=True)
    parser.add_argument("--model", type=str, default="Qwen3-235B-A22B")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--input-metadata", type=str, default="input_metadata.json")
    parser.add_argument("--batch-size", type=int, nargs='+', default=None)
    parser.add_argument("--dtype", type=_data_argtype, default=torch.float16)
    parser.add_argument("--quant-config", type=str, default=None)
    return parser.parse_args()


def set_deterministic_seeds(seed: int = 2025):
    import random, os
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(args: Any) -> None:
    set_deterministic_seeds()
    input_metadata = get_input_metadata(args.input_metadata, args.model)
    if args.task == "compile":
        mp.set_start_method('spawn', force=True)
        configs = get_search_space()
        print(f"[*] Try to compile {len(configs)} configs")
        compiling_fusedmoe_target_kernel(args, configs, input_metadata)
    elif args.task == "tune":
        configs = get_candidate_configs()
        print(f"[*] Try to tune kernel from {len(configs)} compiled configs")
        tuning_fusedmoe_target_kernel(args, configs, input_metadata)
    else:
        print(f"Nothing to do")
        return


if __name__ == "__main__":
    main(parse_args())