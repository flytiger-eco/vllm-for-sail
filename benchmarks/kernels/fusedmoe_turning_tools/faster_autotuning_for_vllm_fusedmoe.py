
import torch
import re, os, ast, sys, json, argparse, subprocess
import pandas as pd
from typing import Any, List, Dict, Tuple, Callable
from run_fusedmoe_target import _data_argtype, get_quant_config, get_input_metadata
from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Run Fusedmoe Tuning")
    parser.add_argument("--model", type=str, default="Qwen3-235B-A22B")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--input-metadata", type=str, default="input_metadata.json")
    parser.add_argument("--batch-size", type=int, nargs='+', default=256)
    parser.add_argument("--dtype", type=_data_argtype, default=torch.float16)
    parser.add_argument("--quant-config", type=str, choices=["int8_w8a8", "fp8_w8a8", None], default=None)
    return parser.parse_args()


def run_cmd(cmd: List):
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, text=True, capture_output=False)
    return result


def nvtx_range_parse(value: str):
    if pd.isna(value):
        return pd.Series([None, None, None])
    try:
        pattern = r"bs=(\d+),\s*up_dn=(\w+),\s*config=(\{[^}]+\})"
        match = re.search(pattern, str(value))

        if match:
            bs = int(match.group(1))
            up_dn = match.group(2)
            config = ast.literal_eval(match.group(3))
            return pd.Series([bs, up_dn, config])
        else:
            return pd.Series([None, None, None])
    except Exception as e:
        print(f"Analysis Error: {value}, {e}")
        return pd.Series([None, None, None])


def remove_middle_consecutive_duplicates_configs(
    df: pd.DataFrame, config_columns: List[str] = ['up_config', 'dn_config'], min_consecutive: int = 3
) -> pd.DataFrame:
    df = df.copy()
    def make_hashable(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(make_hashable(item) for item in obj)
        return obj

    hash_col = '_config_hash'
    df[hash_col] = df.apply(
        lambda row: tuple(make_hashable(row[col]) for col in config_columns),
        axis=1
    )

    df['_group'] = (df[hash_col] != df[hash_col].shift()).cumsum()
    df['_group_size'] = df.groupby('_group')[hash_col].transform('size')
    df['_pos_in_group'] = df.groupby('_group').cumcount()

    keep_condition = (
        (df['_group_size'] < min_consecutive) |
        (df['_pos_in_group'] == 0) |
        (df['_pos_in_group'] == df['_group_size'] - 1)
    )

    result = df[keep_condition].drop(
        columns=[hash_col, '_group', '_group_size', '_pos_in_group']
    ).reset_index(drop=True)

    return result


def find_optimal_config(df: pd.DataFrame) -> pd.DataFrame:
    up_group = df[df['up_dn'] == 'up']
    up_idx = up_group.groupby(['batch_size', 'BLOCK_SIZE_M'])['Kernel Duration (ns)'].idxmin()
    up_gemm_min = up_group.loc[up_idx].reset_index(drop=True)

    dn_group = df[df['up_dn'] == 'dn']
    dn_idx = dn_group.groupby(['batch_size', 'BLOCK_SIZE_M'])['Kernel Duration (ns)'].idxmin()
    dn_gemm_min = dn_group.loc[dn_idx].reset_index(drop=True)

    result = pd.merge(
        up_gemm_min[['Kernel Name', 'batch_size', 'BLOCK_SIZE_M', 'Kernel Duration (ns)', 'config']],
        dn_gemm_min[['Kernel Name', 'batch_size', 'BLOCK_SIZE_M', 'Kernel Duration (ns)', 'config']],
        on=['Kernel Name', 'batch_size', 'BLOCK_SIZE_M'],
        suffixes=('_up', '_dn')
    )

    result = result.rename(columns={
        'Kernel Duration (ns)_up': 'up_gemm_min_duration',
        'Kernel Duration (ns)_dn': 'dn_gemm_min_duration',
        'config_up': 'up_config',
        'config_dn': 'dn_config'
    })

    result['total_duration'] = result['up_gemm_min_duration'] + result['dn_gemm_min_duration']
    min_idx = result.groupby('batch_size')['total_duration'].idxmin()
    result_min = result.loc[min_idx].reset_index(drop=True)
    return result_min


def save_result_as_json(result: pd.DataFrame, filename: str, prune: bool=True):
    if prune:
        result = remove_middle_consecutive_duplicates_configs(result, ['up_config', 'dn_config'])

    config_dict = {
        row['batch_size']: {'BLOCK_SIZE_M': row['BLOCK_SIZE_M'], 'USE_VALU': False, "UP": row['up_config'], "DOWN": row['dn_config']}
        for _, row in result.iterrows()
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f"save result to {filename}")

    with open('optimal_configs.json', 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print("Performance Result: ")
    result['total_duration_us'] = result['total_duration'] / 1000.0
    print(result[['Kernel Name', 'batch_size', 'total_duration_us']])
    print(config_dict)


def postprocess(load_file: str, filename: str):
    df  = pd.read_csv(load_file)
    df = df[df['Kernel Name'] == "fused_moe_kernel"]
    df[['batch_size', 'up_dn', 'config']] = df['Range Name'].apply(nvtx_range_parse)
    df['BLOCK_SIZE_M'] = df['config'].apply(lambda x: x.get('BLOCK_SIZE_M') if isinstance(x, dict) else None)
    my_df = df[['Kernel Name', 'batch_size', 'up_dn', 'BLOCK_SIZE_M', 'config', 'Kernel Duration (ns)']]
    optimal_configs = find_optimal_config(my_df)
    save_result_as_json(optimal_configs, filename)


def main(args: Any) -> None:

    print(f"Step 1: Launch Compilation Task")
    run_cmd(["python3", "run_fusedmoe_target.py", *sys.argv[1:], "--task", "compile"])

    print(f"Step 2: Launch Tuning Task")
    run_cmd(["asys", "profile", "-o", "tune_report", "-f", "true",
            "python3", "run_fusedmoe_target.py", *sys.argv[1:], "--task", "tune"])

    print(f"Step 3: Launch asys/nsys stats the Report")
    run_cmd(["asys", "stats", "--force-overwrite=true", "-r", "hgtx_kern_trace:standalone", "-o", "tuning", "tune_report.asysrep"])

    print(f"Step 4: Generate Optimal Configs")
    model_config = get_input_metadata(args.input_metadata, args.model).get("model_config")
    use_int8_w8a8, use_fp8_w8a8, block_quant_shape = get_quant_config(args.quant_config, model_config)
    dtype_str = _get_config_dtype_str(args.dtype, use_int8_w8a8=use_int8_w8a8, use_fp8_w8a8=use_fp8_w8a8)
    filename = get_config_file_name(model_config["num_experts"], model_config["moe_intermediate_size"] // args.tp_size,
                                    dtype_str, block_quant_shape)
    postprocess("./tuning_hgtx_kern_trace.csv", filename)

    print("OK")


if __name__ == "__main__":
    main(parse_args())