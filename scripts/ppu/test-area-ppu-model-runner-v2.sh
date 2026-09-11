#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-model-runner-v2.sh — PPU Model Runner V2 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-model-runner-v2.yml（容器内，cwd = /workspace）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single | multi
#
# 关键：本 area 所有 step 均需 VLLM_USE_V2_MODEL_RUNNER=1（上游 buildkite
# model_runner_v2.yaml 每个 step 都 export，是本 area 的核心开关）——见 [env] 段。
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# workflow_dispatch 的 pytest_args 透传：按空白切分后追加到每个 step 的
# pytest 命令尾部（如 `-k test_foo -x`），排障时缩小范围而不必改脚本。
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-model-runner-v2-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area 特有 pip 依赖：无。
# ------------------------------------------------------------------------------
# 上游 buildkite model_runner_v2.yaml 的 Examples step 会 `pip install tensorizer`，
# 但本 area PPU scope 不含 Examples（python3 脚本非 pytest，且大量 MM 模型未 staged，
# 见 aone_ci/ppu_extras/model_runner_v2.yaml 审计小结）。core/distributed/spec_decode
# 三档 step 只用镜像预装栈 + ppu_install_dependency.sh 已装的 pytest 工具链
# （pytest-asyncio/tblib/pytest-shard/pyyaml），故此处无补装。

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/model_runner_v2.yaml single/multi 段）
# ------------------------------------------------------------------------------
# single = 上游 buildkite "Model Runner V2 Core Tests" + "Spec Decode" 两 step
# 的 PPU 版（单卡）。逐条对回 aone_ci/ppu_extras/model_runner_v2.yaml single 段。

# mrv2_core_llm_engine：v1/engine/test_llm_engine.py（opt-125m）。
# 排除 test_engine_metrics（上游 buildkite 同排除；PPU 无 metrics 采集依赖）。
MRV2_CORE_LLM_ENGINE_ARGS=(
  tests/v1/engine/test_llm_engine.py
  -k
  "not test_engine_metrics"
)

# mrv2_core_async_scheduling：v1/e2e/general/test_async_scheduling.py。
# 上游 buildkite 用 `not ngram`；PPU extras 追加 `and not eagle3`
# （eagle3 speculator 在 PPU 上 model redirect path rewrite 断言失败，
# LANDMINE #18）。整个 step 需 ENFORCE_EAGER=1（见 mode dispatch 处前缀；
# 上游注释：CG correctness 问题未解前需 eager，PR #32936 合入后可去）。
MRV2_CORE_ASYNC_SCHEDULING_ARGS=(
  tests/v1/e2e/general/test_async_scheduling.py
  -k
  "not ngram and not eagle3"
)

# mrv2_core_context_length：v1/e2e/general/test_context_length.py（JackFram/llama-160m）
MRV2_CORE_CONTEXT_LENGTH_ARGS=(
  tests/v1/e2e/general/test_context_length.py
)

# mrv2_core_min_tokens：v1/e2e/general/test_min_tokens.py（opt-125m）
MRV2_CORE_MIN_TOKENS_ARGS=(
  tests/v1/e2e/general/test_min_tokens.py
)

# mrv2_core_struct_output：entrypoints/llm/test_struct_output_generate.py
# （v0.23 起 tests/v1/entrypoints/ 整体迁入 tests/entrypoints/）。
# 只跑 xgrammar 后端（上游 buildkite 同）；排除 4 个 speculative_config 参数化
# 实例（ngram spec decoding，PPU 侧不跑）。
MRV2_CORE_STRUCT_OUTPUT_ARGS=(
  tests/entrypoints/llm/test_struct_output_generate.py
  -k
  "xgrammar and not speculative_config6 and not speculative_config7 and not speculative_config8 and not speculative_config0"
)

# mrv2_spec_decode_e2e：v1/e2e/spec_decode/test_spec_decode.py -k eagle/mtp。
# 排除 llama3/qwen3 eagle3 speculator（PPU model redirect alias-rewrite 断言失败，
# LANDMINE #18）与 _heavy（超大模型/长跑）。附 --ignore lora spec decode 文件
# （上游 spec decode step 不含 lora 组合，PPU extras 显式 ignore）。
MRV2_SPEC_DECODE_E2E_ARGS=(
  tests/v1/e2e/spec_decode/test_spec_decode.py
  -k
  "(eagle or mtp) and not llama3_eagle3_speculator and not qwen3_eagle3_speculator and not _heavy"
  --ignore=tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py
)

# multi = 上游 buildkite "Distributed (2 GPUs)" + "Pipeline Parallelism" 的
# PPU 版（测试内部用 2 卡）。逐条对回 ppu_extras model_runner_v2.yaml multi 段。

# mrv2_distributed_basic_correctness：basic_correctness/test_basic_correctness.py
# 的 distributed(num_gpus=2) 组。`not ray` 排 ray 后端（PPU 上 hang）；
# `not True` 排 prompt_embeds 参数化实例（PPU 尚不支持，上游注释亦为 hacky filter）。
# 需 TARGET_TEST_SUITE=L4（见 mode dispatch 前缀）。
MRV2_DISTRIBUTED_BASIC_CORRECTNESS_ARGS=(
  tests/basic_correctness/test_basic_correctness.py
  -m
  "distributed(num_gpus=2)"
  -k
  "not ray and not True"
)

# mrv2_distributed_dp：v1/distributed/test_async_llm_dp.py DP_SIZE=2。
# `not ray`（ray DP hang）；`not tiny`（tiny-random 相关）；`not (dp_pause and False)`：
# dp_pause 用例 id 不带模型名（参数化为 expert_parallel=[False,True]），[False] 分支
# 硬编码 tiny-random(head_dim=4) 撞 FlexAttention inductor 编译(需 E>=16)，[True]
# MoE+EP 保留。需 TP_SIZE=1 DP_SIZE=2 NCCL_CUMEM_HOST_ENABLE=0（见前缀）。
MRV2_DISTRIBUTED_DP_ARGS=(
  tests/v1/distributed/test_async_llm_dp.py
  -k
  "not ray and not tiny and not (dp_pause and False)"
)

# mrv2_distributed_eagle_dp：v1/distributed/test_eagle_dp.py DP_SIZE=2。
# 需 TP_SIZE=1 DP_SIZE=2 NCCL_CUMEM_HOST_ENABLE=0（见前缀）。
MRV2_DISTRIBUTED_EAGLE_DP_ARGS=(
  tests/v1/distributed/test_eagle_dp.py
)

# mrv2_pipeline_parallelism：distributed/test_pipeline_parallel.py PP。
# 上游 buildkite 用 `not ray and not Jamba`；PPU extras 追加 `not PowerLM and not
# DeepSeek`（PowerLM-3b / DeepSeek-V2-Lite 架构在 PPU 引擎初始化 crash，LANDMINE #20）。
# 注：本文件 create_new_process_for_each_test 在 PPU 上已走 spawn（tests/utils.py
# is_ppu() 分支），无 fork 子进程设备初始化 landmine。上游 4-GPU PP（Llama-4-Scout）
# 需 4-PPU pod，本 2 卡 area 不含。
MRV2_PIPELINE_PARALLELISM_ARGS=(
  tests/distributed/test_pipeline_parallel.py
  -k
  "not ray and not Jamba and not PowerLM and not DeepSeek"
)

# ------------------------------------------------------------------------------
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# 本 area 核心开关：启用 V2 model runner（上游 buildkite model_runner_v2.yaml
# 每个 step 都 `export VLLM_USE_V2_MODEL_RUNNER=1`，ppu_extras 亦声明为 area 级
# env。aone 自动生成脚本漏了这行；GHA 版按源头补齐，否则整个 area 语义失效）。
export VLLM_USE_V2_MODEL_RUNNER="1"

# 默认离线（模型走 /nas_aisw 预置卷）；需要在线下载时设 PPU_TEST_ONLINE=1
if [[ "${PPU_TEST_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# HF 缓存：workflow 已注入 HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub；
# 未注入时（本地调试）探测 /nas_aisw 下的候选路径
if [ -z "${HF_HUB_CACHE:-}" ]; then
  for _cand in /nas_aisw/datasets/hf_cache/hub "$HOME/.cache/huggingface/hub"; do
    if [ -d "${_cand}" ]; then
      export HF_HUB_CACHE="${_cand}"
      break
    fi
  done
fi
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"

# PPU SDK: Triton/Inductor 编译需要 cuda.h + ptxas + libcuda（缺失时
# torch.compile 类测试 BackendCompilerFailed）
PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# ------------------------------------------------------------------------------
# [setup] HF hub cache symlink：HF id → /nas_aisw 本地检查点
# ------------------------------------------------------------------------------
# 路径优先取红区已存模型清单 scripts/ppu/model_alises/*.json 的 path 字段
# （NAS 绝对路径 = /nas_aisw/datasets/ + path）；清单未收录的按 Aone
# /ppusw 同构路径标注"待确认"。路径不存在时 WARN 并跳过。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # ---- single: core tests ----
    # opt-125m: test_llm_engine.py / test_min_tokens.py / basic_correctness distributed
    # 清单命中：checkpoints_cleaned.json path=checkpoints/LLM/misc/v1.0/opt-125m
    "facebook/opt-125m": "/nas_aisw/datasets/checkpoints/LLM/misc/v1.0/opt-125m",
    # test_llm_engine.py / test_async_scheduling.py / basic_correctness / PP TEST_MODELS
    # 清单命中（cleaned name=Llama-3.2-1B-Instruct path v3.2；Aone /ppusw 为 v3.1，
    # 以红区主清单 v3.2 为准，与 basic-correctness sh 一致）
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
    # test_async_scheduling.py（Qwen3-0.6B）+ spec_decode 轻量 eagle 用例
    # 清单命中：checkpoints/LLM/qwen/v3/Qwen3-0.6B
    "Qwen/Qwen3-0.6B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
    # test_context_length.py（JackFram/llama-160m）
    # 清单命中：checkpoints/LLM/llama/v1.0/llama-160m
    "JackFram/llama-160m": "/nas_aisw/datasets/checkpoints/LLM/llama/v1.0/llama-160m",
    # test_struct_output_generate.py（xgrammar 组）主力模型
    # Ministral: Aone aliases 命中 LLM/Ministral/v1.0
    "mistralai/Ministral-8B-Instruct-2410":
        "/nas_aisw/datasets/checkpoints/LLM/Ministral/v1.0/Ministral-8B-Instruct-2410",
    # Qwen2.5-1.5B: 清单命中 checkpoints/LLM/qwen/v2.5/Qwen2.5-1.5B-Instruct
    "Qwen/Qwen2.5-1.5B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-1.5B-Instruct",
    # struct_output.py 用（同文件另有 speculative 组，已被 -k 排除）
    "meta-llama/Meta-Llama-3.1-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Meta/v1.0/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-1.7B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B",
    # struct_output.py 内 deepseek 蒸馏模型
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        "/nas_aisw/datasets/checkpoints/LLM/deepseek/R1/DeepSeek-R1-Distill-Qwen-1.5B",

    # ---- single: spec_decode eagle/mtp（-k 选中的用例；均 aone aliases/清单命中）----
    # Llama-3.1-8B base（清单主清单 path v3.1）
    "meta-llama/Llama-3.1-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.1/Llama-3.1-8B-Instruct",
    # eagle draft（yuhuili EAGLE/EAGLE3）
    "yuhuili/EAGLE-LLaMA3.1-Instruct-8B":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-LLaMA3.1-Instruct-8B",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE3/v3/EAGLE3-LLaMA3.1-Instruct-8B",
    # qwen3 eagle3 组（eagle3 用例）
    "Qwen/Qwen3-8B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-8B",
    "AngelSlim/Qwen3-8B_eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-8B_eagle3",
    # deepseek_eagle 组（eagle618 随机小模型）
    "eagle618/deepseek-v3-random":
        "/nas_aisw/datasets/checkpoints/LLM/deepseek/v3/deepseek-v3-random",
    "eagle618/eagle-deepseek-v3-random":
        "/nas_aisw/datasets/checkpoints/LLM/eagle/v3/eagle-deepseek-v3-random",
    # mtp 组
    "XiaomiMiMo/MiMo-7B-Base":
        "/nas_aisw/datasets/checkpoints/LLM/MiMo/v1.0/MiMo-7B-Base",
    "ZixiQi/DeepSeek-V3-4layers-MTP-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V3/DeepSeek-V3-4layers-MTP-FP8",
    # 轻量 eagle 参数化组用到的 Qwen3 变体
    "Qwen/Qwen3-1.7B-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B-FP8",
    "Qwen/Qwen3-0.6B-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B-FP8",
    "amd/PARD-Qwen3-0.6B":
        "/nas_aisw/datasets/checkpoints/LLM/PARD/v1.0/PARD-Qwen3-0.6B",
    # speculator 组（被 -k not *_speculator 排除，仍建 link 以防其它引用）
    "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/Llama-3.1-8B-Instruct-speculator.eagle3",
    "RedHatAI/Qwen3-8B-speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3-8B-speculator.eagle3",
    # spec_decode 数据集（likaixin/InstructCoder，datasets 路径）
    "likaixin/InstructCoder":
        "/nas_aisw/datasets/datasets/LLM/likaixin/v1.0/InstructCoder",
    # Llama-4-Scout eagle 组（多为 _heavy/large 被 -k/mark 排除，建 link 兜底）
    "meta-llama/Llama-4-Scout-17B-16E-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/llama/v4/Llama-4-Scout-17B-16E-Instruct",
    "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-Llama-4-Scout-17B-16E-Instruct",
    # VL eagle3 组（对应用例 pytest.mark.skip，Rayzl 命中/ taobao-mnn 待入库）
    "Qwen/Qwen3-VL-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-VL-7B-Instruct",
    "Rayzl/qwen2.5-vl-7b-eagle3-sgl":
        "/nas_aisw/datasets/checkpoints/LLM/qwen2.5/v1.0/qwen2.5-vl-7b-eagle3-sgl",
    # taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3：两处清单 MISS → 待入库
    # （对应 qwen3_vl_eagle3 用例已 pytest.mark.skip，暂不阻塞；入库后补路径）
    # "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3": "<NAS path 待入库后补>",

    # ---- multi: distributed / DP / eagle_dp ----
    # basic_correctness distributed 用 tiny-random（清单未收录，Aone aliases 命中 tiny/v1.0）
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
    "hmellor/tiny-random-Gemma2ForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-Gemma2ForCausalLM",
    # test_async_llm_dp.py（DP_SIZE=2）用 PowerMoE-3b
    # HF org=ibm-research（test 引用）；Aone aliases 命中 PowerMoE/v1.0
    "ibm-research/PowerMoE-3b":
        "/nas_aisw/datasets/checkpoints/LLM/PowerMoE/v1.0/PowerMoE-3b",
    # test_eagle_dp.py 复用上面 Llama-3.1-8B + EAGLE-LLaMA3.1-Instruct-8B

    # ---- multi: pipeline_parallel TEST_MODELS（-k 选中：非 ray/Jamba/PowerLM/DeepSeek）----
    "microsoft/Phi-3.5-MoE-instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Phi/v1.0/Phi-3.5-MoE-instruct",
    "hmellor/Ilama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/Ilama/v3.2/Ilama-3.2-1B",
    "intfloat/e5-mistral-7b-instruct":
        "/nas_aisw/datasets/checkpoints/LLM/e5/v1/e5-mistral-7b-instruct",
    "BAAI/bge-multilingual-gemma2":
        "/nas_aisw/datasets/checkpoints/LLM/bge/v1.0/bge-multilingual-gemma2",
    "OpenGVLab/InternVL2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/InternVL/v1.0/InternVL2-1B",
    "microsoft/Phi-3.5-vision-instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Phi-3/v3.5/Phi-3.5-vision-instruct",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b":
        "/nas_aisw/datasets/checkpoints/LLM/ultravox/v0/ultravox-v0_5-llama-3_2-1b",
    # 说明：PP TEST_MODELS 中 ibm/PowerLM-3b、deepseek-ai/DeepSeek-V2-Lite-Chat、
    # ai21labs/Jamba-tiny-dev 被 -k not(PowerLM/DeepSeek/Jamba) 排除，不建 link。
}

HF_CACHE = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
    "~/.cache/huggingface/hub")
os.makedirs(HF_CACHE, exist_ok=True)

created = skipped = missing = 0
for hf_id, local_path in MODEL_MAP.items():
    if not os.path.isdir(local_path):
        print(f"[setup] MISS  {hf_id} -> {local_path} (tests using it will fail)")
        missing += 1
        continue
    org, repo = hf_id.split("/", 1)
    cache_dir = os.path.join(HF_CACHE, f"models--{org}--{repo}")
    snap_link = os.path.join(cache_dir, "snapshots", "main")
    refs_file = os.path.join(cache_dir, "refs", "main")
    if os.path.islink(snap_link) or os.path.exists(snap_link):
        skipped += 1
        continue
    os.makedirs(os.path.dirname(snap_link), exist_ok=True)
    os.makedirs(os.path.dirname(refs_file), exist_ok=True)
    os.symlink(local_path, snap_link)
    with open(refs_file, "w") as rf:
        rf.write("main")
    created += 1
    print(f"[setup] OK    {hf_id} -> {local_path}")
print(f"[setup] symlinks: created={created} skipped={skipped} "
      f"missing={missing} (total {len(MODEL_MAP)})")
PYEOF

# ------------------------------------------------------------------------------
# [junit] 合并基础设施（含 pytest 崩溃兜底）
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329  # 只由下方 `trap _emit_junit EXIT` 调用
_emit_junit() {
  python3 - <<PYEOF
import glob, os
from xml.etree import ElementTree as ET

OUT = "${RESULTS_DIR}/test.xml"
TMP = "${TMP_JUNIT}"
LABELS = "${STEP_LABELS_LIST}".split()

def _collect(label):
    shard_paths = sorted(glob.glob(f"{TMP}/{label}-shard*.xml"))
    if shard_paths:
        return shard_paths
    single = f"{TMP}/{label}.xml"
    return [single] if os.path.exists(single) else []

root = ET.Element("testsuites", name="vLLM PPU Model Runner V2 (GHA)")
for label in LABELS:
    paths = _collect(label)
    if not paths:
        ts = ET.SubElement(root, "testsuite", name=label, tests="1", errors="1",
                           failures="0", skipped="0", time="0")
        tc = ET.SubElement(ts, "testcase", name=label,
                           classname=f"gha_ci.{label}", time="0")
        err = ET.SubElement(tc, "error",
                            message="junit-xml not produced (pytest crashed before writing)")
        err.text = f"file missing: {TMP}/{label}.xml or shards"
        continue
    for path in paths:
        try:
            tree = ET.parse(path)
            for ts in tree.iter("testsuite"):
                ts.set("name", label)
                root.append(ts)
        except ET.ParseError:
            pass
ET.ElementTree(root).write(OUT, encoding="UTF-8", xml_declaration=True)
print(f"[junit] test.xml emitted -> {OUT}")

# ---- step summary：分 shard 统计表（markdown）
SUMMARY = os.path.join(os.path.dirname(OUT), "summary.md")
COLS = ("tests", "failures", "errors", "skipped", "time")

def _stats(path):
    root_ = ET.parse(path).getroot()
    suites = [root_] if root_.tag == "testsuite" else list(root_.iter("testsuite"))
    agg = dict.fromkeys(COLS, 0.0)
    for s in suites:
        for k in COLS:
            agg[k] += float(s.get(k) or 0)
    return agg

lines = ["### Model Runner V2 Test (PPU)", "",
         "| unit | tests | passed | failed | errors | skipped | time | status |",
         "|---|---:|---:|---:|---:|---:|---:|:-:|"]
tot = dict.fromkeys(COLS, 0.0)
bad_units = 0
for label in LABELS:
    paths = _collect(label)
    if not paths:
        lines.append(f"| {label} | - | - | - | - | - | - | FAILED (no junit, crashed) |")
        bad_units += 1
        continue
    for path in paths:
        unit = label
        base = os.path.basename(path)
        if "-shard" in base:
            unit = f"{label} / shard {base.rsplit('-shard', 1)[1][:-4]}"
        try:
            agg = _stats(path)
        except (ET.ParseError, OSError):
            lines.append(f"| {unit} | - | - | - | - | - | - | FAILED (bad junit) |")
            bad_units += 1
            continue
        n_t, n_f, n_e, n_s = (int(agg[k]) for k in COLS[:4])
        ok = (n_f == 0 and n_e == 0)
        if not ok:
            bad_units += 1
        lines.append(f"| {unit} | {n_t} | {n_t - n_f - n_e - n_s} | {n_f} "
                     f"| {n_e} | {n_s} | {agg['time']:.0f}s | "
                     f"{'PASSED' if ok else 'FAILED'} |")
        for k in COLS:
            tot[k] += agg[k]
n_t, n_f, n_e, n_s = (int(tot[k]) for k in COLS[:4])
status_all = "PASSED" if bad_units == 0 else "FAILED"
lines.append(f"| **Total** | **{n_t}** | **{n_t - n_f - n_e - n_s}** "
             f"| **{n_f}** | **{n_e}** | **{n_s}** | **{tot['time']:.0f}s** "
             f"| **{status_all}** |")
with open(SUMMARY, "w") as sf:
    sf.write("\n".join(lines) + "\n")
print(f"[summary-md] {SUMMARY} emitted")
PYEOF
  # 分片 step 的 pytest 输出重定向到 TMP_JUNIT，一并收进 artifact 便于排障
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
  # 容器以 root 运行，产物须可被 runner 用户读取（upload-artifact）
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
  # [k8s-summary] 经 NAS 回传 per-unit 统计表：RESULTS_DIR 在 Pod 内、回收即失；
  if [ -n "${PPU_SUMMARY_NAS_DIR:-}" ] && [ -f "${RESULTS_DIR}/summary.md" ]; then
    if mkdir -p "${PPU_SUMMARY_NAS_DIR}" 2>/dev/null && \
       cp -f "${RESULTS_DIR}/summary.md" "${PPU_SUMMARY_NAS_DIR}/summary.md"; then
      echo "[k8s-summary] summary.md -> ${PPU_SUMMARY_NAS_DIR}/summary.md"
    else
      echo "[k8s-summary] WARN: NAS 回传失败（${PPU_SUMMARY_NAS_DIR} 不可写？）"
    fi
  fi
}
trap _emit_junit EXIT

# ------------------------------------------------------------------------------
# [run] 单 step 执行器：shards>1 → 并发分片；shards==1 → 单进程
# ------------------------------------------------------------------------------
TOTAL_RC=0

_run_step() {
  local label="$1" shards="$2"
  shift 2
  local args=("$@")
  STEP_LABELS_LIST="${STEP_LABELS_LIST} ${label}"

  if [ "${shards}" -gt 1 ]; then
    echo "========== [step] ${label} shards=${shards} =========="
    local pids=()
    for shard in $(seq 0 $((shards - 1))); do
      local out_xml="${TMP_JUNIT}/${label}-shard${shard}.xml"
      CUDA_VISIBLE_DEVICES="${shard}" pytest -v -s "${args[@]}" ${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"} \
        --shard-id="${shard}" --num-shards="${shards}" \
        --junit-xml="${out_xml}" \
        > "${TMP_JUNIT}/${label}-shard${shard}.log" 2>&1 &
      pids+=($!)
      echo "[shard] launched shard ${shard} pid=${!} CUDA_VISIBLE_DEVICES=${shard}"
    done
    local rc_total=0 i=0
    for pid in "${pids[@]}"; do
      set +e; wait "${pid}"; local rc=$?; set -e
      echo "[shard] shard ${i} pid=${pid} rc=${rc}"
      if [ "${rc}" -ne 0 ]; then rc_total=1; fi
      i=$((i + 1))
    done
    echo "[step] ${label} aggregate rc=${rc_total}"
    for shard in $(seq 0 $((shards - 1))); do
      echo "----- ${label}-shard${shard}.log (tail) -----"
      tail -20 "${TMP_JUNIT}/${label}-shard${shard}.log" 2>/dev/null || echo "(no log)"
    done
    if [ "${rc_total}" -ne 0 ]; then TOTAL_RC=1; fi
  else
    echo "========== [step] ${label} =========="
    local out_xml="${TMP_JUNIT}/${label}.xml"
    set +e
    pytest -v -s "${args[@]}" ${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"} --junit-xml="${out_xml}"
    local rc=$?
    set -e
    echo "[step] ${label} rc=${rc}"
    if [ "${rc}" -ne 0 ]; then TOTAL_RC=1; fi
  fi
}

# 复刻 aone_ci/scripts/test_area_ppu_model_runner_v2.sh 的 single/multi 段：
# 每个 step 独立 pytest 进程；per-step env（ENFORCE_EAGER / TARGET_TEST_SUITE /
# TP_SIZE / DP_SIZE / NCCL_CUMEM_HOST_ENABLE）以命令前缀注入，只作用于该次
# _run_step 调用。所有 step shards=1（本 area 无分片，测试内部自用多卡）。
_run_single_steps() {
  _run_step "mrv2_core_llm_engine" 1 "${MRV2_CORE_LLM_ENGINE_ARGS[@]}"
  ENFORCE_EAGER=1 \
    _run_step "mrv2_core_async_scheduling" 1 "${MRV2_CORE_ASYNC_SCHEDULING_ARGS[@]}"
  _run_step "mrv2_core_context_length" 1 "${MRV2_CORE_CONTEXT_LENGTH_ARGS[@]}"
  _run_step "mrv2_core_min_tokens" 1 "${MRV2_CORE_MIN_TOKENS_ARGS[@]}"
  _run_step "mrv2_core_struct_output" 1 "${MRV2_CORE_STRUCT_OUTPUT_ARGS[@]}"
  _run_step "mrv2_spec_decode_e2e" 1 "${MRV2_SPEC_DECODE_E2E_ARGS[@]}"
}

_run_multi_steps() {
  TARGET_TEST_SUITE=L4 \
    _run_step "mrv2_distributed_basic_correctness" 1 \
      "${MRV2_DISTRIBUTED_BASIC_CORRECTNESS_ARGS[@]}"
  TP_SIZE=1 DP_SIZE=2 NCCL_CUMEM_HOST_ENABLE=0 \
    _run_step "mrv2_distributed_dp" 1 "${MRV2_DISTRIBUTED_DP_ARGS[@]}"
  TP_SIZE=1 DP_SIZE=2 NCCL_CUMEM_HOST_ENABLE=0 \
    _run_step "mrv2_distributed_eagle_dp" 1 "${MRV2_DISTRIBUTED_EAGLE_DP_ARGS[@]}"
  _run_step "mrv2_pipeline_parallelism" 1 "${MRV2_PIPELINE_PARALLELISM_ARGS[@]}"
}

if [ "${MODE}" = "single" ]; then
  _run_single_steps
elif [ "${MODE}" = "multi" ]; then
  _run_multi_steps
else  # all
  _run_single_steps
  _run_multi_steps
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
