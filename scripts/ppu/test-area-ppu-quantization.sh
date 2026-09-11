#!/bin/bash
# [ci-smoke] 第二批 12 area PR 门禁全量验证触碰行（本 PR 勿合并）
# ==============================================================================
# scripts/ppu/test-area-ppu-quantization.sh — PPU Quantization 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-quantization.yml（容器内，cwd = /workspace）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — 本 area 为 single-only（无 multi step）
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac
# quantization 为 single-only；multi 无 step 可跑（见底部 dispatch）

# workflow_dispatch 的 pytest_args 透传：按空白切分后追加到每个 step 的
# pytest 命令尾部（如 `-k test_foo -x`），排障时缩小范围而不必改脚本。
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-quantization-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area 特有依赖：无。
# ------------------------------------------------------------------------------
# 上游 .buildkite/test_areas/quantization.yaml 的 "Quantization" step 装
# torchao==0.14.1 + conch-triton-kernels，但 PPU 侧把 test_torchao.py 整文件
# ignore（torchao 需 SM≥8.9 + 版本过低），conch 相关用例同样不跑，故无需补装。
# bitsandbytes / auto-round / AWQ 相关能力由镜像预装栈提供（ppu_extras 审计
# 确认 bitsandbytes 已装）。若首跑发现缺包，在此按 `import 探测再补` 补装。

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/quantization.yaml single 段）
# ------------------------------------------------------------------------------
# 上游 3 个 step 映射：
#   Step 1 "Quantization"              → tests/quantization/（17 文件，逐文件审计见下）
#   Step 2 "Quantized MoE Test (B200)" → 整段跳过（SM100+ Blackwell，PPU 不支持）
#   Step 3 "Quantized Models Test"     → tests/models/quantization/（9 文件）
#
# quantization_core = 上游 Step 1 的 PPU 版：tests/quantization/ 全量减 14 个
#   ignore + fp8 kv -k 排除 + 10 个 deselect。Run 34580922394 后实际会跑：
#   test_auto_round / test_configs / test_experts_int8 等文件的 PPU 可用子集
#   （各排除项的根因与恢复条件见参数块内注释）。
QUANT_CORE_ARGS=(
  tests/quantization/
  # SM100+ Blackwell，需 flashinfer
  --ignore=tests/quantization/test_blackwell_moe.py
  # CPU-only（module-level pytest.skip if not CPU）
  --ignore=tests/quantization/test_cpu_wna16.py
  # fp8 core，需 SM≥8.9（PPU 报 SM8.0）
  # （上游 ROCm-only 的 test_ptpc_fp8.py 已并入 test_quark.py，原死 ignore 已移除）
  --ignore=tests/quantization/test_fp8.py
  # fp8 依赖（SM≥8.9）+ 大量 HF 模型
  --ignore=tests/quantization/test_cpu_offload.py
  # fp8（SM≥8.9）+ HF snapshot_download
  --ignore=tests/quantization/test_modelopt.py
  # AMD quark + lm_eval + HF access
  --ignore=tests/quantization/test_quark.py
  --ignore=tests/quantization/test_mixed_precision.py
  # torchao 已装但版本过低（需 ≥0.14.0）+ fp8 Float8 需 CUDA≥8.9
  --ignore=tests/quantization/test_torchao.py
  # compressed_tensors 包不在 PPU CI 镜像；且含大量 fp8/SM90+ 用例
  --ignore=tests/quantization/test_compressed_tensors.py
  # 模型未 stage 到红区（无外网）
  --ignore=tests/quantization/test_lm_head.py
  --ignore=tests/quantization/test_gptq_dynamic.py
  --ignore=tests/quantization/test_gptq_v2.py
  # Run 34580922394: 6/6 全挂 —— fp8e4nv 需 SM8.9+（PPU 报 SM8.0），
  # is_quant_method_supported("fp8") 在 PPU 误判可用导致未 skip。
  --ignore=tests/quantization/test_online.py
  # Run 34580922394: test_per_token_kv_cache.py 的 fp8 参数化 23 例全挂
  # （Triton: type fp8e4nv not supported，PPU 仅支持 fp8e4b15/fp8e5）；
  # "[fp8-"/"[fp8]" 前缀精确匹配失败 id，保留 [fp8_per_token_head] 等
  # 无 kernel 路径的通过用例与全部 int8 用例。
  -k
  "not [fp8- and not [fp8]"
  # Run 34580922394: TheBloke/Llama-2-7B-Chat-GPTQ 与
  # TheBloke/OpenHermes-2.5-Mistral-7B-AWQ 未 stage（[setup] MISS），
  # 期望识别成功的 exptype0/1/2/8/9/10 挂（期望 ERROR 的变体仍通过）。
  # 恢复条件：两个检查点入库后移除这 6 个 deselect。
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype0]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype1]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype2]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype8]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype9]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype10]"
  # Run 34580922394: TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ 未 stage（NAS 无
  # 该路径），LocalEntryNotFoundError。恢复条件：检查点入库后移除。
  --deselect "tests/quantization/test_auto_gptq.py::test_auto_gptq_quantization_method[TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ]"
  # Run 34580922394: Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound 引擎初始化
  # 失败（PPU torch_call_dispatcher aten::sum dim_IntList API call failed）；
  # 同文件 OPEA/Qwen2.5 变体通过。恢复条件：PPU aten::sum 修复后移除。
  --deselect "tests/quantization/test_auto_round.py::test_auto_round[Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound]"
  # Run 34580922394: Jamba-tiny-random 引擎启动时 ModelConfig ValidationError
  # （红区 staged 快照内容不被 vllm 接受）；plamo 变体自身 skip。
  --deselect "tests/quantization/test_experts_int8.py::test_model_experts_int8_startup[4-bfloat16-ai21labs/Jamba-tiny-random]"
  # Run 34580922394: PPU torch.linalg.det 数值偏差（|det|-1 = 0.1665，
  # tol=1e-4），单用例排除。
  --deselect "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_det_is_pm1"
)

# quantization_models = 上游 Step 3 的 PPU 版：tests/models/quantization/ 减 10 个
#   ignore + fp8 kv -k 排除。Run 34580922394 后实际会跑：
#   test_per_token_kv_cache.py 的 int8 用例（fp8 变体被 -k 排除）。
QUANT_MODELS_ARGS=(
  tests/models/quantization/
  # SM100+（Blackwell）
  --ignore=tests/models/quantization/test_nvfp4.py
  # AMD quark MXFP4
  --ignore=tests/models/quantization/test_mxfp4.py
  # fp8（SM≥8.9）
  --ignore=tests/models/quantization/test_fp8.py
  --ignore=tests/models/quantization/test_modelopt.py
  # 运行时 hf_hub_download（红区无外网）
  --ignore=tests/models/quantization/test_gguf.py
  # AMD gpt_oss（quark + lm_eval）
  --ignore=tests/models/quantization/test_gpt_oss.py
  # bitsandbytes：拆到 quantization_bitsandbytes step 用 -k 部分跑
  --ignore=tests/models/quantization/test_bitsandbytes.py
  # 模型未 stage（TechxGenus/gemma-1.1-2b-it-GPTQ；因 parametrize 按文件粒度，
  # model1 TheBloke/TinyLlama 虽通过但 model2 gemma 失败 → 整文件 ignore）
  --ignore=tests/models/quantization/test_gptq_marlin.py
  # Run 34580922394: 5/5 全挂 —— gemma4-moe AWQ 模型未 stage
  # （LocalEntryNotFoundError）+ InternVL2-2B EngineDeadError（c10::Error）。
  # 恢复条件：gemma4-moe AWQ 入库 + InternVL2 PPU 修复后 unignore。
  --ignore=tests/models/quantization/test_awq.py
  # Run 34580922394: 4/4 全挂 —— ValueError: Failed to find a kernel that can
  # implement the MXFP8 linear layer（PPU 无 MXFP8 kernel 实现）。
  --ignore=tests/models/quantization/test_mxfp8.py
  # Run 34580922394: fp8_per_token_head 1 例挂（fp8e4nv 需 SM8.9+，PPU 报
  # SM8.0）；同文件 int8 用例通过，故 -k 精确排除而非整文件 ignore。
  -k
  "not fp8_per_token_head"
)

# quantization_bitsandbytes = test_bitsandbytes.py 的部分跑：BNB 4bit/8bit，
#   -k 只选红区已 stage 模型的用例。未选中的用例用未 stage 模型
#   （Mistral-7B-Instruct-v0.3 / Llama-Guard-3-8B-INT8 / PrunaAI 等）。
#   Run 34580922394: test_load_pre_quant_4bit 的 PrunaAI 变体未 stage 秒挂
#   （and not PrunaAI 排除，poedator/opt-125m-bnb-4bit 变体保留）；
#   test_4bit_bnb_embedding（e5-mistral）缺 sentence_transformers，从 -k 移除。
QUANT_BNB_ARGS=(
  tests/models/quantization/test_bitsandbytes.py
  -k
  "(test_load_pre_quant_4bit and not PrunaAI) or test_4bit_bnb_moe or (test_load_4bit_bnb_model and opt) or (test_load_8bit_bnb_model and fbopt)"
)

# ------------------------------------------------------------------------------
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

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
    # ---- quantization_core: tests/quantization/ ----
    # test_auto_round.py（AutoRound INT4）
    # 主清单命中 name=Qwen2-0.5B-Instruct-int4-sym-AutoRound
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.0/"
        "Qwen2-0.5B-Instruct-int4-sym-AutoRound",
    # 主清单命中 name=Qwen2.5-0.5B-Instruct-int4-sym-inc
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/"
        "Qwen2.5-0.5B-Instruct-int4-sym-inc",
    # test_configs.py（GPTQ/AWQ config 解析）
    # 主清单 MISS，Aone alias 命中（/ppusw→/nas_aisw）
    "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v1.0/"
        "TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
    "TheBloke/Llama-2-7B-Chat-GPTQ":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/Llama-2-7B-Chat-GPTQ",
    "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ":
        "/nas_aisw/datasets/checkpoints/LLM/OpenHermes/v1.0/"
        "OpenHermes-2.5-Mistral-7B-AWQ",
    # test_experts_int8.py（INT8 experts 启动）
    # 主清单 MISS，Aone alias 命中
    "ai21labs/Jamba-tiny-random":
        "/nas_aisw/datasets/checkpoints/LLM/Jamba/v1.0/Jamba-tiny-random",
    "pfnet/plamo-2-1b":
        "/nas_aisw/datasets/checkpoints/LLM/plamo/v1.0/plamo-2-1b",
    # test_register_quantization_config.py（自定义量化配置 + 小 LLM）
    # 主清单命中 path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct
    # （清单 ms_name=unsloth/...，按 name 匹配；Aone /ppusw 侧是 v3.1，
    # 以主清单 v3.2 为准）
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",

    # ---- quantization_bitsandbytes: test_bitsandbytes.py（-k 选中的用例）----
    # test_load_pre_quant_4bit（主清单 MISS，Aone alias 命中）
    "poedator/opt-125m-bnb-4bit":
        "/nas_aisw/datasets/checkpoints/LLM/opt/v1.0/opt-125m-bnb-4bit",
    # test_4bit_bnb_moe
    "allenai/OLMoE-1B-7B-0125-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/OLMoE/v1.0/OLMoE-1B-7B-0125-Instruct",
    # test_load_4bit_bnb_model and opt（主清单命中 name=opt-125m）
    "facebook/opt-125m":
        "/nas_aisw/datasets/checkpoints/LLM/misc/v1.0/opt-125m",
    # test_load_8bit_bnb_model and fbopt
    "yec019/fbopt-350m-8bit":
        "/nas_aisw/datasets/checkpoints/LLM/fbopt/v1.0/fbopt-350m-8bit",
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

root = ET.Element("testsuites", name="vLLM PPU Quantization")
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

lines = ["### Quantization Test (PPU)", "",
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

if [ "${MODE}" = "single" ]; then
  # quantization 是 single-only：3 个 step 各限 1 卡（无 multi_gpu_test 用例）
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_core" 1 "${QUANT_CORE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_models" 1 "${QUANT_MODELS_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_bitsandbytes" 1 "${QUANT_BNB_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area quantization has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_core" 1 "${QUANT_CORE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_models" 1 "${QUANT_MODELS_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_bitsandbytes" 1 "${QUANT_BNB_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
