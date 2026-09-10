#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-spec-decode.sh — PPU Spec Decode 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-spec-decode.yml（K8s worker Pod 内，
#         cwd = /workspace/source）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/spec_decode.yaml 的
# 迁移快照（见下方 SD_*_ARGS，调整用例直接改这里）。
# 模型走 /nas_aisw 预置卷（action 默认挂载 + HF_HUB_CACHE）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — spec_decode area 仅 single 段（4 step，
#               各用一个 -k filter），无 multi 段（上游 buildkite 4 step 全 single）
#
# ⚠️ FIRST-DEPLOY RED EXPECTED（首跑期望全 fail）：
#   本 area 第一版依赖的多数模型（Llama-4-Scout gated/heavy、DeepSeek 全量超大、
#   若干 eagle/mtp draft 系列）尚未确认 stage 进 /nas_aisw；离线环境下这些用例会
#   以 LocalEntryNotFoundError 失败。接受 RED baseline，作为同事 stage 完成后的
#   回归 verifier（stage 完 GREEN = 真 PPU code OK；RED→GREEN 翻转即 catch 真回归）。
#   恢复条件：同事把 MODEL_MAP 中标「待入库」的模型 stage 进 NAS 后转 GREEN。
#   故本 workflow 为 dispatch-only（见 yml），暂不挂 PR 链 / nightly。
#
# 机制移植自 aone_ci/scripts/test_area_ppu_spec_decode.sh（该文件
# AUTO-GENERATED 不可手改，故在此复刻）：
#   - single: 4 个独立 step，各单进程单卡（nproc=1），各用同一测试目录 +
#     不同 -k filter（eagle / speculators+mtp / ngram+suffix / draft_model）；
#     tp>1 case 在 1 卡下自动 skip（对齐 Aone 1-PPU pod 语义）
#   - test_lora_with_spec_decode.py 全程 --ignore（Path E lora 暂停期间不 cover）
#   - junit:  每 step 落 xml，EXIT trap 合并到 test-results/test.xml，
#     pytest 崩溃时也要补 error case（不能让 CI 信号失真）
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|all) ;;
  multi) echo "[mode] spec_decode area has no multi-mode steps (single-only)" >&2 ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# workflow_dispatch 的 pytest_args 透传：按空白切分后追加到每个 step 的
# pytest 命令尾部（如 `-k test_foo -x`），排障时缩小范围而不必改脚本。
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-spec-decode-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area 特有依赖：无。
# ------------------------------------------------------------------------------
# spec_decode 用例不需要 ray（single-only，无 distributed_executor_backend="ray"）。
# pytest-asyncio / tblib / pytest-shard 由共享 ppu_install_dependency.sh 兜底安装。
# suffix decoding 需 arctic-inference（仅 source dist，CI pod 缺 build toolchain 无法
# 编译），故 -k filter 里已排除 suffix_decoding_acceptance / speculative_config1，
# 此处不尝试安装；恢复路径 = image rebake 预装 arctic-inference。

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/spec_decode.yaml single 段）
# ------------------------------------------------------------------------------
# 对位上游 .buildkite/test_areas/spec_decode.yaml 的 4 个 step（全跑
# tests/v1/e2e/spec_decode/，各用一个 -k filter）。PPU 合并为 4 single step，
# 每 step 单卡（nproc=1）；test_lora_with_spec_decode.py 全程 --ignore（Path E
# lora 暂停期间不 cover，extras F12 follow-up）。tp>1 case 在 1 卡下自动 skip。
#
# 测试文件清单（tests/v1/e2e/spec_decode/，当前分支均存在，2026-09-09 核对）：
#   - test_spec_decode.py        主 spec_decode（eagle/mtp/ngram/draft）
#   - test_async_spec_decode.py  async spec decode
#   - test_lora_with_spec_decode.py  lora+spec（--ignore，Path E 暂停）

# Step 1：eagle_correctness（排除 _heavy model-dep：llama3/llama4/llama4_mm 未 staged）
SD_EAGLE_ARGS=(
  tests/v1/e2e/spec_decode
  --ignore=tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py
  -k
  "eagle_correctness and not _heavy"
)

# Step 2：speculators or mtp_correctness
#   排除 qwen3_eagle3_speculator / llama3_eagle3_speculator（2026-05-28 / 2026-06-01
#   smoke MR follow-up）：test_speculators_model_integration 断言 draft_model ==
#   'RedHatAI/...speculator.eagle3' 字符串等，但 vLLM alias redirect 会把它重写成
#   /nas_aisw 本地路径 → 字符串不等 → AssertionError。
#   恢复路径：上游 PR 把 test 改成比较 basename 而非完整路径。
SD_SPECULATORS_MTP_ARGS=(
  tests/v1/e2e/spec_decode
  --ignore=tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py
  -k
  "(speculators or mtp_correctness) and not qwen3_eagle3_speculator and not llama3_eagle3_speculator"
)

# Step 3：ngram or suffix
#   排除 suffix_decoding_acceptance / speculative_config1：suffix decoding 需
#   arctic-inference（只有 source dist，需 nanobind+cmake 编译），CI PPU pod 缺
#   build toolchain 编译失败（用户 dev pod 可）。恢复路径：image rebake 预装
#   arctic-inference。
SD_NGRAM_SUFFIX_ARGS=(
  tests/v1/e2e/spec_decode
  --ignore=tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py
  -k
  "(ngram or suffix) and not suffix_decoding_acceptance and not speculative_config1"
)

# Step 4：draft_model + no_sync + batch_inference
#   排除 _tensor_parallelism / _engine_args / _no_sync_with_spec_decode（TP-arg
#   validation 类 + no_sync 的 Llama/DeepSeek model-dep；Phase F+ F13 调查 PPU
#   device detection 路径问题）。
SD_DRAFT_MODEL_ARGS=(
  tests/v1/e2e/spec_decode
  --ignore=tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py
  -k
  "(draft_model or no_sync or batch_inference) and not _tensor_parallelism and not _engine_args and not _no_sync_with_spec_decode"
)

# ------------------------------------------------------------------------------
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# 注意：禁止 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（上游
# TP 场景的 CUDA VMM workaround）——PPU 兼容层疑不支持 VMM API，是虚假
# OOM 头号嫌疑：96 GiB free 时 20 MiB 分配失败且 free>total 统计错乱
# （本 area single uni EngineCore 与 lora multi TP rank1 两案例均发生在
# 设此 env 的 GHA 环境；Aone 侧从不设它且全绿，DEC-0013 当时明确决定
# 不引入）。删除重跑验证；若虚假 OOM 仍现再查 PPU SDK/驱动

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
# /ppusw 同构路径标注"待确认"。路径不存在时 WARN 并跳过（该模型的用例
# 会失败，日志里可见原因）。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # === 已 alias / 红区清单命中（symlink 生效；实际是否在 NAS 上以 MISS 日志为准）===
    # 路径来源标注：(ckpt)=scripts/ppu/model_alises/checkpoints_cleaned.json 主清单
    #             (aone)=aone_ci/scripts/ppu_model_aliases.json（/ppusw/→/nas_aisw/）
    #
    # meta-llama/Llama-3.1-8B-Instruct：extras 明示复用 Meta-Llama-3.1-8B-Instruct
    # mirror（2026-05-28 aliased，smoke MR #87 follow-up）。(ckpt/aone 一致)
    "meta-llama/Llama-3.1-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Meta/v1.0/Meta-Llama-3.1-8B-Instruct",
    # eagle_correctness / draft_model 用；base + eagle draft 系列
    "meta-llama/Llama-3.2-1B-Instruct":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
    "yuhuili/EAGLE-LLaMA3.1-Instruct-8B":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-LLaMA3.1-Instruct-8B",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE3/v3/EAGLE3-LLaMA3.1-Instruct-8B",
    "nm-testing/Llama3_2_1B_speculator.eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v1.0/"
        "Llama3_2_1B_speculator.eagle3",
    # speculators / mtp / ngram 用；Qwen3 + eagle3 speculator 系列
    "Qwen/Qwen3-0.6B":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-FP8":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B-FP8",
    "Qwen/Qwen3-1.7B":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-FP8":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B-FP8",
    "Qwen/Qwen3-8B":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-8B",
    "AngelSlim/Qwen3-1.7B_eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-1.7B_eagle3",
    "AngelSlim/Qwen3-8B_eagle3":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-8B_eagle3",
    # 注：qwen3/llama3 eagle3 speculator 断言用例已在 -k 里 deselect（alias-rewrite
    # AssertionError），symlink 保留供其他 speculators 用例引用。
    "RedHatAI/Qwen3-8B-speculator.eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3-8B-speculator.eagle3",
    "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/"
        "Llama-3.1-8B-Instruct-speculator.eagle3",
    "amd/PARD-Qwen3-0.6B":  # (aone) ngram/pard
        "/nas_aisw/datasets/checkpoints/LLM/PARD/v1.0/PARD-Qwen3-0.6B",
    "XiaomiMiMo/MiMo-7B-Base":  # (aone) mtp
        "/nas_aisw/datasets/checkpoints/LLM/MiMo/v1.0/MiMo-7B-Base",
    # deepseek MTP / no_sync（小型 random 变体，非全量 DeepSeek）
    "ZixiQi/DeepSeek-V3-4layers-MTP-FP8":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V3/DeepSeek-V3-4layers-MTP-FP8",
    "eagle618/deepseek-v3-random":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/deepseek/v3/deepseek-v3-random",
    "eagle618/eagle-deepseek-v3-random":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/eagle/v3/eagle-deepseek-v3-random",
    # VL eagle3（multimodal draft）
    "Qwen/Qwen2.5-VL-7B-Instruct":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-VL-8B-Instruct",
    "Rayzl/qwen2.5-vl-7b-eagle3-sgl":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen2.5/v1.0/qwen2.5-vl-7b-eagle3-sgl",
    "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/"
        "EAGLE-Llama-4-Scout-17B-16E-Instruct",

    # === 待入库（两个红区来源均 MISS，不猜路径；首跑这些用例 fail 属预期 RED）===
    # extras 明示未 staged（gated / 超大）：
    #   meta-llama/Llama-4-Scout-17B-16E-Instruct  gated + ~30 GB
    #     （aone 有 /llama/v4/ 别名但未确认 stage，_heavy 用例已 -k 排除）
    #   deepseek-ai/DeepSeek-R1                     超大，两源 MISS
    #   deepseek-ai/DeepSeek-V3                     超大（no_sync 全量，已 -k 排除）
    # 测试引用但两源 MISS：
    #   taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3
    #   premjatin/qwen-linear-algebra-coder
    #   likaixin/InstructCoder（dataset，非 model；conftest redirect，见 extras）
    # 恢复条件：同事 stage 上述模型进 /nas_aisw 后在此补路径（Phase F+ F10/F11）。
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

root = ET.Element("testsuites", name="vLLM PPU Spec Decode (GHA)")
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

# ---- step summary：分 shard 统计表（markdown）。合并 test.xml 的
# testsuite name 已被改写为 label（丢失 shard 维度），故此处从原始
# shard xml 提取。宿主 workflow 把本文件 cat 进 GITHUB_STEP_SUMMARY，
# 在 run 的 Summary 页直接渲染（容器内拿不到该 env，需 workflow 接力）
SUMMARY = os.path.join(os.path.dirname(OUT), "summary.md")
COLS = ("tests", "failures", "errors", "skipped", "time")

def _stats(path):
    # junit 根节点 pytest 新旧版可能为 <testsuites> 或 <testsuite>
    root_ = ET.parse(path).getroot()
    suites = [root_] if root_.tag == "testsuite" else list(root_.iter("testsuite"))
    agg = dict.fromkeys(COLS, 0.0)
    for s in suites:
        for k in COLS:
            agg[k] += float(s.get(k) or 0)
    return agg

lines = ["### Spec Decode Test (PPU)", "",
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
  # worker Pod 与编排 runner 容器共享同一 NAS（/mnt/wl_nas ↔ /wl_nas），宿主
  # workflow 的 Publish test summary 步骤读取并 cat 进 GITHUB_STEP_SUMMARY。
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
      # 注：不可写成 `[ $rc -ne 0 ] && rc_total=1` —— 条件为假时整个
      # 表达式返回 1，若它是函数/分支的最后一条命令，函数返回码变成 1，
      # 顶层 set -e 会在函数调用处杀掉脚本（测试全过反而 exit 1 的元凶）
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

# spec_decode area 仅 single 段（4 step，各单进程单卡；对齐 aone
# test_area_ppu_spec_decode.sh single 段的 4 个 _run_pytest 调用）。
# 限 1 卡（CUDA_VISIBLE_DEVICES=0）使 tp>1 case 自动 skip，语义对齐 1-PPU pod。
if [ "${MODE}" = "multi" ]; then
  echo "[run] spec_decode area has no multi-mode steps configured — nothing to run" >&2
else  # single | all
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_eagle" 1 "${SD_EAGLE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_speculators_mtp" 1 "${SD_SPECULATORS_MTP_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_ngram_suffix" 1 "${SD_NGRAM_SUFFIX_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_draft_model" 1 "${SD_DRAFT_MODEL_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
