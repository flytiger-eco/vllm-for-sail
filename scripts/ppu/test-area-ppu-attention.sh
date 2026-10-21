#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-attention.sh - PPU Attention test runner (GitHub Actions)
# 由 main 上的 .github/workflows/ppu-attention-pr.yml 调用。
# 在 PPU Pod 内执行，工作目录为 /workspace/source。
# Environment variables: TEST_MODE=all (default)|single (no multi tests in this area)
# Models: no real models; only test_indexer_deepseek_v4_slot_mapping.py needs to
#   parse the Llama-3-8B HF config, which the [stub] section below provides locally.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# pytest_args passthrough from workflow_call: split on whitespace and appended
# to every step's pytest command (e.g. `-k test_foo -x`) to narrow the scope
# while debugging without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-attention-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] Test selection (snapshot of the single stage in
# legacy_ci/ppu_extras/attention.yaml)
# ------------------------------------------------------------------------------
# attention_default = merge of the two upstream steps "V1 attention (H100)" and
# "(B200)":
#   -k "not _correctness": excludes every correctness test that needs a real
#   model (models are not staged in the red zone, so they always fail offline).
#   Restore once the models are staged to /nas_aisw: drop this -k and extend
#   MODEL_MAP. Note that -k is a substring match: `not _backend_correctness`
#   does not cover `test_sparse_backend_decode_correctness`, hence
#   `not _correctness`.
#   Also excludes 2 tests in test_gdn_metadata_builder.py (root cause TBD,
#   tracked as F2).
ATTN_SINGLE_ARGS=(
  tests/v1/attention/
  -k
  'not _correctness and not test_gdn_build_classification and not test_has_initial_state_after_reclassification'
)

# multi: none (tests/v1/attention/ has no @multi_gpu_test)

# ------------------------------------------------------------------------------
# [env] Offline + runtime configuration
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# Offline by default; PPU_TEST_ONLINE=1 lifts it (this area downloads nothing;
# kept only for consistent behavior)
if [[ "${PPU_TEST_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# HF cache: prefer the value injected by the workflow; otherwise (local
# debugging) probe common paths
if [ -z "${HF_HUB_CACHE:-}" ]; then
  for _cand in /nas_aisw/datasets/hf_cache/hub "$HOME/.cache/huggingface/hub"; do
    if [ -d "${_cand}" ]; then
      export HF_HUB_CACHE="${_cand}"
      break
    fi
  done
fi
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"

# ------------------------------------------------------------------------------
# [stub] Offline model stubs for config-only tests (VLLM_MODEL_REDIRECT_PATH)
# ------------------------------------------------------------------------------
# test_indexer_deepseek_v4_slot_mapping.py and others build
# ModelConfig(model="meta-llama/Meta-Llama-3-8B") via create_vllm_config() and
# only parse the HF config; but that gated repo is not in the NAS cache, so
# HF_HUB_OFFLINE=1 raises ValidationError right away.
# The official VLLM_MODEL_REDIRECT_PATH redirects the repo id to a local stub
# (config.json only, with the real parameters) without touching upstream test
# files (zero-diff).
# Restore once Meta-Llama-3-8B is staged into the /nas_aisw HF cache: drop this
# section.
STUB_ROOT="/tmp/ppu-attention-stubs"
STUB_MODEL_DIR="${STUB_ROOT}/Meta-Llama-3-8B"
mkdir -p "${STUB_MODEL_DIR}"
# Public Meta-Llama-3-8B config (for config parsing only; no weights/tokenizer)
cat > "${STUB_MODEL_DIR}/config.json" <<'STUB_CONFIG_EOF'
{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "use_cache": true,
  "vocab_size": 128256
}
STUB_CONFIG_EOF
printf '{"meta-llama/Meta-Llama-3-8B": "%s"}\n' "${STUB_MODEL_DIR}" \
  > "${STUB_ROOT}/model_redirect.json"
export VLLM_MODEL_REDIRECT_PATH="${STUB_ROOT}/model_redirect.json"
echo "[stub] VLLM_MODEL_REDIRECT_PATH -> meta-llama/Meta-Llama-3-8B = ${STUB_MODEL_DIR}"

# PPU SDK: Triton/Inductor compilation needs cuda.h + ptxas + libcuda
# (without them torch.compile tests fail with BackendCompilerFailed)
PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# ------------------------------------------------------------------------------
# [junit] Merge infrastructure (including a fallback for pytest crashes)
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329  # only called via `trap _emit_junit EXIT` below
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

root = ET.Element("testsuites", name="vLLM PPU V1 Attention (GHA)")
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

# ---- step summary (markdown table): the merged test.xml already renamed each
# testsuite to its label (losing the shard dimension), so stats come from the
# raw shard xml files; the host workflow cats this file into
# GITHUB_STEP_SUMMARY (that env is not available in the container, so the
# workflow has to relay it)
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

lines = ["### Attention Test (PPU)", "",
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
  # pytest output of sharded steps goes to TMP_JUNIT; collect it into the
  # artifact as well for debugging
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
  # The container runs as root; artifacts must be readable by the runner user
  # (upload-artifact)
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
  # [k8s-summary] Return the per-unit stats table via NAS: RESULTS_DIR lives in
  # the Pod and is lost once the Pod is reclaimed.
  if [ -n "${PPU_SUMMARY_NAS_DIR:-}" ] && [ -f "${RESULTS_DIR}/summary.md" ]; then
    if mkdir -p "${PPU_SUMMARY_NAS_DIR}" 2>/dev/null && \
       cp -f "${RESULTS_DIR}/summary.md" "${PPU_SUMMARY_NAS_DIR}/summary.md"; then
      echo "[k8s-summary] summary.md -> ${PPU_SUMMARY_NAS_DIR}/summary.md"
    else
      echo "[k8s-summary] WARN: failed to return summary via NAS (${PPU_SUMMARY_NAS_DIR} not writable?)"
    fi
  fi
}
trap _emit_junit EXIT

# ------------------------------------------------------------------------------
# [run] Single-step runner: shards>1 -> concurrent shards; shards==1 -> single process
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
  CUDA_VISIBLE_DEVICES=0 _run_step "attention_default" 1 "${ATTN_SINGLE_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area attention has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "attention_default" 1 "${ATTN_SINGLE_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] Aggregate exit code (script exit code = aggregate rc, so the CI
# signal stays accurate)
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
