#!/bin/bash
# ==============================================================================
# scripts/ci/ppu/ppu_install_dependency.sh — PPU CI 依赖安装（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/pr-test-ppu.yml，在 PPU 基础镜像容器内执行：
#   pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-...-vllm0.23.0-py312
#
# 对位：
#   - Aone CI 侧 wheel/依赖安装分散在 .aoneci/build-wheel-ppu.yaml +
#     aone_ci/scripts/test_area_ppu_lora.sh 的 [setup] 段
#   - sglang-for-sail scripts/ci/ppu/ppu_install_dependency.sh 的组织方式
#
# 核心原则（sglang 踩坑经验，见其脚本注释）：镜像预装的 PPU 栈（带
# +v0.1.0.ppu2.1.1 local version 的 torch/vllm/triton/flashinfer 等）一律
# 不动——blind `--force-reinstall` 会用同版本号的另一种构建覆盖镜像里的
# 版本，甚至降级。只安装镜像里真正缺失的包。
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PIP_INSTALL="python3 -m pip install --no-cache-dir"
# SAIL SDK v2.1.1 PyPI source
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"

# flytiger artifactory 凭证（GitHub Secrets 注入；镜像公开时可无）
if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

# 容器内以 root 运行 git 需要（vllm 版本号由 setuptools-scm 从 git 推导时也依赖它）
git config --global --add safe.directory "${REPO_ROOT}"

# ------------------------------------------------------------------------------
# [diag] 环境盘点：先看清镜像里已有什么，再决定装什么
# ------------------------------------------------------------------------------
echo "========== [diag] preinstalled stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
python3 -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm NOT importable"
python3 -m pip list 2>/dev/null | grep -iE "^(vllm|torch|triton|flashinfer|deep.ep|deep.gemm|tilelang|flash.mla|flash.attn|transformers) " || true

# ------------------------------------------------------------------------------
# [deps] PPU 依赖：镜像预装优先，缺失才从 flytiger PyPI 补
# ------------------------------------------------------------------------------
# 版本来自 SAIL SDK v2.1.1 用户指南的 PyPI 安装命令。注意：如果镜像预装的
# 版本与下表不同（通常更新），以镜像为准，不降级。
_ensure_pip_pkg() {
    local pkg="$1" ver="$2" extra_args="${3:-}"
    local installed
    installed="$(python3 -c "import importlib.metadata as m; print(m.version('${pkg}'))" 2>/dev/null || true)"
    if [[ -n "${installed}" ]]; then
        echo "[deps] ${pkg} already installed: ${installed} — keep image build"
        return 0
    fi
    echo "[deps] installing ${pkg}==${ver}"
    # shellcheck disable=SC2086  # extra_args is intentionally word-split
    ${PIP_INSTALL} "${pkg}==${ver}" ${extra_args} -i "${PPU_PIP_INDEX}"
}

# vllm 本体：不带 --no-deps（需解析依赖树）；其余 PPU 组件按指南 --no-deps
_ensure_pip_pkg "vllm" "0.23.0"
_ensure_pip_pkg "flashinfer_python" "0.6.8.post1" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_ep" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_gemm" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "tilelang" "0.1.8" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_mla" "2.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn_3" "2.8.2" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn" "2.7.4.post1" "--no-deps --force-reinstall"

# ------------------------------------------------------------------------------
# [deps] pytest 测试依赖（对位 test_area_ppu_lora.sh 的 [setup] 段）
# ------------------------------------------------------------------------------
# pytest-asyncio: tests/conftest.py asyncio fixture 需要
# tblib: pytest 跨进程 traceback 序列化（spawn worker 报错时）
# pytest-shard: single 模式 --shard-id/--num-shards 需要
# pyyaml: 备用通用依赖（用例选集已内联在 run_lora_tests.sh，不解析 yaml）
echo "========== [deps] pytest toolchain =========="
${PIP_INSTALL} pytest pytest-asyncio tblib pytest-shard pyyaml -i "${PPU_PIP_INDEX}"

echo "========== [verify] final stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
python3 -c "import vllm; print('vllm', vllm.__version__)"
echo "[deps] done"
