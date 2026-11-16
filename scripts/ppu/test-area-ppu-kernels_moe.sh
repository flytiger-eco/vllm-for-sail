#!/bin/bash
# 独立运行 MoE kernels，避免其他 kernels 步骤挤占其完整超时预算。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PPU_KERNELS_ONLY="moe"
source "${SCRIPT_DIR}/test-area-ppu-kernels.sh"
