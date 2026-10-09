# PPU CI 排除项台账

PPU 上跑不过的用例目前用 `--ignore` / `--deselect` / `-k` 三种方式排除在
`scripts/ppu/test-area-ppu-<area>.sh` 里。**排除项是覆盖率的漏洞**，本文件的作用
是让每个漏洞都有人、有期限、有恢复条件，而不是排掉就忘。

规则：

1. 新增排除项必须在脚本里写明原因，并在下方对应表格补一行（负责人 + 复查时间）。
2. 排除项引用的路径必须真实存在——`PPU CI Selfcheck` workflow 跑
   `scripts/ppu/check-exclusions.py` 校验。**失效的 `--ignore` 会被 pytest 静默吞掉**，
   Aone 侧已发生过（entrypoints 4 条、kernels 1 条排除项指向已删除文件，形同虚设）。
3. 排除项只允许因「平台/环境限制」或「已定位的上游缺陷」存在。「跑不过、原因不明」
   属临时状态，必须挂到下方待办表并给出复查时间。

---

## 1. 规模现状（2026-08-31 实测统计）

| Area | `--ignore` | `--deselect` | `-k` 排除 |
| --- | ---: | ---: | ---: |
| kernels | 56 | 3 | 0 |
| lora | 15 | 3 | 1 |
| engine | 4 | 17 | 0 |
| model-executor | 12 | 0 | 0 |
| entrypoints | 8 | 0 | 0 |
| models-basic | 0 | 8 | 0 |
| samplers | 2 | 0 | 0 |
| basic-correctness | 0 | 2 | 0 |
| models-language | 0 | 2 | 0 |
| entrypoints-llm | 0 | 1 | 0 |
| attention | 0 | 0 | 1 |

统计命令（可复现）：

```bash
for f in scripts/ppu/test-area-ppu-*.sh; do
  printf "%-20s ignore=%-3s deselect=%s\n" "$(basename "$f")" \
    "$(grep -c -- '--ignore' "$f")" "$(grep -c -- '--deselect' "$f")"
done
```

kernels 的 56 条与 engine 的 17 条是两处主要覆盖缺口，也是最该优先收敛的地方。

---

## 2. 待办：原因未定位的排除项

来源为 [批次 1 首跑总结报告](first-run-report-batch1.md)（2026-08-27）。这些是
「跑不过、原因未查明」的临时排除，**必须逐条收敛**，不是可接受的稳定状态。

| # | 范围 | 规模 | 现状 | 负责人 | 复查时间 |
| --- | --- | --- | --- | --- | --- |
| E-1 | `engine`：`tests/v1/engine/` abort 语义 | 6 用例 | deselect，root cause 待查 | 待指派 | 待定 |
| E-2 | `engine`：EngineCore 基础（疑 OOM/调度） | 4 用例 | deselect，root cause 待查 | 待指派 | 待定 |
| E-3 | `engine`：encoder 零 kv-cache 实例 | 6 用例 | deselect，root cause 待查 | 待指派 | 待定 |
| E-4 | `engine`：preprocess 错误处理 | 1 用例 | deselect，root cause 待查 | 待指派 | 待定 |
| K-1 | `kernels`：3 个测试文件整文件排除 | 3 文件 | ignore，原因均待查 | 待指派 | 待定 |
| M-1 | `models-basic`：`test_transformers.py::test_models[hmellor/Ilama-3.2-1B-auto]` | 1 用例 | vLLM vs HF 对拍不一致；模型本体是否就位未确认 | 待指派 | 待定 |
| M-2 | `models-basic`：registry import 链 3 条 | 3 用例 | 报错原因为代码级推断，未经日志核实 | 待指派 | 待定 |
| A-1 | `attention`：`test_gdn_metadata_builder.py` 2 用例 | 2 用例 | Aone 快照既有排除，root cause TBD（F2 跟踪） | 待指派 | 待定 |

> 复核方式：下载对应 run 的 `ppu-<area>-test-results` artifact，读 `test.xml` 里的
> failure message。首跑报告因匿名访问 Actions 日志需登录（403/401）未能直接核实精确
> 报错，这一步仍待补。

---

## 3. 待办：与上游选集的偏差

不是「跑不过」，而是「选集抄漏/抄多」，会造成静默的覆盖差异。

| # | 位置 | 偏差 | 处理 |
| --- | --- | --- | --- |
| D-1 | `lora`：`tests/lora/test_qwen35_densemodel_lora.py` | 上游 `LoRA %N` step 排除（归到 4 卡 TP step），PPU 旧快照漏抄 | 已于本次补上 `--ignore` |
| D-2 | `lora`：`tests/lora/test_minicpmv_tp.py` | 上游 v0.23.0 的 ignore 列表**不含**本文件（即上游在分片 step 里跑），PPU 侧沿用旧快照继续排除 | 多余排除，需确认是否恢复 |
| D-3 | `lora`：多卡段只跑 `test_qwen3_with_multi_loras.py` | 上游 `LoRA TP (Distributed)` 还包含 chatglm3 / llama / olmoe / gptoss / qwen35_densemodel 共 6 个文件 | 需评估 PPU 4 卡可行性后补齐 |
| D-4 | 全体 area | 上游 `.buildkite/test_areas/<area>.yaml` 变更后，PPU 侧脚本无任何同步机制（Aone 侧靠 `aone_ci/pipeline_generator` 生成） | 需要漂移检测；见 `area-migration.md` |

D-4 是结构性问题：只要选集是手抄快照，上游 rebase 就会持续产生 D-1/D-2 这类偏差。
`check-exclusions.py` 只能查出「路径已不存在」，查不出「上游改了排除列表」。

---

## 4. 可接受的稳定排除项

以下类别不需要逐条挂待办，但恢复条件要写在脚本注释里：

- **硬件能力限制**：OAM-810E 为 SM 8.0，fp8e4m3fn kernel 需 SM ≥ 8.9
  （如 `test_punica_ops_fp8.py`）。恢复条件：硬件换代。
- **模型未 stage 到 NAS**：离线环境下 `snapshot_download` 必失败
  （如 `test_moe_lora_ep_load.py`、`test_default_mm_loras.py`）。
  恢复条件：模型/adapter 落到 `/nas_aisw` 后删除对应行并补 `MODEL_MAP`。
- **依赖版本不匹配**：镜像内 transformers 版本不含目标子模块。
  恢复条件：镜像升级。
- **PPU 精度差异**：已确认为平台数值差异且已反馈 PPU 团队的用例。
  恢复条件：PPU 侧修复。

---

## 5. 复查节奏

- 每次定时跑（工作日/周末档）后过一遍新增 red，判定「进台账」还是「当次修复」。
- 每两周 review 一次本文件的待办表，收敛掉已定位项。
- 试用期结束（快速档转 required）前，第 2 节的待办应至少完成 root cause 定位，
  否则门禁的信噪比无法评估。
