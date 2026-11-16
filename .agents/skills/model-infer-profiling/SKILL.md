---
name: model-infer-profiling
description: 昇腾 NPU 上 PyTorch 模型的性能分析数据采集技能。按目标推理入口的采集能力选择复用或注入，产出算子明细、统计和执行时间线，供后续性能分析使用。触发场景：采集 NPU 性能数据、profiling、性能分析、tensorboard、推理性能瓶颈定位、torch_npu.profiler 使用、profiler 解析失败或 JSON 截断。
---

# NPU Profiling 采集技能

本技能帮助你在华为昇腾 NPU 上用 `torch_npu.profiler` 正确采集 PyTorch 模型性能数据，产出与 Ascend 官方 TensorBoard/Profile 插件兼容的 `ASCEND_PROFILER_OUTPUT` 目录。

## 采集目标与路径

依据任务和已有上下文确定采集阶段、窗口和指标，使窗口覆盖目标负载；影响采集方案的未决事项交调用方确认。阶段选择见「关键 3」。单步 prefill 可用 `active=1`；多批次或分块 prefill 按目标范围确定实际采集步数。

目标推理循环已接入 profiler 时，优先复用并按需调整阶段、窗口和指标；未接入时按路径 B 注入。路径 A 给出 infer 仓框架的配置与约束，其他推理框架按其实际采集接口实施。

---

# 路径 A：使用框架内置采集（cann-recipes-infer）

## A.1 怎么开启

在 YAML 的 `model_config` 下把开关打开：

```yaml
model_config:
  enable_profiler: True   # 默认 False
```

然后照常 `bash executor/scripts/infer.sh --model <model> --yaml <yaml>` 跑。

## A.2 框架采集入口与默认配置

infer 仓框架由 `executor/core/engine/execution_engine.py` 调用 `executor/utils/profiler_context.py` 的 `ProfilerManager`，按阶段推进采集状态。

| 阶段 | 当前默认窗口 | 输出目录 |
|---|---|---|
| **prefill** | `active=1, skip_first=0` | `<res_path>/prof/prefill/` |
| **decode** | `active=3, skip_first=3` | `<res_path>/prof/decode/` |
| **mm_encode** | `active=1, skip_first=0` | `<res_path>/prof/mm_encode/` |

各阶段使用 `warmup=0, repeat=1`，默认指标为 `Level1 + PipeUtilization`。`<res_path>` 位于 `models/<model>/res/<date>/<case>/`，多卡按 rank 输出数据。实际采集覆盖由执行阶段和窗口决定。

## A.3 调整采集参数

当前 YAML 暴露 `enable_profiler` 总开关，窗口配置在 `executor/utils/profiler_context.py`：

- Decode 的 `DECODE_PROFILE_ACTIVE_STEPS` 控制记录步数，`DECODE_PROFILE_SKIP_FIRST` 控制跳过步数；它们也用于 `ProfilerManager.step()` 的 `timing_ready` 计时边界，调整时保持一致。
- Prefill / mm_encode 的窗口在 `set_status()` 中共用分支设置；按目标阶段的实际 forward 次数确定覆盖范围。
- 若只采某个阶段，保留 `ProfilerManager` 的状态推进：当前实现要求 Decode 之前已进入过采集阶段；替换采集对象时保持 `start/step/stop` 接口。

实际 Decode 步数需覆盖采集窗口；调整窗口时同步核对 `scheduler_config.max_new_tokens` 和实际结束条件。窗口完成规则见路径 B「关键 2」。

## A.4 验收产物

按下方「产物自检清单」核对本轮各目标阶段和 rank 的产物，并检查对应运行日志中的采集、解析错误。

---

# 路径 B：注入 torch_npu.profiler 采集代码

按以下配置与模板接入目标推理循环。

### 关键 1：按分析目标配置采集指标

分析 cube / vector / MTE 流水利用率时，沿用框架的常用配置：

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)
```

特殊指标、导出格式及版本支持，查询 cannbot-knowledge 中对应版本的 profiling 配置与产物说明。

### 关键 2：执行步数覆盖采集窗口

`skip_first` 跳过起始步，随后按 `wait → warmup → active` 推进；每次实际推理后调用 `prof.step()`。以 torch_npu 2.8 的 schedule、单周期 `repeat=1` 为例，完成 `skip_first + wait + warmup + active` 次推理及对应的 `step()` 后，窗口结束并触发导出。

```python
WARMUP, ACTIVE = 2, 30
with create_profiler("./prof", warmup=WARMUP, active=ACTIVE) as prof:
    for step in range(WARMUP + ACTIVE):  # wait=0, skip_first=0, repeat=1
        run_one_step()
        prof.step()
```

注入模板的四种预置模式使用 `warmup=0`；修改 schedule 时相应调整真实执行步数。

### 关键 3：按目标选择阶段与预热范围

采集稳态性能时，在窗口外完成目标执行路径的图编译 / JIT 和预热。

prefill 本身是真实负载（kernel 分布、attention layout、MC2/all-to-all 调度都和 decode 不同），采不采看你想分析什么：

| 想分析 | 采哪些阶段 | 备注 |
|---|---|---|
| decode TPOT、decode kernel 瓶颈 | **只采 decode** | 最常见；prefill 离群步会拉偏统计 |
| prefill TTFT、长序列 prefill kernel | **只采 prefill** | |
| 一条时间线对照 prefill vs decode | **一次采两者** | `trace_view` 清晰展示 prefill 单步 + 后续 decode |
| 分开对照、不混时间线 | **两次采，两个 save_path** | 每个 phase 一个 `ASCEND_PROFILER_OUTPUT` |

## 标准配置模板（注入）

四种采集模式及 `create_profiler()` 的完整模板见 `scripts/profile_template.py`，在仓库根用 `python3 .agents/skills/model-infer-profiling/scripts/profile_template.py` 运行。使用前按目标负载适配：

1. 把 `build_model()` / `run_prefill()` / `warm_one_decode()` / `run_one_decode()` 中的占位 `Linear` 负载换成真实的模型加载与单步推理
2. `PHASE` 设为 `"prefill"|"decode"|"both"|"separate"`，与上面确认的采集范围一致
3. `DECODE_STEPS` 设为实际要采的 decode 步数
4. `SAVE_PATH` 设为本轮采集输出目录

产物落在 `SAVE_PATH` 下，按「产物自检清单」验收。

---

## 产物自检清单（两条路径通用）

核对本轮目标阶段和 rank 对应的 `ASCEND_PROFILER_OUTPUT/`：

| 检查项 | 验收依据 |
|---|---|
| 阶段 / rank 覆盖 | 产物对应本次采集范围，窗口内有目标负载 |
| `kernel_details.csv` | 有算子记录，包含下游分析所需字段；性能拆解使用 `Step Id`、`Name`、`Duration(us)`、`Start Time(us)`、`Stream ID`，利用率分析再核对选定指标 |
| `op_statistic.csv` / `api_statistic.csv` | 按采集配置核对算子与 host API 统计产物 |
| `trace_view.json` | 能解析为 JSON，并能定位目标阶段和执行步骤 |
| 采集窗口 | 结合 step 标记与运行日志核对实际覆盖步数 |
| 通信分析产物 | 需要分析通信时，确认窗口内捕获了目标 HCCL 操作及对应统计 |

采集或解析异常时，按框架、版本和实际报错查询 cannbot-knowledge 中的采集排障、原始数据重解析及输出目录相关经验。

## 交付与查看结果

交付本轮各阶段 / rank 的产物路径、采集窗口和指标配置。`kernel_details.csv` 用于算子明细，`op_statistic.csv` 用于算子汇总，`trace_view.json` 用于时间线；拆解到模型结构时交给 `model-infer-perf-breakdown`。

## 参考文档

- [采集模板](scripts/profile_template.py) — 路径 B 的完整模板，包含四种采集模式
- [torch_npu Profiler 文档](https://www.hiascend.com/document)
