---
name: model-infer-perf-breakdown
description: 昇腾 NPU 性能数据拆解技能。把 kernel_details.csv 按模型结构切成 component 实例，再按 cluster 规则对 component 内算子分桶，产出 wall_ms / bubble_ms 中位数与异常 layer 的单页 HTML。触发场景：分析 NPU prof、拆解大模型性能、定位抖动 layer。
---

# NPU 性能数据拆解技能

输入一份算子 CSV（可附原始 trace），结合模型实现与任务上下文，先把 op 序列切成 component 实例，再按 cluster 规则统计每类计算的 wall_ms + bubble_ms 中位数 / 抖动 / 异常 layer，输出一页可读的 HTML。内部指标字段用 `*_ms`，HTML 展示按量级自动使用 `µs` 或 `ms`。

---

依据任务上下文、模型实现和 prof 确定结构、样本与分桶；影响结果且无法消解的疑问交调用方确认。输出分析结论、证据和优化候选，供调用方安排后续验证与改造。

## 工作流

```
Step 0: 确定 prof 路径 + 模型脚本路径  →  <run_dir>/inputs.json
Step 1: kernel_details.csv  →  raw_ops.json (单 step 提取)
Step 1.5: (可选,默认不开) 调用 operator-theoretical-perf → operator_analysis.csv → 合并理论列到 raw_ops_details.json
Step 2: Phase 0a 识别模型结构 → structure_spec.json
        Phase 0b 确定每个 component 的 stream sample 范围
                → sample_ack.json
        Phase 1  跑 detect_structure、审查告警 → structure_draft.json (component 实例表)
Step 3: 按模型语义确定 cluster 规则  →  network_spec.json
        → index.html（含 theory median 列）
Step 4: 按 component 切 csv / trace  →  splits/ (单层下钻)
Step 5: 洞察归纳（agent 语义判断 + 脚本统计） → insights/
```

每个 Step 各自的脚本 docstring 里有完整 schema 与算法描述；本文档只列契约与 CLI。

---

## 工作目录约定

本次产物写到 `<run_dir>`，沿用调用方指定的目录，未指定时默认 `./perf_breakdown/<label>/`；`<label>` 标识本次分析。

已有产物时先查看 `HISTORY.md` 和相关规则，优先复用，仅重跑受输入或规则变化影响的阶段。新 prof 的小改 / 中改 / 大改判断及同一份 prof 的复用范围见 `references/reuse_policy.md`。

---

## 反模式

**禁止修改用户提供的任何原始文件**（prof / modeling 源码 / config 等）；产物只写到 `<run_dir>/`。

---

## 输入

| 物料 | 用途 |
|------|------|
| 算子 CSV | Step 1 提取目标 step；Step 4 切片 |
| 原始 trace（可选） | Step 4 提供时间线上下文；未提供时从 raw_ops 合成 |
| 模型脚本路径（`modeling*.py` 及相关 module） | 识别结构、核对 sample 和 cluster 语义 |
| phase、rank 与 `<run_dir>` | 确定分析对象和产物位置 |

从采集产物中定位 `kernel_csv` / `trace_json`，CSV 字段须符合 `references/kernel_data_guide.md` 的输入契约。采集目录、字段含义及版本差异，查询 cannbot-knowledge 中对应版本的 profiling 产物说明。按任务选择采集会话与 `(phase, rank/device)`，每个选中的组合在独立子目录分析。

每个子 run 的 `inputs.json`（agent 写）：

```json
{
  "prof_dir": "/abs/path/to/<prof 根目录>",
  "kernel_csv": "/abs/path/to/.../kernel_details.csv",
  "trace_json": "/abs/path/to/.../trace_view.json 或 msprof_*.json",
  "model_script_paths": ["/abs/path/to/modeling_xxx.py", "/abs/path/to/moe_module.py"],
  "phase": "decode",
  "rank": 0
}
```

`trace_json` 未提供时可省略。`phase`、`rank` 及其与 `device_id` 的对应关系，从采集元信息、日志或已知部署配置确认。

后续 `detect_structure.py` 启动时会校验 `inputs.json` 存在、`model_script_paths` 非空且文件存在、`phase` / `rank` 字段已填，否则 exit 1。

---

## 历史记录（`<run_dir>/HISTORY.md`）

agent 维护的人类可读运行日志，用途：

- 下一个 agent / 用户进 `<run_dir>` 时，一眼看清"这个模型之前是怎么拆的、为什么这么拆"
- 跨 run 归因：某次性能变化是因为换了 sample？改了 cluster 规则？还是模型本身改了？（与 `compare_runs.py` 输出的 `history.html` 定量对比互补）

**契约**：
- 进 `<run_dir>` 时若文件存在 → 必读
- 每次完成 `render.py`（Step 3 Phase 2 出 `runs/<label>/`）后 → 必 append 一条
- append-only，按时间从老到新；禁止回头改旧条目

条目字段、写法、撤回规则见 `references/history_template.md`。

---

### Step 1 — 单 step 提取

在 skill 根目录执行，`<kernel_csv>` 取 `inputs.json` 的同名字段，`<run_dir>` 用绝对路径：

```bash
python scripts/analyze_kernels.py -f <kernel_csv> -s <step_id> \
  -o <run_dir>/raw_ops.json -d <run_dir>/raw_ops_details.json
```

**作用**：从 csv 抽一个稳定 step，后续所有分析以单 step 为单位。`-s` 不指定时取第一个 step，而第一个 step 可能是 warmup，用 `-s` 显式指定一个稳态 step id。

**step 选择**：多 step 的 kernel 分布不同时，未指定 `-s` 会停止并提示选择；显式指定后提取目标 step，其他 step 的差异仅作提示。结合采集阶段与运行记录选择稳态 step。

**输入**：`kernel_details.csv`（同目录通常还有 `trace_view.json`，Step 4 用，Step 1 不消费）。

**输出**：
- `raw_ops.json` — 单 step `operators[]`，每条含 `index` / `org_index`（csv 0-based 数据行号）/ `normalized_name` / `duration_us` / `start_time_us` / `stream_id` / `input_shapes` / `output_shapes`。Step 2、3 消费。
- `raw_ops_details.json` — 单 step 算子详情，用于明细展示、理论统计和洞察。

**行号 domain**（Excel / `org_index` / `index`，三者各自的语义和换算关系）+ AI_CPU drift 注意事项见 `references/phase0b_workflow.md` 末尾的"行号 domain 速查"。

**Fallback 行为**：`accelerator_core ∈ {AI_CPU, AICPU}` 默认剔除；sample 模式严禁加 `--keep-ai-cpu`（破坏 row-order）。若 profiler 同时记录 HCCL/COMMUNICATION 的 collective summary 行和 `Name == AivKernel` 的 task fragment 行，Step 1 只保留 summary 行、丢弃可匹配 summary 的 fragment；没有 summary 证据的通信 fragment 必须保留，不能静默丢失。AI_VECTOR_CORE 中的 Gather/Scatter/Reduce/Slice 等仍按普通算子保留。

---

### Step 1.5 — 理论性能注入（可选，默认不启用）

**可选步骤**——依赖外部 `operator-theoretical-perf` skill。不启用时 `render.py` 的 theory median 列自动留空（`—`）、不阻塞；Step 5 的理论偏离 insight 也自动跳过（输出空 JSON + `data_limits` 说明）。仅当任务需要理论对比、且外部 skill 可用时才启用。

**启用时**：使用已确认的 chip 配置，在 `render.py` 之前完成理论列合并。

调用外部 `operator-theoretical-perf` skill（或直接跑其 `analyze_operators.py --chip-config <chip>.json`），用原始 `kernel_details.csv` + 已确认的 chip 生成 `<run_dir>/operator_analysis.csv`，再把理论列按原 CSV 行序 / `org_index` 合并回 `<run_dir>/raw_ops_details.json`：

```bash
python scripts/merge_theoretical_columns.py \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  --operator-analysis-csv <run_dir>/operator_analysis.csv \
  -o <run_dir>/raw_ops_details.json
```

理论列不参与 Step 2 结构拆解；sub-item / TOTAL 粒度理论耗时在 Step 3 渲染时从合并后的 `raw_ops_details.json` 计算。调用契约、多流选择口径见 `references/theoretical_perf_workflow.md`。

---

### Step 2 — sample-driven 层检测

**Phase 0a：识别模型结构**

结合模型实现、本次运行配置和调用方信息，识别本次 prof 的结构：

- 覆盖主模型、投机解码 / MTP / draft 头及其他独立子模型（如 encoder、vision tower），层数按本次实际执行情况填写。
- `component` 是一层内与 attention / ffn 平级的功能块；一层的 component 组合为 `composition`，记录各 composition 对应的层号。
- 同一种功能跨阶段使用一致的 component 名；`phase` 按子模型语义命名，如 `main`、`mtp_head`、`encoder`。

写入 `<run_dir>/structure_spec.json`，供后续 sample 定位与结构检测使用。schema：

```json
{
  "model_hint": "<自然语言>",
  "phases": [
    {"name": "main", "layers": 6,
     "layer_compositions": [
       {"layer_range": [0, 2], "components": ["mla", "dense"]},
       {"layer_range": [3, 5], "components": ["mla", "moe"]}
     ]},
    {"name": "mtp_head", "layers": 3,
     "layer_compositions": [{"layer_range": [0, 2], "components": ["mtp", "moe"]}]}
  ],
  "expected_components": ["mla", "dense", "moe", "mtp"],
  "expected_layer_count": 9
}
```

每个 composition 的层用 `layer_range`（连续区间，闭区间）**或** `layer_indices`（显式枚举层号）二选一表达，层号均为 **phase 内 0-based**。当相邻层使用不同 composition、连续区间表达不了时（**逐层交替 / 交错**结构），用 `layer_indices` 显式枚举各 composition 占用的层号。下例示意两种 composition 按奇偶层交替（`attn_a` / `attn_b` 为占位名，实际填该模型的 component 名）：

```json
"layer_compositions": [
  {"layer_indices": [0, 2, 4], "components": ["attn_a", "moe"]},
  {"layer_indices": [1, 3, 5], "components": ["attn_b", "moe"]}
]
```

**同一 phase 内各 composition 的层集合不得重叠**——重叠（同一层被多个 composition 覆盖）会被 `_expand_schedule` 判为 hard error 并中止，而非静默取最后一个。

**Phase 0b：确定 stream sample 范围**

每个 `expected_components` 条目都需提供 sample，并转换为 `stream_sample_ack.v1`：一个 `primary` stream，其他语义上属于该 component 的流标为 `auxiliary`。从调用方提供的范围或模型实现定位 sample，核对实际 prof 后写 `<run_dir>/sample_ack.json`。

辅助流归属与时间脱节处理、定位步骤及完整 schema 见 `references/phase0b_workflow.md`。

**Phase 1：跑 detect_structure**

```bash
python scripts/detect_structure.py \
  -r <run_dir>/raw_ops.json \
  --structure-spec <run_dir>/structure_spec.json \
  --inputs <run_dir>/inputs.json \
  --sample-ack <run_dir>/sample_ack.json \
  -o <run_dir>/structure_draft.json
```

`--inputs` 和 `--sample-ack` 在 sample 模式下**必传**：
- `--inputs <run_dir>/inputs.json`：校验 `model_script_paths` 非空（Step 0 物料）
- `--sample-ack <run_dir>/sample_ack.json`：使用 `schema_version: "stream_sample_ack.v1"`，校验每个 component 都有 `stream_samples[]` 且只有一个 `primary`
- 校验失败时修正对应输入后重跑

审查 `warnings[]` 的 component、层号和期望 / 实测差异，修正结构或 sample。hard / ambiguous warnings（含 `auxiliary_stream_temporally_displaced`）默认 exit 1；有证据接受的情况记录依据后加 `--accept-warnings` 重跑，未解决的歧义交调用方确认。

**输出 `structure_draft.json`**（下游 Step 3 / Step 4 直接消费）：
- `mode: "stream_sample_driven"`
- `schema_version: "structure_draft.stream.v1"`
- `structure_spec` — 原样回写
- `samples_used[]` — 每 sample 的 `stream_samples[]`
- `components[]` — 扁平表，每条 = 一个 component instance（`component_id` / `layer_idx` / `phase` / `type` / `op_indices[]` / `stream_segments[]` / stream-aware `scores`；某条 aux 流被判时间脱节时还含 `displaced_op_indices[]`——该流的 op，render 据此从 cluster/bubble/TOTAL 剔除，但仍在 `op_indices` 全集内、partition 不破）
- `op_to_component` — op index 到 component_id 的反查表
- `unmatched_op_indices[]` / `unmatched_stream_segments[]` — 未归属 op，不能静默丢弃
- `warnings[]` — primary/auxiliary stream、op membership、composition mismatch 与 schedule、sample ack 不匹配、stream shape 不匹配、temporally-displaced aux stream 类问题，每条含 `code` + `message`
- `validation` — declared vs detected 层数、per_phase_layer_match、ambiguous_matches（由 `op_membership_conflict` 填充）

warning codes、validation 字段、匹配算法细节见 `detect_structure.py` / `sample_matching.py` docstring。

---

### Step 3 — cluster 规则与渲染

**Phase 1：确定 cluster 规则**

结合模型语义和代表实例的算子序列，为每个 `component_type` 编写 cluster 规则，保存为 `<run_dir>/<network>_spec.json`。规则字段、完整模板、数值定义和复用依据见 `references/cluster_dialogue.md`。

**Phase 2：渲染**

**启用理论时**（Step 1.5 已把逐 kernel 理论列合并进 `raw_ops_details.json`），渲染脚本在 cluster / TOTAL 粒度按 critical stream 聚合理论耗时；**未启用则 theory median 列整列留空（`—`）**，本段其余多流口径不适用。多流时默认取 observed timeline union 最大的 stream，该选择用于理论统计；结合流间依赖与实际重叠判断其是否代表主要路径，其他流不计入该 sub-item 的理论耗时。若某 sub-item / TOTAL 出现多流，agent 应在 `<run_dir>/theory_stream_decisions.json` 填 `semantic_note` 说明这些流的语义；若候选流覆盖范围接近，还要同时判断是否覆盖 `selected_stream`。没有语义说明的多流项会在理论列标 `semantic note required`。

```bash
python scripts/render.py \
  -d <run_dir>/structure_draft.json \
  -r <run_dir>/raw_ops.json \
  --raw-ops-details <run_dir>/raw_ops_details.json \
  -s <run_dir>/<network>_spec.json \
  --label <label> \
  -o <run_dir>/runs/<label>/index.html
```

（首次渲染也走 `runs/<label>/`，与输出文件表 / Step 5 回填 / `compare_runs` 一致；`metrics.json` 落到同目录。）

如需填写多流语义或覆盖 critical stream 选择，再追加：

```bash
--theory-decisions <run_dir>/theory_stream_decisions.json
```

HTML 主表在 `median` 后展示 `theory median` 列（数据来自 Step 1.5 合并进 `raw_ops_details.json` 的 `theoretical_operator_time_us`；**未跑 Step 1.5 时整列留空 `—`**）；两列都按量级自动显示 `µs` 或 `ms`。bubble 行没有 kernel 理论耗时，显示 `—`。

主报告最后一列为 `insight`，首次渲染留空，Step 5 通过 `--insight-annotations` 回填 high / medium 洞察。类别、目标映射及完整模板见 `references/insight_workflow.md`。

`--raw-ops-details` **必传**：缺了 chip 退化成静态 `<span>`，无法下钻 per-op 明细。

**unmatched gate**：`render.py` 在 **compute-core unmatched 比例 > `--unmatched-limit`（默认 5%）** 时 exit 1（通信类 core 不计入该比例）。报错打印按 `accelerator_core` 的 unmatched 分解：若主要是通信 / 采样 / embedding / IO，加 `--accept-unmatched` 放行；若有大段 compute 算子漏匹配，回 Phase 0a/0b 补 sample 或声明 component。`--unmatched-limit` 可调阈值。

**输出**：
- `index.html` — 单页报告，按 spec 中 component_type 顺序排，每 component_type 一张 cluster / bubble / TOTAL 抖动表，含 `unmatched ops` 面板。若 cluster 桶之间存在 timeline overlap 或覆盖率异常，HTML 顶部按下列规则插提示面板：
  - **cluster 覆盖率红字**：Σ cluster_wall.median / TOTAL.median < 80% 时插红字 banner，结合 bubble 与 unmatched 判断时间空隙和分桶遗漏。
  - **overlap 黄字警告**：median gap ≥ 5% TOTAL，提示"分桶不能直接反映真实性能分布"，列主要重叠桶对 + 占比。
  - **overlap 浅蓝 info**：median gap 0 < x < 5% TOTAL，列出仍存在跨 stream 并发的桶对（过滤掉 < 1µs 的噪声项，最多 8 对），方便定位并发来源，但不影响主表读数。
  - **辅流时间脱节提示（浅蓝）**：某 aux 流被判 displaced 时，列其 op_count / median wall / %TOTAL，标注"已从 TOTAL/bubble 排除"、op 仍 matched（逐层明细见 splits/）。
- `metrics.json` — `compare_runs.py` 消费的指标数据；每个 section 含 `overlap: {median_gap_ms, median_gap_pct, max_gap_ms, top_pairs[]}`、`cluster_sum_pct_of_total`、`displaced`（per-stream 脱节摘要）字段。字段名保持 `*_ms`，HTML 展示层会把小于 1ms 的值显示成 `µs`。
- render 完成时若 `<run_dir>/HISTORY.md` 不存在，stdout 打 reminder（不 exit 1）。

cluster / bubble / TOTAL 定义、outlier 检测、overlap 度量、规则求值顺序见 `render.py` docstring。

---

### Step 4 — 按 component 切 csv / trace_view

**必跑**。skill 的交付契约要求 op 序列被完整切片：每个 component 实例与每段 unmatched op 集合各占一个文件夹，整体严格覆盖 `[0, N-1]`、无重无漏。新 draft 以 `op_indices` 为准，component 可以是非连续 op 集合；`op_range` 只作为 manifest 里的展示 envelope。脚本会自检 partition，draft 与 raw_ops 不匹配时直接 abort。

```bash
python scripts/split_artifacts.py \
  -d <run_dir>/structure_draft.json \
  -r <run_dir>/raw_ops.json \
  -f /path/to/kernel_details.csv \
  -t /path/to/trace_view.json \
  -o <run_dir>/splits/
```

文件夹命名（直接对应 layer_id 与 component 类型）：

| kind | 命名 |
|------|------|
| component 实例 | `L<idx:03d>_<phase>_<type>`；同层出现同类型多个实例时追加 `_<lo>_<hi>` 消歧 |
| 未归属区间 | `unmatched_<lo>_<hi>` |

每个文件夹内只放两份产物：
- `kernels.csv` — csv 切片（保留 header，首列 `org_index`）
- `trace_view.json` — trace 证据；传 `-t` 时是该目标 op 的时间包络上下文窗口，可能包含同时间并行的其他目标 op；不传时从 raw_ops 的 `op_indices` 精确合成

顶层 `manifest.json` 形如 `{source, n_ops, targets}`，`targets` 是 `{folder, kind, op_range, op_indices?, ops_count, files}` 的列表，给下游 / 用户导航用；`op_indices` 恒存在，目标 op 连续时为 `null`。`files.trace_scope` 标明 trace 是 `op_membership_exact` 还是 `time_envelope_context`。

`-f` 使用 Step 1 的输入 CSV；有原始 trace 时传 `-t`，未提供时省略 `-t`，从 raw_ops 合成 trace_view.json。`--dry-run` 打印目标清单并跳过 csv / trace 切片，仍会建目录并写 `manifest.json`。

### Step 5 — 洞察归纳

Step 5 在 Step 3 / Step 4 之后执行，基于已核对的结构与 cluster 规则归纳洞察；发现拆解错误时回到对应阶段修正并更新下游产物。事实提取与渲染由 `scripts/gen_insights.py`（先准备 `movement_taxonomy.json`，生成候选后填写 `annotations.json` 并重跑合并）+ `scripts/render_insights.py` 完成，agent 不手写提取逻辑；候选可靠性 / 是否展示由 agent 用 `confidence`（high|medium|low）标注。CLI、脚本/agent 边界、五类 insight、HTML 渲染与落盘契约见 `references/insight_workflow.md`。

默认输出目录：`<run_dir>/runs/<label>/insights/`，和该 run 的 `index.html` / `metrics.json` 放在同一目录下。只有未使用 `runs/<label>/` 的老式单 run 输出，才退回 `<run_dir>/insights/`。Step 5 最后还要生成 `main_report_insights.json`，并重跑主报告渲染（`render.py --insight-annotations`），把 high / medium 洞察连同类别和主映射说明填回 `runs/<label>/index.html` 最后一列。

---

## 增量分析（模型迭代 + 历史对比）

按 `references/reuse_policy.md` 复用有效产物，仅更新受变更影响的阶段。

**每个 run 落盘到独立目录**：`render.py` 的 `--label <label>` + `-o <run_dir>/runs/<label>/index.html`（首次起就这样，见 Step 3 Phase 2）让每个 run 独占 `runs/<label>/`。同目录同时产 `metrics.json`（含 `label` / `model_name` / `outlier` / 每个 `(component_type, cluster)` 的 wall + bubble 指标 + outlier 列表，`compare_runs.py` 消费）。

**对比表**：

```bash
python scripts/compare_runs.py \
  -r <run_dir>/runs/ \
  -o <run_dir>/history.html \
  [--metric median_ms]    # 默认 median_ms；可选 mean_ms / p95_ms / max_ms / std_ms
  [--baseline <label>]    # 默认第一个 run 当 baseline
```

`-r` 给目录时自动扫所有 `*/metrics.json`，按 mtime 升序排（早的当 baseline）。也可以用 `--runs path1 path2 ...` 列出多个 `metrics.json` 路径显式指定顺序（`-r` 只接受一个目录）。

输出 `history.html` — 多 run 跨版本对比表（Δ% vs baseline + outlier diff），列布局见 `compare_runs.py` docstring。

---

## 输出文件

| 文件 | 阶段 | 说明 |
|------|------|------|
| `<run_dir>/raw_ops.json` | Step 1 | 单 step kernel 概要，下游所有脚本消费 |
| `<run_dir>/raw_ops_details.json` | Step 1；Step 1.5 可加列 | 单 step kernel 详情，供渲染和洞察使用；启用理论性能时合并理论列 |
| `<run_dir>/operator_analysis.csv` | Step 1.5 | 外部 `operator-theoretical-perf` skill 对原 CSV 加列后的原生产物 |
| `<run_dir>/theory_stream_decisions.json` | Step 3 / 2 | agent 对多流语义的说明（多流场景必填）；必要时覆盖 critical stream 选择 |
| `<run_dir>/structure_spec.json` | Step 2 / 0a | 本次运行的模型结构 |
| `<run_dir>/structure_draft.json` | Step 2 / 1 | component 实例表，Step 3 / 4 输入 |
| `<run_dir>/<network>_spec.json` | Step 3 / 1 | 本模型的 cluster 规则，新 run 优先复用同名文件 |
| `<run_dir>/runs/<label>/index.html` | Step 3 / 2 | 单 run 的单页 cluster 抖动报告 |
| `<run_dir>/runs/<label>/metrics.json` | Step 3 / 2 | 单 run 指标，`compare_runs.py` 消费 |
| `<run_dir>/runs/<label>/insights/module_bubble.json` | Step 5 | 模块长期 bubble 与具体层/模块 bubble 异常 |
| `<run_dir>/runs/<label>/insights/operator_jitter.json` | Step 5 | 算子 / cluster 性能抖动候选与证据 |
| `<run_dir>/runs/<label>/insights/theoretical_deviation.json` | Step 5 | 逻辑 sub-item / TOTAL / operator slot 与理论性能偏离 Top5 |
| `<run_dir>/runs/<label>/insights/vector_sequence_candidates.json` | Step 5 | 连续 vector 区间融合候选，按 pattern 聚合 |
| `<run_dir>/runs/<label>/insights/data_movement_ops.json` | Step 5 | 冗余 Tensor Movement 消除候选：搬运 / 结构变换算子定位、上下文与消除方向 |
| `<run_dir>/runs/<label>/insights/final_conclusions.md` | Step 5 | agent 基于五类 insight 汇总的最终结论与证据索引 |
| `<run_dir>/runs/<label>/insights/index.html` | Step 5 | 五类 insight 的可视化页面 |
| `<run_dir>/runs/<label>/insights/main_report_insights.json` | Step 5 | agent 选择回填到主报告 `insight` 列的 high / medium 洞察 |
| `<run_dir>/runs/<label>/insights/_review_stub.json` | Step 5 | 列出待 agent 审阅的 key |
| `<run_dir>/runs/<label>/insights/_taxonomy_candidates.json` | Step 5 | op×core 频次，供 movement 分类参考 |
| `<run_dir>/history.html` | 增量 | 多 run 对比表（Δ% vs baseline） |
| `<run_dir>/HISTORY.md` | 每 run 收尾 | agent append 的运行日志，记录复用 / 变更 / 观察 |
| `<run_dir>/splits/` | Step 4 | 按 component 切的 csv / trace_view，含 manifest.json |

---

## 参考

- `references/kernel_data_guide.md` — 算子 CSV 输入契约与行号对应
- `references/phase0b_workflow.md` — Step 2 Phase 0b stream sample 定位 + `sample_ack.json` schema + 行号 domain
- `references/cluster_dialogue.md` — Step 3 cluster 规则生成步骤 + `network_spec.json` schema + 数值定义
- `references/history_template.md` — `HISTORY.md` 条目字段与模板
- `references/reuse_policy.md` — 变更检查依据、三档复用动作与重跑范围
- `references/theoretical_perf_workflow.md` — Step 1.5 理论性能注入：外部 skill 调用、CSV 理论列合并、多流选择口径
- `references/insight_workflow.md` — Step 5 洞察归纳：五类 insight、脚本/agent 边界、HTML 渲染
- `scripts/templates/insight_page.html` — Step 5 洞察页 HTML/CSS/JS 骨架，由 `scripts/render_insights.py` 填充
- 各脚本 docstring — 内部算法、阈值推导、edge case 处理
