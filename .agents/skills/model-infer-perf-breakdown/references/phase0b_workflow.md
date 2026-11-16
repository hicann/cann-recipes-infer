# Phase 0b — stream sample 定位

每个 `expected_components[*]` 都需核对 sample 范围，并转换成 `stream_sample_ack.v1`。每个 component 有且只有一个 `primary` stream；语义上属于该 component 的旁路 / 辅助流写 `auxiliary`。

**辅助流按语义归属**：属于 component 的流即使与本层时间脱节，也保留在该 component；全局通信 / 采样 / embedding / IO 等不属于任何 component 的流留 `unmatched`。pre / post / inter 无需单独提供 sample。

时间脱节触发 `auxiliary_stream_temporally_displaced`：对应 op 仍保留在 `op_indices` 中，供完整切片与逐层查看；`render.py` 按 component type / stream 整体剔除其 cluster / bubble / TOTAL 计时并单独标注。该告警默认阻断，按正文的告警审查流程处理。判定细节见 `sample_matching.py`。

## A：已有 sample 范围

调用方提供 `component:row_lo-row_hi`（Excel 行号）+ 入口/出口算子提示；或只给入口/出口算子提示，agent 去 raw_ops 定位典型窗口。

agent 工作步骤：

1. 把提供的算子名对照 `raw_ops.json` 的 `normalized_name` 定位实际命中位置。提供 row 区间时校验窗口开头/结尾附近的算子是否与入口/出口提示一致。
2. 核对 sample 首尾和邻近 op，展示时使用 Excel 行号（`org_index + 2`）：

   ```
   csa : excel 1210-1455  head=RmsNorm@1210  tail=TransposeBatchMatMul@1455
         邻近：RmsNorm@1209(prev), AddRmsNormDynamicQuant@1456(next)
   moe : excel 1456-1750  head=AddRmsNormDynamicQuant@1456  tail=MoeDistributeCombine@1750
         邻近：HcPost@1751(next)
   ```

3. 读取已核对的 sample 小窗口，把窗口内 op 按 `stream_id` 拆成 `stream_samples[]`：
   - 默认把 op 数最多 / duration 最大、且包含主要 compute 路径的 stream 设为 `primary`。
   - 明确属于同一 component 的 index/cache/vector side path 设为 `auxiliary`。
   - 不确定的 stream 不写入 component；保留到 warning / unmatched，不要为了好看硬归属。
4. 写 `sample_ack.json`，进 Phase 1。

---

## B：从模型实现定位

未提供 sample 范围时，读取 `model_script_paths` 中的模型实现：

1. 读 `forward()` 找每个 component 的第一个 / 最后一个 compute 调用，结合实际 prof 确定对应 kernel。对应关系不明确时，查询 cannbot-knowledge 中相关算子实现与映射资料。
2. （可选）跑 `python scripts/detect_structure.py --explore -r <run_dir>/raw_ops.json -o <run_dir>/explore.json` 看候选算子的频次是否符合预期（count = N × 该 component 在该 phase 出现次数；next_kernels_distinct 低的更适合做锚）。
3. 按 A 的步骤核对 sample、划分主辅流，记录来源脚本和函数，写入 `sample_ack.json`。

---

## `sample_ack.json` schema

```json
{
  "schema_version": "stream_sample_ack.v1",
  "acked_at": "2026-05-22T22:00:00+08:00",
  "components": {
    "csa": {
      "source": "user",
      "primary_stream_id": "189",
      "stream_samples": [
        {
          "stream_id": "189",
          "role": "primary",
          "head_op": "RmsNorm",
          "tail_op": "TransposeBatchMatMul",
          "op_indices": [1208, 1210, 1214]
        },
        {
          "stream_id": "203",
          "role": "auxiliary",
          "head_op": "Slice",
          "tail_op": "ScatterNdUpdate",
          "op_indices": [1209, 1212]
        }
      ],
      "agent_note": "stream 189 is the main compute path; stream 203 carries index/cache side ops"
    },
    "moe": {
      "source": "agent_inferred_from_script:modeling_xxx.py",
      "primary_stream_id": "203",
      "stream_samples": [
        {"stream_id": "203", "role": "primary", "op_indices": [1454, 1456, 1460]}
      ]
    }
  }
}
```

`source` 三种取值：
- `"user"` — 调用方直接提供
- `"agent_inferred_from_script:<path>"` — agent 读脚本推出来的
- `"user_corrected"` — 调用方修正过的样本

`source` 记录样本依据，`acked_at` 记录核对时间。`expected_components` 中任一缺失时，`detect_structure.py` 启动校验失败。

若当前只有人工给出的 Excel 区间，agent 必须先读取对应小窗口，按 `stream_id` 拆成 `stream_samples[]`，再写入上面的 v1 schema（Step 2 输入须为此 v1 schema）。

---

## 行号 domain 速查

- **Excel 行号**（用户口头/书面交换，agent 用它读取小窗口并转换为 `sample_ack.json`）：1-based，header 占 row 1
- **`org_index`**（`raw_ops.json` 字段、Step 4 切 csv 用）：0-based 数据行号，与 Excel 关系 `excel_row = org_index + 2`（标准 csv 阅读器）
- **`index` / `op_indices`**（operators[] 内 sequential，AI_CPU 过滤后致密重编号）：`sample_ack.json`、`structure_draft.json`、`render.py`、`split_artifacts.py` 全部按这个 domain

AI_CPU drift：默认剔除 `accelerator_core ∈ {AI_CPU, AICPU}` 后 `index != org_index`，**手动减 2 后写裸 `op_indices` 必错**。agent 必须先按 Excel 行号反查对应 raw ops，再把确认后的致密 `index` 写入 `stream_samples[].op_indices`。
