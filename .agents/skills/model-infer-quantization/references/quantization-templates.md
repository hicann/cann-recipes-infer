# 量化输出 / 记录模板

供 `model-infer-quantization` skill 第五步（初评估输出）与第八步（量化结果）按需引用。

## 量化方案初评估（输出模板）

```markdown
## 量化方案初评估

> 在 model-infer-optimize 中，将本模板结果写入 progress.md 的量化初评估记录；其他调用按调用方指定位置交付。

### 基本结论
- 模型：
- 结构指纹：
- compressed-tensors 契约结论：
- 当前评估目标：

### 量化产物信息
- 量化产物目录：
- 目标量化模式：
- 期望消费方式：
- 当前核心配置格式：

### 契约核对
- `config.json`：
- `model.safetensors.index.json`（如有）：
- `deploy_quantization.md`：
- 当前 compressed-tensors 判定结果：

### 运行对象映射
- `{quant_target} -> {runtime_object}`（逐项）
- ignore / 显式浮点回退模块：
- 推荐首版无缝接入范围：

### infer 侧重点核查项
- `quantization_config` 是否可直接复用：
- `process_weights_after_loading` 后处理是否写清：
- `kv_cache_scheme` 或等价缓存契约是否写清：
- 融合算子兼容性风险（移交阶段 4 复核）：

### 显存与部署形态初评估
- 当前基线部署形态：
- 量化后部署判断：
- **覆盖率与收益上限提示**（advisory，非契约强校验）：
    - 已量化 / 未量化算力切片估计占比
    - 预期受益环节及依据：
    - 工况收益应可追溯到实测对照点；外推须注明外推起点与假设
    - 量化模式与工况匹配性记入补充诉求（如 batch=1 latency-critical 下 W8A8 vs W4A16 预期差异）

### 接入分级结论
- 各目标模块的接入分级（L0/L1/L2/L3）：
- 判定理由：
- 阻塞项及所需补充（如有）：

### 对后续阶段的影响
- 对阶段 1（并行化）：
- 对阶段 4（量化改造）：
- 若阶段 4 复核发现融合与量化冲突：

### 当前建议
- 建议结论：保留量化改造候选 / 暂不量化 / 先补契约
- 前提条件：
- 若前提不满足：
- 需补充的量化算法或产物契约：
```

## 量化结果字段

- 当前模式：改造 / 验证
- 量化权重目录、量化模式、覆盖范围
- `compressed-tensors` 契约结论
- 结构指纹、接入分级
- `quant_target -> infer runtime object` 映射
- `ignore` / 浮点回退模块
- 改造文件、关键实现点、post-load 处理
- 融合算子回退清单、原始错误、回退点、后续融合量化需求
- 验证命令、日志证据、量化是否真实生效
- 显存、Prefill、Decode、部署卡数对比
- 阻塞点、补充诉求、用户待决策项
