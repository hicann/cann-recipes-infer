---
name: model-infer-prefetch
description: 为模型添加 torch_npu.npu_prefetch 优化特性时使用,当 profiling 显示 MatMul/QBMM/GMM 算子存在 memory-bound 热点时使用,或将 prefetch 模式迁移到新模型时使用
---

# Prefetch 预取优化技能

覆盖预取候选分析、大小计算、代码接入和性能验证。

---

## 预取原则

- **仅在权重搬运瓶颈时使用**:结合 profiling 和模型代码确认目标算子的耗时受权重读取限制
- **必须有可重叠的预取窗口**:依赖 Tensor 就绪后、目标算子读取权重前，有足够时间和 HBM 带宽余量执行预取
- **始终提供开关保护**:所有 prefetch 必须用 `enable_prefetch` 开关保护,默认 `False`

---

## 工作流程

### 方案分析

1. 结合目标阶段、执行模式和当前负载，确定目标权重、依赖窗口及插入位置。
2. 根据本 rank 实际权重存储和窗口确定预取范围（`max_size`、`offset`）。
3. 核对目标后端的依赖与流管理约束，形成改造方案和验证方法。

仅分析的任务在此交付；待决事项返回调用方。

### 实施

按选定方案实施；已有方案时从此接手，结合当前代码复核接入条件。贯通 `enable_prefetch` 开关，按执行模式接入预取调用及所需的流与事件，落实预取范围和依赖位置。

### 验证测试

在目标 NPU 上验证编译和执行，以同负载对比 `enable_prefetch=False` 基线和开启后的输出与性能，按下文「性能测试验证」记录实测结果。

---

## 预取位置确定

已指定目标时，评估其权重和依赖窗口；否则从模型代码和已有 profiling 中发现候选。优先复用匹配当前阶段、负载和配置的 profiling，缺少必要数据时补采。

### 候选判定

| 步骤 | 检查项 | 结果 | 下一步 |
|------|--------|------|--------|
| 1 | 结合模型代码、算子耗时与访存指标筛选热点 | 得到权重读取受限的候选 | 进入步骤 2 |
| 2 | 沿代码依赖和执行时间线向前查找窗口 | 确定候选依赖 Tensor 和目标权重 | 进入步骤 3 |
| 3a | 依赖就绪后有足够时间和带宽余量 | 存在可重叠的预取窗口 | 确定预取大小 |
| 3b | 区间过短或同期工作占满带宽 | 当前窗口不足 | 尝试更早的依赖位置或其他窗口 |
| 3c | 无合适窗口 | 无可用预取位置 | 跳过该算子 |
| 4 | 估算权重字节数并结合窗口选取预取量 | 候选 max_size | 输出候选列表 |

### 资源判断与参考样例

窗口是否可用取决于目标负载下的实际耗时、访存与通信活动。结合 shape 和 profiling 判断带宽占用；产生依赖 Tensor 的算子即使是大型 MatMul，也应检查其完成后的区间。

按当前模型结构，查询 cannbot-knowledge 中同框架、同执行模式的权重预取部署样例与调优经验，参考层内、跨层及 MoE 预取的窗口和参数选择。

> profiling 采集方法参考 model-infer-profiling skill；时间分布拆解参考 model-infer-perf-breakdown skill

---

## 预取大小确定

按本 rank 量化打包及 TP/EP 切分后的权重 Tensor 估算字节数，再结合预取窗口确定 `max_size`。NZ 等带填充格式的预取范围按实际存储大小确定。

### 计算公式

```python
weight_bytes = weight.numel() * weight.element_size()
```

`max_size` 按字节取正整数，先选择与窗口匹配的部分权重验证；`offset` 表示权重内的字节偏移，所选范围需对应实际权重存储。

### 调整策略

| 场景 | 调整方向 | 说明 |
|------|---------|------|
| 目标算子加速明显,预取窗口稳定 | 适度增大 max_size | 复查窗口和整体收益 |
| 目标算子加速不明显 | 按预取覆盖、重叠和带宽情况调整 | 覆盖不足且窗口有余量时增大；争抢时减小，重叠不足时调整位置 |
| 预取窗口明显退化 | 减小 max_size | 核对是否争抢带宽 |
| 预取窗口持续退化 | 调整位置或放弃 | 前移 prefetch 或选择其他预取窗口 |

---

## 目标算子与依赖窗口选择

### 目标算子优先级

优先评估权重读取受限且耗时显著的 MatMul、QBMM、GMM、Dense/MoE 线性层，以及含大权重读取的融合算子（如 MLAProlog）。Norm、ROPE、激活和路由等路径可提供候选窗口，是否可用由实际耗时与带宽余量决定。

### 依赖窗口选择标准

从目标算子向前选择表达预取开始时机的 Tensor，并检查:
- 依赖就绪后到目标读取权重前，存在可与预取重叠的计算区间。
- 该区间及同期其他流有 HBM 带宽余量，结合 profiling 的访存指标和耗时判断。
- 可在层内或跨层寻找窗口；位置前移时，复查预取到的数据能否保留到目标使用时。

---

## 代码实现指南

参数语义、预取流与同步、图捕获约束，按部署的 CANN / torch_npu 版本查询 cannbot-knowledge 或官方文档。npugraph_ex 等显式流路径按目标模型的预取流与事件关系接入，流管理参考 `model-infer-multi-stream`。

### 1. 开关传递路径

infer 仓框架部署的开关传递链路:YAML → `InferenceConfig.model_config.custom_params` → 模型类 `__init__` → `forward`/`decode`/`prefill`

**配置文件示例**(`config.yaml`):
```yaml
model_config:
  custom_params:
    enable_prefetch: true  # Default: false.
```

多流开关及其取值遵循目标模型的实现与配置校验。

### 2. Prefetch 调用模式

infer 仓框架部署复用 [`executor/utils/common_utils.py`](../../../executor/utils/common_utils.py) 的 `npu_prefetch` 封装。

**GE 图模式示例**：`window_layer` 对应已选定的可重叠计算区间，`prefetch_size` 由方案确定。
```python
import torch

from executor.utils.common_utils import npu_prefetch


class PrefetchExample(torch.nn.Module):
    def __init__(self, infer_config, producer, window_layer, target_layer, prefetch_size):
        super().__init__()
        self.enable_prefetch = infer_config.model_config.custom_params.get("enable_prefetch", False)
        self.producer = producer
        self.window_layer = window_layer
        self.target_layer = target_layer
        self.prefetch_size = prefetch_size

    def forward(self, x):
        dependency_output = self.producer(x)

        # Schedule prefetch after the dependency is ready.
        npu_prefetch(
            self.enable_prefetch,
            weight=self.target_layer.weight,
            depend=dependency_output,
            size=self.prefetch_size,
            offset=0,
        )

        # Overlap weight transfer with the selected computation window.
        window_output = self.window_layer(dependency_output)
        return self.target_layer(window_output)
```

### 3. GE 图模式下的依赖节点

GE 图模式下，`dependency` 是约束预取开始时机的 Tensor 节点。上例在其就绪后预取目标权重，并用中间的 `window_layer` 提供重叠窗口。按实际图调度核对预取的开始位置、窗口和目标执行顺序。

---

## 性能测试验证

### 验证检查清单

在 prefetch 前后运行 profiling,按以下清单逐一检查:

- [ ] **目标算子执行时间是否下降**:对比 prefetch 前后目标算子的耗时
- [ ] **关键路径是否缩短**:结合跨流依赖、空洞和等待，对比目标阶段整体耗时
- [ ] **预取窗口是否稳定**:对比窗口内及同期算子耗时，核对带宽争抢
- [ ] **是否出现新的性能瓶颈**:检查其他流是否出现新的长拖尾
- [ ] **功能正确性是否保持**:对比 prefetch 前后输出一致性

### 成功标准

功能与精度验证通过，并在同负载下确认目标场景的时延或吞吐改善。结合目标算子耗时、关键路径及其他流的变化解释收益。

### 失败信号与调整策略

| 失败信号 | 核对项 | 调整策略 |
|----------|---------|---------|
| 目标算子没快,预取窗口反而变慢 | 预取量与同期带宽争抢 | 减小 `max_size` 或调整窗口 |
| 关键路径没有缩短,只是等待位置挪动 | 预取与目标执行的重叠情况 | 调整 prefetch 位置并复测整体收益 |
| 多流上出现新的长拖尾 | 同步等待与带宽争抢 | 调整依赖节点或预取量，评估是否保留该位置 |
| 图模式下编译失败 | 当前后端的依赖表达及接口、图捕获约束 | 按部署版本修正接入方式 |

### 调优顺序

1. **先固定位置,只调 `max_size`**:结合窗口内算子耗时与目标收益增减预取量
2. **再固定 `max_size`,微调 dependency 位置**:前移或后移依赖窗口
3. **最后才考虑多目标叠加**:单个目标验证成功后再扩展

---

## 输出要求

- **分析结果**：目标算子和代码位置、权重读取瓶颈及依赖窗口依据、预取范围及计算依据、接入方案和验证方法。
- **实施结果**：实际改动与开关配置、同负载下的功能与精度验证、性能对比、产物路径及后续调优建议。

---

## 参考资料索引

| 主题 | 文件路径 |
|-----|---------|
| npu_prefetch wrapper 函数 | `executor/utils/common_utils.py`（搜索 `npu_prefetch`） |
| 配置读取与验证 | `executor/core/config/inference_config.py`（`ModelConfig.from_dict`、`InferenceConfig._validate`）+ 目标模型的配置校验 |
| profiling 数据采集 | 参考 model-infer-profiling skill |
| 图模式适配 | 调用 `model-infer-graph-mode` skill |
