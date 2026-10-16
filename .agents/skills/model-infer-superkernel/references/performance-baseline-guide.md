# 性能基线建立指南

本文档提供详细的性能基线建立方法，用于对比 SuperKernel 优化前后的性能差异。

---

## 为什么需要性能基线

性能基线是评估优化效果的参照标准：
- **量化提升**：精确计算性能提升比例
- **问题定位**：性能下降时快速定位问题
- **方案对比**：对比不同 Scope 策略的效果
- **回归测试**：确保后续修改不影响性能

---

## 性能指标

### 核心指标

| 指标 | 说明 | 单位 | 重要性 |
|------|------|------|--------|
| **Decode 单步耗时** | 生成一个 token 的时间 | ms | 高 |
| **吞吐量** | 每秒生成的 token 数 | tokens/s | 高 |
| **首 token 延迟** | Prefill 阶段耗时 | ms | 中 |
| **内存占用** | 推理时的显存使用 | GB | 中 |
| **编译时间** | torch.compile 耗时 | s | 低 |

### 辅助指标

| 指标 | 说明 | 用途 |
|------|------|------|
| **CPU 使用率** | CPU 占用百分比 | 识别 CPU 瓶颈 |
| **NPU 利用率** | NPU 计算资源使用率 | 识别 NPU 瓶颈 |
| **通信开销** | 多卡场景的通信时间 | 优化并行策略 |

---

## 基线建立流程

### 步骤 1：准备测试环境

#### 1.1 确认硬件配置

```bash
# 查看 NPU 信息
npu-smi info

# 确认硬件型号
# 应该是 Atlas A3 系列
```

#### 1.2 确认软件版本

```bash
# CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# PyTorch 版本
python -c "import torch; print(torch.__version__)"

# torch_npu 版本
python -c "import torch_npu; print(torch_npu.__version__)"
```

#### 1.3 设置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/path/to/cann-recipes-infer:$PYTHONPATH
```

---

### 步骤 2：配置基线测试

#### 2.1 禁用 SuperKernel

复制一份关闭 SuperKernel 的配置，保留原配置不动（在仓库根执行）：

```bash
cp models/{model_name}/config/{yaml_name}.yaml models/{model_name}/config/{yaml_name}_sk_off.yaml
sed -i 's/enable_superkernel: True/enable_superkernel: False/' models/{model_name}/config/{yaml_name}_sk_off.yaml
```

再确认副本的 `model_config`：`exe_mode: "ge_graph"` 保持图模式、`enable_cache_compile: False`、`custom_params.enable_multi_streams: False` 排除其他优化干扰。

#### 2.2 配置测试参数

```yaml
scheduler_config:
  batch_size: 1                   # 固定 batch size
  max_new_tokens: 100             # 固定生成长度

data_config:
  dataset: "default"              # 使用 dataset/default_prompt.json 中的固定 prompt
```

### 步骤 3：运行基线测试

推理入口按部署模式选：框架部署在仓库根执行 `bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}_sk_off.yaml`；独立部署在模型目录执行 `python -u infer.py --yaml_file_path config/{yaml_name}_sk_off.yaml`。

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

# 第一次运行预热，不计入统计（等待图编译完成）
bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}_sk_off.yaml

# 正式测试跑 3-5 次
for i in $(seq 1 5); do
    echo "=== Run $i ==="
    bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}_sk_off.yaml
    sleep 5
done
```

对比结束后删除临时副本 `models/{model_name}/config/{yaml_name}_sk_off.yaml`。

每次运行的日志在 `models/{model_name}/res/<date>/<case>/log_<rank>.log`，从中取 prefill 与 decode 耗时，多次运行取中位数，记录格式见「步骤 4：记录基线数据」。

---

### 步骤 4：记录基线数据

#### 4.1 性能数据格式

创建 `baseline_performance.json`：

```json
{
  "test_info": {
    "model_name": "deepseek-r1",
    "hardware": "Atlas A3",
    "cann_version": "8.3.RC1",
    "pytorch_version": "2.6.0",
    "torch_npu_version": "2.6.0",
    "test_date": "2026-03-30",
    "config": {
      "exe_mode": "ge_graph",
      "enable_superkernel": false,
      "batch_size": 1,
      "max_new_tokens": 100
    }
  },
  "performance": {
    "decode_latency_ms": 15.2,
    "throughput_tokens_per_sec": 65.8,
    "prefill_latency_ms": 120.5,
    "memory_usage_gb": 24.3,
    "compile_time_sec": 45.2
  },
  "statistics": {
    "runs": 5,
    "decode_latency_std": 0.3,
    "throughput_std": 1.2
  }
}
```

#### 4.2 日志文件

保存完整的运行日志：
- `baseline.log` - 完整输出
- `baseline_performance.json` - 性能数据
- `baseline_output.json` - 模型输出

---

### 步骤 5：验证基线稳定性

#### 5.1 检查波动

```python
import glob
import numpy as np

# 从步骤 3 的 5 次运行日志里各取一个 decode 单步耗时
data = []
for log_path in sorted(glob.glob(f'{res_dir}/log_0.log')):
    data.append(extract_decode_latency_ms(log_path))

# 计算统计量
mean = np.mean(data)
std = np.std(data)
cv = std / mean * 100  # 变异系数

print(f"Mean: {mean:.2f} ms")
print(f"Std: {std:.2f} ms")
print(f"CV: {cv:.2f}%")

# 判断稳定性
if cv < 5:
    print("基线稳定")
else:
    print("基线波动较大，建议重新测试")
```

#### 5.2 稳定性标准

| 变异系数 (CV) | 稳定性 | 建议 |
|--------------|--------|------|
| < 5% | 优秀 | 可以使用 |
| 5% - 10% | 良好 | 可以使用，注意波动 |
| 10% - 20% | 一般 | 建议增加测试次数 |
| > 20% | 较差 | 需要排查环境问题 |

---

## 优化后性能测试

### 步骤 1：启用 SuperKernel

修改 `models/{model_name}/config/{yaml_name}.yaml`：

```yaml
model_config:
  exe_mode: "ge_graph"
  custom_params:
    enable_superkernel: True      # 启用 SuperKernel
    # 其他配置保持与基线一致
```

### 步骤 2：运行优化测试

用启用 SuperKernel 的原配置，按与基线相同的次数运行：

```bash
bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml

for i in $(seq 1 5); do
    echo "=== Run $i ==="
    bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml
    sleep 5
done
```

### 步骤 3：性能对比

两组各取中位数后逐项对比，判据：

| 指标 | 方向 | 通过判据 |
|------|------|---------|
| decode 单步耗时 | 越低越好 | 优化版低于基线，且降幅超过基线自身的波动范围 |
| prefill 耗时 | 越低越好 | 不回退（SuperKernel 仅在 decode 生效） |
| 显存峰值 | 越低越好 | 不显著上升 |

降幅落在波动范围内的记为「无显著收益」，不记为提升。基线波动范围按步骤 5 的稳定性检查得到。

---

## 性能分析工具

定 scope 需要各模块的 decode 耗时占比，由 profiling 采集与模块拆解得到。

### 自定义计时

```python
import time
import torch_npu

def benchmark_decode(model, input_ids, num_steps=100):
    # 预热
    for _ in range(10):
        _ = model.decode(input_ids)

    # 正式测试
    torch_npu.npu.synchronize()
    start = time.time()

    for _ in range(num_steps):
        _ = model.decode(input_ids)

    torch_npu.npu.synchronize()
    end = time.time()

    avg_latency = (end - start) / num_steps * 1000  # ms
    throughput = num_steps / (end - start)  # steps/s

    return {
        'avg_latency_ms': avg_latency,
        'throughput_steps_per_sec': throughput
    }
```

---

## 性能报告模板

### 完整报告结构

> 下面模板中的数字是占位示例，不是实测值；填报时全部替换为本次运行的实际数据。

```markdown
# SuperKernel 性能测试报告

## 测试环境

- 模型：DeepSeek-R1
- 硬件：Atlas A3 (8 卡)
- CANN 版本：8.3.RC1
- PyTorch 版本：2.6.0
- 测试日期：2026-03-30

## 测试配置

- Batch Size: 1
- Max New Tokens: 100
- Scope 策略：仅 Attention 模块

## 性能对比

| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| Decode 单步耗时 | 15.2 ms | 12.8 ms | +15.8% |
| 吞吐量 | 65.8 tokens/s | 78.1 tokens/s | +18.7% |
| 内存占用 | 24.3 GB | 24.5 GB | -0.8% |

## 详细数据

### 基线性能
- Decode 单步耗时：15.2 ± 0.3 ms
- 吞吐量：65.8 ± 1.2 tokens/s
- Prefill 延迟：120.5 ms
- 内存占用：24.3 GB

### 优化后性能
- Decode 单步耗时：12.8 ± 0.2 ms
- 吞吐量：78.1 ± 1.0 tokens/s
- Prefill 延迟：120.3 ms
- 内存占用：24.5 GB

## 结论

SuperKernel 优化在 Decode 阶段带来了 **15.8%** 的性能提升，吞吐量提升 **18.7%**。内存占用基本持平。优化效果符合预期。

## 下一步

- 尝试扩大 Scope 范围到全 Decoder 层
- 结合多流并行进一步优化
- 在更长序列上测试性能
```

---

## 常见问题

### Q1: 基线性能波动大怎么办？

**可能原因**：
1. 系统负载不稳定
2. NPU 频率波动
3. 测试数据不一致

**解决方法**：
1. 关闭其他进程，确保系统空闲
2. 固定 NPU 频率
3. 使用固定的测试数据和参数
4. 增加测试次数，取中位数而非平均值

### Q2: 如何确保测试的公平性？

**关键点**：
1. 使用相同的测试数据
2. 使用相同的配置参数（除了 enable_superkernel）
3. 在相同的环境下测试
4. 预热后再测试
5. 多次运行取中位数

### Q3: 性能提升不明显怎么办？

**排查步骤**：
1. 确认 SuperKernel 是否真正启用（查看日志）
2. 检查 Scope 范围是否合适
3. 使用 profiler 分析瓶颈
4. 尝试不同的 Scope 策略
5. 结合其他优化技术

---

## 最佳实践

1. **固定测试条件**：确保基线和优化测试使用完全相同的条件
2. **多次测试**：至少运行 3-5 次，取中位数
3. **预热运行**：第一次运行不计入统计
4. **记录详细信息**：保存完整的配置、日志和性能数据
5. **版本控制**：使用 git 记录每次测试的代码版本
6. **自动化**：编写脚本自动化测试流程
7. **可视化**：使用图表展示性能对比

---

## 参考资源

- Scope 分析指南：`scope-analysis-guide.md`
- 官方文档：[CANN 性能分析工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/devtools/auxiliarydevtool/atlasprofiling_16_0001.html)
