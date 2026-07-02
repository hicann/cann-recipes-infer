# NpuOpsTransformerExt

## 简介 | Overview

该目录包含用于 **Qwen3-Next 推理** 的 NPU 融合算子实现，目前包括：

- **rmsnormgated** 融合算子
- **Gated Delta Network (GDN)** 融合算子

这些算子以 PyTorch Extension 的形式实现，并在安装后注册到 `torch.ops` 命名空间中供框架调用。


## 核心组件 | Core Components

本模块主要包含以下关键组件：

1. `gated_delta_net/<op_dir>/` 算子实现目录，主要包含：

   - `<op_name>.cpp`：算子调用文件。
   - `op_kernel/`：算子 Kernel 具体实现代码。

2. `gated_delta_net/<op_dir>/CMakeLists.txt` 算子编译配置文件。

3. `npu_ops_transformer_ext/npu_ops_transformer_ext/npu_ops_def.cpp` 算子接口注册文件。


## 环境要求 | Prerequisites

- Python ≥ 3.8
- CANN Ascend Toolkit
- PyTorch ≥ 2.1.0
- torch_npu (PyTorchAdapter)

上述依赖的安装与环境配置请参考 [Qwen3-Next README](../../README.md)。


## 安装步骤 | Installation

1. 进入算子目录，安装依赖：

```bash
pip install -r requirements.txt
```

2. 从源码构建 `.whl` 包：

```bash
python -m build --wheel -n
```

3. 安装构建好的 `.whl` 包：

```bash
pip install dist/*.whl --force-reinstall --no-deps
```

4. （可选）如果需要重新编译，建议先清理编译缓存：

```bash
python setup.py clean
```


## 算子调用 | Usage

完成编译并安装 `.whl` 包后，自定义算子会注册到 `torch.ops` 命名空间中，可通过如下方式调用：

```python
import torch
import npu_ops_transformer_ext

# 调用自定义算子
out = torch.ops.npu_ops_transformer_ext.my_ops(input)
```

其中：

- `npu_ops_transformer_ext` 为算子注册的 namespace
- `my_ops` 为具体算子名称（在 `npu_ops_def.cpp` 中定义）


当前模块包含的算子示例：

```python
torch.ops.npu_ops_transformer_ext.recurrent_gated_delta_rule(...)
torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated(...)
```

具体输入参数格式请参考对应算子的实现代码。