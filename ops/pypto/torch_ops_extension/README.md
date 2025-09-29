## 概述

本工程目标是基于torch原生支持的cpp extension方式，提供用户自定义算子注册，eager模式和torch.compile模式执行的kernel 实现。

关于eager模式与torch.compile模式的介绍参考：[Link](https://pytorch.org/get-started/pytorch-2.0)。


## 编译与安装

- 安装依赖：

编译自定义算子包之前，请先完成 torch 包和torch_npu包的下载与安装。 取包地址：xxxx

- 源码下载：

克隆代码仓，并更新三方库

```shell
git clone https://gitcode.com/cann/cann-recipes-infer.git
git submodule update --init --recursive
```

- 编译安装：

源码编译与安装

```shell
cd xxxxxx
bash build_and_install.sh
```

执行之后会完成源码编译与自定安装。编译完成之后，会在当前路径新生成一个dist文件夹存放新编译的wheel包。


## 快速上手

- 用例执行：

执行examples路径下的测试用例，验证自定义算子的注册和执行

```shell
bash run.sh
```


## 目录结构介绍

主体功能代码在custom_pypto路径，编译工程在setup.py, 测试用例在test路径。

```
├── custom_pypto
│   ├── __init__.py                               #custom_pypto包初始化文件，用于导入自定义算子python实现和编译的so
│   ├── csrc                                      #自定义算子适配层c++代码目录
│   │   ├── add_custom_autograd.cpp               # 自定义add算子适配代码以及实现注册
│   │   ├── npu_lightning_indexer_pto.cpp         # 自定义算子lightning indexer适配代码以及实现注册
│   │   ├── npu_sparse_attention_pto.cpp          # 自定义算子sparse attention适配代码以及实现注册
│   │   ├── ops_def_registration.cpp              # 自定义算子新增定义
│   │   ├── ops_common.cpp                        # 自定义算子调用和下发框架实现
│   │   ├── ops_common.h                          # 自定义算子调用和下发框架公共接口定义
│   │   └── ops_function.h                        # 正反向接口定义头文件
│   └── converter                                 # 自定义算子包python侧converter代码
│   │   ├── __init__.py                           # python初始化文件，用于converter注册
│   │   ├── add_custom.py                         # 自定义add算子的converter注册
│   │   ├── npu_lightning_indexer_pto.py          # 自定义lightning indexer算子的converter注册
│   │   ├── npu_sparse_attention_pto.py           # 自定义sparse attention算子的converter注册
│   ├──setup.py                                   # wheel包编译文件
│   ├──README.md                                  #当前自定义算子仓库的使用指导
│   ├── build_and_install.sh                      # 自定义算子wheel包编译与安装脚本
```


## 新增自定义算子指导

基于上述代码目录结构介绍，梳理新增自定义算子步骤

### step0：新增pytorch算子前准备工作

C++ extensions插件提供了将自定义算子映射到昇腾AI处理器的功能，为使用PyTorch框架的开发者提供了便捷的NPU算子库调用能力，基于PyTorch原生提供的自定义算子扩展功能，用户可以编译安装自定义算子wheel包并运行。
本指导将以lightning_indexer_pto算子为例，基于C++ extensions的方式介绍如何说明在昇腾NPU上完成自定义算子开发和适配。

### step1：在算子适配层c++代码目录（csrc）中完成C++侧自定义算子代码适配

#### 注册自定义算子schema
PyTorch提供TORCH_LIBRARY宏来定义新的命名空间，并在该命名空间里注册schema。注意命名空间的名字必须是唯一的。具体示例如下：
```cpp 
-- custom_pypto/csrc/ops_def_registration.cpp中添加如下代码

#include <torch/library.h>

TORCH_LIBRARY_IMPL(custom_pypto, Meta, m) {
    m.impl("npu_lightning_indexer_pto", &custom_pypto::npu_lightning_indexer_pto_meta);
}
```

#### 注册Meta函数
通过PyTorch的Meta后端帮助算子完成入图时所需要的shape和data type推导。具体示例如下：
```cpp 
-- 创建custom_pypto/csrc/npu_lightning_indexer_pto.cpp
#include <torch/library.h>
#include "ops_common.h"

namespace custom_pypto {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

// 推导输出shape和data type
at::Tensor construct_lightning_indexer_pto_output_tensor(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights, 
    const c10::optional<at::Tensor> &actual_seq_lengths_key, const c10::optional<at::Tensor> &block_table)
{
    at::SmallVector<int64_t, SIZE> output_size;
    int sparse_count = 2048;
    output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count}; // BSND
    at::Tensor output = at::empty(output_size, query.options().dtype(at::kInt));
    return output;
}

// 为META设备实现前向接口
at::Tensor npu_lightning_indexer_pto_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    return construct_lightning_indexer_pto_output_tensor(query, key, weights, actual_seq_lengths_key, block_table);
}
}

// 为Meta设备注册前向实现
TORCH_LIBRARY_IMPL(custom_pypto, Meta, m) {
    m.impl("npu_lightning_indexer_pto", &custom_pypto::npu_lightning_indexer_pto_meta);
}

```

### step2：在算子适配层python代码目录（converter）中完成python侧自定义算子代码适配
#### 注册自定义算子converter

完成自定义算子在NPU 设备上执行的功能接口，内部会调用自定义算子的 C API，如：
```cpp
-- 创建custom_pypto/converter/npu_sparse_attention_pto.py

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr


# 注意命名空间和schema名称需要与前面注册保持一致
@register_fx_node_ge_converter(torch.ops.custom_pypto.npu_lightning_indexer_pto.default)
def convert_npu_lightning_indexer_pto(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    *,
    actual_seq_lengths_query: Tensor = None,
    actual_seq_lengths_key: Tensor = None,
    block_table: Tensor = None,
    layout_query: str = "BSND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    meta_outputs: Any = None
    ):

    '''NB: npu_lightning_indexer_pto(Tensor query, Tensor key, Tensor weights, Tensor actual_seq_lengths_key, *, Tensor? block_table=None) -> Tensor'''
    
    return torchair.ge.custom_op(
        "LightningIndexerPto",                 # 和REG_OP中算子原型名称保持一致，例如AddCustom
        inputs={"x0": query,                   
                "x1": key,
                "x2": weights,
                "x3": actual_seq_lengths_key,
                "x4": block_table,
                },
        outputs=['y0']
    )
```


## 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/09/22 | 完成readme初始版本 |
