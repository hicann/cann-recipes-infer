# custom.npu_swiglu_group_quant

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

`npu_swiglu_group_quant`在SwiGLU激活函数后执行低比特量化，支持Block FP8量化、MX FP8量化和MX FP4量化。输入`x`最后一维按左右两半拆分，左半部分记为`A`，右半部分记为`B`。

> MX FP4（`FLOAT4_E2M1` / `FLOAT4_E1M2`）量化仅在`quant_mode=1`（MX）下支持，详见下文「MX FP4 量化模式」章节。

### 计算公式

$$
Y_{tmp}=SwiGLU(x)=Swish(A) * B
$$

如果传入`clamp_limit`，在SwiGLU计算前对`A`和`B`执行如下截断：

$$
A=Min(A, clamp\_limit)
$$

$$
B=Min(Max(B, -clamp\_limit), clamp\_limit)
$$

如果传入`weight`，量化前按token乘到`Y_tmp`上：

$$
Y_{tmp}=Y_{tmp} * weight
$$

Block FP8量化模式下：

$$
scale\_out=block\_max(abs(Y_{tmp})) / dstTypeScale
$$

$$
Y=Cast(Y_{tmp} / scale\_out)
$$

MX FP8量化模式下，按每32个通道计算scale，并输出`float8_e8m0`格式scale：

$$
scale\_out=mx\_scale(abs(Y_{tmp}), round\_scale)
$$

$$
Y=Cast(Y_{tmp} / scale\_out)
$$

## 函数原型

```python
custom.npu_swiglu_group_quant(
    Tensor x,
    *,
    Tensor? weight=None,
    Tensor? group_index=None,
    ScalarType dst_type,
    int quant_mode=0,
    int block_size=0,
    bool round_scale=False,
    float? clamp_limit=None,
    bool output_origin=False,
    int dst_type_code=-1
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

> **说明：**
>
> - `T`表示除最后一维外所有维度合轴后的token数。
> - `D`表示输入`x`最后一维，SwiGlu后输出隐藏维为`H = D / 2`。
> - `G`表示`group_index`的元素个数。

- **x**（`Tensor`）：必选输入。SwiGLU的输入。不支持非连续，数据格式支持ND，数据类型支持`float16`和`bfloat16`。shape支持`[..., D]`，典型shape为`[b, s, D]`或`[T, D]`。

- **\***：代表其之前的参数是位置相关参数，必须按照顺序输入；其之后的参数是关键字参数。

- **weight**（`Tensor`，可选）：可选输入。量化前按token乘到SwiGLU输出上。不支持非连续，数据格式支持ND，数据类型支持`float32`。shape通常为`[T]`或`[T, 1]`，需要与`x`合轴后的token维匹配。

- **group_index**（`Tensor`，可选）：可选输入。表示各group中的token数量，当前按count模式使用。不支持非连续，数据格式支持ND，数据类型支持`int64`。shape通常为`[G]`。传入后，算子只保证前`sum(group_index)`个token对应输出有效。

- **dst_type**（`ScalarType`）：必选属性。量化输出`y`的数据类型，支持`torch.float8_e4m3fn`、`torch.float8_e5m2`。MX FP4 场景下若 torch ≥ 2.8 可传入`torch.float4_e2m1fn_x2`；torch < 2.8 无 fp4 ScalarType，请传入占位 ScalarType（如`torch.float8_e4m3fn`）并通过`dst_type_code`显式指定 fp4 子类型，详见「MX FP4 量化模式」。

- **quant_mode**（`int`，可选）：量化模式。默认值为`0`。
  - `0`：Block FP8量化，`scale_out`数据类型为`float32`。
  - `1`：MX FP8量化，`scale_out`数据类型为`float8_e8m0`。

- **block_size**（`int`，可选）：量化block大小。默认值为`0`，表示使用当前`quant_mode`对应默认值。
  - `quant_mode=0`时，仅支持`0`或`128`；`0`按`128`处理。
  - `quant_mode=1`时，仅支持`0`或`32`；`0`按`32`处理。

- **round_scale**（`bool`，可选）：是否对scale进行round处理。
  - `quant_mode=0`时仅支持`False`。
  - `quant_mode=1`时仅支持`True`。

- **clamp_limit**（`float`，可选）：SwiGLU计算前对输入进行截断的阈值。不传入时不执行clamp；传入时取值必须大于等于`0.0`，不支持NaN。

- **output_origin**（`bool`，可选）：是否在MX FP8量化模式下写出量化前的SwiGLU结果`y_origin`，默认值为`False`。`quant_mode=0`时该输出不作为有效结果使用。MX FP4 模式不支持`y_origin`语义（fp4 无量化前原值输出）。

- **dst_type_code**（`int`，可选）：量化输出`y`的目标数据类型对应的`ge::DataType`整型编码，默认值为`-1`，表示由`dst_type`（ScalarType）推导。当目标类型在当前 torch 版本无可用 ScalarType（fp4）时，通过该参数显式覆盖。取值：`FLOAT8_E5M2=35`、`FLOAT8_E4M3FN=36`、`FLOAT4_E2M1=40`、`FLOAT4_E1M2=41`。

## 返回值说明

- **y**（`Tensor`）：量化后的输出。数据格式支持ND，shape为`[..., H]`，数据类型与`dst_type`/`dst_type_code`一致。MX FP4 时`y`为 packed fp4（每 byte 2 个 nibble），末维为`H/2 = D/4`，torch<2.8 以`uint8`容器承载。

- **scale_out**（`Tensor`）：量化scale输出。数据格式支持ND。
  - `quant_mode=0`时，数据类型为`float32`，shape为`[..., ceil(H / 128)]`。
  - `quant_mode=1`时，数据类型为`float8_e8m0`，shape为`[..., ceil(ceil(H / 32) / 2), 2]`。

- **y_origin**（`Tensor`）：量化前的SwiGLU结果。数据格式支持ND，数据类型与`x`一致，shape与`y`一致。仅`quant_mode=1`且`output_origin=True`时结果有效；其他场景下该输出会分配但不保证写入有效业务数据。

## 约束说明

- 输入`x`约束：
  - rank必须大于0。
  - 最后一维`D`必须大于等于`256`，且必须能被`256`整除。
  - `D`会被均分为`A`和`B`两部分，因此SwiGLU输出最后一维为`H=D/2`。
  - 数据类型仅支持`float16`和`bfloat16`，格式仅支持ND。

- 属性取值约束：

  | 属性 | 支持取值 |
  |:--|:--|
  | `dst_type` | `torch.float8_e4m3fn`、`torch.float8_e5m2`；MX FP4 时为`torch.float4_e2m1fn_x2`(torch≥2.8) 或占位 ScalarType + `dst_type_code`(torch<2.8) |
  | `dst_type_code` | `-1`(由 dst_type 推导)、`35`(E5M2)、`36`(E4M3FN)、`40`(FLOAT4_E2M1)、`41`(FLOAT4_E1M2) |
  | `quant_mode` | `0`、`1`；fp4(`dst_type_code` 40/41) 仅支持`1` |
  | `block_size` | `quant_mode=0`时为`0`或`128`；`quant_mode=1`时为`0`或`32` |
  | `round_scale` | `quant_mode=0`时必须为`False`；`quant_mode=1`时必须为`True` |
  | `clamp_limit` | 不传入，或传入大于等于`0.0`的有限/无穷浮点值；不支持NaN |

- 输出dtype约束：
  - `y`数据类型必须与`dst_type`一致。
  - `scale_out`在`quant_mode=0`时必须为`float32`，在`quant_mode=1`时必须为`float8_e8m0`。
  - `y_origin`数据类型必须与`x`一致。

- `group_index`约束：
  - 仅支持count模式，表示每个group的token数量。
  - 需要保证`sum(group_index) <= T`，否则可能访问超出`x`有效token范围。
  - 传入`group_index`后，输出中超过`sum(group_index)`对应token范围的部分不保证为有效业务数据。

- 其他约束：
  - 该接口支持推理场景下使用。
  - 该接口支持aclgraph入图。
  - 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## MX FP4 量化模式（FLOAT4_E2M1 / FLOAT4_E1M2）

MX FP4 在 SwiGLU 后按每 32 个通道（mx block）共享一个`float8_e8m0`指数 scale，并将结果量化为 4-bit 浮点。两种 fp4 子类型：

| 子类型 | `dst_type_code` | 编码（sign·exp·mantissa） | 可表示幅值（含 round/饱和） |
|:--|:--:|:--|:--|
| `FLOAT4_E2M1` | `40` | 1·2·1，max=6.0 | 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 |
| `FLOAT4_E1M2` | `41` | 1·1·2，max=1.75 | 0, ±0.25, ±0.5, ±0.75, ±1.0, ±1.25, ±1.5, ±1.75（步长 0.25，饱和） |

- **使用约束**：fp4 仅在`quant_mode=1`（MX）、`block_size=0或32`、`round_scale=True`下支持。
- **输出`y`（packed fp4）**：每个 byte 打包 2 个 fp4 值（低 nibble 在前），输出末维为`splitD/2 = D/4`。
  - torch ≥ 2.8：`y`使用原生`torch.float4_e2m1fn_x2`（packed）dtype。
  - torch < 2.8：无 fp4 ScalarType，`y`以`uint8`容器承载 packed 字节；接入层通过`TensorWrapper`把 aclTensor 标记为`ACL_FLOAT4_E2M1`/`ACL_FLOAT4_E1M2`（即`dst_type_code` 40/41），底层 kernel/tiling 仍由`dst_type`属性驱动。读取时用`y.view(torch.uint8)`再按需解 nibble。
- **`scale_out`（e8m0）**：数据类型`float8_e8m0`，与 MX FP8 模式相同的 shape 约定（`[..., ceil(ceil(H/32)/2), 2]`）。e8m0 偏置指数 = `max(E_amax - f4Emax_exp, 0)`，其中`f4Emax_exp`：E2M1 为 2，E1M2 为 0。
- **`y_origin`**：MX FP4 不支持量化前原值输出（无 fp4 `y_origin`语义）。

### 已知限制 / 注意事项

- `D`必须能被`256`整除（与其他模式一致）；fp4 packed 末维`D/4`与 e8m0 scale 列数`splitD/32`在该约束下天然满足对齐与打包要求。
- 非对齐 / 小`d`场景：当`scaleDFactor = splitD/32`非 16 对齐（如`D=256`→4、`D=512`→8）且单核多行全载（`rowFactor>1`）时，e8m0 scale 的搬出需走`PaddingMode::Compact`通路；当前实现已按此处理，已覆盖`D=256/512`、`bs=64/512`用例验证逐行 scale 正确。
- 多 d-loop（大`D`单行无法全载，`dLoop>1`）场景：tiling 从最小 mx block 递增选取`dFactor`，并保持`scaleCol`为整行宽度，使每个 d-chunk 的 e8m0 scale 落到正确 GM 偏移；已用`D=49152`（splitD=24576，dLoop>1）验证 y 与 scale 正确。
- e1m2 无 `ml_dtypes` 对应 dtype，golden 采用按 kernel 反推的均匀网格编码（见 examples 测试）。fp4 比对采用 nibble/packed-byte 匹配率阈值（>0.99），残差来自 bf16 中间计算在 block 边界的舍入，与 e2m1 同量级。

## 调用示例

```python
import torch
import torch_npu
import custom_ops

x = torch.randn(1, 128, 4096, device="npu", dtype=torch.float16)

# Block FP8 quant
y, scale, y_origin = torch.ops._ascend_dsv4.npu_swiglu_group_quant(
    x,
    dst_type=torch.float8_e5m2,
    quant_mode=0,
    block_size=0,
    round_scale=False,
)

# MX FP8 quant
y_mx, scale_mx, y_origin_mx = torch.ops._ascend_dsv4.npu_swiglu_group_quant(
    x,
    dst_type=torch.float8_e4m3fn,
    quant_mode=1,
    block_size=32,
    round_scale=True,
    output_origin=True,
)

# MX FP4 quant (torch < 2.8: uint8 container + dst_type_code 选择 fp4 子类型)
# E2M1=40, E1M2=41
y_fp4, scale_fp4, _ = torch.ops.custom.npu_swiglu_group_quant(
    x,
    dst_type=torch.float8_e4m3fn,   # 占位 ScalarType
    quant_mode=1,
    block_size=32,
    round_scale=True,
    dst_type_code=40,               # FLOAT4_E2M1
)
y_packed = y_fp4.view(torch.uint8)  # packed fp4，末维 D/4，每 byte 两个 nibble
```

- 详见[test_npu_swiglu_group_quant.py](../examples/test_npu_swiglu_group_quant.py)
