# custom-npu_kv_compress_epilog_v2

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

将输入 x 按 quant_group_size 分组进行 MXFP8 或 packed MXFP4 量化，量化数据与 BF16 类型的 scale 拼接后按 32 字节向上对齐，根据 slot_mapping 中的下标映射（-1 表示无效数据，不进行更新），原地更新到 cache 对应的行上。MXFP8 固定为 32 个元素一组，MXFP4 支持 16 或 32 个元素一组。与 [custom-npu_kv_compress_epilog](./custom-npu_kv_compress_epilog.md) 相比，x 的全部 d 个元素均参与量化（不再拆分 rope/nope 分区），并新增 packed MXFP4 模式。

### cache 行布局

记 $G = d / quant\_group\_size$，每行 cache 布局为：`[量化数据区 | BF16 scale 区 | 0 填充至 32 字节对齐]`：

$$
dataCol = d\ (quant\_mode=2)\ 或\ d/2\ (quant\_mode=4)
$$

$$
concatCol = dataCol + 2G
$$

$$
kvCacheCol = \lceil concatCol / 32 \rceil \times 32
$$

| quant_mode | 量化格式 | cache 数据类型 | dataCol | scale 区 |
|:---:|:---|:---|:---:|:---|
| 2 | MXFP8（E4M3FN/E5M2） | `float8_e4m3fn` / `float8_e5m2` | d | G 个 BF16 |
| 4 | packed MXFP4（E2M1，两个 nibble 打包为 1 字节） | `uint8` | d/2 | G 个 BF16 |

MXFP8 模式下 scale 为每组 amax 与量化上限（E4M3FN 为 448、E5M2 为 57344）之比（round_scale=True 时向上取整为 2 的幂）；MXFP4 模式下 scale 由组内最大指数按共享指数方式生成。scale 以 BF16 位模式写入行内 `dataCol + 2g` 偏移处（g 为组下标）。

## 函数原型

Torch 接口（quant_mode 为字符串，大小写不敏感、允许首尾空白）：
```
custom.kv_compress_epilog_v2(Tensor(a!) cache, Tensor x, Tensor slot_mapping, *, int quant_group_size = 32, str quant_mode = "mxfp8_bf16", bool round_scale = True, float x_scale = 1.0) -> ()
```

GE/ACLNN 层 quant_mode 为 int，取值 2（mxfp8_bf16）或 4（mxfp4_bf16），Torch 接口在内部完成字符串到 int 的映射。ACLNN 两段式 API 原型：
```
aclnnStatus aclnnKvCompressEpilogV2GetWorkspaceSize(
    aclTensor *cacheRef, const aclTensor *x, const aclTensor *slotMapping,
    int64_t quantGroupSize, int64_t quantMode, bool roundScale, double xScale,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnKvCompressEpilogV2(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

## 参数说明

>**说明：**<br>
>
>- T 表示输入 token 数，d 表示每个 token 的尾轴维度，N 表示 cache 的行数。

-   **cache**（`Tensor`）：必选参数，输入tensor，待更新的cache数据，量化结果原地更新到此Tensor。不支持非连续，数据格式支持ND，shape为[ N, kvCacheCol ]（kvCacheCol 计算公式见 cache 行布局，允许大于 kvCacheCol 的加宽行）。数据类型：quant_mode 为 mxfp8_bf16 时支持`float8_e4m3fn`和`float8_e5m2`；quant_mode 为 mxfp4_bf16 时为`uint8`（承载 packed E2M1 数据与 BF16 scale）。

-   **x**（`Tensor`）：必选参数，输入tensor，待量化的输入数据。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[ T, d ]。

-   **slot_mapping**（`Tensor`）：必选参数，输入tensor，表示下标映射，其值为cache的行下标。不支持非连续，数据格式支持ND，数据类型支持`int32`和`int64`，shape为[ T ]，取值范围为-1或[ 0, N )内的合法行下标。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **quant_group_size**（`int`，可选）：quant_group 的大小，默认为 32。MXFP8 仅支持 32，MXFP4 支持 16 或 32。

-   **quant_mode**（`str`，可选）：量化模式，取值为`mxfp8_bf16`（MXFP8，默认）或`mxfp4_bf16`（packed MXFP4）。仅接受这两个字符串（大小写不敏感、允许首尾空白），裸名（如`mxfp4`）、空串和 int 入参均会被拒绝；底层 GE/ACLNN 层映射为 int 2/4。

-   **round_scale**（`bool`, 可选）：是否对 scale 做 2 的幂向上取整，默认为True，仅 MXFP8 模式生效。

-   **x_scale**（`float`，可选）：预留参数，取值固定为1.0。

## 返回值说明

无返回值，cache 做原地更新操作（cache 同时是输入与输出）。

## 约束说明
-  shape 字段取值范围约束
    | 字段名       | 取值规则与说明                                                                 |
    |--------------|-------------------------------------------|
    | d           |  取值需大于 0、能被 quant_group_size 整除且不超过 8192                                  |
    | kvCacheCol  |  按 cache 行布局公式计算，cache 第二维不小于该值                                          |
    | T           |  取值需大于 0，等于 slot_mapping 的长度                                                 |
- quant_mode/quant_group_size 只支持 `(mxfp8_bf16, 32)`、`(mxfp4_bf16, 32)` 和 `(mxfp4_bf16, 16)`；x_scale 固定为 1.0。
- quant_mode 与 cache 数据类型的组合见参数说明，不匹配时报错。
- slot_mapping 取值为 -1 时跳过该行；越界下标（大于等于 N 或小于 -1）静默跳过，不更新 cache 且无副作用；多个有效行写同一 slot 时跨核写回顺序不保证。
- `(mxfp8_bf16, 32)` 的 TilingKey 为 2000，`(mxfp4_bf16, 32)` 为 2001，`(mxfp4_bf16, 16)` 为 2002。
- Torch NPU 与 Meta 路径共用接口校验：rank、x/slot shape、dtype、连续性、字符串 mode/groupSize 组合、d 整除和 x_scale 在调用 ACLNN 或构图前检查；Torch 层不校验 cache size，最小 cache 行宽由 host tiling 检查。
- TorchAir converter 在生成 `KvCompressEpilogV2` 节点前再次校验 mode/groupSize 组合，并将字符串 quant_mode 映射为 GE int 属性。
- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 调用示例

- Torch eager、NPU/Meta/converter 接口校验与 torchair ACL Graph 图模式：详见 [test_npu_kv_compress_epilog_v2.py](../examples/test_npu_kv_compress_epilog_v2.py)
- ACLNN 两段式调用：详见 [test_aclnn_kv_compress_epilog_v2.cpp](../examples/test_aclnn_kv_compress_epilog_v2.cpp)
- GEIR 图模式调用：详见 [test_geir_kv_compress_epilog_v2.cpp](../examples/test_geir_kv_compress_epilog_v2.cpp)
