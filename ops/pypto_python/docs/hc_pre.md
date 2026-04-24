# hc_pre

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Ascend 950PR/Ascend 950DT 推理系列产品</term>   | √  |

## 功能说明

- API功能：hc_pre算子旨在完成以下计算过程。
- 计算过程：

1. 计算 RMSNorm 的分母

$$
rsqrt = \sqrt{\frac{1}{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}}
$$

2. 计算 mixes

$$
mixes = (x @ hc\_fn) \odot rsqrt
$$

3. Sinkhorn-Knopp 算法

$$
pre, post, comb = sinkhorn(mixes, hc\_scale, hc\_base, hc\_mult, hc\_sinkhorn\_iters)
$$

Sinkhorn-Knopp 算法每次迭代会进行逐行归一化，再做逐列归一化，hc_sinkhorn_iters 控制迭代次数。

4. 利用 pre 和 x 计算 y

$$
y = rowsum(pre \odot x)
$$

## 函数原型

```
torch.ops.pypto.hc_pre(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult: int=4,
    hc_split_sinkhorn_iters: int=20,
    hc_eps: float=1e-6
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，对应公式中的$x$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_x`为TND时shape为[t, hc_mult, h]。
- **hc_fn**（`Tensor`）：必选参数，对应公式中的$hc\_fn$，不支持非连续，数据格式支持ND，数据类型支持`float32`，`layout_x`为TND时shape为[mix_hc, hc_mult*h]，其中mix_hc = (2+hc_mult)*hc_mult。
- **hc_scale**（`Tensor`）：必选参数，对应公式中的$hc\_scale$，不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[3, ]。
- **hc_base**（`Tensor`）：对应公式中的$hc\_base$，不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[mix_hc, ]。
- **hc_mult**（`int`）：可选参数，表示mHC中的expansion rate，数据类型支持`int`，默认值为`4`。
- **hc_split_sinkhorn_iters**（`int`）：可选参数，表示sinkhornde 迭代次数，数据类型支持`int`， 默认值`20`。
- **hc_eps**（`float`）：可选参数，表示RMSNorm分母计算与Sinkhorn-Knopp计算中用于数值稳定的加法值，数据类型支持`float`， 默认值为`1e-6`。

## 返回值说明

- **y**（`Tensor`）：公式中的输出。数据格式为ND，数据类型为`bfloat16`。当layout\_x为TND时shape为[t, h]。
- **post**（`Tensor`）：公式中sinkhorn的输出post，数据格式为ND，数据类型为`float`。当layout\_x为TND时shape为[t, hc_mult]。
- **comb**（`Tensor`）：公式中sinkhorn的输出comb，数据格式为ND，数据类型为`float`。当layout\_x为TND时shape为[t, hc_mult, hc_mult]。

## 约束说明

- 该接口支持推理场景下使用。
- 入参x中的shape [t, hc_mult, h]中，h仅支持`4096`。
- 入参的shape、dtype等需与参数说明保持一致。
- t的值域范围为[1, 64k]

## 调用方法

```
python3 ops/pypto_python/example/test_hc_pre_pypto.py

```
