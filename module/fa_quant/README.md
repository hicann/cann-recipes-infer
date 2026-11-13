# fa_quant — NPU MXFP8 动态量化 Flash Attention 公共模块

本模块提供 Q/K/V 动态 MXFP8 量化 + `npu_fused_infer_attention_score_v2`（FA v2）封装，用于在 Ascend 950 上以 MXFP8 精度执行 Flash Attention（950PR 实测通过；MXFP8 quant mode 依赖 CANN ≥ 9.1.0，其他型号暂未验证），降低注意力算子访存与计算开销。

## 环境要求

- **CANN ≥ 9.1.0**。
- torch_npu：Ascend 950 实测 `v26.0.0-pytorch2.10.0`（torch_npu 2.10.0）与 `v26.0.0-pytorch2.7.1`（2.7.1.post4）均可用。

## 接口

### npu_mxfp8_attn(q, k, v, dst_type, softmax_scale)

Dense attention 快捷入口。输入 `(B, S, N, D)` BSND（支持 GQA：`num_key_value_heads = k.shape[2]`）。内部完成动态量化（Q/K 逐 token 组、V 每 64-token 组）并调用 FA v2（`input_layout="TND"`、`query/key_quant_mode=6`、`value_quant_mode=8`、dequant scale dtype `float8_e8m0fnu`），输出恢复输入 dtype。另含 keyword-only 参数 `scale_alg`（scale 指数取整策略，0=floor 历史默认 / 1=向上取整，见使用约束 5），默认值保持历史行为。调用方：HunyuanVideo 等。

### npu_mxfp8_dense_attn(query, key, value, softmax_scale, *, scale_alg, cp_world_size)

带守卫的 dense 封装：先检查支持条件（NPU 张量、`head_dim` 为 32 的倍数、`cp_world_size ≤ 1`），再将 Q/K/V 转 bfloat16 后调用 `npu_mxfp8_attn`。不支持组合与 kernel 拒绝统一抛 `FaMxfp8Unsupported`（携带原因字符串），回退/fail-fast 策略与审计日志由调用方决定；`cp_world_size` 由调用方传入（分布式上下文属于模型侧状态）。

## 使用约束（Ascend 950PR + CANN 9.1.0 实测）

1. 禁止「量化后立即反量化再调 BF16 FA」——量化收益来自 FA v2 的 fp8 计算；
2. scale 由 `npu_dynamic_mx_quant` 动态产出（e8m0），禁止伪造/全 1 scale；
3. head_dim 须为 32 的倍数；
4. V 的 64-token 组 scale 不支持逐 token 模式（value_quant_mode 1-6 实测全拒绝）；
5. `scale_alg=1` 表示 scale 指数向上取整（保证块内数值不超出 FP8 可表示范围；默认 0=floor 保持历史行为）。

## 验证参考

Ascend 950PR 实测（daVinci-MagiHuman base，GQA 40:8 / head_dim 128）：FA 单算子 1.14×~1.44×（S=4025→16384 随序列增大，量化开销 <0.1 ms/call），denoise 阶段 **-4.40%**、整网 **-4.24%**（448×256，950PR）；详见 `models/davinci-magihuman` 优化说明。
