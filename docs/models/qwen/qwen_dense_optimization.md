# 基于Atlas A2/A3的Qwen Dense模型推理性能优化实践
## 概述
本文主要介绍Qwen2/Qwen3 Dense（非MoE）模型基于NPU的推理优化策略。基于Atlas A2/A3 训练/推理系列产品，实现BF16场景下的推理。

## Tensor Parallel (TP)优化
### Attention TP优化
#### 切分策略
对Attention的张量切分策略可以分为对QKV头的切分和对线性层的切分。

在对QKV头切分时，attention的多头计算机制可以方便进行张量切分，每个头先独立计算，再将结果concat起来。假设模型的attention层需要对`num_heads`个query按照切分数量`attn_tp_size`进行切分，要求`num_heads`必须能被`attn_tp_size`整除，每张卡放置query头个数为`num_heads_per_rank = num_heads // attn_tp_size`；key和value头数相等，且可能小于等于query头个数（在GQA场景下会小于）。为了确保每张卡至少放置一个key和value头，每张卡放置的key或value头数计算方法为
`num_key_value_heads_per_rank = max(num_key_value_heads // attn_tp_size, 1)`。

在对线性层`o_proj`进行切分时，按照行切分即可。

#### 计算分解
该优化策略先将Q、K、V的线性层计算合并为一次Matmul计算（merged_qkv_proj），提升计算性能。将`merged_qkv_proj`的输出结果按Q、K、V拆分后，对Q和K进行归一化操作（Qwen3启用QK-Norm，Qwen2无此步骤）并使用旋转位置编码，再计算attention，最后通过o_proj层输出。

### MLP TP优化
#### 切分策略
对MLP层的`gate_proj`与`up_proj`进行列切分，对`down_proj`进行行切分。同时对`gate_proj`与`up_proj`线性层采用合并计算的优化方式，得到`gate_up_proj`。

#### 计算分解
MLP层存在gate_proj、up_proj与down_proj三个matmul运算，具体运算为 x = down( SiLU(gate(x)) * up(x) )。本优化将张量切分后的gate_proj和up_proj合并为gate_up_proj一次Matmul计算，将输出按中间维度切分为两块后，分别执行SiLU激活和element-wise乘法，最终通过down_proj输出。

通过将gate_proj与up_proj合并计算，减少kernel launch次数，提升整体计算效率。

## 残差融合优化
在原始transformers实现中，每个DecoderLayer的残差连接和RMSNorm是分开执行的：
```
residual = hidden_states
hidden_states = RMSNorm(hidden_states)
hidden_states = Attention(hidden_states)
hidden_states = residual + hidden_states        ← 独立residual add
```

本优化通过使能[torch_npu.npu_add_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000140.html)融合算子，将残差加法与RMSNorm合并为一次kernel调用：
```
hidden_states, residual = npu_add_rms_norm(residual, hidden_states, weight, eps)
```

每层DecoderLayer包含两处残差连接（Attention前和MLP前），融合后每层减少两次独立的tensor加法算子及对应的显存读写开销。残差（residual）作为参数在层间传递，不再在每层内部做独立的add操作。

## 使能融合算子
### RmsNorm算子优化
通过使能[torch_npu.npu_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000140.html)算子，替代原始手动计算variance + rsqrt的实现，提升模型推理性能。

### RoPE算子优化
通过使能[torch_npu.npu_apply_rotary_pos_emb](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000140.html)融合算子，替代原始`rotate_half` + 手动乘加的实现。同时对cos/sin进行预计算并缓存，避免每次forward动态计算。

### FlashAttention融合算子优化
通过使能[torch.ops.npu.npu_fused_infer_attention_score_v2](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00086.html)推理场景下的FlashAttention融合算子，既可以支持全量计算场景，也可支持增量计算场景。

## 使能图模式
使用静态图可以获得更好的推理性能。通过覆写`executor/model_runner.py`中的`ModelRunner`的`graph_compile`函数，将模型编译为静态图。

## W8A8 INT8 量化

Qwen3-8B 支持 W8A8 INT8 量化，其中权重采用 per-channel 静态量化，激活采用 per-token 动态量化。量化范围覆盖 Attention 的 `merged_qkv_proj` / `o_proj` 与 MLP 的 `gate_up_proj` / `down_proj`，共 4 个 Linear 层。

启用量化后，MLP 计算切换至专用的融合算子路径：`gate_up_proj` 输出 int32 张量与 per-token scale，并作为 [torch_npu.npu_dequant_swiglu_quant](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_dequant_swiglu_quant.md) 的输入，由该算子一次性完成反量化、SwiGLU 激活与再量化操作；输出的 int8 张量与对应 scale 由 `down_proj` 直接接收并执行后续矩阵乘运算。该实现可使 ge_graph 静态图模式下命中现有融合算子实现，避免因 `AddRmsNormDynamicQuant` 等组合算子尚未提供而触发的回退路径。

量化后的模型权重以 compressed-tensors 格式加载。量化相关代码逻辑由 `mm_quant_mode` 字段控制：仅当字段值为 `w8a8int8` 时启用 W8A8 量化路径，否则保持原 BF16 推理路径不变。

## 集合通信使能AIV展开
利用Device的Vector Core计算单元来加速通信操作的执行，可参考[HCCL_OP_EXPANSION_MODE环境变量](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0096.html)：

```shell
export HCCL_OP_EXPANSION_MODE=AIV
```

## 附录
[环境部署以及样例执行](../../../models/qwen/README.md)
