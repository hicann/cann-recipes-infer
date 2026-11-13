# NPU daVinci-MagiHuman模型推理优化实践

daVinci-MagiHuman 模型是一款多模态视频生成模型，提供了文生视频、图文生视频功能。本文档主要介绍 daVinci-MagiHuman 模型基于NPU的推理适配、优化，在昇腾Atlas A2/A3环境上做了验证。

##  NPU 全局适配

在脚本入口，即`models/davinci-magihuman/sample_video.py`，添加如下代码：

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## NPU 导入torch_npu包

在有调用到torch_npu接口（比如`torch_npu.npu_fused_infer_attention_score`、`torch_npu.npu_rotary_mul`等）的Python文件头部，导入torch_npu包：

```python
import torch
# 增加
import torch_npu
```

## NPU npu_fused_infer_attention_score算子适配

本样例使用torch_npu内置的npu_fused_infer_attention_score融合算子替代FlashAttention算子，该算子详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md)。

在`models/davinci-magihuman/inference/model/vae2_2/vae2_2_module.py`、`models/davinci-magihuman/inference/model/dit/dit_module.py`使用了npu_fused_infer_attention_score算子。

以`models/davinci-magihuman/inference/model/dit/dit_module.py`中的`flash_attn_func`函数示例说明调用如下：

```python
def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    b, s, n, d = query.shape
    _, _, kv_n, _ = key.shape
    scale = 1.0 / (d ** 0.5)

    original_dtype = query.dtype

    query = query.to(torch.bfloat16).contiguous()
    key = key.to(torch.bfloat16).contiguous()
    value = value.to(torch.bfloat16).contiguous()

    output, _ = torch_npu.npu_fused_infer_attention_score(
        query, key, value,
        num_heads=n,
        num_key_value_heads=kv_n,
        scale=scale,
        input_layout="BSND",
        sparse_mode=0
    )
    return output.to(original_dtype)
```

## NPU npu_rotary_mul算子适配

本样例使用torch_npu内置的npu_rotary_mul融合算子替换源代码中的小算子实现，npu_rotary_mul详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md)。

在`models/davinci-magihuman/inference/model/dit/dit_module.py`使用了npu_rotary_mul算子。

以`models/davinci-magihuman/inference/model/dit/dit_module.py`中的`apply_rotary_emb_npu`函数示例说明调用如下：

```python
def apply_rotary_emb_npu(x, cos, sin):
    """
    Apply rotary embedding using npu_rotary_mul with pre-expanded cos/sin.
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (1, seqlen, 1, rotary_dim) - pre-expanded via cat([d, d], dim=-1)
    """
    ro_dim = cos.shape[-1]
    if ro_dim == x.shape[-1]:
        return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode='half')
    x_rot = torch_npu.npu_rotary_mul(x[..., :ro_dim], cos, sin, rotary_mode='half')
    return torch.cat([x_rot, x[..., ro_dim:]], dim=-1)
```

## NPU npu_rms_norm算子适配

本样例使用torch_npu内置的npu_rms_norm融合算子替换源代码中的小算子实现。npu_rms_norm详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md)。

在`models/davinci-magihuman/inference/model/dit/dit_module.py`使用了npu_rms_norm算子。

以`models/davinci-magihuman/inference/model/dit/dit_module.py`中的`forward_single_expert`函数示例说明调用如下：

```python
    def forward_single_expert(self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None, residual: Optional[torch.Tensor] = None):
        gamma = (self.weight + 1).contiguous()
        if residual is not None:
            gamma = gamma.to(torch.bfloat16)
            out, _, new_residual = torch_npu.npu_add_rms_norm(
                residual.to(gamma.dtype), x.to(gamma.dtype), gamma, self.eps)
            return out, new_residual
        out = torch_npu.npu_rms_norm(x, gamma, self.eps)[0]
        return out
```

## NPU npu_clipped_swiglu算子适配

本样例使用torch_npu内置的npu_clipped_swiglu融合算子替换源代码中的小算子实现。npu_clipped_swiglu详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_clipped_swiglu.md)。

在`models/davinci-magihuman/inference/model/dit/dit_module.py`使用了npu_clipped_swiglu算子。

以`models/davinci-magihuman/inference/model/dit/dit_module.py`中的`swiglu7`函数示例说明调用如下：

```python
def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x_f32 = x.to(torch.float32).contiguous()
    out = torch_npu.npu_clipped_swiglu(x_f32, alpha=alpha, limit=limit, bias=1.0, interleaved=True)
    return out.to(out_dtype)
```

## NPU npu_fast_gelu算子适配

本样例使用torch_npu内置的npu_fast_gelu融合算子替换源代码中的小算子实现。npu_fast_gelu详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fast_gelu.md)。

在`models/davinci-magihuman/inference/model/dit/dit_module.py`使用了npu_fast_gelu算子。

以`models/davinci-magihuman/inference/model/dit/dit_module.py`中的`gelu7`函数示例说明调用如下：

```python
def gelu7(x, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x.clamp_(max=limit)
    x = torch_npu.npu_fast_gelu(x)
    return x.to(out_dtype)
```

## NPU npu_add_rms_norm算子适配

本样例使用torch_npu内置的npu_add_rms_norm融合算子替换源代码中的小算子实现。npu_add_rms_norm详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md)。

在`models/davinci-magihuman/inference/model/dit/dit_module.py`使用了npu_add_rms_norm算子。

以`models/davinci-magihuman/inference/model/dit/dit_module.py`中的`forward_single_expert`函数示例说明调用如下：

```python
    def forward_single_expert(self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None, residual: Optional[torch.Tensor] = None):
        gamma = (self.weight + 1).contiguous()
        if residual is not None:
            gamma = gamma.to(torch.bfloat16)
            out, _, new_residual = torch_npu.npu_add_rms_norm(
                residual.to(gamma.dtype), x.to(gamma.dtype), gamma, self.eps)
            return out, new_residual
        out = torch_npu.npu_rms_norm(x, gamma, self.eps)[0]
        return out
```

## Offload优化

daVinci-MagiHuman开源代码是在H100 GPU（显存80GB）设备上运行的，但是昇腾Atlas A2/A3 NPU单卡显存只有64GB，如果不做offload的话运行sr_540p/sr_540p_ti2v/sr_1080p/sr_1080p_ti2v模式会OOM（out of memory），因此做了offload优化。

首先修改了`models/davinci-magihuman/inference/common/cpu_offload_wrapper.py`中`CPUOffloadWrapper`类的实现，增加了`_save_cpu_copies`、`prepare_resident_backup`、`disable_offload`、`enable_offload`函数，修改了`_run_with_optional_offload`函数。

其次，在`models/davinci-magihuman/inference/pipeline/video_generate.py`中调用`CPUOffloadWrapper`类的`prepare_resident_backup`、`disable_offload`、`enable_offload`函数。

详细改动可以对比本仓库与开源代码仓这2个文件的差异。

## FA MXFP8 动态量化（Ascend 950）

daVinci-MagiHuman 的 DiT 注意力支持通过配置开关切换到 MXFP8 动态量化执行：Q/K/V 经 `module/fa_quant` 的 `npu_mxfp8_quant` 动态量化为 FP8 E4M3（scale 为 e8m0，逐 token / 每 64-token 组动态产出），注意力经 `npu_fused_infer_attention_score_v2` 以 MXFP8 quant mode 计算，输出恢复原 dtype。该能力依赖 **CANN ≥ 9.1.0** 与 Ascend 950（950PR 实测通过）。支持范围以实测为准，见下表：

| 注意力路径 | fa-mxfp8 | 行为 |
|---|---|---|
| dense `flash_attn_func`（含 GQA） | 支持（950PR 实测） | MXFP8 量化执行（FA v2 quant mode） |
| `flex_flash_attn_func`（局部/变长） | 不支持（无 MXFP8 实现） | 回退 BF16（审计日志）或 fail-fast |
| context-parallel（cp_size>1） | 不支持（未验证） | 同上 |

### 启用方式

一键入口：`bash infer.sh fa_mxfp8`。该模式复用 `config/base.yaml`，由 `infer.sh` 在生成 runtime yaml 时注入以下开关（无独立 YAML）：

```yaml
model_args:
  fa-mxfp8: true
  fa-mxfp8-fallback: true
  output_path: "output_fa_mxfp8"
```

- `fa-mxfp8`（默认 `false`）：开启后 DiT 注意力切换 MXFP8 路径；`false` 时保持 BF16 `npu_fused_infer_attention_score` 路径，行为不变；
- `fa-mxfp8-fallback`（默认 `false`）：遇不支持的组合（`flex_flash_attn_func` 局部/变长、context-parallel>1、head_dim 非 32 倍数、非 NPU 张量、CANN 版本过低等）时置 `true` 自动回退 BF16 并输出 `[FA_MXFP8] fallback: ...` 审计日志（经仓库统一 logging 输出，级别由 `CANN_RECIPES_LOG_LEVEL` 控制）；`false`（默认）时 fail-fast 硬报错。`bash infer.sh fa_mxfp8` 的注入值为 `true`。

### 实现要点

- 实现收敛在公共模块 `module/fa_quant`：`npu_mxfp8_quant` / `npu_mxfp8_attn` 为通用量化与 FA v2 封装，`npu_mxfp8_dense_attn` 为带守卫的 dense 封装（不支持时抛 `FaMxfp8Unsupported`）；DiT 侧注册算子 `infra::flash_attn_func_mxfp8` 复用之，仅保留模型侧策略（首层 K 中心化、context-parallel 世界大小获取）；
- 量化策略：`scale_alg=1`（scale 指数向上取整）全局生效；仅首层 K 中心化为模型侧精度策略（量化前减去 K 沿序列维的逐头均值，softmax 对 key 维常数平移不变）；
- 精度验证：BF16/MXFP8 同 seed 双跑 dump 对比（FA 输出 cos 0.99967，DiT block0 rel 2.43%），方法与材料见 #490。
