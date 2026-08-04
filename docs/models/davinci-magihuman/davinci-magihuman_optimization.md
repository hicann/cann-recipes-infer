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
