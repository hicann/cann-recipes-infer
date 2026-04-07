# NPU SANA-Video模型推理优化实践
本文档主要介绍SANA-Video模型基于NPU的推理优化策略。相关优化逻辑位于`models/sana-video/patches/npu_patches.py`。

## 1×1 conv2d优化
在`patch_conv1x1_matmul()`中，对`ConvLayer.forward`加入1×1 conv2d的判断与转换逻辑：

```python
if self.conv.kernel_size == (1, 1):
    B, Ci, H, W = x.shape
    Co, *_ = self.conv.weight.shape
    x_trans = x.view(B, Ci, -1).permute(0, 2, 1).contiguous()
    kernel_trans = self.conv.weight.view(Co, Ci)
    x = x_trans @ kernel_trans.T
    if self.conv.bias is not None:
        x += self.conv.bias
    x = x.permute(0, 2, 1).contiguous().view(B, Co, H, W)
else:
    x = self.conv(x)
```

## 时序conv2d的hw轴调换优化
在`patch_temporal_conv_swap()`中，对`GLUMBConvTemp`增加`t_conv_swap`并在`forward`中调用：

```python
self.t_conv_swap = nn.Conv2d(
    out_feature,
    out_feature,
    kernel_size=(1, t_kernel_size),
    stride=1,
    padding=(0, t_padding),
    bias=False,
)

x_reshaped = x.view(batch_size, time_steps, channels, height * width).permute(0, 2, 1, 3)
x_out = x_reshaped + self.t_conv_swap(x_reshaped.transpose(-1, -2)).transpose(-1, -2)
```

在`inference_video_scripts/inference_sana_video.py`加载模型参数后，再同步一次交换后的权重：

```python
for block in model.blocks:
    block.mlp.t_conv_swap.weight = nn.Parameter(block.mlp.t_conv.weight.permute(0, 1, 3, 2))
```

## npu_rms_norm算子适配
本样例使用`torch_npu`内置的`npu_rms_norm`融合算子替换原始小算子实现，`npu_rms_norm`详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md)。在`patch_rms_norm()`中使能`npu_rms_norm`融合算子：

```python
def forward(self, x):
    if _npu_available:
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
    return original_forward(self, x)
```

## npu_rotary_mul算子适配
本样例使用`torch_npu`内置的`npu_rotary_mul`融合算子替换原始 rotary 小算子实现，`npu_rotary_mul`详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_rotary_mul.md)。在`patch_rotary_mul()`中使能`npu_rotary_mul`融合算子：

```python
def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
    if _npu_available:
        cos = freqs.real.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(-2)
        sin = freqs.imag.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(-2)
        x_out = torch_npu.npu_rotary_mul(hidden_states.permute(0, 1, 3, 2), cos, sin, "interleave")
        return x_out.permute(0, 1, 3, 2)
    else:
        x_rotated = torch.view_as_complex(hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return x_out.type_as(hidden_states)

q_rotated = apply_rotary_emb(q, rotary_emb)
k_rotated = apply_rotary_emb(k, rotary_emb)
```

## npu_fusion_attention算子适配
本样例使用`torch_npu`内置的`npu_fusion_attention`融合算子替换原始 attention 小算子实现，`npu_fusion_attention`详细可见[Ascend社区文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fusion_attention.md)。在`patch_fusion_attention()`中使能`npu_fusion_attention`融合算子：

```python
if mask is not None and mask.ndim == 2:
    mask = (1 - mask.to(q.dtype)) * -10000.0
    mask = mask[:, None, None].repeat(1, self.num_heads, num_tokens, 1)

if _npu_available:
    bool_mask = None if mask is None else mask < -1000
    x = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        input_layout="BNSD",
        pse=None,
        atten_mask=bool_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        pre_tockens=2147483647,
        next_tockens=2147483647,
        keep_prob=1,
    )[0]
else:
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

x = x.transpose(1, 2).contiguous()
```
