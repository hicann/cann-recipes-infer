# 说明

本示例对 DeepSeek 模型进行 TileLang-ascend 算子替换与 inductor + [autofuse](https://gitcode.com/cann/ge/tree/master/compiler/graph/optimize/autofuse) 融合编译，展示其在昇腾 A3 NPU 上的加速效果。

- TileLang-ascend 仓库地址：[tilelang-ascend](https://github.com/tile-ai/tilelang-ascend/)
- Inductor 对接 autofuse 的参考实现：[inductor_npu_ext](https://gitcode.com/Ascend/torchair/blob/master/experimental/_inductor_npu_ext/)

# 软件安装

## 获取 docker 基础镜像

点击 [镜像下载连接](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann8.5/ds_cann_nightly_aarch_pta2.8_image_v1_1.tar)下载镜像，下载完成后执行如下命令导入镜像：

```
docker load -i ds_cann_nightly_aarch_pta2.8_image_v1_1.tar
```

> 注意，截至文档编写时，inductor + autofuse 依赖 CANN 社区 nightly 版本，所需依赖已打包至镜像，请确保从镜像启动。


## 拉起 docker 容器

执行如下命令拉起 docker 容器，示例使用的容器名为 `tilelang_and_inductor`。
```
docker run -u root -itd --name tilelang_and_inductor --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci8     --device=/dev/davinci9 \
    --device=/dev/davinci10    --device=/dev/davinci11 \
    --device=/dev/davinci12    --device=/dev/davinci13 \
    --device=/dev/davinci14    --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home/:/home \
    -v /data:/data \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
    -v /usr/bin/hostname:/usr/bin/hostname \
    --net=host \
    --shm-size=128g \
    --privileged \
    ds_cann_nightly_aarch_pta2.8_image_v1_1:v1.1 /bin/bash
```
## 进入容器

```
docker attach tilelang_and_inductor
```

# 收益复现

## 典型模型片段

以下示例为 DeepSeek 模型中的 hc_post 片段，测试数据基于模型执行时的典型输入构造。

### 运行示例

```python
import torch
import torch_npu
import inductor_npu_ext # 导入 inductor_npu_ext 以注册基于 ascendc 的 inductor 后端扩展

def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb:torch.Tensor) -> torch.Tensor:
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)

# 添加 @torch.compile 装饰以启用 inductor 编译
@torch.compile
def hc_post_compiled(x, residual, post, comb):
    return hc_post(x, residual, post, comb)

@torch.compile(dynamic=True)
def hc_post_compiled_dynamic(x, residual, post, comb):
    return hc_post(x, residual, post, comb)

x = torch.randn(1, 1, 4096, dtype=torch.bfloat16, device='npu')
residual = torch.randn(1, 1, 4, 4096, dtype=torch.bfloat16, device='npu')
post = torch.randn(1, 1, 4, dtype=torch.float32, device='npu')
comb = torch.randn(1, 1, 4, 4, dtype=torch.float32, device='npu')

hc_post(x, residual, post, comb)
hc_post_compiled(x, residual, post, comb)
hc_post_compiled_dynamic(x, residual, post, comb)

experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
)

for func in [hc_post_compiled, hc_post, hc_post_compiled_dynamic]:
    test_datas = []
    for i in range(10):
        test_datas.append([v.clone() for v in (x, residual, post, comb)]) # minimal cache hint

    with torch_npu.profiler.profile(
            activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU
                    ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=100, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"./prof_{func.__name__}"),
            experimental_config=experimental_config) as prof:
                for data in test_datas:
                    func(*data)
```

运行以上脚本后，会采集 profiling 到以下目录：
- prof_hc_post：单算子执行的 profiling
- prof_hc_post_compiled：以静态 shape 编译后的执行 profiling
- prof_hc_post_compiled_dynamic：以动态 shape 编译后的执行 profiling

您可以查看目录下的 profiler 结果，了解 kernel 耗时的细节数据。

### 片段性能收益

| 模型部分 | 未编译耗时 | 编译后耗时(dynamic=False) | 性能提升 | 编译后耗时(dynamic=True) | 性能提升 |
| -------- | ---------- | ---------- | -------- | ---------- | -------- |
| hc_post  | 30 us      | 7.5 us     | 4x     | 10 us     | 3x     |

# DeepSeek 模型验证

您可以提前阅读 [模型改动说明](#模型改动说明) 了解我们对 DeepSeek 模型的改动细节。

## 下载 cann-recipes-infer 源码

```shell
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer/models/deepseek-v4-tilelang-and-inductorAF
```

后续示例命令均在 `cann-recipes-infer/models/deepseek-v4-tilelang-and-inductorAF/` 目录下执行。

## DeepSeek 模型下载与权重切分

示例模型使用 bf16 格式权重，采用 MoE(MP=16) + Attn(MP=4) 方式部署在 A3 昇腾卡上。执行模型前需要对模型权重进行转换与切分。

> 注意，示例主要用于展示加速效果，尽可能减少对原始模型的改动，切分方式以及脚本实现并非最佳实践。

### 下载模型权重

您需要先下载 [DeepSeek-v4-Flash原始权重](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main)。
请将下载得到的权重文件解压至 `/data/models/deepseek_v4` 目录下。

> 如果您使用的是其他路径，请在后续命令中替换为对应路径。

### 权重转换为bf16格式

使用 [convert_model.py](../deepseek-v4/utils/convert_model.py) 执行如下命令将权重转换为 bf16 格式：

> 入参介绍：`input_fp8_hf_path`：原始fp8权重路径；`output_hf_path`：转换后输出的权重路径；`quant_type`：量化模式
> 示例输出路径为 `/data/models/deepseek_v4_bf16`，您可以根据需要修改为其他路径。

```bash
python3 ../deepseek-v4/utils/convert_model.py \
    --input_fp8_hf_path /data/models/deepseek_v4 \
    --output_hf_path /data/models/deepseek_v4_bf16 \
    --quant_type bfloat16
```

### 切分模型权重文件

使用 [convert.py](./utils/convert.py) 执行如下命令将 bf16 权重切分为 MoE(MP=16) + Attn(MP=4) 格式：

> 相比于原始 convert.py 脚本，我们对 Attn 层参数以 mp4 切分方式。

```bash
python3 ./utils/convert.py \
    --hf-ckpt-path=/data/models/deepseek_v4_bf16 \
    --save-path=/data/models/deepseek_v4_bf16_moe16_attn4 \
    --n-expert=256 \
    --model-parallel=16
```

详细的切分策略如下：

- 量化标准=W8A8代表主量化层，即模型的关键计算层，每一路权重都会独立做量化校准 / 转换。
- 量化标准=W8A8-SHARE代表辅助量化层，参与量化，按照对应的TP值共享同一套量化参数。
- 量化标准=BF16代表跳过量化，保持原始精度。

|       模型层        |   参数名称   |  量化标准  | TP切分 |
| :-----------------: | :----------: | :--------: | :----: |
|    embed_tokens     |    embed     | W8A8-SHARE |   16   |
|   input_layernorm   |  attn_norm   |    BF16    |   16   |
| post_attn_layernorm |   ffn_norm   |    BF16    |   16   |
|       q_proj        |      wq      | W8A8-SHARE |   16   |
|      q_a_proj       |     wq_a     |    BF16    |   16   |
|    q_a_layernorm    |    q_norm    |    BF16    |   16   |
|      q_b_proj       |     wq_b     | W8A8-SHARE |   16   |
| kv_a_proj_with_mqa  |    wkv_a     |    BF16    |   16   |
|   kv_a_layernorm    |   kv_norm    |    BF16    |   16   |
|      kv_b_proj      |    wkv_b     | W8A8-SHARE |   16   |
|       o_proj        |      wo      |    W8A8    |   16   |
|      gate_proj      |      w1      | W8A8-SHARE |   16   |
|      down_proj      |      w2      |    W8A8    |   16   |
|       up_proj       |      w3      | W8A8-SHARE |   16   |
|       lm_head       |     head     | W8A8-SHARE |   16   |
|        embed        |    embed     | W8A8-SHARE |   16   |
|        wq_b         |     wq_b     | W8A8-SHARE |   4    |
|        wo_a         |     wo_a     | W8A8-SHARE |   4    |
|        wo_b         |     wo_b     |    W8A8    |   4    |
|        head         |     head     | W8A8-SHARE |   16   |
|      attn_sink      |  attn_sink   | W8A8-SHARE |   4    |
|    weights_proj     | weights_proj | W8A8-SHARE |   4    |

## 运行验证

执行以下命令启动 DeepSeek 模型：
```bash
torchrun --nproc_per_node=16 generate.py \
    --ckpt-path=/data/models/deepseek_v4_bf16_moe16_attn4 \
    --config=config.json \
    --input-file=./prompts.txt
```

首次启动由于磁盘缓存未生成，耗时较长，另外包含权重加载、dynamo tracing、inductor 编译等耗时环节，整个过程约 60min。后续启动将显著加快。

## 模型收益说明

由于未消除模型中的动态结构，我们未使能 aclgraph 下沉调度。统计模型耗时时，我们只统计 Device 上非 Hcom 类算子的 Kernel 耗时总合，而非直接统计 Host 上的函数调用耗时。

性能基准为替换了 TileLang-ascend 算子但未启用 inductor 编译的模型版本。

> 我们去除 Hcom 相关 Kernel 耗时由于其显著受到 Host 调度性能的影响。

| Step | 未编译Kernel总耗时（不含Hcom） | 编译后Kernel总耗时（不含Hcom） | 性能提升 |
| -------- | ---------- | ---------- | -------- |
| 2（无Compressor）  | 49.8 ms      | 39.5 ms     | 20.6%     |
| 4（触发Compressor）  | 52.8 ms      | 42.0 ms     | 20.4%     |

## 模型改动说明

### NPU 适配的改动

我们对模型进行了少量改动以使其能在昇腾 NPU 上运行。包括：

- 导入了 torch_npu 包以启用 NPU 设备支持。
- 模型中所有涉及设备指定的地方均改为使用 NPU 设备。

### TileLang-ascend 算子替换

替换了原始模型中 Tilelang 算子为 [TileLang-ascend](./tilelang_kernels/) 实现。

### 适配 inductor 编译的改动

在脚本开头，我们导入了 inductor_npu_ext 以注册基于 ascendc 的 inductor 编译后端扩展：

```python
import inductor_npu_ext
```

我们没有对模型进行过多改造来支持 fullgraph 编译，而是基于以下规则选择 inductor 编译范围：

1. 编译范围内图结构稳定，不包含执行时分支选择、.item()内存同步等行为，避免过多的断图（Graph Break）。
2. 避免编译范围过大，过大的编译范围由于 Guard 叠加，更容易触发重新编译。
3. 选择包含较多 Pintwise/Reduce 计算的范围进行编译，inductor 对该类计算的融合效果较好，能以较小的编译代价获取性能收益。

基于以上规则，我们对模型以下部分添加 @torch.compile 装饰以进行 inductor 编译：

- Expert.forward
- RMSNorm.forward
- Transformer.hc_head
- hc_pre
- hc_post

同时，我们将 hc_pre 片段中使用的 TileLang 算子 `hc_split_sinkhorn` 封装为 torch 自定义算子，这是自定义函数与 torch.compile 配合的良好方式，有助于减少 dyanmo 的断图。

```python
from tilelang_kernels.hc_split_sinkhorn_kernel import hc_split_sinkhorn as tl_hc_split_sinkhorn

tllib = torch.library.Library("tl", "FRAGMENT")
tllib.define("hc_split_sinkhorn(Tensor mixes, Tensor hc_scale, Tensor hc_base, int hc_mult=4, int sinkhorn_iters = 20, float eps=1e-6) -> (Tensor, Tensor, Tensor)")


@torch.library.impl(tllib, "hc_split_sinkhorn", "Meta")
def hc_split_sinkhorn_meta(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    return pre, post, comb


@torch.library.impl(tllib, "hc_split_sinkhorn", "NPU")
def hc_split_sinkhorn_npu_impl(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    return tl_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


hc_split_sinkhorn = torch.ops.tl.hc_split_sinkhorn
```

此外，我们对脚本进行了部分修改以避免触发 dynamo 在 torch 2.8 版本上的已知 bug ：

- [[dynamo] Fix tensor factory functions ignoring default device in torch.compile](https://github.com/pytorch/pytorch/pull/165473)

  > Tracing torch.arange 等 tensor 创建操作时，dynamo tracing 未使用默认 device 导致错误。将其纳入编译范围时，需要修改为指定 device 参数的形式。
  >
- [[dynamo] 8*s72 is not tracked with proxy for torch.fx.experimental.proxy_tensor](https://github.com/pytorch/pytorch/issues/163713)

  > 当从 tensor 发起 unflatten 操作时，dynamo tracing 会失败。将其纳入编译范围时，需要修改为使用 torch.unflatten 接口。
  >

注意，我们没有对模型输入 padding 以避免 prompt shape 变化，示例模型在输入长度变化时，可能会触发 dyanmo 重新编译。

### 其他改动

> 详情可以参考[模型文件](./model.py)。

- 我们对 Attn 层进行了少量修改以适配 MoE(MP=16) + Attn(MP=4) 混合切分方式。
- 修改 hadamard_transform 的实现以去除对 GPU 库的依赖，详情可以参考。
