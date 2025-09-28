NPU TileLang 使用指南
============================================================================
# 1. 准备
## a) 镜像准备
### 获取 docker 镜像
从[ARM镜像地址](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann8.3.rc1.alpha002/pt2.5.1/aarch/ascendc/cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image.tar)中下载 docker 镜像，然后上传到需要A3服务器每个节点上，并通过命令导入镜像 `docker load -i cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image.tar`

### 拉起 docker 容器

  容器拉起脚本如下，默认容器名为 cann_recipes_infer。
  ```
  docker run -u root -itd --name cann_recipes_infer --ulimit nproc=65535:65535 --ipc=host \
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
      swr.cn-east-4.myhuaweicloud.com/cann_recipes/cann_recipes-a3-arm:v202509221500 /bin/bash
  ```
  进入容器：
  ```
  docker attach cann_recipes_infer
  ```

## b) 设置环境变量
  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```

# 2. TileLang 安装

安装TileLang有几种方法，更多详情请参考TileLang社区的安装指南：

https://github.com/tile-ai/tilelang/blob/main/docs/get_started/Installation.md#method-3-install-using-the-provided-script



这里使用从源码编译的方式安装。

## a) 下载代码

    git clone --recursive https://github.com/tile-ai/tilelang-ascend
    cd tilelang-ascend

## b) 编译和安装
    bash install_ascend.sh

## c) 环境变量设置

    source set_env.sh

# 3. 运行

----------------------------------

在本节中，您将学习如何调用NPU TileLang算子。

这里以**sparse_flash_attention**算子为例来介绍。

```
cd {your_cann_recipes_path}/ops/tilelang/examples # 比如{your_cann_recipes_path} 改成 /home/code/cann-recipes-infer
python3 test_sparse_flash_attention.py
```

成功后会打印：

```
Test passed!
```



以下的代码片段都包含在**sparse_flash_attention.py**中，这里详细来介绍下如何运行和测试该算子。

## a) 算子调用

**sparse_fa_func**是算子的入口，调用他来进行**sparse_flash_attention**的计算：

```
sparse_fa_func = sparse_attention_fwd(
    heads=128,
    dim=512,
    tail_dim=64,
    topk=2048,
    kv_stride=1,
)
```

## b) 生成Golden值

下面的代码来生成算子的golden值。

    def ref_sparse_attention_fwd_interface(q, kv, indices, q_start_index_s, kv_stride=4, sm_scale=None, is_casual=True):
        q = q.float()
        kv = kv.float()
        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        if q_start_index_s is None:
            q_start_index_s = sk * kv_stride - sq
    
    assert kv.shape[-1] == 576, 'you should assign dim otherwise'
    dim = 512
    k = kv
    v = kv[..., :dim]
    
    b, _, _, dim_v = v.shape
    num_kv_per_index = 1
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(q_start_index_s, sq + q_start_index_s, dtype=torch.int32).view(-1, 1) >= torch.arange(kv_stride-1, sk * kv_stride, kv_stride, dtype=torch.int32).view(1, -1)
    
    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)
    
    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q ** -0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.float16)

## c) 算子的整个调用和测试对比

```
os.environ["ACL_OP_INIT_MODE"] = "1"
B, S, SKV, H, HKV, DQK, DV, topk = 1, 128, 32768, 128, 1, 576, 512, 2048
dtype = torch.float16

KV_stride = 1
q_start_s_index = 4096 * 7

q = torch.randn((B, S, H, DQK), dtype=dtype)
kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype)
indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32)
for b in range(B):
    for t in range(S):
        for h in range(HKV):
            i_i = torch.randperm(max(1, ((t + q_start_s_index) // KV_stride)))[:topk]
            indices[b, t, h, :len(i_i)] = i_i

output = torch.empty((B, S, H, DV), dtype=dtype)

workspace_1 = torch.zeros((256, 64, 512), dtype=dtype)
workspace_2 = torch.zeros((256, 64, 64), dtype=dtype)
workspace_3 = torch.zeros((256, 64, 64), dtype=torch.float)
workspace_4 = torch.zeros((256, 64, 64), dtype=dtype)
workspace_5 = torch.zeros((256, 64, 512), dtype=torch.float)

torch.npu.synchronize()
print("init successful!")

output = func(q, kv, indices, workspace_1, workspace_2, workspace_3, workspace_4, workspace_5)

torch.npu.synchronize()

ref_output = ref_sparse_attention_fwd_interface(q, kv, indices, q_start_s_index, KV_stride)
torch.npu.synchronize()
torch.testing.assert_close(ref_output, output, rtol=1e-2, atol=1e-2)
```


