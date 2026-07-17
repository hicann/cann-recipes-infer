# GLM-5.2 Inference on NPU
## 概述
智谱团队发布了 GLM-5.2，本样例基于 [GLM-5](../glm_5) 样例适配迁移，并在 CANN 平台上完成对应的优化适配，可在华为 Atlas A3 集群上运行起来。

- GLM-5.2 模型结构沿用 GLM-5 / DeepSeek-V3.2-Exp 的 DSA（MLA + Lightning Indexer）+ MoE + MTP，本样例的并行策略方案沿用 DeepSeek-V3.2-Exp。详细方案请参考[NPU DeepSeek-V3.2-Exp推理优化实践](../../docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_inference_guide.md)。
- **GLM-5.2 相对 GLM-5 的核心新增是 IndexShare（跨层 top-k 复用）**：按 `config.indexer_types` 每层取 `full`（本层运行 indexer）或 `shared`（复用上一个 full 层的 top-k，不持有 indexer 权重）。
- **CANN 版本要求**：GLM-5.2 真实注意力维度（`q_lora_rank=2048`、`qk_nope_head_dim=192`、`v_head_dim=256`）需 **CANN 9.1** 的 `mla_prolog_v3` 内核支持。

---

## 硬件要求
产品型号：Atlas A3 系列

操作系统：Linux ARM

镜像版本：cann9.1_pt2.8.0_glm_aarch_image_v0.1.tar

驱动版本：Ascend HDK 25.2.0
> npu-smi info 检查Ascend NPU固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为 25.2.0。如果未安装或者版本不是 25.2.0，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=All&driver=Ascend+HDK+25.2.0)，然后根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。


## 快速启动


### 下载源码

  在各个节点上执行如下命令下载 cann-recipes-infer 源码。
  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone https://gitcode.com/cann/cann-recipes-infer.git
  cd cann-recipes-infer
  ```
### 下载数据集
  从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)中下载长序列输入数据集longbook_qa_eng，并上传到各个节点上新建的路径 dataset/InfiniteBench下。
  ```shell
  mkdir -p dataset/InfiniteBench
  ```

### 下载权重
  智谱团队开源了 GLM-5.2 模型的 Bfloat16 权重。
  - 下载[GLM-5.2原始Bfloat16权重](https://huggingface.co/zai-org/GLM-5.2)，并上传到Atlas A3各节点某个固定的路径下，比如`/data/models/GLM-5.2-BF16`。

### 获取 docker 镜像
  从[ARM镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/GLM/cann9.1_pt2.8.0_glm_aarch_image_v0.1.tar)中下载 docker 镜像，然后上传到A3服务器的每个节点上，并通过命令导入镜像 `docker load -i cann9.1_pt2.8.0_glm_aarch_image_v0.1.tar`。

### 拉起 docker 容器

  在各个节点上通过如下脚本拉起容器，默认容器名为 cann_recipes_infer。注意：需要将权重路径和源码路径挂载到容器里。
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
      cann9.1_pt2.8.0_glm_aarch_image:v0.1 /bin/bash
  ```
  在各个节点上通过如下命令进入容器：
  ```
  docker attach cann_recipes_infer
  cd /home/code/cann-recipes-infer/models/glm_5_2
  ```

### 转换权重

  在各个节点上使用`weight_convert.sh` 脚本完成BF16到Int8权重转换。

  >入参介绍：`input_hf_path`：原始权重路径；`output_hf_path`：转换后输出的权重路径；`quant_mode`：量化模式

如果权重转换的运行环境为NPU，需要先执行：

```shell
cann_path=/usr/local/Ascend/ascend-toolkit/latest  # CANN包安装路径
source ${cann_path}/bin/setenv.bash
```

  权重转换拉起示例：

  ```shell
  # 转换为W8A8C16权重
  bash utils/weight_convert.sh --input_hf_path /data/models/GLM-5.2-BF16 --output_hf_path /data/models/GLM-5.2-W8A8 --quant_mode w8a8c16
  ```

  > **GLM-5.2 IndexShare 说明**：`utils/convert_model.py` 已按 `config.indexer_types` 感知 IndexShare —
  > 仅对 `full` 层(+MTP)生成 `self_attn.indexer.*` 的 ignore / 量化 / hadamard 项，`shared` 层不再生成多余
  > indexer 量化产物。

### 修改代码
- 修改`cann-recipes-infer/executor/scripts/set_env.sh`中的如下字段:
  ```shell
  export IPs=('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx') # 所有节点的IP，确保第1个IP是master，多个节点的ip通过空格分开
  cann_path="your_cann_pkgs_path" # CANN软件包安装路径，镜像默认CANN包路径为`/usr/local/Ascend/ascend-toolkit/latest`
  ```
- 在各个节点上修改 `config/` 路径下需要执行的yaml文件中的model_path路径。关于YAML文件中的更多配置说明可参见[YAML参数描述](./config/README.md)。

  ```
  # W8A8
  model_path: "/data/models/GLM-5.2-W8A8"
  ```

- 在各个节点上修改 infer.sh 文件中的YAML_FILE_NAME，指定为上一步需要执行的yaml文件名。默认的yaml路径为32卡推理。

  ```
  # 默认 32 卡部署（W8A8 + KV offload）
  export YAML_FILE_NAME=glm_5_2_rank_32_32ep_w8a8_offload.yaml
  ```

### 拉起多卡推理
  在各个节点上同步执行如下命令即可拉起多卡推理任务。
  ```shell
  bash infer.sh
  ```

## KV Offload 启用（长序列，需自定义算子）

GLM-5.2 的 KV offload（`enable_offload: True`）把全量 MLA KV 卸载到 host swapped memory，decode 时按 DSA top-k 把命中 block gather 回 device，面向长上下文（全量 KV 放不下 HBM）场景。开启 `shared_indexer_offload: True` 时，模型会利用 GLM-5.2 的 IndexShare 特性复用同一组 top-k 规划，减少共享层重复的命中判断和缓存管理开销，并将常驻池回填与 SFA、MoE 计算并行。该路径释放 HBM 以支持更长上下文或更大 batch；`enlarge_pool_size: True` 会把设备侧常驻 token 池从 8K 扩大到 16K，进一步提高命中率并减少 host 到 device 的 KV 搬运。

KV offload 依赖**仓内自定义 AscendC 算子**（非 torch_npu 内置），须先编译安装：

```bash
# 1) 编译算子内核（A3 默认 ascend910_93；bisheng 随 CANN 提供）
cd ops/ascendc
bash build.sh -n "gather_selection_kv_cache;dsa_plan;dsa_serve;dsa_install"  # 安装包位于 output/CANN-custom_ops-*-linux.<arch>.run

# 2) 安装内核到 CANN opp
cd output
./CANN-custom_ops-*-linux.*.run --quiet --install-path="${ASCEND_HOME_PATH}/opp"

# 3) 编译并安装 torch 绑定（缺 ninja 时先 pip install ninja）
cd ../torch_ops_extension
bash build_and_install.sh                              # 生成并 pip 安装 custom_ops wheel

# 4) 运行前 source 自定义算子环境（设置 ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH）
source "${ASCEND_HOME_PATH}/opp/vendors/customize/bin/set_env.bash"
```

> `models/modeling_glm.py` 在 import 时自动把基础 offload 算子挂到 `torch_npu.npu_gather_selection_kv_cache`；开启 `shared_indexer_offload` 时，模型 Runner 初始化会按需加载共享 IndexShare offload 相关自定义算子。因此模型侧无需改动，只要上面的算子已安装、且运行前 source 了第 4 步的环境即可。

运行（在 `config/*.yaml` 的 `model_config` 中设置 offload 选项）：

```bash
enable_offload: True
shared_indexer_offload: True
enlarge_pool_size: False
# eager
exe_mode: "eager"
# 或图模式
exe_mode: "npugraph_ex"
```


## 附录
### FAQ
- **HCCL_BUFFSIZE不足问题**：如果报错日志中出现关键字"HCCL_BUFFSIZE is too SMALL, ..., NEEDED_HCCL_BUFFSIZE..., HCCL_BUFFSIZE=200MB, ..."，可通过配置环境变量 `export HCCL_BUFFSIZE=实际需要的大小` 解决，所有Rank上的该环境变量需保持一致。HCCL_BUFFSIZE参数介绍可参考[昇腾资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/maintenref/envvar/envref_07_0080.html)中的详细描述。
- **自定义算子导入失败**：如果报错日志中出现类似关键字"'_OpNamespace' 'custom' object has no attribute"，可参考[自定义算子指南](../../ops/ascendc/README.md)编译所需算子。
