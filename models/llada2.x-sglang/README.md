# LLaDA2.0模型基于sglang框架在NPU实现推理

## 概述
LLaDA2.0模型是2025年开源的Diffusion LLM大语言模型，本样例基于sglang开源框架[llada2.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llada2.py)，完成LLaDA2.0模型在sglang框架上的的适配。

## 支持的产品型号
<term>Atlas A2 系列产品</term>

## 环境准备

参考sglang-npu官方教程准备conda或者docker环境(具体的版本和docker链接也可直接参考官方链接)：https://docs.sglang.io/platforms/ascend_npu.html
当前版本复现docker如下，也可参考上述提到的官方环境准备教程：
1. 拉取镜像
```
docker pull quay.io/ascend/sglang:main-cann8.3.rc2-910b
```
2. 基于基础镜像创建sglang镜像

```
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
docker build -t <image_name> -f npu.Dockerfile .
```
3. 创建容器并执行：其中{your_offline_model_path}为host上下载好的模型权重路径
```
#!/usr/bin/env bash
set -euo pipefail

# ===== 用户配置 =====
export IMAGE=${IMAGE:-quay.io/ascend/sglang:main-cann8.3.rc2-910b}
export NAME=${NAME:-sglang-main-cann8.3.rc2-910b}
export SHM_SIZE=${SHM_SIZE:-16g}
export CACHE_DIR=${CACHE_DIR:-$HOME/.cache}
export MODEL_DIR=${MODEL_DIR:-/your_offline_model_path}
export NPU_VISIBLE_DEVICES=${NPU_VISIBLE_DEVICES:-auto}
# =====================

DEVICES=()
for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -e "$d" ]] && DEVICES+=(--device "$d")
done

if [[ "$NPU_VISIBLE_DEVICES" == "auto" ]]; then
  mapfile -t DLIST < <(ls -1 /dev/davinci[0-9]* 2>/dev/null | sort -V)
else
  IFS=',' read -ra IDS <<<"$NPU_VISIBLE_DEVICES"
  DLIST=()
  for i in "${IDS[@]}"; do
    [[ -e "/dev/davinci${i}" ]] && DLIST+=("/dev/davinci${i}")
  done
fi
for d in "${DLIST[@]}"; do DEVICES+=(--device "$d"); done
VISIBLE_IDS=$(printf "%s\n" "${DLIST[@]}" | sed -E 's#.*/davinci([0-9]+)#\1#' | paste -sd, -)

MOUNTS=(
  -v /usr/local/sbin:/usr/local/sbin
  -v /usr/local/dcmi:/usr/local/dcmi
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v /var/queue_schedule:/var/queue_schedule
  -v "${MODEL_DIR}":/workspace/models/
  -v "${CACHE_DIR}":/root/.cache
)

# 组装容器内部初始化命令（注意必须是一行）
CMD='
set -e
echo "==> 可见 NPU: $NPU_VISIBLE_DEVICES"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
fi
echo "容器初始化完毕，已进入交互模式。"
exec bash
'
docker run --rm \
  --name "${NAME}" \
  "${DEVICES[@]}" \
  "${MOUNTS[@]}" \
  -e ASCEND_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
  -e NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"\
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size "${SHM_SIZE}" \
  -it "${IMAGE}" bash -c "${CMD}"
```
4. 卸载容器内的官方sglang版本
```
pip uninstall -y sglang
```

## 权重准备
1. 下载[LLaDA2.x权重]，以LLaDA2.1-mini为例 (https://huggingface.co/inclusionAI/LLaDA2.1-mini) 或者 LLaDA2.0-mini （https://huggingface.co/inclusionAI/LLaDA2.0-mini）

## 代码准备
1. 下载本样例所在代码仓，以master分支为例
   ```shell
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   ```

2. 获取sglang主仓源码，并应用patch
   ```shell
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git reset --hard 615a02dcd45adf875de698d2ba66b1cbef161aa5

   # 将修改本仓中的修改patch应用到sglang代码中
   git am ../cann-recipes-infer/models/llada2.x-sglang/patches/*.patch
   ```
3. 基于patched之后的代码，重装sglang. 
   ```shell
   cd  ./python/
   cp  ../../cann-recipes-infer/models/llada2.0-sglang/install_sglang_dev.sh ./
   cp pyproject_npu.toml.toml pyproject.toml
   chmod +x install_sglang_dev.sh
   ./install_sglang_dev.sh
   ```
   
## 推理执行
1. llada 2.1拉起服务, 如果模型文件以及下载到本地，替换--model-path的参数为本地参数路径
   ```shell
    export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
    python -m sglang.launch_server \
            --model-path inclusionAI/LLaDA2.1-mini \
            --host 0.0.0.0 \
            --port 8000 \
            --device npu \
            --attention-backend ascend \
            --dtype bfloat16 \
            --kv-cache-dtype auto \
            --trust-remote-code \
            --disable-radix-cache \
            --mem-fraction-static 0.90 \
            --max-running-requests 1 \
            --dllm-algorithm "JointThreshold" \
            --enable-tokenizer-batch-encode \
            --skip-server-warmup \
            --enable-cache-report \
            --tp 1 \
   ```
2. llada 2.0拉起服务, 如果模型文件以及下载到本地，替换--model-path的参数为本地参数路径
   ```shell
    export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
    python -m sglang.launch_server \
            --model-path inclusionAI/LLaDA2.0-mini \
            --host 0.0.0.0 \
            --port 8000 \
            --device npu \
            --attention-backend ascend \
            --dtype bfloat16 \
            --kv-cache-dtype auto \
            --trust-remote-code \
            --disable-radix-cache \
            --mem-fraction-static 0.90 \
            --max-running-requests 1 \
            --dllm-algorithm "LowConfidence" \
            --enable-tokenizer-batch-encode \
            --skip-server-warmup \
            --enable-cache-report \
            --tp 1 \
   ```
   
2. 执行推理
	```shell
    curl http://localhost:8000/v1/chat/completions
    -H "Content-Type: application/json"
    -d '{
    "model": "LLaDA",
    "messages": [
    {"role": "user", "content": "(Question: Elizas rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week? Answer:"}
    ],
    "temperature": 0.0,
    "max_tokens": 25600,
    "ignore_eos": false
    }'
    ```

## gms8k精度测试，修改对应的模型和算法
```shell
vim test/registered/dllm/test_llada2_mini_ascend.py
```
llada2.1 对应参数:
```shell
 		cls.model = "inclusionAI/LLaDA2.1-mini"
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "1",
            "--tp",
            "1",
            "--attention-backend",
            "ascend",
            "--dllm-algorithm",
            "JointThreshold",  # TODO: Add dLLM configurations
        ]
```
llada2.0 对应参数:
```shell
 		cls.model = "inclusionAI/LLaDA2.0-mini"
        other_args = [
            "--trust-remote-code",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.9",
            "--max-running-requests",
            "1",
            "--tp",
            "1",
            "--attention-backend",
            "ascend",
            "--dllm-algorithm",
            "LowConfidence",  # TODO: Add dLLM configurations
            "--dllm-algorithm-config",
            "dllm_config.yaml",
        ]
```
执行脚本
```shell
python test_llada2_mini_ascend.py
```

## 附录
### speed 以及 gms8k 精度测试

- llada2.1 tp1-bs1 5shot

![image.png](https://raw.gitcode.com/user-images/assets/8916162/d289103f-ccf2-414a-8e82-717393067b2c/image.png 'image.png')
![image.png](https://raw.gitcode.com/user-images/assets/8916162/f22bff48-523f-42f5-bf0a-22a45a827523/image.png 'image.png')

- llada2.1 tp1-bs1 zero shot

![image.png](https://raw.gitcode.com/user-images/assets/8916162/26297662-4a7d-4ee4-a12f-412b6ce2c52d/image.png 'image.png')


- llada2.0 tp1-bs1 5shot

![image.png](https://raw.gitcode.com/user-images/assets/8916162/3e233c6e-69e8-4ef6-8c19-81fb96c48f9b/image.png 'image.png')
![image.png](https://raw.gitcode.com/user-images/assets/8916162/6c21f192-d581-401b-8dfc-9cafffd0cad4/image.png 'image.png')

- llada2.0 tp2-bs1 5shot

![image.png](https://raw.gitcode.com/user-images/assets/8916162/74918396-1906-419c-8ead-f5fafe1cac93/image.png 'image.png')
![image.png](https://raw.gitcode.com/user-images/assets/8916162/1a7ec2d1-1c7e-4665-b0f9-ad9b97b26b5e/image.png 'image.png')
### 文件说明
|文件路径|说明|
|-------|--------|
|[0001-NPU-support-DLLM-ascend-backend-on-NPU-with-LLaDA2-t.patch](patches/0001-NPU-support-DLLM-ascend-backend-on-NPU-with-LLaDA2-t.patch)| LLaDA2.0 sglang-ascend支持，主要涉及到ascend_backend的dllm适配，page_size适配（默认128，需要适配到dllm block size大小）.
|[0002-NPU-support-dllm-model-LLaDA2.0-graph-capture-and-re.patch](patches/0002-NPU-support-dllm-model-LLaDA2.0-graph-capture-and-re.patch)| 支持dllm在昇腾上的入图
|[0003-NPU-speed-up-llada2.0-on-npu-with-graph-mode.patch](patches/0003-NPU-speed-up-llada2.0-on-npu-with-graph-mode.patch)| 入图后的算子融合及优化
|[0004-NPU-update-test_llada2_mini_ascend.py-with-graph-mod.patch](patches/0004-NPU-update-test_llada2_mini_ascend.py-with-graph-mod.patch)| 更新入图模式下的benchmark脚本
|[0005-NPU-fix-bug-in-python-sglang-srt-server_args.py.patch](patches/0005-NPU-fix-bug-in-python-sglang-srt-server_args.py.patch)| fix bug in python/sglang/srt/server_args.py
|[0006-NPU-remove-some-redundent-code.patch](patches/0006-NPU-remove-some-redundent-code.patch)| 删除一些无关代码
|[0007-NPU-Update-dllm-llada2.0-ci-for-npu.patch](patches/0007-NPU-Update-dllm-llada2.0-ci-for-npu.patch)|  Update dllm-llada2.0 ci for npu
|[0008-NPU-dllm-update-format-code-with-pre-commit.patch](patches/0008-NPU-dllm-update-format-code-with-pre-commit.patch)|   dllm update: format code with pre-commit
|[0009-NPU-update-ci-by-introducing-the-SGLANG_NPU_DISABLE_.patch](patches/0009-NPU-update-ci-by-introducing-the-SGLANG_NPU_DISABLE_.patch)|   update ci by introducing the SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT env parameter, to avoid oom with tp1
|[0011-Update-joint_threshold.py.patch](patches/0009-NPU-update-ci-by-introducing-the-SGLANG_NPU_DISABLE_.patch)|   在npu上加速update-joint threhold 算法（for llada2.1
|[0012-fuse-shared-output-into-moe-groupmatmul2-to-speedup-.patch](patches/0012-fuse-shared-output-into-moe-groupmatmul2-to-speedup-.patch)|  fuse shared output into moe groupmatmul2 to speedup on npu
|[0013-Update-unquant.py.patch](patches/0013-Update-unquant.py.patch)|  cache the row idx for fusedMoE with NPU inference
|[0014-update-the-ascend-dllm-accuracy-test-config.patch](patches/0014-update-the-ascend-dllm-accuracy-test-config.patch)|  update the accuracy test config file
|[0015-update-the-comments-in-test_llada2_mini_ascend.py.patch](py.patch)|  update the commaccuracy test config file
|[0016-revert-the-moe-gate-accuracy-to-the-model-prededfine.patch](patches/0016-revert-the-moe-gate-accuracy-to-the-model-prededfine.patch)|  revert the moe gate accuracy to the model predefined accuracy


