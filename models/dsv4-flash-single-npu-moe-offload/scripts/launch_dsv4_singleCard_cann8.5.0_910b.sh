#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 起 DeepSeek-V4-Flash 单卡推理容器（参考脚本，按需改下面几个可配置项）。
# 镜像：910B -> lmsysorg/sglang:deepseek-v4-npu-910b ；A3 -> lmsysorg/sglang:deepseek-v4-npu-a3
set -euo pipefail

# ===== 可配置（按你的环境改）=====
IMAGE="${IMAGE:-lmsysorg/sglang:deepseek-v4-npu-910b}"
NAME="${NAME:-dsv4_singlecard}"
WORKSPACE="${WORKSPACE:?宿主机代码目录（放本工程），将挂到容器 /workspace/code}"
MODEL_DIR="${MODEL_DIR:?宿主机权重目录（含 W8A8 + MXFP4 两份权重），将挂到 /workspace/models}"
SERVICE_PORT="${SERVICE_PORT:-8020}"            # 容器内服务端口，映射到宿主同号端口
SHM_SIZE="${SHM_SIZE:-16g}"
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-auto}"   # auto=挂所有 davinci 卡；或指定 "0,3"

# ===== NPU 设备发现 =====
DEVICES=()
for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -e "$d" ]] && DEVICES+=(--device "$d")
done
if [[ "$NPU_VISIBLE_DEVICES" == "auto" ]]; then
  shopt -s nullglob
  mapfile -t DLIST < <(printf '%s\n' /dev/davinci[0-9]* | sort -V)
  shopt -u nullglob
else
  IFS=',' read -ra IDS <<<"$NPU_VISIBLE_DEVICES"; DLIST=()
  for i in "${IDS[@]}"; do [[ -e "/dev/davinci${i}" ]] && DLIST+=("/dev/davinci${i}"); done
fi
for d in "${DLIST[@]}"; do DEVICES+=(--device "$d"); done

# ===== 驱动 + 数据挂载 =====
MOUNTS=(
  -v /usr/local/sbin:/usr/local/sbin
  -v /usr/local/dcmi:/usr/local/dcmi
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
  -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v /var/queue_schedule:/var/queue_schedule
  -v "${WORKSPACE}":/workspace/code
  -v "${MODEL_DIR}":/workspace/models
)

docker run --rm -it \
  --name "${NAME}" \
  "${DEVICES[@]}" "${MOUNTS[@]}" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --privileged=true \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size "${SHM_SIZE}" \
  -p "${SERVICE_PORT}:${SERVICE_PORT}" \
  "${IMAGE}" bash
# 进容器后按 docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md 继续（装 libhwloc → 拉代码 → 打补丁 → 编译 → 转 GGUF → 拉起）。
