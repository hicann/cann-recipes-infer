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
# Start the DeepSeek-V4-Flash single-card inference container (reference script; adjust the
# configurable items below for your host).
# Images: 910B -> lmsysorg/sglang:deepseek-v4-npu-910b, A3 -> lmsysorg/sglang:deepseek-v4-npu-a3
set -euo pipefail

# ===== Configurable (adjust for your environment) =====
IMAGE="${IMAGE:-lmsysorg/sglang:deepseek-v4-npu-910b}"
NAME="${NAME:-dsv4_singlecard}"
WORKSPACE="${WORKSPACE:?host directory holding this project, mounted at /workspace/code}"
MODEL_DIR="${MODEL_DIR:?host weight directory (both the W8A8 and MXFP4 copies), mounted at /workspace/models}"
SERVICE_PORT="${SERVICE_PORT:-8020}"            # in-container service port, published on the same host port
SHM_SIZE="${SHM_SIZE:-16g}"
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-auto}"   # auto = mount every davinci device; or list them, e.g. "0,3"

# ===== NPU device discovery =====
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

# ===== Driver and data mounts =====
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
# Once inside the container follow docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md
# (install libhwloc -> clone -> apply patches -> build -> convert GGUF -> launch).
