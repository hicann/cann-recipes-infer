# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

#!/bin/bash
pkill -9 python* | pkill -9 sglang
sleep 3
pkill -9 python* | pkill -9 sglang
sleep 2
rm -rf kernel_meta
rm -rf extra-info
rm -rf /root/ascend/
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=10
sysctl -w kernel.numa_balancing=0

## variables need to modify
# IP of the all decode servers
IPs=('IP0' 'IP1' 'IP2' 'IP3')
# Socket prefix, to obtain Host IP for HCCL and HCCL group; modify to enp/eth accordingly
IFNAMES=('IFNAMES1' 'IFNAMES2' 'IFNAMES3' 'IFNAMES4')
# model path
MODEL_PATH=/PATH/TO/MODEL_WEIGHT
# add sglang code path to PYTHONPATH
PATH_TO_SGLANG=/PATH/TO/SGLANG
# prefill master server ip required to connect between pd disaggregation
Prefill_Master_Server_IP="xx.xx.xx.xx"

export PYTHONPATH=${PATH_TO_SGLANG}/python:$PYTHONPATH
export ASCEND_MF_STORE_URL=tcp://${Prefill_Master_Server_IP}:24667

# CANN environment variables
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export ASCEND_USE_FIA=True
export HCCL_BUFFSIZE=1000
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

# SGLang environment variables
export SGLANG_SET_CPU_AFFINITY=1

LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
echo "LOCAL_HOST = " ${LOCAL_HOST}

for i in "${!IPs[@]}";
do
  echo "LOCAL_HOST=${LOCAL_HOST}, IPs[${i}]=${IPs[$i]}"
  if [ "$LOCAL_HOST" == "${IPs[$i]}" ]; then
      echo "Node Rank : ${i}"
      VC_TASK_INDEX=$i
      break
  fi
done
export HCCL_SOCKET_IFNAME=${IFNAMES[$VC_TASK_INDEX]}
export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
echo "HCCL_SOCKET_IFNAME : ${HCCL_SOCKET_IFNAME}"
nnodes=${#IPs[@]}
tp_size=`expr 16 \* ${nnodes}`
dp_size=`expr $tp_size / 4`

python3 -m sglang.launch_server \
--model-path ${MODEL_PATH} \
--tp-size $tp_size \
--ep-size $tp_size \
--dp-size $dp_size \
--trust-remote-code \
--max-total-token 436000 \
--dtype bfloat16 \
--quantization w8a8_int8 \
--attention-backend ascend \
--device npu \
--watchdog-timeout 9000 \
--host 0.0.0.0 \
--port 30001 \
--mem-fraction-static 0.85 \
--context-length 6144  \
--disable-radix-cache \
--chunked-prefill-size 12288 \
--max-prefill-tokens 5120 \
--max-running-requests 64 \
--disable-overlap-schedule \
--enable-dp-attention \
--enable-dp-lm-head \
--enable-torch-compile \
--moe-a2a-backend deepep \
--deepep-mode auto \
--enable-metrics \
--disaggregation-mode decode \
--disaggregation-transfer-backend ascend \
--prefill-round-robin-balance \
--load-balance-method round_robin \
--dist-init-addr ${IPs[0]}:10000 \
--nnodes $nnodes \
--node-rank $VC_TASK_INDEX 2>&1 | tee launch.log &