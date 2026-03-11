#!/bin/sh
# 手动配置项
MODEL_PATH=/data/models/Qwen3-Next-80B-A3B-Instruct  # 权重路径
IP_NODE_P0=x.x.x.x   # P0节点ip
IP_NODE_D0=y.y.y.y   # D0节点ip
SOCKET_IFNAME="enp23s0f3"  # 网卡名称

# cpu高性能
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# 绑核
export SGLANG_SET_CPU_AFFINITY=1

# 设置 SGLANG PYTHONPATH
export PYTHONPATH=${PWD}/python:$PYTHONPATH

# CANN 相关
source /usr/local/Ascend/cann-8.5.0/set_env.sh
which bishengir-compile

# 内存碎片
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

# 通信
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export HCCL_BUFFSIZE=2048

# MTP相关
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1

# 节点信息
export nnodes=1
export node_rank=0
export p0=$IP_NODE_P0  # p0节点ip
export d0=$IP_NODE_D0  # d0节点ip
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME  # 根据实际情况配置
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
echo "HCCL_SOCKET_IFNAME : $SOCKET_IFNAME"
export ASCEND_MF_STORE_URL=tcp://$p0:24667

export SGLANG_DEEPEP_BF16_DISPATCH=1  # deepep, bf16权重
# export DEEP_NORMAL_MODE_USE_INT8_QUANT=1  # deepep, int8权重

export ENABLE_ASCENDC_FUSION_GDN="true"  # 使能GDN融合算子

export USE_CUSTOM_NORM_KERNEL="true"

export ENABLE_NPU_DEEPEP_MOE_MULTI_STREAM=1 # 使能MoE多流

export ASCEND_USE_FIA=1

# export ASCEND_USE_C8=1 # 使能C8