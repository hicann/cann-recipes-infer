#!/bin/bash
# =============================================================================
# 昇腾 NPU 环境初始化脚本
# 支持 aarch64 架构
# =============================================================================

ARCH=$(uname -m)

# =============================================================================
# Ascend 工具链路径
# =============================================================================
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH=$ASCEND_TOOLKIT_HOME
export ASCEND_AICPU_PATH=$ASCEND_TOOLKIT_HOME
export ASCEND_OPP_PATH=$ASCEND_TOOLKIT_HOME/opp
export TOOLCHAIN_HOME=$ASCEND_TOOLKIT_HOME/toolkit
source /usr/local/Ascend/nnal/atb/set_env.sh

# =============================================================================
# 基础库路径
# =============================================================================
BASE_LIB_PATH=$ASCEND_TOOLKIT_HOME/${ARCH}-linux/lib64
DRIVER_LIB_PATH=/usr/local/Ascend/driver/lib64

# 动态库路径
export LD_LIBRARY_PATH=$DRIVER_LIB_PATH/driver:$DRIVER_LIB_PATH/common:$BASE_LIB_PATH:$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/lib64/plugin/opskernel:$ASCEND_TOOLKIT_HOME/lib64/plugin/nnengine:$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/${ARCH}:$ASCEND_TOOLKIT_HOME/tools/aml/lib64:$ASCEND_TOOLKIT_HOME/tools/aml/lib64/plugin:$LD_LIBRARY_PATH

# Python 路径
export PYTHONPATH=$ASCEND_TOOLKIT_HOME/python/site-packages:$ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH

# 执行路径
export PATH=$ASCEND_TOOLKIT_HOME/bin:$ASCEND_TOOLKIT_HOME/compiler/ccec_compiler/bin:$ASCEND_TOOLKIT_HOME/tools/ccec_compiler/bin:$PATH

# =============================================================================
# ATB (加速库) 环境变量
# =============================================================================
export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
export LD_LIBRARY_PATH=$ATB_HOME_PATH/lib:$LD_LIBRARY_PATH

# ATB 运行时配置 - 基础优化
export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
export ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
export ATB_OPSRUNNER_SETUP_CACHE_ENABLE=1
export ATB_OPSRUNNER_KERNEL_CACHE_TYPE=3
export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
export ATB_LAUNCH_KERNEL_WITH_TILING=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=1

# ATB 进一步优化 - 新增
export ATB_SPLIT_MMF_ENABLE=1
export ATB_MEMORY_REUSE_ENABLE=1
export ATB_WORKSPACE_MEM_REUSE_ENABLE=1

# =============================================================================
# 算子库环境变量
# =============================================================================
export ASDOPS_HOME_PATH=$ATB_HOME_PATH
export ASDOPS_MATMUL_PP_FLAG=1
export ASDOPS_LOG_LEVEL=ERROR
export ASDOPS_LOG_TO_FILE=1

# =============================================================================
# CANN 性能优化 - 新增
# =============================================================================
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

# =============================================================================
# NPU 内存优化 - 新增
# =============================================================================
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"
export NPU_MEMORY_FRACTION=0.95
export TASK_QUEUE_ENABLE=2

# =============================================================================
# 算子/日志优化 - 新增
# =============================================================================
export OP_DEBUG_LEVEL=0
export ASCEND_GLOBAL_LOG_LEVEL=3          # 0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# =============================================================================
# 图编译优化 - 新增 (配合 torch.compile)
# =============================================================================
export GRAPH_OP_RUN=1
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export ACL_PRECISION_MODE=allow_fp32_to_fp16

# =============================================================================
# Python/Tokenizer 优化 - 新增
# =============================================================================
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# =============================================================================
# vLLM 优化 - 新增
# =============================================================================
export VLLM_ATTENTION_BACKEND=NPU
export VLLM_USE_TRITON_FLASH_ATTN=0

# =============================================================================
# DeepSeek-OCR-2 项目特定配置 - 新增
# =============================================================================
# 启用视觉编码器 torch.compile (实验性)
# export DPSK_ENABLE_COMPILE=1

# 默认模型路径 (可通过环境变量覆盖)
# export DPSK_MODEL_PATH=/data/models/DeepSeek-OCR-2
# export DPSK_INPUT_PATH=/data/test_images
# export DPSK_OUTPUT_PATH=./output

# =============================================================================
# 输出环境信息
# =============================================================================
echo "[INFO] Ascend NPU environment initialized (Enhanced v2)"
echo "[INFO] Architecture: $ARCH"
echo "[INFO] ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
echo "[INFO] ATB_HOME_PATH: $ATB_HOME_PATH"
