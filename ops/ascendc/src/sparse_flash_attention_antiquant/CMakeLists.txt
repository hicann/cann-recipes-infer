# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME SparseFlashAttentionAntiquant
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(sparse_flash_attention_antiquant_depends transformer/attention/sparse_flash_attention_antiquant PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        op_host/sparse_flash_attention_antiquant_def.cpp
)

target_sources(optiling PRIVATE
        op_host/sparse_flash_attention_antiquant_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/sparse_flash_attention_antiquant_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        op_host/sparse_flash_attention_antiquant_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
