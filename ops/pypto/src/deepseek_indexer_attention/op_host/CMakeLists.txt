# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

target_sources(opsproto_rt2.0 PRIVATE
        deepseek_indexer_attention_def.cpp
        deepseek_indexer_attention_proto.cpp
)

target_include_directories(opsproto_rt2.0 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME_PATH}/include
)

target_sources(opmaster_ct PRIVATE
        deepseek_indexer_attention_proto.cpp
)

target_include_directories(opmaster_ct PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME_PATH}/include
)

target_sources(opmaster_rt2.0 PRIVATE
        deepseek_indexer_attention_tiling.cpp
)

target_include_directories(opmaster_rt2.0 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_HOME_PATH}/include
)
