# Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(layers_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_rmsnorm_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_rotary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transpose_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/insert_kv_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deepseek_v3_mqa_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gated_silu_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sample_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/per_token_per_channel_dequant_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/per_token_quant_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_matmul_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sigmoid_group_topk_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/expert_token_dispatcher_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_routing_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/linear_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cast_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quant_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fused_quant_linear_dequant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/group_linear_dequant_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scaled_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multinomial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scaled_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/div_op.cpp
)

target_include_directories(layers_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUST_PKG_PATH}/include
)

target_link_libraries(layers_lib PUBLIC
    opapi cust_opapi runtime_lib common_lib
)

set_target_properties(layers_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
