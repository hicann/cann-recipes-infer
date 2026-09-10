# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# KvCompressEpilogV2 单元测试独立构建入口。
#
# 前置条件（环境变量）：
#   ASCEND_HOME_PATH       CANN toolkit 路径（source <ascend-toolkit>/set_env.sh 后自动设置）
#   ASCEND_CUSTOM_OPP_PATH 自定义算子 vendor 路径（source vendors/customize/bin/set_env.bash 后自动设置）
#
# 构建运行：
#   cd <repo>/ops/ascendc/src/kv_compress_epilog_v2/test
#   mkdir -p build && cd build
#   cmake .. && make
#   ./test_host_tiling && ./test_kernel

cmake_minimum_required(VERSION 3.16)
project(kv_compress_epilog_v2_test LANGUAGES CXX)

if(NOT DEFINED ASCEND_HOME_PATH OR ASCEND_HOME_PATH STREQUAL "")
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT DEFINED ASCEND_CUSTOM_OPP_PATH OR ASCEND_CUSTOM_OPP_PATH STREQUAL "")
    set(ASCEND_CUSTOM_OPP_PATH "$ENV{ASCEND_CUSTOM_OPP_PATH}")
endif()
if(ASCEND_HOME_PATH STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH is required (source <ascend-toolkit>/set_env.sh first)")
endif()
# ASCEND_CUSTOM_OPP_PATH 可能包含多个冒号分隔的路径，取第一个包含 op_api 的 vendor 路径
string(REPLACE ":" ";" _OPP_PATHS "${ASCEND_CUSTOM_OPP_PATH}")
set(_VENDOR_DIR "")
foreach(_dir ${_OPP_PATHS})
    if(EXISTS "${_dir}/op_api/include" AND _VENDOR_DIR STREQUAL "")
        set(_VENDOR_DIR "${_dir}")
    endif()
endforeach()
if(_VENDOR_DIR STREQUAL "")
    message(FATAL_ERROR "ASCEND_CUSTOM_OPP_PATH with op_api is required "
            "(source vendors/customize/bin/set_env.bash first)")
endif()
message(STATUS "CANN toolkit: ${ASCEND_HOME_PATH}")
message(STATUS "custom op vendor: ${_VENDOR_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(HOST_INCLUDE_DIRS
    "${ASCEND_HOME_PATH}/include"
    "${_VENDOR_DIR}/op_api/include"
)

# Host UT：GetWorkspaceSize 正/负例驱动 host tiling 校验
add_executable(test_host_tiling test_host_tiling.cpp)
target_include_directories(test_host_tiling PRIVATE ${HOST_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(test_host_tiling PRIVATE
    "${ASCEND_HOME_PATH}/lib64"
    "${_VENDOR_DIR}/op_api/lib"
)
set_target_properties(test_host_tiling PROPERTIES
    BUILD_RPATH "${ASCEND_HOME_PATH}/lib64;${_VENDOR_DIR}/op_api/lib"
)
target_link_libraries(test_host_tiling PRIVATE ascendcl nnopbase cust_opapi pthread dl)

# Kernel UT：两段式执行并逐字节校验 cache 写回（groupSize 16/32、尾 chunk、slot 语义）
add_executable(test_kernel test_kernel.cpp)
target_include_directories(test_kernel PRIVATE ${HOST_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(test_kernel PRIVATE
    "${ASCEND_HOME_PATH}/lib64"
    "${_VENDOR_DIR}/op_api/lib"
)
set_target_properties(test_kernel PROPERTIES
    BUILD_RPATH "${ASCEND_HOME_PATH}/lib64;${_VENDOR_DIR}/op_api/lib"
)
target_link_libraries(test_kernel PRIVATE ascendcl nnopbase cust_opapi pthread dl)
