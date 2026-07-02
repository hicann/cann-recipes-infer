# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)
project(NpuOpsTransformerExt)
option(BUILD_TORCH_OPS "Build torch ops project (PyTorch extension style)" OFF)
message(STATUS "Build torch ops: ${BUILD_TORCH_OPS}")

if(NOT BUILD_TORCH_OPS)
    message(STATUS "BUILD_TORCH_OPS is OFF, exiting CMake configuration")
    return()
endif()

#================================
# Torch Ops 项目特定配置
#================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release/Debug)" FORCE)
endif()

#================================
# Ascend 环境配置
#================================
message(STATUS "$ENV{ASCEND_HOME_PATH}")
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend installation path" FORCE)
else()
    set(ASCEND_HOME "/usr/local/Ascend/latest" CACHE PATH "Ascend installation path")
endif()

set(BISHENG "${ASCEND_HOME}/aarch64-linux/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
message(STATUS "ASCEND_HOME = ${ASCEND_HOME}")
message(STATUS "BISHENG     = ${BISHENG}")
# 设置编译器为 bisheng
set(CMAKE_C_COMPILER   ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER       ${BISHENG})

# Python 配置
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3_EXECUTABLE   = ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES    = ${Python3_LIBRARIES}")

# Torch 配置
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Torch_DIR "${TORCH_CMAKE_PATH}/Torch")
find_package(Torch REQUIRED)
message(STATUS "Torch_DIR       = ${TORCH_DIR}")
message(STATUS "Torch_LIBRARIES = ${TORCH_LIBRARIES}")
message(STATUS "Torch_INCLUDES  = ${TORCH_INCLUDE_DIRS}")

# Torch-NPU 配置
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_NPU_INCLUDE_PATH "${TORCH_NPU_PATH}/include")
set(TORCH_NPU_LIB_PATH     "${TORCH_NPU_PATH}/lib")

message(STATUS "TORCH_NPU_PATH         = ${TORCH_NPU_PATH}")
message(STATUS "TORCH_NPU_INCLUDE_DIR  = ${TORCH_NPU_INCLUDE_PATH}")
message(STATUS "TORCH_NPU_LIB_DIR      = ${TORCH_NPU_LIB_PATH}")

# Torch Ops 公共配置
set(COMMON_INCLUDE_DIRS
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME}/compiler/ascendc/include/basic_api/impl
    ${ASCEND_HOME}/compiler/ascendc/include/basic_api/interface
    ${ASCEND_HOME}/compiler/ascendc/include/highlevel_api
    ${ASCEND_HOME}/compiler/ascendc/include/highlevel_api/tiling
    ${ASCEND_HOME}/compiler/ascendc/impl/aicore/basic_api
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/inc
)

set(COMMON_LINK_DIRS
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_HOME}/lib64
)

set(COMMON_LINK_LIBS
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
    ${Python3_LIBRARIES}
)

set(COMMON_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O2
    -fdiagnostics-color=always
    -DPy_LIMITED_API=0x03090000
    -w
    -DTORCH_MODE
)

message(STATUS "Building Torch Ops project...")

# These macros are stubs added for compatibility purporses
macro(add_graph_plugin_sources)
    message(STATUS "skip add_graph_plugin_sources...")
endmacro()

macro(add_modules_sources)
    message(STATUS "skip add_modules_sources...")
endmacro()

set(PARENT_DIRS attention gated_delta_net mamba ffn gmm mc2 moe posembedding)
set(OPERATOR_TARGETS "")

foreach(DIR_NAME ${PARENT_DIRS})
    set(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${DIR_NAME}")

    if(IS_DIRECTORY ${PARENT_DIR})
        message(STATUS "Scanning ${DIR_NAME} directory ...")
        file(GLOB SUB_DIRS "${PARENT_DIR}/*")
        foreach(SUB_DIR ${SUB_DIRS})
            if(IS_DIRECTORY ${SUB_DIR} AND EXISTS "${SUB_DIR}/CMakeLists.txt")
                # 临时变量用于接收子CMake配置
                unset(OPERATOR_CONFIG)

                get_filename_component(OP_DIR_NAME ${SUB_DIR} NAME)
                set(BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/npu_ops_transformer_ext/csrc/${OP_DIR_NAME}")
                add_subdirectory(${SUB_DIR} ${BINARY_DIR})

                # 使用子CMake提供的配置
                if(OPERATOR_CONFIG)
                    string(REPLACE ":" ";" CONFIG_LIST ${OPERATOR_CONFIG})
                    list(GET CONFIG_LIST 0 OP_NAME)
                    list(GET CONFIG_LIST 1 OP_TARGET)
                    message(STATUS "Found operator: ${OP_NAME}")
                    list(APPEND OPERATOR_TARGETS $<TARGET_OBJECTS:${OP_TARGET}>)
                else()
                    message(WARNING "No operator config found in ${SUB_DIR}")
                endif()
            endif()
        endforeach()
    endif()
endforeach()

message(STATUS "Discovered torch operator targets: ${OPERATOR_TARGETS}")

# 主库源文件
set(MAIN_LIBRARY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/npu_ops_transformer_ext/npu_ops_def.cpp")

# 创建主库
add_library(_C SHARED
    ${MAIN_LIBRARY_SOURCE}
    ${OPERATOR_TARGETS}
)

set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LINK_FLAGS "--cce-fatobj-link"
    PREFIX ""
    SUFFIX ".abi3.so"
    OUTPUT_NAME "_C"
)

target_compile_options(_C PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(_C PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(_C PRIVATE ${COMMON_LINK_LIBS})
return()
