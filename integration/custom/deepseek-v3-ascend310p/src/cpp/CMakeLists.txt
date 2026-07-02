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

# Minimum required CMake version
cmake_minimum_required(VERSION 3.15)

# Project Information
project(mlguider_llm_ascend LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build Options
option(BUILD_TESTS "Build Google tests" ON) # for test

# for Debug
# set(CMAKE_BUILD_TYPE Debug)
# for release
set(CMAKE_BUILD_TYPE Release)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Build type: Debug")
  add_compile_definitions(ENABLE_CHECK_MIACL_ERROR)
  add_compile_definitions(TM_LOG_ENABLE)
  add_compile_definitions(TM_LOG_PRINT_RANK)
else()
  message(STATUS "Build type: Release")
endif()

# add_compile_definitions(PLATFORM_910A)
# add_compile_definitions(USE_NTD)
add_compile_definitions(USE_PAGE_NZ)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -Wall")

# =================================================================================
# Set CANN include and library paths
set(INC_PATH
  $ENV{DDK_PATH}
  CACHE STRING "CANN include path")
set(LIB_PATH
  $ENV{NPU_HOST_LIB}
  CACHE STRING "CANN library path")
if(NOT INC_PATH)
  set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  message(STATUS "Default INC_PATH: ${INC_PATH}")
endif()
if(NOT LIB_PATH)
  set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub/")
  set(LIB_PATH1 "/usr/local/Ascend/ascend-toolkit/latest/atc/lib64/stub/")
  message(STATUS "Default LIB_PATH: ${LIB_PATH}")
endif()

# CANN custom package path
set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/micustomize/op_api")
message(STATUS "CUST_PKG_PATH: ${CUST_PKG_PATH}")

include_directories(
  ${INC_PATH}/runtime/include
  ${INC_PATH}/atc/include
  ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
  ${LIB_PATH}
  ${LIB_PATH1}
  ${CUST_PKG_PATH}/lib
)

set(MPI_INCLUDE_PATH
  "$ENV{MPI_PATH}/include"
  CACHE STRING "MPI include path")
set(MPI_LIB_PATH
  "$ENV{MPI_PATH}/lib"
  CACHE STRING "MPI library path")

add_subdirectory(mlguider_llm_ascend)
