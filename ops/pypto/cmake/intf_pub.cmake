# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

include_guard(GLOBAL)
if (TARGET intf_pub)
    return()
endif()

# intf_pub_base
add_library(intf_pub_base INTERFACE)
target_compile_options(intf_pub_base INTERFACE
        -Wall
        -fPIC
        $<IF:$<STREQUAL:${CMAKE_SYSTEM_NAME},centos>,-fstack-protector-all,-fstack-protector-strong>
)
target_compile_definitions(intf_pub_base INTERFACE
        _GLIBCXX_USE_CXX11_ABI=0
        $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
        $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
        LINUX=0
        LOG_CPP
        $<$<STREQUAL:${ENABLE_TEST},True>:ENABLE_AIR_D_TEST>
        $<$<BOOL:${ENABLE_TEST}>:SUPPORT_LARGE_MODEL_ENABLE=1>
)
target_link_options(intf_pub_base INTERFACE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-Wl,--build-id=none>
)
target_link_directories(intf_pub_base INTERFACE
)
target_link_libraries(intf_pub_base INTERFACE
        -lpthread
)

# intf_pub c++11
add_library(intf_pub INTERFACE)
target_compile_features(intf_pub INTERFACE cxx_std_11)

target_link_libraries(intf_pub INTERFACE
        $<BUILD_INTERFACE:intf_pub_base>
)

# intf_pub c++17
add_library(intf_pub_cxx17 INTERFACE)
target_compile_options(intf_pub_cxx17 INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)
target_link_libraries(intf_pub_cxx17 INTERFACE
        $<BUILD_INTERFACE:intf_pub_base>
)

# intf_pub c++14
add_library(intf_pub_cxx14 INTERFACE)
target_compile_options(intf_pub_cxx14 INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)
target_link_libraries(intf_pub_cxx14 INTERFACE
        $<BUILD_INTERFACE:intf_pub_base>
)

# ccache
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND} CACHE PATH "cache Compiler")
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND} CACHE PATH "cache Compiler")
endif()
message(STATUS "CMAKE_C_COMPILER_LAUNCHER:${CMAKE_C_COMPILER_LAUNCHER}")
message(STATUS "CMAKE_CXX_COMPILER_LAUNCHER:${CMAKE_CXX_COMPILER_LAUNCHER}")
