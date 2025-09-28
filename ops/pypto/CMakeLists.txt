# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.14)
project(tile_fwk_ops)

option(BUILD_OPEN_PROJECT "Build open ascend ops project."  FALSE)
option(ENABLE_BUILD_HOST "Build host."  TRUE)
option(ENABLE_BUILD_BINARY "Build binary."  FALSE)

set(TILE_FWK_OPS_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_VERBOSE_MAKEFILE True)
set(INSTALL_LIBRARY_DIR lib)
set(CMAKE_LIBRARY_PATH ${ASCEND_HOME_PATH}/lib64)

if (NOT DEFINED CMAKE_MODULE_PATH)
        set(CMAKE_MODULE_PATH
                ${CMAKE_CURRENT_LIST_DIR}/cmake/modules
        )
endif()

if (NOT DEFINED CMAKE_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH
                ${ASCEND_HOME_PATH}
        )
endif()

# cann packages
include(cmake/find_dependence_packages.cmake)

string(REPLACE "," ";" COMPUTE_UNIT_LIST "${ASCEND_COMPUTE_UNIT}")

if (ENABLE_BUILD_BINARY)
        # tile_fwk_impl
        add_library(tile_fwk_impl SHARED)

        target_compile_options(tile_fwk_impl PRIVATE
                -fvisibility=hidden
        )

        target_link_directories(tile_fwk_impl PRIVATE
                ${ASCEND_HOME_PATH}/lib64
        )

        target_link_libraries(tile_fwk_impl PRIVATE
                $<BUILD_INTERFACE:intf_pub_cxx17>
                c_sec
                ascendalog
                tile_fwk_interface
                tile_fwk_operator
                nlohmann_json::nlohmann_json
        )

        target_include_directories(tile_fwk_impl PRIVATE
                ${ASCEND_HOME_PATH}/include
        )

        install(TARGETS tile_fwk_impl OPTIONAL
                EXPORT tile_fwk_impl-targets
                LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        )

        # nsa op kernel binary
        foreach(COMPUTE_UNIT IN LISTS COMPUTE_UNIT_LIST)
                # deepseek_indexer_attention
                add_custom_command(
                        TARGET tile_fwk_impl POST_BUILD
                        COMMAND bash ${TILE_FWK_OPS_ROOT_PATH}/scripts/kernels_compile.sh
                                ${COMPUTE_UNIT}
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/deepseek_indexer_attention/op_kernel/bin_param
                                ${CMAKE_CURRENT_SOURCE_DIR}/python/tile_fwk_compiler.py
                                ${CMAKE_CURRENT_BINARY_DIR}/libtile_fwk_impl.so
                                ascendcpp_compile_op
                                ${BINARY_OUTPUT_PATH}/${COMPUTE_UNIT}/deepseek_indexer_attention
                        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/insert_simplified_keys.py
                                -p ${BINARY_OUTPUT_PATH}/${COMPUTE_UNIT}/deepseek_indexer_attention
                        COMMAND ${CMAKE_COMMAND} -E make_directory ${BINARY_OUTPUT_PATH}/config/${COMPUTE_UNIT}/
                        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/gen_binary_info_config.py
                                -p ${BINARY_OUTPUT_PATH}/${COMPUTE_UNIT}/
                                -o ${BINARY_OUTPUT_PATH}/config/${COMPUTE_UNIT}/ -s ${COMPUTE_UNIT}                        
                )
                # lightning_indexer_pto
                add_custom_command(
                        TARGET tile_fwk_impl POST_BUILD
                        COMMAND bash ${TILE_FWK_OPS_ROOT_PATH}/scripts/kernels_compile.sh
                                ${COMPUTE_UNIT}
                                ${CMAKE_CURRENT_SOURCE_DIR}/src/lightning_indexer_pto/op_kernel/bin_param
                                ${CMAKE_CURRENT_SOURCE_DIR}/python/tile_fwk_compiler.py
                                ${CMAKE_CURRENT_BINARY_DIR}/libtile_fwk_impl.so
                                ascendcpp_compile_op
                                ${BINARY_OUTPUT_PATH}/${COMPUTE_UNIT}/lightning_indexer_pto
                        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/insert_simplified_keys.py
                                -p ${BINARY_OUTPUT_PATH}/${COMPUTE_UNIT}/lightning_indexer_pto
                        COMMAND ${CMAKE_COMMAND} -E make_directory ${BINARY_OUTPUT_PATH}/config/${COMPUTE_UNIT}/
                        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils/gen_binary_info_config.py
                                -p ${BINARY_OUTPUT_PATH}/${COMPUTE_UNIT}/
                                -o ${BINARY_OUTPUT_PATH}/config/${COMPUTE_UNIT}/ -s ${COMPUTE_UNIT}                        
                )
        endforeach()
endif()

if (BUILD_OPEN_PROJECT AND ENABLE_BUILD_HOST)
        # op proto
        add_library(opsproto_rt2.0 SHARED)
        target_compile_options(opsproto_rt2.0 PRIVATE
                -g
                -fvisibility=hidden
                -Dgoogle=ascend_private
        )
        target_compile_definitions(opsproto_rt2.0 PRIVATE
                LOG_CPP
                PROCESS_LOG
        )
        target_link_directories(opsproto_rt2.0 PRIVATE
                ${ASCEND_HOME_PATH}/lib64
        )
        target_link_libraries(opsproto_rt2.0 PRIVATE
                $<BUILD_INTERFACE:intf_pub>
                $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
                -Wl,--whole-archive
                rt2_registry
                -Wl,--no-whole-archive
                -Wl,--no-as-needed
                exe_graph
                graph
                graph_base
                register
                alog
                error_manager
                platform
                -Wl,--as-needed
                c_sec
        )
        set_target_properties(opsproto_rt2.0 PROPERTIES OUTPUT_NAME
                cust_opsproto_rt2.0
        )
        install(TARGETS opsproto_rt2.0
                LIBRARY DESTINATION op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )

        # op proto
        add_library(opmaster_ct SHARED)
        target_compile_options(opmaster_ct PRIVATE
                -fvisibility=hidden
                -Dgoogle=ascend_private
        )
        target_compile_definitions(opmaster_ct PRIVATE
                LOG_CPP
                PROCESS_LOG
        )
        target_link_directories(opmaster_ct PRIVATE
                ${ASCEND_HOME_PATH}/lib64
        )
        target_link_libraries(opmaster_ct PRIVATE
                $<BUILD_INTERFACE:intf_pub>
                $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
                -Wl,--whole-archive
                rt2_registry
                -Wl,--no-whole-archive
                -Wl,--no-as-needed
                exe_graph
                graph
                graph_base
                register
                alog
                error_manager
                platform
                -Wl,--as-needed
                c_sec
        )
        set_target_properties(opmaster_ct PROPERTIES OUTPUT_NAME
                cust_opmaster_ct
        )
        install(TARGETS opmaster_ct
                LIBRARY DESTINATION op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )

        # op tiling
        add_library(opmaster_rt2.0 SHARED)
        target_compile_options(opmaster_rt2.0 PRIVATE
                -fvisibility=hidden
                -Dgoogle=ascend_private
        )
        target_compile_definitions(opmaster_rt2.0 PRIVATE
                LOG_CPP
                PROCESS_LOG
        )
        target_link_directories(opmaster_rt2.0 PRIVATE
                ${ASCEND_HOME_PATH}/lib64
        )
        target_link_libraries(opmaster_rt2.0 PRIVATE
                $<BUILD_INTERFACE:intf_pub>
                $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
                -Wl,--whole-archive
                rt2_registry
                -Wl,--no-whole-archive
                -Wl,--no-as-needed
                graph
                graph_base
                exe_graph
                platform
                register
                alog
                error_manager
                -Wl,--as-needed
                -Wl,--whole-archive
                -Wl,--no-whole-archive
                mmpa
                c_sec
        )
        set_target_properties(opmaster_rt2.0 PROPERTIES OUTPUT_NAME
                cust_opmaster_rt2.0
        )
        install(TARGETS opmaster_rt2.0
                LIBRARY DESTINATION op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
        )

        foreach(COMPUTE_UNIT IN LISTS COMPUTE_UNIT_LIST)
                add_custom_target(OP_STORE_${COMPUTE_UNIT}_2 ALL
                        COMMAND echo "Begin to build op store of ${COMPUTE_UNIT}."
                        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/op_impl/ai_core/tbe/config/${COMPUTE_UNIT}
                        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/deepseek_indexer_attention/op_host/op_store/deepseek_indexer_attention_store.json ${CMAKE_INSTALL_PREFIX}/op_impl/ai_core/tbe/config/${COMPUTE_UNIT}/
                        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/lightning_indexer_pto/op_host/op_store/lightning_indexer_store.json ${CMAKE_INSTALL_PREFIX}/op_impl/ai_core/tbe/config/${COMPUTE_UNIT}/
                        COMMAND echo "Finish building op store of ${COMPUTE_UNIT}."
                )
        endforeach()
endif ()

add_subdirectory(src/lightning_indexer_pto)
add_subdirectory(src/deepseek_indexer_attention)
