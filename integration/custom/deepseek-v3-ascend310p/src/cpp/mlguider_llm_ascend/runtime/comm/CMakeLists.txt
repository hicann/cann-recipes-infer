# coding=utf-8
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
add_library(comm_lib STATIC comm_domain_manager.cpp)

target_include_directories(comm_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${MPI_INCLUDE_PATH})

target_link_directories(comm_lib PUBLIC ${MPI_LIB_PATH})

target_link_libraries(comm_lib PUBLIC ascendcl hccl config_lib mpi)
set_target_properties(comm_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
