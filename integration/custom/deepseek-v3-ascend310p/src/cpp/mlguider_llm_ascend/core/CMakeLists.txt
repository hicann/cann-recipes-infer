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
add_subdirectory(layers)

find_package(Boost REQUIRED COMPONENTS iostreams)

add_library(core_lib OBJECT
    text_generator.cpp
    model_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/models/deepseek_v3_w8a8_model.cpp
)
message(STATUS "CUST_PKG_PATH: ${CUST_PKG_PATH}")
target_include_directories(core_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)

target_include_directories(core_lib PUBLIC ${MPI_INCLUDE_PATH})
target_link_directories(core_lib PUBLIC ${MPI_LIB_PATH})

target_link_libraries(core_lib
    PUBLIC
    comm_lib
    data_processor_lib
    layers_lib
    Boost::iostreams
    mpi
)
set_target_properties(core_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
