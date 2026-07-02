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
add_subdirectory(logger)
add_subdirectory(comm)


add_library(runtime_lib OBJECT file_utils.cpp)


target_include_directories(
  runtime_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} # 暴露core/ 目录为include路径
)

target_link_libraries(runtime_lib PUBLIC ascendcl logger_lib comm_lib)


set_target_properties(runtime_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
