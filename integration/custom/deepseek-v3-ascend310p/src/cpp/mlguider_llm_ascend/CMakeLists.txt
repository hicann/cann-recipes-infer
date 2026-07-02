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

set(TARGET_NAME mlguider_llm_ascend)

set(SHARED_LIB_TARGET ${TARGET_NAME}_lib)
set(SHARED_LIB_TARGET ${SHARED_LIB_TARGET} PARENT_SCOPE)

add_compile_definitions(USE_POOL)

add_subdirectory(config)
add_subdirectory(runtime)
add_subdirectory(common)
add_subdirectory(data_processor)
add_subdirectory(core)


set(LLM_LINK_LIBS core_lib mpi)


add_library(${SHARED_LIB_TARGET} SHARED)
target_include_directories(${SHARED_LIB_TARGET} PUBLIC ${MPI_INCLUDE_PATH})
target_link_directories(${SHARED_LIB_TARGET} PUBLIC ${MPI_LIB_PATH})
target_link_libraries(${SHARED_LIB_TARGET} PUBLIC ${LLM_LINK_LIBS})

add_executable(main_inference main_inference.cpp)

target_include_directories(main_inference PRIVATE ${MPI_INCLUDE_PATH})
target_link_directories(main_inference PRIVATE ${MPI_LIB_PATH})
target_link_libraries(main_inference PUBLIC ${SHARED_LIB_TARGET})
