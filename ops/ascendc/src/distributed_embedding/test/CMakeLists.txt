cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED ASCEND_HOME_PATH OR ASCEND_HOME_PATH STREQUAL "")
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT DEFINED SHMEM_ROOT OR SHMEM_ROOT STREQUAL "")
    set(SHMEM_ROOT "$ENV{SHMEM_ROOT}")
endif()
if(SHMEM_ROOT STREQUAL "" AND DEFINED ENV{SHMEM_HOME_PATH})
    set(SHMEM_ROOT "$ENV{SHMEM_HOME_PATH}/shmem")
endif()
set(SHMEM_BACKEND "950" CACHE STRING "SHMEM backend directory name")

if(ASCEND_HOME_PATH STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH is required")
endif()
if(SHMEM_ROOT STREQUAL "")
    message(FATAL_ERROR "SHMEM_ROOT is required")
endif()
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")

set(BISHENG "${ASCEND_HOME_PATH}/x86_64-linux/ccec_compiler/bin/bisheng")
if(NOT DEFINED SHMEM_LIBRARY_DIR OR SHMEM_LIBRARY_DIR STREQUAL "")
    set(SHMEM_LIBRARY_DIR "${SHMEM_ROOT}/lib")
endif()
if(NOT EXISTS "${BISHENG}")
    message(FATAL_ERROR "Bisheng compiler not found: ${BISHENG}")
endif()
if(NOT EXISTS "${SHMEM_ROOT}/include/shmem.h")
    message(FATAL_ERROR "SHMEM host headers not found: ${SHMEM_ROOT}/include/shmem.h")
endif()
if(NOT EXISTS "${SHMEM_LIBRARY_DIR}/libshmem.so")
    message(FATAL_ERROR "SHMEM library not found: ${SHMEM_LIBRARY_DIR}/libshmem.so")
endif()
message(STATUS "SHMEM library directory: ${SHMEM_LIBRARY_DIR}")

set(CMAKE_CXX_COMPILER "${BISHENG}" CACHE FILEPATH "Bisheng compiler" FORCE)
project(distributed_embedding_test LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH
    "${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake/asc_modules"
)
set(CMAKE_ASC_STANDARD 17)
enable_language(ASC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(HOST_INCLUDE_DIRS
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/x86_64-linux/asc/include"
    "${ASCEND_HOME_PATH}/x86_64-linux/include"
    "${SHMEM_ROOT}/include"
    "${SHMEM_ROOT}/src"
    "${SHMEM_ROOT}/src/device"
    "${SHMEM_ROOT}/src/device_simt"
    "${OP_DIR}/op_host"
    "${OP_DIR}/op_kernel"
)

add_library(distributed_embedding_host STATIC
    "${OP_DIR}/op_host/dist_embedding_container.cpp"
)
target_compile_features(distributed_embedding_host PRIVATE cxx_std_17)
set_source_files_properties(
    "${OP_DIR}/op_host/dist_embedding_container.cpp"
    PROPERTIES LANGUAGE ASC
)
target_compile_options(distributed_embedding_host PRIVATE
    -xasc
    --npu-arch=dav-3510
)
target_compile_definitions(distributed_embedding_host PRIVATE USE_SIMT)
target_include_directories(distributed_embedding_host PUBLIC ${HOST_INCLUDE_DIRS})

add_subdirectory("${OP_DIR}/op_kernel" "distributed_embedding_kernel")

add_executable(test_rank1 test_rank1.cpp)
target_compile_features(test_rank1 PRIVATE cxx_std_17)
target_compile_options(test_rank1 PRIVATE -xc++)
target_include_directories(test_rank1 PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(test_rank1 PRIVATE
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64"
    "${SHMEM_LIBRARY_DIR}"
)
set_target_properties(test_rank1 PROPERTIES
    BUILD_RPATH "${ASCEND_HOME_PATH}/x86_64-linux/lib64;${SHMEM_LIBRARY_DIR}"
)
target_link_libraries(test_rank1 PRIVATE
    distributed_embedding_host
    shmem
    ascendcl
    tiling_api
    platform
    unified_dlog
    error_manager
    profapi
    ascend_dump
    mmpa
    c_sec
    nnopbase
    runtime
    pthread
    dl
)
add_dependencies(test_rank1 distributed_embedding_kernel)

add_executable(test_rank2 test_rank2.cpp)
target_compile_features(test_rank2 PRIVATE cxx_std_17)
target_compile_options(test_rank2 PRIVATE -xc++)
target_include_directories(test_rank2 PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(test_rank2 PRIVATE
    "${ASCEND_HOME_PATH}/x86_64-linux/lib64"
    "${SHMEM_LIBRARY_DIR}"
)
set_target_properties(test_rank2 PROPERTIES
    BUILD_RPATH "${ASCEND_HOME_PATH}/x86_64-linux/lib64;${SHMEM_LIBRARY_DIR}"
)
target_link_libraries(test_rank2 PRIVATE
    distributed_embedding_host
    shmem
    ascendcl
    tiling_api
    platform
    unified_dlog
    error_manager
    profapi
    ascend_dump
    mmpa
    c_sec
    nnopbase
    runtime
    pthread
    dl
)
add_dependencies(test_rank2 distributed_embedding_kernel)
