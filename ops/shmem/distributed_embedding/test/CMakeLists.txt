cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(../cmake/dependencies.cmake)
    project(distributed_embedding_test LANGUAGES CXX ASC)
endif()

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

if(NOT TARGET distributed_embedding_container)
    add_subdirectory("${OP_DIR}/op_host" "distributed_embedding_host")
    add_subdirectory("${OP_DIR}/op_kernel" "distributed_embedding_kernel")
endif()

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
    distributed_embedding_container
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
    distributed_embedding_container
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
