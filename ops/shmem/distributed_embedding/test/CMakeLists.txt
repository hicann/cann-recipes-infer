cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(../cmake/dependencies.cmake)
    project(distributed_embedding_test LANGUAGES CXX ASC)
endif()

set(OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(HOST_INCLUDE_DIRS
    "${ASCEND_HOME_PATH}/include"
    "${CANN_HOST_DIR}/asc/include"
    "${CANN_HOST_DIR}/include"
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

foreach(scenario IN ITEMS d2d d2h)
    foreach(rank IN ITEMS 1 2)
        set(test_target "${scenario}_rank${rank}")
        add_executable(${test_target} ${scenario}/test_rank${rank}.cpp)
        target_compile_features(${test_target} PRIVATE cxx_std_17)
        target_compile_options(${test_target} PRIVATE -xc++)
        target_include_directories(${test_target} PRIVATE ${HOST_INCLUDE_DIRS})
        target_link_directories(${test_target} PRIVATE
            "${CANN_HOST_DIR}/lib64"
            "${SHMEM_LIBRARY_DIR}"
        )
        set_target_properties(${test_target} PROPERTIES
            BUILD_RPATH "${CANN_HOST_DIR}/lib64;${SHMEM_LIBRARY_DIR}"
        )
        target_link_libraries(${test_target} PRIVATE
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
        add_dependencies(${test_target} distributed_embedding_kernel)
    endforeach()
endforeach()
