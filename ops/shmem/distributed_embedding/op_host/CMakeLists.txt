# The distributed embedding container is a host-side direct-invocation
# library. Its source includes the Ascend C kernel definitions and is compiled
# with -xasc so it can contain <<<>>> launch expressions.

add_library(distributed_embedding_container STATIC
        dist_embedding_container.cpp
)

set_source_files_properties(dist_embedding_container.cpp PROPERTIES LANGUAGE ASC)

target_compile_features(distributed_embedding_container PRIVATE cxx_std_17)
target_link_libraries(distributed_embedding_container PUBLIC m)

target_compile_options(distributed_embedding_container PRIVATE
        -xasc
        --npu-arch=${DISTRIBUTED_EMBEDDING_NPU_ARCH}
)

target_compile_definitions(distributed_embedding_container PRIVATE USE_SIMT)

target_include_directories(distributed_embedding_container
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${ASCEND_CANN_PACKAGE_PATH}/include
)

# The host translation unit also compiles the kernel definitions, which require
# SHMEM's device implementation headers in addition to its public headers.
if (DEFINED SHMEM_ROOT AND EXISTS "${SHMEM_ROOT}/include")
    target_include_directories(distributed_embedding_container
            PUBLIC
            ${SHMEM_ROOT}/include
            PRIVATE
            ${SHMEM_ROOT}/src
            ${SHMEM_ROOT}/src/device
            ${SHMEM_ROOT}/src/device_simt
    )
endif ()

install(TARGETS distributed_embedding_container
        ARCHIVE DESTINATION lib
)

install(FILES
        dist_embedding_container.h
        DESTINATION include/distributed_embedding
)
