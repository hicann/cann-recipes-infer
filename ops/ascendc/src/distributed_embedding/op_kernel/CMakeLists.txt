# Kernel sources are compiled directly because this container intentionally has
# no opbuild registration metadata. The default architecture is the SHMEM
# supported Ascend 950 architecture; callers may override it explicitly.

set(DISTRIBUTED_EMBEDDING_NPU_ARCH "dav-3510" CACHE STRING
        "NPU architecture for the distributed embedding kernel")
set(DISTRIBUTED_EMBEDDING_BISHENG
        "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ccec_compiler/bin/bisheng")
set(DISTRIBUTED_EMBEDDING_KERNEL_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/distributed_hashtable.h
        ${CMAKE_CURRENT_SOURCE_DIR}/distributed_hashtable.cpp
)
set(DISTRIBUTED_EMBEDDING_KERNEL_OBJECT
        "${CMAKE_CURRENT_BINARY_DIR}/distributed_hashtable.o")

add_custom_command(
        OUTPUT ${DISTRIBUTED_EMBEDDING_KERNEL_OBJECT}
        COMMAND ${DISTRIBUTED_EMBEDDING_BISHENG}
                -xasc
                --npu-arch=${DISTRIBUTED_EMBEDDING_NPU_ARCH}
                -std=c++17
                -O0
                -DUSE_SIMT
                -I${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
                -I${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/common
                -I${SHMEM_ROOT}/include
                -I${SHMEM_ROOT}/src/device
                -I${SHMEM_ROOT}/src/device_simt
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/distributed_hashtable.cpp
                -o ${DISTRIBUTED_EMBEDDING_KERNEL_OBJECT}
        DEPENDS ${DISTRIBUTED_EMBEDDING_KERNEL_HEADERS}
        VERBATIM
)

add_custom_target(distributed_embedding_kernel
        DEPENDS ${DISTRIBUTED_EMBEDDING_KERNEL_OBJECT}
)

# Keep kernel sources visible to IDEs and downstream direct-invoke projects.
add_custom_target(distributed_embedding_kernel_sources
        SOURCES ${DISTRIBUTED_EMBEDDING_KERNEL_HEADERS}
)

install(FILES
        distributed_hashtable.h
        distributed_hashtable.cpp
        DESTINATION include/distributed_embedding
)
