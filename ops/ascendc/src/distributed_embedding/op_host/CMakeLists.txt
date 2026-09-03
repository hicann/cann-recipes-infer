# The distributed embedding container is a host-side direct-invocation
# library. Its source includes the Ascend C kernel definitions and is compiled
# with -xasc so it can contain <<<>>> launch expressions.

# Use CMake's ASC language so a parent CANN build that uses a normal host C++
# compiler still dispatches this translation unit to Bisheng.
if (DEFINED ENV{ASCEND_HOME_PATH})
    list(APPEND CMAKE_MODULE_PATH
            "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/asc_modules")
    set(CMAKE_ASC_STANDARD 17)
    enable_language(ASC)
endif ()

add_library(distributed_embedding_container STATIC
        dist_embedding_container.cpp
)

set_source_files_properties(dist_embedding_container.cpp PROPERTIES LANGUAGE ASC)

target_compile_features(distributed_embedding_container PRIVATE cxx_std_17)

target_compile_options(distributed_embedding_container PRIVATE
        -xasc
        --npu-arch=dav-3510
)

target_include_directories(distributed_embedding_container
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${ASCEND_CANN_PACKAGE_PATH}/include
)

# A caller can provide the SHMEM installation independently of the main
# ascendc operator package. The empty implementation does not link SHMEM, but
# exposing the include path keeps the public host header compatible with it.
if (DEFINED SHMEM_ROOT AND EXISTS "${SHMEM_ROOT}/include")
    target_include_directories(distributed_embedding_container
            PUBLIC
            ${SHMEM_ROOT}/include
    )
endif ()

install(TARGETS distributed_embedding_container
        ARCHIVE DESTINATION lib
)

install(FILES
        dist_embedding_container.h
        DESTINATION include/distributed_embedding
)
