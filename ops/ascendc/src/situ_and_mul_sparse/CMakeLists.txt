add_ops_compile_options(
        OP_NAME SituAndMulSparse
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE op_host/situ_and_mul_sparse_def.cpp)
target_sources(optiling PRIVATE op_host/situ_and_mul_sparse_tiling.cpp)
if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE op_host/situ_and_mul_sparse_tiling.cpp)
endif ()
target_include_directories(optiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
target_sources(opsproto PRIVATE op_host/situ_and_mul_sparse_proto.cpp)
