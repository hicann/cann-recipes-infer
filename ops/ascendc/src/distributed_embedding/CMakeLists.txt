# The distributed embedding implementation has independent host and kernel
# build rules. Keep this file as the operator entry point discovered by the
# AscendC parent build, and keep the side-specific rules in their source dirs.

if (NOT DEFINED ASCEND_CANN_PACKAGE_PATH OR
        "${ASCEND_CANN_PACKAGE_PATH}" STREQUAL "")
    if (DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
    else ()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/latest")
    endif ()
endif ()

add_subdirectory(op_host)
add_subdirectory(op_kernel)

add_custom_target(distributed_embedding ALL
        DEPENDS distributed_embedding_container distributed_embedding_kernel
)
