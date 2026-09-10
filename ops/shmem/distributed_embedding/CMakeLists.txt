cmake_minimum_required(VERSION 3.16)

include(cmake/dependencies.cmake)
project(distributed_embedding LANGUAGES CXX ASC)

add_subdirectory(op_host)
add_subdirectory(op_kernel)
add_custom_target(distributed_embedding ALL
    DEPENDS distributed_embedding_container distributed_embedding_kernel
)

option(DISTRIBUTED_EMBEDDING_BUILD_TESTS "Build single-PE and two-PE tests" ON)
if(DISTRIBUTED_EMBEDDING_BUILD_TESTS)
    add_subdirectory(test)
endif()
