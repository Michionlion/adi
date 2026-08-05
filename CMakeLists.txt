cmake_minimum_required(VERSION 3.20)
project(adi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ADI_BUILD_TESTS "Build ADI tests" ON)

add_library(adi_runtime STATIC
    src/executor.cpp
    src/gguf.cpp
    src/kernels.cpp
    src/model.cpp
    src/version.cpp
)
target_include_directories(adi_runtime PUBLIC include)
target_compile_options(adi_runtime PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_executable(adi src/main.cpp)
target_link_libraries(adi PRIVATE adi_runtime)

if(ADI_BUILD_TESTS)
    enable_testing()
    add_executable(adi_smoke_test tests/smoke.cpp)
    target_link_libraries(adi_smoke_test PRIVATE adi_runtime)
    add_test(NAME smoke COMMAND adi_smoke_test)

    add_executable(adi_gguf_test tests/gguf_test.cpp)
    target_link_libraries(adi_gguf_test PRIVATE adi_runtime)
    add_test(NAME gguf COMMAND adi_gguf_test)

    add_executable(adi_kernels_test tests/kernels_test.cpp)
    target_link_libraries(adi_kernels_test PRIVATE adi_runtime)
    add_test(NAME kernels COMMAND adi_kernels_test)

    add_executable(adi_executor_test tests/executor_test.cpp)
    target_link_libraries(adi_executor_test PRIVATE adi_runtime)
    add_test(NAME executor COMMAND adi_executor_test)

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME pack_mach
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/pack_mach.py --self-test
        )
    endif()
endif()
