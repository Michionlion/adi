cmake_minimum_required(VERSION 3.20)
project(adi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ADI_BUILD_TESTS "Build ADI tests" ON)
set(ADI_TEST_MODEL "" CACHE FILEPATH "Mach-1 GGUF used for checkpoint tests")

add_library(adi_runtime STATIC
    src/backend.cpp
    src/bpe.cpp
    src/chat.cpp
    src/executor.cpp
    src/generation.cpp
    src/gguf.cpp
    src/json.cpp
    src/kernels.cpp
    src/model.cpp
    src/server.cpp
    src/tokenizer.cpp
    src/unicode.cpp
    src/utf8.cpp
    src/version.cpp
)
target_include_directories(adi_runtime PUBLIC include)
if(MSVC)
    target_compile_options(adi_runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(adi_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(adi src/main.cpp)
target_link_libraries(adi PRIVATE adi_runtime)

if(WIN32)
    target_link_libraries(adi_runtime PUBLIC ws2_32)
endif()

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
    target_include_directories(adi_kernels_test PRIVATE src)
    add_test(NAME kernels COMMAND adi_kernels_test)

    add_executable(adi_executor_test tests/executor_test.cpp)
    target_link_libraries(adi_executor_test PRIVATE adi_runtime)
    add_test(NAME executor COMMAND adi_executor_test)

    add_executable(adi_generation_test tests/generation_test.cpp)
    target_link_libraries(adi_generation_test PRIVATE adi_runtime)
    add_test(NAME generation COMMAND adi_generation_test)

    add_executable(adi_json_test tests/json_test.cpp)
    target_link_libraries(adi_json_test PRIVATE adi_runtime)
    add_test(NAME json COMMAND adi_json_test)

    add_executable(adi_unicode_test tests/unicode_test.cpp)
    target_link_libraries(adi_unicode_test PRIVATE adi_runtime)
    target_include_directories(adi_unicode_test PRIVATE src)
    add_test(NAME unicode COMMAND adi_unicode_test)

    add_executable(adi_bpe_test tests/bpe_test.cpp)
    target_link_libraries(adi_bpe_test PRIVATE adi_runtime)
    target_include_directories(adi_bpe_test PRIVATE src)
    add_test(NAME bpe COMMAND adi_bpe_test)

    add_executable(adi_chat_test tests/chat_test.cpp)
    target_link_libraries(adi_chat_test PRIVATE adi_runtime)
    target_include_directories(adi_chat_test PRIVATE src)
    add_test(NAME chat COMMAND adi_chat_test)

    add_executable(adi_server_test tests/server_test.cpp)
    target_link_libraries(adi_server_test PRIVATE adi_runtime)
    target_include_directories(adi_server_test PRIVATE src)
    add_test(NAME server COMMAND adi_server_test)

    add_executable(adi_tokenizer_test tests/tokenizer_test.cpp)
    target_link_libraries(adi_tokenizer_test PRIVATE adi_runtime)
    target_include_directories(adi_tokenizer_test PRIVATE src)
    add_executable(adi_model_e2e_test tests/model_e2e_test.cpp)
    target_link_libraries(adi_model_e2e_test PRIVATE adi_runtime)
    add_executable(adi_model_batch_test tests/model_batch_test.cpp)
    target_link_libraries(adi_model_batch_test PRIVATE adi_runtime)
    if(ADI_TEST_MODEL)
        add_test(NAME tokenizer COMMAND adi_tokenizer_test ${ADI_TEST_MODEL})
        add_test(NAME model_e2e COMMAND adi_model_e2e_test ${ADI_TEST_MODEL})
        add_test(NAME model_batch COMMAND adi_model_batch_test ${ADI_TEST_MODEL})
    endif()

    foreach(test_target
            adi_smoke_test
            adi_gguf_test
            adi_kernels_test
            adi_executor_test
            adi_generation_test
            adi_json_test
            adi_unicode_test
            adi_bpe_test
            adi_chat_test
            adi_server_test
            adi_tokenizer_test
            adi_model_e2e_test
            adi_model_batch_test)
        if(MSVC)
            target_compile_options(${test_target} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${test_target} PRIVATE -UNDEBUG)
        endif()
    endforeach()

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME pack_mach
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/pack_mach.py --self-test
        )
    endif()
endif()
