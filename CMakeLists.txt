cmake_minimum_required(VERSION 3.20)
project(adi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ADI_BUILD_TESTS "Build ADI tests" ON)
set(ADI_TEST_MODEL "" CACHE FILEPATH "Mach-1 GGUF used for checkpoint tests")

add_library(adi_runtime STATIC
    src/attention.cpp
    src/backend.cpp
    src/bpe.cpp
    src/chat.cpp
    src/codec_cache.cpp
    src/executor.cpp
    src/expert_batch.cpp
    src/expert_batch_x86_avx2.cpp
    src/expert_batch_x86_avx512.cpp
    src/generation.cpp
    src/gguf.cpp
    src/json.cpp
    src/kernels.cpp
    src/model.cpp
    src/ne_batch.cpp
    src/ne_batch_x86_avx2.cpp
    src/ne_batch_x86_avx512.cpp
    src/options.cpp
    src/parallel.cpp
    src/profiling.cpp
    src/server.cpp
    src/simd.cpp
    src/simd_arm.cpp
    src/simd_x86.cpp
    src/simd_x86_avx2.cpp
    src/simd_x86_avx512.cpp
    src/simd_x86_vbmi.cpp
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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64|x64|i[3-6]86)")
    if(MSVC)
        set_source_files_properties(
            src/simd_x86_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties(
            src/simd_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set_source_files_properties(
                src/simd_x86_vbmi.cpp
                PROPERTIES COMPILE_OPTIONS
                    "/arch:AVX512;/clang:-mavx512vbmi")
        else()
            set_source_files_properties(
                src/simd_x86_vbmi.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
        endif()
        # /fp:precise already forbids contraction across a statement, so the
        # separate multiply and add in the batch kernels stay separate.
        set_source_files_properties(
            src/expert_batch_x86_avx2.cpp
            src/ne_batch_x86_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:precise")
        set_source_files_properties(
            src/expert_batch_x86_avx512.cpp
            src/ne_batch_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX512;/fp:precise")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(
            src/simd_x86_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2")
        set_source_files_properties(
            src/simd_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw")
        set_source_files_properties(
            src/simd_x86_vbmi.cpp
            PROPERTIES COMPILE_OPTIONS
                "-mavx512f;-mavx512bw;-mavx512vbmi")
        # The batch kernels must not fuse their multiply and add into an FMA:
        # that would round differently from the scalar reference and break
        # bit-exactness. AVX2 and AVX-512 both make FMA available, so the
        # contraction has to be disabled explicitly.
        set_source_files_properties(
            src/expert_batch_x86_avx2.cpp
            src/ne_batch_x86_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
        set_source_files_properties(
            src/expert_batch_x86_avx512.cpp
            src/ne_batch_x86_avx512.cpp
            PROPERTIES COMPILE_OPTIONS
                "-mavx512f;-mavx512bw;-ffp-contract=off")
    endif()
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

    add_executable(adi_int5_batch_test tests/int5_batch_test.cpp)
    target_link_libraries(adi_int5_batch_test PRIVATE adi_runtime)
    target_include_directories(adi_int5_batch_test PRIVATE src)
    add_test(NAME int5_batch COMMAND adi_int5_batch_test)

    add_executable(adi_parallel_test tests/parallel_test.cpp)
    target_link_libraries(adi_parallel_test PRIVATE adi_runtime)
    target_include_directories(adi_parallel_test PRIVATE src)
    add_test(NAME parallel COMMAND adi_parallel_test)

    add_executable(adi_ne_trellis_test tests/ne_trellis_test.cpp)
    target_link_libraries(adi_ne_trellis_test PRIVATE adi_runtime)
    target_include_directories(adi_ne_trellis_test PRIVATE src)
    add_test(NAME ne_trellis COMMAND adi_ne_trellis_test)

    add_executable(adi_expert_trellis_test tests/expert_trellis_test.cpp)
    target_link_libraries(adi_expert_trellis_test PRIVATE adi_runtime)
    target_include_directories(adi_expert_trellis_test PRIVATE src)
    add_test(NAME expert_trellis COMMAND adi_expert_trellis_test)

    add_executable(adi_ne_batch_test tests/ne_batch_test.cpp)
    target_link_libraries(adi_ne_batch_test PRIVATE adi_runtime)
    target_include_directories(adi_ne_batch_test PRIVATE src)
    add_test(NAME ne_batch COMMAND adi_ne_batch_test)

    add_executable(adi_expert_batch_test tests/expert_batch_test.cpp)
    target_link_libraries(adi_expert_batch_test PRIVATE adi_runtime)
    target_include_directories(adi_expert_batch_test PRIVATE src)
    add_test(NAME expert_batch COMMAND adi_expert_batch_test)

    add_executable(adi_codec_cache_test tests/codec_cache_test.cpp)
    target_link_libraries(adi_codec_cache_test PRIVATE adi_runtime)
    target_include_directories(adi_codec_cache_test PRIVATE src)
    add_test(NAME codec_cache COMMAND adi_codec_cache_test)

    add_executable(adi_gated_delta_test tests/gated_delta_test.cpp)
    target_link_libraries(adi_gated_delta_test PRIVATE adi_runtime)
    target_include_directories(adi_gated_delta_test PRIVATE src)
    add_test(NAME gated_delta COMMAND adi_gated_delta_test)

    add_executable(adi_simd_dispatch_test tests/simd_dispatch_test.cpp)
    target_link_libraries(adi_simd_dispatch_test PRIVATE adi_runtime)
    target_include_directories(adi_simd_dispatch_test PRIVATE src)
    add_test(NAME simd_dispatch COMMAND adi_simd_dispatch_test)

    add_executable(adi_int5_vbmi_test tests/int5_vbmi_test.cpp)
    target_link_libraries(adi_int5_vbmi_test PRIVATE adi_runtime)
    target_include_directories(adi_int5_vbmi_test PRIVATE src)
    add_test(NAME int5_vbmi COMMAND adi_int5_vbmi_test)

    add_executable(adi_int5_vbmi_batch_test tests/int5_vbmi_batch_test.cpp)
    target_link_libraries(adi_int5_vbmi_batch_test PRIVATE adi_runtime)
    target_include_directories(adi_int5_vbmi_batch_test PRIVATE src)
    add_test(NAME int5_vbmi_batch COMMAND adi_int5_vbmi_batch_test)

    add_executable(adi_executor_test tests/executor_test.cpp)
    target_link_libraries(adi_executor_test PRIVATE adi_runtime)
    target_include_directories(adi_executor_test PRIVATE src)
    add_test(NAME executor COMMAND adi_executor_test)

    add_executable(adi_generation_test tests/generation_test.cpp)
    target_link_libraries(adi_generation_test PRIVATE adi_runtime)
    target_include_directories(adi_generation_test PRIVATE src)
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
    add_executable(adi_prefill_ubatch_test tests/prefill_ubatch_test.cpp)
    target_link_libraries(adi_prefill_ubatch_test PRIVATE adi_runtime)
    add_executable(adi_linear_prefill_test tests/linear_prefill_test.cpp)
    target_link_libraries(adi_linear_prefill_test PRIVATE adi_runtime)
    add_executable(adi_moe_batch_test tests/moe_batch_test.cpp)
    target_link_libraries(adi_moe_batch_test PRIVATE adi_runtime)
    if(ADI_TEST_MODEL)
        add_test(
            NAME linear_prefill
            COMMAND adi_linear_prefill_test ${ADI_TEST_MODEL})
        add_test(
            NAME moe_batch
            COMMAND adi_moe_batch_test ${ADI_TEST_MODEL})
        add_test(NAME tokenizer COMMAND adi_tokenizer_test ${ADI_TEST_MODEL})
        add_test(NAME model_e2e COMMAND adi_model_e2e_test ${ADI_TEST_MODEL})
        add_test(NAME model_batch COMMAND adi_model_batch_test ${ADI_TEST_MODEL})
        # The wide matrix from the plan runs as "adi_prefill_ubatch_test MODEL
        # --full"; chunk handling does not depend on prompt size, so the
        # per-commit test keeps prompts short.
        add_test(
            NAME prefill_ubatch
            COMMAND adi_prefill_ubatch_test ${ADI_TEST_MODEL})
    endif()

    foreach(test_target
            adi_smoke_test
            adi_gguf_test
            adi_kernels_test
            adi_int5_batch_test
            adi_parallel_test
            adi_ne_trellis_test
            adi_expert_trellis_test
            adi_ne_batch_test
            adi_expert_batch_test
            adi_codec_cache_test
            adi_gated_delta_test
            adi_simd_dispatch_test
            adi_int5_vbmi_test
            adi_int5_vbmi_batch_test
            adi_executor_test
            adi_generation_test
            adi_json_test
            adi_unicode_test
            adi_bpe_test
            adi_chat_test
            adi_server_test
            adi_tokenizer_test
            adi_model_e2e_test
            adi_model_batch_test
            adi_prefill_ubatch_test
            adi_linear_prefill_test
            adi_moe_batch_test)
        if(MSVC)
            target_compile_options(${test_target} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${test_target} PRIVATE -UNDEBUG)
        endif()
    endforeach()

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME adi_chat_client
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/adi_chat_test.py
        )
        add_test(
            NAME pack_mach
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/pack_mach.py --self-test
        )
    endif()
endif()
