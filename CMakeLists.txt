cmake_minimum_required(VERSION 3.20)
project(adi VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ADI_BUILD_TESTS "Build ADI tests" ON)

add_library(adi_runtime STATIC
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
endif()
